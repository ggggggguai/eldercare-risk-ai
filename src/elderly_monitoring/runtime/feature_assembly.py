from __future__ import annotations

import json
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from elderly_monitoring.modules.fall_risk.baseline import MODEL_VERSION as BASELINE_MODEL_VERSION
from elderly_monitoring.modules.fall_risk.baseline import (
    build_personal_baselines,
    score_baseline_deviation,
)
from elderly_monitoring.modules.fall_risk.gait import GAIT_KEYPOINT_NAMES, extract_gait_windows
from elderly_monitoring.modules.fall_risk.near_fall import (
    MODEL_VERSION as NEAR_FALL_MODEL_VERSION,
    NEAR_FALL_KEYPOINT_NAMES,
    extract_near_fall_events,
)
from elderly_monitoring.modules.fall_risk.pose_quality import CORE_KEYPOINT_NAMES, process_pose_records
from elderly_monitoring.modules.fall_risk.sit_stand import (
    MODEL_VERSION as SIT_STAND_MODEL_VERSION,
    extract_sit_stand_events,
)
from elderly_monitoring.runtime.fall_state import FallStateConfig, FallStateDetector


@dataclass(frozen=True)
class BranchQualityGate:
    required_keypoints: tuple[str, ...]
    min_frames: int
    min_valid_frame_ratio: float
    min_keypoint_coverage: float
    min_window_coverage_sec: float
    min_effective_fps: float
    max_gap_sec: float


@dataclass(frozen=True)
class FeatureAssemblyConfig:
    window_sec: float = 10.0
    analysis_interval_sec: float = 0.5
    gait_gate: BranchQualityGate = BranchQualityGate(
        required_keypoints=tuple(GAIT_KEYPOINT_NAMES),
        min_frames=8,
        min_valid_frame_ratio=0.60,
        min_keypoint_coverage=0.70,
        min_window_coverage_sec=1.0,
        min_effective_fps=4.0,
        max_gap_sec=1.0,
    )
    sit_stand_gate: BranchQualityGate = BranchQualityGate(
        required_keypoints=tuple(CORE_KEYPOINT_NAMES),
        min_frames=8,
        min_valid_frame_ratio=0.60,
        min_keypoint_coverage=0.70,
        min_window_coverage_sec=1.0,
        min_effective_fps=4.0,
        max_gap_sec=1.0,
    )
    near_fall_gate: BranchQualityGate = BranchQualityGate(
        required_keypoints=tuple(NEAR_FALL_KEYPOINT_NAMES),
        min_frames=5,
        min_valid_frame_ratio=0.60,
        min_keypoint_coverage=0.60,
        min_window_coverage_sec=0.5,
        min_effective_fps=6.0,
        max_gap_sec=0.75,
    )
    fall_state_gate: BranchQualityGate = BranchQualityGate(
        required_keypoints=tuple(CORE_KEYPOINT_NAMES),
        min_frames=2,
        min_valid_frame_ratio=0.60,
        min_keypoint_coverage=0.60,
        min_window_coverage_sec=0.1,
        min_effective_fps=6.0,
        max_gap_sec=0.75,
    )


@dataclass(frozen=True)
class FeatureSnapshot:
    features: dict[str, Any]
    quality_flags: list[str]
    usable: bool = True
    urgent: bool = False
    branch_diagnostics: dict[str, dict[str, Any]] = field(default_factory=dict)
    stage_timings_ms: dict[str, float] = field(default_factory=dict)


class FeatureAssembler:
    def __init__(
        self,
        *,
        person_id: str,
        scene_region: str,
        device_id: str | None = None,
        scene_risk_scores: Mapping[str, float] | None = None,
        config: FeatureAssemblyConfig | None = None,
        baseline_history: Iterable[Mapping[str, Any]] | None = None,
        fall_state_config: FallStateConfig | None = None,
        gait_predictor: Any | None = None,
        sit_stand_predictor: Any | None = None,
        fall_event_predictor: Any | None = None,
        fall_event_runtime_mode: str = "shadow",
    ) -> None:
        self.person_id = person_id
        self.device_id = device_id
        self.scene_region = scene_region
        self.scene_risk_scores = dict(scene_risk_scores or {})
        self.config = config or FeatureAssemblyConfig()
        self.records: deque[dict[str, Any]] = deque()
        self._last_analysis: float | None = None
        self._fall_state = FallStateDetector(fall_state_config)
        self._gait_predictor = gait_predictor
        self._sit_stand_predictor = sit_stand_predictor
        self._fall_event_predictor = fall_event_predictor
        if fall_event_runtime_mode not in {"shadow", "experimental_tcn"}:
            raise ValueError(
                "fall_event_runtime_mode must be shadow or experimental_tcn"
            )
        self._fall_event_runtime_mode = fall_event_runtime_mode
        self._baselines = build_personal_baselines(baseline_history or []) if baseline_history else {}
        self._baseline_lock = threading.RLock()
        self._baseline_period_result: dict[str, Any] | None = None
        self.analysis_count = 0
        self.stream_epoch = 0
        self.reset_count = 0
        self.last_reset_reason: str | None = None
        self._cached_signature: tuple[Any, ...] | None = None
        self._cached_snapshot: FeatureSnapshot | None = None

    def reset(self, *, reason: str = "manual_reset", stream_epoch: int | None = None) -> None:
        self.records.clear()
        self._last_analysis = None
        self._fall_state.reset()
        self._cached_signature = None
        self._cached_snapshot = None
        self.reset_count += 1
        self.last_reset_reason = reason
        if stream_epoch is not None:
            self.stream_epoch = stream_epoch

    def update_baseline_period(self, record: Mapping[str, Any] | None) -> dict[str, Any] | None:
        """Submit one completed period result for the next realtime fusion window.

        Pose windows remain deliberately unsupported as baseline periods. The
        caller is expected to provide an upstream daily/hourly aggregate that
        satisfies the baseline period schema. Passing ``None`` clears the
        current result and restores the unavailable fallback.
        """
        if record is None:
            with self._baseline_lock:
                self._baseline_period_result = None
                self._cached_signature = None
                self._cached_snapshot = None
            return None
        person_id = str(record.get("person_id", ""))
        if person_id != self.person_id:
            raise ValueError(
                f"baseline period person_id {person_id!r} does not match assembler person_id {self.person_id!r}"
            )
        results = score_baseline_deviation([record], self._baselines)
        if not results:
            raise ValueError(
                "baseline period must be a valid completed fall-baseline-period-features-v1 record"
            )
        result = results[0]
        with self._baseline_lock:
            self._baseline_period_result = dict(result)
            self._cached_signature = None
            self._cached_snapshot = None
            return dict(self._baseline_period_result)

    def add_pose(self, record: Mapping[str, Any], *, monotonic_sec: float) -> FeatureSnapshot | None:
        item = dict(record)
        item["person_id"] = self.person_id
        self.records.append(item)
        timestamp = _number(item.get("timestamp_sec"), monotonic_sec)
        while self.records and timestamp - _number(self.records[0].get("timestamp_sec"), timestamp) > self.config.window_sec:
            self.records.popleft()
        if self._last_analysis is not None and monotonic_sec - self._last_analysis < self.config.analysis_interval_sec:
            return None
        self._last_analysis = monotonic_sec
        return self._assemble()

    def _assemble(self) -> FeatureSnapshot:
        records = list(self.records)
        signature = _window_signature(records)
        if signature == self._cached_signature and self._cached_snapshot is not None:
            return self._cached_snapshot
        self.analysis_count += 1
        assembly_started = time.perf_counter()
        quality_started = time.perf_counter()
        try:
            cleaned = process_pose_records(records)
        except Exception as exc:
            snapshot = self._quality_failure_snapshot(records, exc, quality_started, assembly_started)
            self._cached_signature = signature
            self._cached_snapshot = snapshot
            return snapshot
        stage_timings = {"pose_quality": _elapsed_ms(quality_started)}
        quality_flags: list[str] = []
        quality_values = [_number(record.get("keypoint_quality"), 0.0) for record in cleaned]
        keypoint_quality = sum(quality_values) / len(quality_values) if quality_values else 0.0
        branch_diagnostics: dict[str, dict[str, Any]] = {}

        gait_item, branch_diagnostics["gait"] = self._run_gait(cleaned)
        sit_item, branch_diagnostics["sit_stand"] = self._run_sit_stand(cleaned)
        near_item, branch_diagnostics["near_fall"] = self._run_near_fall(cleaned)
        state, branch_diagnostics["fall_state"] = self._run_fall_state(cleaned)
        if self._fall_event_runtime_mode == "experimental_tcn":
            fall_event_tcn_item, branch_diagnostics["fall_event_tcn"] = (
                self._run_fall_event_tcn_runtime(cleaned)
            )
            fall_event_tcn_diagnostic = branch_diagnostics["fall_event_tcn"]
        else:
            fall_event_tcn_item, branch_diagnostics["fall_event_tcn_shadow"] = (
                self._run_fall_event_tcn_shadow(cleaned)
            )
            fall_event_tcn_diagnostic = branch_diagnostics["fall_event_tcn_shadow"]
        baseline_item, branch_diagnostics["baseline"] = self._run_baseline(cleaned)
        scene_score = max(
            0.0,
            min(1.0, float(self.scene_risk_scores.get(self.scene_region, 0.0))),
        )
        branch_diagnostics["scene"] = _static_branch_diagnostic(
            score=scene_score,
            model_version="scene-risk-config-v1",
            input_frame_count=len(cleaned),
        )
        for name, diagnostic in branch_diagnostics.items():
            stage_timings[name] = float(diagnostic.get("duration_ms", 0.0))
            if diagnostic["status"] != "valid" and not diagnostic.get("optional"):
                quality_flags.extend(f"{name}:{reason}" for reason in diagnostic["reasons"])
        if branch_diagnostics["baseline"]["status"] == "unavailable":
            quality_flags.append("insufficient_baseline_history")

        contextual_signal_available = any(
            branch_diagnostics[name]["status"] == "valid"
            for name in ("gait", "sit_stand", "near_fall", "baseline")
        )
        rule_fall_score = state.fall_event_score if state else None
        fall_event_score = rule_fall_score
        fall_event_score_source = (
            "fall_state_rule"
            if branch_diagnostics["fall_state"]["status"] == "valid"
            else "unavailable"
        )
        fall_event_score_available = (
            branch_diagnostics["fall_state"]["status"] == "valid"
        )
        if self._fall_event_runtime_mode == "experimental_tcn":
            if fall_event_tcn_diagnostic["status"] == "valid":
                detected = bool(fall_event_tcn_item.get("fall_event_tcn_detected"))
                # Preserve the existing event-feature contract: a detected fall is
                # a strong binary trigger, while the raw model probability remains
                # available separately for diagnostics and threshold tuning.
                fall_event_score = 0.9 if detected else 0.0
                fall_event_score_source = "continuous_tcn"
                fall_event_score_available = True
            elif fall_event_score_available:
                fall_event_score_source = "fall_state_rule_fallback"

        fusion_mask = {
            "gait_risk_score": branch_diagnostics["gait"]["status"] == "valid",
            "sit_stand_risk_score": branch_diagnostics["sit_stand"]["status"] == "valid",
            "near_fall_event_score": branch_diagnostics["near_fall"]["status"] == "valid",
            "baseline_deviation_score": branch_diagnostics["baseline"]["status"] == "valid",
            "activity_rhythm_score": (
                branch_diagnostics["baseline"]["status"] == "valid"
                and baseline_item.get("activity_rhythm_score") is not None
            ),
            # Scene risk is contextual and must never drive a warning by itself.
            "scene_risk_score": contextual_signal_available,
            "fall_event_score": fall_event_score_available,
            "long_static_score": branch_diagnostics["fall_state"]["status"] == "valid",
        }
        coverage_names = (
            "gait_risk_score",
            "sit_stand_risk_score",
            "near_fall_event_score",
            "baseline_deviation_score",
            "scene_risk_score",
            "activity_rhythm_score",
        )
        feature_coverage = sum(bool(fusion_mask[name]) for name in coverage_names) / len(coverage_names)
        gait_score = _valid_score(gait_item, "gait_risk_score", branch_diagnostics["gait"])
        sit_score = _valid_score(sit_item, "sit_stand_risk_score", branch_diagnostics["sit_stand"])
        near_score = _valid_score(near_item, "near_fall_event_score", branch_diagnostics["near_fall"])
        baseline_score = _valid_score(baseline_item, "baseline_deviation_score", branch_diagnostics["baseline"])
        activity_score = _valid_score(baseline_item, "activity_rhythm_score", branch_diagnostics["baseline"])
        features = {
            "person_id": self.person_id,
            "device_id": self.device_id,
            "scene_region": self.scene_region,
            "stream_epoch": self.stream_epoch,
            "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
            "start_time": _number(cleaned[0].get("timestamp_sec"), 0.0) if cleaned else None,
            "end_time": _number(cleaned[-1].get("timestamp_sec"), 0.0) if cleaned else None,
            "keypoint_quality": keypoint_quality,
            "feature_coverage": round(feature_coverage, 4),
            "gait_risk_score": gait_score,
            "gait_score_source": gait_item.get("score_source", "unavailable"),
            "gait_risk_factors": list(gait_item.get("risk_factors", [])),
            "sit_stand_risk_score": sit_score,
            "near_fall_event_score": near_score,
            "baseline_deviation_score": baseline_score,
            "activity_rhythm_score": activity_score,
            "baseline_state": baseline_item.get("baseline_state"),
            "baseline_confidence": baseline_item.get("baseline_confidence"),
            "baseline_fusion_weight": baseline_item.get("baseline_fusion_weight"),
            "baseline_deviation_factors": list(
                baseline_item.get("deviation_factors", [])
            ),
            "scene_risk_score": scene_score,
            "fall_event_score": fall_event_score,
            "fall_event_score_source": fall_event_score_source,
            "long_static_score": state.long_static_score if state else None,
            "fall_event_tcn_score": fall_event_tcn_item.get(
                "fall_event_tcn_score"
            ),
            "fall_event_tcn_detected": bool(
                fall_event_tcn_item.get("fall_event_tcn_detected", False)
            ),
            "fall_event_tcn_onset_frame": fall_event_tcn_item.get(
                "fall_event_tcn_onset_frame"
            ),
            "fall_event_tcn_shadow_score": fall_event_tcn_item.get(
                "fall_event_tcn_shadow_score"
            ),
            "fall_event_tcn_shadow_detected": bool(
                fall_event_tcn_item.get("fall_event_tcn_shadow_detected", False)
            ),
            "fall_event_tcn_shadow_onset_frame": fall_event_tcn_item.get(
                "fall_event_tcn_shadow_onset_frame"
            ),
            "fusion_mask": fusion_mask,
            "branch_diagnostics": branch_diagnostics,
        }
        usable_branches = ["gait", "sit_stand", "near_fall", "fall_state"]
        if self._fall_event_runtime_mode == "experimental_tcn":
            usable_branches.append("fall_event_tcn")
        usable = any(
            branch_diagnostics[name]["status"] == "valid"
            for name in usable_branches
        )
        urgent = bool(
            (state and (state.triggered_now or state.long_static_score >= 0.8))
            or (
                self._fall_event_runtime_mode == "experimental_tcn"
                and fall_event_tcn_item.get("fall_event_tcn_detected")
            )
        )
        stage_timings["feature_assembly_total"] = _elapsed_ms(assembly_started)
        snapshot = FeatureSnapshot(
            features=features,
            quality_flags=list(dict.fromkeys(quality_flags)),
            usable=usable,
            urgent=urgent,
            branch_diagnostics=branch_diagnostics,
            stage_timings_ms=stage_timings,
        )
        self._cached_signature = signature
        self._cached_snapshot = snapshot
        return snapshot

    def _run_gait(self, cleaned: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
        return self._run_scored_branch(
            name="gait",
            cleaned=cleaned,
            gate=self.config.gait_gate,
            score_field="gait_risk_score",
            default_version="gait-risk-rule-v0.2",
            run=lambda: extract_gait_windows(cleaned, model_predictor=self._gait_predictor),
            unavailable_flag="insufficient_gait_quality",
        )

    def _run_sit_stand(self, cleaned: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
        predictor = self._sit_stand_predictor
        return self._run_scored_branch(
            name="sit_stand",
            cleaned=cleaned,
            gate=self.config.sit_stand_gate,
            score_field="sit_stand_risk_score",
            default_version=str(
                getattr(predictor, "model_version", SIT_STAND_MODEL_VERSION)
            ),
            run=(
                (lambda: predictor.predict_records(cleaned))
                if predictor is not None
                else (lambda: extract_sit_stand_events(cleaned))
            ),
            unavailable_flag="insufficient_sit_stand_quality",
        )

    def _run_near_fall(self, cleaned: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
        return self._run_scored_branch(
            name="near_fall",
            cleaned=cleaned,
            gate=self.config.near_fall_gate,
            score_field="near_fall_event_score",
            default_version=NEAR_FALL_MODEL_VERSION,
            run=lambda: extract_near_fall_events(cleaned),
            unavailable_flag="insufficient_near_fall_quality",
        )

    def _run_scored_branch(
        self,
        *,
        name: str,
        cleaned: list[dict[str, Any]],
        gate: BranchQualityGate,
        score_field: str,
        default_version: str,
        run: Any,
        unavailable_flag: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        started = time.perf_counter()
        diagnostic = _quality_diagnostic(cleaned, gate)
        if diagnostic["reasons"]:
            diagnostic.update({
                "status": "unavailable",
                "score": None,
                "model_version": default_version,
                "duration_ms": _elapsed_ms(started),
            })
            return {}, diagnostic
        try:
            outputs = run()
            item = outputs[-1] if outputs else {}
        except Exception as exc:
            diagnostic.update({
                "status": "inference_error",
                "score": None,
                "reasons": ["branch_inference_failed"],
                "error_type": type(exc).__name__,
                "model_version": default_version,
                "duration_ms": _elapsed_ms(started),
            })
            return {}, diagnostic
        risk_factors = set(str(value) for value in item.get("risk_factors", []))
        if unavailable_flag in risk_factors or item.get("score_source") == "unavailable":
            reason = str(item.get("fallback_reason") or unavailable_flag)
            diagnostic.update({
                "status": "unavailable",
                "score": None,
                "reasons": [reason],
                "model_version": str(item.get("model_version", default_version)),
                "score_source": item.get("score_source"),
                "fallback_reason": item.get("fallback_reason"),
                "duration_ms": _elapsed_ms(started),
            })
            return item, diagnostic
        score = _number(item.get(score_field), 0.0) if item else 0.0
        diagnostic.update({
            "status": "valid",
            "score": score,
            "reasons": [],
            "model_version": str(item.get("model_version", default_version)),
            "score_source": item.get("score_source"),
            "fallback_reason": item.get("fallback_reason"),
            "duration_ms": _elapsed_ms(started),
        })
        return item, diagnostic

    def _run_fall_state(self, cleaned: list[dict[str, Any]]) -> tuple[Any | None, dict[str, Any]]:
        started = time.perf_counter()
        diagnostic = _quality_diagnostic(cleaned, self.config.fall_state_gate)
        if diagnostic["reasons"]:
            diagnostic.update({
                "status": "unavailable",
                "score": None,
                "model_version": "fall-state-rule-v0.1",
                "duration_ms": _elapsed_ms(started),
            })
            return None, diagnostic
        try:
            state = self._fall_state.update(
                _state_observation(cleaned[-1], cleaned[-2] if len(cleaned) > 1 else None)
            )
        except Exception as exc:
            diagnostic.update({
                "status": "inference_error",
                "score": None,
                "reasons": ["branch_inference_failed"],
                "error_type": type(exc).__name__,
                "model_version": "fall-state-rule-v0.1",
                "duration_ms": _elapsed_ms(started),
            })
            return None, diagnostic
        diagnostic.update({
            "status": "valid",
            "score": max(state.fall_event_score, state.long_static_score),
            "reasons": [],
            "model_version": "fall-state-rule-v0.1",
            "duration_ms": _elapsed_ms(started),
        })
        return state, diagnostic

    def _run_fall_event_tcn_shadow(
        self, cleaned: list[dict[str, Any]]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        started = time.perf_counter()
        if self._fall_event_predictor is None:
            return {}, {
                "status": "disabled",
                "score": None,
                "reasons": ["shadow_predictor_disabled"],
                "model_version": "fall-event-continuous-tcn-shadow-none",
                "optional": True,
                "duration_ms": _elapsed_ms(started),
            }
        try:
            item = dict(self._fall_event_predictor.predict_records(cleaned))
        except Exception as exc:
            return {}, {
                "status": "inference_error",
                "score": None,
                "reasons": ["shadow_inference_failed"],
                "error_type": type(exc).__name__,
                "model_version": str(
                    getattr(self._fall_event_predictor, "model_version", "unknown")
                ),
                "optional": True,
                "duration_ms": _elapsed_ms(started),
            }
        score = item.get("fall_event_tcn_shadow_score")
        if score is None:
            return item, {
                "status": "unavailable",
                "score": None,
                "reasons": [str(item.get("fall_event_tcn_shadow_reason", "shadow_unavailable"))],
                "model_version": str(
                    item.get(
                        "fall_event_tcn_shadow_model_version",
                        getattr(self._fall_event_predictor, "model_version", "unknown"),
                    )
                ),
                "optional": True,
                "duration_ms": _elapsed_ms(started),
            }
        return item, {
            "status": "valid",
            "score": float(score),
            "reasons": [],
            "model_version": str(
                item.get(
                    "fall_event_tcn_shadow_model_version",
                    getattr(self._fall_event_predictor, "model_version", "unknown"),
                )
            ),
            "score_source": "provisional_shadow",
            "optional": True,
            "duration_ms": _elapsed_ms(started),
        }

    def _run_fall_event_tcn_runtime(
        self, cleaned: list[dict[str, Any]]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        started = time.perf_counter()
        if self._fall_event_predictor is None:
            return {}, {
                "status": "unavailable",
                "score": None,
                "reasons": ["runtime_predictor_disabled"],
                "model_version": "fall-event-continuous-tcn-runtime-none",
                "score_source": "fall_state_rule_fallback",
                "optional": True,
                "duration_ms": _elapsed_ms(started),
            }
        try:
            item = dict(self._fall_event_predictor.predict_records(cleaned))
        except Exception as exc:
            return {}, {
                "status": "inference_error",
                "score": None,
                "reasons": ["runtime_inference_failed"],
                "error_type": type(exc).__name__,
                "model_version": str(
                    getattr(self._fall_event_predictor, "model_version", "unknown")
                ),
                "score_source": "fall_state_rule_fallback",
                "optional": True,
                "duration_ms": _elapsed_ms(started),
            }
        score = item.get("fall_event_tcn_score")
        if score is None:
            return item, {
                "status": "unavailable",
                "score": None,
                "reasons": [
                    str(item.get("fall_event_tcn_reason", "runtime_window_unavailable"))
                ],
                "model_version": str(
                    item.get(
                        "fall_event_tcn_model_version",
                        getattr(self._fall_event_predictor, "model_version", "unknown"),
                    )
                ),
                "score_source": "fall_state_rule_fallback",
                "optional": True,
                "duration_ms": _elapsed_ms(started),
            }
        return item, {
            "status": "valid",
            "score": float(score),
            "reasons": [],
            "model_version": str(
                item.get(
                    "fall_event_tcn_model_version",
                    getattr(self._fall_event_predictor, "model_version", "unknown"),
                )
            ),
            "score_source": "continuous_tcn",
            "provisional": True,
            "optional": True,
            "duration_ms": _elapsed_ms(started),
        }

    def _run_baseline(self, cleaned: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
        started = time.perf_counter()
        with self._baseline_lock:
            period_result = dict(self._baseline_period_result) if self._baseline_period_result is not None else None
        if period_result is not None:
            item = period_result
            score = item.get("baseline_deviation_score")
            if score is not None:
                return item, {
                    "status": "valid",
                    "score": float(score),
                    "reasons": [],
                    "model_version": str(item.get("model_version", BASELINE_MODEL_VERSION)),
                    "score_source": "personal_baseline_period",
                    "input_frame_count": len(cleaned),
                    "valid_frame_count": len(cleaned),
                    "baseline_state": item.get("baseline_state"),
                    "duration_ms": _elapsed_ms(started),
                }
            reasons = list(item.get("deviation_factors", [])) or [
                "baseline_score_unavailable"
            ]
            return item, {
                "status": "unavailable",
                "score": None,
                "reasons": reasons,
                "model_version": str(item.get("model_version", BASELINE_MODEL_VERSION)),
                "score_source": "personal_baseline_period",
                "input_frame_count": len(cleaned),
                "valid_frame_count": 0,
                "baseline_state": item.get("baseline_state"),
                "duration_ms": _elapsed_ms(started),
            }
        reasons = (
            ["completed_baseline_period_unavailable"]
            if self._baselines
            else ["insufficient_baseline_history", "completed_baseline_period_unavailable"]
        )
        return {}, {
            "status": "unavailable",
            "score": None,
            "reasons": reasons,
            "input_frame_count": len(cleaned),
            "valid_frame_count": 0,
            "model_version": BASELINE_MODEL_VERSION,
            "duration_ms": _elapsed_ms(started),
        }

    def _quality_failure_snapshot(
        self,
        records: list[dict[str, Any]],
        exc: Exception,
        quality_started: float,
        assembly_started: float,
    ) -> FeatureSnapshot:
        diagnostic = {
            "status": "inference_error",
            "score": None,
            "reasons": ["pose_quality_failed"],
            "error_type": type(exc).__name__,
            "input_frame_count": len(records),
            "valid_frame_count": 0,
            "model_version": "unavailable",
            "duration_ms": _elapsed_ms(quality_started),
        }
        branches = {
            name: dict(diagnostic)
            for name in ("gait", "sit_stand", "near_fall", "fall_state", "baseline")
        }
        features = {
            "person_id": self.person_id,
            "device_id": self.device_id,
            "scene_region": self.scene_region,
            "stream_epoch": self.stream_epoch,
            "feature_coverage": 0.0,
            "fusion_mask": {},
            "branch_diagnostics": branches,
        }
        return FeatureSnapshot(
            features=features,
            quality_flags=["pose_quality:pose_quality_failed"],
            usable=False,
            branch_diagnostics=branches,
            stage_timings_ms={
                "pose_quality": diagnostic["duration_ms"],
                "feature_assembly_total": _elapsed_ms(assembly_started),
            },
        )


def _quality_diagnostic(
    records: list[dict[str, Any]],
    gate: BranchQualityGate,
) -> dict[str, Any]:
    timestamps = [_optional_number(record.get("timestamp_sec")) for record in records]
    source_times = [value for value in timestamps if value is not None]
    source_times.sort()
    gaps = [later - earlier for earlier, later in zip(source_times, source_times[1:])]
    window_start = source_times[0] if source_times else None
    window_end = source_times[-1] if source_times else None
    coverage_sec = (
        max(0.0, window_end - window_start)
        if window_start is not None and window_end is not None
        else 0.0
    )
    effective_fps = (
        (len(source_times) - 1) / coverage_sec
        if len(source_times) >= 2 and coverage_sec > 0
        else 0.0
    )
    max_gap_sec = max(gaps, default=0.0)
    required = gate.required_keypoints
    total_required = len(records) * len(required)
    valid_required = 0
    valid_frames = 0
    core_total = len(records) * len(CORE_KEYPOINT_NAMES)
    valid_core = 0
    for record in records:
        points = {
            str(point.get("name")): point
            for point in record.get("keypoints", [])
            if isinstance(point, Mapping)
        }
        valid_in_frame = sum(
            1
            for name in required
            if _usable_keypoint(points.get(name))
        )
        valid_required += valid_in_frame
        if required and valid_in_frame / len(required) >= gate.min_keypoint_coverage:
            valid_frames += 1
        valid_core += sum(
            1 for name in CORE_KEYPOINT_NAMES if _usable_keypoint(points.get(name))
        )
    keypoint_coverage = valid_required / total_required if total_required else 0.0
    core_keypoint_coverage = valid_core / core_total if core_total else 0.0
    valid_frame_ratio = valid_frames / len(records) if records else 0.0
    reasons: list[str] = []
    if len(records) < gate.min_frames:
        reasons.append("insufficient_frames")
    if valid_frame_ratio < gate.min_valid_frame_ratio:
        reasons.append("insufficient_valid_frame_ratio")
    if keypoint_coverage < gate.min_keypoint_coverage:
        reasons.append("insufficient_branch_keypoint_coverage")
    if coverage_sec < gate.min_window_coverage_sec:
        reasons.append("insufficient_window_coverage")
    if effective_fps < gate.min_effective_fps:
        reasons.append("insufficient_effective_fps")
    if max_gap_sec > gate.max_gap_sec:
        reasons.append("source_gap_exceeded")
    return {
        "status": "valid" if not reasons else "unavailable",
        "reasons": reasons,
        "input_frame_count": len(records),
        "valid_frame_count": valid_frames,
        "valid_frame_ratio": round(valid_frame_ratio, 4),
        "core_keypoint_coverage": round(core_keypoint_coverage, 4),
        "branch_keypoint_coverage": round(keypoint_coverage, 4),
        "effective_fps": round(effective_fps, 4),
        "max_gap_sec": round(max_gap_sec, 4),
        "window_start_sec": None if window_start is None else round(window_start, 4),
        "window_end_sec": None if window_end is None else round(window_end, 4),
        "window_coverage_sec": round(coverage_sec, 4),
    }


def _static_branch_diagnostic(
    *,
    score: float,
    model_version: str,
    input_frame_count: int,
) -> dict[str, Any]:
    return {
        "status": "valid",
        "score": score,
        "reasons": [],
        "input_frame_count": input_frame_count,
        "valid_frame_count": input_frame_count,
        "model_version": model_version,
        "duration_ms": 0.0,
    }


def _valid_score(
    item: Mapping[str, Any],
    score_field: str,
    diagnostic: Mapping[str, Any],
) -> float | None:
    if diagnostic.get("status") != "valid":
        return None
    return _number(item.get(score_field), 0.0)


def _usable_keypoint(point: Mapping[str, Any] | None) -> bool:
    return bool(
        point
        and point.get("valid") is True
        and point.get("is_jump_outlier") is not True
    )


def _window_signature(records: list[dict[str, Any]]) -> tuple[Any, ...]:
    if not records:
        return (0,)
    first = records[0]
    last = records[-1]
    return (
        len(records),
        first.get("frame_id"),
        first.get("timestamp_sec"),
        first.get("track_id"),
        last.get("frame_id"),
        last.get("timestamp_sec"),
        last.get("track_id"),
    )


def _optional_number(value: Any) -> float | None:
    try:
        if value is None:
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 4)


def _state_observation(record: Mapping[str, Any], previous: Mapping[str, Any] | None = None) -> dict[str, float]:
    points = {str(point.get("name")): point for point in record.get("keypoints", []) if isinstance(point, Mapping)}
    def center(left: str, right: str) -> tuple[float, float]:
        first, second = points.get(left), points.get(right)
        if not first or not second:
            return 0.0, 0.0
        return (_number(first.get("x"), 0.0) + _number(second.get("x"), 0.0)) / 2, (_number(first.get("y"), 0.0) + _number(second.get("y"), 0.0)) / 2
    shoulder = center("left_shoulder", "right_shoulder")
    hip = center("left_hip", "right_hip")
    angle = math.degrees(math.atan2(abs(shoulder[0] - hip[0]), max(0.001, abs(shoulder[1] - hip[1]))))
    bbox = record.get("bbox") or [0.0, 0.0, 0.0, 0.0]
    motion = 1.0
    if previous is not None:
        previous_points = {str(point.get("name")): point for point in previous.get("keypoints", []) if isinstance(point, Mapping)}
        previous_hips = [previous_points.get("left_hip"), previous_points.get("right_hip")]
        if all(previous_hips):
            previous_x = sum(_number(point.get("x"), 0.0) for point in previous_hips) / 2
            previous_y = sum(_number(point.get("y"), 0.0) for point in previous_hips) / 2
            motion = math.hypot(hip[0] - previous_x, hip[1] - previous_y)
    return {
        "timestamp_sec": _number(record.get("timestamp_sec"), 0.0),
        "hip_center_y": hip[1],
        "bbox_center_y": (_number(bbox[1], 0.0) + _number(bbox[3], 0.0)) / 2 if len(bbox) >= 4 else hip[1],
        "trunk_angle_deg": angle,
        "core_keypoint_quality": _number(record.get("core_keypoint_quality", record.get("keypoint_quality")), 0.0),
        "motion_score": _number(record.get("motion_score"), motion),
    }


def _number(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
