from __future__ import annotations

from dataclasses import replace
import json
import time
from pathlib import Path
from typing import Any, Mapping

from elderly_monitoring.common.schemas import (
    AlgorithmEvent,
    EvidenceWindow,
    action_for_level,
)
from elderly_monitoring.modules.fall_risk.pipeline import FallRiskPipeline
from elderly_monitoring.runtime.event_policy import EventPolicy
from elderly_monitoring.runtime.fall_state import (
    EpisodeState,
    EpisodeTransition,
    FallEpisodeStateMachine,
    FallStateConfig,
)
from elderly_monitoring.runtime.feature_assembly import FeatureAssembler, FeatureAssemblyConfig
from elderly_monitoring.runtime.sampling import SamplingMonitor
from elderly_monitoring.runtime.streaming_pose import StreamingPoseTracker
from elderly_monitoring.service.callback import CallbackSender
from elderly_monitoring.service.outbox import CallbackOutbox


class RealtimeFallRiskEngine:
    def __init__(self, *, assembler: Any, fusion_interval_sec: float = 2.0, pipeline: FallRiskPipeline | None = None) -> None:
        self.assembler = assembler
        self.fusion_interval_sec = fusion_interval_sec
        self.pipeline = pipeline or FallRiskPipeline()
        self._last_fusion: float | None = None
        self.last_snapshot: Any | None = None
        self.last_fusion_duration_ms: float | None = None

    def process_pose(self, record: Mapping[str, Any], *, monotonic_sec: float) -> AlgorithmEvent | None:
        snapshot = self.assembler.add_pose(record, monotonic_sec=monotonic_sec)
        self.last_snapshot = snapshot
        if snapshot is None or not snapshot.usable:
            return None
        if self._last_fusion is not None and monotonic_sec - self._last_fusion < self.fusion_interval_sec and not snapshot.urgent:
            return None
        self._last_fusion = monotonic_sec
        started = time.perf_counter()
        event = self.pipeline.predict_from_features(snapshot.features)
        self.last_fusion_duration_ms = round(
            (time.perf_counter() - started) * 1000.0, 4
        )
        return event

    def update_baseline_period(self, record: Mapping[str, Any] | None) -> dict[str, Any] | None:
        """Forward an upstream completed period to the feature assembler."""
        updater = getattr(self.assembler, "update_baseline_period", None)
        if updater is None:
            raise RuntimeError("feature assembler does not support baseline period updates")
        return updater(record)

    def reset_window(
        self,
        *,
        reason: str = "manual_reset",
        stream_epoch: int | None = None,
    ) -> None:
        self._last_fusion = None
        self.last_snapshot = None
        self.last_fusion_duration_ms = None
        self.assembler.reset(reason=reason, stream_epoch=stream_epoch)


class FallRiskSessionEngine:
    """Compose one pose tracker, in-memory feature analysis and callback policy."""

    def __init__(self, *, session: Any, model_path: str, callback_token: str = "", scene_risk_scores: Mapping[str, float] | None = None, baseline_history_path: str | Path | None = None, **kwargs: Any) -> None:
        self.session = session
        self.tracker = StreamingPoseTracker(
            model_name=model_path,
            person_id=session.person_id,
            scene_region=session.scene_region,
            lost_timeout_sec=float(kwargs.get("primary_lost_timeout_sec", 2.0)),
            inference_size=int(kwargs.get("pose_inference_size", 640)),
        )
        history = _load_jsonl(Path(baseline_history_path)) if baseline_history_path else []
        gait_predictor = None
        gait_model_path = kwargs.get("gait_model_path")
        if gait_model_path:
            from elderly_monitoring.modules.fall_risk.gait_tcn import GaitTCNPredictor

            gait_predictor = GaitTCNPredictor(
                gait_model_path,
                device=str(kwargs.get("gait_model_device", "auto")),
                window_frames=int(kwargs.get("gait_model_window_frames", 16)),
                expected_task="gait_instability_vs_normal_activity",
            )
        sit_stand_predictor = None
        sit_stand_runtime_mode = str(
            kwargs.get("sit_stand_runtime_mode", "rule_baseline")
        )
        if sit_stand_runtime_mode == "experimental_tcn":
            sit_stand_model_path = kwargs.get("sit_stand_model_path")
            if not sit_stand_model_path:
                raise ValueError(
                    "sit_stand_model_path is required for experimental_tcn"
                )
            from elderly_monitoring.modules.fall_risk.sit_stand_continuous_inference import (
                ExperimentalSitStandTCNPredictor,
            )

            sit_stand_predictor = ExperimentalSitStandTCNPredictor(
                sit_stand_model_path,
                device=str(kwargs.get("sit_stand_model_device", "cpu")),
                batch_size=int(kwargs.get("sit_stand_model_batch_size", 128)),
            )
        elif sit_stand_runtime_mode != "rule_baseline":
            raise ValueError(
                "sit_stand_runtime_mode must be rule_baseline or experimental_tcn"
            )
        fall_event_predictor = None
        fall_event_runtime_mode = str(
            kwargs.get("fall_event_runtime_mode", "shadow")
        )
        if fall_event_runtime_mode not in {"shadow", "experimental_tcn"}:
            raise ValueError(
                "fall_event_runtime_mode must be shadow or experimental_tcn"
            )
        shadow_paths = tuple(kwargs.get("fall_event_shadow_checkpoint_paths") or ())
        if shadow_paths:
            if fall_event_runtime_mode == "experimental_tcn":
                from elderly_monitoring.modules.fall_risk.fall_event_continuous_tcn import (
                    ContinuousFallTCNRuntimePredictor as Predictor,
                )
            else:
                from elderly_monitoring.modules.fall_risk.fall_event_continuous_tcn import (
                    ContinuousFallTCNShadowPredictor as Predictor,
                )

            fall_event_predictor = Predictor(
                shadow_paths,
                device=str(kwargs.get("fall_event_shadow_device", "cpu")),
                threshold=float(kwargs.get("fall_event_shadow_threshold", 0.5)),
            )
        self.assembler = FeatureAssembler(
            person_id=session.person_id,
            device_id=session.device_id,
            scene_region=session.scene_region,
            scene_risk_scores=scene_risk_scores,
            config=_feature_assembly_config(kwargs),
            baseline_history=history,
            fall_state_config=FallStateConfig(**dict(kwargs.get("fall_state") or {})),
            gait_predictor=gait_predictor,
            sit_stand_predictor=sit_stand_predictor,
            fall_event_predictor=fall_event_predictor,
            fall_event_runtime_mode=fall_event_runtime_mode,
        )
        self.engine = RealtimeFallRiskEngine(assembler=self.assembler, fusion_interval_sec=float(kwargs.get("fusion_interval_sec", 2.0)))
        self.policy = EventPolicy(cooldown_sec=float(kwargs.get("event_cooldown_sec", 30.0)))
        configured_outbox = kwargs.get("callback_outbox")
        if configured_outbox is not None:
            self.outbox = configured_outbox
            self.callback = getattr(
                configured_outbox, "sender", kwargs.get("callback_sender")
            )
        else:
            self.callback = kwargs.get("callback_sender") or CallbackSender(
                token=callback_token,
                timeout=float(kwargs.get("callback_timeout_sec", 5.0)),
            )
            self.outbox = CallbackOutbox(
                sender=self.callback,
                capacity=int(kwargs.get("outbox_capacity", 32)),
                retry_delays=kwargs.get(
                    "callback_retry_delays_sec", (0.5, 1.0, 2.0)
                ),
            )
        state_config = dict(kwargs.get("fall_state") or {})
        self.episode = FallEpisodeStateMachine(
            recovery_confirmation_sec=float(
                state_config.get("recovery_confirmation_sec", 3.0)
            ),
            episode_ttl_sec=float(state_config.get("episode_ttl_sec", 30.0)),
        )
        self.outbox_drain_timeout_sec = float(
            kwargs.get("outbox_drain_timeout_sec", 3.0)
        )
        self._last_episode_event: AlgorithmEvent | None = None
        self._last_source_time_sec = 0.0
        self.frame_id = 0
        self.primary_pose_count = 0
        self.max_inference_fps = float(kwargs.get("max_inference_fps", 8.0))
        self._last_inference_at: float | None = None
        self.stream_epoch = 0
        self._epoch_started_monotonic_sec = time.monotonic()
        self.sampling_monitor = SamplingMonitor(
            window_sec=float(kwargs.get("pose_window_sec", 10.0))
        )
        self.time_boundary_count = 0
        self.last_time_boundary_reason: str | None = None
        self.epoch_reset_history: list[dict[str, Any]] = []
        self.last_frame_diagnostics: dict[str, Any] = {}
        self.close_diagnostics: dict[str, Any] | None = None
        self.episode.begin_epoch(
            0,
            source_time_sec=0.0,
            monotonic_sec=self._epoch_started_monotonic_sec,
        )

    def update_baseline_period(self, record: Mapping[str, Any] | None) -> dict[str, Any] | None:
        """Submit or clear the latest completed personal-baseline period."""
        return self.engine.update_baseline_period(record)

    def begin_stream_epoch(
        self,
        stream_epoch: int,
        *,
        reason: str,
        started_monotonic_sec: float | None = None,
    ) -> None:
        boundary_monotonic = (
            time.monotonic()
            if started_monotonic_sec is None
            else started_monotonic_sec
        )
        self._clear_episode(
            reason=reason,
            source_time_sec=self._last_source_time_sec,
            monotonic_sec=boundary_monotonic,
        )
        previous_epoch = self.stream_epoch
        pending_records = len(self.assembler.records)
        previous_track_id = self.tracker.primary_track_id
        self.stream_epoch = stream_epoch
        self._epoch_started_monotonic_sec = (
            boundary_monotonic
        )
        self._last_inference_at = None
        self.frame_id = 0
        self.tracker.reset()
        self.engine.reset_window(reason=reason, stream_epoch=stream_epoch)
        self.policy.reset()
        self.episode.begin_epoch(
            stream_epoch,
            source_time_sec=0.0,
            monotonic_sec=boundary_monotonic,
        )
        self._last_episode_event = None
        self._last_source_time_sec = 0.0
        self.sampling_monitor.reset(reason=reason)
        self.last_time_boundary_reason = reason
        self.epoch_reset_history.append({
            "previous_stream_epoch": previous_epoch,
            "stream_epoch": stream_epoch,
            "reason": reason,
            "discarded_pose_records": pending_records,
            "discarded_primary_track_id": previous_track_id,
        })

    def process_frame(
        self,
        frame: Any,
        *,
        source_pts_sec: float | None = None,
        received_monotonic_sec: float | None = None,
        stream_epoch: int | None = None,
        timestamp_sec: float | None = None,
    ) -> None:
        received = time.monotonic() if received_monotonic_sec is None else received_monotonic_sec
        if stream_epoch is not None and stream_epoch != self.stream_epoch:
            return
        source_time = self._source_time(
            source_pts_sec=source_pts_sec,
            received_monotonic_sec=received,
            legacy_timestamp_sec=timestamp_sec,
        )
        self._last_source_time_sec = source_time
        if self._last_inference_at is not None and self.max_inference_fps > 0 and received - self._last_inference_at < 1.0 / self.max_inference_fps:
            return
        self._last_inference_at = received
        boundary_reason = self.sampling_monitor.observe(
            source_time_sec=source_time,
            received_monotonic_sec=received,
            inference_started_monotonic_sec=time.monotonic(),
        )
        if boundary_reason is not None:
            self._clear_episode(
                reason=boundary_reason,
                source_time_sec=source_time,
                monotonic_sec=received,
            )
            self.engine.reset_window(
                reason=boundary_reason,
                stream_epoch=self.stream_epoch,
            )
            self.tracker.reset()
            self.time_boundary_count += 1
            self.last_time_boundary_reason = boundary_reason
            self.policy.reset()
        self.frame_id += 1
        height, width = frame.shape[:2]
        result = self.tracker.process_frame(frame, frame_id=self.frame_id, timestamp_sec=source_time, frame_size=(float(width), float(height)))
        if result.window_reset:
            self._clear_episode(
                reason=result.target_reason,
                source_time_sec=source_time,
                monotonic_sec=received,
            )
            self.engine.reset_window(
                reason=result.target_reason,
                stream_epoch=self.stream_epoch,
            )
            self.policy.reset()
        self.last_frame_diagnostics = {
            "stream_epoch": self.stream_epoch,
            "source_pts_sec": source_time,
            "target": {
                **self.tracker.target_diagnostics,
                "state": result.target_state,
                "reason": result.target_reason,
                "target_changed": result.target_changed,
                "candidate_count": len(result.poses),
            },
            "stage_timings_ms": dict(result.stage_timings_ms or {}),
        }
        if result.primary_pose is None:
            self._expire_episode(monotonic_sec=received)
            return
        self.primary_pose_count += 1
        event = self.engine.process_pose(result.primary_pose.to_dict(), monotonic_sec=received)
        snapshot = self.engine.last_snapshot
        if snapshot is not None:
            self.last_frame_diagnostics["window"] = {
                "usable": snapshot.usable,
                "quality_flags": list(snapshot.quality_flags),
                "branches": snapshot.branch_diagnostics,
            }
            self.last_frame_diagnostics["stage_timings_ms"].update(
                snapshot.stage_timings_ms
            )
        if self.engine.last_fusion_duration_ms is not None:
            self.last_frame_diagnostics["stage_timings_ms"]["fusion"] = (
                self.engine.last_fusion_duration_ms
            )
        observation_status = "unavailable"
        if snapshot is not None:
            score_source = snapshot.features.get("fall_event_score_source")
            branch_name = (
                "fall_event_tcn"
                if score_source == "continuous_tcn"
                else "fall_state"
            )
            fall_diagnostic = snapshot.branch_diagnostics.get(branch_name, {})
            observation_status = str(fall_diagnostic.get("status", "unavailable"))
        if event is not None:
            self._process_algorithm_event(
                event,
                observation_status=observation_status,
                source_time_sec=source_time,
                monotonic_sec=received,
                snapshot=snapshot,
            )
        self._expire_episode(monotonic_sec=received)

    def close(self) -> None:
        close_started = time.monotonic()
        self._clear_episode(
            reason="session_stopped",
            source_time_sec=self._last_source_time_sec,
            monotonic_sec=close_started,
        )
        pending_records = len(self.assembler.records)
        previous_track_id = self.tracker.primary_track_id
        self.engine.reset_window(
            reason="session_stopped",
            stream_epoch=self.stream_epoch,
        )
        self.tracker.reset()
        self.policy.reset()
        self.sampling_monitor.reset(reason="session_stopped")
        drain_result = self.outbox.stop(
            drain_budget_sec=self.outbox_drain_timeout_sec
        )
        self.close_diagnostics = {
            "reason": "session_stopped",
            "stream_epoch": self.stream_epoch,
            "discarded_pose_records": pending_records,
            "discarded_primary_track_id": previous_track_id,
            "outbox_drain": drain_result,
        }
        self.tracker.close()

    def _process_algorithm_event(
        self,
        event: AlgorithmEvent,
        *,
        observation_status: str,
        source_time_sec: float,
        monotonic_sec: float,
        snapshot: Any,
    ) -> None:
        static_condition = bool(
            snapshot is not None
            and float(snapshot.features.get("long_static_score") or 0.0) >= 0.8
        )
        transition = self.episode.observe(
            risk_level=event.risk_level,
            static_condition=static_condition,
            observation_status=observation_status,
            source_time_sec=source_time_sec,
            monotonic_sec=monotonic_sec,
            trigger_condition=(
                "static_duration_threshold"
                if static_condition
                else f"risk_level_{event.risk_level}"
            ),
            stream_epoch=self.stream_epoch,
        )
        if observation_status != "valid":
            return
        if event.risk_level > 0:
            self._last_episode_event = event
        episode_id = self.episode.episode_id
        if episode_id is None:
            return
        force = transition is not None
        version_kind = self._version_kind(transition)
        lifecycle_state = self.episode.state.value
        version_event = (
            self._event_for_transition(event, transition)
            if transition is not None
            else event
        )
        version = self.policy.create_version(
            version_event,
            monotonic_sec=monotonic_sec,
            episode_id=episode_id,
            session_id=self.session.session_id,
            stream_epoch=self.stream_epoch,
            lifecycle_state=lifecycle_state,
            version_kind=version_kind,
            force=force,
        )
        if version is not None:
            self.outbox.enqueue(self.session.callback_url, version)
        if transition is not None and transition.to_state == EpisodeState.RECOVERED:
            self.policy.reset()

    def _expire_episode(self, *, monotonic_sec: float) -> None:
        transition = self.episode.expire(monotonic_sec=monotonic_sec)
        if transition is not None:
            self._enqueue_terminal_transition(transition, monotonic_sec=monotonic_sec)

    def _clear_episode(
        self,
        *,
        reason: str,
        source_time_sec: float,
        monotonic_sec: float,
    ) -> None:
        transition = self.episode.clear(
            reason=reason,
            source_time_sec=source_time_sec,
            monotonic_sec=monotonic_sec,
        )
        if transition is not None:
            self._enqueue_terminal_transition(transition, monotonic_sec=monotonic_sec)

    def _enqueue_terminal_transition(
        self,
        transition: EpisodeTransition,
        *,
        monotonic_sec: float,
    ) -> None:
        if self._last_episode_event is None:
            return
        terminal = self._event_for_transition(self._last_episode_event, transition)
        version = self.policy.create_version(
            terminal,
            monotonic_sec=monotonic_sec,
            episode_id=transition.episode_id,
            session_id=self.session.session_id,
            stream_epoch=transition.stream_epoch,
            lifecycle_state=transition.to_state.value,
            version_kind=transition.to_state.value,
            force=True,
        )
        if version is not None:
            self.outbox.enqueue(self.session.callback_url, version)
        self.policy.reset()

    @staticmethod
    def _version_kind(transition: EpisodeTransition | None) -> str:
        if transition is None:
            return "observation_update"
        if transition.to_state == EpisodeState.SUSPECTED:
            return "initial"
        return transition.to_state.value

    @staticmethod
    def _event_for_transition(
        event: AlgorithmEvent,
        transition: EpisodeTransition | None,
    ) -> AlgorithmEvent:
        if transition is None:
            return event
        terminal = transition.to_state in {
            EpisodeState.RECOVERED,
            EpisodeState.UNRESOLVED,
        }
        metadata = dict(event.metadata)
        metadata["episode_transition"] = {
            "from_state": transition.from_state.value,
            "to_state": transition.to_state.value,
            "source_time_sec": transition.source_time_sec,
            "entered_source_time_sec": transition.entered_source_time_sec,
            "entered_monotonic_sec": transition.entered_monotonic_sec,
            "peak_source_time_sec": transition.peak_source_time_sec,
            "last_observed_source_time_sec": transition.last_observed_source_time_sec,
            "ttl_sec": transition.ttl_sec,
            "trigger_condition": transition.trigger_condition,
            "cleanup_reason": transition.cleanup_reason,
        }
        if not terminal:
            return replace(event, metadata=metadata)
        start_time = (
            event.evidence_window.start_time
            if event.evidence_window is not None
            else None
        )
        return replace(
            event,
            risk_level=0,
            risk_score=0.0,
            trigger_event=f"episode_{transition.to_state.value}",
            risk_factors=[f"episode_{transition.to_state.value}"],
            recommended_action=action_for_level(0),
            evidence_window=EvidenceWindow(
                start_time=start_time,
                end_time=transition.source_time_sec,
            ),
            metadata=metadata,
        )

    def _source_time(
        self,
        *,
        source_pts_sec: float | None,
        received_monotonic_sec: float,
        legacy_timestamp_sec: float | None,
    ) -> float:
        if source_pts_sec is None:
            if legacy_timestamp_sec is not None:
                return float(legacy_timestamp_sec)
            return max(0.0, received_monotonic_sec - self._epoch_started_monotonic_sec)
        return float(source_pts_sec)

    @property
    def sampling_diagnostics(self) -> dict[str, Any]:
        return self.sampling_monitor.snapshot()

    @property
    def runtime_diagnostics(self) -> dict[str, Any]:
        return {
            "stream_epoch": self.stream_epoch,
            "target": self.tracker.target_diagnostics,
            "last_frame": self.last_frame_diagnostics,
            "epoch_resets": list(self.epoch_reset_history),
            "close": self.close_diagnostics,
            "episode": self.episode.snapshot(),
            "outbox": self.outbox.snapshot(),
        }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _feature_assembly_config(kwargs: Mapping[str, Any]) -> FeatureAssemblyConfig:
    config = FeatureAssemblyConfig(
        window_sec=float(kwargs.get("pose_window_sec", 10.0)),
        analysis_interval_sec=float(kwargs.get("analysis_interval_sec", 0.5)),
    )
    configured = kwargs.get("branch_quality")
    if not isinstance(configured, Mapping):
        return config
    fields = {
        "gait": "gait_gate",
        "sit_stand": "sit_stand_gate",
        "near_fall": "near_fall_gate",
        "fall_state": "fall_state_gate",
    }
    updates: dict[str, Any] = {}
    allowed = {
        "min_frames",
        "min_valid_frame_ratio",
        "min_keypoint_coverage",
        "min_window_coverage_sec",
        "min_effective_fps",
        "max_gap_sec",
    }
    for branch_name, field_name in fields.items():
        raw = configured.get(branch_name)
        if not isinstance(raw, Mapping):
            continue
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(
                f"unsupported {branch_name} quality fields: {sorted(unknown)}"
            )
        values = {
            key: (int(value) if key == "min_frames" else float(value))
            for key, value in raw.items()
        }
        updates[field_name] = replace(getattr(config, field_name), **values)
    return replace(config, **updates)
