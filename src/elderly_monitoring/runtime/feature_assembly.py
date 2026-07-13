from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from elderly_monitoring.modules.fall_risk.baseline import build_personal_baselines, score_baseline_deviation
from elderly_monitoring.modules.fall_risk.gait import extract_gait_windows
from elderly_monitoring.modules.fall_risk.near_fall import extract_near_fall_events
from elderly_monitoring.modules.fall_risk.pose_quality import process_pose_records
from elderly_monitoring.modules.fall_risk.sit_stand import extract_sit_stand_events
from elderly_monitoring.runtime.fall_state import FallStateConfig, FallStateDetector


@dataclass(frozen=True)
class FeatureAssemblyConfig:
    window_sec: float = 10.0
    analysis_interval_sec: float = 0.5


@dataclass(frozen=True)
class FeatureSnapshot:
    features: dict[str, Any]
    quality_flags: list[str]
    usable: bool = True
    urgent: bool = False


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
    ) -> None:
        self.person_id = person_id
        self.device_id = device_id
        self.scene_region = scene_region
        self.scene_risk_scores = dict(scene_risk_scores or {})
        self.config = config or FeatureAssemblyConfig()
        self.records: deque[dict[str, Any]] = deque()
        self._last_analysis: float | None = None
        self._fall_state = FallStateDetector(fall_state_config)
        self._baselines = build_personal_baselines(baseline_history or []) if baseline_history else {}
        self.analysis_count = 0

    def reset(self) -> None:
        self.records.clear()
        self._last_analysis = None
        self._fall_state.reset()

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
        self.analysis_count += 1
        records = list(self.records)
        cleaned = process_pose_records(records)
        quality_flags: list[str] = []
        quality_values = [_number(record.get("keypoint_quality"), 0.0) for record in cleaned]
        keypoint_quality = sum(quality_values) / len(quality_values) if quality_values else 0.0
        usable = bool(cleaned) and keypoint_quality >= 0.45
        if not usable:
            quality_flags.append("insufficient_pose_quality")
        gait = extract_gait_windows(cleaned)
        sit_stand = extract_sit_stand_events(cleaned)
        near_fall = extract_near_fall_events(cleaned)
        gait_item = gait[-1] if gait else {}
        sit_item = sit_stand[-1] if sit_stand else {}
        near_item = near_fall[-1] if near_fall else {}
        baseline_score = 0.0
        activity_score = 0.0
        if self._baselines:
            baseline_records = score_baseline_deviation(cleaned, self._baselines)
            baseline_item = baseline_records[-1] if baseline_records else {}
            baseline_score = _number(baseline_item.get("baseline_deviation_score"), 0.0)
            activity_score = _number(baseline_item.get("activity_rhythm_score"), 0.0)
        else:
            quality_flags.append("insufficient_baseline_history")
        state = self._fall_state.update(_state_observation(cleaned[-1], cleaned[-2] if len(cleaned) > 1 else None)) if usable else None
        features = {
            "person_id": self.person_id,
            "device_id": self.device_id,
            "scene_region": self.scene_region,
            "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
            "start_time": _number(cleaned[0].get("timestamp_sec"), 0.0) if cleaned else None,
            "end_time": _number(cleaned[-1].get("timestamp_sec"), 0.0) if cleaned else None,
            "keypoint_quality": keypoint_quality,
            "feature_coverage": (1.0 if self._baselines else 0.6667) if usable else 0.0,
            "gait_risk_score": _number(gait_item.get("gait_risk_score"), 0.0),
            "sit_stand_risk_score": _number(sit_item.get("sit_stand_risk_score"), 0.0),
            "near_fall_event_score": _number(near_item.get("near_fall_event_score"), 0.0),
            "baseline_deviation_score": baseline_score,
            "activity_rhythm_score": activity_score,
            "scene_risk_score": max(0.0, min(1.0, float(self.scene_risk_scores.get(self.scene_region, 0.0)))),
            "fall_event_score": state.fall_event_score if state else 0.0,
            "long_static_score": state.long_static_score if state else 0.0,
        }
        urgent = bool(state and (state.triggered_now or state.long_static_score >= 0.8))
        return FeatureSnapshot(features=features, quality_flags=quality_flags, usable=usable, urgent=urgent)


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
