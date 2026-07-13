from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class FallStateConfig:
    observation_window_sec: float = 1.0
    min_quality: float = 0.6
    hip_drop_threshold: float = 0.18
    center_drop_threshold: float = 0.15
    horizontal_angle_threshold: float = 60.0
    static_duration_sec: float = 10.0
    static_motion_threshold: float = 0.02


@dataclass(frozen=True)
class FallStateResult:
    fall_event_score: float
    long_static_score: float
    suspected_fall: bool
    triggered_now: bool = False


class FallStateDetector:
    def __init__(self, config: FallStateConfig | None = None) -> None:
        self.config = config or FallStateConfig()
        self._history: deque[dict[str, float]] = deque()
        self._suspected_at: float | None = None
        self._static_since: float | None = None

    def reset(self) -> None:
        self._history.clear()
        self._suspected_at = None
        self._static_since = None

    def update(self, observation: Mapping[str, Any]) -> FallStateResult:
        current = {key: float(observation.get(key, default)) for key, default in (
            ("timestamp_sec", 0.0), ("hip_center_y", 0.0), ("bbox_center_y", 0.0),
            ("trunk_angle_deg", 0.0), ("core_keypoint_quality", 0.0), ("motion_score", 1.0),
        )}
        now = current["timestamp_sec"]
        while self._history and now - self._history[0]["timestamp_sec"] > self.config.observation_window_sec:
            self._history.popleft()
        reference = self._history[0] if self._history else None
        triggered = False
        if reference is not None and current["core_keypoint_quality"] >= self.config.min_quality:
            hip_drop = current["hip_center_y"] - reference["hip_center_y"]
            center_drop = current["bbox_center_y"] - reference["bbox_center_y"]
            if (
                hip_drop >= self.config.hip_drop_threshold
                and center_drop >= self.config.center_drop_threshold
                and current["trunk_angle_deg"] >= self.config.horizontal_angle_threshold
            ):
                self._suspected_at = now
                self._static_since = None
                triggered = True
        self._history.append(current)

        if self._suspected_at is not None:
            if current["motion_score"] <= self.config.static_motion_threshold:
                if self._static_since is None:
                    self._static_since = now
            else:
                self._static_since = None
        static_duration = 0.0 if self._static_since is None else now - self._static_since
        return FallStateResult(
            fall_event_score=0.9 if self._suspected_at is not None else 0.0,
            long_static_score=0.9 if static_duration >= self.config.static_duration_sec else 0.0,
            suspected_fall=self._suspected_at is not None,
            triggered_now=triggered,
        )
