from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
from typing import Any, Mapping


@dataclass(frozen=True)
class BranchSamplingRequirement:
    min_effective_fps: float
    max_gap_sec: float
    min_frames: int


DEFAULT_BRANCH_REQUIREMENTS = {
    "gait": BranchSamplingRequirement(4.0, 1.0, 8),
    "sit_stand": BranchSamplingRequirement(4.0, 1.0, 8),
    "near_fall": BranchSamplingRequirement(6.0, 0.75, 3),
    "fall_state": BranchSamplingRequirement(6.0, 0.75, 2),
}


class SamplingMonitor:
    def __init__(
        self,
        *,
        window_sec: float = 10.0,
        requirements: Mapping[str, BranchSamplingRequirement] | None = None,
    ) -> None:
        if window_sec <= 0:
            raise ValueError("window_sec must be positive")
        self.window_sec = window_sec
        self.requirements = dict(requirements or DEFAULT_BRANCH_REQUIREMENTS)
        self._source_times: deque[float] = deque()
        self._received_times: deque[float] = deque()
        self._last_received_monotonic_sec: float | None = None
        self._last_frame_age_sec: float | None = None
        self._boundary_count = 0
        self._boundary_reason: str | None = None
        self._lock = threading.RLock()

    def reset(self, *, reason: str) -> None:
        with self._lock:
            self._source_times.clear()
            self._received_times.clear()
            self._last_received_monotonic_sec = None
            self._last_frame_age_sec = None
            self._boundary_reason = reason

    def observe(
        self,
        *,
        source_time_sec: float,
        received_monotonic_sec: float,
        inference_started_monotonic_sec: float,
    ) -> str | None:
        with self._lock:
            boundary_reason: str | None = None
            if self._source_times and source_time_sec <= self._source_times[-1]:
                boundary_reason = (
                    "duplicate_source_pts"
                    if source_time_sec == self._source_times[-1]
                    else "source_pts_regression"
                )
                self._source_times.clear()
                self._received_times.clear()
                self._boundary_count += 1
                self._boundary_reason = boundary_reason
            self._source_times.append(source_time_sec)
            self._received_times.append(received_monotonic_sec)
            while self._source_times and source_time_sec - self._source_times[0] > self.window_sec:
                self._source_times.popleft()
                self._received_times.popleft()
            self._last_received_monotonic_sec = received_monotonic_sec
            self._last_frame_age_sec = max(
                0.0, inference_started_monotonic_sec - received_monotonic_sec
            )
            return boundary_reason

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._snapshot_unlocked()

    def _snapshot_unlocked(self) -> dict[str, Any]:
        source_times = list(self._source_times)
        received_times = list(self._received_times)
        gaps = [later - earlier for earlier, later in zip(source_times, source_times[1:])]
        coverage_sec = (
            source_times[-1] - source_times[0]
            if len(source_times) >= 2
            else 0.0
        )
        effective_fps = (
            (len(source_times) - 1) / coverage_sec if coverage_sec > 0 else 0.0
        )
        received_coverage_sec = (
            received_times[-1] - received_times[0]
            if len(received_times) >= 2
            else 0.0
        )
        received_effective_fps = (
            (len(received_times) - 1) / received_coverage_sec
            if received_coverage_sec > 0
            else 0.0
        )
        max_gap_sec = max(gaps, default=0.0)
        branches = {}
        for name, requirement in self.requirements.items():
            reasons = []
            if len(source_times) < requirement.min_frames:
                reasons.append("insufficient_frames")
            if effective_fps < requirement.min_effective_fps:
                reasons.append("insufficient_effective_fps")
            if max_gap_sec > requirement.max_gap_sec:
                reasons.append("source_gap_exceeded")
            branches[name] = {
                "status": "valid" if not reasons else "unavailable",
                "reasons": reasons,
                "min_effective_fps": requirement.min_effective_fps,
                "max_allowed_gap_sec": requirement.max_gap_sec,
                "min_frames": requirement.min_frames,
            }
        return {
            "frame_count": len(source_times),
            "window_coverage_sec": round(coverage_sec, 4),
            "effective_fps": round(effective_fps, 4),
            "source_effective_fps": round(effective_fps, 4),
            "received_effective_fps": round(received_effective_fps, 4),
            "max_gap_sec": round(max_gap_sec, 4),
            "last_frame_age_sec": (
                None if self._last_frame_age_sec is None else round(self._last_frame_age_sec, 6)
            ),
            "time_boundary_count": self._boundary_count,
            "last_time_boundary_reason": self._boundary_reason,
            "branches": branches,
        }
