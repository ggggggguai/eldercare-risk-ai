from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping
import uuid


class EpisodeState(str, Enum):
    IDLE = "idle"
    SUSPECTED = "suspected"
    CONFIRMED_STATIC = "confirmed_static"
    RECOVERED = "recovered"
    UNRESOLVED = "unresolved"


class InvalidEpisodeTransition(ValueError):
    pass


class CrossEpochEpisodeError(ValueError):
    pass


@dataclass(frozen=True)
class EpisodeTransition:
    episode_id: str
    stream_epoch: int
    from_state: EpisodeState
    to_state: EpisodeState
    source_time_sec: float
    entered_source_time_sec: float
    entered_monotonic_sec: float
    peak_source_time_sec: float
    last_observed_source_time_sec: float
    ttl_sec: float
    trigger_condition: str
    cleanup_reason: str | None = None


class FallEpisodeStateMachine:
    """Track engineering lifecycle state separately from recognition scores."""

    _ACTIVE = {EpisodeState.SUSPECTED, EpisodeState.CONFIRMED_STATIC}
    _LEGAL = {
        EpisodeState.IDLE: {EpisodeState.SUSPECTED},
        EpisodeState.SUSPECTED: {
            EpisodeState.CONFIRMED_STATIC,
            EpisodeState.RECOVERED,
            EpisodeState.UNRESOLVED,
        },
        EpisodeState.CONFIRMED_STATIC: {
            EpisodeState.RECOVERED,
            EpisodeState.UNRESOLVED,
        },
        EpisodeState.RECOVERED: {EpisodeState.SUSPECTED},
        EpisodeState.UNRESOLVED: {EpisodeState.SUSPECTED},
    }

    def __init__(
        self,
        *,
        recovery_confirmation_sec: float = 3.0,
        episode_ttl_sec: float = 30.0,
        id_factory: Any = None,
    ) -> None:
        if recovery_confirmation_sec < 0:
            raise ValueError("recovery_confirmation_sec must be non-negative")
        if episode_ttl_sec <= 0:
            raise ValueError("episode_ttl_sec must be positive")
        self.recovery_confirmation_sec = float(recovery_confirmation_sec)
        self.episode_ttl_sec = float(episode_ttl_sec)
        self._id_factory = id_factory or (lambda: str(uuid.uuid4()))
        self.state = EpisodeState.IDLE
        self.stream_epoch = 0
        self.episode_id: str | None = None
        self.entered_source_time_sec: float | None = None
        self.entered_monotonic_sec: float | None = None
        self.peak_source_time_sec: float | None = None
        self.last_observed_source_time_sec: float | None = None
        self.last_observed_monotonic_sec: float | None = None
        self.last_cleanup_reason: str | None = None
        self._peak_level = 0
        self._recovery_started_source_time_sec: float | None = None
        self.transitions: deque[EpisodeTransition] = deque(maxlen=100)

    def begin_epoch(
        self,
        stream_epoch: int,
        *,
        source_time_sec: float,
        monotonic_sec: float,
    ) -> None:
        if self.state in self._ACTIVE:
            raise InvalidEpisodeTransition("active episode must be cleared before epoch switch")
        self.state = EpisodeState.IDLE
        self.stream_epoch = int(stream_epoch)
        self.episode_id = None
        self.entered_source_time_sec = float(source_time_sec)
        self.entered_monotonic_sec = float(monotonic_sec)
        self.peak_source_time_sec = None
        self.last_observed_source_time_sec = None
        self.last_observed_monotonic_sec = None
        self.last_cleanup_reason = None
        self._peak_level = 0
        self._recovery_started_source_time_sec = None

    def observe(
        self,
        *,
        risk_level: int,
        static_condition: bool,
        observation_status: str,
        source_time_sec: float,
        monotonic_sec: float,
        trigger_condition: str,
        stream_epoch: int | None = None,
    ) -> EpisodeTransition | None:
        if stream_epoch is not None and int(stream_epoch) != self.stream_epoch:
            raise CrossEpochEpisodeError(
                f"observation epoch {stream_epoch} does not match {self.stream_epoch}"
            )
        if observation_status not in {"valid", "unavailable", "inference_error"}:
            raise ValueError(f"unsupported observation status: {observation_status}")
        if observation_status != "valid":
            return None

        source_time = float(source_time_sec)
        monotonic_time = float(monotonic_sec)
        self.last_observed_source_time_sec = source_time
        self.last_observed_monotonic_sec = monotonic_time
        level = int(risk_level)

        if level > 0:
            self._recovery_started_source_time_sec = None
            if self.state not in self._ACTIVE:
                self.episode_id = self._id_factory()
                self._peak_level = level
                self.peak_source_time_sec = source_time
                return self._transition(
                    EpisodeState.SUSPECTED,
                    source_time_sec=source_time,
                    monotonic_sec=monotonic_time,
                    trigger_condition=trigger_condition,
                )
            if level >= self._peak_level:
                self._peak_level = level
                self.peak_source_time_sec = source_time
            if static_condition and self.state == EpisodeState.SUSPECTED:
                return self._transition(
                    EpisodeState.CONFIRMED_STATIC,
                    source_time_sec=source_time,
                    monotonic_sec=monotonic_time,
                    trigger_condition=trigger_condition,
                )
            return None

        if self.state not in self._ACTIVE:
            return None
        if self._recovery_started_source_time_sec is None:
            self._recovery_started_source_time_sec = source_time
            return None
        if source_time - self._recovery_started_source_time_sec < self.recovery_confirmation_sec:
            return None
        return self._transition(
            EpisodeState.RECOVERED,
            source_time_sec=source_time,
            monotonic_sec=monotonic_time,
            trigger_condition="recovery_confirmation_elapsed",
        )

    def expire(self, *, monotonic_sec: float) -> EpisodeTransition | None:
        if self.state not in self._ACTIVE or self.last_observed_monotonic_sec is None:
            return None
        if float(monotonic_sec) - self.last_observed_monotonic_sec < self.episode_ttl_sec:
            return None
        return self._transition(
            EpisodeState.UNRESOLVED,
            source_time_sec=float(self.last_observed_source_time_sec or 0.0),
            monotonic_sec=float(monotonic_sec),
            trigger_condition="episode_observation_ttl_elapsed",
            cleanup_reason="observation_ttl_expired",
        )

    def clear(
        self,
        *,
        reason: str,
        source_time_sec: float,
        monotonic_sec: float,
    ) -> EpisodeTransition | None:
        self.last_cleanup_reason = str(reason)
        if self.state not in self._ACTIVE:
            return None
        return self._transition(
            EpisodeState.UNRESOLVED,
            source_time_sec=float(source_time_sec),
            monotonic_sec=float(monotonic_sec),
            trigger_condition="technical_cleanup",
            cleanup_reason=str(reason),
        )

    def transition_for_test(
        self,
        target: EpisodeState,
        *,
        source_time_sec: float,
        monotonic_sec: float,
        trigger_condition: str,
    ) -> EpisodeTransition:
        return self._transition(
            target,
            source_time_sec=source_time_sec,
            monotonic_sec=monotonic_sec,
            trigger_condition=trigger_condition,
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "episode_id": self.episode_id,
            "stream_epoch": self.stream_epoch,
            "entered_source_time_sec": self.entered_source_time_sec,
            "entered_monotonic_sec": self.entered_monotonic_sec,
            "peak_source_time_sec": self.peak_source_time_sec,
            "last_observed_source_time_sec": self.last_observed_source_time_sec,
            "last_observed_monotonic_sec": self.last_observed_monotonic_sec,
            "ttl_sec": self.episode_ttl_sec,
            "recovery_confirmation_sec": self.recovery_confirmation_sec,
            "last_cleanup_reason": self.last_cleanup_reason,
            "transitions": [
                {
                    "episode_id": item.episode_id,
                    "stream_epoch": item.stream_epoch,
                    "from_state": item.from_state.value,
                    "to_state": item.to_state.value,
                    "source_time_sec": item.source_time_sec,
                    "entered_source_time_sec": item.entered_source_time_sec,
                    "entered_monotonic_sec": item.entered_monotonic_sec,
                    "peak_source_time_sec": item.peak_source_time_sec,
                    "last_observed_source_time_sec": item.last_observed_source_time_sec,
                    "ttl_sec": item.ttl_sec,
                    "trigger_condition": item.trigger_condition,
                    "cleanup_reason": item.cleanup_reason,
                }
                for item in list(self.transitions)[-20:]
            ],
        }

    def _transition(
        self,
        target: EpisodeState,
        *,
        source_time_sec: float,
        monotonic_sec: float,
        trigger_condition: str,
        cleanup_reason: str | None = None,
    ) -> EpisodeTransition:
        if target == self.state:
            raise InvalidEpisodeTransition(f"repeated transition to {target.value}")
        if target not in self._LEGAL[self.state]:
            raise InvalidEpisodeTransition(
                f"illegal episode transition {self.state.value}->{target.value}"
            )
        if target == EpisodeState.SUSPECTED and self.episode_id is None:
            self.episode_id = self._id_factory()
        if self.episode_id is None:
            raise InvalidEpisodeTransition("episode_id is required for non-idle states")
        previous = self.state
        self.state = target
        self.entered_source_time_sec = float(source_time_sec)
        self.entered_monotonic_sec = float(monotonic_sec)
        self.last_cleanup_reason = cleanup_reason
        transition = EpisodeTransition(
            episode_id=self.episode_id,
            stream_epoch=self.stream_epoch,
            from_state=previous,
            to_state=target,
            source_time_sec=float(source_time_sec),
            entered_source_time_sec=float(source_time_sec),
            entered_monotonic_sec=float(monotonic_sec),
            peak_source_time_sec=float(self.peak_source_time_sec or source_time_sec),
            last_observed_source_time_sec=float(
                self.last_observed_source_time_sec
                if self.last_observed_source_time_sec is not None
                else source_time_sec
            ),
            ttl_sec=self.episode_ttl_sec,
            trigger_condition=str(trigger_condition),
            cleanup_reason=cleanup_reason,
        )
        self.transitions.append(transition)
        return transition


@dataclass(frozen=True)
class FallStateConfig:
    observation_window_sec: float = 1.0
    min_quality: float = 0.6
    hip_drop_threshold: float = 0.18
    center_drop_threshold: float = 0.15
    horizontal_angle_threshold: float = 60.0
    static_duration_sec: float = 10.0
    static_motion_threshold: float = 0.02
    recovery_upright_angle_threshold: float = 45.0
    recovery_motion_threshold: float = 0.05
    recovery_confirmation_sec: float = 3.0
    episode_ttl_sec: float = 30.0


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
        self._last_valid_timestamp: float | None = None

    def reset(self) -> None:
        self._history.clear()
        self._suspected_at = None
        self._static_since = None
        self._last_valid_timestamp = None

    def update(self, observation: Mapping[str, Any]) -> FallStateResult:
        current = {key: float(observation.get(key, default)) for key, default in (
            ("timestamp_sec", 0.0), ("hip_center_y", 0.0), ("bbox_center_y", 0.0),
            ("trunk_angle_deg", 0.0), ("core_keypoint_quality", 0.0), ("motion_score", 1.0),
        )}
        now = current["timestamp_sec"]
        triggered = False
        valid_observation = current["core_keypoint_quality"] >= self.config.min_quality
        if valid_observation:
            while (
                self._history
                and now - self._history[0]["timestamp_sec"]
                > self.config.observation_window_sec
            ):
                self._history.popleft()
        reference = self._history[0] if self._history else None
        if reference is not None and valid_observation:
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
        if valid_observation:
            self._history.append(current)
            self._last_valid_timestamp = now

        if self._suspected_at is not None and valid_observation:
            if (
                current["trunk_angle_deg"]
                <= self.config.recovery_upright_angle_threshold
                and current["motion_score"] >= self.config.recovery_motion_threshold
            ):
                self._suspected_at = None
                self._static_since = None
            elif current["motion_score"] <= self.config.static_motion_threshold:
                if self._static_since is None:
                    self._static_since = now
            else:
                self._static_since = None
        static_duration = (
            0.0
            if self._static_since is None or self._last_valid_timestamp is None
            else self._last_valid_timestamp - self._static_since
        )
        return FallStateResult(
            fall_event_score=0.9 if self._suspected_at is not None else 0.0,
            long_static_score=0.9 if static_duration >= self.config.static_duration_sec else 0.0,
            suspected_fall=self._suspected_at is not None,
            triggered_now=triggered,
        )
