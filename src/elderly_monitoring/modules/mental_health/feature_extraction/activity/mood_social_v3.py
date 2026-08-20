from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from numbers import Real
from statistics import median
from typing import Any, Iterable, Mapping
from zoneinfo import ZoneInfo

from elderly_monitoring.modules.mental_health.feature_extraction.activity.daytime import (
    ActivityFrame,
    DaytimeActivityConfig,
    _adapt_activity_frame,
    _build_window,
    _window_epoch,
)
from elderly_monitoring.modules.mental_health.mood_social.config import (
    MoodSocialConfig,
    load_mood_social_config,
)
from elderly_monitoring.modules.mental_health.mood_social.schemas import (
    MoodSocialActivity,
)


__all__ = [
    "aggregate_mood_social_camera_daily",
    "aggregate_mood_social_camera_windows",
    "extract_mood_social_camera_features",
]


_VALID_DETECTION_RATIO_THRESHOLD = 0.60
_FRAME_SUPPORT_SECONDS = 1.0
_MAX_GAIT_POINT_GAP_SECONDS = 1.0
_MIN_GAIT_KEYPOINT_SCORE = 0.45
_CORE_KEYPOINTS = (
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
_FORBIDDEN_ACTIVITY_FIELDS = {
    "daily_step_count",
    "sedentary_total_minutes",
    "hourly_activity_vector",
    "gait_speed_mps",
    "walking_speed_norm",
    "walking_speed_norm_camera",
}
_FORBIDDEN_GAIT_FIELDS = {
    "daily_step_count",
    "gait_speed_mps",
    "walking_speed_norm",
    "walking_speed_norm_camera",
}


@dataclass(frozen=True)
class _CameraRuntime:
    timezone: ZoneInfo
    window_seconds: int
    daytime_start_hour: int
    daytime_end_hour: int
    active_score_threshold: float
    low_activity_score_threshold: float
    sedentary_min_seconds: float
    daytime_config: DaytimeActivityConfig


@dataclass(frozen=True)
class _RawFrame:
    frame: ActivityFrame
    source: Mapping[str, Any]
    camera_id: str
    scene_version: str
    identity_confidence: float
    pose_validity: float
    tracking_confidence: float

    @property
    def rank(self) -> tuple[int, float, float, float]:
        return (
            int(self.frame.valid_detection),
            self.identity_confidence,
            self.pose_validity,
            self.tracking_confidence,
        )


@dataclass(frozen=True)
class _CameraWindow:
    window_start: datetime
    window_end: datetime
    person_id: str
    camera_id: str
    scene_version: str
    active_score: float | None
    valid_detection_ratio: float
    data_quality: str
    identity_confidence: float
    pose_validity: float
    tracking_confidence: float

    @property
    def is_valid(self) -> bool:
        return (
            self.data_quality == "valid"
            and self.active_score is not None
            and self.valid_detection_ratio >= _VALID_DETECTION_RATIO_THRESHOLD
        )

    @property
    def selection_key(self) -> tuple[float, float, float, str, str]:
        return (
            -self.identity_confidence,
            -self.pose_validity,
            -self.tracking_confidence,
            self.camera_id,
            self.scene_version,
        )


@dataclass
class _GaitAccumulator:
    gait_speeds: list[float] = field(default_factory=list)
    sit_to_stand_durations: list[float] = field(default_factory=list)
    turn_durations: list[float] = field(default_factory=list)
    direct_stabilities: list[float] = field(default_factory=list)
    gait_stabilities: list[float] = field(default_factory=list)
    turn_stabilities: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class _GaitPosePoint:
    observed_at: datetime
    center: tuple[float, float] | None


def aggregate_mood_social_camera_windows(
    records: Iterable[Mapping[str, Any]],
    *,
    config: MoodSocialConfig | None = None,
) -> list[dict[str, Any]]:
    """Build per-camera, per-scene 10-second candidates from raw pose frames.

    Coverage is based on the union of one-second support intervals, so duplicate
    frames at one timestamp cannot turn a partial observation into a valid
    10-second window. Cross-camera de-duplication intentionally happens later,
    after each camera/scene has produced an independent activity score.
    """

    runtime = _runtime(config)
    source_records = list(records)
    if not source_records:
        raise ValueError("No camera activity frame records were provided")

    grouped: dict[tuple[str, str, str, float], list[_RawFrame]] = defaultdict(list)
    for record_number, record in enumerate(source_records, start=1):
        _require_mapping(record, f"camera activity frame {record_number}")
        _reject_forbidden_fields(
            record,
            _FORBIDDEN_ACTIVITY_FIELDS,
            f"camera activity frame {record_number}",
        )
        frame = _adapt_activity_frame(
            record,
            config=runtime.daytime_config,
            record_number=record_number,
        )
        camera_id = _required_text(
            frame.camera_id,
            f"camera activity frame {record_number} field 'camera_id'",
        )
        scene_version = _required_text(
            record.get("scene_version"),
            f"camera activity frame {record_number} field 'scene_version'",
        )
        raw_frame = _RawFrame(
            frame=frame,
            source=record,
            camera_id=camera_id,
            scene_version=scene_version,
            identity_confidence=_confidence(
                record.get("identity_confidence"),
                default=0.0,
                field=f"camera activity frame {record_number} identity_confidence",
            ),
            pose_validity=_pose_validity(record, frame),
            tracking_confidence=_confidence(
                record.get("tracking_confidence"),
                default=0.0,
                field=f"camera activity frame {record_number} tracking_confidence",
            ),
        )
        grouped[
            (
                frame.person_id,
                camera_id,
                scene_version,
                _window_epoch(frame.observed_at, runtime.daytime_config),
            )
        ].append(raw_frame)

    windows = [
        _build_camera_window(key, frames, runtime)
        for key, frames in grouped.items()
    ]
    return [
        _camera_window_to_dict(window)
        for window in sorted(
            windows,
            key=lambda item: (
                item.window_start,
                item.person_id,
                item.camera_id,
                item.scene_version,
            ),
        )
    ]


def aggregate_mood_social_camera_daily(
    windows: Iterable[Mapping[str, Any]],
    *,
    camera_gait_records: Iterable[Mapping[str, Any]] = (),
    config: MoodSocialConfig | None = None,
) -> list[dict[str, Any]]:
    """Aggregate camera candidates into strict V3.3.3 daily activity objects."""

    runtime = _runtime(config)
    source_windows = list(windows)
    if not source_windows:
        raise ValueError(
            "No camera activity windows were provided; represent a completely "
            "missing camera day as activity=null in the caller"
        )

    parsed = [
        _parse_camera_window(record, runtime, record_number=index)
        for index, record in enumerate(source_windows, start=1)
    ]
    _reject_duplicate_scene_windows(parsed)
    gait_by_day = _aggregate_camera_gait_records(camera_gait_records, runtime)

    buckets: dict[tuple[str, date], list[_CameraWindow]] = defaultdict(list)
    for window in parsed:
        local_start = window.window_start.astimezone(runtime.timezone)
        buckets[(window.person_id, local_start.date())].append(window)

    outputs: list[dict[str, Any]] = []
    for (person_id, day), day_windows in sorted(
        buckets.items(),
        key=lambda item: (item[0][1], item[0][0]),
    ):
        activity = _finalize_camera_day(
            day_windows,
            gait_by_day.get((person_id, day), []),
            runtime,
        )
        validated = MoodSocialActivity.model_validate(activity)
        outputs.append(
            {
                "person_id": person_id,
                "date": day.isoformat(),
                "activity": validated.model_dump(mode="python"),
            }
        )
    return outputs


def extract_mood_social_camera_features(
    records: Iterable[Mapping[str, Any]],
    *,
    camera_gait_records: Iterable[Mapping[str, Any]] = (),
    config: MoodSocialConfig | None = None,
) -> list[dict[str, Any]]:
    """Run the V3 camera path from raw frames or pre-aggregated candidates."""

    source_records = list(records)
    if not source_records:
        raise ValueError(
            "No camera records were provided; represent a completely missing "
            "camera day as activity=null in the caller"
        )
    for index, record in enumerate(source_records, start=1):
        _require_mapping(record, f"camera record {index}")
    mood_config = config or load_mood_social_config()
    are_windows = ["window_start" in record for record in source_records]
    if any(are_windows) and not all(are_windows):
        raise ValueError("Camera records must be all raw frames or all activity windows")
    windows = (
        source_records
        if all(are_windows)
        else aggregate_mood_social_camera_windows(source_records, config=mood_config)
    )
    return aggregate_mood_social_camera_daily(
        windows,
        camera_gait_records=camera_gait_records,
        config=mood_config,
    )


def _runtime(config: MoodSocialConfig | None) -> _CameraRuntime:
    mood_config = config or load_mood_social_config()
    camera = mood_config.camera
    timezone = mood_config.runtime.timezone_info
    daytime_config = DaytimeActivityConfig(
        timezone=mood_config.runtime.timezone,
        window_seconds=camera.activity_window_seconds,
        daytime_start=camera.daytime_start,
        daytime_end=camera.daytime_end,
        valid_detection_ratio_threshold=_VALID_DETECTION_RATIO_THRESHOLD,
        active_score_threshold=camera.active_score_threshold,
        low_motion_score_threshold=camera.low_activity_score_threshold,
        sedentary_min_minutes=float(camera.sedentary_min_minutes),
        center_motion_weight=camera.active_score_weights.center_motion_score,
        pose_motion_weight=camera.active_score_weights.pose_motion_score,
        zone_transition_weight=camera.active_score_weights.zone_transition_score,
        posture_change_weight=camera.active_score_weights.posture_change_score,
    )
    return _CameraRuntime(
        timezone=timezone,
        window_seconds=camera.activity_window_seconds,
        daytime_start_hour=camera.daytime_start_time.hour,
        daytime_end_hour=camera.daytime_end_time.hour,
        active_score_threshold=camera.active_score_threshold,
        low_activity_score_threshold=camera.low_activity_score_threshold,
        sedentary_min_seconds=float(camera.sedentary_min_minutes * 60),
        daytime_config=daytime_config,
    )


def _build_camera_window(
    key: tuple[str, str, str, float],
    raw_frames: list[_RawFrame],
    runtime: _CameraRuntime,
) -> _CameraWindow:
    person_id, camera_id, scene_version, window_epoch = key
    frames_by_timestamp: dict[float, _RawFrame] = {}
    for raw_frame in raw_frames:
        timestamp = raw_frame.frame.observed_at.timestamp()
        existing = frames_by_timestamp.get(timestamp)
        if existing is None or raw_frame.rank > existing.rank:
            frames_by_timestamp[timestamp] = raw_frame
    distinct_frames = sorted(
        frames_by_timestamp.values(),
        key=lambda item: item.frame.observed_at,
    )
    valid_raw_frames = [
        raw_frame
        for raw_frame in distinct_frames
        if raw_frame.frame.valid_detection
    ]
    window_start = datetime.fromtimestamp(window_epoch, tz=runtime.timezone)
    window_end = window_start + timedelta(seconds=runtime.window_seconds)
    valid_support_seconds = _support_seconds(
        [item.frame.observed_at for item in valid_raw_frames],
        window_start,
        window_end,
    )
    valid_ratio = min(1.0, valid_support_seconds / runtime.window_seconds)

    built = None
    if valid_raw_frames:
        built = _build_window(
            person_id,
            window_epoch,
            [item.frame for item in valid_raw_frames],
            runtime.daytime_config,
        )
    has_two_times = len(valid_raw_frames) >= 2
    has_complete_motion = (
        built is not None
        and built.center_path_norm is not None
        and built.pose_motion_norm is not None
        and built.active_score is not None
    )
    is_valid = (
        valid_ratio >= _VALID_DETECTION_RATIO_THRESHOLD
        and has_two_times
        and has_complete_motion
    )

    return _CameraWindow(
        window_start=window_start,
        window_end=window_end,
        person_id=person_id,
        camera_id=camera_id,
        scene_version=scene_version,
        active_score=float(built.active_score) if is_valid and built is not None else None,
        valid_detection_ratio=valid_ratio,
        data_quality="valid" if is_valid else _invalid_window_quality(raw_frames),
        identity_confidence=_median_confidence(
            item.identity_confidence for item in valid_raw_frames
        ),
        pose_validity=_median_confidence(
            item.pose_validity for item in valid_raw_frames
        ),
        tracking_confidence=_median_confidence(
            item.tracking_confidence for item in valid_raw_frames
        ),
    )


def _camera_window_to_dict(window: _CameraWindow) -> dict[str, Any]:
    return {
        "window_start": window.window_start.isoformat(),
        "window_end": window.window_end.isoformat(),
        "person_id": window.person_id,
        "camera_id": window.camera_id,
        "scene_version": window.scene_version,
        "active_score": _round_optional(window.active_score),
        "valid_detection_ratio": _round(window.valid_detection_ratio),
        "data_quality": window.data_quality,
        "identity_confidence": _round(window.identity_confidence),
        "pose_validity": _round(window.pose_validity),
        "tracking_confidence": _round(window.tracking_confidence),
    }


def _parse_camera_window(
    record: Mapping[str, Any],
    runtime: _CameraRuntime,
    *,
    record_number: int,
) -> _CameraWindow:
    label = f"camera activity window {record_number}"
    _require_mapping(record, label)
    _reject_forbidden_fields(record, _FORBIDDEN_ACTIVITY_FIELDS, label)
    start = _aware_datetime(record.get("window_start"), f"{label} window_start", runtime.timezone)
    end = _aware_datetime(record.get("window_end"), f"{label} window_end", runtime.timezone)
    duration = (end - start).total_seconds()
    if not math.isclose(duration, runtime.window_seconds, abs_tol=1e-6):
        raise ValueError(f"{label} must be exactly {runtime.window_seconds} seconds")
    remainder = start.timestamp() % runtime.window_seconds
    if not (
        math.isclose(remainder, 0.0, abs_tol=1e-6)
        or math.isclose(remainder, runtime.window_seconds, abs_tol=1e-6)
    ):
        raise ValueError(f"{label} window_start must align to the 10-second epoch grid")

    score = _optional_probability(record.get("active_score"), f"{label} active_score")
    ratio = _optional_probability(
        record.get("valid_detection_ratio"),
        f"{label} valid_detection_ratio",
    )
    if ratio is None:
        raise ValueError(f"{label} valid_detection_ratio is required")
    data_quality = _required_text(
        record.get("data_quality", "valid"),
        f"{label} data_quality",
    ).lower()
    is_valid = (
        data_quality == "valid"
        and score is not None
        and ratio >= _VALID_DETECTION_RATIO_THRESHOLD
    )
    return _CameraWindow(
        window_start=start,
        window_end=end,
        person_id=_required_text(record.get("person_id"), f"{label} person_id"),
        camera_id=_required_text(record.get("camera_id"), f"{label} camera_id"),
        scene_version=_required_text(
            record.get("scene_version"),
            f"{label} scene_version",
        ),
        active_score=score if is_valid else None,
        valid_detection_ratio=ratio,
        data_quality=(
            "valid"
            if is_valid
            else (
                data_quality
                if data_quality != "valid"
                else "insufficient_coverage"
            )
        ),
        identity_confidence=_confidence(
            record.get("identity_confidence"),
            default=0.0,
            field=f"{label} identity_confidence",
        ),
        pose_validity=_confidence(
            record.get("pose_validity"),
            default=0.0,
            field=f"{label} pose_validity",
        ),
        tracking_confidence=_confidence(
            record.get("tracking_confidence"),
            default=0.0,
            field=f"{label} tracking_confidence",
        ),
    )


def _reject_duplicate_scene_windows(windows: list[_CameraWindow]) -> None:
    seen: set[tuple[str, datetime, str, str]] = set()
    for window in windows:
        key = (
            window.person_id,
            window.window_start,
            window.camera_id,
            window.scene_version,
        )
        if key in seen:
            raise ValueError(
                "Duplicate camera activity candidate for the same person, "
                "camera_id, scene_version, and 10-second window"
            )
        seen.add(key)


def _finalize_camera_day(
    windows: list[_CameraWindow],
    gait_metrics: list[dict[str, Any]],
    runtime: _CameraRuntime,
) -> dict[str, Any]:
    daytime_windows = [
        window
        for window in windows
        if _is_daytime_window(window, runtime)
    ]
    candidates_by_slot: dict[tuple[str, datetime], list[_CameraWindow]] = defaultdict(list)
    for window in daytime_windows:
        candidates_by_slot[(window.person_id, window.window_start)].append(window)

    selected: list[_CameraWindow] = []
    for candidates in candidates_by_slot.values():
        valid_candidates = [candidate for candidate in candidates if candidate.is_valid]
        pool = valid_candidates or candidates
        selected.append(min(pool, key=lambda item: item.selection_key))
    selected.sort(key=lambda item: item.window_start)
    valid = [window for window in selected if window.is_valid]

    hourly_coverage_seconds = [0.0] * 24
    hourly_weighted_seconds = [0.0] * 24
    for window in valid:
        hour = window.window_start.astimezone(runtime.timezone).hour
        hourly_coverage_seconds[hour] += runtime.window_seconds
        hourly_weighted_seconds[hour] += (
            float(window.active_score) * runtime.window_seconds
        )

    valid_seconds = sum(hourly_coverage_seconds)
    if valid_seconds <= 0.0:
        return {
            "daytime_active_minutes": None,
            "weighted_daytime_activity": None,
            "valid_daytime_detection_minutes": 0.0,
            "low_activity_minutes": None,
            "sedentary_bout_total_minutes": None,
            "longest_sedentary_bout_minutes": None,
            "activity_peak_minute_of_day": None,
            "hourly_activity_intensity": [None] * 24,
            "hourly_valid_detection_minutes": [0.0] * 24,
            "camera_gait_metrics": [],
        }

    active_seconds = sum(
        runtime.window_seconds
        for window in valid
        if float(window.active_score) >= runtime.active_score_threshold
    )
    low_seconds = sum(
        runtime.window_seconds
        for window in valid
        if float(window.active_score) <= runtime.low_activity_score_threshold
    )
    weighted_seconds = sum(
        float(window.active_score) * runtime.window_seconds
        for window in valid
    )
    bout_seconds = _low_activity_bouts(selected, runtime)
    hourly_coverage_minutes = [
        _round(seconds / 60.0)
        for seconds in hourly_coverage_seconds
    ]
    hourly_intensity = [
        (
            _round(hourly_weighted_seconds[hour] / coverage_seconds, digits=6)
            if coverage_seconds > 0.0
            else None
        )
        for hour, coverage_seconds in enumerate(hourly_coverage_seconds)
    ]
    return {
        "daytime_active_minutes": _round(active_seconds / 60.0),
        "weighted_daytime_activity": _round(weighted_seconds / 60.0),
        "valid_daytime_detection_minutes": _round(valid_seconds / 60.0),
        "low_activity_minutes": _round(low_seconds / 60.0),
        "sedentary_bout_total_minutes": _round(sum(bout_seconds) / 60.0),
        "longest_sedentary_bout_minutes": _round(
            max(bout_seconds, default=0.0) / 60.0
        ),
        "activity_peak_minute_of_day": _activity_peak_minute(valid, runtime),
        "hourly_activity_intensity": hourly_intensity,
        "hourly_valid_detection_minutes": hourly_coverage_minutes,
        "camera_gait_metrics": gait_metrics,
    }


def _is_daytime_window(window: _CameraWindow, runtime: _CameraRuntime) -> bool:
    local_start = window.window_start.astimezone(runtime.timezone)
    local_end = window.window_end.astimezone(runtime.timezone)
    day_start = local_start.replace(
        hour=runtime.daytime_start_hour,
        minute=0,
        second=0,
        microsecond=0,
    )
    day_end = local_start.replace(
        hour=runtime.daytime_end_hour,
        minute=0,
        second=0,
        microsecond=0,
    )
    return day_start <= local_start and local_end <= day_end


def _low_activity_bouts(
    selected: list[_CameraWindow],
    runtime: _CameraRuntime,
) -> list[float]:
    bouts: list[float] = []
    current_seconds = 0.0
    previous_end: datetime | None = None
    for window in selected:
        contiguous = previous_end is not None and window.window_start == previous_end
        low_activity = (
            window.is_valid
            and float(window.active_score) <= runtime.low_activity_score_threshold
        )
        if low_activity and (previous_end is None or contiguous):
            current_seconds += runtime.window_seconds
        else:
            if current_seconds >= runtime.sedentary_min_seconds:
                bouts.append(current_seconds)
            current_seconds = runtime.window_seconds if low_activity else 0.0
        previous_end = window.window_end
    if current_seconds >= runtime.sedentary_min_seconds:
        bouts.append(current_seconds)
    return bouts


def _activity_peak_minute(
    valid_windows: list[_CameraWindow],
    runtime: _CameraRuntime,
) -> int | None:
    minute_mass: dict[int, float] = defaultdict(float)
    for window in valid_windows:
        local_start = window.window_start.astimezone(runtime.timezone)
        minute_of_day = local_start.hour * 60 + local_start.minute
        minute_mass[minute_of_day] += (
            float(window.active_score) * runtime.window_seconds / 60.0
        )
    if not minute_mass:
        return None
    peak_mass = max(minute_mass.values())
    if peak_mass <= 0.0:
        return None
    return min(
        minute
        for minute, mass in minute_mass.items()
        if math.isclose(mass, peak_mass, abs_tol=1e-12)
    )


def _aggregate_camera_gait_records(
    records: Iterable[Mapping[str, Any]],
    runtime: _CameraRuntime,
) -> dict[tuple[str, date], list[dict[str, Any]]]:
    accumulators: dict[
        tuple[str, date, str, str],
        _GaitAccumulator,
    ] = defaultdict(_GaitAccumulator)
    pose_points: dict[
        tuple[str, date, str, str, str, str],
        list[_GaitPosePoint],
    ] = defaultdict(list)

    for record_number, record in enumerate(records, start=1):
        label = f"camera gait record {record_number}"
        _require_mapping(record, label)
        _reject_forbidden_fields(record, _FORBIDDEN_GAIT_FIELDS, label)
        person_id = _required_text(record.get("person_id"), f"{label} person_id")
        camera_id = _required_text(record.get("camera_id"), f"{label} camera_id")
        scene_version = _required_text(
            record.get("scene_version"),
            f"{label} scene_version",
        )
        observed_at = _optional_record_datetime(record, runtime.timezone, label)
        day = _record_day(record, observed_at, runtime.timezone, label)
        if observed_at is not None and not _is_daytime_timestamp(observed_at, runtime):
            continue
        key = (person_id, day, camera_id, scene_version)
        accumulator = accumulators[key]
        record_is_valid = _gait_frame_is_valid(record)

        speed = (
            _metric_value(
                record,
                ("gait_speed_image_norm_per_sec", "gait_speed_norm_per_sec"),
                minimum=0.0,
                strictly_greater=False,
                field=f"{label} gait speed",
            )
            if record_is_valid
            else None
        )
        sit_to_stand = (
            _metric_value(
                record,
                ("sit_to_stand_duration_seconds",),
                minimum=0.0,
                strictly_greater=True,
                field=f"{label} sit-to-stand duration",
            )
            if record_is_valid
            else None
        )
        if (
            record_is_valid
            and sit_to_stand is None
            and str(record.get("transition_type") or "") == "sit_to_stand"
        ):
            sit_to_stand = _metric_value(
                record,
                ("duration",),
                minimum=0.0,
                strictly_greater=True,
                field=f"{label} sit-to-stand duration",
            )
        turn_duration = (
            _metric_value(
                record,
                ("turn_duration_seconds",),
                minimum=0.0,
                strictly_greater=True,
                field=f"{label} turn duration",
            )
            if record_is_valid
            else None
        )
        direct_stability = (
            _metric_value(
                record,
                ("postural_stability",),
                minimum=0.0,
                maximum=1.0,
                strictly_greater=False,
                field=f"{label} postural stability",
            )
            if record_is_valid
            else None
        )
        gait_stability = (
            _metric_value(
                record,
                ("gait_cycle_stability_score",),
                minimum=0.0,
                maximum=1.0,
                strictly_greater=False,
                field=f"{label} gait stability",
            )
            if record_is_valid
            else None
        )
        turn_stability = (
            _metric_value(
                record,
                ("turn_stability_score",),
                minimum=0.0,
                maximum=1.0,
                strictly_greater=False,
                field=f"{label} turn stability",
            )
            if record_is_valid
            else None
        )
        _append_optional(accumulator.gait_speeds, speed)
        _append_optional(accumulator.sit_to_stand_durations, sit_to_stand)
        _append_optional(accumulator.turn_durations, turn_duration)
        _append_optional(accumulator.direct_stabilities, direct_stability)
        _append_optional(accumulator.gait_stabilities, gait_stability)
        _append_optional(accumulator.turn_stabilities, turn_stability)

        walking_segment = record.get(
            "walking_segment_id",
            record.get("gait_segment_id"),
        )
        if observed_at is not None and walking_segment is not None and speed is None:
            segment_id = _required_text(
                walking_segment,
                f"{label} walking_segment_id",
            )
            track_id = str(record.get("track_id") or "__single_track__")
            center = _hip_center(record) if record_is_valid else None
            pose_points[
                (
                    person_id,
                    day,
                    camera_id,
                    scene_version,
                    segment_id,
                    track_id,
                )
            ].append(_GaitPosePoint(observed_at=observed_at, center=center))

    for pose_key, points in pose_points.items():
        person_id, day, camera_id, scene_version, _, _ = pose_key
        accumulator = accumulators[(person_id, day, camera_id, scene_version)]
        by_timestamp: dict[datetime, _GaitPosePoint] = {}
        for point in points:
            existing = by_timestamp.get(point.observed_at)
            if existing is None or (existing.center is None and point.center is not None):
                by_timestamp[point.observed_at] = point
        ordered = sorted(by_timestamp.values(), key=lambda item: item.observed_at)
        for left, right in zip(ordered, ordered[1:]):
            delta_seconds = (right.observed_at - left.observed_at).total_seconds()
            if (
                left.center is None
                or right.center is None
                or not 0.0 < delta_seconds <= _MAX_GAIT_POINT_GAP_SECONDS
            ):
                continue
            speed = math.hypot(
                right.center[0] - left.center[0],
                right.center[1] - left.center[1],
            ) / delta_seconds
            accumulator.gait_speeds.append(speed)

    by_day: dict[tuple[str, date], list[dict[str, Any]]] = defaultdict(list)
    for (person_id, day, camera_id, scene_version), values in sorted(
        accumulators.items(),
        key=lambda item: item[0],
    ):
        stability_values = (
            values.direct_stabilities
            or values.gait_stabilities
            or values.turn_stabilities
        )
        metric = {
            "camera_id": camera_id,
            "scene_version": scene_version,
            "gait_speed_image_norm_per_sec": _median_optional(values.gait_speeds),
            "sit_to_stand_duration_seconds": _median_optional(
                values.sit_to_stand_durations
            ),
            "turn_duration_seconds": _median_optional(values.turn_durations),
            "postural_stability": _median_optional(stability_values),
        }
        if all(value is None for value in tuple(metric.values())[2:]):
            continue
        by_day[(person_id, day)].append(metric)
    return by_day


def _support_seconds(
    timestamps: list[datetime],
    window_start: datetime,
    window_end: datetime,
) -> float:
    intervals = sorted(
        (
            max(timestamp, window_start),
            min(timestamp + timedelta(seconds=_FRAME_SUPPORT_SECONDS), window_end),
        )
        for timestamp in timestamps
        if timestamp < window_end
        and timestamp + timedelta(seconds=_FRAME_SUPPORT_SECONDS) > window_start
    )
    total = 0.0
    current_start: datetime | None = None
    current_end: datetime | None = None
    for start, end in intervals:
        if end <= start:
            continue
        if current_start is None:
            current_start, current_end = start, end
        elif start <= current_end:
            current_end = max(current_end, end)
        else:
            total += (current_end - current_start).total_seconds()
            current_start, current_end = start, end
    if current_start is not None and current_end is not None:
        total += (current_end - current_start).total_seconds()
    return total


def _pose_validity(record: Mapping[str, Any], frame: ActivityFrame) -> float:
    explicit = record.get("pose_validity")
    if explicit is not None:
        return _confidence(explicit, default=0.0, field="pose_validity")
    names = {name for name, _, _ in frame.keypoints}
    return len(names.intersection(_CORE_KEYPOINTS)) / len(_CORE_KEYPOINTS)


def _invalid_window_quality(raw_frames: list[_RawFrame]) -> str:
    states = {
        str(item.frame.quality_state).strip().lower()
        for item in raw_frames
        if str(item.frame.quality_state).strip()
    }
    if len(states) == 1:
        state = next(iter(states))
        if state != "valid":
            return state
    return "insufficient_coverage"


def _median_confidence(values: Iterable[float]) -> float:
    collected = list(values)
    return float(median(collected)) if collected else 0.0


def _optional_record_datetime(
    record: Mapping[str, Any],
    timezone: ZoneInfo,
    label: str,
) -> datetime | None:
    for field_name in (
        "observed_at",
        "timestamp",
        "start_time",
        "window_start",
    ):
        value = record.get(field_name)
        if value is not None:
            return _aware_datetime(value, f"{label} {field_name}", timezone)
    return None


def _record_day(
    record: Mapping[str, Any],
    observed_at: datetime | None,
    timezone: ZoneInfo,
    label: str,
) -> date:
    raw_day = record.get("date")
    parsed_day: date | None = None
    if isinstance(raw_day, date) and not isinstance(raw_day, datetime):
        parsed_day = raw_day
    elif isinstance(raw_day, str):
        try:
            parsed_day = date.fromisoformat(raw_day)
        except ValueError as exc:
            raise ValueError(f"{label} date must use YYYY-MM-DD") from exc
    elif raw_day is not None:
        raise ValueError(f"{label} date must use YYYY-MM-DD")
    observed_day = observed_at.astimezone(timezone).date() if observed_at is not None else None
    if parsed_day is not None and observed_day is not None and parsed_day != observed_day:
        raise ValueError(f"{label} date does not match its timestamp in Asia/Shanghai")
    if parsed_day is not None:
        return parsed_day
    if observed_day is not None:
        return observed_day
    raise ValueError(f"{label} requires date or a timezone-aware timestamp")


def _is_daytime_timestamp(value: datetime, runtime: _CameraRuntime) -> bool:
    local = value.astimezone(runtime.timezone)
    return runtime.daytime_start_hour <= local.hour < runtime.daytime_end_hour


def _hip_center(record: Mapping[str, Any]) -> tuple[float, float] | None:
    raw_keypoints = record.get("keypoints")
    if not isinstance(raw_keypoints, (list, tuple, Mapping)):
        return None
    points: dict[str, Mapping[str, Any]] = {}
    if isinstance(raw_keypoints, Mapping):
        for name, value in raw_keypoints.items():
            if isinstance(value, Mapping):
                points[str(name)] = value
    else:
        for value in raw_keypoints:
            if isinstance(value, Mapping) and isinstance(value.get("name"), str):
                points[str(value["name"])] = value
    left = _normalized_keypoint(points.get("left_hip"), record)
    right = _normalized_keypoint(points.get("right_hip"), record)
    if left is None or right is None:
        return None
    return ((left[0] + right[0]) / 2.0, (left[1] + right[1]) / 2.0)


def _normalized_keypoint(
    keypoint: Mapping[str, Any] | None,
    record: Mapping[str, Any],
) -> tuple[float, float] | None:
    if keypoint is None or keypoint.get("valid") is False:
        return None
    score = keypoint.get("score")
    if score is not None:
        parsed_score = _number(score, "gait keypoint score")
        if parsed_score < _MIN_GAIT_KEYPOINT_SCORE:
            return None
    x_value = keypoint.get("x_smooth")
    y_value = keypoint.get("y_smooth")
    if x_value is None or y_value is None:
        x_value, y_value = keypoint.get("x"), keypoint.get("y")
    if x_value is None or y_value is None:
        return None
    x = _number(x_value, "gait keypoint x")
    y = _number(y_value, "gait keypoint y")
    if abs(x) > 1.0:
        width = _positive_number(record.get("image_width"), "image_width")
        if width is None:
            return None
        x /= width
    if abs(y) > 1.0:
        height = _positive_number(record.get("image_height"), "image_height")
        if height is None:
            return None
        y /= height
    if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
        return None
    return x, y


def _gait_frame_is_valid(record: Mapping[str, Any]) -> bool:
    quality = str(
        record.get("data_quality", record.get("quality_state", "valid"))
    ).strip().lower()
    if quality not in {"valid", "good", "high"}:
        return False
    window_quality = record.get("window_quality")
    if isinstance(window_quality, Mapping) and window_quality.get("usable_for_gait") is False:
        return False
    return record.get("quality_rejected") is not True and record.get("rejected") is not True


def _metric_value(
    record: Mapping[str, Any],
    fields: tuple[str, ...],
    *,
    minimum: float,
    strictly_greater: bool,
    field: str,
    maximum: float | None = None,
) -> float | None:
    raw_value = None
    found = False
    for field_name in fields:
        if field_name in record:
            found = True
            if record[field_name] is not None:
                raw_value = record[field_name]
                break
    if not found or raw_value is None:
        return None
    value = _number(raw_value, field)
    if strictly_greater and value <= minimum:
        raise ValueError(f"{field} must be greater than {minimum}")
    if not strictly_greater and value < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{field} must not exceed {maximum}")
    return value


def _append_optional(values: list[float], value: float | None) -> None:
    if value is not None:
        values.append(value)


def _median_optional(values: list[float]) -> float | None:
    return _round(float(median(values))) if values else None


def _reject_forbidden_fields(
    record: Mapping[str, Any],
    forbidden: set[str],
    label: str,
) -> None:
    present = sorted(forbidden.intersection(record))
    if present:
        raise ValueError(f"{label} contains forbidden V3 fields: {', '.join(present)}")


def _require_mapping(value: Any, label: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _aware_datetime(value: Any, field: str, timezone: ZoneInfo) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        normalized = value.strip().replace(" ", "T")
        if normalized.endswith("Z"):
            normalized = normalized[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError as exc:
            raise ValueError(f"{field} must be a timezone-aware ISO-8601 timestamp") from exc
    else:
        raise ValueError(f"{field} must be a timezone-aware ISO-8601 timestamp")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must include a timezone")
    return parsed.astimezone(timezone)


def _optional_probability(value: Any, field: str) -> float | None:
    if value is None:
        return None
    number = _number(value, field)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{field} must be between 0 and 1")
    return number


def _confidence(
    value: Any,
    *,
    default: float,
    field: str,
) -> float:
    parsed = _optional_probability(value, field)
    return default if parsed is None else parsed


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be a finite number")
    return number


def _positive_number(value: Any, field: str) -> float | None:
    if value is None:
        return None
    parsed = _number(value, field)
    return parsed if parsed > 0.0 else None


def _round(value: float, *, digits: int = 4) -> float:
    return round(float(value), digits)


def _round_optional(value: float | None) -> float | None:
    return None if value is None else _round(value)
