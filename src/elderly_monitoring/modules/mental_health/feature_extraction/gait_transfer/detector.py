from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from numbers import Real
from statistics import fmean
from typing import Any, Iterable, Mapping
from zoneinfo import ZoneInfo

from elderly_monitoring.modules.fall_risk.gait import (
    GaitAnalysisConfig,
    extract_gait_windows,
)
from elderly_monitoring.modules.fall_risk.sit_stand import (
    SitStandAnalysisConfig,
    extract_sit_stand_events,
)


CORE_KEYPOINTS = (
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
MODEL_VERSION = "mental-cognitive-gait-v0.1"


@dataclass(frozen=True)
class CognitiveGaitConfig:
    """Engineering thresholds for motor-cognitive clue features, not diagnosis."""

    timezone: str = "Asia/Shanghai"
    min_keypoint_quality: float = 0.45
    min_turn_points: int = 5
    turn_angle_degrees: float = 70.0
    turn_merge_gap_seconds: float = 0.8
    max_point_gap_seconds: float = 1.0
    pause_speed_threshold_norm_per_sec: float = 0.03
    speed_cv_high_risk: float = 0.60
    turn_sway_high_risk: float = 0.08
    gait_speed_low_norm_per_sec: float = 0.08
    sit_stand_slow_seconds: float = 4.0
    turn_slow_seconds: float = 3.0
    gait_window_frames: int = 10

    @property
    def timezone_info(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)


@dataclass(frozen=True)
class _PosePoint:
    person_id: str
    device_id: str | None
    track_id: str | None
    observed_at: datetime
    timestamp_sec: float
    center: tuple[float, float]
    quality: float


def extract_cognitive_gait_features(
    records: Iterable[Mapping[str, Any]],
    *,
    config: CognitiveGaitConfig | None = None,
    gait_config: GaitAnalysisConfig | None = None,
    sit_stand_config: SitStandAnalysisConfig | None = None,
) -> list[dict[str, Any]]:
    """Extract daily gait clues for the cognitive-change module from pose records."""
    cognitive_config = config or CognitiveGaitConfig()
    normalized = [
        _normalize_pose_record(record, cognitive_config, record_number=index)
        for index, record in enumerate(records, start=1)
    ]
    if not normalized:
        raise ValueError("No pose records were provided")

    gait_windows = extract_gait_windows(
        normalized,
        config=gait_config or GaitAnalysisConfig(window_frames=cognitive_config.gait_window_frames),
    )
    sit_stand_events = extract_sit_stand_events(
        normalized,
        config=sit_stand_config or SitStandAnalysisConfig(),
    )
    gait_windows = _attach_absolute_event_times(gait_windows, normalized, cognitive_config)
    sit_stand_events = _attach_absolute_event_times(sit_stand_events, normalized, cognitive_config)
    turn_events = detect_turn_events(normalized, config=cognitive_config)
    return aggregate_daily_cognitive_gait(
        normalized,
        gait_windows=gait_windows,
        sit_stand_events=sit_stand_events,
        turn_events=turn_events,
        config=cognitive_config,
    )


def detect_turn_events(
    records: Iterable[Mapping[str, Any]],
    *,
    config: CognitiveGaitConfig | None = None,
) -> list[dict[str, Any]]:
    """Detect turning episodes from hip-center trajectory direction changes."""
    cognitive_config = config or CognitiveGaitConfig()
    normalized = [
        _normalize_pose_record(record, cognitive_config, record_number=index)
        for index, record in enumerate(records, start=1)
    ]
    grouped: dict[tuple[str, str], list[_PosePoint]] = defaultdict(list)
    for record in normalized:
        point = _pose_point(record, cognitive_config)
        if point is None:
            continue
        grouped[(point.person_id, point.track_id or "__single_track__")].append(point)

    events: list[dict[str, Any]] = []
    for points in grouped.values():
        ordered = sorted(points, key=lambda item: item.timestamp_sec)
        events.extend(_turn_events_for_track(ordered, cognitive_config))
    return sorted(
        events,
        key=lambda item: (
            str(item["person_id"]),
            str(item.get("track_id") or ""),
            str(item["start_time"]),
        ),
    )


def aggregate_daily_cognitive_gait(
    pose_records: Iterable[Mapping[str, Any]],
    *,
    gait_windows: Iterable[Mapping[str, Any]],
    sit_stand_events: Iterable[Mapping[str, Any]],
    turn_events: Iterable[Mapping[str, Any]],
    config: CognitiveGaitConfig | None = None,
) -> list[dict[str, Any]]:
    """Aggregate event/window outputs into person-day motor-cognitive features."""
    cognitive_config = config or CognitiveGaitConfig()
    buckets: dict[tuple[str, date], dict[str, Any]] = {}
    latest_time: dict[tuple[str, date], datetime] = {}
    scale_by_day: dict[tuple[str, date], list[float]] = defaultdict(list)

    normalized_records = [
        _normalize_pose_record(record, cognitive_config, record_number=index)
        for index, record in enumerate(pose_records, start=1)
    ]
    gait_windows = _attach_absolute_event_times(gait_windows, normalized_records, cognitive_config)
    sit_stand_events = _attach_absolute_event_times(sit_stand_events, normalized_records, cognitive_config)
    turn_events = _attach_absolute_event_times(turn_events, normalized_records, cognitive_config)
    for record in normalized_records:
        observed_at = _record_observed_at(record, cognitive_config)
        person_id = _required_string(record, "person_id", "pose record")
        day = observed_at.date()
        bucket = _bucket_for(buckets, person_id, day)
        bucket["pose_frame_count"] += 1
        quality = _optional_number(record.get("core_keypoint_quality", record.get("keypoint_quality")))
        if quality is not None and quality >= cognitive_config.min_keypoint_quality:
            bucket["valid_pose_frame_count"] += 1
        scale = _meters_per_norm_unit(record)
        if scale is not None:
            scale_by_day[(person_id, day)].append(scale)
        _set_latest(latest_time, person_id, day, observed_at)

    for window in gait_windows:
        if _is_insufficient(window, "quality_coverage", "insufficient_gait_quality"):
            _add_quality_flag(buckets, latest_time, window, cognitive_config, "insufficient_gait_quality")
            continue
        person_id, day = _event_person_day(window, cognitive_config)
        bucket = _bucket_for(buckets, person_id, day)
        features = window.get("gait_stability_features")
        if not isinstance(features, Mapping):
            continue
        speed = _optional_number(features.get("mean_center_speed_norm_per_sec"))
        if speed is not None:
            bucket["gait_speeds"].append(speed)
        stability = _gait_cycle_stability(features, cognitive_config)
        bucket["gait_cycle_stability_scores"].append(stability)
        bucket["gait_window_count"] += 1

    for event in sit_stand_events:
        if _is_insufficient(event, "quality_coverage", "insufficient_sit_stand_quality"):
            _add_quality_flag(buckets, latest_time, event, cognitive_config, "insufficient_sit_stand_quality")
            continue
        person_id, day = _event_person_day(event, cognitive_config)
        bucket = _bucket_for(buckets, person_id, day)
        duration = _optional_number(event.get("duration"))
        transition_type = str(event.get("transition_type") or "")
        if duration is not None:
            bucket["sit_stand_durations"].append(duration)
            if transition_type == "sit_to_stand":
                bucket["sit_to_stand_durations"].append(duration)
            elif transition_type == "stand_to_sit":
                bucket["stand_to_sit_durations"].append(duration)
        bucket["sit_stand_event_count"] += 1

    for event in turn_events:
        if event.get("data_quality") == "insufficient":
            _add_quality_flag(buckets, latest_time, event, cognitive_config, "insufficient_turn_quality")
            continue
        person_id, day = _event_person_day(event, cognitive_config)
        bucket = _bucket_for(buckets, person_id, day)
        duration = _optional_number(event.get("turn_duration_seconds"))
        stability = _optional_number(event.get("turn_stability_score"))
        if duration is not None:
            bucket["turn_durations"].append(duration)
        if stability is not None:
            bucket["turn_stability_scores"].append(stability)
        bucket["turn_event_count"] += 1

    outputs = []
    for key, bucket in sorted(buckets.items(), key=lambda item: (item[0][1], item[0][0])):
        person_id, day = key
        output = _finalize_bucket(
            person_id,
            day,
            bucket,
            latest_at=latest_time.get(key),
            meters_per_norm_unit=_mean(scale_by_day.get(key, [])),
            config=cognitive_config,
        )
        outputs.append(output)
    return outputs


def _turn_events_for_track(points: list[_PosePoint], config: CognitiveGaitConfig) -> list[dict[str, Any]]:
    if len(points) < config.min_turn_points:
        return []
    angles: list[tuple[int, float]] = []
    for index, (previous, current) in enumerate(zip(points, points[1:]), start=1):
        dt = current.timestamp_sec - previous.timestamp_sec
        if dt <= 0 or dt > config.max_point_gap_seconds:
            continue
        dx = current.center[0] - previous.center[0]
        dy = current.center[1] - previous.center[1]
        if math.hypot(dx, dy) < 1e-5:
            continue
        angles.append((index, math.atan2(dy, dx)))

    threshold = math.radians(config.turn_angle_degrees)
    candidates: list[tuple[int, int, float]] = []
    start_index: int | None = None
    accumulated = 0.0
    for (previous_index, previous_angle), (current_index, current_angle) in zip(angles, angles[1:]):
        delta = abs(_angle_delta(previous_angle, current_angle))
        if delta <= math.radians(5.0):
            if start_index is not None and points[current_index].timestamp_sec - points[previous_index].timestamp_sec > config.turn_merge_gap_seconds:
                start_index = None
                accumulated = 0.0
            continue
        if start_index is None:
            start_index = max(0, previous_index - 1)
            accumulated = 0.0
        accumulated += delta
        if accumulated >= threshold:
            candidates.append((start_index, current_index, accumulated))
            start_index = None
            accumulated = 0.0

    events = []
    for start, end, angle_change in candidates:
        segment = points[start : end + 1]
        if len(segment) < config.min_turn_points:
            continue
        events.append(_build_turn_event(segment, angle_change, config))
    return events


def _build_turn_event(
    points: list[_PosePoint],
    angle_change: float,
    config: CognitiveGaitConfig,
) -> dict[str, Any]:
    start = points[0]
    end = points[-1]
    duration = max(0.0, end.timestamp_sec - start.timestamp_sec)
    speeds = _speeds(points)
    mean_speed = _mean(speeds)
    speed_cv = _std(speeds) / mean_speed if mean_speed and mean_speed > 1e-6 else 0.0
    sway = _path_max_deviation(points)
    pause_ratio = _pause_ratio(speeds, config.pause_speed_threshold_norm_per_sec)
    speed_component = _ratio_score(speed_cv, config.speed_cv_high_risk)
    sway_component = _ratio_score(sway, config.turn_sway_high_risk)
    pause_component = pause_ratio
    instability = _clamp(0.45 * speed_component + 0.35 * sway_component + 0.20 * pause_component)
    stability = 1.0 - instability
    quality = _clamp(fmean(point.quality for point in points))
    if quality < config.min_keypoint_quality:
        data_quality = "insufficient"
        stability = 0.0
    elif quality < 0.75:
        data_quality = "medium"
    else:
        data_quality = "high"

    risk_factors: list[str] = []
    if speed_component >= 0.5:
        risk_factors.append("turn_speed_instability")
    if sway_component >= 0.5:
        risk_factors.append("turn_path_sway")
    if pause_component >= 0.5:
        risk_factors.append("turn_pause_or_hesitation")

    return {
        "event_type": "turning",
        "person_id": start.person_id,
        "device_id": start.device_id,
        "track_id": start.track_id,
        "start_time": start.observed_at.isoformat(),
        "end_time": end.observed_at.isoformat(),
        "turn_duration_seconds": _round(duration),
        "turn_angle_degrees": _round(math.degrees(angle_change)),
        "turn_stability_score": _round(stability),
        "turn_speed_cv": _round(speed_cv),
        "turn_path_sway": _round(sway),
        "turn_pause_ratio": _round(pause_ratio),
        "data_quality": data_quality,
        "quality_score": _round(quality),
        "risk_factors": risk_factors,
        "diagnosis": False,
        "model_version": MODEL_VERSION,
    }


def _normalize_pose_record(
    record: Mapping[str, Any],
    config: CognitiveGaitConfig,
    *,
    record_number: int,
) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise ValueError(f"pose record {record_number}: expected an object")
    normalized = dict(record)
    normalized["person_id"] = _required_string(record, "person_id", f"pose record {record_number}")
    _record_observed_at(record, config)
    if _optional_number(normalized.get("timestamp_sec")) is None:
        first_at = normalized.get("session_start_time")
        if first_at is not None:
            normalized["timestamp_sec"] = 0.0
        else:
            observed_at = _record_observed_at(record, config)
            midnight = datetime.combine(observed_at.date(), datetime.min.time(), tzinfo=config.timezone_info)
            normalized["timestamp_sec"] = observed_at.timestamp() - midnight.timestamp()
    if _optional_number(normalized.get("core_keypoint_quality")) is None:
        normalized["core_keypoint_quality"] = _optional_number(normalized.get("keypoint_quality")) or 0.0

    keypoints = normalized.get("keypoints")
    if not isinstance(keypoints, (list, tuple)):
        raise ValueError(f"pose record {record_number}: field 'keypoints' must be a list")
    normalized_points = []
    for point in keypoints:
        if not isinstance(point, Mapping):
            continue
        item = dict(point)
        score = _optional_number(item.get("score"))
        if item.get("valid") is None:
            item["valid"] = score is not None and score >= config.min_keypoint_quality
        for axis in ("x", "y"):
            smooth_key = f"{axis}_smooth"
            if _optional_number(item.get(smooth_key)) is None and _optional_number(item.get(axis)) is not None:
                item[smooth_key] = item.get(axis)
        item.setdefault("is_jump_outlier", False)
        normalized_points.append(item)
    normalized["keypoints"] = normalized_points
    return normalized


def _pose_point(record: Mapping[str, Any], config: CognitiveGaitConfig) -> _PosePoint | None:
    left_hip = _point(record, "left_hip")
    right_hip = _point(record, "right_hip")
    if not (_has_point(left_hip) and _has_point(right_hip)):
        return None
    observed_at = _record_observed_at(record, config)
    timestamp_sec = _optional_number(record.get("timestamp_sec"))
    if timestamp_sec is None:
        return None
    left = _point_xy(left_hip)
    right = _point_xy(right_hip)
    return _PosePoint(
        person_id=_required_string(record, "person_id", "pose record"),
        device_id=_optional_string(record.get("device_id")) or _optional_string(record.get("camera_id")),
        track_id=_optional_string(record.get("track_id")),
        observed_at=observed_at,
        timestamp_sec=timestamp_sec,
        center=((left[0] + right[0]) / 2.0, (left[1] + right[1]) / 2.0),
        quality=_optional_number(record.get("core_keypoint_quality", record.get("keypoint_quality"))) or 0.0,
    )


def _finalize_bucket(
    person_id: str,
    day: date,
    bucket: Mapping[str, Any],
    *,
    latest_at: datetime | None,
    meters_per_norm_unit: float | None,
    config: CognitiveGaitConfig,
) -> dict[str, Any]:
    gait_speed = _mean(bucket["gait_speeds"])
    gait_speed_mps = None
    if gait_speed is not None and meters_per_norm_unit is not None:
        gait_speed_mps = gait_speed * meters_per_norm_unit
    sit_stand_duration = _mean(bucket["sit_stand_durations"])
    turn_duration = _mean(bucket["turn_durations"])
    turn_stability = _mean(bucket["turn_stability_scores"])
    gait_cycle_stability = _mean(bucket["gait_cycle_stability_scores"])
    clue_score = _motor_cognitive_clue_score(
        gait_speed_norm_per_sec=gait_speed,
        sit_stand_duration=sit_stand_duration,
        turn_duration=turn_duration,
        turn_stability=turn_stability,
        gait_cycle_stability=gait_cycle_stability,
        config=config,
    )
    quality_flags = set(bucket["data_quality_flags"])
    if gait_speed is None:
        quality_flags.add("gait_speed_unavailable")
    if sit_stand_duration is None:
        quality_flags.add("sit_stand_duration_unavailable")
    if turn_duration is None:
        quality_flags.add("turn_duration_unavailable")
    if turn_stability is None:
        quality_flags.add("turn_stability_unavailable")
    if gait_cycle_stability is None:
        quality_flags.add("gait_cycle_stability_unavailable")

    pose_frame_count = int(bucket["pose_frame_count"])
    valid_pose_frame_count = int(bucket["valid_pose_frame_count"])
    pose_quality_coverage = (
        valid_pose_frame_count / pose_frame_count if pose_frame_count > 0 else None
    )
    return {
        "person_id": person_id,
        "date": day.isoformat(),
        "timestamp": latest_at.isoformat() if latest_at is not None else None,
        "gait_speed_norm_per_sec": _round_optional(gait_speed),
        "gait_speed_mps": _round_optional(gait_speed_mps),
        "sit_stand_duration_seconds": _round_optional(sit_stand_duration),
        "sit_to_stand_duration_seconds": _round_optional(_mean(bucket["sit_to_stand_durations"])),
        "stand_to_sit_duration_seconds": _round_optional(_mean(bucket["stand_to_sit_durations"])),
        "turn_duration_seconds": _round_optional(turn_duration),
        "turn_stability_score": _round_optional(turn_stability),
        "gait_cycle_stability_score": _round_optional(gait_cycle_stability),
        "motor_cognitive_clue_score": _round_optional(clue_score),
        "gait_window_count": int(bucket["gait_window_count"]),
        "sit_stand_event_count": int(bucket["sit_stand_event_count"]),
        "turn_event_count": int(bucket["turn_event_count"]),
        "pose_frame_count": pose_frame_count,
        "valid_pose_frame_count": valid_pose_frame_count,
        "pose_quality_coverage": _round_optional(pose_quality_coverage),
        "data_quality_flags": sorted(quality_flags),
        "diagnosis": False,
        "family_copy_hint": _family_copy_hint(clue_score),
        "model_version": MODEL_VERSION,
    }


def _bucket_for(
    buckets: dict[tuple[str, date], dict[str, Any]],
    person_id: str,
    day: date,
) -> dict[str, Any]:
    key = (person_id, day)
    if key not in buckets:
        buckets[key] = {
            "pose_frame_count": 0,
            "valid_pose_frame_count": 0,
            "gait_speeds": [],
            "gait_cycle_stability_scores": [],
            "sit_stand_durations": [],
            "sit_to_stand_durations": [],
            "stand_to_sit_durations": [],
            "turn_durations": [],
            "turn_stability_scores": [],
            "gait_window_count": 0,
            "sit_stand_event_count": 0,
            "turn_event_count": 0,
            "data_quality_flags": set(),
        }
    return buckets[key]


def _gait_cycle_stability(features: Mapping[str, Any], config: CognitiveGaitConfig) -> float:
    speed_cv = _optional_number(features.get("center_speed_cv")) or 0.0
    asymmetry = _optional_number(features.get("ankle_motion_asymmetry")) or 0.0
    pause_ratio = _optional_number(features.get("pause_frame_ratio")) or 0.0
    instability = _clamp(
        0.45 * _ratio_score(speed_cv, config.speed_cv_high_risk)
        + 0.35 * asymmetry
        + 0.20 * pause_ratio
    )
    return _round(1.0 - instability)


def _motor_cognitive_clue_score(
    *,
    gait_speed_norm_per_sec: float | None,
    sit_stand_duration: float | None,
    turn_duration: float | None,
    turn_stability: float | None,
    gait_cycle_stability: float | None,
    config: CognitiveGaitConfig,
) -> float | None:
    components: list[tuple[float, float]] = []
    if gait_speed_norm_per_sec is not None:
        low_speed = 1.0 - min(gait_speed_norm_per_sec / config.gait_speed_low_norm_per_sec, 1.0)
        components.append((low_speed, 0.25))
    if sit_stand_duration is not None:
        components.append((_ratio_score(sit_stand_duration, config.sit_stand_slow_seconds), 0.20))
    if turn_duration is not None:
        components.append((_ratio_score(turn_duration, config.turn_slow_seconds), 0.20))
    if turn_stability is not None:
        components.append((1.0 - turn_stability, 0.20))
    if gait_cycle_stability is not None:
        components.append((1.0 - gait_cycle_stability, 0.15))
    if not components:
        return None
    denominator = sum(weight for _, weight in components)
    return _clamp(sum(value * weight for value, weight in components) / denominator)


def _event_person_day(
    event: Mapping[str, Any],
    config: CognitiveGaitConfig,
) -> tuple[str, date]:
    person_id = _required_string(event, "person_id", "event")
    timestamp = event.get("end_time") or event.get("start_time")
    if isinstance(timestamp, str):
        observed_at = _aware_datetime(timestamp, "event time", config)
        return person_id, observed_at.date()
    if isinstance(timestamp, Real) and not isinstance(timestamp, bool):
        # Relative-time fall-risk outputs cannot assign a natural day on their own.
        # In this aggregation path they only appear alongside pose records, so use
        # the first existing bucket day if available via the caller-added flag.
        raise ValueError("event time must be timezone-aware for cognitive gait daily aggregation")
    raise ValueError("event time must be timezone-aware for cognitive gait daily aggregation")


def _add_quality_flag(
    buckets: dict[tuple[str, date], dict[str, Any]],
    latest_time: dict[tuple[str, date], datetime],
    event: Mapping[str, Any],
    config: CognitiveGaitConfig,
    flag: str,
) -> None:
    try:
        person_id, day = _event_person_day(event, config)
    except ValueError:
        return
    _bucket_for(buckets, person_id, day)["data_quality_flags"].add(flag)
    timestamp = event.get("end_time") or event.get("start_time")
    if isinstance(timestamp, str):
        _set_latest(latest_time, person_id, day, _aware_datetime(timestamp, "event time", config))


def _attach_absolute_event_times(
    events: Iterable[Mapping[str, Any]],
    pose_records: list[Mapping[str, Any]],
    config: CognitiveGaitConfig,
) -> list[dict[str, Any]]:
    records_by_key: dict[tuple[str, str], list[tuple[float, datetime]]] = defaultdict(list)
    for record in pose_records:
        person_id = _required_string(record, "person_id", "pose record")
        track_key = _optional_string(record.get("track_id")) or "__single_track__"
        timestamp_sec = _optional_number(record.get("timestamp_sec"))
        if timestamp_sec is None:
            continue
        records_by_key[(person_id, track_key)].append((timestamp_sec, _record_observed_at(record, config)))
    for values in records_by_key.values():
        values.sort(key=lambda item: item[0])

    outputs: list[dict[str, Any]] = []
    for event in events:
        item = dict(event)
        person_id = _optional_string(item.get("person_id"))
        if person_id is None:
            outputs.append(item)
            continue
        track_key = _optional_string(item.get("track_id")) or "__single_track__"
        timeline = records_by_key.get((person_id, track_key)) or records_by_key.get((person_id, "__single_track__"))
        if not timeline:
            outputs.append(item)
            continue
        start_relative = _optional_number(item.get("start_time"))
        end_relative = _optional_number(item.get("end_time"))
        if start_relative is not None:
            item["start_time"] = _nearest_time(timeline, start_relative).isoformat()
        if end_relative is not None:
            item["end_time"] = _nearest_time(timeline, end_relative).isoformat()
        outputs.append(item)
    return outputs


def _nearest_time(timeline: list[tuple[float, datetime]], timestamp_sec: float) -> datetime:
    best = min(timeline, key=lambda item: abs(item[0] - timestamp_sec))
    return best[1]


def _is_insufficient(event: Mapping[str, Any], quality_field: str, flag_field: str) -> bool:
    quality = event.get(quality_field)
    return isinstance(quality, Mapping) and quality.get(flag_field) is True


def _set_latest(
    latest_time: dict[tuple[str, date], datetime],
    person_id: str,
    day: date,
    value: datetime,
) -> None:
    key = (person_id, day)
    if key not in latest_time or value.timestamp() > latest_time[key].timestamp():
        latest_time[key] = value


def _record_observed_at(record: Mapping[str, Any], config: CognitiveGaitConfig) -> datetime:
    if record.get("observed_at") is not None:
        return _aware_datetime(record.get("observed_at"), "observed_at", config)
    if record.get("timestamp") is not None:
        return _aware_datetime(record.get("timestamp"), "timestamp", config)
    if record.get("session_start_time") is not None:
        start = _aware_datetime(record.get("session_start_time"), "session_start_time", config)
        timestamp_sec = _optional_number(record.get("timestamp_sec"))
        if timestamp_sec is None or timestamp_sec < 0:
            raise ValueError("pose record requires timestamp_sec with session_start_time")
        return start + timedelta(seconds=timestamp_sec)
    raise ValueError("pose record requires observed_at, timestamp, or session_start_time + timestamp_sec")


def _aware_datetime(value: Any, field: str, config: CognitiveGaitConfig) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a timezone-aware ISO-8601 timestamp")
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"{field} must be a timezone-aware ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must include a timezone")
    return parsed.astimezone(config.timezone_info)


def _point(record: Mapping[str, Any], name: str) -> Mapping[str, Any] | None:
    keypoints = record.get("keypoints")
    if not isinstance(keypoints, (list, tuple)):
        return None
    for point in keypoints:
        if isinstance(point, Mapping) and point.get("name") == name:
            return point
    return None


def _has_point(point: Mapping[str, Any] | None) -> bool:
    if point is None or point.get("valid") is not True or point.get("is_jump_outlier") is True:
        return False
    return _coordinate(point, "x") is not None and _coordinate(point, "y") is not None


def _point_xy(point: Mapping[str, Any] | None) -> tuple[float, float]:
    if point is None:
        raise ValueError("point is required")
    x = _coordinate(point, "x")
    y = _coordinate(point, "y")
    if x is None or y is None:
        raise ValueError("point coordinates are required")
    return x, y


def _coordinate(point: Mapping[str, Any], axis: str) -> float | None:
    smooth = _optional_number(point.get(f"{axis}_smooth"))
    if smooth is not None:
        return smooth
    return _optional_number(point.get(axis))


def _speeds(points: list[_PosePoint]) -> list[float]:
    speeds = []
    for previous, current in zip(points, points[1:]):
        dt = current.timestamp_sec - previous.timestamp_sec
        if dt <= 0:
            continue
        speeds.append(math.hypot(current.center[0] - previous.center[0], current.center[1] - previous.center[1]) / dt)
    return speeds


def _path_max_deviation(points: list[_PosePoint]) -> float:
    if len(points) < 3:
        return 0.0
    start = points[0].center
    end = points[-1].center
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy)
    if length <= 1e-6:
        xs = [point.center[0] for point in points]
        ys = [point.center[1] for point in points]
        return max(max(xs) - min(xs), max(ys) - min(ys))
    distances = []
    for point in points[1:-1]:
        x, y = point.center
        numerator = abs((dy * x) - (dx * y) + (end[0] * start[1]) - (end[1] * start[0]))
        distances.append(numerator / length)
    return max(distances) if distances else 0.0


def _pause_ratio(speeds: list[float], threshold: float) -> float:
    if not speeds:
        return 0.0
    return sum(1 for speed in speeds if speed <= threshold) / len(speeds)


def _meters_per_norm_unit(record: Mapping[str, Any]) -> float | None:
    for field in ("meters_per_norm_unit", "meter_scale", "scene_meter_scale"):
        value = _optional_number(record.get(field))
        if value is not None and value > 0:
            return value
    return None


def _family_copy_hint(score: float | None) -> str:
    if score is None:
        return "No usable gait clue was available for this day."
    if score >= 0.60:
        return "Recent walking, sit-stand, or turning patterns changed noticeably; family follow-up is recommended."
    if score >= 0.35:
        return "Some motor-cognitive behavior clues changed mildly; continue observing trends."
    return "No obvious motor-cognitive behavior clue was detected today."


def _angle_delta(left: float, right: float) -> float:
    delta = right - left
    while delta > math.pi:
        delta -= 2 * math.pi
    while delta < -math.pi:
        delta += 2 * math.pi
    return delta


def _ratio_score(value: float, threshold: float) -> float:
    if threshold <= 0:
        return 0.0
    return _clamp(value / threshold)


def _required_string(record: Mapping[str, Any], field: str, context: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}: field '{field}' must be a non-empty string")
    return value.strip()


def _optional_string(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _optional_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool) or not isinstance(value, Real):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _mean(values: Iterable[float] | None) -> float | None:
    value_list = [float(value) for value in (values or []) if math.isfinite(float(value))]
    return fmean(value_list) if value_list else None


def _std(values: Iterable[float]) -> float:
    value_list = list(values)
    if len(value_list) < 2:
        return 0.0
    mean = fmean(value_list)
    return math.sqrt(fmean((value - mean) ** 2 for value in value_list))


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _round(value: float) -> float:
    return round(float(value), 4)


def _round_optional(value: float | None) -> float | None:
    return None if value is None else _round(value)
