from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from numbers import Real
from statistics import fmean, median
from typing import Any, Iterable, Mapping
from zoneinfo import ZoneInfo


_UNKNOWN_VALUES = {"", "unknown", "none", "null", "nan"}
_REJECTED_QUALITY_STATES = {
    "offline",
    "occluded",
    "identity_uncertain",
    "low",
    "low_quality",
    "invalid",
    "rejected",
    "data_missing",
}
_SLEEP_END_FIELDS = (
    "sleep_end_or_out_of_bed_time",
    "main_sleep_end_time",
    "sleep_end_time",
    "out_of_bed_time",
    "wake_time",
)


@dataclass(frozen=True)
class DaytimeActivityConfig:
    """Thresholds for V1 daytime activity feature engineering."""

    timezone: str = "Asia/Shanghai"
    window_seconds: int = 10
    daytime_start: str = "06:00"
    daytime_end: str = "18:00"
    valid_detection_ratio_threshold: float = 0.60
    active_score_threshold: float = 0.40
    low_motion_score_threshold: float = 0.20
    effective_activity_minutes: float = 5.0
    effective_detection_coverage: float = 0.70
    effective_active_score: float = 0.35
    stable_zone_seconds: float = 3.0
    room_transition_stable_seconds: float = 10.0
    sedentary_min_minutes: float = 30.0
    bed_stay_min_minutes: float = 60.0
    outdoor_absence_min_minutes: float = 10.0
    meal_activity_min_minutes: float = 5.0
    min_valid_daytime_minutes: float = 30.0
    min_bbox_confidence: float = 0.30
    min_keypoint_confidence: float = 0.30
    min_tracking_confidence: float = 0.50
    center_motion_weight: float = 0.55
    pose_motion_weight: float = 0.30
    zone_transition_weight: float = 0.10
    posture_change_weight: float = 0.05
    core_keypoints: tuple[str, ...] = (
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    )
    sedentary_zones: tuple[str, ...] = (
        "sofa",
        "sofa_area",
        "chair",
        "chair_area",
        "living_room_fixed_area",
        "living_room",
    )
    bed_zones: tuple[str, ...] = (
        "bed",
        "bed_area",
        "bedroom_bed",
        "bedroom_bedside",
    )
    bedroom_rooms: tuple[str, ...] = ("bedroom",)
    door_zones: tuple[str, ...] = (
        "door",
        "door_area",
        "entry",
        "entrance",
        "hall_door",
    )
    meal_zones: tuple[str, ...] = (
        "kitchen",
        "kitchen_area",
        "dining",
        "dining_area",
        "dining_table",
        "meal_table",
    )
    sedentary_postures: tuple[str, ...] = ("sitting", "unknown_non_bed", "unknown")
    bed_postures: tuple[str, ...] = ("lying", "sleeping", "unknown_bed")
    meal_windows: tuple[tuple[str, str, str], ...] = (
        ("breakfast", "06:30", "09:00"),
        ("lunch", "11:00", "13:30"),
        ("dinner", "17:00", "19:30"),
    )

    @property
    def timezone_info(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)

    @property
    def daytime_start_time(self) -> time:
        return _parse_clock(self.daytime_start, "daytime_start")

    @property
    def daytime_end_time(self) -> time:
        return _parse_clock(self.daytime_end, "daytime_end")


@dataclass(frozen=True)
class ActivityFrame:
    person_id: str
    observed_at: datetime
    camera_id: str | None
    bbox_center: tuple[float, float] | None
    bbox_height: float | None
    keypoints: tuple[tuple[str, float, float], ...]
    zone: str
    room: str
    posture: str
    valid_detection: bool
    quality_state: str
    quality_flags: tuple[str, ...]

    @property
    def keypoint_map(self) -> dict[str, tuple[float, float]]:
        return {name: (x, y) for name, x, y in self.keypoints}


@dataclass(frozen=True)
class ActivityWindow:
    window_start: datetime
    window_end: datetime
    person_id: str
    room: str
    zone: str
    active_score: float | None
    motion_state: str
    posture: str
    valid_detection_ratio: float
    data_quality: str
    center_path_norm: float | None
    pose_motion_norm: float | None
    zone_transition_score: float
    posture_change_score: float
    quality_flags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_start": self.window_start.isoformat(),
            "window_end": self.window_end.isoformat(),
            "person_id": self.person_id,
            "room": self.room,
            "zone": self.zone,
            "active_score": self.active_score,
            "motion_state": self.motion_state,
            "posture": self.posture,
            "valid_detection_ratio": self.valid_detection_ratio,
            "data_quality": self.data_quality,
            "center_path_norm": self.center_path_norm,
            "pose_motion_norm": self.pose_motion_norm,
            "zone_transition_score": self.zone_transition_score,
            "posture_change_score": self.posture_change_score,
            "quality_flags": list(self.quality_flags),
        }


@dataclass
class _DayBucket:
    person_id: str
    day: date
    windows: list[ActivityWindow] = field(default_factory=list)


def aggregate_activity_windows(
    records: Iterable[Mapping[str, Any]],
    *,
    config: DaytimeActivityConfig | None = None,
) -> list[dict[str, Any]]:
    """Aggregate frame/second-level structure records into 10-second activity windows."""
    activity_config = config or DaytimeActivityConfig()
    frames = [
        _adapt_activity_frame(record, config=activity_config, record_number=index)
        for index, record in enumerate(records, start=1)
    ]
    if not frames:
        raise ValueError("No daytime activity records were provided")

    by_person_window: dict[tuple[str, float], list[ActivityFrame]] = defaultdict(list)
    for frame in frames:
        window_epoch = _window_epoch(frame.observed_at, activity_config)
        by_person_window[(frame.person_id, window_epoch)].append(frame)

    windows = [
        _build_window(person_id, window_epoch, window_frames, activity_config)
        for (person_id, window_epoch), window_frames in by_person_window.items()
    ]
    return [
        window.to_dict()
        for window in sorted(windows, key=lambda item: (item.window_start, item.person_id))
    ]


def aggregate_daytime_activity_from_windows(
    windows: Iterable[Mapping[str, Any]],
    *,
    sleep_records: Iterable[Mapping[str, Any]] | None = None,
    history_daily_features: Iterable[Mapping[str, Any]] | None = None,
    config: DaytimeActivityConfig | None = None,
) -> list[dict[str, Any]]:
    """Aggregate activity windows into V1 daytime activity features."""
    activity_config = config or DaytimeActivityConfig()
    window_objects = [_coerce_window(window, activity_config) for window in windows]
    if not window_objects:
        raise ValueError("No activity windows were provided")

    sleep_by_key = _sleep_records_by_person_day(sleep_records or (), activity_config)
    history_by_person = _history_by_person(history_daily_features or ())

    buckets: dict[tuple[str, date], _DayBucket] = {}
    for window in window_objects:
        key = (window.person_id, window.window_start.astimezone(activity_config.timezone_info).date())
        if key not in buckets:
            buckets[key] = _DayBucket(person_id=key[0], day=key[1])
        buckets[key].windows.append(window)

    outputs = [
        _finalize_day_bucket(
            bucket,
            sleep_record=sleep_by_key.get((bucket.person_id, bucket.day)),
            history=history_by_person.get(bucket.person_id, []),
            config=activity_config,
        )
        for bucket in buckets.values()
    ]
    return sorted(outputs, key=lambda item: (str(item["date"]), str(item["person_id"])))


def extract_daytime_activity_features(
    records: Iterable[Mapping[str, Any]],
    *,
    sleep_records: Iterable[Mapping[str, Any]] | None = None,
    history_daily_features: Iterable[Mapping[str, Any]] | None = None,
    config: DaytimeActivityConfig | None = None,
) -> list[dict[str, Any]]:
    """End-to-end V1 extraction from frame/second-level records to daily features."""
    activity_config = config or DaytimeActivityConfig()
    windows = aggregate_activity_windows(records, config=activity_config)
    return aggregate_daytime_activity_from_windows(
        windows,
        sleep_records=sleep_records,
        history_daily_features=history_daily_features,
        config=activity_config,
    )


def _adapt_activity_frame(
    record: Mapping[str, Any],
    *,
    config: DaytimeActivityConfig,
    record_number: int,
) -> ActivityFrame:
    if not isinstance(record, Mapping):
        raise ValueError(f"activity record {record_number}: expected an object")
    person_id = _required_string(record, "person_id", record_number)
    observed_at = _record_time(record, config, record_number)
    camera_id = _optional_string(record.get("camera_id")) or _optional_string(record.get("device_id"))
    bbox_center, bbox_height, bbox_flags = _bbox(record)
    keypoints, keypoint_flags = _keypoints(record.get("keypoints"), config)
    zone = _label(record.get("zone")) or _label(record.get("scene_region")) or "unknown"
    room = _label(record.get("room")) or _label(record.get("scene_region")) or zone
    posture = _label(record.get("posture")) or "unknown"

    flags = bbox_flags + keypoint_flags
    quality_state = _label(record.get("data_quality")) or _label(record.get("quality_state")) or "valid"
    bbox_confidence = _optional_unit(record.get("bbox_confidence", record.get("bbox_score")))
    keypoint_confidence = _optional_unit(
        record.get("keypoint_confidence", record.get("core_keypoint_quality", record.get("keypoint_quality")))
    )
    tracking_confidence = _optional_unit(record.get("tracking_confidence"))

    has_detection = bbox_center is not None or bool(keypoints)
    if not has_detection:
        flags.append("missing_detection")
    if bbox_confidence is not None and bbox_confidence < config.min_bbox_confidence:
        flags.append("low_bbox_confidence")
    if keypoint_confidence is not None and keypoint_confidence < config.min_keypoint_confidence:
        flags.append("low_keypoint_confidence")
    if tracking_confidence is not None and tracking_confidence < config.min_tracking_confidence:
        flags.append("low_tracking_confidence")
    if quality_state in _REJECTED_QUALITY_STATES:
        flags.append(quality_state)
    if record.get("quality_rejected") is True or record.get("rejected") is True:
        flags.append("quality_rejected")

    valid_detection = (
        has_detection
        and not any(flag in _rejecting_flags() for flag in flags)
        and quality_state not in _REJECTED_QUALITY_STATES
    )
    return ActivityFrame(
        person_id=person_id,
        observed_at=observed_at,
        camera_id=camera_id,
        bbox_center=bbox_center,
        bbox_height=bbox_height,
        keypoints=keypoints,
        zone=zone,
        room=room,
        posture=posture,
        valid_detection=valid_detection,
        quality_state=quality_state,
        quality_flags=tuple(_dedupe(flags)),
    )


def _build_window(
    person_id: str,
    window_epoch: float,
    frames: list[ActivityFrame],
    config: DaytimeActivityConfig,
) -> ActivityWindow:
    ordered = sorted(frames, key=lambda item: item.observed_at.timestamp())
    window_start = datetime.fromtimestamp(window_epoch, tz=config.timezone_info)
    window_end = window_start + timedelta(seconds=config.window_seconds)
    valid_frames = [frame for frame in ordered if frame.valid_detection]
    valid_ratio = _round(len(valid_frames) / len(ordered)) if ordered else 0.0
    flags = _dedupe(flag for frame in ordered for flag in frame.quality_flags)
    data_quality = _window_quality(ordered, valid_ratio, config)

    if data_quality != "valid" or not valid_frames:
        return ActivityWindow(
            window_start=window_start,
            window_end=window_end,
            person_id=person_id,
            room=_dominant_label(ordered, "room"),
            zone=_dominant_label(ordered, "zone"),
            active_score=None,
            motion_state="data_missing",
            posture=_dominant_label(ordered, "posture"),
            valid_detection_ratio=valid_ratio,
            data_quality=data_quality,
            center_path_norm=None,
            pose_motion_norm=None,
            zone_transition_score=0.0,
            posture_change_score=0.0,
            quality_flags=tuple(flags),
        )

    center_path_norm = _center_path_norm(valid_frames)
    pose_motion_norm = _pose_motion_norm(valid_frames, config)
    zone_changed = _stable_label_changed(
        valid_frames,
        "zone",
        window_start,
        window_end,
        config.stable_zone_seconds,
    )
    posture_changed = _label_changed(valid_frames, "posture")
    center_score = _piecewise_score(center_path_norm, ((0.05, 0.0), (0.30, 0.3), (0.80, 0.7)))
    pose_score = _piecewise_score(pose_motion_norm, ((0.03, 0.0), (0.15, 0.3), (0.50, 0.7)))
    zone_score = 1.0 if zone_changed else 0.0
    posture_score = 1.0 if posture_changed else 0.0
    active_score = _round(
        config.center_motion_weight * center_score
        + config.pose_motion_weight * pose_score
        + config.zone_transition_weight * zone_score
        + config.posture_change_weight * posture_score
    )
    low_motion = (
        active_score <= config.low_motion_score_threshold
        and center_path_norm is not None
        and pose_motion_norm is not None
        and center_path_norm < 0.05
        and pose_motion_norm < 0.03
        and not zone_changed
    )
    if low_motion:
        motion_state = "low_motion"
    elif active_score >= config.active_score_threshold:
        motion_state = "active"
    else:
        motion_state = "moderate_motion"

    return ActivityWindow(
        window_start=window_start,
        window_end=window_end,
        person_id=person_id,
        room=_dominant_label(valid_frames, "room"),
        zone=_dominant_label(valid_frames, "zone"),
        active_score=active_score,
        motion_state=motion_state,
        posture=_dominant_label(valid_frames, "posture"),
        valid_detection_ratio=valid_ratio,
        data_quality="valid",
        center_path_norm=_round_optional(center_path_norm),
        pose_motion_norm=_round_optional(pose_motion_norm),
        zone_transition_score=zone_score,
        posture_change_score=posture_score,
        quality_flags=tuple(flags),
    )


def _finalize_day_bucket(
    bucket: _DayBucket,
    *,
    sleep_record: Mapping[str, Any] | None,
    history: list[Mapping[str, Any]],
    config: DaytimeActivityConfig,
) -> dict[str, Any]:
    windows = sorted(bucket.windows, key=lambda item: item.window_start)
    daytime_windows = [window for window in windows if _is_daytime(window.window_start, config)]
    valid_daytime = [window for window in daytime_windows if window.data_quality == "valid"]
    duration_minutes = config.window_seconds / 60.0
    valid_daytime_minutes = len(valid_daytime) * duration_minutes
    active_windows = [
        window
        for window in valid_daytime
        if window.active_score is not None and window.active_score >= config.active_score_threshold
    ]
    active_minutes = len(active_windows) * duration_minutes
    weighted_activity = sum((window.active_score or 0.0) * duration_minutes for window in valid_daytime)
    sedentary_bouts = _low_motion_bouts(
        valid_daytime,
        duration_minutes=duration_minutes,
        min_minutes=config.sedentary_min_minutes,
        zone_predicate=lambda window: _label_in(window.zone, config.sedentary_zones)
        or _label_in(window.room, config.sedentary_zones),
        posture_predicate=lambda window: _label_in(window.posture, config.sedentary_postures),
    )
    bed_bouts = _low_motion_bouts(
        valid_daytime,
        duration_minutes=duration_minutes,
        min_minutes=config.bed_stay_min_minutes,
        zone_predicate=lambda window: _is_bed_zone(window, config),
        posture_predicate=lambda window: True,
    )
    room_transition_count = _room_transition_count(valid_daytime, config)
    bedroom_minutes = sum(
        duration_minutes
        for window in valid_daytime
        if _is_bedroom_window(window, config)
    )
    bedroom_ratio = _round(bedroom_minutes / valid_daytime_minutes) if valid_daytime_minutes > 0 else None
    outdoor_count, outdoor_minutes, outdoor_flags = _outdoor_events(valid_daytime, bucket.day, config)
    sleep_end = _sleep_end_time(sleep_record, config) if sleep_record is not None else None
    first_effective = _first_effective_activity(
        windows,
        start_at=_day_start(bucket.day, config) if sleep_end is None else sleep_end,
        config=config,
    )
    wake_activation_delay = (
        _round((first_effective - sleep_end).total_seconds() / 60.0)
        if first_effective is not None and sleep_end is not None
        else None
    )
    hourly_vector = _hourly_activity_vector(windows, config)
    peak_minute = _activity_peak_minute(hourly_vector)
    meal_flags = _meal_window_activity(valid_daytime, config)
    meal_count = sum(1 for value in meal_flags.values() if value)
    routine_score, peak_shift, first_shift, meal_consistency = _routine_metrics(
        bucket.person_id,
        bucket.day,
        hourly_vector,
        peak_minute,
        _minute_of_day(first_effective) if first_effective is not None else None,
        meal_count,
        history,
    )

    quality_flags = set(outdoor_flags)
    for window in windows:
        quality_flags.update(window.quality_flags)
        if window.data_quality != "valid":
            quality_flags.add(f"{window.data_quality}_window")
    if sleep_record is None:
        quality_flags.add("wake_activation_sleep_time_unavailable")
    if routine_score is None:
        quality_flags.add("routine_baseline_unavailable")
    if valid_daytime_minutes < config.min_valid_daytime_minutes:
        quality_flags.add("insufficient_daytime_valid_detection")
        data_quality = "data_insufficient"
    elif any(window.data_quality != "valid" for window in daytime_windows):
        data_quality = "partial"
    else:
        data_quality = "valid"

    return {
        "person_id": bucket.person_id,
        "date": bucket.day.isoformat(),
        "daytime_active_minutes": _round(active_minutes),
        "weighted_daytime_activity": _round(weighted_activity),
        "valid_daytime_detection_minutes": _round(valid_daytime_minutes),
        "sedentary_bouts_count": len(sedentary_bouts),
        "sedentary_total_minutes": _round(sum(sedentary_bouts)),
        "longest_sedentary_bout_minutes": _round(max(sedentary_bouts, default=0.0)),
        "daytime_bed_stay_minutes": _round(sum(bed_bouts)),
        "daytime_bed_bouts_count": len(bed_bouts),
        "room_transition_count": room_transition_count,
        "bedroom_stay_ratio": bedroom_ratio,
        "outdoor_event_count": outdoor_count,
        "outdoor_total_duration_minutes": _round(outdoor_minutes),
        "wake_activation_delay_minutes": wake_activation_delay,
        "first_effective_activity_time": first_effective.isoformat() if first_effective is not None else None,
        "first_effective_activity_minute_of_day": _minute_of_day(first_effective),
        "routine_stability_score": routine_score,
        "activity_peak_shift_minutes": peak_shift,
        "first_activity_shift_minutes": first_shift,
        "sleep_midpoint_shift_minutes": None,
        "activity_peak_minute_of_day": peak_minute,
        "hourly_activity_vector": [_round(value) for value in hourly_vector],
        "meal_window_activity_count": meal_count,
        "breakfast_related_activity": meal_flags["breakfast"],
        "lunch_related_activity": meal_flags["lunch"],
        "dinner_related_activity": meal_flags["dinner"],
        "meal_routine_consistency": meal_consistency,
        "data_quality": data_quality,
        "data_quality_flags": sorted(quality_flags),
    }


def _coerce_window(record: Mapping[str, Any], config: DaytimeActivityConfig) -> ActivityWindow:
    if isinstance(record, ActivityWindow):
        return record
    if not isinstance(record, Mapping):
        raise ValueError("activity window must be an object")
    start = _aware_datetime(record.get("window_start"), "window_start", config)
    end = _aware_datetime(record.get("window_end"), "window_end", config)
    if end <= start:
        raise ValueError("activity window field 'window_end' must be after 'window_start'")
    person_id = record.get("person_id")
    if not isinstance(person_id, str) or not person_id.strip():
        raise ValueError("activity window field 'person_id' must be a non-empty string")
    active_score = _optional_unit(record.get("active_score"))
    ratio = _optional_unit(record.get("valid_detection_ratio"))
    return ActivityWindow(
        window_start=start,
        window_end=end,
        person_id=person_id.strip(),
        room=_label(record.get("room")) or "unknown",
        zone=_label(record.get("zone")) or "unknown",
        active_score=active_score,
        motion_state=_label(record.get("motion_state")) or "data_missing",
        posture=_label(record.get("posture")) or "unknown",
        valid_detection_ratio=0.0 if ratio is None else ratio,
        data_quality=_label(record.get("data_quality")) or "valid",
        center_path_norm=_optional_nonnegative(record.get("center_path_norm")),
        pose_motion_norm=_optional_nonnegative(record.get("pose_motion_norm")),
        zone_transition_score=float(_optional_unit(record.get("zone_transition_score")) or 0.0),
        posture_change_score=float(_optional_unit(record.get("posture_change_score")) or 0.0),
        quality_flags=tuple(_string_list(record.get("quality_flags"))),
    )


def _center_path_norm(frames: list[ActivityFrame]) -> float | None:
    heights = [frame.bbox_height for frame in frames if frame.bbox_height is not None and frame.bbox_height > 0]
    pairs = _same_camera_pairs(frames)
    distance = 0.0
    for left, right in pairs:
        if left.bbox_center is None or right.bbox_center is None:
            continue
        distance += math.hypot(
            right.bbox_center[0] - left.bbox_center[0],
            right.bbox_center[1] - left.bbox_center[1],
        )
    if distance <= 0 and heights:
        return 0.0
    if not heights:
        return None
    return distance / fmean(heights)


def _pose_motion_norm(frames: list[ActivityFrame], config: DaytimeActivityConfig) -> float | None:
    heights = [frame.bbox_height for frame in frames if frame.bbox_height is not None and frame.bbox_height > 0]
    if not heights:
        return None
    total = 0.0
    usable_pairs = 0
    for left, right in _same_camera_pairs(frames):
        left_points = left.keypoint_map
        right_points = right.keypoint_map
        names = [name for name in config.core_keypoints if name in left_points and name in right_points]
        if not names:
            names = sorted(set(left_points) & set(right_points))
        if not names:
            continue
        total += fmean(
            math.hypot(
                right_points[name][0] - left_points[name][0],
                right_points[name][1] - left_points[name][1],
            )
            for name in names
        )
        usable_pairs += 1
    if usable_pairs == 0:
        return None
    return total / fmean(heights)


def _same_camera_pairs(frames: list[ActivityFrame]) -> list[tuple[ActivityFrame, ActivityFrame]]:
    pairs = []
    ordered = sorted(frames, key=lambda item: item.observed_at.timestamp())
    for left, right in zip(ordered, ordered[1:]):
        if (left.camera_id or "__single_camera__") == (right.camera_id or "__single_camera__"):
            pairs.append((left, right))
    return pairs


def _stable_label_changed(
    frames: list[ActivityFrame],
    attr: str,
    window_start: datetime,
    window_end: datetime,
    stable_seconds: float,
) -> bool:
    segments = _label_segments(frames, attr, window_start, window_end)
    stable_labels = [
        label
        for label, start, end in segments
        if _known(label) and (end - start).total_seconds() >= stable_seconds
    ]
    return len(dict.fromkeys(stable_labels)) > 1


def _label_segments(
    frames: list[ActivityFrame],
    attr: str,
    window_start: datetime,
    window_end: datetime,
) -> list[tuple[str, datetime, datetime]]:
    ordered = sorted(frames, key=lambda item: item.observed_at.timestamp())
    raw = []
    for index, frame in enumerate(ordered):
        segment_start = max(frame.observed_at, window_start)
        segment_end = ordered[index + 1].observed_at if index + 1 < len(ordered) else window_end
        segment_end = min(segment_end, window_end)
        label = getattr(frame, attr)
        if segment_end > segment_start:
            raw.append((label, segment_start, segment_end))
    merged: list[tuple[str, datetime, datetime]] = []
    for label, start, end in raw:
        if merged and merged[-1][0] == label:
            merged[-1] = (label, merged[-1][1], end)
        else:
            merged.append((label, start, end))
    return merged


def _label_changed(frames: list[ActivityFrame], attr: str) -> bool:
    labels = [getattr(frame, attr) for frame in frames if _known(getattr(frame, attr))]
    return len(set(labels)) > 1


def _piecewise_score(value: float | None, thresholds: tuple[tuple[float, float], ...]) -> float:
    if value is None or not math.isfinite(value):
        return 0.0
    for upper, score in thresholds:
        if value < upper:
            return score
    return 1.0


def _window_quality(
    frames: list[ActivityFrame],
    valid_ratio: float,
    config: DaytimeActivityConfig,
) -> str:
    states = [frame.quality_state for frame in frames]
    if states and all(state == "offline" for state in states):
        return "offline"
    if states and all(state == "occluded" for state in states):
        return "occluded"
    if states and all(state == "identity_uncertain" for state in states):
        return "identity_uncertain"
    if valid_ratio < config.valid_detection_ratio_threshold:
        return "low"
    return "valid"


def _low_motion_bouts(
    windows: list[ActivityWindow],
    *,
    duration_minutes: float,
    min_minutes: float,
    zone_predicate: Any,
    posture_predicate: Any,
) -> list[float]:
    bouts = []
    current = 0.0
    previous_end: datetime | None = None
    for window in windows:
        contiguous = previous_end is None or window.window_start <= previous_end + timedelta(seconds=1)
        matches = (
            window.motion_state == "low_motion"
            and zone_predicate(window)
            and posture_predicate(window)
        )
        if matches and contiguous:
            current += duration_minutes
        else:
            if current >= min_minutes:
                bouts.append(current)
            current = duration_minutes if matches else 0.0
        previous_end = window.window_end
    if current >= min_minutes:
        bouts.append(current)
    return bouts


def _room_transition_count(windows: list[ActivityWindow], config: DaytimeActivityConfig) -> int:
    runs: list[tuple[str, float]] = []
    duration_seconds = float(config.window_seconds)
    for window in windows:
        room = window.room
        if not _known(room):
            continue
        if runs and runs[-1][0] == room:
            runs[-1] = (room, runs[-1][1] + duration_seconds)
        else:
            runs.append((room, duration_seconds))
    stable_rooms = [room for room, seconds in runs if seconds >= config.room_transition_stable_seconds]
    count = 0
    previous: str | None = None
    for room in stable_rooms:
        if previous is not None and room != previous:
            count += 1
        previous = room
    return count


def _outdoor_events(
    windows: list[ActivityWindow],
    day: date,
    config: DaytimeActivityConfig,
) -> tuple[int, float, list[str]]:
    count = 0
    total_minutes = 0.0
    flags: list[str] = []
    index = 0
    while index < len(windows):
        window = windows[index]
        if not (_label_in(window.zone, config.door_zones) or _label_in(window.room, config.door_zones)):
            index += 1
            continue
        next_window = windows[index + 1] if index + 1 < len(windows) else None
        if next_window is None:
            day_end = datetime.combine(day, config.daytime_end_time, tzinfo=config.timezone_info)
            gap_seconds = max(0.0, (day_end - window.window_end).total_seconds())
            flags.append("outdoor_event_open_ended")
        else:
            gap_seconds = (next_window.window_start - window.window_end).total_seconds()
        gap_minutes = gap_seconds / 60.0
        if gap_minutes >= config.outdoor_absence_min_minutes:
            count += 1
            total_minutes += gap_minutes
            if next_window is None:
                break
            while index + 1 < len(windows) and windows[index + 1].window_start < next_window.window_start:
                index += 1
        index += 1
    return count, total_minutes, flags


def _first_effective_activity(
    windows: list[ActivityWindow],
    *,
    start_at: datetime,
    config: DaytimeActivityConfig,
) -> datetime | None:
    ordered = [window for window in sorted(windows, key=lambda item: item.window_start) if window.window_start >= start_at]
    if not ordered:
        return None
    required_seconds = config.effective_activity_minutes * 60.0
    for candidate in ordered:
        search_end = candidate.window_start + timedelta(seconds=required_seconds)
        group = [window for window in ordered if candidate.window_start <= window.window_start < search_end]
        if not group:
            continue
        valid = [window for window in group if window.data_quality == "valid"]
        valid_seconds = len(valid) * config.window_seconds
        coverage = valid_seconds / required_seconds
        if coverage < config.effective_detection_coverage:
            continue
        average_score = (
            fmean(window.active_score or 0.0 for window in valid)
            if valid
            else 0.0
        )
        has_transition = any(window.zone_transition_score > 0 for window in valid) or _room_transition_count(valid, config) > 0
        bed_only_micro_motion = valid and all(_is_bed_zone(window, config) and window.motion_state == "low_motion" for window in valid)
        if not bed_only_micro_motion and (
            average_score >= config.effective_active_score or has_transition
        ):
            return candidate.window_start
    return None


def _hourly_activity_vector(windows: list[ActivityWindow], config: DaytimeActivityConfig) -> list[float]:
    vector = [0.0 for _ in range(24)]
    for window in windows:
        if window.data_quality != "valid" or window.active_score is None:
            continue
        hour = window.window_start.astimezone(config.timezone_info).hour
        vector[hour] += window.active_score * config.window_seconds / 60.0
    return vector


def _activity_peak_minute(vector: list[float]) -> int | None:
    if not vector or max(vector) <= 0:
        return None
    peak_hour = max(range(len(vector)), key=lambda index: vector[index])
    return peak_hour * 60 + 30


def _routine_metrics(
    person_id: str,
    day: date,
    vector: list[float],
    peak_minute: int | None,
    first_activity_minute: int | None,
    meal_count: int,
    history: list[Mapping[str, Any]],
) -> tuple[float | None, float | None, float | None, float | None]:
    eligible = [
        item
        for item in history
        if str(item.get("person_id", person_id)) == person_id
        and str(item.get("date", "")) < day.isoformat()
    ]
    historical_vectors = [
        [float(value) for value in item.get("hourly_activity_vector", [])]
        for item in eligible
        if isinstance(item.get("hourly_activity_vector"), list)
        and len(item.get("hourly_activity_vector", [])) == 24
        and all(isinstance(value, Real) and not isinstance(value, bool) for value in item.get("hourly_activity_vector", []))
    ]
    routine_score = None
    if historical_vectors:
        baseline = [fmean(values) for values in zip(*historical_vectors)]
        routine_score = _round(_cosine_similarity(vector, baseline))

    historical_peaks = _number_values(item.get("activity_peak_minute_of_day") for item in eligible)
    peak_shift = (
        _round(abs(float(peak_minute) - median(historical_peaks)))
        if peak_minute is not None and historical_peaks
        else None
    )
    historical_first = _number_values(item.get("first_effective_activity_minute_of_day") for item in eligible)
    first_shift = (
        _round(abs(float(first_activity_minute) - median(historical_first)))
        if first_activity_minute is not None and historical_first
        else None
    )
    historical_meals = _number_values(item.get("meal_window_activity_count") for item in eligible)
    meal_consistency = None
    if historical_meals:
        baseline_meals = median(historical_meals)
        if baseline_meals > 0:
            meal_consistency = _round(min(1.0, max(0.0, meal_count / baseline_meals)))
    return routine_score, peak_shift, first_shift, meal_consistency


def _meal_window_activity(windows: list[ActivityWindow], config: DaytimeActivityConfig) -> dict[str, bool]:
    result = {name: False for name, _, _ in config.meal_windows}
    for name, start_text, end_text in config.meal_windows:
        start_time = _parse_clock(start_text, f"meal_windows.{name}.start")
        end_time = _parse_clock(end_text, f"meal_windows.{name}.end")
        meal_windows = [
            window
            for window in windows
            if _clock_in_range(window.window_start.timetz().replace(tzinfo=None), start_time, end_time)
            and (_label_in(window.zone, config.meal_zones) or _label_in(window.room, config.meal_zones))
            and window.active_score is not None
            and window.active_score >= config.effective_active_score
        ]
        result[name] = _continuous_minutes(meal_windows, config) >= config.meal_activity_min_minutes
    return result


def _continuous_minutes(windows: list[ActivityWindow], config: DaytimeActivityConfig) -> float:
    if not windows:
        return 0.0
    ordered = sorted(windows, key=lambda item: item.window_start)
    best = 0.0
    current = config.window_seconds / 60.0
    previous = ordered[0]
    for window in ordered[1:]:
        if window.window_start <= previous.window_end + timedelta(seconds=1):
            current += config.window_seconds / 60.0
        else:
            best = max(best, current)
            current = config.window_seconds / 60.0
        previous = window
    return max(best, current)


def _sleep_records_by_person_day(
    records: Iterable[Mapping[str, Any]],
    config: DaytimeActivityConfig,
) -> dict[tuple[str, date], Mapping[str, Any]]:
    outputs: dict[tuple[str, date], Mapping[str, Any]] = {}
    for record in records:
        person_id = record.get("person_id")
        if not isinstance(person_id, str) or not person_id.strip():
            continue
        end_time = _sleep_end_time(record, config)
        if end_time is None:
            day_text = record.get("date")
            if not isinstance(day_text, str):
                continue
            try:
                day = date.fromisoformat(day_text)
            except ValueError:
                continue
        else:
            day = end_time.astimezone(config.timezone_info).date()
        outputs[(person_id.strip(), day)] = record
    return outputs


def _sleep_end_time(record: Mapping[str, Any] | None, config: DaytimeActivityConfig) -> datetime | None:
    if record is None:
        return None
    for field in _SLEEP_END_FIELDS:
        value = record.get(field)
        if value is None:
            continue
        return _aware_datetime(value, field, config)
    return None


def _history_by_person(records: Iterable[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        person_id = record.get("person_id")
        if isinstance(person_id, str) and person_id.strip():
            grouped[person_id.strip()].append(record)
    return grouped


def _bbox(record: Mapping[str, Any]) -> tuple[tuple[float, float] | None, float | None, list[str]]:
    raw = record.get("bbox")
    if raw is None:
        return None, None, ["missing_bbox"]
    if not isinstance(raw, (list, tuple)) or len(raw) < 4:
        return None, None, ["invalid_bbox"]
    try:
        x, y, third, fourth = [float(value) for value in raw[:4]]
    except (TypeError, ValueError):
        return None, None, ["invalid_bbox"]
    fmt = str(record.get("bbox_format", "xywh")).strip().lower()
    if fmt in {"xyxy", "x1y1x2y2"}:
        height = abs(fourth - y)
        center = ((x + third) / 2.0, (y + fourth) / 2.0)
    else:
        height = fourth
        center = (x + third / 2.0, y + fourth / 2.0)
    if not math.isfinite(height) or height <= 0:
        return None, None, ["invalid_bbox"]
    if any(not math.isfinite(value) for value in (*center,)):
        return None, None, ["invalid_bbox"]
    return center, height, []


def _keypoints(raw: Any, config: DaytimeActivityConfig) -> tuple[tuple[tuple[str, float, float], ...], list[str]]:
    if raw is None:
        return (), ["missing_keypoints"]
    if not isinstance(raw, (list, tuple)):
        return (), ["invalid_keypoints"]
    points: list[tuple[str, float, float]] = []
    invalid = False
    for index, item in enumerate(raw):
        if isinstance(item, Mapping):
            name = _label(item.get("name")) or f"kp_{index}"
            x_value = item.get("x")
            y_value = item.get("y")
            score = _optional_unit(item.get("score", item.get("confidence")))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            name = f"kp_{index}"
            x_value = item[0]
            y_value = item[1]
            score = _optional_unit(item[2]) if len(item) > 2 else 1.0
        else:
            invalid = True
            continue
        try:
            x = float(x_value)
            y = float(y_value)
        except (TypeError, ValueError):
            invalid = True
            continue
        if not math.isfinite(x) or not math.isfinite(y):
            invalid = True
            continue
        if score is not None and score < config.min_keypoint_confidence:
            continue
        points.append((name, x, y))
    flags = []
    if invalid:
        flags.append("invalid_keypoints")
    if not points:
        flags.append("missing_keypoints")
    return tuple(points), flags


def _required_string(record: Mapping[str, Any], field: str, record_number: int) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"activity record {record_number}: field '{field}' must be a non-empty string")
    return value.strip()


def _record_time(record: Mapping[str, Any], config: DaytimeActivityConfig, record_number: int) -> datetime:
    value = record.get("observed_at", record.get("timestamp"))
    return _aware_datetime(value, f"activity record {record_number} timestamp", config)


def _aware_datetime(value: Any, field: str, config: DaytimeActivityConfig) -> datetime:
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
    return parsed.astimezone(config.timezone_info)


def _window_epoch(value: datetime, config: DaytimeActivityConfig) -> float:
    timestamp = value.timestamp()
    return timestamp - (timestamp % config.window_seconds)


def _dominant_label(frames: Iterable[Any], attr: str) -> str:
    labels = [getattr(frame, attr, "unknown") for frame in frames]
    known = [label for label in labels if _known(label)]
    if not known:
        return "unknown"
    return Counter(known).most_common(1)[0][0]


def _label(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    return "" if text in _UNKNOWN_VALUES else text


def _known(value: str) -> bool:
    return _label(value) != ""


def _label_in(value: str, candidates: tuple[str, ...]) -> bool:
    normalized = _label(value)
    return any(normalized == item or item in normalized for item in candidates)


def _is_daytime(value: datetime, config: DaytimeActivityConfig) -> bool:
    current = value.astimezone(config.timezone_info).time()
    return _clock_in_range(current, config.daytime_start_time, config.daytime_end_time)


def _clock_in_range(current: time, start: time, end: time) -> bool:
    if start <= end:
        return start <= current < end
    return current >= start or current < end


def _day_start(day: date, config: DaytimeActivityConfig) -> datetime:
    return datetime.combine(day, config.daytime_start_time, tzinfo=config.timezone_info)


def _is_bed_zone(window: ActivityWindow, config: DaytimeActivityConfig) -> bool:
    return _label_in(window.zone, config.bed_zones) or _label_in(window.room, config.bed_zones)


def _is_bedroom_window(window: ActivityWindow, config: DaytimeActivityConfig) -> bool:
    return _label_in(window.room, config.bedroom_rooms) or _is_bed_zone(window, config)


def _minute_of_day(value: datetime | None) -> int | None:
    if value is None:
        return None
    return value.hour * 60 + value.minute


def _number_values(values: Iterable[Any]) -> list[float]:
    return [
        float(value)
        for value in values
        if isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value))
    ]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return max(0.0, min(1.0, numerator / (left_norm * right_norm)))


def _parse_clock(value: str, field: str) -> time:
    try:
        hour_text, minute_text = value.split(":", 1)
        return time(hour=int(hour_text), minute=int(minute_text))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"{field} must use HH:MM format") from exc


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _optional_unit(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        return None
    return number


def _optional_nonnegative(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("activity window numeric fields must be finite non-negative numbers")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError("activity window numeric fields must be finite non-negative numbers")
    return number


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _rejecting_flags() -> set[str]:
    return {
        "invalid_bbox",
        "missing_detection",
        "low_bbox_confidence",
        "low_keypoint_confidence",
        "low_tracking_confidence",
        "offline",
        "occluded",
        "identity_uncertain",
        "quality_rejected",
    }


def _dedupe(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _round(value: float) -> float:
    return round(float(value), 4)


def _round_optional(value: float | None) -> float | None:
    return None if value is None else _round(value)
