from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from statistics import median
from typing import Any, Iterable, Mapping

from elderly_monitoring.modules.mental_health.adapters import (
    BehaviorObservation,
    MentalHealthDataError,
    adapt_behavior_record,
)
from elderly_monitoring.modules.mental_health.config import AggregationConfig, load_aggregation_config


@dataclass
class _DayAccumulator:
    person_id: str
    date: date
    start_time: datetime | None = None
    end_time: datetime | None = None
    observation_seconds: float = 0.0
    valid_observation_seconds: float = 0.0
    activity_volume: float = 0.0
    active_valid_seconds: float = 0.0
    nighttime_valid_seconds: float = 0.0
    nighttime_active_valid_seconds: float = 0.0
    scene_seconds: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    scene_transition_count: int = 0
    flags: set[str] = field(default_factory=set)

    def include_time(self, value: datetime) -> None:
        if self.start_time is None or _epoch_seconds(value) < _epoch_seconds(self.start_time):
            self.start_time = value
        if self.end_time is None or _epoch_seconds(value) > _epoch_seconds(self.end_time):
            self.end_time = value


@dataclass(frozen=True)
class _Interval:
    person_id: str
    device_id: str
    start: BehaviorObservation
    end: BehaviorObservation
    start_at: datetime
    end_at: datetime
    duration_seconds: float
    motion_proxy: float | None
    valid: bool
    quality_score: float
    common_keypoint_count: int
    transition: bool
    flags: tuple[str, ...]

def aggregate_daily_behavior(
    records: Iterable[Mapping[str, Any]],
    *,
    config: AggregationConfig | None = None,
) -> list[dict[str, Any]]:
    """Aggregate absolute-time pose observations into deterministic person-day features."""
    aggregation_config = config or load_aggregation_config()
    adapted = [
        adapt_behavior_record(record, config=aggregation_config, record_number=index)
        for index, record in enumerate(records, start=1)
    ]
    if not adapted:
        raise ValueError("No behavior records were provided; absolute event time is required")

    by_person: dict[str, list[BehaviorObservation]] = defaultdict(list)
    for observation in adapted:
        by_person[observation.person_id].append(observation)

    for person_id, observations in by_person.items():
        timeline_observations = [
            item
            for item in observations
            if item.usable_for_daily_aggregation and item.observed_at is not None
        ]
        _validate_device_contract(person_id, timeline_observations)
        if not timeline_observations:
            raise ValueError(
                f"person {person_id!r} has no usable absolute event time; "
                "observed_at or session_start_time with timestamp_sec is required"
            )

    outputs: list[dict[str, Any]] = []
    for person_id in sorted(by_person):
        outputs.extend(_aggregate_person(person_id, by_person[person_id], aggregation_config))
    return sorted(outputs, key=lambda item: (str(item["date"]), str(item["person_id"])))


def _aggregate_person(
    person_id: str,
    observations: list[BehaviorObservation],
    config: AggregationConfig,
) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, date], _DayAccumulator] = {}
    duplicate_dates: set[date] = set()
    usable = [item for item in observations if item.usable_for_daily_aggregation and item.observed_at is not None]

    for item in observations:
        if item.observed_at is None:
            continue
        bucket = _bucket_for(buckets, person_id, item.observed_at.date())
        bucket.include_time(item.observed_at)
        bucket.flags.update(item.data_quality_flags)

    deduped = _deduplicate(usable, duplicate_dates)
    for duplicate_date in duplicate_dates:
        _bucket_for(buckets, person_id, duplicate_date).flags.add("duplicate_observation")
    resolved, simultaneous_overlap_times = _resolve_simultaneous_device_observations(deduped)

    grouped_by_device: dict[str, list[BehaviorObservation]] = defaultdict(list)
    for item in resolved:
        grouped_by_device[item.device_id or "__single_device__"].append(item)

    intervals: list[_Interval] = []
    for device_id, device_observations in grouped_by_device.items():
        ordered = sorted(device_observations, key=_observation_sort_key)
        for start, end in zip(ordered, ordered[1:]):
            interval = _build_interval(person_id, device_id, start, end, config)
            if interval is None:
                continue
            if "large_observation_gap" in interval.flags:
                _add_flag_to_dates(buckets, person_id, (start.observed_at, end.observed_at), "large_observation_gap")
                continue
            if "non_positive_observation_gap" in interval.flags:
                _add_flag_to_dates(
                    buckets,
                    person_id,
                    (start.observed_at, end.observed_at),
                    "non_positive_observation_gap",
                )
                continue
            intervals.append(interval)
            for flag in interval.flags:
                if flag != "large_observation_gap":
                    _add_flag_to_interval_dates(buckets, person_id, interval, flag, config)

    point_overlap_times = simultaneous_overlap_times | _device_overlap_observation_times(
        resolved,
        intervals,
    )
    _, pieces, interval_overlap_segments = _resolve_interval_overlaps(intervals, config)
    for timestamp in point_overlap_times:
        overlap_time = datetime.fromtimestamp(timestamp, tz=config.timezone_info)
        _bucket_for(buckets, person_id, overlap_time.date()).flags.add(
            "overlapping_device_observations"
        )
    for overlap_start, overlap_end in interval_overlap_segments:
        for day, _, _ in _split_at_midnight(overlap_start, overlap_end, config):
            _bucket_for(buckets, person_id, day).flags.add("overlapping_device_observations")

    sorted_point_overlap_times = sorted(point_overlap_times)
    for interval, piece_start, piece_end, piece_overlapped in pieces:
        for day, segment_start, segment_end in _split_at_midnight(piece_start, piece_end, config):
            bucket = _bucket_for(buckets, person_id, day)
            bucket.include_time(segment_start)
            bucket.include_time(segment_end)
            duration = _elapsed_seconds(segment_start, segment_end)
            if duration <= 0:
                continue
            bucket.observation_seconds += duration
            if not interval.valid:
                continue

            bucket.valid_observation_seconds += duration
            if interval.motion_proxy is not None:
                bucket.activity_volume += interval.motion_proxy * duration
                if _is_active(interval.motion_proxy, config.active_motion_threshold):
                    bucket.active_valid_seconds += duration
            night_seconds = _night_overlap_seconds(segment_start, segment_end, config)
            bucket.nighttime_valid_seconds += night_seconds
            if interval.motion_proxy is not None and _is_active(
                interval.motion_proxy,
                config.active_motion_threshold,
            ):
                bucket.nighttime_active_valid_seconds += night_seconds
            bucket.scene_seconds[interval.start.scene_region] += duration

        # A transition is evidence about the complete adjacent interval. If any
        # competing device overlapped it, suppress it to avoid inventing a scene move.
        if interval.valid and interval.transition and not piece_overlapped:
            overlap_index = bisect_left(
                sorted_point_overlap_times,
                _epoch_seconds(interval.start_at),
            )
            point_overlap_in_interval = (
                overlap_index < len(sorted_point_overlap_times)
                and sorted_point_overlap_times[overlap_index] <= _epoch_seconds(interval.end_at)
            )
            if not point_overlap_in_interval:
                transition_day = interval.end_at.date()
                _bucket_for(buckets, person_id, transition_day).scene_transition_count += 1

    for bucket in buckets.values():
        if bucket.observation_seconds <= 0:
            bucket.flags.add("no_observation_seconds")
        if bucket.valid_observation_seconds <= 0:
            bucket.flags.add("no_valid_observation_seconds")
        if bucket.nighttime_valid_seconds <= 0:
            bucket.flags.add("no_nighttime_observation_seconds")

    if not buckets:
        raise ValueError(
            f"person {person_id!r} has no usable absolute event time; "
            "observed_at or session_start_time with timestamp_sec is required"
        )

    return [_finalize_bucket(bucket) for bucket in buckets.values()]


def _validate_device_contract(person_id: str, observations: list[BehaviorObservation]) -> None:
    device_ids = {item.device_id for item in observations if item.device_id is not None}
    if device_ids and any(item.device_id is None for item in observations):
        raise MentalHealthDataError(
            f"person {person_id!r}: device_id is required when named devices are present; "
            "refusing cross-device merge"
        )


def _deduplicate(
    observations: list[BehaviorObservation],
    duplicate_dates: set[date],
) -> list[BehaviorObservation]:
    selected: dict[tuple[str, str, str, int], BehaviorObservation] = {}
    for observation in observations:
        device_key = observation.device_id or "__single_device__"
        key = (
            observation.person_id,
            device_key,
            observation.observed_at.isoformat() if observation.observed_at else "",
            observation.frame_id,
        )
        previous = selected.get(key)
        if previous is None:
            selected[key] = observation
            continue
        duplicate_dates.add(observation.observed_at.date())
        if _observation_quality_rank(observation) < _observation_quality_rank(previous):
            selected[key] = observation
    return list(selected.values())


def _observation_quality_rank(observation: BehaviorObservation) -> tuple[Any, ...]:
    return (
        0 if observation.usable_for_valid_interval else 1,
        -(observation.keypoint_quality if observation.keypoint_quality is not None else -1.0),
        -len(observation.core_coordinates),
        observation.device_id or "",
        observation.scene_region,
        observation.core_coordinates,
        observation.data_quality_flags,
        observation.timestamp_sec if observation.timestamp_sec is not None else math.inf,
        observation.record_number,
    )


def _resolve_simultaneous_device_observations(
    observations: list[BehaviorObservation],
) -> tuple[list[BehaviorObservation], set[float]]:
    by_time: dict[float, list[BehaviorObservation]] = defaultdict(list)
    for observation in observations:
        if observation.observed_at is None:
            continue
        by_time[_epoch_seconds(observation.observed_at)].append(observation)

    resolved: list[BehaviorObservation] = []
    overlap_times: set[float] = set()
    for timestamp in sorted(by_time):
        items = by_time[timestamp]
        devices = {item.device_id or "__single_device__" for item in items}
        if len(devices) > 1:
            overlap_times.add(timestamp)
            resolved.append(min(items, key=_observation_quality_rank))
        else:
            resolved.extend(items)
    return resolved, overlap_times


def _build_interval(
    person_id: str,
    device_id: str,
    start: BehaviorObservation,
    end: BehaviorObservation,
    config: AggregationConfig,
) -> _Interval | None:
    if start.observed_at is None or end.observed_at is None:
        return None
    delta = _elapsed_seconds(start.observed_at, end.observed_at)
    if delta <= 0:
        return _Interval(
            person_id=person_id,
            device_id=device_id,
            start=start,
            end=end,
            start_at=start.observed_at,
            end_at=end.observed_at,
            duration_seconds=0.0,
            motion_proxy=None,
            valid=False,
            quality_score=0.0,
            common_keypoint_count=0,
            transition=False,
            flags=("non_positive_observation_gap",),
        )
    if delta > config.max_gap_seconds:
        return _Interval(
            person_id=person_id,
            device_id=device_id,
            start=start,
            end=end,
            start_at=start.observed_at,
            end_at=end.observed_at,
            duration_seconds=0.0,
            motion_proxy=None,
            valid=False,
            quality_score=0.0,
            common_keypoint_count=0,
            transition=False,
            flags=("large_observation_gap",),
        )

    start_points = start.coordinate_map
    end_points = end.coordinate_map
    common_names = [name for name in config.core_keypoints if name in start_points and name in end_points]
    flags = list(start.data_quality_flags) + list(end.data_quality_flags)
    flags = _dedupe_flags(flags)
    valid = start.usable_for_valid_interval and end.usable_for_valid_interval
    if len(common_names) < config.min_common_core_keypoints:
        flags.append("insufficient_common_core_keypoints")
        valid = False

    motion_proxy: float | None = None
    if common_names:
        displacements = [
            math.hypot(
                end_points[name][0] - start_points[name][0],
                end_points[name][1] - start_points[name][1],
            )
            for name in common_names
        ]
        motion_proxy = median(displacements) / delta
    elif valid:
        valid = False
        flags.append("insufficient_common_core_keypoints")

    quality_values = [value for value in (start.keypoint_quality, end.keypoint_quality) if value is not None]
    quality_score = min(quality_values) if quality_values else 0.0
    transition = (
        _known_scene(start.scene_region)
        and _known_scene(end.scene_region)
        and start.scene_region != end.scene_region
    )
    return _Interval(
        person_id=person_id,
        device_id=device_id,
        start=start,
        end=end,
        start_at=start.observed_at,
        end_at=end.observed_at,
        duration_seconds=delta,
        motion_proxy=motion_proxy,
        valid=valid,
        quality_score=quality_score,
        common_keypoint_count=len(common_names),
        transition=transition,
        flags=tuple(_dedupe_flags(flags)),
    )


def _resolve_interval_overlaps(
    intervals: list[_Interval],
    config: AggregationConfig,
) -> tuple[
    bool,
    list[tuple[_Interval, datetime, datetime, bool]],
    list[tuple[datetime, datetime]],
]:
    if not intervals:
        return False, [], []
    starts_at: dict[float, list[_Interval]] = defaultdict(list)
    ends_at: dict[float, list[_Interval]] = defaultdict(list)
    for interval in intervals:
        starts_at[_epoch_seconds(interval.start_at)].append(interval)
        ends_at[_epoch_seconds(interval.end_at)].append(interval)
    boundaries = sorted(set(starts_at) | set(ends_at))
    raw_pieces: list[tuple[_Interval, float, float, bool]] = []
    raw_overlap_segments: list[tuple[float, float]] = []
    overlapped_interval_ids: set[int] = set()
    active: dict[int, _Interval] = {}
    overlap_present = False
    for left, right in zip(boundaries, boundaries[1:]):
        if right <= left:
            continue
        for interval in ends_at.get(left, ()):
            active.pop(id(interval), None)
        for interval in starts_at.get(left, ()):
            active[id(interval)] = interval
        active_intervals = list(active.values())
        if not active_intervals:
            continue
        overlapped = len({interval.device_id for interval in active_intervals}) > 1
        if overlapped:
            overlap_present = True
            overlapped_interval_ids.update(id(interval) for interval in active_intervals)
            raw_overlap_segments.append((left, right))
        selected = min(active_intervals, key=_interval_rank)
        raw_pieces.append((selected, left, right, overlapped))
    pieces = [
        (
            interval,
            datetime.fromtimestamp(left, tz=config.timezone_info),
            datetime.fromtimestamp(right, tz=config.timezone_info),
            segment_overlapped or id(interval) in overlapped_interval_ids,
        )
        for interval, left, right, segment_overlapped in raw_pieces
    ]
    overlap_segments = [
        (
            datetime.fromtimestamp(left, tz=config.timezone_info),
            datetime.fromtimestamp(right, tz=config.timezone_info),
        )
        for left, right in raw_overlap_segments
    ]
    return overlap_present, pieces, overlap_segments


def _interval_rank(interval: _Interval) -> tuple[int, float, int, str, int, int]:
    return (
        0 if interval.valid else 1,
        -interval.quality_score,
        -interval.common_keypoint_count,
        interval.device_id,
        interval.start.frame_id,
        interval.end.frame_id,
    )


def _device_overlap_observation_times(
    observations: list[BehaviorObservation],
    intervals: list[_Interval],
) -> set[float]:
    devices = {observation.device_id or "__single_device__" for observation in observations}
    if len(devices) <= 1:
        return set()

    by_time: dict[float, set[str]] = defaultdict(set)
    for observation in observations:
        if observation.observed_at is None:
            continue
        timestamp = _epoch_seconds(observation.observed_at)
        by_time[timestamp].add(observation.device_id or "__single_device__")
    overlap_times = {timestamp for timestamp, device_ids in by_time.items() if len(device_ids) > 1}
    intervals_by_device: dict[str, list[_Interval]] = defaultdict(list)
    for interval in intervals:
        intervals_by_device[interval.device_id].append(interval)
    starts_by_device: dict[str, list[float]] = {}
    for device_id, device_intervals in intervals_by_device.items():
        device_intervals.sort(key=lambda interval: _epoch_seconds(interval.start_at))
        starts_by_device[device_id] = [
            _epoch_seconds(interval.start_at) for interval in device_intervals
        ]

    for observation in observations:
        if observation.observed_at is None:
            continue
        timestamp = _epoch_seconds(observation.observed_at)
        device_id = observation.device_id or "__single_device__"
        for other_device_id, device_intervals in intervals_by_device.items():
            if other_device_id == device_id:
                continue
            candidate_index = bisect_right(starts_by_device[other_device_id], timestamp) - 1
            if candidate_index < 0:
                continue
            candidate = device_intervals[candidate_index]
            if timestamp <= _epoch_seconds(candidate.end_at):
                overlap_times.add(timestamp)
                break
    return overlap_times


def _split_at_midnight(
    start: datetime,
    end: datetime,
    config: AggregationConfig,
) -> list[tuple[date, datetime, datetime]]:
    segments: list[tuple[date, datetime, datetime]] = []
    cursor_timestamp = _epoch_seconds(start)
    end_timestamp = _epoch_seconds(end)
    while cursor_timestamp < end_timestamp:
        cursor = datetime.fromtimestamp(cursor_timestamp, tz=config.timezone_info)
        next_midnight = datetime.combine(
            cursor.date() + timedelta(days=1),
            time.min,
            tzinfo=config.timezone_info,
        )
        segment_end_timestamp = min(end_timestamp, _epoch_seconds(next_midnight))
        if segment_end_timestamp <= cursor_timestamp:
            raise ValueError("Unable to advance across configured timezone day boundary")
        segment_end = datetime.fromtimestamp(segment_end_timestamp, tz=config.timezone_info)
        segments.append((cursor.date(), cursor, segment_end))
        cursor_timestamp = segment_end_timestamp
    return segments


def _night_overlap_seconds(start: datetime, end: datetime, config: AggregationConfig) -> float:
    start_timestamp = _epoch_seconds(start)
    end_timestamp = _epoch_seconds(end)
    if end_timestamp <= start_timestamp:
        return 0.0
    night_start = config.night_start_time
    night_end = config.night_end_time
    total = 0.0
    for offset in (-1, 0, 1):
        current_date = start.date() + timedelta(days=offset)
        window_start = datetime.combine(current_date, night_start, tzinfo=config.timezone_info)
        end_date = current_date + timedelta(days=1) if night_start > night_end else current_date
        window_end = datetime.combine(end_date, night_end, tzinfo=config.timezone_info)
        overlap_start = max(start_timestamp, _epoch_seconds(window_start))
        overlap_end = min(end_timestamp, _epoch_seconds(window_end))
        if overlap_end > overlap_start:
            total += overlap_end - overlap_start
    return total


def _bucket_for(
    buckets: dict[tuple[str, date], _DayAccumulator],
    person_id: str,
    day: date,
) -> _DayAccumulator:
    key = (person_id, day)
    if key not in buckets:
        buckets[key] = _DayAccumulator(person_id=person_id, date=day)
    return buckets[key]


def _add_flag_to_dates(
    buckets: dict[tuple[str, date], _DayAccumulator],
    person_id: str,
    values: tuple[datetime | None, datetime | None],
    flag: str,
) -> None:
    for value in values:
        if value is not None:
            _bucket_for(buckets, person_id, value.date()).flags.add(flag)


def _add_flag_to_interval_dates(
    buckets: dict[tuple[str, date], _DayAccumulator],
    person_id: str,
    interval: _Interval,
    flag: str,
    config: AggregationConfig,
) -> None:
    for day, _, _ in _split_at_midnight(interval.start_at, interval.end_at, config):
        _bucket_for(buckets, person_id, day).flags.add(flag)


def _finalize_bucket(bucket: _DayAccumulator) -> dict[str, Any]:
    observation_seconds = _round_measurement(bucket.observation_seconds)
    valid_seconds = _round_measurement(bucket.valid_observation_seconds)
    active_ratio = (
        _round(bucket.active_valid_seconds / bucket.valid_observation_seconds)
        if bucket.valid_observation_seconds > 0
        else None
    )
    nighttime_ratio = (
        _round(bucket.nighttime_active_valid_seconds / bucket.nighttime_valid_seconds)
        if bucket.nighttime_valid_seconds > 0
        else None
    )
    coverage = (
        _round(bucket.valid_observation_seconds / bucket.observation_seconds)
        if bucket.observation_seconds > 0
        else None
    )
    if active_ratio is None:
        bucket.flags.add("active_ratio_unavailable")
    if nighttime_ratio is None:
        bucket.flags.add("nighttime_activity_ratio_unavailable")
    scene_distribution = {}
    if bucket.valid_observation_seconds > 0:
        scene_distribution = {
            scene: _round(seconds / bucket.valid_observation_seconds)
            for scene, seconds in sorted(bucket.scene_seconds.items())
        }

    return {
        "person_id": bucket.person_id,
        "date": bucket.date.isoformat(),
        "start_time": bucket.start_time.isoformat() if bucket.start_time is not None else None,
        "end_time": bucket.end_time.isoformat() if bucket.end_time is not None else None,
        "observation_seconds": observation_seconds,
        "valid_observation_seconds": valid_seconds,
        "activity_volume": (
            _round_measurement(bucket.activity_volume)
            if bucket.valid_observation_seconds > 0
            else None
        ),
        "active_ratio": active_ratio,
        "nighttime_activity_ratio": nighttime_ratio,
        "scene_region_distribution": scene_distribution,
        "scene_transition_count": bucket.scene_transition_count,
        "observation_coverage": coverage,
        "data_quality_flags": sorted(bucket.flags),
    }


def _observation_sort_key(observation: BehaviorObservation) -> tuple[float, int, int]:
    if observation.observed_at is None:
        raise ValueError("Cannot sort a behavior observation without absolute event time")
    return _epoch_seconds(observation.observed_at), observation.frame_id, observation.record_number


def _known_scene(scene: str) -> bool:
    return bool(scene) and scene.strip().lower() not in {"unknown", "none", "null"}


def _is_active(motion_proxy: float, threshold: float) -> bool:
    return motion_proxy >= threshold or math.isclose(
        motion_proxy,
        threshold,
        rel_tol=1e-9,
        abs_tol=1e-12,
    )


def _dedupe_flags(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _epoch_seconds(value: datetime) -> float:
    return value.timestamp()


def _elapsed_seconds(start: datetime, end: datetime) -> float:
    return _epoch_seconds(end) - _epoch_seconds(start)


def _round(value: float) -> float:
    return round(float(value), 4)


def _round_measurement(value: float) -> float:
    return round(float(value), 6)
