from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from numbers import Real
from statistics import fmean
from typing import Any, Iterable, Mapping
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class WanderingDetectionConfig:
    """Engineering thresholds for MVP wandering-clue detection."""

    timezone: str = "Asia/Shanghai"
    window_seconds: int = 300
    step_seconds: int = 30
    frame_width: float = 640.0
    frame_height: float = 480.0
    grid_size: int = 8
    turn_angle_degrees: float = 60.0
    min_duration_seconds: float = 60.0
    event_duration_seconds: float = 90.0
    random_duration_seconds: float = 120.0
    min_path_length: float = 0.05
    min_track_quality: float = 0.60
    candidate_score_threshold: float = 45.0
    event_score_threshold: float = 65.0
    max_path_efficiency_pacing: float = 0.35
    max_path_efficiency_lapping: float = 0.25
    max_path_efficiency_random: float = 0.40
    min_turn_count_pacing: int = 4
    min_turn_count_random: int = 6
    min_revisit_ratio_random: float = 0.30
    min_loop_score_lapping: float = 0.60
    min_major_axis_ratio_pacing: float = 2.50
    max_point_gap_seconds: float = 10.0
    high_risk_roi_hover_seconds: float = 30.0
    doorway_hover_risk_seconds: float = 60.0
    max_ignore_area_ratio: float = 0.50
    night_start: str = "22:00"
    night_end: str = "06:00"

    @property
    def timezone_info(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)

    @property
    def night_start_time(self) -> time:
        return _parse_clock(self.night_start, "night_start")

    @property
    def night_end_time(self) -> time:
        return _parse_clock(self.night_end, "night_end")


@dataclass(frozen=True)
class TrackPoint:
    person_id: str
    observed_at: datetime
    x: float
    y: float
    camera_id: str | None
    device_id: str | None
    track_id: str | None
    roi_id: str | None
    roi_type: str | None
    roi_quality: str | None
    det_confidence: float
    is_interpolated: bool
    quality_flags: tuple[str, ...]


@dataclass(frozen=True)
class WanderingWindow:
    person_id: str
    camera_id: str | None
    device_id: str | None
    track_id: str | None
    window_start: datetime
    window_end: datetime
    points: tuple[TrackPoint, ...]


@dataclass(frozen=True)
class RoiRegion:
    roi_id: str
    roi_type: str
    points: tuple[tuple[float, float], ...]
    bbox: tuple[float, float, float, float] | None = None
    camera_id: str | None = None
    device_id: str | None = None
    roi_quality: str = "confirmed"
    roi_version_id: str | None = None


def detect_wandering_events(
    records: Iterable[Mapping[str, Any]],
    *,
    roi_annotations: Iterable[Mapping[str, Any]] | None = None,
    config: WanderingDetectionConfig | None = None,
    include_normal: bool = False,
) -> list[dict[str, Any]]:
    """Detect event-level wandering clues from person center-point tracks."""
    detection_config = config or WanderingDetectionConfig()
    points = [
        _adapt_track_point(record, config=detection_config, record_number=index)
        for index, record in enumerate(records, start=1)
    ]
    if not points:
        raise ValueError("No track points were provided")
    roi_regions = _adapt_roi_regions(roi_annotations or ())

    grouped: dict[tuple[str, str], list[TrackPoint]] = defaultdict(list)
    for point in points:
        camera_key = point.camera_id or "__single_camera__"
        grouped[(point.person_id, camera_key)].append(point)

    outputs: list[dict[str, Any]] = []
    for (_, _), group_points in grouped.items():
        for window in _sliding_windows(group_points, detection_config):
            event = _evaluate_window(window, detection_config, roi_regions)
            if include_normal or event["decision"] != "normal_movement":
                outputs.append(event)
    return sorted(
        outputs,
        key=lambda item: (
            str(item["window_start"]),
            str(item["person_id"]),
            str(item.get("camera_id") or ""),
        ),
    )


def aggregate_daily_wandering(
    events: Iterable[Mapping[str, Any]],
    *,
    history_daily_features: Iterable[Mapping[str, Any]] | None = None,
    config: WanderingDetectionConfig | None = None,
) -> list[dict[str, Any]]:
    """Aggregate event-level wandering clues into day-level cognitive-safety features."""
    detection_config = config or WanderingDetectionConfig()
    buckets: dict[tuple[str, date], list[Mapping[str, Any]]] = defaultdict(list)
    for event in events:
        if not isinstance(event, Mapping):
            raise ValueError("wandering event records must be objects")
        person_id = _required_string(event, "person_id", "wandering event")
        start = _aware_datetime(event.get("window_start"), "wandering event.window_start", detection_config)
        buckets[(person_id, start.date())].append(event)

    history_by_person = _history_by_person(history_daily_features or ())
    outputs = []
    for (person_id, day), day_events in sorted(buckets.items(), key=lambda item: (item[0][1], item[0][0])):
        outputs.append(
            _finalize_daily_bucket(
                person_id,
                day,
                day_events,
                history=history_by_person.get(person_id, []),
                config=detection_config,
            )
        )
    return outputs


def extract_wandering_features(
    records: Iterable[Mapping[str, Any]],
    *,
    roi_annotations: Iterable[Mapping[str, Any]] | None = None,
    history_daily_features: Iterable[Mapping[str, Any]] | None = None,
    config: WanderingDetectionConfig | None = None,
) -> list[dict[str, Any]]:
    """End-to-end MVP extraction from center-point tracks to daily wandering features."""
    detection_config = config or WanderingDetectionConfig()
    events = detect_wandering_events(records, roi_annotations=roi_annotations, config=detection_config)
    return aggregate_daily_wandering(
        events,
        history_daily_features=history_daily_features,
        config=detection_config,
    )


def _adapt_track_point(
    record: Mapping[str, Any],
    *,
    config: WanderingDetectionConfig,
    record_number: int,
) -> TrackPoint:
    if not isinstance(record, Mapping):
        raise ValueError(f"track point {record_number}: expected an object")
    person_id = _optional_string(record.get("person_id")) or _optional_string(record.get("track_id"))
    if person_id is None:
        raise ValueError(f"track point {record_number}: field 'person_id' or 'track_id' is required")
    observed_at = _aware_datetime(
        record.get("observed_at", record.get("timestamp")),
        f"track point {record_number}.timestamp",
        config,
    )
    x = _finite_number(record.get("x"), f"track point {record_number}.x")
    y = _finite_number(record.get("y"), f"track point {record_number}.y")
    det_confidence = _optional_unit(
        record.get(
            "det_confidence",
            record.get("detection_confidence", record.get("bbox_confidence")),
        ),
        default=1.0,
    )
    flags = []
    if det_confidence < config.min_track_quality:
        flags.append("low_detection_confidence")
    quality_state = _optional_string(record.get("data_quality")) or _optional_string(record.get("quality_state"))
    if quality_state in {"offline", "occluded", "identity_uncertain", "invalid", "rejected", "low_quality"}:
        flags.append(quality_state)
    return TrackPoint(
        person_id=person_id,
        observed_at=observed_at,
        x=x,
        y=y,
        camera_id=_optional_string(record.get("camera_id")) or _optional_string(record.get("device_id")),
        device_id=_optional_string(record.get("device_id")) or _optional_string(record.get("camera_id")),
        track_id=_optional_string(record.get("track_id")),
        roi_id=_optional_string(record.get("roi_id")) or _optional_string(record.get("zone_id")),
        roi_type=_optional_string(record.get("roi_type")) or _optional_string(record.get("zone_type")) or _optional_string(record.get("zone")),
        roi_quality=_optional_string(record.get("roi_quality")) or _optional_string(record.get("zone_quality")),
        det_confidence=det_confidence,
        is_interpolated=bool(record.get("is_interpolated", False)),
        quality_flags=tuple(_dedupe(flags)),
    )


def _adapt_roi_regions(records: Iterable[Mapping[str, Any]]) -> tuple[RoiRegion, ...]:
    regions: list[RoiRegion] = []
    for index, record in enumerate(records, start=1):
        if not isinstance(record, Mapping):
            raise ValueError(f"ROI region {index}: expected an object")
        roi_id = _optional_string(record.get("roi_id")) or _optional_string(record.get("zone_id")) or f"roi_{index}"
        roi_type = _optional_string(record.get("type")) or _optional_string(record.get("roi_type")) or "unknown"
        shape = record.get("shape") if isinstance(record.get("shape"), Mapping) else record
        points = _roi_points(shape.get("points") if isinstance(shape, Mapping) else None)
        if len(points) < 3:
            continue
        bbox = _roi_bbox(shape.get("bbox") if isinstance(shape, Mapping) else None)
        regions.append(
            RoiRegion(
                roi_id=roi_id,
                roi_type=roi_type,
                points=tuple(points),
                bbox=bbox,
                camera_id=_optional_string(record.get("camera_id")),
                device_id=_optional_string(record.get("device_id")),
                roi_quality=_optional_string(record.get("roi_quality")) or _optional_string(record.get("status")) or "confirmed",
                roi_version_id=_optional_string(record.get("roi_version_id")) or _optional_string(record.get("roi_set_id")),
            )
        )
    return tuple(regions)


def _roi_points(value: Any) -> list[tuple[float, float]]:
    points = []
    if not isinstance(value, list):
        return points
    for point in value:
        if not isinstance(point, Mapping):
            continue
        x = _maybe_number(point.get("x"))
        y = _maybe_number(point.get("y"))
        if x is not None and y is not None:
            points.append((max(0.0, min(1.0, x)), max(0.0, min(1.0, y))))
    return points


def _roi_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, Mapping):
        return None
    x1 = _maybe_number(value.get("x1"))
    y1 = _maybe_number(value.get("y1"))
    x2 = _maybe_number(value.get("x2"))
    y2 = _maybe_number(value.get("y2"))
    if None in (x1, y1, x2, y2):
        return None
    return (
        max(0.0, min(1.0, float(x1))),
        max(0.0, min(1.0, float(y1))),
        max(0.0, min(1.0, float(x2))),
        max(0.0, min(1.0, float(y2))),
    )


def _sliding_windows(points: list[TrackPoint], config: WanderingDetectionConfig) -> list[WanderingWindow]:
    ordered = sorted(points, key=lambda item: item.observed_at.timestamp())
    if len(ordered) < 2:
        return []
    start_epoch = ordered[0].observed_at.timestamp()
    end_epoch = ordered[-1].observed_at.timestamp()
    windows = []
    cursor = start_epoch - (start_epoch % config.step_seconds)
    while cursor <= end_epoch:
        window_start = datetime.fromtimestamp(cursor, tz=config.timezone_info)
        window_end = window_start + timedelta(seconds=config.window_seconds)
        window_points = tuple(
            point
            for point in ordered
            if cursor <= point.observed_at.timestamp() < window_end.timestamp()
        )
        if len(window_points) >= 2:
            windows.append(
                WanderingWindow(
                    person_id=ordered[0].person_id,
                    camera_id=ordered[0].camera_id,
                    device_id=ordered[0].device_id,
                    track_id=ordered[0].track_id,
                    window_start=window_start,
                    window_end=window_end,
                    points=window_points,
                )
            )
        cursor += config.step_seconds
    return windows


def _evaluate_window(window: WanderingWindow, config: WanderingDetectionConfig, roi_regions: tuple[RoiRegion, ...]) -> dict[str, Any]:
    points = window.points
    duration_seconds = points[-1].observed_at.timestamp() - points[0].observed_at.timestamp()
    path_length = _path_length(points)
    net_displacement = _distance(points[0], points[-1])
    path_efficiency = net_displacement / path_length if path_length >= config.min_path_length else None
    turn_count = _turn_count(points, config)
    revisit_ratio = _revisit_ratio(points, config)
    loop_score = _loop_score(path_efficiency)
    major_axis_ratio = _major_axis_ratio(points)
    track_quality = _track_quality(points, config)
    roi_features = _roi_features(points, roi_regions, config)
    is_night = _window_overlaps_night(points[0].observed_at, points[-1].observed_at, config)
    quality_flags = _dedupe(flag for point in points for flag in point.quality_flags)
    if path_efficiency is None:
        quality_flags.append("path_length_too_short")
    if roi_features["ignore_area_ratio"] > config.max_ignore_area_ratio:
        quality_flags.append("ignore_area_dominant")

    wandering_score = _wandering_score(
        duration_seconds=duration_seconds,
        path_efficiency=path_efficiency,
        turn_count=turn_count,
        revisit_ratio=revisit_ratio,
        loop_score=loop_score,
        roi_risk_score=roi_features["roi_risk_score"],
    )
    wandering_type = _wandering_type(
        duration_seconds=duration_seconds,
        path_efficiency=path_efficiency,
        turn_count=turn_count,
        revisit_ratio=revisit_ratio,
        loop_score=loop_score,
        major_axis_ratio=major_axis_ratio,
        track_quality=track_quality,
        config=config,
    )
    decision = _decision(
        duration_seconds=duration_seconds,
        path_efficiency=path_efficiency,
        track_quality=track_quality,
        ignore_area_ratio=roi_features["ignore_area_ratio"],
        wandering_score=wandering_score,
        config=config,
    )
    event_type = {
        "record_as_wandering_event": "wandering",
        "record_as_wandering_candidate": "wandering",
        "record_as_low_confidence_candidate": "wandering",
        "normal_movement": "movement_window",
    }[decision]

    return {
        "event_type": event_type,
        "person_id": window.person_id,
        "device_id": window.device_id,
        "camera_id": window.camera_id,
        "track_id": window.track_id,
        "roi_version_id": roi_features["roi_version_id"],
        "window_start": window.window_start.isoformat(),
        "window_end": window.window_end.isoformat(),
        "is_wandering_event": decision == "record_as_wandering_event",
        "wandering_type": wandering_type,
        "wandering_score": _round(wandering_score),
        "duration_seconds": _round(duration_seconds),
        "path_length_norm": _round(path_length),
        "net_displacement_norm": _round(net_displacement),
        "path_length": _round(path_length),
        "net_displacement": _round(net_displacement),
        "path_efficiency": _round_optional(path_efficiency),
        "turn_count": turn_count,
        "revisit_ratio": _round(revisit_ratio),
        "loop_score": _round(loop_score),
        "roi_hits": roi_features["roi_hits"],
        "high_risk_roi_hit": roi_features["high_risk_roi_hit"],
        "doorway_hover_seconds": _round(roi_features["doorway_hover_seconds"]),
        "bathroom_entrance_hover_seconds": _round(roi_features["bathroom_entrance_hover_seconds"]),
        "high_risk_passage_hover_seconds": _round(roi_features["high_risk_passage_hover_seconds"]),
        "ignore_area_ratio": _round(roi_features["ignore_area_ratio"]),
        "roi_quality": roi_features["roi_quality"],
        "major_axis_ratio": _round_optional(None if math.isinf(major_axis_ratio) else major_axis_ratio),
        "track_quality": _round(track_quality),
        "is_night": is_night,
        "decision": decision,
        "quality_flags": quality_flags,
        "diagnosis": False,
    }


def _finalize_daily_bucket(
    person_id: str,
    day: date,
    events: list[Mapping[str, Any]],
    *,
    history: list[Mapping[str, Any]],
    config: WanderingDetectionConfig,
) -> dict[str, Any]:
    event_records = [item for item in events if item.get("decision") == "record_as_wandering_event"]
    candidate_records = [item for item in events if item.get("decision") == "record_as_wandering_candidate"]
    low_confidence_records = [item for item in events if item.get("decision") == "record_as_low_confidence_candidate"]
    scored_records = event_records + candidate_records + low_confidence_records
    night_events = [item for item in event_records if item.get("is_night") is True]
    scores = [_number(item.get("wandering_score")) for item in scored_records]
    event_scores = [_number(item.get("wandering_score")) for item in event_records]
    track_qualities = [_number(item.get("track_quality")) for item in scored_records]
    total_minutes = sum(_number(item.get("duration_seconds")) for item in event_records) / 60.0
    total_night_minutes = sum(_number(item.get("duration_seconds")) for item in night_events) / 60.0
    history_counts = _historical_night_counts(person_id, day, history)
    baseline_sigma = _baseline_sigma(len(night_events), history_counts)
    consecutive_nights = _consecutive_nights_with_wandering(person_id, day, history, len(night_events))

    quality_flags = set()
    for item in events:
        raw_flags = item.get("quality_flags")
        if isinstance(raw_flags, list):
            quality_flags.update(str(flag) for flag in raw_flags if str(flag))
    if low_confidence_records:
        quality_flags.add("low_confidence_wandering_candidate")
    if not event_records and candidate_records:
        quality_flags.add("candidate_only_no_confirmed_event")
    data_quality = _daily_quality(track_qualities, quality_flags)

    return {
        "person_id": person_id,
        "date": day.isoformat(),
        "night_wandering_count": len(night_events),
        "total_night_wandering_minutes": _round(total_night_minutes),
        "total_wandering_minutes": _round(total_minutes),
        "max_wandering_duration_seconds": _round(max((_number(item.get("duration_seconds")) for item in event_records), default=0.0)),
        "pacing_count": _count_type(event_records, "pacing"),
        "lapping_count": _count_type(event_records, "lapping"),
        "random_wandering_count": _count_type(event_records, "random"),
        "mixed_wandering_count": _count_type(event_records, "mixed"),
        "repeated_path_count": sum(1 for item in event_records if _number(item.get("revisit_ratio")) >= config.min_revisit_ratio_random),
        "doorway_wandering_count": sum(1 for item in event_records if _number(item.get("doorway_hover_seconds")) > 0),
        "bathroom_entrance_wandering_count": sum(1 for item in event_records if _number(item.get("bathroom_entrance_hover_seconds")) > 0),
        "high_risk_roi_wandering_count": sum(1 for item in event_records if item.get("high_risk_roi_hit") is True),
        "wandering_candidate_count": len(candidate_records),
        "low_confidence_candidate_count": len(low_confidence_records),
        "wandering_score_daily_max": _round(max(scores, default=0.0)),
        "wandering_score_daily_avg": _round(fmean(scores)) if scores else 0.0,
        "wandering_event_score_avg": _round(fmean(event_scores)) if event_scores else 0.0,
        "consecutive_nights_with_wandering": consecutive_nights,
        "wandering_baseline_sigma": _round_optional(baseline_sigma),
        "wandering_data_quality": data_quality,
        "data_quality_flags": sorted(quality_flags),
        "diagnosis": False,
        "family_copy_hint": _family_copy_hint(len(night_events), consecutive_nights),
    }


def _path_length(points: tuple[TrackPoint, ...]) -> float:
    return sum(_distance(left, right) for left, right in zip(points, points[1:]))


def _distance(left: TrackPoint, right: TrackPoint) -> float:
    return math.hypot(right.x - left.x, right.y - left.y)


def _turn_count(points: tuple[TrackPoint, ...], config: WanderingDetectionConfig) -> int:
    angles = []
    for left, right in zip(points, points[1:]):
        distance = _distance(left, right)
        if distance < config.min_path_length:
            continue
        angles.append(math.atan2(right.y - left.y, right.x - left.x))
    threshold = math.radians(config.turn_angle_degrees)
    count = 0
    accumulated = 0.0
    for previous, current in zip(angles, angles[1:]):
        accumulated += abs(_angle_delta(previous, current))
        if accumulated >= threshold:
            count += 1
            accumulated = 0.0
    return count


def _angle_delta(left: float, right: float) -> float:
    delta = right - left
    while delta > math.pi:
        delta -= 2 * math.pi
    while delta < -math.pi:
        delta += 2 * math.pi
    return delta


def _revisit_ratio(points: tuple[TrackPoint, ...], config: WanderingDetectionConfig) -> float:
    cells: dict[tuple[int, int], int] = defaultdict(int)
    normalized = all(0.0 <= point.x <= 1.0 and 0.0 <= point.y <= 1.0 for point in points)
    width = 1.0 if normalized else config.frame_width
    height = 1.0 if normalized else config.frame_height
    for point in points:
        col = min(config.grid_size - 1, max(0, int((point.x / width) * config.grid_size)))
        row = min(config.grid_size - 1, max(0, int((point.y / height) * config.grid_size)))
        cells[(col, row)] += 1
    if not cells:
        return 0.0
    repeated = sum(1 for count in cells.values() if count > 1)
    return repeated / len(cells)


def _loop_score(path_efficiency: float | None) -> float:
    if path_efficiency is None:
        return 0.0
    return max(0.0, 1.0 - min(path_efficiency / 0.2, 1.0))


def _major_axis_ratio(points: tuple[TrackPoint, ...]) -> float:
    if len(points) < 2:
        return 1.0
    mean_x = fmean(point.x for point in points)
    mean_y = fmean(point.y for point in points)
    centered = [(point.x - mean_x, point.y - mean_y) for point in points]
    var_x = fmean(x * x for x, _ in centered)
    var_y = fmean(y * y for _, y in centered)
    cov_xy = fmean(x * y for x, y in centered)
    trace = var_x + var_y
    determinant = var_x * var_y - cov_xy * cov_xy
    discriminant = max(0.0, trace * trace - 4 * determinant)
    major = (trace + math.sqrt(discriminant)) / 2.0
    minor = (trace - math.sqrt(discriminant)) / 2.0
    if major <= 0:
        return 1.0
    if minor <= 1e-12:
        return math.inf
    return math.sqrt(major / minor)


def _track_quality(points: tuple[TrackPoint, ...], config: WanderingDetectionConfig) -> float:
    confidence = fmean(point.det_confidence for point in points)
    non_interpolated_ratio = sum(1 for point in points if not point.is_interpolated) / len(points)
    if len(points) < 2:
        continuity = 0.0
    else:
        gaps = [
            right.observed_at.timestamp() - left.observed_at.timestamp()
            for left, right in zip(points, points[1:])
        ]
        continuity = sum(1 for gap in gaps if 0 < gap <= config.max_point_gap_seconds) / len(gaps)
    return max(0.0, min(1.0, 0.4 * confidence + 0.3 * continuity + 0.3 * non_interpolated_ratio))


def _roi_features(points: tuple[TrackPoint, ...], roi_regions: tuple[RoiRegion, ...], config: WanderingDetectionConfig) -> dict[str, Any]:
    hover_seconds = {
        "doorway": 0.0,
        "bathroom_entrance": 0.0,
        "high_risk_passage": 0.0,
    }
    roi_hits: list[str] = []
    roi_versions: list[str] = []
    roi_qualities: list[str] = []
    ignore_hits = 0
    valid_hits = 0
    for index, point in enumerate(points):
        roi = _point_roi(point, roi_regions, config)
        if roi is None:
            continue
        valid_hits += 1
        roi_id, roi_type, roi_quality, roi_version_id = roi
        roi_hits.append(roi_id)
        if roi_version_id:
            roi_versions.append(roi_version_id)
        if roi_quality:
            roi_qualities.append(roi_quality)
        if roi_type == "ignore_area":
            ignore_hits += 1
        delta = _point_interval_seconds(points, index, config)
        if roi_type in hover_seconds:
            hover_seconds[roi_type] += delta

    high_risk_roi_hit = (
        hover_seconds["high_risk_passage"] >= config.high_risk_roi_hover_seconds
        or hover_seconds["bathroom_entrance"] >= config.high_risk_roi_hover_seconds
        or hover_seconds["doorway"] >= config.doorway_hover_risk_seconds
    )
    doorway_score = min(hover_seconds["doorway"] / config.doorway_hover_risk_seconds, 1.0) * 0.6
    roi_risk_score = 1.0 if high_risk_roi_hit else doorway_score
    ignore_area_ratio = ignore_hits / valid_hits if valid_hits else 0.0
    return {
        "roi_hits": _dedupe(roi_hits),
        "roi_version_id": _dedupe(roi_versions)[0] if roi_versions else None,
        "roi_quality": _roi_quality(roi_qualities, roi_regions),
        "high_risk_roi_hit": high_risk_roi_hit,
        "doorway_hover_seconds": hover_seconds["doorway"],
        "bathroom_entrance_hover_seconds": hover_seconds["bathroom_entrance"],
        "high_risk_passage_hover_seconds": hover_seconds["high_risk_passage"],
        "ignore_area_ratio": ignore_area_ratio,
        "roi_risk_score": roi_risk_score,
    }


def _point_roi(
    point: TrackPoint,
    roi_regions: tuple[RoiRegion, ...],
    config: WanderingDetectionConfig,
) -> tuple[str, str, str | None, str | None] | None:
    if point.roi_id or point.roi_type:
        roi_type = point.roi_type or "unknown"
        return point.roi_id or roi_type, roi_type, point.roi_quality, None
    normalized = _normalized_point(point, config)
    candidates = []
    for region in roi_regions:
        if region.device_id and point.device_id and region.device_id != point.device_id:
            continue
        if region.camera_id and point.camera_id and region.camera_id != point.camera_id:
            continue
        if region.bbox is not None:
            x1, y1, x2, y2 = region.bbox
            if not (x1 <= normalized[0] <= x2 and y1 <= normalized[1] <= y2):
                continue
        if _point_in_polygon(normalized, region.points):
            candidates.append(region)
    if not candidates:
        return None
    region = candidates[0]
    return region.roi_id, region.roi_type, region.roi_quality, region.roi_version_id


def _normalized_point(point: TrackPoint, config: WanderingDetectionConfig) -> tuple[float, float]:
    if 0.0 <= point.x <= 1.0 and 0.0 <= point.y <= 1.0:
        return point.x, point.y
    return (
        max(0.0, min(1.0, point.x / config.frame_width)),
        max(0.0, min(1.0, point.y / config.frame_height)),
    )


def _point_interval_seconds(points: tuple[TrackPoint, ...], index: int, config: WanderingDetectionConfig) -> float:
    if index + 1 >= len(points):
        return 0.0
    delta = points[index + 1].observed_at.timestamp() - points[index].observed_at.timestamp()
    if delta <= 0:
        return 0.0
    return min(delta, config.max_point_gap_seconds)


def _point_in_polygon(point: tuple[float, float], polygon: tuple[tuple[float, float], ...]) -> bool:
    x, y = point
    inside = False
    if len(polygon) < 3:
        return False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi):
            inside = not inside
        j = i
    return inside


def _roi_quality(qualities: list[str], roi_regions: tuple[RoiRegion, ...]) -> str:
    if not qualities and not roi_regions:
        return "unavailable"
    if any(value != "confirmed" for value in qualities):
        return "unconfirmed"
    return "confirmed"


def _wandering_score(
    *,
    duration_seconds: float,
    path_efficiency: float | None,
    turn_count: int,
    revisit_ratio: float,
    loop_score: float,
    roi_risk_score: float,
) -> float:
    duration_score = min(max(duration_seconds, 0.0) / 180.0, 1.0)
    inefficiency_score = 0.0 if path_efficiency is None else 1.0 - max(0.0, min(path_efficiency, 1.0))
    turn_score = min(turn_count / 8.0, 1.0)
    return max(
        0.0,
        min(
            100.0,
            20 * duration_score
            + 25 * inefficiency_score
            + 15 * turn_score
            + 15 * max(0.0, min(revisit_ratio, 1.0))
            + 10 * max(0.0, min(loop_score, 1.0))
            + 15 * max(0.0, min(roi_risk_score, 1.0)),
        ),
    )


def _wandering_type(
    *,
    duration_seconds: float,
    path_efficiency: float | None,
    turn_count: int,
    revisit_ratio: float,
    loop_score: float,
    major_axis_ratio: float,
    track_quality: float,
    config: WanderingDetectionConfig,
) -> str:
    if path_efficiency is None or track_quality < config.min_track_quality:
        return "uncertain"
    if (
        duration_seconds >= config.event_duration_seconds
        and path_efficiency <= config.max_path_efficiency_pacing
        and turn_count >= config.min_turn_count_pacing
        and major_axis_ratio >= config.min_major_axis_ratio_pacing
    ):
        return "pacing"
    if (
        duration_seconds >= config.event_duration_seconds
        and path_efficiency <= config.max_path_efficiency_lapping
        and loop_score >= config.min_loop_score_lapping
    ):
        return "lapping"
    if (
        duration_seconds >= config.random_duration_seconds
        and path_efficiency <= config.max_path_efficiency_random
        and turn_count >= config.min_turn_count_random
        and revisit_ratio >= config.min_revisit_ratio_random
    ):
        return "random"
    return "mixed"


def _decision(
    *,
    duration_seconds: float,
    path_efficiency: float | None,
    track_quality: float,
    ignore_area_ratio: float,
    wandering_score: float,
    config: WanderingDetectionConfig,
) -> str:
    if duration_seconds < config.min_duration_seconds or path_efficiency is None:
        return "normal_movement"
    if ignore_area_ratio > config.max_ignore_area_ratio:
        return "normal_movement"
    if wandering_score < config.candidate_score_threshold:
        return "normal_movement"
    if track_quality < config.min_track_quality:
        return "record_as_low_confidence_candidate"
    if wandering_score >= config.event_score_threshold:
        return "record_as_wandering_event"
    return "record_as_wandering_candidate"


def _window_overlaps_night(start: datetime, end: datetime, config: WanderingDetectionConfig) -> bool:
    if end < start:
        return False
    start_ts = start.timestamp()
    end_ts = end.timestamp()
    for offset in (-1, 0, 1):
        base_day = start.astimezone(config.timezone_info).date() + timedelta(days=offset)
        night_start = datetime.combine(base_day, config.night_start_time, tzinfo=config.timezone_info)
        night_end_day = base_day + timedelta(days=1) if config.night_start_time > config.night_end_time else base_day
        night_end = datetime.combine(night_end_day, config.night_end_time, tzinfo=config.timezone_info)
        if max(start_ts, night_start.timestamp()) <= min(end_ts, night_end.timestamp()):
            return True
    return False


def _historical_night_counts(person_id: str, day: date, history: list[Mapping[str, Any]]) -> list[float]:
    values = []
    for item in history:
        if str(item.get("person_id", person_id)) != person_id:
            continue
        item_date = _date_value(item.get("date"))
        if item_date is None or item_date >= day:
            continue
        values.append(_number(item.get("night_wandering_count")))
    return values


def _baseline_sigma(current_count: int, history_counts: list[float]) -> float | None:
    if len(history_counts) < 3:
        return None
    mean = fmean(history_counts)
    variance = fmean((value - mean) ** 2 for value in history_counts)
    sigma = math.sqrt(variance)
    if sigma <= 1e-9:
        return current_count - mean
    return (current_count - mean) / sigma


def _consecutive_nights_with_wandering(
    person_id: str,
    day: date,
    history: list[Mapping[str, Any]],
    current_count: int,
) -> int:
    by_day: dict[date, int] = {}
    for item in history:
        if str(item.get("person_id", person_id)) != person_id:
            continue
        item_date = _date_value(item.get("date"))
        if item_date is None or item_date >= day:
            continue
        by_day[item_date] = int(_number(item.get("night_wandering_count")))
    by_day[day] = current_count
    count = 0
    cursor = day
    while by_day.get(cursor, 0) > 0:
        count += 1
        cursor -= timedelta(days=1)
    return count


def _history_by_person(records: Iterable[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        if not isinstance(record, Mapping):
            continue
        person_id = _optional_string(record.get("person_id"))
        if person_id is not None:
            grouped[person_id].append(record)
    return grouped


def _count_type(events: list[Mapping[str, Any]], value: str) -> int:
    return sum(1 for item in events if item.get("wandering_type") == value)


def _daily_quality(track_qualities: list[float], flags: set[str]) -> str:
    if not track_qualities:
        return "data_insufficient"
    average = fmean(track_qualities)
    if average < 0.60 or any("low" in flag for flag in flags):
        return "low"
    if average < 0.80 or flags:
        return "medium"
    return "high"


def _family_copy_hint(night_count: int, consecutive_nights: int) -> str:
    if night_count >= 2 and consecutive_nights >= 2:
        return "近两晚出现多次夜间反复走动线索，建议家属确认夜间安全。"
    if night_count > 0:
        return "夜间出现反复走动线索，建议家属关注休息和居家安全。"
    return "未发现需要单独提醒的夜间徘徊线索。"


def _required_string(record: Mapping[str, Any], field: str, context: str) -> str:
    value = _optional_string(record.get(field))
    if value is None:
        raise ValueError(f"{context}: field '{field}' must be a non-empty string")
    return value


def _aware_datetime(value: Any, field: str, config: WanderingDetectionConfig) -> datetime:
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


def _date_value(value: Any) -> date | None:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return date.fromisoformat(value.strip())
        except ValueError:
            return None
    return None


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be a finite number")
    return number


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        return 0.0
    number = float(value)
    return number if math.isfinite(number) else 0.0


def _maybe_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _optional_unit(value: Any, *, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, Real):
        return default
    number = float(value)
    if not math.isfinite(number):
        return default
    return max(0.0, min(1.0, number))


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _parse_clock(value: str, field: str) -> time:
    try:
        hour_text, minute_text = value.split(":", 1)
        return time(hour=int(hour_text), minute=int(minute_text))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"{field} must use HH:MM format") from exc


def _dedupe(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _round(value: float) -> float:
    return round(float(value), 4)


def _round_optional(value: float | None) -> float | None:
    return None if value is None else _round(value)
