from __future__ import annotations

from typing import Any, Iterable, Mapping

from elderly_monitoring.modules.mental_health.feature_extraction.activity import (
    aggregate_activity_windows,
    aggregate_daytime_activity_from_windows,
)
from elderly_monitoring.service.schemas import DaytimeActivityRequest


def build_daytime_activity_result(request: DaytimeActivityRequest) -> dict[str, Any]:
    """Run the mental-health daytime activity feature pipeline for one person."""
    frames = [
        _with_person_id(item.model_dump(mode="json", exclude_none=True), request.person_id)
        for item in request.frames
    ]
    windows = [
        _with_person_id(item.model_dump(mode="json", exclude_none=True), request.person_id)
        for item in request.windows
    ]

    quality_flags: list[str] = []
    if frames:
        frames = _apply_roi_annotations(
            frames,
            request.roi_annotations,
            default_width=request.image_width,
            default_height=request.image_height,
        )
        windows = aggregate_activity_windows(frames)
        quality_flags.append("aggregated_from_frames")
    elif windows:
        quality_flags.append("aggregated_from_windows")
    else:
        raise ValueError("frames or windows must be provided")

    daily_features = aggregate_daytime_activity_from_windows(
        windows,
        sleep_records=request.sleep_records,
        history_daily_features=request.history_daily_features,
    )
    if request.date is not None:
        requested = request.date.isoformat()
        daily_features = [item for item in daily_features if item.get("date") == requested]
        if not daily_features:
            quality_flags.append("requested_date_has_no_features")

    return {
        "person_id": request.person_id,
        "requested_date": request.date,
        "windows": windows,
        "daily_features": daily_features,
        "quality_flags": quality_flags,
    }


def _with_person_id(record: dict[str, Any], person_id: str) -> dict[str, Any]:
    if not record.get("person_id"):
        record["person_id"] = person_id
    return record


def _apply_roi_annotations(
    frames: list[dict[str, Any]],
    annotations: Iterable[Mapping[str, Any]],
    *,
    default_width: int | None,
    default_height: int | None,
) -> list[dict[str, Any]]:
    rois = _flatten_rois(annotations)
    if not rois:
        return frames

    enriched: list[dict[str, Any]] = []
    for frame in frames:
        if frame.get("zone") and frame.get("room"):
            enriched.append(frame)
            continue

        center = _bbox_center(frame.get("bbox"), str(frame.get("bbox_format") or "xywh"))
        if center is None:
            enriched.append(frame)
            continue

        width = _int_or_none(frame.get("image_width")) or default_width
        height = _int_or_none(frame.get("image_height")) or default_height
        match = _find_roi(center, rois, image_width=width, image_height=height)
        if match is None:
            enriched.append(frame)
            continue

        updated = dict(frame)
        if not updated.get("zone"):
            updated["zone"] = _roi_zone(match)
        if not updated.get("room"):
            updated["room"] = _roi_room(match)
        if not updated.get("zone_id") and match.get("zone_id"):
            updated["zone_id"] = match.get("zone_id")
        if not updated.get("room_id") and match.get("room_id"):
            updated["room_id"] = match.get("room_id")
        enriched.append(updated)
    return enriched


def _flatten_rois(annotations: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    rois: list[Mapping[str, Any]] = []
    for item in annotations:
        nested = item.get("rois")
        if isinstance(nested, list):
            rois.extend(roi for roi in nested if isinstance(roi, Mapping))
        elif isinstance(item, Mapping):
            rois.append(item)
    return rois


def _find_roi(
    center: tuple[float, float],
    rois: Iterable[Mapping[str, Any]],
    *,
    image_width: int | None,
    image_height: int | None,
) -> Mapping[str, Any] | None:
    for roi in rois:
        shape = roi.get("shape")
        if not isinstance(shape, Mapping):
            continue
        if _point_in_roi(center, shape, image_width=image_width, image_height=image_height):
            return roi
    return None


def _point_in_roi(
    point: tuple[float, float],
    shape: Mapping[str, Any],
    *,
    image_width: int | None,
    image_height: int | None,
) -> bool:
    bbox = shape.get("bbox")
    if isinstance(bbox, Mapping):
        normalized = _normalize_bbox(bbox, image_width=image_width, image_height=image_height)
        if normalized is not None:
            x1, y1, x2, y2 = normalized
            if x1 <= point[0] <= x2 and y1 <= point[1] <= y2:
                return True

    points = shape.get("points")
    if isinstance(points, list) and len(points) >= 3:
        polygon = [
            normalized
            for raw in points
            if (normalized := _normalize_point(raw, image_width=image_width, image_height=image_height)) is not None
        ]
        if len(polygon) >= 3:
            return _point_in_polygon(point, polygon)
    return False


def _bbox_center(raw: Any, bbox_format: str) -> tuple[float, float] | None:
    if not isinstance(raw, list | tuple) or len(raw) < 4:
        return None
    try:
        x1, y1, third, fourth = (float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]))
    except (TypeError, ValueError):
        return None
    if bbox_format == "xyxy":
        x, y = (x1 + third) / 2.0, (y1 + fourth) / 2.0
    else:
        x, y = x1 + third / 2.0, y1 + fourth / 2.0
    if x > 1.0 or y > 1.0:
        return None
    return x, y


def _normalize_bbox(
    raw: Mapping[str, Any],
    *,
    image_width: int | None,
    image_height: int | None,
) -> tuple[float, float, float, float] | None:
    try:
        x1 = float(raw["x1"])
        y1 = float(raw["y1"])
        x2 = float(raw["x2"])
        y2 = float(raw["y2"])
    except (KeyError, TypeError, ValueError):
        return None
    if max(abs(x1), abs(x2)) > 1.0:
        if not image_width:
            return None
        x1, x2 = x1 / image_width, x2 / image_width
    if max(abs(y1), abs(y2)) > 1.0:
        if not image_height:
            return None
        y1, y2 = y1 / image_height, y2 / image_height
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def _normalize_point(
    raw: Any,
    *,
    image_width: int | None,
    image_height: int | None,
) -> tuple[float, float] | None:
    if isinstance(raw, Mapping):
        x_raw, y_raw = raw.get("x"), raw.get("y")
    elif isinstance(raw, list | tuple) and len(raw) >= 2:
        x_raw, y_raw = raw[0], raw[1]
    else:
        return None
    try:
        x = float(x_raw)
        y = float(y_raw)
    except (TypeError, ValueError):
        return None
    if x > 1.0:
        if not image_width:
            return None
        x /= image_width
    if y > 1.0:
        if not image_height:
            return None
        y /= image_height
    return x, y


def _point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    x, y = point
    inside = False
    previous_x, previous_y = polygon[-1]
    for current_x, current_y in polygon:
        intersects = (current_y > y) != (previous_y > y)
        if intersects:
            slope_x = (previous_x - current_x) * (y - current_y) / (previous_y - current_y) + current_x
            if x < slope_x:
                inside = not inside
        previous_x, previous_y = current_x, current_y
    return inside


def _roi_zone(roi: Mapping[str, Any]) -> str:
    return str(roi.get("type") or roi.get("zone") or roi.get("zone_id") or "unknown")


def _roi_room(roi: Mapping[str, Any]) -> str:
    return str(roi.get("room") or roi.get("room_id") or "unknown")


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
