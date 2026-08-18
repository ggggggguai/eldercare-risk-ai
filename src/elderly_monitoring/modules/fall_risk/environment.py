"""Pure feature functions for the optional environment risk branch."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence


def clamp01(value: float | int | None) -> float:
    if value is None:
        return 0.0
    return max(0.0, min(1.0, float(value)))


def low_light_score(
    illumination_lux: float | int | None,
    *,
    normal_lux: float | int | None,
    severe_low_lux: float | int | None,
    illumination_status: str = "ok",
) -> float | None:
    """Return a calibrated 0--1 weak-light score or ``None`` if unavailable."""
    if str(illumination_status).lower() != "ok" or illumination_lux is None:
        return None
    try:
        lux = float(illumination_lux)
        normal = float(normal_lux) if normal_lux is not None else math.nan
        severe = float(severe_low_lux) if severe_low_lux is not None else math.nan
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (lux, normal, severe)):
        return None
    if normal <= severe or severe < 0 or lux < 0:
        return None
    if lux >= normal:
        return 0.0
    if lux <= severe:
        return 1.0
    return round(clamp01((normal - lux) / (normal - severe)), 4)


def behavior_anchor(scores: Mapping[str, Any]) -> float | None:
    """Use only valid gait, sit-stand and near-fall branches as anchors."""
    available: list[float] = []
    mask = scores.get("environment_behavior_mask")
    mask = mask if isinstance(mask, Mapping) else {}
    for name in ("gait_risk_score", "sit_stand_risk_score", "near_fall_event_score"):
        if mask.get(name) is False:
            continue
        value = scores.get(name)
        if value is None:
            continue
        available.append(clamp01(value))
    return max(available) if available else None


def point_in_polygon(point: tuple[float, float], polygon: Sequence[Sequence[float]]) -> bool:
    """Ray-casting point-in-polygon for normalized ROI coordinates."""
    if len(polygon) < 3:
        return False
    x, y = point
    inside = False
    previous = polygon[-1]
    try:
        px, py = float(previous[0]), float(previous[1])
    except (TypeError, IndexError, ValueError):
        return False
    for current in polygon:
        try:
            cx, cy = float(current[0]), float(current[1])
        except (TypeError, IndexError, ValueError):
            return False
        intersects = ((cy > y) != (py > y)) and (
            x < (px - cx) * (y - cy) / ((py - cy) or 1e-12) + cx
        )
        if intersects:
            inside = not inside
        px, py = cx, cy
    return inside


def _usable_point(point: Mapping[str, Any] | None) -> tuple[float, float] | None:
    if not isinstance(point, Mapping):
        return None
    if point.get("valid") is not True or point.get("is_jump_outlier") is True:
        return None
    try:
        x, y = float(point["x"]), float(point["y"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return x, y


def _ankle_points(record: Mapping[str, Any]) -> tuple[tuple[float, float], ...]:
    points = {
        str(point.get("name")): point
        for point in record.get("keypoints", [])
        if isinstance(point, Mapping)
    }
    left = _usable_point(points.get("left_ankle"))
    right = _usable_point(points.get("right_ankle"))
    return tuple(point for point in (left, right) if point is not None)


@dataclass(frozen=True)
class WaterExposureResult:
    score: float | None
    status: str
    reason: str | None
    active_probe_names: tuple[str, ...] = ()
    usable_frame_count: int = 0
    hit_count: int = 0


def water_exposure_score(
    records: Iterable[Mapping[str, Any]],
    water_probes: Mapping[str, Mapping[str, Any]],
    probe_rois: Mapping[str, Mapping[str, Any]],
    *,
    window_frames: int = 3,
    min_usable_frames: int = 2,
    min_hits: int = 2,
    max_span_sec: float = 1.0,
) -> WaterExposureResult:
    """Determine whether a person is repeatedly inside a wet probe ROI."""
    active: list[str] = []
    for name, probe in water_probes.items():
        state = str(probe.get("state", "unknown")).lower()
        sensor_status = str(probe.get("sensor_status", "ok")).lower()
        if sensor_status not in {"ok", "degraded"} or state == "unknown":
            continue
        if state == "water":
            active.append(str(name))
        elif state != "dry":
            return WaterExposureResult(None, "unavailable", "invalid_water_state")
    if not active:
        if any(str(probe.get("state", "unknown")).lower() == "unknown" for probe in water_probes.values()):
            return WaterExposureResult(None, "unavailable", "water_probe_unknown")
        return WaterExposureResult(0.0, "valid", None)

    recent = list(records)[-max(1, int(window_frames)) :]
    usable_times: list[float] = []
    hits = 0
    for record in recent:
        points = _ankle_points(record)
        if not points:
            continue
        timestamp = record.get("timestamp_sec")
        try:
            timestamp_value = float(timestamp)
        except (TypeError, ValueError):
            timestamp_value = None
        usable_times.append(timestamp_value if timestamp_value is not None else float(len(usable_times)))
        frame_hit = False
        for name in active:
            roi = probe_rois.get(name)
            polygon = roi.get("polygon") if isinstance(roi, Mapping) else None
            if isinstance(polygon, Sequence) and any(point_in_polygon(point, polygon) for point in points):
                frame_hit = True
                break
        hits += int(frame_hit)
    if len(usable_times) < int(min_usable_frames):
        return WaterExposureResult(None, "unavailable", "insufficient_ankle_quality", tuple(active), len(usable_times), hits)
    if max(usable_times) - min(usable_times) > float(max_span_sec):
        return WaterExposureResult(None, "unavailable", "roi_window_span_exceeded", tuple(active), len(usable_times), hits)
    if hits < int(min_hits):
        return WaterExposureResult(None, "unavailable", "person_not_in_water_roi", tuple(active), len(usable_times), hits)
    return WaterExposureResult(1.0, "valid", None, tuple(active), len(usable_times), hits)


def interaction_scores(
    *,
    low_light: float | None,
    water_exposure: float | None,
    anchor: float | None,
) -> dict[str, float | None]:
    if anchor is None:
        return {
            "behavior_anchor": None,
            "light_interaction_score": None,
            "water_interaction_score": None,
            "environment_interaction_score": None,
        }
    light = None if low_light is None else round(clamp01(low_light) * clamp01(anchor), 4)
    water = None if water_exposure is None else round(clamp01(water_exposure) * clamp01(anchor), 4)
    candidates = [value for value in (light, water) if value is not None]
    return {
        "behavior_anchor": round(clamp01(anchor), 4),
        "light_interaction_score": light,
        "water_interaction_score": water,
        "environment_interaction_score": max(candidates) if candidates else None,
    }
