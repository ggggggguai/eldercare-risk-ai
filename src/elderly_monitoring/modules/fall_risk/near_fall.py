"""基于清洗后姿态关键点的近跌倒事件检测规则 baseline。

本模块消费 pose_quality.py 输出的稳定 2D 姿态序列，提取短时失衡、
快速下沉恢复、急停恢复和疑似支撑接触等 proxy 特征。输出是局部
near_fall_event_score，不是最终跌倒风险等级，也不是临床判断。
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from elderly_monitoring.modules.fall_risk.pose import write_jsonl


NEAR_FALL_KEYPOINT_NAMES = (
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
WRIST_KEYPOINT_NAMES = ("left_wrist", "right_wrist")
MODEL_VERSION = "near-fall-rule-v0.1"


@dataclass(frozen=True)
class NearFallDetectionConfig:
    # 阈值基于归一化 2D 图像坐标 proxy，仅适合作为可解释 baseline。
    # 换摄像头视角或拿到标注数据后应重新校准。
    window_sec: float = 1.5
    step_sec: float = 0.5
    window_frames: int | None = None
    step_frames: int | None = None
    min_event_frames: int = 5
    merge_gap_sec: float = 0.5
    min_output_score: float = 0.25
    min_usable_frame_ratio: float = 0.60
    min_core_keypoint_coverage: float = 0.70
    lateral_velocity_threshold: float = 0.22
    lateral_acceleration_threshold: float = 0.90
    path_deviation_threshold: float = 0.055
    hip_drop_threshold: float = 0.10
    hip_recovery_ratio_threshold: float = 0.55
    stop_speed_threshold: float = 0.025
    moving_speed_threshold: float = 0.20
    trunk_angle_delta_threshold_deg: float = 22.0
    trunk_angle_speed_threshold_deg: float = 110.0
    crouch_leg_compression_threshold: float = 0.08
    support_lateral_distance_threshold: float = 0.22
    support_speed_threshold: float = 0.35
    support_stationary_range_threshold: float = 0.035
    support_min_frame_ratio: float = 0.45


@dataclass(frozen=True)
class _FrameGeometry:
    record: Mapping[str, Any]
    position: int
    timestamp: float
    frame_id: int
    shoulder_center: tuple[float, float]
    hip_center: tuple[float, float]
    knee_center: tuple[float, float]
    ankle_center: tuple[float, float]
    body_center: tuple[float, float]
    torso_length: float
    leg_extension: float
    trunk_angle_deg: float
    wrists: dict[str, tuple[float, float]]


@dataclass(frozen=True)
class _ScoredWindow:
    start_index: int
    end_index: int
    event_type: str
    score: float
    features: dict[str, Any]
    risk_factors: list[str]
    evidence: list[dict[str, Any]]


@dataclass(frozen=True)
class _MergedCandidate:
    start_index: int
    end_index: int
    event_type: str
    score: float


def extract_near_fall_events(
    records: Iterable[Mapping[str, Any]],
    *,
    config: NearFallDetectionConfig | None = None,
) -> list[dict[str, Any]]:
    near_fall_config = config or NearFallDetectionConfig()
    indexed_records = [(index, dict(record)) for index, record in enumerate(records)]
    groups: dict[tuple[str, str], list[tuple[int, dict[str, Any]]]] = defaultdict(list)

    for index, record in indexed_records:
        groups[_group_key(record)].append((index, record))

    events: list[dict[str, Any]] = []
    for group_items in groups.values():
        sorted_items = sorted(group_items, key=lambda item: _record_sort_key(item[1], item[0]))
        group_records = [record for _, record in sorted_items]
        events.extend(_extract_group_events(group_records, near_fall_config))

    return sorted(
        events,
        key=lambda item: (
            str(item.get("person_id", "")),
            str(item.get("track_id", "")),
            float(item.get("start_time", 0.0) or 0.0),
        ),
    )


def run_near_fall_jsonl(
    *,
    input_path: Path,
    output_path: Path,
    config: NearFallDetectionConfig | None = None,
) -> int:
    records = _read_jsonl(input_path)
    near_fall_records = extract_near_fall_events(records, config=config)
    return write_jsonl(near_fall_records, output_path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def _extract_group_events(records: list[dict[str, Any]], config: NearFallDetectionConfig) -> list[dict[str, Any]]:
    if not records:
        return []

    group_quality = _quality_coverage(records, config)
    if group_quality["insufficient_near_fall_quality"]:
        return [_insufficient_quality_event(records, group_quality)]

    geometries = [
        geometry
        for position, record in enumerate(records)
        if (geometry := _frame_geometry(record, position)) is not None
    ]
    if len(geometries) < config.min_event_frames:
        quality = dict(group_quality)
        quality["insufficient_near_fall_quality"] = True
        return [_insufficient_quality_event(records, quality)]

    scored_windows: list[_ScoredWindow] = []
    for start_index, end_index in _window_ranges(geometries, config):
        event_geometries = geometries[start_index : end_index + 1]
        event_records = _records_between(records, event_geometries[0].timestamp, event_geometries[-1].timestamp)
        quality = _quality_coverage(event_records, config)
        if quality["insufficient_near_fall_quality"]:
            scored_windows.append(
                _ScoredWindow(
                    start_index=start_index,
                    end_index=end_index,
                    event_type="unknown_near_fall",
                    score=0.0,
                    features=_empty_features(),
                    risk_factors=["insufficient_near_fall_quality"],
                    evidence=[],
                )
            )
            continue
        scored_windows.append(_score_window(start_index, end_index, event_geometries, quality, config))

    selected = [
        window
        for window in scored_windows
        if window.score >= config.min_output_score
        or (
            window.score == 0.0
            and "insufficient_near_fall_quality" in window.risk_factors
        )
    ]
    if not selected and scored_windows:
        selected = [max(scored_windows, key=lambda window: window.score)]

    merged_candidates = _merge_windows(selected, geometries, config)
    outputs: list[dict[str, Any]] = []
    for candidate in merged_candidates:
        event_geometries = geometries[candidate.start_index : candidate.end_index + 1]
        event_records = _records_between(records, event_geometries[0].timestamp, event_geometries[-1].timestamp)
        quality = _quality_coverage(event_records, config)
        if quality["insufficient_near_fall_quality"]:
            outputs.append(_insufficient_quality_event(event_records or records, quality))
            continue
        scored = _score_window(candidate.start_index, candidate.end_index, event_geometries, quality, config)
        outputs.append(_build_event(scored, event_geometries, event_records, quality))

    return outputs


def _window_ranges(geometries: list[_FrameGeometry], config: NearFallDetectionConfig) -> list[tuple[int, int]]:
    total = len(geometries)
    if total < config.min_event_frames:
        return []

    if config.window_frames is not None and config.window_frames > 0:
        window_size = max(config.window_frames, config.min_event_frames)
        step_size = max(config.step_frames or window_size, 1)
        ranges = []
        start = 0
        while start + config.min_event_frames <= total:
            end = min(total - 1, start + window_size - 1)
            ranges.append((start, end))
            if end == total - 1:
                break
            start += step_size
        return ranges

    ranges = []
    start = 0
    while start + config.min_event_frames <= total:
        start_time = geometries[start].timestamp
        max_end_time = start_time + max(config.window_sec, 0.0)
        end = start
        while end + 1 < total and geometries[end + 1].timestamp <= max_end_time:
            end += 1
        if end - start + 1 >= config.min_event_frames:
            ranges.append((start, end))
        if end == total - 1:
            break
        next_start_time = start_time + max(config.step_sec, 1e-6)
        next_start = start + 1
        while next_start < total and geometries[next_start].timestamp < next_start_time:
            next_start += 1
        start = max(start + 1, next_start)

    return ranges


def _score_window(
    start_index: int,
    end_index: int,
    event_geometries: list[_FrameGeometry],
    quality: Mapping[str, Any],
    config: NearFallDetectionConfig,
) -> _ScoredWindow:
    features = _near_fall_features(event_geometries, quality, config)
    components = features["score_components"]
    raw_score = (
        0.25 * components["lateral_instability_component"]
        + 0.20 * components["rapid_descent_recovery_component"]
        + 0.15 * components["sudden_stop_recovery_component"]
        + 0.10 * components["support_contact_component"]
        + 0.15 * components["abnormal_crouch_component"]
        + 0.15 * components["trunk_angle_change_component"]
    )
    quality_multiplier = _quality_multiplier(quality, components["quality_penalty"])
    adjusted_score = _clamp(raw_score * quality_multiplier)

    risk_factors = _risk_factors(features, components)
    event_type = _event_type(components, features)
    if not risk_factors:
        event_type = "unknown_near_fall"
    evidence = _evidence(features, components)

    return _ScoredWindow(
        start_index=start_index,
        end_index=end_index,
        event_type=event_type,
        score=round(adjusted_score, 4),
        features=features,
        risk_factors=risk_factors,
        evidence=evidence,
    )


def _near_fall_features(
    event_geometries: list[_FrameGeometry],
    quality: Mapping[str, Any],
    config: NearFallDetectionConfig,
) -> dict[str, Any]:
    start = event_geometries[0]
    end = event_geometries[-1]
    duration = max(0.0, end.timestamp - start.timestamp)
    centers = [(geometry.timestamp, geometry.body_center[0], geometry.body_center[1]) for geometry in event_geometries]
    hips = [(geometry.timestamp, geometry.hip_center[0], geometry.hip_center[1]) for geometry in event_geometries]
    lateral_velocities = _axis_velocities(centers, axis="x")
    lateral_accelerations = _axis_accelerations(centers, axis="x")
    speeds = _speeds(centers)
    body_center_lateral_velocity_peak = max((abs(value) for value in lateral_velocities), default=0.0)
    body_center_lateral_acceleration_peak = max((abs(value) for value in lateral_accelerations), default=0.0)
    body_center_path_deviation = _path_max_deviation(centers)
    body_center_lateral_excursion = _axis_excursion_from_endpoint_trend(centers, axis="x")
    hip_vertical_drop, hip_recovery_ratio = _hip_drop_recovery(hips)
    trunk_angles = [(geometry.timestamp, geometry.trunk_angle_deg) for geometry in event_geometries]
    trunk_angle_delta = _axis_range_time_value(trunk_angles)
    trunk_angle_speed_peak = max((abs(value) for value in _value_velocities(trunk_angles)), default=0.0)
    sudden_stop_ratio, stop_recovery = _sudden_stop_recovery(speeds, config)
    leg_compression = _leg_compression(event_geometries)
    wrist_support_proxy = _support_proxy(event_geometries, config)

    quality_penalty = _quality_penalty(quality)
    lateral_excursion_component = max(
        _ratio_score(body_center_path_deviation, config.path_deviation_threshold),
        _ratio_score(body_center_lateral_excursion, config.path_deviation_threshold),
    )
    lateral_component = max(
        min(
            _ratio_score(body_center_lateral_acceleration_peak, config.lateral_acceleration_threshold),
            lateral_excursion_component,
        ),
        min(
            _ratio_score(body_center_lateral_velocity_peak, config.lateral_velocity_threshold),
            lateral_excursion_component,
        )
        * 0.75,
        lateral_excursion_component,
    )
    rapid_descent_component = _ratio_score(hip_vertical_drop, config.hip_drop_threshold)
    if hip_recovery_ratio < config.hip_recovery_ratio_threshold:
        rapid_descent_component *= max(0.0, hip_recovery_ratio / max(config.hip_recovery_ratio_threshold, 1e-6))

    sudden_stop_component = 0.0
    if stop_recovery:
        sudden_stop_component = max(
            _ratio_score(sudden_stop_ratio, 0.35),
            0.6 * _ratio_score(max(speeds, default=0.0), config.moving_speed_threshold),
        )

    support_component = 1.0 if wrist_support_proxy["suspected"] else 0.0
    abnormal_crouch_component = 0.0
    if hip_recovery_ratio >= config.hip_recovery_ratio_threshold:
        abnormal_crouch_component = min(
            _ratio_score(hip_vertical_drop, config.hip_drop_threshold),
            _ratio_score(leg_compression, config.crouch_leg_compression_threshold),
        )
    trunk_component = max(
        _ratio_score(trunk_angle_delta, config.trunk_angle_delta_threshold_deg),
        _ratio_score(trunk_angle_speed_peak, config.trunk_angle_speed_threshold_deg) * 0.75,
    )

    return {
        "duration_sec": round(duration, 4),
        "body_center_lateral_velocity_peak": round(body_center_lateral_velocity_peak, 4),
        "body_center_lateral_acceleration_peak": round(body_center_lateral_acceleration_peak, 4),
        "hip_vertical_drop": round(hip_vertical_drop, 4),
        "hip_recovery_ratio": round(hip_recovery_ratio, 4),
        "body_center_path_deviation": round(body_center_path_deviation, 4),
        "body_center_lateral_excursion": round(body_center_lateral_excursion, 4),
        "trunk_angle_delta_deg": round(trunk_angle_delta, 4),
        "trunk_angle_speed_peak_deg_per_sec": round(trunk_angle_speed_peak, 4),
        "sudden_stop_ratio": round(sudden_stop_ratio, 4),
        "leg_compression_delta": round(leg_compression, 4),
        "wrist_support_proxy": wrist_support_proxy,
        "score_components": {
            "lateral_instability_component": round(_clamp(lateral_component), 4),
            "rapid_descent_recovery_component": round(_clamp(rapid_descent_component), 4),
            "sudden_stop_recovery_component": round(_clamp(sudden_stop_component), 4),
            "support_contact_component": round(_clamp(support_component), 4),
            "abnormal_crouch_component": round(_clamp(abnormal_crouch_component), 4),
            "trunk_angle_change_component": round(_clamp(trunk_component), 4),
            "quality_penalty": round(_clamp(quality_penalty), 4),
        },
    }


def _risk_factors(features: Mapping[str, Any], components: Mapping[str, float]) -> list[str]:
    risk_factors: list[str] = []
    if components["lateral_instability_component"] >= 0.50:
        risk_factors.append("body_center_lateral_acceleration_spike")
    if (
        float(features["body_center_path_deviation"]) >= 0.055
        or float(features["body_center_lateral_excursion"]) >= 0.055
    ):
        risk_factors.append("body_center_path_deviation")
    if components["rapid_descent_recovery_component"] >= 0.50:
        risk_factors.append("hip_vertical_drop_recovery")
    if components["sudden_stop_recovery_component"] >= 0.50:
        risk_factors.append("sudden_stop_recovery")
    if components["support_contact_component"] >= 1.0:
        risk_factors.append("support_contact_proxy")
    if components["abnormal_crouch_component"] >= 0.50:
        risk_factors.append("abnormal_crouch_recovery")
    if components["trunk_angle_change_component"] >= 0.50:
        risk_factors.append("trunk_angle_sudden_change")
    if components["quality_penalty"] >= 0.25:
        risk_factors.append("reduced_near_fall_quality_coverage")
    return risk_factors


def _event_type(components: Mapping[str, float], features: Mapping[str, Any]) -> str:
    if components["sudden_stop_recovery_component"] >= 0.50:
        return "sudden_stop_recovery"
    strong_components = {
        "stumble_or_lateral_loss": components["lateral_instability_component"],
        "rapid_descent_recovery": components["rapid_descent_recovery_component"],
        "abnormal_crouch_recovery": components["abnormal_crouch_component"],
    }
    best_type, best_score = max(strong_components.items(), key=lambda item: item[1])
    if best_score >= 0.50:
        return best_type
    if bool(features["wrist_support_proxy"].get("suspected")):
        return "support_contact_proxy"
    return "unknown_near_fall"


def _evidence(features: Mapping[str, Any], components: Mapping[str, float]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for name, component in components.items():
        if name == "quality_penalty" or component < 0.50:
            continue
        evidence.append({"trigger": name, "value": round(component, 4)})

    wrist_support = features.get("wrist_support_proxy")
    if isinstance(wrist_support, Mapping):
        for item in wrist_support.get("evidence", []):
            evidence.append({"trigger": "support_contact_proxy", "detail": item})
    return evidence


def _quality_coverage(records: list[dict[str, Any]], config: NearFallDetectionConfig) -> dict[str, Any]:
    total_frames = len(records)
    usable_records = [record for record in records if _usable_for_near_fall(record)]
    usable_frames = len(usable_records)
    total_core_points = total_frames * len(NEAR_FALL_KEYPOINT_NAMES)
    total_wrist_points = total_frames * len(WRIST_KEYPOINT_NAMES)
    valid_core_points = 0
    valid_wrist_points = 0
    interpolated_points = 0
    jump_outliers = 0
    quality_values: list[float] = []

    for record in records:
        quality_values.append(float(record.get("core_keypoint_quality", 0.0)))
        for name in NEAR_FALL_KEYPOINT_NAMES:
            point = _point(record, name)
            if point is None:
                continue
            if _has_point(point):
                valid_core_points += 1
            if point.get("source") == "interpolated":
                interpolated_points += 1
            if point.get("is_jump_outlier") is True:
                jump_outliers += 1
        for name in WRIST_KEYPOINT_NAMES:
            point = _point(record, name)
            if _has_point(point):
                valid_wrist_points += 1
            if point is not None and point.get("source") == "interpolated":
                interpolated_points += 1
            if point is not None and point.get("is_jump_outlier") is True:
                jump_outliers += 1

    usable_frame_ratio = usable_frames / total_frames if total_frames > 0 else 0.0
    core_coverage = valid_core_points / total_core_points if total_core_points > 0 else 0.0
    wrist_coverage = valid_wrist_points / total_wrist_points if total_wrist_points > 0 else 0.0
    total_points = total_core_points + total_wrist_points
    interpolated_ratio = interpolated_points / total_points if total_points > 0 else 0.0
    insufficient = (
        total_frames < config.min_event_frames
        or usable_frame_ratio < config.min_usable_frame_ratio
        or core_coverage < config.min_core_keypoint_coverage
    )
    return {
        "frame_count": total_frames,
        "usable_frame_count": usable_frames,
        "usable_frame_ratio": round(usable_frame_ratio, 4),
        "core_keypoint_coverage": round(core_coverage, 4),
        "wrist_keypoint_coverage": round(wrist_coverage, 4),
        "mean_core_keypoint_quality": round(_mean(quality_values), 4),
        "interpolated_point_ratio": round(interpolated_ratio, 4),
        "jump_outlier_count": jump_outliers,
        "usable_near_fall_window_ratio": round(usable_frame_ratio, 4),
        "insufficient_near_fall_quality": insufficient,
    }


def _quality_penalty(quality: Mapping[str, Any]) -> float:
    usable_penalty = max(0.0, 1.0 - float(quality.get("usable_frame_ratio", 0.0)))
    coverage_penalty = max(0.0, 1.0 - float(quality.get("core_keypoint_coverage", 0.0)))
    interpolation_penalty = float(quality.get("interpolated_point_ratio", 0.0))
    jump_penalty = min(0.35, int(quality.get("jump_outlier_count", 0)) * 0.04)
    return _clamp((0.35 * usable_penalty) + (0.30 * coverage_penalty) + (0.20 * interpolation_penalty) + jump_penalty)


def _quality_multiplier(quality: Mapping[str, Any], quality_penalty: float) -> float:
    usable_frame_ratio = float(quality.get("usable_frame_ratio", 0.0))
    return _clamp(0.4 + (0.6 * usable_frame_ratio) - quality_penalty)


def _merge_windows(
    windows: list[_ScoredWindow],
    geometries: list[_FrameGeometry],
    config: NearFallDetectionConfig,
) -> list[_MergedCandidate]:
    if not windows:
        return []

    sorted_windows = sorted(windows, key=lambda window: (window.start_index, window.end_index))
    merged: list[_MergedCandidate] = []
    current = _MergedCandidate(
        start_index=sorted_windows[0].start_index,
        end_index=sorted_windows[0].end_index,
        event_type=sorted_windows[0].event_type,
        score=sorted_windows[0].score,
    )

    for window in sorted_windows[1:]:
        gap = max(0.0, geometries[window.start_index].timestamp - geometries[current.end_index].timestamp)
        if window.event_type == current.event_type and gap <= config.merge_gap_sec:
            current = _MergedCandidate(
                start_index=current.start_index,
                end_index=max(current.end_index, window.end_index),
                event_type=current.event_type,
                score=max(current.score, window.score),
            )
            continue
        merged.append(current)
        current = _MergedCandidate(
            start_index=window.start_index,
            end_index=window.end_index,
            event_type=window.event_type,
            score=window.score,
        )

    merged.append(current)
    return merged


def _build_event(
    scored: _ScoredWindow,
    event_geometries: list[_FrameGeometry],
    event_records: list[dict[str, Any]],
    quality: Mapping[str, Any],
) -> dict[str, Any]:
    first_geometry = event_geometries[0]
    last_geometry = event_geometries[-1]
    first_record = event_records[0] if event_records else dict(first_geometry.record)
    person_id = str(first_record.get("person_id", "unknown"))
    track_id = first_record.get("track_id")
    start_time = round(first_geometry.timestamp, 4)
    end_time = round(last_geometry.timestamp, 4)
    return {
        "event_id": _event_id(person_id, track_id, start_time, end_time),
        "person_id": person_id,
        "track_id": track_id,
        "scene_region": first_record.get("scene_region"),
        "start_time": start_time,
        "end_time": end_time,
        "frame_start": first_geometry.frame_id,
        "frame_end": last_geometry.frame_id,
        "near_fall_event_score": scored.score,
        "event_type": scored.event_type,
        "near_fall_features": scored.features,
        "quality_coverage": dict(quality),
        "risk_factors": scored.risk_factors,
        "evidence": scored.evidence,
        "model_version": MODEL_VERSION,
    }


def _insufficient_quality_event(records: list[dict[str, Any]], quality: Mapping[str, Any]) -> dict[str, Any]:
    first_record = records[0]
    start_time, end_time = _record_bounds(records)
    frame_start, frame_end = _frame_bounds(records)
    person_id = str(first_record.get("person_id", "unknown"))
    track_id = first_record.get("track_id")
    return {
        "event_id": _event_id(person_id, track_id, start_time, end_time),
        "person_id": person_id,
        "track_id": track_id,
        "scene_region": first_record.get("scene_region"),
        "start_time": start_time,
        "end_time": end_time,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "near_fall_event_score": 0.0,
        "event_type": "unknown_near_fall",
        "near_fall_features": _empty_features(),
        "quality_coverage": dict(quality),
        "risk_factors": ["insufficient_near_fall_quality"],
        "evidence": [],
        "model_version": MODEL_VERSION,
    }


def _empty_features() -> dict[str, Any]:
    return {
        "duration_sec": None,
        "body_center_lateral_velocity_peak": None,
        "body_center_lateral_acceleration_peak": None,
        "hip_vertical_drop": None,
        "hip_recovery_ratio": None,
        "body_center_path_deviation": None,
        "body_center_lateral_excursion": None,
        "trunk_angle_delta_deg": None,
        "trunk_angle_speed_peak_deg_per_sec": None,
        "sudden_stop_ratio": None,
        "leg_compression_delta": None,
        "wrist_support_proxy": {"suspected": False, "evidence": []},
        "score_components": {
            "lateral_instability_component": 0.0,
            "rapid_descent_recovery_component": 0.0,
            "sudden_stop_recovery_component": 0.0,
            "support_contact_component": 0.0,
            "abnormal_crouch_component": 0.0,
            "trunk_angle_change_component": 0.0,
            "quality_penalty": 1.0,
        },
    }


def _frame_geometry(record: Mapping[str, Any], position: int) -> _FrameGeometry | None:
    if not _usable_for_near_fall(record):
        return None

    left_shoulder = _point(record, "left_shoulder")
    right_shoulder = _point(record, "right_shoulder")
    left_hip = _point(record, "left_hip")
    right_hip = _point(record, "right_hip")
    left_knee = _point(record, "left_knee")
    right_knee = _point(record, "right_knee")
    left_ankle = _point(record, "left_ankle")
    right_ankle = _point(record, "right_ankle")
    required_points = [
        left_shoulder,
        right_shoulder,
        left_hip,
        right_hip,
        left_knee,
        right_knee,
        left_ankle,
        right_ankle,
    ]
    if not all(_has_point(point) for point in required_points):
        return None

    shoulder_center = _midpoint(_point_xy(left_shoulder), _point_xy(right_shoulder))
    hip_center = _midpoint(_point_xy(left_hip), _point_xy(right_hip))
    knee_center = _midpoint(_point_xy(left_knee), _point_xy(right_knee))
    ankle_center = _midpoint(_point_xy(left_ankle), _point_xy(right_ankle))
    body_center = (
        (0.45 * shoulder_center[0]) + (0.55 * hip_center[0]),
        (0.45 * shoulder_center[1]) + (0.55 * hip_center[1]),
    )
    torso_length = max(_distance(shoulder_center, hip_center), 1e-6)
    leg_extension = _distance(hip_center, knee_center) + _distance(knee_center, ankle_center)
    trunk_angle = _trunk_angle_deg(shoulder_center, hip_center)
    wrists: dict[str, tuple[float, float]] = {}
    for name in WRIST_KEYPOINT_NAMES:
        point = _point(record, name)
        if _has_point(point):
            wrists[name] = _point_xy(point)

    return _FrameGeometry(
        record=record,
        position=position,
        timestamp=_record_time(record, position),
        frame_id=int(_number_or_default(record.get("frame_id"), position)),
        shoulder_center=shoulder_center,
        hip_center=hip_center,
        knee_center=knee_center,
        ankle_center=ankle_center,
        body_center=body_center,
        torso_length=torso_length,
        leg_extension=leg_extension,
        trunk_angle_deg=trunk_angle,
        wrists=wrists,
    )


def _usable_for_near_fall(record: Mapping[str, Any]) -> bool:
    window_quality = record.get("window_quality")
    if isinstance(window_quality, Mapping) and window_quality.get("usable_for_near_fall") is False:
        return False
    return all(_has_point(_point(record, name)) for name in NEAR_FALL_KEYPOINT_NAMES)


def _support_proxy(event_geometries: list[_FrameGeometry], config: NearFallDetectionConfig) -> dict[str, Any]:
    evidence: list[str] = []
    for wrist_name in WRIST_KEYPOINT_NAMES:
        wrist_points: list[tuple[float, float, float]] = []
        body_points: list[tuple[float, float, float]] = []
        for geometry in event_geometries:
            wrist = geometry.wrists.get(wrist_name)
            if wrist is None:
                continue
            wrist_points.append((geometry.timestamp, wrist[0], wrist[1]))
            body_points.append((geometry.timestamp, geometry.body_center[0], geometry.body_center[1]))

        if len(wrist_points) / len(event_geometries) < config.support_min_frame_ratio:
            continue

        lateral_distances = [wrist[1] - body[1] for wrist, body in zip(wrist_points, body_points, strict=False)]
        max_lateral_distance = max((abs(value) for value in lateral_distances), default=0.0)
        wrist_speeds = _speeds(wrist_points)
        wrist_speed_peak = max(wrist_speeds, default=0.0)
        tail_count = max(2, len(wrist_points) // 3)
        tail_points = [(point[1], point[2]) for point in wrist_points[-tail_count:]]
        stationary_tail = max(_axis_range_xy(tail_points, axis="x"), _axis_range_xy(tail_points, axis="y"))
        close_to_boundary = (
            min(point[1] for point in wrist_points) <= 0.10
            or max(point[1] for point in wrist_points) >= 0.90
        )
        if (
            max_lateral_distance >= config.support_lateral_distance_threshold
            and wrist_speed_peak >= config.support_speed_threshold
            and stationary_tail <= config.support_stationary_range_threshold
        ) or (close_to_boundary and stationary_tail <= config.support_stationary_range_threshold):
            side = "left" if wrist_name == "left_wrist" else "right"
            evidence.append(f"{side}_wrist_moves_toward_side_boundary_then_stabilizes")

    return {"suspected": bool(evidence), "evidence": evidence}


def _hip_drop_recovery(hips: list[tuple[float, float, float]]) -> tuple[float, float]:
    if len(hips) < 3:
        return 0.0, 0.0
    start_y = hips[0][2]
    end_y = hips[-1][2]
    peak_y = max(point[2] for point in hips)
    peak_index = max(range(len(hips)), key=lambda index: hips[index][2])
    before_min = min(point[2] for point in hips[: peak_index + 1])
    after_min = min(point[2] for point in hips[peak_index:])
    drop = max(0.0, peak_y - before_min)
    recovered_amount = max(0.0, peak_y - min(end_y, after_min))
    recovery_ratio = recovered_amount / drop if drop > 1e-6 else 0.0
    if peak_index in {0, len(hips) - 1}:
        return 0.0, 0.0
    return drop, min(1.0, recovery_ratio)


def _sudden_stop_recovery(speeds: list[float], config: NearFallDetectionConfig) -> tuple[float, bool]:
    if len(speeds) < 4:
        return 0.0, False
    low_indices = [index for index, speed in enumerate(speeds) if speed <= config.stop_speed_threshold]
    low_ratio = len(low_indices) / len(speeds)
    if not low_indices:
        return 0.0, False
    for index in low_indices:
        before_moving = any(speed >= config.moving_speed_threshold for speed in speeds[:index])
        after_moving = any(speed >= config.moving_speed_threshold for speed in speeds[index + 1 :])
        if before_moving and after_moving:
            return low_ratio, True
    return low_ratio, False


def _leg_compression(event_geometries: list[_FrameGeometry]) -> float:
    if len(event_geometries) < 3:
        return 0.0
    values = [geometry.leg_extension / max(geometry.torso_length, 1e-6) for geometry in event_geometries]
    start_value = values[0]
    min_value = min(values)
    end_value = values[-1]
    compression = max(0.0, start_value - min_value)
    recovery = max(0.0, end_value - min_value)
    if compression <= 1e-6:
        return 0.0
    return compression if recovery / compression >= 0.45 else 0.0


def _records_between(records: list[dict[str, Any]], start_time: float, end_time: float) -> list[dict[str, Any]]:
    return [record for index, record in enumerate(records) if start_time <= _record_time(record, index) <= end_time]


def _record_bounds(records: list[Mapping[str, Any]]) -> tuple[float, float]:
    times = [_record_time(record, index) for index, record in enumerate(records)]
    return round(min(times), 4), round(max(times), 4)


def _frame_bounds(records: list[Mapping[str, Any]]) -> tuple[int | None, int | None]:
    frames = [_optional_number(record.get("frame_id")) for record in records]
    numeric_frames = [int(frame) for frame in frames if frame is not None]
    if not numeric_frames:
        return None, None
    return min(numeric_frames), max(numeric_frames)


def _event_id(person_id: str, track_id: Any, start_time: float, end_time: float) -> str:
    track = "none" if track_id is None else str(track_id)
    return f"{person_id}_{track}_{start_time:.4f}_{end_time:.4f}"


def _group_key(record: Mapping[str, Any]) -> tuple[str, str]:
    person_id = str(record.get("person_id", "unknown"))
    track_id = record.get("track_id")
    return person_id, "none" if track_id is None else str(track_id)


def _record_sort_key(record: Mapping[str, Any], index: int) -> tuple[float, int, int]:
    timestamp = _number_or_default(record.get("timestamp_sec"), math.inf)
    frame_id = int(_number_or_default(record.get("frame_id"), index))
    return timestamp, frame_id, index


def _record_time(record: Mapping[str, Any], fallback: int) -> float:
    timestamp = _optional_number(record.get("timestamp_sec"))
    if timestamp is not None:
        return timestamp
    frame_id = _optional_number(record.get("frame_id"))
    return float(fallback if frame_id is None else frame_id)


def _point(record: Mapping[str, Any], name: str) -> Mapping[str, Any] | None:
    for point in record.get("keypoints", []):
        if isinstance(point, Mapping) and point.get("name") == name:
            return point
    return None


def _has_point(point: Mapping[str, Any] | None) -> bool:
    if point is None or point.get("valid") is not True or point.get("is_jump_outlier") is True:
        return False
    return _optional_number(point.get("x_smooth")) is not None and _optional_number(point.get("y_smooth")) is not None


def _point_xy(point: Mapping[str, Any] | None) -> tuple[float, float]:
    if point is None:
        return 0.0, 0.0
    return float(point["x_smooth"]), float(point["y_smooth"])


def _midpoint(left: tuple[float, float], right: tuple[float, float]) -> tuple[float, float]:
    return (left[0] + right[0]) / 2.0, (left[1] + right[1]) / 2.0


def _distance(first: tuple[float, float], second: tuple[float, float]) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def _trunk_angle_deg(shoulder_center: tuple[float, float], hip_center: tuple[float, float]) -> float:
    dx = hip_center[0] - shoulder_center[0]
    dy = hip_center[1] - shoulder_center[1]
    if abs(dy) <= 1e-6:
        return 90.0
    return abs(math.degrees(math.atan2(dx, dy)))


def _axis_velocities(points: list[tuple[float, float, float]], *, axis: str) -> list[float]:
    velocities: list[float] = []
    axis_index = 1 if axis == "x" else 2
    for previous, current in zip(points, points[1:], strict=False):
        dt = current[0] - previous[0]
        if dt <= 0:
            continue
        velocities.append((current[axis_index] - previous[axis_index]) / dt)
    return velocities


def _axis_accelerations(points: list[tuple[float, float, float]], *, axis: str) -> list[float]:
    velocities = _axis_velocities(points, axis=axis)
    timestamps = [point[0] for point in points]
    accelerations: list[float] = []
    for index, (previous, current) in enumerate(zip(velocities, velocities[1:], strict=False), start=1):
        dt = timestamps[index + 1] - timestamps[index]
        if dt <= 0:
            continue
        accelerations.append((current - previous) / dt)
    return accelerations


def _value_velocities(points: list[tuple[float, float]]) -> list[float]:
    velocities: list[float] = []
    for previous, current in zip(points, points[1:], strict=False):
        dt = current[0] - previous[0]
        if dt <= 0:
            continue
        velocities.append((current[1] - previous[1]) / dt)
    return velocities


def _speeds(points: list[tuple[float, float, float]]) -> list[float]:
    speeds: list[float] = []
    for previous, current in zip(points, points[1:], strict=False):
        dt = current[0] - previous[0]
        if dt <= 0:
            continue
        distance = math.hypot(current[1] - previous[1], current[2] - previous[2])
        speeds.append(distance / dt)
    return speeds


def _path_max_deviation(points: list[tuple[float, float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    start = points[0]
    end = points[-1]
    line_dx = end[1] - start[1]
    line_dy = end[2] - start[2]
    line_length = math.hypot(line_dx, line_dy)
    if line_length <= 1e-6:
        return _axis_range(points, axis="x")

    distances = []
    for _, x, y in points[1:-1]:
        numerator = abs((line_dy * x) - (line_dx * y) + (end[1] * start[2]) - (end[2] * start[1]))
        distances.append(numerator / line_length)
    return max(distances) if distances else 0.0


def _axis_excursion_from_endpoint_trend(points: list[tuple[float, float, float]], *, axis: str) -> float:
    if len(points) < 3:
        return 0.0
    axis_index = 1 if axis == "x" else 2
    start_value = points[0][axis_index]
    end_value = points[-1][axis_index]
    max_excursion = 0.0
    denominator = max(len(points) - 1, 1)
    for index, point in enumerate(points[1:-1], start=1):
        expected = start_value + ((end_value - start_value) * index / denominator)
        max_excursion = max(max_excursion, abs(point[axis_index] - expected))
    return max_excursion


def _axis_range(points: list[tuple[float, float, float]], *, axis: str) -> float:
    if not points:
        return 0.0
    values = [point[1] if axis == "x" else point[2] for point in points]
    return max(values) - min(values)


def _axis_range_xy(points: list[tuple[float, float]], *, axis: str) -> float:
    if not points:
        return 0.0
    values = [point[0] if axis == "x" else point[1] for point in points]
    return max(values) - min(values)


def _axis_range_time_value(points: list[tuple[float, float]]) -> float:
    if not points:
        return 0.0
    values = [point[1] for point in points]
    return max(values) - min(values)


def _ratio_score(value: float, threshold: float) -> float:
    if threshold <= 0:
        return 0.0
    return _clamp(value / threshold)


def _mean(values: Iterable[float]) -> float:
    value_list = list(values)
    if not value_list:
        return 0.0
    return sum(value_list) / len(value_list)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _optional_number(value: Any) -> float | None:
    try:
        if value is None:
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _number_or_default(value: Any, default: float) -> float:
    number = _optional_number(value)
    return default if number is None else number
