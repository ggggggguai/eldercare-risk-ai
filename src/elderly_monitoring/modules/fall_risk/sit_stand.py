"""基于清洗后姿态关键点的坐站转换能力规则 baseline。

本模块消费 pose_quality.py 输出的稳定关键点序列，识别候选坐站事件并
输出局部坐站风险分。分数只表示坐站转换这一局部能力线索，不是最终
跌倒风险等级，也不是医学诊断结论。
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from elderly_monitoring.modules.fall_risk.pose import write_jsonl


SIT_STAND_KEYPOINT_NAMES = (
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
MODEL_VERSION = "sit-stand-risk-rule-v0.1"


@dataclass(frozen=True)
class SitStandAnalysisConfig:
    # 阈值基于归一化 2D 图像坐标，只适合作为可复现 baseline；
    # 换相机视角或拿到动作标注后应重新校准。
    min_event_frames: int = 5
    min_usable_frame_ratio: float = 0.60
    min_sit_stand_keypoint_coverage: float = 0.70
    movement_start_delta: float = 0.02
    min_vertical_displacement: float = 0.10
    stable_height_epsilon: float = 0.018
    posture_stable_frames: int = 2
    post_stand_window_sec: float = 2.0
    min_post_window_frames: int = 3
    stable_lateral_speed_threshold: float = 0.06
    stable_trunk_angle_speed_threshold_deg: float = 15.0
    normal_duration_sec: float = 3.0
    high_duration_sec: float = 6.0
    failed_attempt_min_displacement: float = 0.04
    failed_attempt_return_tolerance: float = 0.03
    trunk_forward_angle_threshold_deg: float = 30.0
    post_stand_sway_threshold: float = 0.18
    normal_stabilization_sec: float = 0.8
    high_stabilization_sec: float = 2.0
    support_lateral_distance_threshold: float = 0.18
    support_stationary_range_threshold: float = 0.035
    support_min_frame_ratio: float = 0.60


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
    trunk_forward_angle: float
    wrists: dict[str, tuple[float, float]]


@dataclass(frozen=True)
class _CandidateEvent:
    start_index: int
    end_index: int
    transition_type: str


def extract_sit_stand_events(
    records: Iterable[Mapping[str, Any]],
    *,
    config: SitStandAnalysisConfig | None = None,
) -> list[dict[str, Any]]:
    sit_stand_config = config or SitStandAnalysisConfig()
    indexed_records = [(index, dict(record)) for index, record in enumerate(records)]
    groups: dict[tuple[str, str], list[tuple[int, dict[str, Any]]]] = defaultdict(list)

    for index, record in indexed_records:
        groups[_group_key(record)].append((index, record))

    events: list[dict[str, Any]] = []
    for group_items in groups.values():
        sorted_items = sorted(group_items, key=lambda item: _record_sort_key(item[1], item[0]))
        group_records = [record for _, record in sorted_items]
        events.extend(_extract_group_events(group_records, sit_stand_config))

    return sorted(
        events,
        key=lambda item: (
            str(item.get("person_id", "")),
            str(item.get("track_id", "")),
            float(item.get("start_time", 0.0) or 0.0),
        ),
    )


def run_sit_stand_jsonl(
    *,
    input_path: Path,
    output_path: Path,
    config: SitStandAnalysisConfig | None = None,
) -> int:
    records = _read_jsonl(input_path)
    sit_stand_records = extract_sit_stand_events(records, config=config)
    return write_jsonl(sit_stand_records, output_path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def _extract_group_events(records: list[dict[str, Any]], config: SitStandAnalysisConfig) -> list[dict[str, Any]]:
    if not records:
        return []

    group_quality = _quality_coverage(records, config)
    if group_quality["insufficient_sit_stand_quality"]:
        return [_insufficient_quality_event(records, group_quality)]

    geometries = [
        geometry
        for position, record in enumerate(records)
        if (geometry := _frame_geometry(record, position)) is not None
    ]
    if len(geometries) < config.min_event_frames:
        quality = dict(group_quality)
        quality["insufficient_sit_stand_quality"] = True
        return [_insufficient_quality_event(records, quality)]

    outputs: list[dict[str, Any]] = []
    for candidate in _detect_candidates(geometries, config):
        start = geometries[candidate.start_index]
        end = geometries[candidate.end_index]
        event_records = _records_between(records, start.timestamp, end.timestamp)
        event_quality = _quality_coverage(event_records, config)
        if event_quality["insufficient_sit_stand_quality"]:
            outputs.append(_insufficient_quality_event(event_records or records, event_quality))
            continue
        outputs.append(_build_event(candidate, geometries, event_records, event_quality, config))

    return outputs


def _detect_candidates(
    geometries: list[_FrameGeometry],
    config: SitStandAnalysisConfig,
) -> list[_CandidateEvent]:
    candidates: list[_CandidateEvent] = []
    index = 0
    while index <= len(geometries) - config.min_event_frames:
        movement = _find_next_movement(geometries, index, config)
        if movement is None:
            break
        start_index, direction = movement
        end_index = _find_event_end(geometries, start_index, direction, config)
        if end_index is None:
            index = start_index + 1
            continue
        if end_index - start_index + 1 < config.min_event_frames:
            index = start_index + 1
            continue
        transition_type = "sit_to_stand" if direction < 0 else "stand_to_sit"
        candidates.append(_CandidateEvent(start_index, end_index, transition_type))
        index = max(end_index + 1, start_index + 1)
    return candidates


def _find_next_movement(
    geometries: list[_FrameGeometry],
    start_at: int,
    config: SitStandAnalysisConfig,
) -> tuple[int, int] | None:
    for index in range(start_at, len(geometries) - 1):
        anchor_y = geometries[index].hip_center[1]
        for lookahead in range(index + 1, len(geometries)):
            delta = geometries[lookahead].hip_center[1] - anchor_y
            if abs(delta) >= config.movement_start_delta:
                direction = -1 if delta < 0 else 1
                return index, direction
            if lookahead - index >= config.min_event_frames:
                break
    return None


def _find_event_end(
    geometries: list[_FrameGeometry],
    start_index: int,
    direction: int,
    config: SitStandAnalysisConfig,
) -> int | None:
    start_y = geometries[start_index].hip_center[1]
    crossed_threshold = False
    best_index = start_index
    best_displacement = 0.0

    for index in range(start_index + 1, len(geometries)):
        displacement = _directional_displacement(start_y, geometries[index].hip_center[1], direction)
        if displacement > best_displacement:
            best_displacement = displacement
            best_index = index
        if displacement >= config.min_vertical_displacement:
            crossed_threshold = True
        if crossed_threshold and _height_is_stable_from(geometries, index, config):
            return index

    if crossed_threshold and best_displacement >= config.min_vertical_displacement:
        return best_index
    return None


def _height_is_stable_from(
    geometries: list[_FrameGeometry],
    index: int,
    config: SitStandAnalysisConfig,
) -> bool:
    end = index + max(config.posture_stable_frames, 1)
    if end > len(geometries):
        return False
    values = [geometry.hip_center[1] for geometry in geometries[index:end]]
    return max(values) - min(values) <= config.stable_height_epsilon


def _build_event(
    candidate: _CandidateEvent,
    geometries: list[_FrameGeometry],
    event_records: list[dict[str, Any]],
    quality: dict[str, Any],
    config: SitStandAnalysisConfig,
) -> dict[str, Any]:
    event_geometries = geometries[candidate.start_index : candidate.end_index + 1]
    start = event_geometries[0]
    end = event_geometries[-1]
    duration = max(0.0, end.timestamp - start.timestamp)
    failed_attempts = (
        _failed_attempts(event_geometries, config)
        if candidate.transition_type == "sit_to_stand"
        else 0
    )
    trunk_forward_angle = _percentile([geometry.trunk_forward_angle for geometry in event_geometries], 0.90)
    support_usage = _support_usage(event_geometries, candidate.transition_type, config)
    post_window = _post_window(geometries, candidate.end_index, config)
    if len(post_window) < config.min_post_window_frames:
        post_stand_sway = None
        stabilization_time = None
        quality["post_window_insufficient"] = True
    elif candidate.transition_type == "sit_to_stand":
        post_stand_sway = _post_stand_sway(post_window)
        stabilization_time = _stabilization_time(post_window, end.timestamp, config)
        quality["post_window_insufficient"] = False
    else:
        post_stand_sway = None
        stabilization_time = None
        quality["post_window_insufficient"] = False

    features = _features(
        event_geometries=event_geometries,
        duration=duration,
        failed_attempts=failed_attempts,
        trunk_forward_angle=trunk_forward_angle,
        post_stand_sway=post_stand_sway,
        stabilization_time=stabilization_time,
        support_usage=support_usage,
        config=config,
    )
    score, risk_factors = _score_event(features, quality, candidate.transition_type, config)
    first_record = event_records[0] if event_records else dict(start.record)

    return {
        "person_id": str(first_record.get("person_id", "unknown")),
        "track_id": first_record.get("track_id"),
        "scene_region": first_record.get("scene_region"),
        "start_time": round(start.timestamp, 4),
        "end_time": round(end.timestamp, 4),
        "transition_type": candidate.transition_type,
        "sit_stand_risk_score": score,
        "duration": round(duration, 4),
        "failed_attempts": failed_attempts,
        "trunk_forward_angle": round(trunk_forward_angle, 4),
        "post_stand_sway": None if post_stand_sway is None else round(post_stand_sway, 4),
        "support_usage": support_usage,
        "stabilization_time": None if stabilization_time is None else round(stabilization_time, 4),
        "sit_stand_features": features,
        "quality_coverage": quality,
        "risk_factors": risk_factors,
        "model_version": MODEL_VERSION,
    }


def _features(
    *,
    event_geometries: list[_FrameGeometry],
    duration: float,
    failed_attempts: int,
    trunk_forward_angle: float,
    post_stand_sway: float | None,
    stabilization_time: float | None,
    support_usage: Mapping[str, Any],
    config: SitStandAnalysisConfig,
) -> dict[str, Any]:
    start = event_geometries[0]
    end = event_geometries[-1]
    hip_vertical_displacement = start.hip_center[1] - end.hip_center[1]
    shoulder_vertical_displacement = start.shoulder_center[1] - end.shoulder_center[1]
    leg_extension_delta = end.leg_extension - start.leg_extension
    components = _score_components(
        duration=duration,
        failed_attempts=failed_attempts,
        trunk_forward_angle=trunk_forward_angle,
        post_stand_sway=post_stand_sway,
        stabilization_time=stabilization_time,
        support_suspected=bool(support_usage.get("suspected")),
        config=config,
    )
    return {
        "hip_start_y": round(start.hip_center[1], 4),
        "hip_end_y": round(end.hip_center[1], 4),
        "hip_vertical_displacement": round(hip_vertical_displacement, 4),
        "shoulder_vertical_displacement": round(shoulder_vertical_displacement, 4),
        "leg_extension_start": round(start.leg_extension, 4),
        "leg_extension_end": round(end.leg_extension, 4),
        "leg_extension_delta": round(leg_extension_delta, 4),
        "torso_length_mean": round(_mean(geometry.torso_length for geometry in event_geometries), 4),
        "trunk_forward_angle_p90": round(trunk_forward_angle, 4),
        "failed_attempts": failed_attempts,
        "duration_component": round(components["duration_component"], 4),
        "failed_attempt_component": round(components["failed_attempt_component"], 4),
        "trunk_forward_component": round(components["trunk_forward_component"], 4),
        "post_stand_sway_component": round(components["post_stand_sway_component"], 4),
        "support_usage_component": round(components["support_usage_component"], 4),
        "stabilization_time_component": round(components["stabilization_time_component"], 4),
    }


def _score_event(
    features: Mapping[str, Any],
    quality: Mapping[str, Any],
    transition_type: str,
    config: SitStandAnalysisConfig,
) -> tuple[float, list[str]]:
    components = {
        "duration_component": float(features["duration_component"]),
        "failed_attempt_component": float(features["failed_attempt_component"]),
        "trunk_forward_component": float(features["trunk_forward_component"]),
        "post_stand_sway_component": float(features["post_stand_sway_component"]),
        "support_usage_component": float(features["support_usage_component"]),
        "stabilization_time_component": float(features["stabilization_time_component"]),
    }
    score = (
        0.30 * components["duration_component"]
        + 0.20 * components["failed_attempt_component"]
        + 0.15 * components["trunk_forward_component"]
        + 0.15 * components["post_stand_sway_component"]
        + 0.10 * components["support_usage_component"]
        + 0.10 * components["stabilization_time_component"]
    )

    risk_factors: list[str] = []
    if components["duration_component"] >= 0.5:
        risk_factors.append("起身耗时较长" if transition_type == "sit_to_stand" else "坐下耗时较长")
    if transition_type == "sit_to_stand" and int(features.get("failed_attempts", 0)) > 0:
        risk_factors.append("疑似多次起身失败")
    if components["trunk_forward_component"] >= 1.0:
        risk_factors.append("起身过程中躯干明显前倾")
    if transition_type == "sit_to_stand" and components["post_stand_sway_component"] >= 0.5:
        risk_factors.append("起身后存在明显摇晃")
    if components["support_usage_component"] >= 1.0:
        risk_factors.append("疑似借助支撑")
    if transition_type == "sit_to_stand" and components["stabilization_time_component"] >= 0.5:
        risk_factors.append("站稳时间较长")
    if float(quality.get("usable_frame_ratio", 1.0)) < 0.80:
        risk_factors.append("reduced_sit_stand_quality_coverage")

    return round(_clamp(score), 4), risk_factors


def _score_components(
    *,
    duration: float,
    failed_attempts: int,
    trunk_forward_angle: float,
    post_stand_sway: float | None,
    stabilization_time: float | None,
    support_suspected: bool,
    config: SitStandAnalysisConfig,
) -> dict[str, float]:
    duration_component = _range_score(duration, config.normal_duration_sec, config.high_duration_sec)
    failed_attempt_component = _clamp(failed_attempts / 2.0)
    trunk_forward_component = _ratio_score(trunk_forward_angle, config.trunk_forward_angle_threshold_deg)
    post_stand_sway_component = (
        0.0 if post_stand_sway is None else _ratio_score(post_stand_sway, config.post_stand_sway_threshold)
    )
    stabilization_time_component = (
        0.0
        if stabilization_time is None
        else _range_score(stabilization_time, config.normal_stabilization_sec, config.high_stabilization_sec)
    )
    return {
        "duration_component": duration_component,
        "failed_attempt_component": failed_attempt_component,
        "trunk_forward_component": trunk_forward_component,
        "post_stand_sway_component": post_stand_sway_component,
        "support_usage_component": 1.0 if support_suspected else 0.0,
        "stabilization_time_component": stabilization_time_component,
    }


def _failed_attempts(event_geometries: list[_FrameGeometry], config: SitStandAnalysisConfig) -> int:
    if len(event_geometries) < 4:
        return 0
    start_y = event_geometries[0].hip_center[1]
    count = 0
    for index in range(1, len(event_geometries) - 2):
        previous_y = event_geometries[index - 1].hip_center[1]
        current_y = event_geometries[index].hip_center[1]
        next_y = event_geometries[index + 1].hip_center[1]
        drop = start_y - current_y
        returned_after = any(
            geometry.hip_center[1] >= start_y - config.failed_attempt_return_tolerance
            for geometry in event_geometries[index + 1 : -1]
        )
        if (
            previous_y > current_y
            and next_y > current_y
            and drop >= config.failed_attempt_min_displacement
            and returned_after
        ):
            count += 1
    return count


def _support_usage(
    event_geometries: list[_FrameGeometry],
    transition_type: str,
    config: SitStandAnalysisConfig,
) -> dict[str, Any]:
    if transition_type not in {"sit_to_stand", "stand_to_sit"}:
        return {"suspected": False, "evidence": []}

    evidence: list[str] = []
    for wrist_name in WRIST_KEYPOINT_NAMES:
        wrist_points: list[tuple[float, float]] = []
        body_points: list[tuple[float, float]] = []
        for geometry in event_geometries:
            wrist = geometry.wrists.get(wrist_name)
            if wrist is None:
                continue
            wrist_points.append(wrist)
            body_points.append(geometry.body_center)

        if len(wrist_points) / len(event_geometries) < config.support_min_frame_ratio:
            continue

        lateral_distances = [wrist[0] - body[0] for wrist, body in zip(wrist_points, body_points, strict=False)]
        median_lateral_distance = _median(abs(value) for value in lateral_distances)
        wrist_x_range = _axis_range_xy(wrist_points, axis="x")
        wrist_y_range = _axis_range_xy(wrist_points, axis="y")
        if (
            median_lateral_distance >= config.support_lateral_distance_threshold
            and max(wrist_x_range, wrist_y_range) <= config.support_stationary_range_threshold
        ):
            side = "left" if wrist_name == "left_wrist" else "right"
            evidence.append(f"{side}_wrist_stays_near_side_surface")

    return {"suspected": bool(evidence), "evidence": evidence}


def _post_window(
    geometries: list[_FrameGeometry],
    end_index: int,
    config: SitStandAnalysisConfig,
) -> list[_FrameGeometry]:
    end_time = geometries[end_index].timestamp
    max_time = end_time + config.post_stand_window_sec
    return [geometry for geometry in geometries[end_index:] if geometry.timestamp <= max_time]


def _post_stand_sway(post_window: list[_FrameGeometry]) -> float:
    centers = [geometry.body_center for geometry in post_window]
    lateral_range = _axis_range_xy(centers, axis="x")
    lateral_std = _std(point[0] for point in centers)
    torso_scale = max(_median(geometry.torso_length for geometry in post_window), 1e-6)
    return (lateral_range + lateral_std) / torso_scale


def _stabilization_time(
    post_window: list[_FrameGeometry],
    event_end_time: float,
    config: SitStandAnalysisConfig,
) -> float:
    required_intervals = max(config.posture_stable_frames, 1)
    speeds = _lateral_speeds(post_window)
    angle_speeds = _trunk_angle_speeds(post_window)
    for start_interval in range(0, len(speeds) - required_intervals + 1):
        speed_slice = speeds[start_interval : start_interval + required_intervals]
        angle_slice = angle_speeds[start_interval : start_interval + required_intervals]
        if (
            all(speed <= config.stable_lateral_speed_threshold for speed in speed_slice)
            and all(speed <= config.stable_trunk_angle_speed_threshold_deg for speed in angle_slice)
        ):
            stable_point = post_window[start_interval + 1]
            return max(0.0, stable_point.timestamp - event_end_time)
    return max(0.0, min(post_window[-1].timestamp - event_end_time, config.post_stand_window_sec))


def _quality_coverage(records: list[dict[str, Any]], config: SitStandAnalysisConfig) -> dict[str, Any]:
    total_frames = len(records)
    usable_records = [record for record in records if _usable_for_sit_stand(record)]
    usable_frames = len(usable_records)
    total_required_points = total_frames * len(SIT_STAND_KEYPOINT_NAMES)
    valid_required_points = 0
    interpolated_points = 0
    jump_outliers = 0
    quality_values: list[float] = []

    for record in records:
        quality_values.append(float(record.get("core_keypoint_quality", 0.0)))
        for name in SIT_STAND_KEYPOINT_NAMES:
            point = _point(record, name)
            if point is None:
                continue
            if _has_point(point):
                valid_required_points += 1
            if point.get("source") == "interpolated":
                interpolated_points += 1
            if point.get("is_jump_outlier") is True:
                jump_outliers += 1

    usable_frame_ratio = usable_frames / total_frames if total_frames > 0 else 0.0
    coverage = valid_required_points / total_required_points if total_required_points > 0 else 0.0
    interpolated_ratio = interpolated_points / total_required_points if total_required_points > 0 else 0.0
    insufficient = (
        total_frames < config.min_event_frames
        or usable_frame_ratio < config.min_usable_frame_ratio
        or coverage < config.min_sit_stand_keypoint_coverage
    )
    return {
        "frame_count": total_frames,
        "usable_frame_count": usable_frames,
        "usable_frame_ratio": round(usable_frame_ratio, 4),
        "sit_stand_keypoint_coverage": round(coverage, 4),
        "mean_core_keypoint_quality": round(_mean(quality_values), 4),
        "interpolated_point_ratio": round(interpolated_ratio, 4),
        "jump_outlier_count": jump_outliers,
        "insufficient_sit_stand_quality": insufficient,
    }


def _insufficient_quality_event(records: list[dict[str, Any]], quality: Mapping[str, Any]) -> dict[str, Any]:
    first_record = records[0]
    start_time, end_time = _record_bounds(records)
    return {
        "person_id": str(first_record.get("person_id", "unknown")),
        "track_id": first_record.get("track_id"),
        "scene_region": first_record.get("scene_region"),
        "start_time": start_time,
        "end_time": end_time,
        "transition_type": "unknown_transition",
        "sit_stand_risk_score": 0.0,
        "duration": None,
        "failed_attempts": 0,
        "trunk_forward_angle": None,
        "post_stand_sway": None,
        "support_usage": {"suspected": False, "evidence": []},
        "stabilization_time": None,
        "sit_stand_features": _empty_features(),
        "quality_coverage": dict(quality),
        "risk_factors": ["insufficient_sit_stand_quality"],
        "model_version": MODEL_VERSION,
    }


def _empty_features() -> dict[str, Any]:
    return {
        "hip_start_y": None,
        "hip_end_y": None,
        "hip_vertical_displacement": None,
        "shoulder_vertical_displacement": None,
        "leg_extension_start": None,
        "leg_extension_end": None,
        "leg_extension_delta": None,
        "torso_length_mean": None,
        "trunk_forward_angle_p90": None,
        "failed_attempts": None,
        "duration_component": None,
        "failed_attempt_component": None,
        "trunk_forward_component": None,
        "post_stand_sway_component": None,
        "support_usage_component": None,
        "stabilization_time_component": None,
    }


def _frame_geometry(record: Mapping[str, Any], position: int) -> _FrameGeometry | None:
    if not _usable_for_sit_stand(record):
        return None

    shoulder_center = _pair_center(record, "left_shoulder", "right_shoulder")
    hip_center = _pair_center(record, "left_hip", "right_hip")
    knee_center = _pair_center(record, "left_knee", "right_knee")
    ankle_center = _pair_center(record, "left_ankle", "right_ankle")
    if shoulder_center is None or hip_center is None or knee_center is None or ankle_center is None:
        return None

    body_center = (
        (0.45 * shoulder_center[0]) + (0.55 * hip_center[0]),
        (0.45 * shoulder_center[1]) + (0.55 * hip_center[1]),
    )
    torso_length = math.hypot(shoulder_center[0] - hip_center[0], shoulder_center[1] - hip_center[1])
    leg_extension = ankle_center[1] - hip_center[1]
    trunk_forward_angle = _trunk_angle(shoulder_center, hip_center)
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
        trunk_forward_angle=trunk_forward_angle,
        wrists=wrists,
    )


def _usable_for_sit_stand(record: Mapping[str, Any]) -> bool:
    window_quality = record.get("window_quality")
    if isinstance(window_quality, Mapping) and window_quality.get("usable_for_sit_stand") is False:
        return False
    return all(_has_point(_point(record, name)) for name in SIT_STAND_KEYPOINT_NAMES)


def _pair_center(record: Mapping[str, Any], left_name: str, right_name: str) -> tuple[float, float] | None:
    left = _point(record, left_name)
    right = _point(record, right_name)
    if not (_has_point(left) and _has_point(right)):
        return None
    left_xy = _point_xy(left)
    right_xy = _point_xy(right)
    return (left_xy[0] + right_xy[0]) / 2.0, (left_xy[1] + right_xy[1]) / 2.0


def _records_between(records: list[dict[str, Any]], start_time: float, end_time: float) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for position, record in enumerate(records):
        timestamp = _record_time(record, position)
        if start_time <= timestamp <= end_time:
            selected.append(record)
    return selected


def _record_bounds(records: list[Mapping[str, Any]]) -> tuple[float, float]:
    timestamps = [_record_time(record, index) for index, record in enumerate(records)]
    if not timestamps:
        return 0.0, 0.0
    return round(min(timestamps), 4), round(max(timestamps), 4)


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
    smooth_value = _optional_number(point.get(f"{axis}_smooth"))
    if smooth_value is not None:
        return smooth_value
    return _optional_number(point.get(axis))


def _trunk_angle(shoulder_center: tuple[float, float], hip_center: tuple[float, float]) -> float:
    dx = shoulder_center[0] - hip_center[0]
    dy = hip_center[1] - shoulder_center[1]
    if dy <= 1e-6:
        return 90.0
    return math.degrees(math.atan2(abs(dx), abs(dy)))


def _directional_displacement(start_y: float, current_y: float, direction: int) -> float:
    return start_y - current_y if direction < 0 else current_y - start_y


def _lateral_speeds(points: list[_FrameGeometry]) -> list[float]:
    speeds: list[float] = []
    for previous, current in zip(points, points[1:], strict=False):
        dt = current.timestamp - previous.timestamp
        if dt <= 0:
            continue
        speeds.append(abs(current.body_center[0] - previous.body_center[0]) / dt)
    return speeds


def _trunk_angle_speeds(points: list[_FrameGeometry]) -> list[float]:
    speeds: list[float] = []
    for previous, current in zip(points, points[1:], strict=False):
        dt = current.timestamp - previous.timestamp
        if dt <= 0:
            continue
        speeds.append(abs(current.trunk_forward_angle - previous.trunk_forward_angle) / dt)
    return speeds


def _axis_range_xy(points: Iterable[tuple[float, float]], *, axis: str) -> float:
    values = [point[0] if axis == "x" else point[1] for point in points]
    if not values:
        return 0.0
    return max(values) - min(values)


def _range_score(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return _clamp((value - low) / (high - low))


def _ratio_score(value: float, threshold: float) -> float:
    if threshold <= 0:
        return 0.0
    return _clamp(value / threshold)


def _percentile(values: Iterable[float], percentile: float) -> float:
    value_list = sorted(values)
    if not value_list:
        return 0.0
    if len(value_list) == 1:
        return value_list[0]
    position = _clamp(percentile) * (len(value_list) - 1)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return value_list[lower_index]
    ratio = position - lower_index
    return value_list[lower_index] + ((value_list[upper_index] - value_list[lower_index]) * ratio)


def _mean(values: Iterable[float]) -> float:
    value_list = list(values)
    if not value_list:
        return 0.0
    return sum(value_list) / len(value_list)


def _median(values: Iterable[float]) -> float:
    value_list = sorted(values)
    if not value_list:
        return 0.0
    middle = len(value_list) // 2
    if len(value_list) % 2 == 1:
        return value_list[middle]
    return (value_list[middle - 1] + value_list[middle]) / 2.0


def _std(values: Iterable[float]) -> float:
    value_list = list(values)
    if len(value_list) < 2:
        return 0.0
    mean = _mean(value_list)
    variance = sum((value - mean) ** 2 for value in value_list) / len(value_list)
    return math.sqrt(variance)


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
