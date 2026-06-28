"""基于清洗后姿态关键点的步态稳定性规则 baseline。

本模块消费平滑后的关键点，并按窗口输出步态特征 JSONL。当前分数是
可解释工程特征，不是经过医学标定的诊断结论。
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from elderly_monitoring.modules.fall_risk.pose import write_jsonl


GAIT_KEYPOINT_NAMES = (
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


@dataclass(frozen=True)
class GaitAnalysisConfig:
    # 阈值基于归一化图像坐标，适合做可复现 baseline；
    # 后续换相机、视角或数据集时需要重新校准。
    window_sec: float = 2.0
    window_frames: int | None = None
    min_window_frames: int = 5
    min_usable_frame_ratio: float = 0.60
    min_gait_keypoint_coverage: float = 0.70
    pause_speed_threshold_norm_per_sec: float = 0.03
    center_speed_cv_risk_threshold: float = 0.60
    ankle_asymmetry_risk_threshold: float = 0.45
    hip_sway_risk_threshold: float = 0.035
    pause_ratio_risk_threshold: float = 0.25
    shuffling_motion_threshold: float = 0.018


def extract_gait_windows(
    records: Iterable[Mapping[str, Any]],
    *,
    config: GaitAnalysisConfig | None = None,
) -> list[dict[str, Any]]:
    gait_config = config or GaitAnalysisConfig()
    indexed_records = [(index, dict(record)) for index, record in enumerate(records)]
    groups: dict[tuple[str, str], list[tuple[int, dict[str, Any]]]] = defaultdict(list)

    for index, record in indexed_records:
        groups[_group_key(record)].append((index, record))

    windows: list[dict[str, Any]] = []
    for group_items in groups.values():
        sorted_items = sorted(group_items, key=lambda item: _record_sort_key(item[1], item[0]))
        windows.extend(_extract_group_windows([record for _, record in sorted_items], gait_config))

    return sorted(windows, key=lambda item: (str(item.get("person_id", "")), int(item.get("track_id", -1)), float(item.get("start_time", 0.0))))


def run_gait_jsonl(
    *,
    input_path: Path,
    output_path: Path,
    config: GaitAnalysisConfig | None = None,
) -> int:
    records = _read_jsonl(input_path)
    gait_records = extract_gait_windows(records, config=config)
    return write_jsonl(gait_records, output_path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def _extract_group_windows(records: list[dict[str, Any]], config: GaitAnalysisConfig) -> list[dict[str, Any]]:
    windows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for position, record in enumerate(records):
        windows[_window_index(record, position, config)].append(record)

    outputs: list[dict[str, Any]] = []
    for window_index in sorted(windows):
        window_records = windows[window_index]
        if len(window_records) < config.min_window_frames:
            continue
        outputs.append(_analyze_window(window_index, window_records, config))
    return outputs


def _analyze_window(
    window_index: int,
    records: list[dict[str, Any]],
    config: GaitAnalysisConfig,
) -> dict[str, Any]:
    usable_records = [record for record in records if _usable_for_gait(record)]
    quality = _quality_coverage(records, usable_records, config)
    start_time, end_time = _window_bounds(window_index, records, config)
    first_record = records[0]

    if quality["insufficient_gait_quality"]:
        # 低质量窗口标为不可用，而不是高风险，避免遮挡或脚踝缺失被误判成不稳。
        return {
            "person_id": str(first_record.get("person_id", "unknown")),
            "track_id": first_record.get("track_id"),
            "scene_region": first_record.get("scene_region"),
            "start_time": start_time,
            "end_time": end_time,
            "gait_risk_score": 0.0,
            "gait_stability_features": _empty_features(),
            "quality_coverage": quality,
            "risk_factors": ["insufficient_gait_quality"],
            "model_version": "gait-risk-rule-v0.1",
        }

    features = _gait_features(usable_records, config)
    score, risk_factors = _score_gait(features, quality, config)
    return {
        "person_id": str(first_record.get("person_id", "unknown")),
        "track_id": first_record.get("track_id"),
        "scene_region": first_record.get("scene_region"),
        "start_time": start_time,
        "end_time": end_time,
        "gait_risk_score": score,
        "gait_stability_features": features,
        "quality_coverage": quality,
        "risk_factors": risk_factors,
        "model_version": "gait-risk-rule-v0.1",
    }


def _gait_features(records: list[dict[str, Any]], config: GaitAnalysisConfig) -> dict[str, Any]:
    centers: list[tuple[float, float, float]] = []
    left_ankles: list[tuple[float, float, float]] = []
    right_ankles: list[tuple[float, float, float]] = []
    hip_widths: list[float] = []

    for index, record in enumerate(records):
        timestamp = _record_time(record, index)
        left_hip = _point(record, "left_hip")
        right_hip = _point(record, "right_hip")
        left_ankle = _point(record, "left_ankle")
        right_ankle = _point(record, "right_ankle")

        # 用左右髋中心作为相机坐标系下的身体中心；脚踝运动相对身体中心计算，
        # 可以弱化整个人在画面中平移造成的影响。
        center: tuple[float, float] | None = None
        if _has_point(left_hip) and _has_point(right_hip):
            left_hip_xy = _point_xy(left_hip)
            right_hip_xy = _point_xy(right_hip)
            center = ((left_hip_xy[0] + right_hip_xy[0]) / 2.0, (left_hip_xy[1] + right_hip_xy[1]) / 2.0)
            centers.append((timestamp, center[0], center[1]))
            hip_widths.append(abs(right_hip_xy[0] - left_hip_xy[0]))

        if center is not None and _has_point(left_ankle):
            left_xy = _point_xy(left_ankle)
            left_ankles.append((timestamp, left_xy[0] - center[0], left_xy[1] - center[1]))
        if center is not None and _has_point(right_ankle):
            right_xy = _point_xy(right_ankle)
            right_ankles.append((timestamp, right_xy[0] - center[0], right_xy[1] - center[1]))

    center_speeds = _speeds(centers)
    mean_speed = _mean(center_speeds)
    speed_std = _std(center_speeds)
    center_speed_cv = speed_std / mean_speed if mean_speed > 1e-6 else 0.0
    pause_frame_ratio = _pause_ratio(center_speeds, config.pause_speed_threshold_norm_per_sec)
    hip_lateral_sway = _path_max_deviation(centers)
    path_deviation = _path_deviation(centers)
    left_ankle_motion = _axis_range(left_ankles, axis="x")
    right_ankle_motion = _axis_range(right_ankles, axis="x")
    ankle_motion_asymmetry = _asymmetry(left_ankle_motion, right_ankle_motion)
    ankle_motion_mean = _mean([left_ankle_motion, right_ankle_motion])
    cadence_proxy = _cadence_proxy(left_ankles, right_ankles)
    hip_width_mean = _mean(hip_widths)

    return {
        "mean_center_speed_norm_per_sec": round(mean_speed, 4),
        "center_speed_std_norm_per_sec": round(speed_std, 4),
        "center_speed_cv": round(center_speed_cv, 4),
        "pause_frame_ratio": round(pause_frame_ratio, 4),
        "hip_lateral_sway": round(hip_lateral_sway, 4),
        "path_deviation": round(path_deviation, 4),
        "left_ankle_motion_range": round(left_ankle_motion, 4),
        "right_ankle_motion_range": round(right_ankle_motion, 4),
        "ankle_motion_asymmetry": round(ankle_motion_asymmetry, 4),
        "ankle_motion_mean": round(ankle_motion_mean, 4),
        "cadence_proxy_peaks_per_sec": round(cadence_proxy, 4),
        "hip_width_mean": round(hip_width_mean, 4),
    }


def _score_gait(
    features: Mapping[str, Any],
    quality: Mapping[str, Any],
    config: GaitAnalysisConfig,
) -> tuple[float, list[str]]:
    speed_component = _ratio_score(float(features["center_speed_cv"]), config.center_speed_cv_risk_threshold)
    asymmetry_component = _ratio_score(float(features["ankle_motion_asymmetry"]), config.ankle_asymmetry_risk_threshold)
    sway_component = _ratio_score(float(features["hip_lateral_sway"]), config.hip_sway_risk_threshold)
    pause_component = _ratio_score(float(features["pause_frame_ratio"]), config.pause_ratio_risk_threshold)

    ankle_motion = float(features["ankle_motion_mean"])
    mean_speed = float(features["mean_center_speed_norm_per_sec"])
    shuffling_component = 0.0
    if mean_speed > config.pause_speed_threshold_norm_per_sec and ankle_motion < config.shuffling_motion_threshold:
        shuffling_component = 1.0 - min(1.0, ankle_motion / max(config.shuffling_motion_threshold, 1e-6))

    quality_penalty = 1.0 - float(quality["usable_frame_ratio"])
    # 下面的权重是透明规则 baseline。拿到有标签步态数据后，应替换或校准成
    # 轻量学习模型/统计模型。
    score = (
        0.30 * speed_component
        + 0.25 * asymmetry_component
        + 0.20 * sway_component
        + 0.15 * pause_component
        + 0.10 * shuffling_component
        + 0.10 * quality_penalty
    )

    risk_factors: list[str] = []
    if speed_component >= 0.5:
        risk_factors.append("center_speed_instability")
    if asymmetry_component >= 0.5:
        risk_factors.append("lower_limb_asymmetry")
    if sway_component >= 0.5:
        risk_factors.append("hip_lateral_sway")
    if pause_component >= 0.5:
        risk_factors.append("pause_or_hesitation")
    if shuffling_component >= 0.5:
        risk_factors.append("shuffling_or_dragging")
    if quality_penalty >= 0.25:
        risk_factors.append("reduced_gait_quality_coverage")

    return round(_clamp(score), 4), risk_factors


def _quality_coverage(
    records: list[dict[str, Any]],
    usable_records: list[dict[str, Any]],
    config: GaitAnalysisConfig,
) -> dict[str, Any]:
    total_frames = len(records)
    usable_frames = len(usable_records)
    total_required_points = total_frames * len(GAIT_KEYPOINT_NAMES)
    valid_required_points = 0
    interpolated_points = 0
    jump_outliers = 0
    quality_values: list[float] = []

    for record in records:
        quality_values.append(float(record.get("core_keypoint_quality", 0.0)))
        for name in GAIT_KEYPOINT_NAMES:
            point = _point(record, name)
            if point is None:
                continue
            if point.get("valid") is True and point.get("is_jump_outlier") is not True and _has_point(point):
                valid_required_points += 1
            if point.get("source") == "interpolated":
                interpolated_points += 1
            if point.get("is_jump_outlier") is True:
                jump_outliers += 1

    usable_frame_ratio = usable_frames / total_frames if total_frames > 0 else 0.0
    gait_keypoint_coverage = valid_required_points / total_required_points if total_required_points > 0 else 0.0
    interpolated_point_ratio = interpolated_points / total_required_points if total_required_points > 0 else 0.0
    insufficient = (
        total_frames < config.min_window_frames
        or usable_frame_ratio < config.min_usable_frame_ratio
        or gait_keypoint_coverage < config.min_gait_keypoint_coverage
    )
    return {
        "frame_count": total_frames,
        "usable_frame_count": usable_frames,
        "usable_frame_ratio": round(usable_frame_ratio, 4),
        "gait_keypoint_coverage": round(gait_keypoint_coverage, 4),
        "mean_core_keypoint_quality": round(_mean(quality_values), 4),
        "interpolated_point_ratio": round(interpolated_point_ratio, 4),
        "jump_outlier_count": jump_outliers,
        "insufficient_gait_quality": insufficient,
    }


def _usable_for_gait(record: Mapping[str, Any]) -> bool:
    window_quality = record.get("window_quality")
    if isinstance(window_quality, Mapping) and window_quality.get("usable_for_gait") is False:
        return False
    return all(_has_point(_point(record, name)) for name in GAIT_KEYPOINT_NAMES)


def _window_index(record: Mapping[str, Any], position: int, config: GaitAnalysisConfig) -> int:
    if config.window_frames is not None and config.window_frames > 0:
        return position // config.window_frames

    timestamp = _optional_number(record.get("timestamp_sec"))
    if timestamp is None or config.window_sec <= 0:
        return position // max(config.min_window_frames, 1)
    return math.floor(timestamp / config.window_sec)


def _window_bounds(window_index: int, records: list[Mapping[str, Any]], config: GaitAnalysisConfig) -> tuple[float, float]:
    timestamps = [_optional_number(record.get("timestamp_sec")) for record in records]
    numeric_timestamps = [timestamp for timestamp in timestamps if timestamp is not None]
    if numeric_timestamps:
        return round(min(numeric_timestamps), 4), round(max(numeric_timestamps), 4)
    if config.window_frames is not None and config.window_frames > 0:
        return float(window_index * config.window_frames), float((window_index + 1) * config.window_frames)
    return round(window_index * config.window_sec, 4), round((window_index + 1) * config.window_sec, 4)


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


def _point_xy(point: Mapping[str, Any]) -> tuple[float, float]:
    return float(point["x_smooth"]), float(point["y_smooth"])


def _speeds(points: list[tuple[float, float, float]]) -> list[float]:
    speeds: list[float] = []
    for previous, current in zip(points, points[1:], strict=False):
        dt = current[0] - previous[0]
        if dt <= 0:
            continue
        distance = math.hypot(current[1] - previous[1], current[2] - previous[2])
        speeds.append(distance / dt)
    return speeds


def _pause_ratio(speeds: list[float], threshold: float) -> float:
    if not speeds:
        return 0.0
    pause_count = sum(1 for speed in speeds if speed <= threshold)
    return pause_count / len(speeds)


def _axis_range(points: list[tuple[float, float, float]], *, axis: str) -> float:
    if not points:
        return 0.0
    values = [point[1] if axis == "x" else point[2] for point in points]
    return max(values) - min(values)


def _path_deviation(points: list[tuple[float, float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    start = points[0]
    end = points[-1]
    line_dx = end[1] - start[1]
    line_dy = end[2] - start[2]
    line_length = math.hypot(line_dx, line_dy)
    if line_length <= 1e-6:
        return 0.0

    distances = []
    for _, x, y in points[1:-1]:
        numerator = abs((line_dy * x) - (line_dx * y) + (end[1] * start[2]) - (end[2] * start[1]))
        distances.append(numerator / line_length)
    return _mean(distances)


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


def _asymmetry(left_value: float, right_value: float) -> float:
    denominator = max(left_value, right_value, 1e-6)
    return abs(left_value - right_value) / denominator


def _cadence_proxy(left_points: list[tuple[float, float, float]], right_points: list[tuple[float, float, float]]) -> float:
    peaks = _count_direction_changes(left_points) + _count_direction_changes(right_points)
    timestamps = [point[0] for point in left_points + right_points]
    if len(timestamps) < 2:
        return 0.0
    duration = max(timestamps) - min(timestamps)
    if duration <= 0:
        return 0.0
    return peaks / duration


def _count_direction_changes(points: list[tuple[float, float, float]]) -> int:
    if len(points) < 3:
        return 0
    signs: list[int] = []
    for previous, current in zip(points, points[1:], strict=False):
        delta = current[1] - previous[1]
        if abs(delta) <= 1e-6:
            signs.append(0)
        else:
            signs.append(1 if delta > 0 else -1)

    changes = 0
    previous_sign = 0
    for sign in signs:
        if sign == 0:
            continue
        if previous_sign != 0 and sign != previous_sign:
            changes += 1
        previous_sign = sign
    return changes


def _ratio_score(value: float, threshold: float) -> float:
    if threshold <= 0:
        return 0.0
    return _clamp(value / threshold)


def _empty_features() -> dict[str, Any]:
    return {
        "mean_center_speed_norm_per_sec": None,
        "center_speed_std_norm_per_sec": None,
        "center_speed_cv": None,
        "pause_frame_ratio": None,
        "hip_lateral_sway": None,
        "path_deviation": None,
        "left_ankle_motion_range": None,
        "right_ankle_motion_range": None,
        "ankle_motion_asymmetry": None,
        "ankle_motion_mean": None,
        "cadence_proxy_peaks_per_sec": None,
        "hip_width_mean": None,
    }


def _mean(values: Iterable[float]) -> float:
    value_list = list(values)
    if not value_list:
        return 0.0
    return sum(value_list) / len(value_list)


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
