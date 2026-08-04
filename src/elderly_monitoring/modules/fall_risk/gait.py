"""步态稳定性 TCN 主预测分支与可解释规则 fallback。"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

from elderly_monitoring.modules.fall_risk.pose import write_jsonl


GAIT_KEYPOINT_NAMES = (
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

GAIT_STABILITY_FEATURE_NAMES = (
    "mean_center_speed_norm_per_sec",
    "center_speed_std_norm_per_sec",
    "center_speed_cv",
    "pause_frame_ratio",
    "hip_lateral_sway",
    "path_deviation",
    "left_ankle_motion_range",
    "right_ankle_motion_range",
    "ankle_motion_asymmetry",
    "ankle_motion_mean",
    "cadence_proxy_peaks_per_sec",
    "hip_width_mean",
    "gait_speed_norm_per_sec",
    "cadence_steps_per_min_proxy",
    "stride_length_norm_proxy",
    "left_right_asymmetry",
    "center_lateral_sway",
    "trunk_tilt_mean_deg",
    "trunk_tilt_std_deg",
    "turn_instability_proxy",
)


class GaitModelPredictor(Protocol):
    model_version: str

    def predict_records(self, records: Sequence[Mapping[str, Any]]) -> float: ...


@dataclass(frozen=True)
class GaitAnalysisConfig:
    # 阈值基于归一化图像坐标，适合做可复现 baseline；
    # 后续换相机、视角或数据集时需要重新校准。
    window_sec: float = 4.0
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
    low_speed_risk_threshold_norm_per_sec: float = 0.05
    low_cadence_risk_threshold_steps_per_min: float = 40.0
    trunk_tilt_risk_threshold_deg: float = 15.0
    path_deviation_risk_threshold: float = 0.025
    turn_instability_risk_threshold: float = 0.50


def extract_gait_windows(
    records: Iterable[Mapping[str, Any]],
    *,
    config: GaitAnalysisConfig | None = None,
    model_predictor: GaitModelPredictor | None = None,
) -> list[dict[str, Any]]:
    gait_config = config or GaitAnalysisConfig()
    indexed_records = [(index, dict(record)) for index, record in enumerate(records)]
    groups: dict[tuple[str, str], list[tuple[int, dict[str, Any]]]] = defaultdict(list)

    for index, record in indexed_records:
        groups[_group_key(record)].append((index, record))

    windows: list[dict[str, Any]] = []
    for group_items in groups.values():
        sorted_items = sorted(group_items, key=lambda item: _record_sort_key(item[1], item[0]))
        windows.extend(
            _extract_group_windows(
                [record for _, record in sorted_items],
                gait_config,
                model_predictor,
            )
        )

    return sorted(windows, key=lambda item: (str(item.get("person_id", "")), int(item.get("track_id", -1)), float(item.get("start_time", 0.0))))


def run_gait_jsonl(
    *,
    input_path: Path,
    output_path: Path,
    config: GaitAnalysisConfig | None = None,
    model_predictor: GaitModelPredictor | None = None,
) -> int:
    records = _read_jsonl(input_path)
    gait_records = extract_gait_windows(
        records,
        config=config,
        model_predictor=model_predictor,
    )
    return write_jsonl(gait_records, output_path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def _extract_group_windows(
    records: list[dict[str, Any]],
    config: GaitAnalysisConfig,
    model_predictor: GaitModelPredictor | None,
) -> list[dict[str, Any]]:
    windows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for position, record in enumerate(records):
        windows[_window_index(record, position, config)].append(record)

    outputs: list[dict[str, Any]] = []
    for window_index in sorted(windows):
        window_records = windows[window_index]
        if len(window_records) < config.min_window_frames:
            continue
        outputs.append(
            _analyze_window(
                window_index,
                window_records,
                config,
                model_predictor,
            )
        )
    return outputs


def _analyze_window(
    window_index: int,
    records: list[dict[str, Any]],
    config: GaitAnalysisConfig,
    model_predictor: GaitModelPredictor | None,
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
            "model_score": None,
            "fallback_score": 0.0,
            "score_source": "unavailable",
            "fallback_reason": "insufficient_gait_quality",
            "gait_stability_features": _empty_features(),
            "quality_coverage": quality,
            "risk_factors": ["insufficient_gait_quality"],
            "model_version": "gait-risk-unavailable-v0.2",
        }

    features = _gait_features(usable_records, config)
    fallback_score, risk_factors = _score_gait(features, quality, config)
    model_score: float | None = None
    fallback_reason: str | None = None
    if model_predictor is None:
        score = fallback_score
        score_source = "rule_fallback"
        fallback_reason = "model_unavailable"
        model_version = "gait-risk-rule-v0.2"
    else:
        try:
            predicted = _optional_number(model_predictor.predict_records(records))
            if predicted is None:
                raise ValueError("model returned a non-finite score")
            model_score = round(_clamp(predicted), 4)
            score = model_score
            score_source = "tcn"
            model_version = str(model_predictor.model_version)
        except (OSError, RuntimeError, TypeError, ValueError):
            score = fallback_score
            score_source = "rule_fallback"
            fallback_reason = "model_inference_failed"
            model_version = "gait-risk-rule-v0.2"
    return {
        "person_id": str(first_record.get("person_id", "unknown")),
        "track_id": first_record.get("track_id"),
        "scene_region": first_record.get("scene_region"),
        "start_time": start_time,
        "end_time": end_time,
        "gait_risk_score": score,
        "model_score": model_score,
        "fallback_score": fallback_score,
        "score_source": score_source,
        "fallback_reason": fallback_reason,
        "gait_stability_features": features,
        "quality_coverage": quality,
        "risk_factors": risk_factors,
        "model_version": model_version,
    }


def _gait_features(records: list[dict[str, Any]], config: GaitAnalysisConfig) -> dict[str, Any]:
    centers: list[tuple[float, float, float]] = []
    left_ankles: list[tuple[float, float, float]] = []
    right_ankles: list[tuple[float, float, float]] = []
    hip_widths: list[float] = []
    trunk_tilts: list[float] = []
    shoulder_width_ratios: list[tuple[float, float]] = []

    for index, record in enumerate(records):
        timestamp = _record_time(record, index)
        left_hip = _point(record, "left_hip")
        right_hip = _point(record, "right_hip")
        left_ankle = _point(record, "left_ankle")
        right_ankle = _point(record, "right_ankle")
        left_shoulder = _point(record, "left_shoulder")
        right_shoulder = _point(record, "right_shoulder")

        # 用左右髋中心作为相机坐标系下的身体中心；脚踝运动相对身体中心计算，
        # 可以弱化整个人在画面中平移造成的影响。
        center: tuple[float, float] | None = None
        if _has_point(left_hip) and _has_point(right_hip):
            left_hip_xy = _point_xy(left_hip)
            right_hip_xy = _point_xy(right_hip)
            center = ((left_hip_xy[0] + right_hip_xy[0]) / 2.0, (left_hip_xy[1] + right_hip_xy[1]) / 2.0)
            centers.append((timestamp, center[0], center[1]))
            hip_width = math.hypot(
                right_hip_xy[0] - left_hip_xy[0],
                right_hip_xy[1] - left_hip_xy[1],
            )
            hip_widths.append(hip_width)
            if _has_point(left_shoulder) and _has_point(right_shoulder):
                left_shoulder_xy = _point_xy(left_shoulder)
                right_shoulder_xy = _point_xy(right_shoulder)
                shoulder_center = (
                    (left_shoulder_xy[0] + right_shoulder_xy[0]) / 2.0,
                    (left_shoulder_xy[1] + right_shoulder_xy[1]) / 2.0,
                )
                trunk_tilts.append(
                    math.degrees(
                        math.atan2(
                            abs(shoulder_center[0] - center[0]),
                            max(abs(center[1] - shoulder_center[1]), 1e-6),
                        )
                    )
                )
                shoulder_width = math.hypot(
                    right_shoulder_xy[0] - left_shoulder_xy[0],
                    right_shoulder_xy[1] - left_shoulder_xy[1],
                )
                if hip_width > 1e-6:
                    shoulder_width_ratios.append(
                        (timestamp, shoulder_width / hip_width)
                    )

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
    cadence_steps_per_min = (
        None if cadence_proxy is None else round(cadence_proxy * 60.0, 4)
    )
    hip_width_mean = _mean(hip_widths)
    trunk_tilt_mean = _mean(trunk_tilts)
    trunk_tilt_std = _std(trunk_tilts)
    turn_instability = _turn_instability_proxy(shoulder_width_ratios)

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
        "cadence_proxy_peaks_per_sec": (
            None if cadence_proxy is None else round(cadence_proxy, 4)
        ),
        "hip_width_mean": round(hip_width_mean, 4),
        "gait_speed_norm_per_sec": round(mean_speed, 4),
        "cadence_steps_per_min_proxy": cadence_steps_per_min,
        "stride_length_norm_proxy": round(ankle_motion_mean, 4),
        "left_right_asymmetry": round(ankle_motion_asymmetry, 4),
        "center_lateral_sway": round(hip_lateral_sway, 4),
        "trunk_tilt_mean_deg": round(trunk_tilt_mean, 4),
        "trunk_tilt_std_deg": round(trunk_tilt_std, 4),
        "turn_instability_proxy": round(turn_instability, 4),
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
    low_speed_component = _inverse_ratio_score(
        float(features["gait_speed_norm_per_sec"]),
        config.low_speed_risk_threshold_norm_per_sec,
    )
    cadence = _optional_number(features["cadence_steps_per_min_proxy"])
    cadence_component = (
        0.0
        if cadence is None
        else _inverse_ratio_score(
            cadence,
            config.low_cadence_risk_threshold_steps_per_min,
        )
    )
    trunk_component = _ratio_score(
        float(features["trunk_tilt_mean_deg"]),
        config.trunk_tilt_risk_threshold_deg,
    )
    path_component = _ratio_score(
        float(features["path_deviation"]),
        config.path_deviation_risk_threshold,
    )
    turn_component = _ratio_score(
        float(features["turn_instability_proxy"]),
        config.turn_instability_risk_threshold,
    )

    ankle_motion = float(features["ankle_motion_mean"])
    mean_speed = float(features["mean_center_speed_norm_per_sec"])
    shuffling_component = 0.0
    if mean_speed > config.pause_speed_threshold_norm_per_sec and ankle_motion < config.shuffling_motion_threshold:
        shuffling_component = 1.0 - min(1.0, ankle_motion / max(config.shuffling_motion_threshold, 1e-6))

    quality_penalty = 1.0 - float(quality["usable_frame_ratio"])
    # 规则分只用于解释、对照和模型不可用时的 fallback。
    score = (
        0.18 * speed_component
        + 0.08 * low_speed_component
        + 0.14 * asymmetry_component
        + 0.11 * sway_component
        + 0.10 * pause_component
        + 0.10 * shuffling_component
        + 0.08 * cadence_component
        + 0.08 * trunk_component
        + 0.05 * path_component
        + 0.06 * turn_component
        + 0.02 * quality_penalty
    )

    risk_factors: list[str] = []
    if speed_component >= 0.5:
        risk_factors.append("center_speed_instability")
    if low_speed_component >= 0.5:
        risk_factors.append("gait_speed_reduced")
    if cadence_component >= 0.5:
        risk_factors.append("cadence_reduced")
    if asymmetry_component >= 0.5:
        risk_factors.append("lower_limb_asymmetry")
    if sway_component >= 0.5:
        risk_factors.append("hip_lateral_sway")
    if pause_component >= 0.5:
        risk_factors.append("pause_or_hesitation")
    if shuffling_component >= 0.5:
        risk_factors.append("shuffling_or_dragging")
    if trunk_component >= 0.5:
        risk_factors.append("trunk_tilt_increased")
    if path_component >= 0.5:
        risk_factors.append("path_deviation")
    if turn_component >= 0.5:
        risk_factors.append("turn_instability")
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


def _cadence_proxy(
    left_points: list[tuple[float, float, float]],
    right_points: list[tuple[float, float, float]],
) -> float | None:
    peaks = _count_motion_peaks(left_points) + _count_motion_peaks(right_points)
    timestamps = [point[0] for point in left_points + right_points]
    if len(timestamps) < 2:
        return None
    duration = max(timestamps) - min(timestamps)
    if duration < 1.5:
        return None
    return min(4.0, peaks / duration)


def _turn_instability_proxy(observations: list[tuple[float, float]]) -> float:
    if len(observations) < 4:
        return 0.0
    values = [value for _, value in observations]
    if max(values) - min(values) < 0.08:
        return 0.0
    deltas = [
        (current[1] - previous[1]) / (current[0] - previous[0])
        for previous, current in zip(observations, observations[1:], strict=False)
        if current[0] > previous[0]
    ]
    if len(deltas) < 2:
        return 0.0
    direction_changes = _direction_change_ratio(deltas)
    magnitude_variation = _std(abs(value) for value in deltas) / max(
        _mean(abs(value) for value in deltas),
        1e-6,
    )
    return _clamp((0.6 * direction_changes) + (0.4 * magnitude_variation))


def _direction_change_ratio(values: Sequence[float]) -> float:
    signs = [1 if value > 1e-6 else -1 if value < -1e-6 else 0 for value in values]
    non_zero = [sign for sign in signs if sign]
    if len(non_zero) < 2:
        return 0.0
    changes = sum(
        current != previous
        for previous, current in zip(non_zero, non_zero[1:], strict=False)
    )
    return changes / (len(non_zero) - 1)


def _count_motion_peaks(points: list[tuple[float, float, float]]) -> int:
    if len(points) < 5:
        return 0
    smoothed = [
        _mean(point[1] for point in points[max(0, index - 1) : index + 2])
        for index in range(len(points))
    ]
    motion_range = max(smoothed) - min(smoothed)
    min_prominence = max(0.002, motion_range * 0.10)
    peak_times: list[float] = []
    for index in range(1, len(points) - 1):
        value = smoothed[index]
        if value <= smoothed[index - 1] or value < smoothed[index + 1]:
            continue
        local_floor = min(smoothed[index - 1], smoothed[index + 1])
        if value - local_floor < min_prominence:
            continue
        timestamp = points[index][0]
        if peak_times and timestamp - peak_times[-1] < 0.40:
            if value > smoothed[_nearest_time_index(points, peak_times[-1])]:
                peak_times[-1] = timestamp
            continue
        peak_times.append(timestamp)
    return len(peak_times)


def _nearest_time_index(
    points: Sequence[tuple[float, float, float]],
    timestamp: float,
) -> int:
    return min(range(len(points)), key=lambda index: abs(points[index][0] - timestamp))


def _ratio_score(value: float, threshold: float) -> float:
    if threshold <= 0:
        return 0.0
    return _clamp(value / threshold)


def _inverse_ratio_score(value: float, threshold: float) -> float:
    if threshold <= 0:
        return 0.0
    return _clamp((threshold - value) / threshold)


def _empty_features() -> dict[str, Any]:
    return {name: None for name in GAIT_STABILITY_FEATURE_NAMES}


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
