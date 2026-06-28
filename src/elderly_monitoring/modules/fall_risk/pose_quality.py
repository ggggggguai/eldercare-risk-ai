"""姿态关键点质量控制与时序平滑。

本层为步态、坐站、近跌倒模块准备更稳定的关键点序列。它优先标记质量问题，
而不是直接删除帧，方便下游决定跳过、降权或解释低置信度窗口。
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from elderly_monitoring.modules.fall_risk.pose import write_jsonl


CORE_KEYPOINT_NAMES = (
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

GAIT_KEYPOINT_NAMES = (
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


@dataclass(frozen=True)
class PoseQualityConfig:
    min_keypoint_score: float = 0.30
    low_quality_threshold: float = 0.45
    low_quality_run_frames: int = 3
    max_interp_gap_frames: int = 2
    alpha: float = 0.40
    jump_threshold_norm: float = 0.18
    window_sec: float = 1.0
    window_frames: int | None = None


def process_pose_records(
    records: Iterable[Mapping[str, Any]],
    *,
    config: PoseQualityConfig | None = None,
) -> list[dict[str, Any]]:
    quality_config = config or PoseQualityConfig()
    indexed_records = [(index, dict(record)) for index, record in enumerate(records)]
    groups: dict[tuple[str, str], list[tuple[int, dict[str, Any]]]] = defaultdict(list)

    for index, record in indexed_records:
        # 质量统计和平滑必须限制在同一个人/同一条轨迹内；混合轨迹会制造
        # 人为跳变，也会让插值结果误导下游。
        groups[_group_key(record)].append((index, record))

    processed_by_index: dict[int, dict[str, Any]] = {}
    for group_items in groups.values():
        for index, record in _process_group(group_items, quality_config):
            processed_by_index[index] = record

    return [processed_by_index[index] for index, _ in indexed_records]


def run_pose_quality_jsonl(
    *,
    input_path: Path,
    output_path: Path,
    config: PoseQualityConfig | None = None,
) -> int:
    records = _read_jsonl(input_path)
    cleaned_records = process_pose_records(records, config=config)
    return write_jsonl(cleaned_records, output_path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def _process_group(
    group_items: list[tuple[int, dict[str, Any]]],
    config: PoseQualityConfig,
) -> list[tuple[int, dict[str, Any]]]:
    sorted_items = sorted(group_items, key=lambda item: _record_sort_key(item[1], item[0]))
    keypoint_names = _ordered_keypoint_names(record for _, record in sorted_items)
    working_records = [_build_working_record(index, record, keypoint_names, config) for index, record in sorted_items]

    # 顺序很重要：先补短缺失，再评估单帧质量、标记异常跳变、做时序平滑，
    # 最后生成给下游模块使用的窗口质量摘要。
    _interpolate_short_gaps(working_records, keypoint_names, config)
    _apply_frame_quality(working_records, config)
    _mark_jump_outliers(working_records, keypoint_names, config)
    _smooth_keypoints(working_records, keypoint_names, config)
    _attach_window_quality(working_records, config)

    return [(item["index"], item["record"]) for item in working_records]


def _group_key(record: Mapping[str, Any]) -> tuple[str, str]:
    person_id = str(record.get("person_id", "unknown"))
    track_id = record.get("track_id")
    return person_id, "none" if track_id is None else str(track_id)


def _record_sort_key(record: Mapping[str, Any], index: int) -> tuple[float, int, int]:
    timestamp = _number_or_default(record.get("timestamp_sec"), math.inf)
    frame_id = int(_number_or_default(record.get("frame_id"), index))
    return timestamp, frame_id, index


def _ordered_keypoint_names(records: Iterable[Mapping[str, Any]]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for record in records:
        for keypoint in record.get("keypoints", []):
            if not isinstance(keypoint, Mapping):
                continue
            name = str(keypoint.get("name", ""))
            if name and name not in seen:
                seen.add(name)
                names.append(name)
    for name in CORE_KEYPOINT_NAMES:
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


def _build_working_record(
    index: int,
    record: Mapping[str, Any],
    keypoint_names: list[str],
    config: PoseQualityConfig,
) -> dict[str, Any]:
    raw_keypoints: dict[str, Mapping[str, Any]] = {}
    for keypoint in record.get("keypoints", []):
        if isinstance(keypoint, Mapping):
            name = str(keypoint.get("name", ""))
            if name and name not in raw_keypoints:
                raw_keypoints[name] = keypoint

    payload = dict(record)
    points_by_name = {
        name: _build_quality_keypoint(name, raw_keypoints.get(name), config) for name in keypoint_names
    }
    payload["keypoints"] = [points_by_name[name] for name in keypoint_names]
    return {"index": index, "record": payload, "points_by_name": points_by_name}


def _build_quality_keypoint(
    name: str,
    raw_keypoint: Mapping[str, Any] | None,
    config: PoseQualityConfig,
) -> dict[str, Any]:
    if raw_keypoint is None:
        return _missing_keypoint(name)

    point = dict(raw_keypoint)
    point["name"] = name
    x = _optional_number(point.get("x"))
    y = _optional_number(point.get("y"))
    score = _number_or_default(point.get("score"), 0.0)
    point["score"] = round(score, 4)
    point["is_jump_outlier"] = False
    point["x_smooth"] = None
    point["y_smooth"] = None

    if x is None or y is None:
        point["x"] = None
        point["y"] = None
        point["valid"] = False
        point["source"] = "missing"
        point["quality_weight"] = 0.0
        return point

    point["x"] = round(x, 4)
    point["y"] = round(y, 4)
    if score >= config.min_keypoint_score:
        point["valid"] = True
        point["source"] = "observed"
        point["quality_weight"] = round(score, 4)
    else:
        point["valid"] = False
        point["source"] = "low_confidence"
        point["quality_weight"] = 0.0
    return point


def _missing_keypoint(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "x": None,
        "y": None,
        "score": 0.0,
        "valid": False,
        "source": "missing",
        "quality_weight": 0.0,
        "x_smooth": None,
        "y_smooth": None,
        "is_jump_outlier": False,
    }


def _interpolate_short_gaps(
    working_records: list[dict[str, Any]],
    keypoint_names: list[str],
    config: PoseQualityConfig,
) -> None:
    if config.max_interp_gap_frames <= 0:
        return

    for name in keypoint_names:
        index = 0
        while index < len(working_records):
            point = working_records[index]["points_by_name"][name]
            if point.get("valid") is True:
                index += 1
                continue

            gap_start = index
            while index < len(working_records) and working_records[index]["points_by_name"][name].get("valid") is not True:
                index += 1
            gap_end = index - 1

            previous_index = gap_start - 1
            next_index = index
            gap_length = gap_end - gap_start + 1
            if (
                previous_index < 0
                or next_index >= len(working_records)
                or gap_length > config.max_interp_gap_frames
            ):
                continue

            previous_point = working_records[previous_index]["points_by_name"][name]
            next_point = working_records[next_index]["points_by_name"][name]
            if not (_point_has_coordinates(previous_point) and _point_has_coordinates(next_point)):
                continue

            # 只填补很短的缺失片段。较长缺失通常意味着遮挡或跟踪失败，
            # 应保留下来让下游感知质量风险。
            for offset, record_index in enumerate(range(gap_start, gap_end + 1), start=1):
                ratio = offset / (gap_length + 1)
                point = working_records[record_index]["points_by_name"][name]
                point["x"] = round(_lerp(float(previous_point["x"]), float(next_point["x"]), ratio), 4)
                point["y"] = round(_lerp(float(previous_point["y"]), float(next_point["y"]), ratio), 4)
                point["score"] = round(
                    min(float(previous_point.get("score", 0.0)), float(next_point.get("score", 0.0)), config.min_keypoint_score),
                    4,
                )
                point["valid"] = True
                point["source"] = "interpolated"
                point["quality_weight"] = point["score"]


def _apply_frame_quality(working_records: list[dict[str, Any]], config: PoseQualityConfig) -> None:
    low_quality_run_length = 0
    for item in working_records:
        record = item["record"]
        points_by_name = item["points_by_name"]
        core_points = [points_by_name[name] for name in CORE_KEYPOINT_NAMES]
        valid_core_points = [point for point in core_points if point.get("valid") is True]
        valid_core_count = len(valid_core_points)
        missing_core_names = [point["name"] for point in core_points if point.get("valid") is not True]
        core_coverage = valid_core_count / len(CORE_KEYPOINT_NAMES)
        mean_core_score = (
            sum(float(point.get("quality_weight", 0.0)) for point in valid_core_points) / valid_core_count
            if valid_core_count > 0
            else 0.0
        )
        core_keypoint_quality = round((0.7 * core_coverage) + (0.3 * mean_core_score), 4)

        if core_keypoint_quality < config.low_quality_threshold:
            low_quality_run_length += 1
            quality_state = (
                "low_quality_run"
                if low_quality_run_length >= config.low_quality_run_frames
                else "low_quality"
            )
        else:
            low_quality_run_length = 0
            quality_state = "missing_core" if missing_core_names else "usable"

        record["core_keypoint_quality"] = core_keypoint_quality
        record["valid_core_count"] = valid_core_count
        record["missing_core_names"] = missing_core_names
        record["quality_state"] = quality_state
        record["low_quality_run_length"] = low_quality_run_length


def _mark_jump_outliers(
    working_records: list[dict[str, Any]],
    keypoint_names: list[str],
    config: PoseQualityConfig,
) -> None:
    for name in keypoint_names:
        previous_point: dict[str, Any] | None = None
        for item in working_records:
            point = item["points_by_name"][name]
            point["is_jump_outlier"] = False
            if point.get("valid") is not True or not _point_has_coordinates(point):
                previous_point = None
                continue

            if (
                name in CORE_KEYPOINT_NAMES
                and previous_point is not None
                and _point_has_coordinates(previous_point)
            ):
                distance = _point_distance(previous_point, point)
                if distance > config.jump_threshold_norm:
                    point["is_jump_outlier"] = True
                    point["quality_weight"] = round(float(point.get("quality_weight", 0.0)) * 0.5, 4)
            previous_point = point


def _smooth_keypoints(
    working_records: list[dict[str, Any]],
    keypoint_names: list[str],
    config: PoseQualityConfig,
) -> None:
    alpha = max(0.0, min(1.0, config.alpha))
    for name in keypoint_names:
        previous_smooth: tuple[float, float] | None = None
        for item in working_records:
            point = item["points_by_name"][name]
            if point.get("valid") is not True or not _point_has_coordinates(point):
                point["x_smooth"] = None
                point["y_smooth"] = None
                previous_smooth = None
                continue

            x = float(point["x"])
            y = float(point["y"])
            if previous_smooth is None:
                smooth_x = x
                smooth_y = y
            else:
                # 异常跳变点仍参与平滑，但降低 alpha，避免单个坏帧把骨架
                # 位置突然拉偏。
                effective_alpha = alpha * (0.25 if point.get("is_jump_outlier") is True else 1.0)
                smooth_x = (effective_alpha * x) + ((1.0 - effective_alpha) * previous_smooth[0])
                smooth_y = (effective_alpha * y) + ((1.0 - effective_alpha) * previous_smooth[1])

            point["x_smooth"] = round(smooth_x, 4)
            point["y_smooth"] = round(smooth_y, 4)
            previous_smooth = (smooth_x, smooth_y)


def _attach_window_quality(working_records: list[dict[str, Any]], config: PoseQualityConfig) -> None:
    windows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for position, item in enumerate(working_records):
        windows[_window_index(item["record"], position, config)].append(item)

    for window_index, items in windows.items():
        summary = _window_summary(window_index, items, config)
        for item in items:
            item["record"]["window_quality"] = dict(summary)


def _window_index(record: Mapping[str, Any], position: int, config: PoseQualityConfig) -> int:
    if config.window_frames is not None and config.window_frames > 0:
        return position // config.window_frames

    timestamp = _optional_number(record.get("timestamp_sec"))
    if timestamp is None or config.window_sec <= 0:
        return position // 10
    return math.floor(timestamp / config.window_sec)


def _window_summary(window_index: int, items: list[dict[str, Any]], config: PoseQualityConfig) -> dict[str, Any]:
    records = [item["record"] for item in items]
    mean_quality = sum(float(record.get("core_keypoint_quality", 0.0)) for record in records) / len(records)
    low_quality_count = sum(1 for record in records if str(record.get("quality_state")) in {"low_quality", "low_quality_run"})
    total_points = sum(len(record.get("keypoints", [])) for record in records)
    interpolated_points = sum(
        1
        for record in records
        for point in record.get("keypoints", [])
        if point.get("source") == "interpolated"
    )
    jump_outlier_count = sum(
        1
        for record in records
        for point in record.get("keypoints", [])
        if point.get("is_jump_outlier") is True
    )
    low_quality_ratio = low_quality_count / len(records)
    interpolated_ratio = interpolated_points / total_points if total_points > 0 else 0.0
    gait_coverage = _window_keypoint_coverage(items, GAIT_KEYPOINT_NAMES)
    sit_stand_coverage = _window_keypoint_coverage(items, CORE_KEYPOINT_NAMES)
    near_fall_coverage = _window_keypoint_coverage(items, CORE_KEYPOINT_NAMES)
    window_start, window_end = _window_bounds(window_index, records, config)

    # 这些标志只是下游特征提取的质量门控，不是风险判断；目的是避免
    # 低质量姿态窗口被误当成步态、坐站或近跌倒证据。
    return {
        "window_start_sec": round(window_start, 4),
        "window_end_sec": round(window_end, 4),
        "mean_core_keypoint_quality": round(mean_quality, 4),
        "low_quality_frame_ratio": round(low_quality_ratio, 4),
        "interpolated_point_ratio": round(interpolated_ratio, 4),
        "jump_outlier_count": int(jump_outlier_count),
        "usable_for_gait": gait_coverage >= 0.75 and mean_quality >= 0.55 and low_quality_ratio <= 0.40,
        "usable_for_sit_stand": sit_stand_coverage >= 0.75 and mean_quality >= 0.55 and low_quality_ratio <= 0.40,
        "usable_for_near_fall": near_fall_coverage >= 0.60 and mean_quality >= 0.50 and low_quality_ratio <= 0.50,
    }


def _window_bounds(window_index: int, records: list[Mapping[str, Any]], config: PoseQualityConfig) -> tuple[float, float]:
    timestamps = [_optional_number(record.get("timestamp_sec")) for record in records]
    numeric_timestamps = [timestamp for timestamp in timestamps if timestamp is not None]
    if config.window_frames is not None and config.window_frames > 0:
        if numeric_timestamps:
            return min(numeric_timestamps), max(numeric_timestamps)
        return float(window_index * config.window_frames), float((window_index + 1) * config.window_frames)

    if config.window_sec > 0:
        start = window_index * config.window_sec
        return start, start + config.window_sec

    if numeric_timestamps:
        return min(numeric_timestamps), max(numeric_timestamps)
    return float(window_index * 10), float((window_index + 1) * 10)


def _window_keypoint_coverage(items: list[dict[str, Any]], required_names: Iterable[str]) -> float:
    required = tuple(required_names)
    total = len(items) * len(required)
    if total == 0:
        return 0.0
    usable = 0
    for item in items:
        points_by_name = item["points_by_name"]
        for name in required:
            point = points_by_name.get(name)
            if point is not None and point.get("valid") is True and point.get("is_jump_outlier") is not True:
                usable += 1
    return usable / total


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


def _point_has_coordinates(point: Mapping[str, Any]) -> bool:
    return _optional_number(point.get("x")) is not None and _optional_number(point.get("y")) is not None


def _point_distance(previous_point: Mapping[str, Any], current_point: Mapping[str, Any]) -> float:
    previous_x = float(previous_point["x"])
    previous_y = float(previous_point["y"])
    current_x = float(current_point["x"])
    current_y = float(current_point["y"])
    return math.hypot(current_x - previous_x, current_y - previous_y)


def _lerp(start: float, end: float, ratio: float) -> float:
    return start + ((end - start) * ratio)
