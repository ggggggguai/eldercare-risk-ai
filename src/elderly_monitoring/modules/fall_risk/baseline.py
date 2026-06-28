"""个体化行为基线建模的规则/统计 baseline。

本模块消费步态、坐站、近跌倒、活动节律和场景聚合 JSONL，按
person_id 建立个人历史统计，并输出当前窗口相对个人历史的偏离分。
输出是供后续风险融合层消费的工程特征，不是最终跌倒风险等级或
医疗诊断结论。
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from elderly_monitoring.modules.fall_risk.pose import write_jsonl


MODEL_VERSION = "fall-baseline-rule-v0.1"


@dataclass(frozen=True)
class BaselineModelConfig:
    # 初始基线按 3-7 天可用，稳定基线按 7-14 天滚动统计。
    min_history_days: int = 3
    stable_history_days: int = 7
    max_history_days: int = 14
    min_history_records: int = 10
    aggregation_period: str = "day"
    min_quality_score: float = 0.60
    insufficient_history_score_cap: float = 0.20
    reduced_quality_score_cap: float = 0.25
    near_fall_score_count_threshold: float = 0.25
    scene_shift_probability_threshold: float = 0.20


def build_personal_baselines(
    records: Iterable[Mapping[str, Any]],
    *,
    config: BaselineModelConfig | None = None,
) -> dict[str, dict[str, Any]]:
    """按 person_id 从历史结构化结果中建立个人滚动统计摘要。"""
    baseline_config = config or BaselineModelConfig()
    record_list = [dict(record) for record in records]
    period_features = _aggregate_records(record_list, baseline_config)

    by_person: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for features in period_features:
        by_person[str(features["person_id"])].append(features)

    baselines: dict[str, dict[str, Any]] = {}
    for person_id, features in by_person.items():
        rolling_features = _select_rolling_features(features, baseline_config)
        baselines[person_id] = _build_reference(person_id, rolling_features, baseline_config)
    return baselines


def score_baseline_deviation(
    current_records: Iterable[Mapping[str, Any]],
    baselines: Mapping[str, Mapping[str, Any]],
    *,
    config: BaselineModelConfig | None = None,
) -> list[dict[str, Any]]:
    """计算当前观测窗口相对个人历史基线的偏离分和解释因子。"""
    baseline_config = config or BaselineModelConfig()
    current_features = _aggregate_records([dict(record) for record in current_records], baseline_config)
    outputs = [
        _score_current_features(features, baselines.get(str(features["person_id"])), baseline_config)
        for features in current_features
    ]
    return sorted(
        outputs,
        key=lambda item: (
            str(item.get("person_id", "")),
            _sort_time_value(item.get("start_time")),
            _sort_time_value(item.get("end_time")),
        ),
    )


def run_baseline_jsonl(
    *,
    baseline_input_path: Path,
    current_input_path: Path,
    output_path: Path,
    config: BaselineModelConfig | None = None,
) -> int:
    """从历史 JSONL 和当前 JSONL 生成个体化行为基线偏离 JSONL。"""
    baseline_config = config or BaselineModelConfig()
    history_records = _read_jsonl(baseline_input_path)
    current_records = _read_jsonl(current_input_path)
    baselines = build_personal_baselines(history_records, config=baseline_config)
    outputs = score_baseline_deviation(current_records, baselines, config=baseline_config)
    return write_jsonl(outputs, output_path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def _aggregate_records(records: list[dict[str, Any]], config: BaselineModelConfig) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, record in enumerate(records):
        person_id = str(record.get("person_id", "unknown"))
        groups[(person_id, _period_key(record, index, config))].append((index, record))

    features = [
        _aggregate_group(person_id, period_key, items, config)
        for (person_id, period_key), items in groups.items()
    ]
    return sorted(
        features,
        key=lambda item: (
            str(item.get("person_id", "")),
            _sort_time_value(item.get("start_time")),
            str(item.get("period_key", "")),
        ),
    )


def _aggregate_group(
    person_id: str,
    period_key: str,
    items: list[tuple[int, dict[str, Any]]],
    config: BaselineModelConfig,
) -> dict[str, Any]:
    sorted_items = sorted(items, key=lambda item: _record_sort_key(item[1], item[0]))
    records = [record for _, record in sorted_items]

    gait_speeds: list[float] = []
    center_speed_cvs: list[float] = []
    hip_lateral_sways: list[float] = []
    gait_risk_scores: list[float] = []
    sit_stand_durations: list[float] = []
    failed_attempts = 0.0
    stabilization_times: list[float] = []
    sit_stand_risk_scores: list[float] = []
    near_fall_event_scores: list[float] = []
    near_fall_frequency = 0.0
    nighttime_activity_count = 0.0
    activity_values: list[float] = []
    quality_values: list[float] = []
    scene_counts: Counter[str] = Counter()
    track_ids: set[str] = set()

    for index, record in sorted_items:
        track_id = record.get("track_id")
        if track_id is not None:
            track_ids.add(str(track_id))

        scene_region = record.get("scene_region")
        if scene_region:
            scene_counts[str(scene_region)] += 1

        quality_values.append(_quality_score(record))

        gait_features = record.get("gait_stability_features")
        if isinstance(gait_features, Mapping):
            _append_number(gait_speeds, gait_features.get("mean_center_speed_norm_per_sec"))
            _append_number(center_speed_cvs, gait_features.get("center_speed_cv"))
            _append_number(hip_lateral_sways, gait_features.get("hip_lateral_sway"))
        _append_number(gait_risk_scores, record.get("gait_risk_score"))

        _append_number(sit_stand_durations, record.get("duration"))
        failed_attempts += _number_or_default(record.get("failed_attempts"), 0.0)
        _append_number(stabilization_times, record.get("stabilization_time"))
        _append_number(sit_stand_risk_scores, record.get("sit_stand_risk_score"))

        _append_number(near_fall_event_scores, record.get("near_fall_event_score"))
        near_fall_frequency += _near_fall_count(record, config)
        nighttime_activity_count += _nighttime_activity_count(record, index)
        activity_value = _activity_value(record)
        if activity_value is not None:
            activity_values.append(activity_value)

    start_time, end_time = _record_bounds(sorted_items)
    quality_mean = _mean(quality_values)
    activity_volume = sum(activity_values) if activity_values else None
    dominant_scene = scene_counts.most_common(1)[0][0] if scene_counts else None
    turn_instability_proxy = _turn_instability_proxy(center_speed_cvs, hip_lateral_sways)

    return {
        "person_id": person_id,
        "period_key": period_key,
        "day_key": _day_key(records[0], sorted_items[0][0]) if records else period_key,
        "aggregation_period": config.aggregation_period,
        "start_time": start_time,
        "timestamp": start_time,
        "end_time": end_time,
        "record_count": len(records),
        "track_ids": sorted(track_ids),
        "mean_gait_speed": _rounded_or_none(_mean_or_none(gait_speeds)),
        "gait_speed_observation_count": len(gait_speeds),
        "center_speed_cv": _rounded_or_none(_mean_or_none(center_speed_cvs)),
        "hip_lateral_sway": _rounded_or_none(_mean_or_none(hip_lateral_sways)),
        "turn_instability_proxy": _rounded_or_none(turn_instability_proxy),
        "mean_gait_risk_score": _rounded_or_none(_mean_or_none(gait_risk_scores)),
        "mean_sit_stand_duration": _rounded_or_none(_mean_or_none(sit_stand_durations)),
        "sit_stand_observation_count": len(sit_stand_durations),
        "failed_attempts": round(failed_attempts, 4),
        "mean_stabilization_time": _rounded_or_none(_mean_or_none(stabilization_times)),
        "mean_sit_stand_risk_score": _rounded_or_none(_mean_or_none(sit_stand_risk_scores)),
        "near_fall_frequency": round(near_fall_frequency, 4),
        "mean_near_fall_event_score": _rounded_or_none(_mean_or_none(near_fall_event_scores)),
        "nighttime_activity_count": round(nighttime_activity_count, 4),
        "activity_volume": _rounded_or_none(activity_volume),
        "dominant_scene_region": dominant_scene,
        "scene_region_counts": dict(sorted(scene_counts.items())),
        "scene_region_distribution": _distribution(scene_counts),
        "quality_mean": round(quality_mean, 4),
        "quality_min": round(min(quality_values), 4) if quality_values else 0.0,
        "low_quality_record_count": sum(1 for value in quality_values if value < config.min_quality_score),
    }


def _build_reference(
    person_id: str,
    features: list[dict[str, Any]],
    config: BaselineModelConfig,
) -> dict[str, Any]:
    day_count = len({str(feature.get("day_key")) for feature in features})
    record_count = sum(int(feature.get("record_count", 0)) for feature in features)
    quality_values = [float(feature.get("quality_mean", 0.0)) for feature in features]
    history_quality_mean = _mean(quality_values)
    reduced_quality = bool(quality_values) and history_quality_mean < config.min_quality_score
    insufficient = day_count < config.min_history_days or record_count < config.min_history_records

    metric_references = {
        metric: _metric_stats([feature.get(metric) for feature in features], config)
        for metric in _REFERENCE_METRICS
    }
    scene_counts: Counter[str] = Counter()
    track_ids: set[str] = set()
    for feature in features:
        scene_counts.update({str(key): int(value) for key, value in feature.get("scene_region_counts", {}).items()})
        track_ids.update(str(track_id) for track_id in feature.get("track_ids", []))

    start_time = _min_output_time(feature.get("start_time") for feature in features)
    end_time = _max_output_time(feature.get("end_time") for feature in features)
    return {
        "person_id": person_id,
        "history_record_count": record_count,
        "history_period_count": len(features),
        "history_day_count": day_count,
        "window_start": start_time,
        "window_end": end_time,
        "track_ids": sorted(track_ids),
        "metric_references": metric_references,
        "scene_region_distribution": _distribution(scene_counts),
        "dominant_scene_region": scene_counts.most_common(1)[0][0] if scene_counts else None,
        "baseline_quality": {
            "history_record_count": record_count,
            "history_period_count": len(features),
            "history_day_count": day_count,
            "initial_baseline_ready": day_count >= config.min_history_days,
            "stable_baseline_ready": day_count >= config.stable_history_days,
            "insufficient_baseline_history": insufficient,
            "history_quality_mean": round(history_quality_mean, 4),
            "reduced_baseline_quality": reduced_quality,
        },
        "model_version": MODEL_VERSION,
    }


_REFERENCE_METRICS = (
    "mean_gait_speed",
    "center_speed_cv",
    "hip_lateral_sway",
    "turn_instability_proxy",
    "mean_sit_stand_duration",
    "failed_attempts",
    "mean_stabilization_time",
    "near_fall_frequency",
    "nighttime_activity_count",
    "activity_volume",
)


def _score_current_features(
    features: Mapping[str, Any],
    reference: Mapping[str, Any] | None,
    config: BaselineModelConfig,
) -> dict[str, Any]:
    empty_reference = reference is None
    reference_data = reference or _empty_reference(str(features.get("person_id", "unknown")))
    reference_quality = reference_data.get("baseline_quality", {})
    history_insufficient = bool(reference_quality.get("insufficient_baseline_history", True)) or empty_reference
    current_quality_mean = float(features.get("quality_mean", 0.0))
    reduced_quality = (
        bool(reference_quality.get("reduced_baseline_quality", False))
        or current_quality_mean < config.min_quality_score
    )

    baseline_quality = {
        "history_record_count": int(reference_quality.get("history_record_count", 0)),
        "history_period_count": int(reference_quality.get("history_period_count", 0)),
        "history_day_count": int(reference_quality.get("history_day_count", 0)),
        "current_record_count": int(features.get("record_count", 0)),
        "current_quality_mean": round(current_quality_mean, 4),
        "history_quality_mean": round(float(reference_quality.get("history_quality_mean", 0.0)), 4),
        "initial_baseline_ready": bool(reference_quality.get("initial_baseline_ready", False)),
        "stable_baseline_ready": bool(reference_quality.get("stable_baseline_ready", False)),
        "insufficient_baseline_history": history_insufficient,
        "reduced_baseline_quality": reduced_quality,
    }
    baseline_quality["baseline_confidence"] = _baseline_confidence(baseline_quality, config)

    deviation_factors: list[str] = []
    factor_details: list[dict[str, Any]] = []
    if history_insufficient:
        deviation_factors.append("insufficient_baseline_history")
    if reduced_quality:
        deviation_factors.append("reduced_baseline_quality")

    raw_score = 0.0
    metric_references = reference_data.get("metric_references", {})
    if not history_insufficient and not reduced_quality:
        raw_score, factor_details = _deviation_score_components(features, metric_references, reference_data, config)
        deviation_factors.extend(detail["factor"] for detail in factor_details)

    score = _clamp(raw_score)
    if history_insufficient:
        score = min(score, config.insufficient_history_score_cap)
    if reduced_quality:
        score = min(score, config.reduced_quality_score_cap)

    return {
        "person_id": str(features.get("person_id", "unknown")),
        "start_time": features.get("start_time"),
        "timestamp": features.get("timestamp") or features.get("start_time"),
        "end_time": features.get("end_time"),
        "baseline_deviation_score": round(score, 4),
        "baseline_features": _public_current_features(features),
        "baseline_reference": _public_reference(reference_data),
        "deviation_factors": _dedupe(deviation_factors),
        "deviation_factor_details": factor_details,
        "baseline_quality": baseline_quality,
        "model_version": MODEL_VERSION,
    }


def _deviation_score_components(
    features: Mapping[str, Any],
    metric_references: Mapping[str, Any],
    reference_data: Mapping[str, Any],
    config: BaselineModelConfig,
) -> tuple[float, list[dict[str, Any]]]:
    details: list[dict[str, Any]] = []
    weighted_score = 0.0

    for metric_name, factor, direction, weight in (
        ("mean_gait_speed", "gait_speed_drop_from_baseline", "low", 0.22),
        ("mean_sit_stand_duration", "sit_stand_duration_increase_from_baseline", "high", 0.28),
        ("near_fall_frequency", "near_fall_frequency_increase", "high", 0.22),
        ("nighttime_activity_count", "nighttime_activity_increase", "high", 0.10),
        ("activity_volume", "activity_volume_drop", "low", 0.12),
    ):
        component, detail = _metric_deviation_component(
            metric_name=metric_name,
            current_value=features.get(metric_name),
            stats=metric_references.get(metric_name),
            direction=direction,
            factor=factor,
        )
        weighted_score += weight * component
        if detail is not None:
            details.append(detail)

    scene_component, scene_detail = _scene_shift_component(features, reference_data, config)
    weighted_score += 0.06 * scene_component
    if scene_detail is not None:
        details.append(scene_detail)

    return _clamp(weighted_score), details


def _metric_deviation_component(
    *,
    metric_name: str,
    current_value: Any,
    stats: Mapping[str, Any] | None,
    direction: str,
    factor: str,
) -> tuple[float, dict[str, Any] | None]:
    value = _optional_number(current_value)
    if value is None or not stats or int(stats.get("count", 0)) <= 0:
        return 0.0, None

    mean = _number_or_default(stats.get("mean"), 0.0)
    std = _number_or_default(stats.get("std"), 0.0)
    p10 = _number_or_default(stats.get("p10"), mean)
    p25 = _number_or_default(stats.get("p25"), mean)
    p75 = _number_or_default(stats.get("p75"), mean)
    p90 = _number_or_default(stats.get("p90"), mean)
    recent_mean = _optional_number(stats.get("recent_mean"))

    if direction == "low":
        diff = mean - value
        z = diff / _std_scale(std, mean)
        quantile_component = 0.0
        if value < p25:
            quantile_component = 0.45
        if value < p10:
            quantile_component = 0.85
        relative_change = diff / max(abs(mean), 1e-6)
        trend_change = (
            (recent_mean - value) / max(abs(recent_mean), 1e-6)
            if recent_mean is not None
            else relative_change
        )
    else:
        diff = value - mean
        z = diff / _std_scale(std, mean)
        quantile_component = 0.0
        if value > p75:
            quantile_component = 0.45
        if value > p90:
            quantile_component = 0.85
        relative_change = diff / max(abs(mean), 1.0 if mean == 0 else abs(mean))
        trend_change = (
            (value - recent_mean) / max(abs(recent_mean), 1.0 if recent_mean == 0 else abs(recent_mean))
            if recent_mean is not None
            else relative_change
        )

    if diff <= 0:
        return 0.0, None

    z_component = _range_score(z, 1.0, 3.0)
    relative_component = _range_score(relative_change, 0.15, 0.35)
    trend_component = _range_score(trend_change, 0.10, 0.30)
    component = max(z_component, quantile_component, relative_component, trend_component)
    component = _clamp(component)
    if component < 0.35:
        return component, None

    return component, {
        "factor": factor,
        "metric": metric_name,
        "current_value": round(value, 4),
        "baseline_mean": round(mean, 4),
        "baseline_std": round(std, 4),
        "z_score": round(max(0.0, z), 4),
        "relative_change": round(max(0.0, relative_change), 4),
        "trend_change": round(max(0.0, trend_change), 4),
        "component_score": round(component, 4),
    }


def _scene_shift_component(
    features: Mapping[str, Any],
    reference_data: Mapping[str, Any],
    config: BaselineModelConfig,
) -> tuple[float, dict[str, Any] | None]:
    current_scene = features.get("dominant_scene_region")
    if not current_scene:
        return 0.0, None

    distribution = reference_data.get("scene_region_distribution")
    if not isinstance(distribution, Mapping) or not distribution:
        return 0.0, None

    baseline_probability = float(distribution.get(str(current_scene), 0.0))
    component = _clamp(1.0 - (baseline_probability / max(config.scene_shift_probability_threshold, 1e-6)))
    if baseline_probability >= config.scene_shift_probability_threshold:
        return 0.0, None

    return component, {
        "factor": "scene_region_pattern_shift",
        "metric": "dominant_scene_region",
        "current_value": str(current_scene),
        "baseline_probability": round(baseline_probability, 4),
        "component_score": round(component, 4),
    }


def _metric_stats(values: Iterable[Any], config: BaselineModelConfig) -> dict[str, Any]:
    numeric_values = [_optional_number(value) for value in values]
    value_list = [value for value in numeric_values if value is not None]
    if not value_list:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "p10": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p90": None,
            "recent_mean": None,
            "earlier_mean": None,
            "recent_change_ratio": None,
        }

    recent_count = min(max(config.min_history_days, 1), len(value_list))
    recent_values = value_list[-recent_count:]
    earlier_values = value_list[:-recent_count]
    recent_mean = _mean(recent_values)
    earlier_mean = _mean_or_none(earlier_values)
    recent_change_ratio = None
    if earlier_mean is not None and abs(earlier_mean) > 1e-6:
        recent_change_ratio = (recent_mean - earlier_mean) / abs(earlier_mean)

    return {
        "count": len(value_list),
        "mean": round(_mean(value_list), 4),
        "std": round(_std(value_list), 4),
        "min": round(min(value_list), 4),
        "max": round(max(value_list), 4),
        "p10": round(_percentile(value_list, 0.10), 4),
        "p25": round(_percentile(value_list, 0.25), 4),
        "p50": round(_percentile(value_list, 0.50), 4),
        "p75": round(_percentile(value_list, 0.75), 4),
        "p90": round(_percentile(value_list, 0.90), 4),
        "recent_mean": round(recent_mean, 4),
        "earlier_mean": _rounded_or_none(earlier_mean),
        "recent_change_ratio": _rounded_or_none(recent_change_ratio),
    }


def _select_rolling_features(
    features: list[dict[str, Any]],
    config: BaselineModelConfig,
) -> list[dict[str, Any]]:
    sorted_features = sorted(features, key=lambda feature: _sort_time_value(feature.get("start_time")))
    if config.max_history_days <= 0:
        return sorted_features

    seen_days: list[str] = []
    for feature in sorted_features:
        day_key = str(feature.get("day_key", feature.get("period_key", "")))
        if day_key not in seen_days:
            seen_days.append(day_key)
    selected_days = set(seen_days[-config.max_history_days :])
    return [feature for feature in sorted_features if str(feature.get("day_key")) in selected_days]


def _public_current_features(features: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: features.get(key)
        for key in (
            "aggregation_period",
            "period_key",
            "record_count",
            "track_ids",
            "mean_gait_speed",
            "center_speed_cv",
            "hip_lateral_sway",
            "turn_instability_proxy",
            "mean_sit_stand_duration",
            "failed_attempts",
            "mean_stabilization_time",
            "near_fall_frequency",
            "nighttime_activity_count",
            "activity_volume",
            "dominant_scene_region",
            "scene_region_distribution",
            "quality_mean",
            "quality_min",
            "low_quality_record_count",
        )
    }


def _public_reference(reference: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "person_id": reference.get("person_id"),
        "history_record_count": reference.get("history_record_count", 0),
        "history_period_count": reference.get("history_period_count", 0),
        "history_day_count": reference.get("history_day_count", 0),
        "window_start": reference.get("window_start"),
        "window_end": reference.get("window_end"),
        "metric_references": reference.get("metric_references", {}),
        "scene_region_distribution": reference.get("scene_region_distribution", {}),
        "dominant_scene_region": reference.get("dominant_scene_region"),
    }


def _empty_reference(person_id: str) -> dict[str, Any]:
    return {
        "person_id": person_id,
        "history_record_count": 0,
        "history_period_count": 0,
        "history_day_count": 0,
        "metric_references": {},
        "scene_region_distribution": {},
        "dominant_scene_region": None,
        "baseline_quality": {
            "history_record_count": 0,
            "history_period_count": 0,
            "history_day_count": 0,
            "initial_baseline_ready": False,
            "stable_baseline_ready": False,
            "insufficient_baseline_history": True,
            "history_quality_mean": 0.0,
            "reduced_baseline_quality": False,
        },
        "model_version": MODEL_VERSION,
    }


def _quality_score(record: Mapping[str, Any]) -> float:
    quality = record.get("quality_coverage")
    values: list[float] = []
    insufficient = False
    if isinstance(quality, Mapping):
        for key in (
            "usable_frame_ratio",
            "mean_core_keypoint_quality",
            "gait_keypoint_coverage",
            "sit_stand_keypoint_coverage",
            "core_keypoint_coverage",
            "usable_near_fall_window_ratio",
        ):
            value = _optional_number(quality.get(key))
            if value is not None:
                values.append(_clamp(value))
        insufficient = any(
            quality.get(key) is True
            for key in (
                "insufficient_gait_quality",
                "insufficient_sit_stand_quality",
                "insufficient_near_fall_quality",
            )
        )
    direct_quality = _optional_number(record.get("baseline_quality"))
    if direct_quality is not None:
        values.append(_clamp(direct_quality))
    if not values:
        keypoint_quality = _optional_number(record.get("keypoint_quality"))
        values.append(_clamp(keypoint_quality) if keypoint_quality is not None else 0.8)

    score = _mean(values)
    if insufficient:
        score = min(score, 0.45)
    return round(_clamp(score), 4)


def _near_fall_count(record: Mapping[str, Any], config: BaselineModelConfig) -> float:
    for key in ("near_fall_event_count", "near_fall_count", "near_fall_frequency"):
        value = _optional_number(record.get(key))
        if value is not None:
            return max(0.0, value)

    score = _number_or_default(record.get("near_fall_event_score"), 0.0)
    event_type = str(record.get("event_type", ""))
    if score >= config.near_fall_score_count_threshold:
        return 1.0
    if event_type and event_type not in {"unknown", "unknown_near_fall", "none"}:
        return 1.0
    return 0.0


def _nighttime_activity_count(record: Mapping[str, Any], fallback_index: int) -> float:
    for key in ("nighttime_activity_count", "night_activity_count", "nighttime_activity_events"):
        value = _optional_number(record.get(key))
        if value is not None:
            return max(0.0, value)
    return 1.0 if _is_night_record(record, fallback_index) else 0.0


def _activity_value(record: Mapping[str, Any]) -> float | None:
    for key in ("activity_volume", "daily_activity_volume", "active_duration_sec", "activity_duration_sec"):
        value = _optional_number(record.get(key))
        if value is not None:
            return max(0.0, value)
    return None


def _turn_instability_proxy(center_speed_cvs: list[float], hip_lateral_sways: list[float]) -> float | None:
    if not center_speed_cvs and not hip_lateral_sways:
        return None
    cv_component = _ratio_score(_mean(center_speed_cvs), 0.60) if center_speed_cvs else 0.0
    sway_component = _ratio_score(_mean(hip_lateral_sways), 0.05) if hip_lateral_sways else 0.0
    return round(_clamp(max(cv_component, sway_component)), 4)


def _period_key(record: Mapping[str, Any], fallback_index: int, config: BaselineModelConfig) -> str:
    period = config.aggregation_period
    if period not in {"day", "hour"}:
        period = "day"

    parsed = _record_datetime(record)
    if parsed is not None:
        if period == "hour":
            return parsed.strftime("%Y-%m-%dT%H")
        return parsed.date().isoformat()

    seconds = _record_seconds(record)
    if seconds is not None:
        unit = 3600.0 if period == "hour" else 86400.0
        prefix = "hour" if period == "hour" else "day"
        return f"{prefix}_{math.floor(seconds / unit)}"
    return f"{period}_unknown_{fallback_index}"


def _day_key(record: Mapping[str, Any], fallback_index: int) -> str:
    parsed = _record_datetime(record)
    if parsed is not None:
        return parsed.date().isoformat()
    seconds = _record_seconds(record)
    if seconds is not None:
        return f"day_{math.floor(seconds / 86400.0)}"
    return f"day_unknown_{fallback_index}"


def _record_datetime(record: Mapping[str, Any]) -> datetime | None:
    for key in ("timestamp", "start_time", "end_time"):
        value = record.get(key)
        if isinstance(value, str):
            parsed = _parse_datetime(value)
            if parsed is not None:
                return parsed
    return None


def _record_seconds(record: Mapping[str, Any]) -> float | None:
    for key in ("timestamp_sec", "start_time", "timestamp", "end_time"):
        value = _optional_number(record.get(key))
        if value is not None:
            return value
    return None


def _parse_datetime(value: str) -> datetime | None:
    normalized = value.strip()
    if not normalized:
        return None
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _is_night_record(record: Mapping[str, Any], fallback_index: int) -> bool:
    parsed = _record_datetime(record)
    if parsed is None:
        return False
    hour = parsed.hour
    return hour >= 22 or hour < 6


def _record_bounds(items: list[tuple[int, Mapping[str, Any]]]) -> tuple[Any, Any]:
    if not items:
        return None, None
    start_candidates = [
        (_record_sort_key(record, index), _output_start_time(record, index))
        for index, record in items
    ]
    end_candidates = [
        (_record_end_sort_key(record, index), _output_end_time(record, index))
        for index, record in items
    ]
    return min(start_candidates, key=lambda item: item[0])[1], max(end_candidates, key=lambda item: item[0])[1]


def _output_start_time(record: Mapping[str, Any], fallback_index: int) -> Any:
    for key in ("start_time", "timestamp", "timestamp_sec"):
        if record.get(key) is not None:
            return record.get(key)
    return float(fallback_index)


def _output_end_time(record: Mapping[str, Any], fallback_index: int) -> Any:
    for key in ("end_time", "timestamp", "timestamp_sec"):
        if record.get(key) is not None:
            return record.get(key)
    return float(fallback_index)


def _record_sort_key(record: Mapping[str, Any], index: int) -> tuple[float, int]:
    return _sort_time_value(_output_start_time(record, index)), index


def _record_end_sort_key(record: Mapping[str, Any], index: int) -> tuple[float, int]:
    return _sort_time_value(_output_end_time(record, index)), index


def _sort_time_value(value: Any) -> float:
    number = _optional_number(value)
    if number is not None:
        return number
    if isinstance(value, str):
        parsed = _parse_datetime(value)
        if parsed is not None:
            return parsed.timestamp()
    return math.inf


def _min_output_time(values: Iterable[Any]) -> Any:
    value_list = [value for value in values if value is not None]
    if not value_list:
        return None
    return min(value_list, key=_sort_time_value)


def _max_output_time(values: Iterable[Any]) -> Any:
    value_list = [value for value in values if value is not None]
    if not value_list:
        return None
    return max(value_list, key=_sort_time_value)


def _distribution(counter: Counter[str]) -> dict[str, float]:
    total = sum(counter.values())
    if total <= 0:
        return {}
    return {key: round(value / total, 4) for key, value in sorted(counter.items())}


def _baseline_confidence(quality: Mapping[str, Any], config: BaselineModelConfig) -> float:
    history_days = min(1.0, float(quality.get("history_day_count", 0)) / max(config.stable_history_days, 1))
    history_records = min(1.0, float(quality.get("history_record_count", 0)) / max(config.min_history_records, 1))
    history_quality = _clamp(float(quality.get("history_quality_mean", 0.0)))
    current_quality = _clamp(float(quality.get("current_quality_mean", 0.0)))
    confidence = 0.30 * history_days + 0.25 * history_records + 0.25 * history_quality + 0.20 * current_quality
    if quality.get("insufficient_baseline_history"):
        confidence *= 0.45
    if quality.get("reduced_baseline_quality"):
        confidence *= 0.50
    return round(_clamp(confidence), 4)


def _std_scale(std: float, mean: float) -> float:
    return max(std, abs(mean) * 0.05, 0.05)


def _append_number(values: list[float], value: Any) -> None:
    number = _optional_number(value)
    if number is not None:
        values.append(number)


def _mean_or_none(values: Iterable[float]) -> float | None:
    value_list = list(values)
    if not value_list:
        return None
    return _mean(value_list)


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


def _range_score(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return _clamp((value - low) / (high - low))


def _ratio_score(value: float, threshold: float) -> float:
    if threshold <= 0:
        return 0.0
    return _clamp(value / threshold)


def _rounded_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 4)


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


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
