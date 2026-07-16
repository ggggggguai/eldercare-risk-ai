from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import time
from numbers import Real
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from elderly_monitoring.common.config import load_yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[4] / "configs" / "modules" / "mental_health.yaml"
DEFAULT_CORE_KEYPOINTS = (
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
MENTAL_HEALTH_SCORE_FEATURES = (
    "activity_drop_score",
    "sleep_disturbance_score",
    "social_withdrawal_score",
    "routine_irregularity_score",
    "negative_affect_score",
    "self_report_risk_score",
)


@dataclass(frozen=True)
class AggregationConfig:
    timezone: str
    max_gap_seconds: float
    timestamp_conflict_tolerance_seconds: float
    min_keypoint_quality: float
    min_common_core_keypoints: int
    active_motion_threshold: float
    night_start: str
    night_end: str
    core_keypoints: tuple[str, ...]

    @property
    def timezone_info(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)

    @property
    def night_start_time(self) -> time:
        return _parse_clock(self.night_start, "aggregation.night_start")

    @property
    def night_end_time(self) -> time:
        return _parse_clock(self.night_end, "aggregation.night_end")


@dataclass(frozen=True)
class BaselineConfig:
    initial_days: int
    stable_days: int
    max_window_days: int
    abnormal_score_threshold: float
    zero_variance_relative_floor: float
    zero_variance_absolute_floor: float
    z_score_full_scale: float
    relative_change_full_scale: float
    lower_quantile: float
    upper_quantile: float
    missing_quality_default: float


@dataclass(frozen=True)
class ConfidenceConfig:
    feature_coverage_weight: float
    baseline_quality_weight: float
    persistence_weight: float


@dataclass(frozen=True)
class ScoringCaps:
    coverage_below_0_40: int
    coverage_below_0_60: int
    initial_baseline_not_ready: int
    stable_baseline_not_ready: int
    persistent_days_below_minimum: int


@dataclass(frozen=True)
class ScoringConfig:
    weights: dict[str, float]
    thresholds: tuple[float, float, float]
    min_persistent_days_for_level_3: int
    passive_max_level: int
    self_report_emergency_threshold: float
    confidence: ConfidenceConfig
    coverage_expected_features: tuple[str, ...]
    caps: ScoringCaps


@dataclass(frozen=True)
class MentalHealthConfig:
    version: str
    aggregation: AggregationConfig
    baseline: BaselineConfig
    scoring: ScoringConfig


def load_aggregation_config(path: str | Path | None = None) -> AggregationConfig:
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    root = load_yaml(config_path)
    aggregation = _required_mapping(root, "aggregation", source=config_path)
    return aggregation_config_from_mapping(aggregation, source=config_path)


def load_mental_health_config(path: str | Path | None = None) -> MentalHealthConfig:
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    return mental_health_config_from_mapping(load_yaml(config_path), source=config_path)


def mental_health_config_from_mapping(
    root: Mapping[str, Any],
    *,
    source: str | Path = "configuration",
) -> MentalHealthConfig:
    if not isinstance(root, Mapping):
        raise ValueError(f"{source}: configuration root must be a mapping")
    version = _path_string(root, "version", source)
    aggregation = aggregation_config_from_mapping(
        _required_mapping(root, "aggregation", source=source),
        source=source,
    )
    baseline = baseline_config_from_mapping(
        _required_mapping(root, "baseline", source=source),
        source=source,
    )
    scoring = scoring_config_from_mapping(
        _required_mapping(root, "scoring", source=source),
        source=source,
    )
    return MentalHealthConfig(
        version=version,
        aggregation=aggregation,
        baseline=baseline,
        scoring=scoring,
    )


def baseline_config_from_mapping(
    values: Mapping[str, Any],
    *,
    source: str | Path = "configuration",
) -> BaselineConfig:
    integer_fields = ("initial_days", "stable_days", "max_window_days")
    days = {field: _path_positive_integer(values, field, f"{source}: baseline") for field in integer_fields}
    if not days["initial_days"] <= days["stable_days"] <= days["max_window_days"]:
        raise ValueError(
            f"{source}: baseline days must satisfy initial_days <= stable_days <= max_window_days"
        )

    bounded_fields = (
        "abnormal_score_threshold",
        "zero_variance_relative_floor",
        "zero_variance_absolute_floor",
        "lower_quantile",
        "upper_quantile",
        "missing_quality_default",
    )
    numbers = {
        field: _path_finite_number(values, field, f"{source}: baseline")
        for field in bounded_fields
    }
    if not 0.0 <= numbers["abnormal_score_threshold"] <= 1.0:
        raise ValueError(f"{source}: baseline.abnormal_score_threshold must be in [0, 1]")
    for field in ("zero_variance_relative_floor", "zero_variance_absolute_floor"):
        if numbers[field] <= 0:
            raise ValueError(f"{source}: baseline.{field} must be greater than 0")
    if not 0.0 < numbers["lower_quantile"] < numbers["upper_quantile"] < 1.0:
        raise ValueError(
            f"{source}: baseline quantiles must satisfy 0 < lower_quantile < upper_quantile < 1"
        )
    if not 0.0 <= numbers["missing_quality_default"] <= 1.0:
        raise ValueError(f"{source}: baseline.missing_quality_default must be in [0, 1]")

    z_scale = _path_finite_number(values, "z_score_full_scale", f"{source}: baseline")
    relative_scale = _path_finite_number(
        values,
        "relative_change_full_scale",
        f"{source}: baseline",
    )
    if z_scale <= 0:
        raise ValueError(f"{source}: baseline.z_score_full_scale must be greater than 0")
    if relative_scale <= 0:
        raise ValueError(
            f"{source}: baseline.relative_change_full_scale must be greater than 0"
        )

    return BaselineConfig(
        initial_days=days["initial_days"],
        stable_days=days["stable_days"],
        max_window_days=days["max_window_days"],
        abnormal_score_threshold=numbers["abnormal_score_threshold"],
        zero_variance_relative_floor=numbers["zero_variance_relative_floor"],
        zero_variance_absolute_floor=numbers["zero_variance_absolute_floor"],
        z_score_full_scale=z_scale,
        relative_change_full_scale=relative_scale,
        lower_quantile=numbers["lower_quantile"],
        upper_quantile=numbers["upper_quantile"],
        missing_quality_default=numbers["missing_quality_default"],
    )


def scoring_config_from_mapping(
    values: Mapping[str, Any],
    *,
    source: str | Path = "configuration",
) -> ScoringConfig:
    weights_values = _path_mapping(values, "weights", f"{source}: scoring")
    expected_weight_names = set(MENTAL_HEALTH_SCORE_FEATURES)
    actual_weight_names = set(weights_values)
    if actual_weight_names != expected_weight_names:
        missing = sorted(expected_weight_names - actual_weight_names)
        extra = sorted(actual_weight_names - expected_weight_names)
        raise ValueError(
            f"{source}: scoring.weights must contain exactly the supported features; "
            f"missing={missing}, extra={extra}"
        )
    weights: dict[str, float] = {}
    for name in MENTAL_HEALTH_SCORE_FEATURES:
        weight = _path_finite_number(weights_values, name, f"{source}: scoring.weights")
        if weight <= 0:
            raise ValueError(f"{source}: scoring.weights.{name} must be greater than 0")
        weights[name] = weight

    threshold_values = _path_mapping(values, "thresholds", f"{source}: scoring")
    thresholds = tuple(
        _path_finite_number(threshold_values, field, f"{source}: scoring.thresholds")
        for field in ("level_1", "level_2", "level_3")
    )
    if not 0.0 <= thresholds[0] < thresholds[1] < thresholds[2] <= 1.0:
        raise ValueError(
            f"{source}: scoring.thresholds must satisfy 0 <= level_1 < level_2 < level_3 <= 1"
        )

    persistence_days = _path_positive_integer(
        values,
        "min_persistent_days_for_level_3",
        f"{source}: scoring",
    )
    passive_max_level = _path_integer(values, "passive_max_level", f"{source}: scoring")
    if not 0 <= passive_max_level <= 3:
        raise ValueError(f"{source}: scoring.passive_max_level must be an integer in [0, 3]")
    emergency_threshold = _path_finite_number(
        values,
        "self_report_emergency_threshold",
        f"{source}: scoring",
    )
    if not 0.0 <= emergency_threshold <= 1.0:
        raise ValueError(
            f"{source}: scoring.self_report_emergency_threshold must be in [0, 1]"
        )

    confidence_values = _path_mapping(values, "confidence", f"{source}: scoring")
    confidence_numbers = {
        field: _path_finite_number(
            confidence_values,
            field,
            f"{source}: scoring.confidence",
        )
        for field in (
            "feature_coverage_weight",
            "baseline_quality_weight",
            "persistence_weight",
        )
    }
    if any(value < 0 for value in confidence_numbers.values()):
        raise ValueError(f"{source}: scoring.confidence weights must be non-negative")
    if not math.isclose(sum(confidence_numbers.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"{source}: scoring.confidence weights must sum to 1.0")
    confidence = ConfidenceConfig(**confidence_numbers)

    raw_expected = values.get("coverage_expected_features")
    if not isinstance(raw_expected, list) or not raw_expected:
        raise ValueError(
            f"{source}: scoring.coverage_expected_features must be a non-empty list"
        )
    if any(not isinstance(name, str) or not name.strip() for name in raw_expected):
        raise ValueError(
            f"{source}: scoring.coverage_expected_features must contain non-empty strings"
        )
    coverage_expected = tuple(name.strip() for name in raw_expected)
    if len(set(coverage_expected)) != len(coverage_expected):
        raise ValueError(
            f"{source}: scoring.coverage_expected_features must not contain duplicates"
        )
    unknown_expected = sorted(set(coverage_expected) - expected_weight_names)
    if unknown_expected:
        raise ValueError(
            f"{source}: scoring.coverage_expected_features contains unsupported features: "
            f"{unknown_expected}"
        )

    cap_values = _path_mapping(values, "caps", f"{source}: scoring")
    cap_names = (
        "coverage_below_0_40",
        "coverage_below_0_60",
        "initial_baseline_not_ready",
        "stable_baseline_not_ready",
        "persistent_days_below_minimum",
    )
    caps_data: dict[str, int] = {}
    for field in cap_names:
        cap = _path_integer(cap_values, field, f"{source}: scoring.caps")
        if not 0 <= cap <= 3:
            raise ValueError(f"{source}: scoring.caps.{field} must be an integer in [0, 3]")
        caps_data[field] = cap

    return ScoringConfig(
        weights=weights,
        thresholds=(thresholds[0], thresholds[1], thresholds[2]),
        min_persistent_days_for_level_3=persistence_days,
        passive_max_level=passive_max_level,
        self_report_emergency_threshold=emergency_threshold,
        confidence=confidence,
        coverage_expected_features=coverage_expected,
        caps=ScoringCaps(**caps_data),
    )


def aggregation_config_from_mapping(
    values: Mapping[str, Any],
    *,
    source: str | Path = "configuration",
) -> AggregationConfig:
    timezone_name = _required_string(values, "timezone", source)
    try:
        ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"{source}: aggregation.timezone is unknown: {timezone_name!r}") from exc

    max_gap_seconds = _finite_number(values, "max_gap_seconds", source)
    if max_gap_seconds <= 0:
        raise ValueError(f"{source}: aggregation.max_gap_seconds must be greater than 0")

    conflict_tolerance = _finite_number(values, "timestamp_conflict_tolerance_seconds", source)
    if conflict_tolerance < 0:
        raise ValueError(
            f"{source}: aggregation.timestamp_conflict_tolerance_seconds must be non-negative"
        )

    min_quality = _finite_number(values, "min_keypoint_quality", source)
    if not 0.0 <= min_quality <= 1.0:
        raise ValueError(f"{source}: aggregation.min_keypoint_quality must be in [0, 1]")

    common_count = values.get("min_common_core_keypoints")
    if isinstance(common_count, bool) or not isinstance(common_count, int) or common_count <= 0:
        raise ValueError(
            f"{source}: aggregation.min_common_core_keypoints must be a positive integer"
        )

    active_threshold = _finite_number(values, "active_motion_threshold", source)
    if active_threshold < 0:
        raise ValueError(f"{source}: aggregation.active_motion_threshold must be non-negative")

    night_start = _required_string(values, "night_start", source)
    night_end = _required_string(values, "night_end", source)
    _parse_clock(night_start, f"{source}: aggregation.night_start")
    _parse_clock(night_end, f"{source}: aggregation.night_end")
    if night_start == night_end:
        raise ValueError(f"{source}: aggregation night window cannot have identical boundaries")

    raw_core_keypoints = values.get("core_keypoints", list(DEFAULT_CORE_KEYPOINTS))
    if not isinstance(raw_core_keypoints, list) or not raw_core_keypoints:
        raise ValueError(f"{source}: aggregation.core_keypoints must be a non-empty list")
    if any(not isinstance(name, str) or not name.strip() for name in raw_core_keypoints):
        raise ValueError(f"{source}: aggregation.core_keypoints must contain non-empty strings")
    core_keypoints = tuple(name.strip() for name in raw_core_keypoints)
    if len(set(core_keypoints)) != len(core_keypoints):
        raise ValueError(f"{source}: aggregation.core_keypoints must not contain duplicates")
    if common_count > len(core_keypoints):
        raise ValueError(
            f"{source}: aggregation.min_common_core_keypoints cannot exceed core_keypoints length"
        )

    return AggregationConfig(
        timezone=timezone_name,
        max_gap_seconds=max_gap_seconds,
        timestamp_conflict_tolerance_seconds=conflict_tolerance,
        min_keypoint_quality=min_quality,
        min_common_core_keypoints=common_count,
        active_motion_threshold=active_threshold,
        night_start=night_start,
        night_end=night_end,
        core_keypoints=core_keypoints,
    )


def _required_mapping(root: Mapping[str, Any], key: str, *, source: str | Path) -> Mapping[str, Any]:
    value = root.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{source}: {key} must be a mapping")
    return value


def _required_string(values: Mapping[str, Any], key: str, source: str | Path) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{source}: aggregation.{key} must be a non-empty string")
    return value.strip()


def _finite_number(values: Mapping[str, Any], key: str, source: str | Path) -> float:
    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{source}: aggregation.{key} must be a finite number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{source}: aggregation.{key} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{source}: aggregation.{key} must be a finite number")
    return number


def _parse_clock(value: str, path: str) -> time:
    if len(value) != 5 or value[2] != ":":
        raise ValueError(f"{path} must use HH:MM format")
    try:
        hour = int(value[:2])
        minute = int(value[3:])
        parsed = time(hour=hour, minute=minute)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} must use a valid HH:MM time") from exc
    if parsed.strftime("%H:%M") != value:
        raise ValueError(f"{path} must use zero-padded HH:MM format")
    return parsed


def _path_mapping(
    values: Mapping[str, Any],
    key: str,
    parent_path: str | Path,
) -> Mapping[str, Any]:
    value = values.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{parent_path}.{key} must be a mapping")
    return value


def _path_string(values: Mapping[str, Any], key: str, parent_path: str | Path) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{parent_path}: {key} must be a non-empty string")
    return value.strip()


def _path_finite_number(
    values: Mapping[str, Any],
    key: str,
    parent_path: str | Path,
) -> float:
    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{parent_path}.{key} must be a finite number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{parent_path}.{key} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{parent_path}.{key} must be a finite number")
    return number


def _path_integer(values: Mapping[str, Any], key: str, parent_path: str | Path) -> int:
    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{parent_path}.{key} must be an integer")
    return value


def _path_positive_integer(
    values: Mapping[str, Any],
    key: str,
    parent_path: str | Path,
) -> int:
    value = _path_integer(values, key, parent_path)
    if value <= 0:
        raise ValueError(f"{parent_path}.{key} must be a positive integer")
    return value
