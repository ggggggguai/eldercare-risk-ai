"""Causal, robust personal behaviour baseline for the fall-risk module.

The public batch API consumes completed period features. It deliberately does
not accept pose frames: aggregation into an hour/day record is an upstream
responsibility. Outputs are internal fall-risk features, not risk events or
medical conclusions.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from elderly_monitoring.common.config import load_yaml
from elderly_monitoring.modules.fall_risk.pose import write_jsonl


MODEL_VERSION = "fall-personal-baseline-robust-v1"
PERIOD_SCHEMA_VERSION = "fall-baseline-period-features-v1"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[4] / "configs/modules/fall_risk_baseline.yaml"


def _default_metric_weights() -> dict[str, float]:
    return {
        "mean_gait_speed": 0.22,
        "mean_sit_stand_duration": 0.28,
        "near_fall_rate_per_hour": 0.22,
        "nighttime_activity_rate_per_hour": 0.10,
        "activity_volume": 0.12,
    }


@dataclass(frozen=True)
class BaselineModelConfig:
    config_version: str = "fall-personal-baseline-config-v1"
    aggregation_period: str = "day"
    min_history_days: int = 3
    stable_history_days: int = 7
    max_history_days: int = 14
    min_history_records: int = 10
    min_metric_observations: int = 3
    min_quality_score: float = 0.60
    min_metric_quality: float = 0.30
    initial_score_cap: float = 0.35
    initial_fusion_weight: float = 0.25
    reduced_quality_score_cap: float = 0.25
    relative_scale_floor: float = 0.05
    absolute_scale_floor: float = 0.05
    robust_z_start: float = 1.0
    robust_z_full: float = 3.0
    relative_change_start: float = 0.15
    relative_change_full: float = 0.35
    short_change_start: float = 0.10
    short_change_full: float = 0.30
    scene_shift_probability_threshold: float = 0.20
    fast_ewma_alpha: float = 0.50
    drift_metric_score_threshold: float = 0.75
    drift_consecutive_periods: int = 2
    cusum_allowance: float = 0.35
    cusum_threshold: float = 1.00
    recovery_consecutive_periods: int = 3
    metric_weights: Mapping[str, float] = field(default_factory=_default_metric_weights)


def load_baseline_config(path: Path | str | None = None) -> BaselineModelConfig:
    """Load the versioned provisional algorithm thresholds."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        return BaselineModelConfig()
    payload = load_yaml(config_path)
    allowed = {item.name for item in fields(BaselineModelConfig)}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"unknown baseline config fields: {', '.join(unknown)}")
    return BaselineModelConfig(**payload)


def build_personal_baselines(
    records: Iterable[Mapping[str, Any]],
    *,
    config: BaselineModelConfig | None = None,
) -> dict[str, dict[str, Any]]:
    """Build references grouped by person and camera profile.

    Invalid/non-period inputs fail closed. Exact duplicate periods are ignored;
    conflicting records for the same identity and period are rejected.
    """
    baseline_config = config or load_baseline_config()
    periods = _normalise_periods(records, baseline_config)
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for period in periods:
        person = str(period["person_id"])
        camera = str(period["camera_profile_id"])
        grouped.setdefault(person, {}).setdefault(camera, []).append(period)

    output: dict[str, dict[str, Any]] = {}
    for person, camera_periods in grouped.items():
        output[person] = {
            "person_id": person,
            "camera_references": {
                camera: _build_reference(
                    person,
                    camera,
                    _select_rolling_periods(periods_for_camera, baseline_config),
                    baseline_config,
                )
                for camera, periods_for_camera in sorted(camera_periods.items())
            },
            "model_version": MODEL_VERSION,
        }
    return output


def score_baseline_deviation(
    current_records: Iterable[Mapping[str, Any]],
    baselines: Mapping[str, Mapping[str, Any]],
    *,
    config: BaselineModelConfig | None = None,
) -> list[dict[str, Any]]:
    """Score each completed current period against strictly earlier history."""
    baseline_config = config or load_baseline_config()
    current_periods = _normalise_periods(current_records, baseline_config)
    outputs = []
    for current in current_periods:
        person_bundle = baselines.get(str(current["person_id"]))
        reference = _causal_reference(person_bundle, current, baseline_config)
        outputs.append(_score_period(current, reference, baseline_config))
    return sorted(outputs, key=lambda item: (item["person_id"], _time_value(item["start_time"])))


def run_baseline_jsonl(
    *,
    baseline_input_path: Path,
    current_input_path: Path,
    output_path: Path,
    config: BaselineModelConfig | None = None,
) -> int:
    baseline_config = config or load_baseline_config()
    baselines = build_personal_baselines(_read_jsonl(baseline_input_path), config=baseline_config)
    outputs = score_baseline_deviation(
        _read_jsonl(current_input_path), baselines, config=baseline_config
    )
    return write_jsonl(outputs, output_path)


class PersonalBaselineTracker:
    """Stateful slow-reference/fast-state tracker for ordered period records."""

    def __init__(self, *, config: BaselineModelConfig | None = None) -> None:
        self.config = config or load_baseline_config()
        self._slow_periods: list[dict[str, Any]] = []
        self._identity: tuple[str, str] | None = None
        self._last_end: float | None = None
        self._drift_streak = 0
        self._recovery_streak = 0
        self._state = "none"
        self._fast_levels: dict[str, float] = {}
        self._cusum: dict[str, float] = {}

    def update(self, record: Mapping[str, Any]) -> dict[str, Any]:
        periods = _normalise_periods([record], self.config)
        if len(periods) != 1:
            raise ValueError("tracker requires one valid completed baseline period")
        current = periods[0]
        identity = (str(current["person_id"]), str(current["camera_profile_id"]))
        if self._identity is not None and identity != self._identity:
            raise ValueError("tracker identity or camera profile changed")
        if self._last_end is not None and _time_value(current["period_start"]) <= self._last_end:
            raise ValueError("tracker periods must be strictly ordered and non-overlapping")
        self._identity = identity

        reference = None
        if self._slow_periods:
            reference = _build_reference(
                identity[0],
                identity[1],
                _select_rolling_periods(self._slow_periods, self.config),
                self.config,
            )
        result = _score_period(current, reference, self.config)
        metric_scores = [
            float(value)
            for value in result["metric_deviation_scores"].values()
            if value is not None
        ]
        for metric, value in result["metric_deviation_scores"].items():
            if value is None:
                continue
            self._cusum[metric] = max(
                0.0,
                self._cusum.get(metric, 0.0) + float(value) - self.config.cusum_allowance,
            )
        cusum_triggered = any(
            value >= self.config.cusum_threshold for value in self._cusum.values()
        )
        current_anomalous = (
            bool(metric_scores)
            and max(metric_scores) >= self.config.drift_metric_score_threshold
        )
        previous_state = self._state
        # CUSUM can enter drift from a stable state, but recovery must follow
        # current evidence; otherwise old accumulated evidence prevents exit.
        anomalous = current_anomalous or (
            previous_state not in {"drift_suspected", "recovery"} and cusum_triggered
        )
        if previous_state in {"drift_suspected", "recovery"}:
            if anomalous:
                self._state = "drift_suspected"
                self._recovery_streak = 0
            else:
                self._cusum = {
                    metric: max(0.0, value - self.config.cusum_allowance)
                    for metric, value in self._cusum.items()
                }
                self._recovery_streak += 1
                if self._recovery_streak >= self.config.recovery_consecutive_periods:
                    self._state = "stable"
                    self._drift_streak = 0
                    self._recovery_streak = 0
                    self._slow_periods.append(current)
                else:
                    self._state = "recovery"
        elif anomalous and result["baseline_state"] == "stable":
            self._drift_streak += 1
            if self._drift_streak >= self.config.drift_consecutive_periods:
                self._state = "drift_suspected"
            else:
                self._state = "stable"
        else:
            self._drift_streak = 0
            self._state = str(result["baseline_state"])
            self._slow_periods.append(current)

        for metric in _SCORING_METRICS:
            value = _optional_number(current.get(metric))
            if value is None or not current["metric_status"][metric]["available"]:
                continue
            old = self._fast_levels.get(metric, value)
            alpha = self.config.fast_ewma_alpha
            self._fast_levels[metric] = alpha * value + (1.0 - alpha) * old

        result["baseline_state"] = self._state
        result["slow_reference_frozen"] = self._state in {"drift_suspected", "recovery"} or (
            anomalous and result["baseline_state"] == "stable"
        )
        result["fast_state"] = {
            "ewma": {key: round(value, 4) for key, value in sorted(self._fast_levels.items())},
            "cusum": {key: round(value, 4) for key, value in sorted(self._cusum.items())},
            "cusum_triggered": cusum_triggered,
            "drift_streak": self._drift_streak,
            "recovery_streak": self._recovery_streak,
        }
        self._last_end = _time_value(current["period_end"])
        return result


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


_REFERENCE_METRICS = (
    "mean_gait_speed",
    "center_speed_cv",
    "hip_lateral_sway",
    "turn_instability_proxy",
    "mean_sit_stand_duration",
    "failed_attempts",
    "mean_stabilization_time",
    "near_fall_rate_per_hour",
    "nighttime_activity_rate_per_hour",
    "activity_volume",
)

_SCORING_METRICS = {
    "mean_gait_speed": ("gait_speed_drop_from_baseline", "low"),
    "mean_sit_stand_duration": ("sit_stand_duration_increase_from_baseline", "high"),
    "near_fall_rate_per_hour": ("near_fall_frequency_increase", "high"),
    "nighttime_activity_rate_per_hour": ("nighttime_activity_increase", "high"),
    "activity_volume": ("activity_volume_drop", "low"),
}


def _normalise_periods(
    records: Iterable[Mapping[str, Any]], config: BaselineModelConfig
) -> list[dict[str, Any]]:
    unique: dict[tuple[str, str, str], tuple[str, dict[str, Any]]] = {}
    for source in records:
        record = dict(source)
        if record.get("record_type") != "fall_baseline_period_features":
            continue
        if record.get("schema_version") != PERIOD_SCHEMA_VERSION or record.get("completed") is not True:
            continue
        person = str(record.get("person_id", "")).strip()
        camera = str(record.get("camera_profile_id", "")).strip()
        period_id = str(record.get("period_id", "")).strip()
        device = str(record.get("device_id", "")).strip()
        upstream_versions = record.get("upstream_versions")
        input_summary = record.get("input_summary")
        metric_quality = record.get("metric_quality")
        monitoring_hours = _optional_number(record.get("valid_monitoring_hours"))
        aggregation_version = str(record.get("aggregation_version", "")).strip()
        if (
            not person
            or person.lower() == "unknown"
            or not device
            or not camera
            or not period_id
            or not aggregation_version
            or not isinstance(upstream_versions, Mapping)
            or not upstream_versions
            or not isinstance(input_summary, Mapping)
            or not input_summary
            or not isinstance(metric_quality, Mapping)
            or not metric_quality
            or monitoring_hours is None
            or monitoring_hours <= 0
        ):
            continue
        start = _strict_datetime(record.get("period_start"))
        end = _strict_datetime(record.get("period_end"))
        if start is None or end is None or end <= start or not str(record.get("timezone", "")).strip():
            continue
        if config.aggregation_period not in {"day", "hour"}:
            raise ValueError("aggregation_period must be day or hour")

        normalised = _period_metrics(record, config)
        normalised.update({
            "person_id": person,
            "device_id": device,
            "camera_profile_id": camera,
            "period_id": period_id,
            "period_start": record["period_start"],
            "period_end": record["period_end"],
            "start_time": record["period_start"],
            "timestamp": record["period_start"],
            "end_time": record["period_end"],
            "timezone": str(record["timezone"]),
            "day_key": start.date().isoformat(),
            "aggregation_period": config.aggregation_period,
            "aggregation_version": aggregation_version,
            "upstream_versions": dict(upstream_versions),
            "input_summary": dict(input_summary),
            "record_count": int(_number(record.get("record_count"), 1.0)),
            "dominant_scene_region": record.get("dominant_scene_region") or record.get("scene_region"),
            "scene_region_distribution": _scene_distribution(record),
            "quality_mean": _quality_score(record),
        })
        identity = (person, camera, period_id)
        fingerprint = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        prior = unique.get(identity)
        if prior is not None and prior[0] != fingerprint:
            raise ValueError(f"conflicting baseline period: {person}/{camera}/{period_id}")
        unique[identity] = (fingerprint, normalised)
    return sorted(
        (item[1] for item in unique.values()),
        key=lambda period: (period["person_id"], period["camera_profile_id"], _time_value(period["period_start"])),
    )


def _period_metrics(record: Mapping[str, Any], config: BaselineModelConfig) -> dict[str, Any]:
    gait = record.get("gait_stability_features")
    gait = gait if isinstance(gait, Mapping) else {}
    exposure = _optional_number(record.get("valid_monitoring_hours"))
    near_count = _first_number(record, "near_fall_event_count", "near_fall_count")
    night_count = _first_number(record, "nighttime_activity_count", "night_activity_count")
    values = {
        "mean_gait_speed": _first_number(gait, "mean_center_speed_norm_per_sec"),
        "center_speed_cv": _first_number(gait, "center_speed_cv"),
        "hip_lateral_sway": _first_number(gait, "hip_lateral_sway"),
        "turn_instability_proxy": _first_number(record, "turn_instability_proxy"),
        "mean_sit_stand_duration": _first_number(record, "mean_sit_stand_duration", "duration"),
        "failed_attempts": _first_number(record, "failed_attempts"),
        "mean_stabilization_time": _first_number(record, "mean_stabilization_time", "stabilization_time"),
        "near_fall_rate_per_hour": near_count / exposure if near_count is not None and exposure and exposure > 0 else None,
        "nighttime_activity_rate_per_hour": night_count / exposure if night_count is not None and exposure and exposure > 0 else None,
        "activity_volume": _first_number(record, "activity_volume", "daily_activity_volume"),
    }
    status = {
        metric: _metric_status(record, metric, value, exposure, config)
        for metric, value in values.items()
    }
    return {
        **{key: _round(value) for key, value in values.items()},
        "valid_monitoring_hours": _round(exposure),
        "metric_status": status,
    }


def _metric_status(
    record: Mapping[str, Any],
    metric: str,
    value: float | None,
    exposure: float | None,
    config: BaselineModelConfig,
) -> dict[str, Any]:
    quality_map = record.get("metric_quality")
    quality_map = quality_map if isinstance(quality_map, Mapping) else {}
    raw = quality_map.get(metric)
    raw = raw if isinstance(raw, Mapping) else {}
    count = int(_number(raw.get("observation_count"), 1.0 if value is not None else 0.0))
    quality = _clamp(_number(raw.get("quality"), _quality_score(record)))
    coverage = _clamp(_number(raw.get("coverage"), quality))
    metric_exposure = _optional_number(raw.get("exposure_hours"))
    if metric_exposure is None:
        metric_exposure = exposure
    available = bool(raw.get("available", value is not None)) and value is not None and count > 0
    if metric.endswith("_rate_per_hour") and (metric_exposure is None or metric_exposure <= 0):
        available = False
    missing_reason = raw.get("missing_reason")
    if not available and not missing_reason:
        missing_reason = "missing_value_or_exposure"
    return {
        "available": available,
        "observation_count": count,
        "quality": round(quality, 4),
        "coverage": round(coverage, 4),
        "exposure_hours": _round(metric_exposure),
        "missing_reason": missing_reason,
    }


def _causal_reference(
    person_bundle: Mapping[str, Any] | None,
    current: Mapping[str, Any],
    config: BaselineModelConfig,
) -> dict[str, Any] | None:
    if not isinstance(person_bundle, Mapping):
        return None
    cameras = person_bundle.get("camera_references")
    if not isinstance(cameras, Mapping):
        return None
    reference = cameras.get(str(current["camera_profile_id"]))
    if not isinstance(reference, Mapping):
        return None
    cutoff = _time_value(current["period_start"])
    history = [
        dict(period)
        for period in reference.get("_periods", [])
        if _time_value(period.get("period_end")) < cutoff
    ]
    if not history:
        return None
    history = _select_rolling_periods(history, config)
    return _build_reference(
        str(current["person_id"]), str(current["camera_profile_id"]), history, config
    )


def _select_rolling_periods(
    periods: Iterable[Mapping[str, Any]], config: BaselineModelConfig
) -> list[dict[str, Any]]:
    ordered = sorted((dict(period) for period in periods), key=lambda item: _time_value(item["period_start"]))
    if config.max_history_days <= 0:
        return ordered
    days: list[str] = []
    for period in ordered:
        if period["day_key"] not in days:
            days.append(period["day_key"])
    selected = set(days[-config.max_history_days :])
    return [period for period in ordered if period["day_key"] in selected]


def _build_reference(
    person: str,
    camera: str,
    periods: list[dict[str, Any]],
    config: BaselineModelConfig,
) -> dict[str, Any]:
    metric_references: dict[str, dict[str, Any]] = {}
    for metric in _REFERENCE_METRICS:
        observations = [
            (float(period[metric]), period["metric_status"][metric])
            for period in periods
            if period.get(metric) is not None
            and period["metric_status"][metric]["available"]
            and period["metric_status"][metric]["quality"] >= config.min_metric_quality
        ]
        metric_references[metric] = _metric_stats(observations, config)
        unavailable = [
            period["metric_status"][metric]
            for period in periods
            if not period["metric_status"][metric]["available"]
        ]
        metric_references[metric]["unavailable_period_count"] = len(unavailable)
        metric_references[metric]["missing_reason_counts"] = dict(sorted(Counter(
            str(status.get("missing_reason") or "unspecified") for status in unavailable
        ).items()))

    days = {period["day_key"] for period in periods}
    record_count = sum(int(period.get("record_count", 1)) for period in periods)
    quality_mean = _mean(float(period.get("quality_mean", 0.0)) for period in periods)
    scene = Counter()
    for period in periods:
        for key, value in period.get("scene_region_distribution", {}).items():
            scene[str(key)] += float(value)
    return {
        "person_id": person,
        "camera_profile_id": camera,
        "history_record_count": record_count,
        "history_period_count": len(periods),
        "history_day_count": len(days),
        "window_start": periods[0]["period_start"] if periods else None,
        "window_end": periods[-1]["period_end"] if periods else None,
        "metric_references": metric_references,
        "scene_region_distribution": _distribution(scene),
        "history_quality_mean": round(quality_mean, 4),
        "_periods": periods,
        "model_version": MODEL_VERSION,
    }


def _metric_stats(
    observations: list[tuple[float, Mapping[str, Any]]], config: BaselineModelConfig
) -> dict[str, Any]:
    values = [value for value, _ in observations]
    if not values:
        return {
            "count": 0, "mean": None, "std": None, "median": None, "mad": None,
            "winsorized_mean": None, "p10": None, "p25": None, "p50": None,
            "p75": None, "p90": None, "observed_period_count": 0,
            "total_exposure": 0.0, "quality_weight_sum": 0.0,
            "recent_level": None, "slow_level": None,
        }
    median = _percentile(values, 0.5)
    mad = _percentile([abs(value - median) for value in values], 0.5)
    p10, p90 = _percentile(values, 0.1), _percentile(values, 0.9)
    winsorized = [min(p90, max(p10, value)) for value in values]
    recent_count = min(max(config.min_history_days, 1), len(values))
    recent = values[-recent_count:]
    ewma = values[0]
    for value in values[1:]:
        ewma = config.fast_ewma_alpha * value + (1.0 - config.fast_ewma_alpha) * ewma
    return {
        "count": len(values),
        "mean": _round(_mean(values)),
        "std": _round(_std(values)),
        "median": _round(median),
        "mad": _round(mad),
        "winsorized_mean": _round(_mean(winsorized)),
        "min": _round(min(values)),
        "max": _round(max(values)),
        "p10": _round(p10),
        "p25": _round(_percentile(values, 0.25)),
        "p50": _round(median),
        "p75": _round(_percentile(values, 0.75)),
        "p90": _round(p90),
        "observed_period_count": len(values),
        "total_exposure": _round(sum(_number(status.get("exposure_hours"), 0.0) for _, status in observations)),
        "quality_weight_sum": _round(sum(_number(status.get("quality"), 0.0) for _, status in observations)),
        "recent_level": _round(_mean(recent)),
        "slow_level": _round(median),
        "fast_ewma": _round(ewma),
    }


def _score_period(
    current: Mapping[str, Any],
    reference: Mapping[str, Any] | None,
    config: BaselineModelConfig,
) -> dict[str, Any]:
    day_count = int(reference.get("history_day_count", 0)) if reference else 0
    record_count = int(reference.get("history_record_count", 0)) if reference else 0
    if day_count == 0:
        state = "none"
    elif day_count < config.min_history_days or record_count < config.min_history_records:
        state = "cold"
    elif day_count < config.stable_history_days:
        state = "initial"
    else:
        state = "stable"

    refs = reference.get("metric_references", {}) if reference else {}
    mask: dict[str, bool] = {}
    metric_scores: dict[str, float | None] = {}
    legacy_scores: dict[str, float | None] = {}
    details: list[dict[str, Any]] = []
    weighted_score = 0.0
    weight_sum = 0.0
    for metric, (factor, direction) in _SCORING_METRICS.items():
        current_status = current["metric_status"][metric]
        stats = refs.get(metric)
        available = (
            state in {"initial", "stable"}
            and current_status["available"]
            and float(current_status["quality"]) >= config.min_metric_quality
            and isinstance(stats, Mapping)
            and int(stats.get("observed_period_count", 0)) >= config.min_metric_observations
        )
        mask[metric] = available
        if not available:
            metric_scores[metric] = None
            legacy_scores[metric] = None
            continue
        component, detail = _metric_component(
            metric, float(current[metric]), stats, direction, factor, current_status, config
        )
        metric_scores[metric] = round(component, 4)
        legacy_scores[metric] = round(
            _legacy_mean_std_component(float(current[metric]), stats, direction, config), 4
        )
        quality_weight = config.metric_weights[metric] * float(current_status["quality"])
        weighted_score += component * quality_weight
        weight_sum += quality_weight
        if detail is not None:
            details.append(detail)

    scene_score, scene_detail = _scene_component(current, reference, config)
    if scene_detail is not None and state in {"initial", "stable"}:
        details.append(scene_detail)
        weighted_score += 0.06 * scene_score * float(current["quality_mean"])
        weight_sum += 0.06 * float(current["quality_mean"])

    score = weighted_score / weight_sum if weight_sum > 0 else None
    reduced_quality = bool(reference) and (
        float(reference.get("history_quality_mean", 0.0)) < config.min_quality_score
        or float(current["quality_mean"]) < config.min_quality_score
    )
    if score is not None and state == "initial":
        score = min(score, config.initial_score_cap)
    if score is not None and reduced_quality:
        score = min(score, config.reduced_quality_score_cap)

    factors = [detail["factor"] for detail in details]
    if state in {"none", "cold"}:
        factors.insert(0, "insufficient_baseline_history")
    if reduced_quality:
        factors.insert(0, "reduced_baseline_quality")
    confidence = _baseline_confidence(day_count, record_count, current, reference, state, config)
    return {
        "person_id": current["person_id"],
        "camera_profile_id": current["camera_profile_id"],
        "start_time": current["period_start"],
        "timestamp": current["period_start"],
        "end_time": current["period_end"],
        "baseline_deviation_score": _round(score),
        "baseline_state": state,
        "baseline_confidence": confidence,
        "available_metric_mask": mask,
        "metric_deviation_scores": metric_scores,
        "legacy_mean_std_metric_scores": legacy_scores,
        "scene_deviation_score": round(scene_score, 4),
        "baseline_fusion_weight": (
            1.0 if state == "stable" else config.initial_fusion_weight if state == "initial" else 0.0
        ),
        "baseline_features": _public_period(current),
        "baseline_reference": _public_reference(reference),
        "history_cutoff": current["period_start"],
        "deviation_factors": list(dict.fromkeys(factors)),
        "deviation_factor_details": details,
        "baseline_quality": {
            "history_record_count": record_count,
            "history_period_count": int(reference.get("history_period_count", 0)) if reference else 0,
            "history_day_count": day_count,
            "current_quality_mean": current["quality_mean"],
            "history_quality_mean": reference.get("history_quality_mean", 0.0) if reference else 0.0,
            "initial_baseline_ready": state in {"initial", "stable"},
            "stable_baseline_ready": state == "stable",
            "insufficient_baseline_history": state in {"none", "cold"},
            "reduced_baseline_quality": reduced_quality,
            "baseline_confidence": confidence,
        },
        "slow_reference_frozen": False,
        "model_version": MODEL_VERSION,
        "config_version": config.config_version,
    }


def _metric_component(
    metric: str,
    value: float,
    stats: Mapping[str, Any],
    direction: str,
    factor: str,
    status: Mapping[str, Any],
    config: BaselineModelConfig,
) -> tuple[float, dict[str, Any] | None]:
    median = float(stats["median"])
    mad = float(stats["mad"])
    recent = _number(stats.get("recent_level"), median)
    directional_delta = median - value if direction == "low" else value - median
    if directional_delta <= 0:
        return 0.0, None
    scale = max(
        1.4826 * mad,
        abs(median) * config.relative_scale_floor,
        config.absolute_scale_floor,
    )
    robust_z = directional_delta / scale
    relative_change = directional_delta / max(abs(median), config.absolute_scale_floor)
    short_delta = recent - value if direction == "low" else value - recent
    short_change = max(0.0, short_delta) / max(abs(recent), config.absolute_scale_floor)
    p10, p25 = float(stats["p10"]), float(stats["p25"])
    p75, p90 = float(stats["p75"]), float(stats["p90"])
    quantile = 0.0
    if direction == "low":
        quantile = 0.85 if value < p10 else 0.45 if value < p25 else 0.0
    else:
        quantile = 0.85 if value > p90 else 0.45 if value > p75 else 0.0
    component = max(
        _range_score(robust_z, config.robust_z_start, config.robust_z_full),
        quantile,
        _range_score(relative_change, config.relative_change_start, config.relative_change_full),
        _range_score(short_change, config.short_change_start, config.short_change_full),
    )
    detail = None
    if component >= 0.35:
        detail = {
            "factor": factor,
            "metric": metric,
            "current_value": round(value, 4),
            "baseline_median": round(median, 4),
            "baseline_mad": round(mad, 4),
            "winsorized_mean": stats.get("winsorized_mean"),
            "robust_z": round(robust_z, 4),
            "relative_change": round(relative_change, 4),
            "short_term_change": round(short_change, 4),
            "quality": status["quality"],
            "observation_count": status["observation_count"],
            "exposure_hours": status["exposure_hours"],
            "component_score": round(component, 4),
        }
    return _clamp(component), detail


def _scene_component(
    current: Mapping[str, Any],
    reference: Mapping[str, Any] | None,
    config: BaselineModelConfig,
) -> tuple[float, dict[str, Any] | None]:
    scene = current.get("dominant_scene_region")
    distribution = reference.get("scene_region_distribution", {}) if reference else {}
    if not scene or not distribution:
        return 0.0, None
    probability = float(distribution.get(str(scene), 0.0))
    if probability >= config.scene_shift_probability_threshold:
        return 0.0, None
    score = _clamp(1.0 - probability / max(config.scene_shift_probability_threshold, 1e-6))
    return score, {
        "factor": "scene_region_pattern_shift",
        "metric": "dominant_scene_region",
        "current_value": scene,
        "baseline_probability": round(probability, 4),
        "component_score": round(score, 4),
    }


def _legacy_mean_std_component(
    value: float,
    stats: Mapping[str, Any],
    direction: str,
    config: BaselineModelConfig,
) -> float:
    mean = float(stats["mean"])
    std = float(stats["std"])
    delta = mean - value if direction == "low" else value - mean
    if delta <= 0:
        return 0.0
    scale = max(std, abs(mean) * config.relative_scale_floor, config.absolute_scale_floor)
    return _range_score(delta / scale, config.robust_z_start, config.robust_z_full)


def _baseline_confidence(
    days: int,
    records: int,
    current: Mapping[str, Any],
    reference: Mapping[str, Any] | None,
    state: str,
    config: BaselineModelConfig,
) -> float:
    history_quality = float(reference.get("history_quality_mean", 0.0)) if reference else 0.0
    confidence = (
        0.30 * min(1.0, days / max(config.stable_history_days, 1))
        + 0.25 * min(1.0, records / max(config.min_history_records, 1))
        + 0.25 * _clamp(history_quality)
        + 0.20 * _clamp(float(current["quality_mean"]))
    )
    if state in {"none", "cold"}:
        confidence *= 0.45
    return round(_clamp(confidence), 4)


def _public_period(period: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "aggregation_period", "period_id", "period_start", "period_end", "timezone",
        "camera_profile_id", "record_count", "valid_monitoring_hours", "mean_gait_speed",
        "center_speed_cv", "hip_lateral_sway", "turn_instability_proxy",
        "mean_sit_stand_duration", "failed_attempts", "mean_stabilization_time",
        "near_fall_rate_per_hour", "nighttime_activity_rate_per_hour", "activity_volume",
        "dominant_scene_region", "scene_region_distribution", "quality_mean", "metric_status",
        "aggregation_version",
        "upstream_versions", "input_summary",
    )
    return {key: period.get(key) for key in keys}


def _public_reference(reference: Mapping[str, Any] | None) -> dict[str, Any]:
    if not reference:
        return {
            "person_id": None, "camera_profile_id": None, "history_record_count": 0,
            "history_period_count": 0, "history_day_count": 0, "window_start": None,
            "window_end": None, "metric_references": {}, "scene_region_distribution": {},
        }
    return {
        key: reference.get(key)
        for key in (
            "person_id", "camera_profile_id", "history_record_count", "history_period_count",
            "history_day_count", "window_start", "window_end", "metric_references",
            "scene_region_distribution", "history_quality_mean",
        )
    }


def _quality_score(record: Mapping[str, Any]) -> float:
    direct = _optional_number(record.get("baseline_quality"))
    if direct is not None:
        return round(_clamp(direct), 4)
    coverage = record.get("quality_coverage")
    if isinstance(coverage, Mapping):
        values = [
            _optional_number(coverage.get(key))
            for key in (
                "usable_frame_ratio", "mean_core_keypoint_quality", "gait_keypoint_coverage",
                "sit_stand_keypoint_coverage", "core_keypoint_coverage",
            )
        ]
        valid = [value for value in values if value is not None]
        if valid:
            return round(_clamp(_mean(valid)), 4)
    metric_quality = record.get("metric_quality")
    if isinstance(metric_quality, Mapping):
        values = [
            _optional_number(item.get("quality"))
            for item in metric_quality.values()
            if isinstance(item, Mapping)
        ]
        valid = [value for value in values if value is not None]
        if valid:
            return round(_clamp(_mean(valid)), 4)
    return 0.0


def _scene_distribution(record: Mapping[str, Any]) -> dict[str, float]:
    raw = record.get("scene_region_distribution")
    if isinstance(raw, Mapping):
        values = Counter({str(key): max(0.0, _number(value, 0.0)) for key, value in raw.items()})
        return _distribution(values)
    scene = record.get("dominant_scene_region") or record.get("scene_region")
    return {str(scene): 1.0} if scene else {}


def _strict_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _time_value(value: Any) -> float:
    parsed = _strict_datetime(value)
    return parsed.timestamp() if parsed is not None else math.inf


def _first_number(mapping: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _optional_number(mapping.get(key))
        if value is not None:
            return value
    return None


def _optional_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _number(value: Any, default: float) -> float:
    parsed = _optional_number(value)
    return default if parsed is None else parsed


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def _std(values: Iterable[float]) -> float:
    items = list(values)
    if len(items) < 2:
        return 0.0
    mean = _mean(items)
    return math.sqrt(sum((value - mean) ** 2 for value in items) / len(items))


def _percentile(values: Iterable[float], quantile: float) -> float:
    items = sorted(values)
    if not items:
        return 0.0
    position = _clamp(quantile) * (len(items) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return items[lower]
    fraction = position - lower
    return items[lower] * (1.0 - fraction) + items[upper] * fraction


def _distribution(counter: Mapping[str, float]) -> dict[str, float]:
    total = sum(float(value) for value in counter.values())
    if total <= 0:
        return {}
    return {key: round(float(value) / total, 4) for key, value in sorted(counter.items())}


def _range_score(value: float, start: float, full: float) -> float:
    if value <= start:
        return 0.0
    if full <= start:
        return 1.0
    return _clamp((value - start) / (full - start))


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _round(value: float | None) -> float | None:
    return round(value, 4) if value is not None else None
