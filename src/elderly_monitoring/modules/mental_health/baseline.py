from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import date, datetime, time, timedelta
from numbers import Real
from statistics import fmean, pstdev
from typing import Any, Iterable, Mapping

from elderly_monitoring.modules.mental_health.config import (
    BaselineConfig,
    MentalHealthConfig,
    load_mental_health_config,
)


_METRIC_SPECS = {
    "activity_volume": ("decrease", "behavior"),
    "active_ratio": ("decrease", "behavior"),
    "daytime_active_minutes": ("decrease", "behavior"),
    "weighted_daytime_activity": ("decrease", "behavior"),
    "bedroom_stay_ratio": ("increase", "behavior"),
    "outdoor_event_count": ("decrease", "behavior"),
    "outdoor_total_duration_minutes": ("decrease", "behavior"),
    "sleep_onset_latency": ("increase", "sleep"),
    "night_awakenings": ("increase", "sleep"),
    "sleep_efficiency": ("decrease", "sleep"),
    "night_leave_bed_count": ("increase", "sleep"),
    "night_leave_bed_minutes": ("increase", "sleep"),
    "sleep_midpoint_minute_of_day": ("two_sided", "sleep"),
    "sleep_midpoint_shift_minutes": ("increase", "sleep"),
    "nighttime_activity_ratio": ("two_sided", "behavior"),
    "scene_transition_count": ("two_sided", "behavior"),
    "call_count_7d": ("decrease", "social"),
    "answered_call_count_7d": ("decrease", "social"),
    "call_answer_rate_7d": ("decrease", "social"),
    "call_duration_minutes_7d": ("decrease", "social"),
    "active_call_count_7d": ("decrease", "social"),
    "missed_call_count_7d": ("increase", "social"),
    "mean_heart_rate": ("two_sided", "physiology"),
    "mean_breath_rate": ("two_sided", "physiology"),
    "heart_rate_std": ("increase", "physiology"),
    "breath_rate_std": ("increase", "physiology"),
    "heart_rate_range": ("increase", "physiology"),
    "breath_rate_range": ("increase", "physiology"),
    "heart_rate_outlier_ratio": ("increase", "physiology"),
    "breath_rate_outlier_ratio": ("increase", "physiology"),
    "abnormal_heart_rate_count": ("increase", "physiology"),
    "abnormal_breath_rate_count": ("increase", "physiology"),
    "gait_speed_norm_per_sec": ("decrease", "movement"),
    "sit_stand_duration_seconds": ("increase", "movement"),
    "turn_duration_seconds": ("increase", "movement"),
    "turn_stability_score": ("decrease", "movement"),
    "gait_cycle_stability_score": ("decrease", "movement"),
}
_GROUP_METRICS = {
    "activity_drop_score": (
        "activity_volume",
        "active_ratio",
        "daytime_active_minutes",
        "weighted_daytime_activity",
        "outdoor_event_count",
        "outdoor_total_duration_minutes",
    ),
    "sleep_disturbance_score": (
        "sleep_onset_latency",
        "night_awakenings",
        "sleep_efficiency",
        "night_leave_bed_count",
        "night_leave_bed_minutes",
        "sleep_midpoint_minute_of_day",
        "sleep_midpoint_shift_minutes",
    ),
    "social_withdrawal_score": (
        "call_count_7d",
        "answered_call_count_7d",
        "call_answer_rate_7d",
        "call_duration_minutes_7d",
        "active_call_count_7d",
        "missed_call_count_7d",
    ),
    "routine_irregularity_score": (
        "nighttime_activity_ratio",
        "scene_transition_count",
        "bedroom_stay_ratio",
    ),
    "night_physiology_score": (
        "mean_heart_rate",
        "mean_breath_rate",
        "heart_rate_std",
        "breath_rate_std",
        "heart_rate_range",
        "breath_rate_range",
        "heart_rate_outlier_ratio",
        "breath_rate_outlier_ratio",
        "abnormal_heart_rate_count",
        "abnormal_breath_rate_count",
    ),
    "movement_vitality_score": (
        "gait_speed_norm_per_sec",
        "sit_stand_duration_seconds",
        "turn_duration_seconds",
        "turn_stability_score",
        "gait_cycle_stability_score",
    ),
}
_OPTIONAL_FEATURES = (
    "social_withdrawal_score",
    "negative_affect_score",
    "self_report_risk_score",
)
_REJECTING_SOCIAL_FLAGS = {
    "invalid",
    "low_quality",
    "data_missing",
    "social_connection_score_unavailable",
}
_REJECTING_DAY_FLAGS = {
    "daily_quality_rejected",
    "invalid_daily_observation",
    "low_quality_day",
    "rejected",
}
_REJECTING_SLEEP_FLAGS = {
    "invalid",
    "low_quality",
    "rejected",
    "sleep_quality_rejected",
}


def score_daily_mental_health(
    history_records: Iterable[Mapping[str, Any]],
    current_records: Iterable[Mapping[str, Any]],
    *,
    config: MentalHealthConfig | None = None,
) -> list[dict[str, Any]]:
    """Score current person-days against prior qualified natural-day history."""
    mental_config = config or load_mental_health_config()
    history_days = _coalesce_days(history_records, source="history")
    current_days = _coalesce_days(current_records, source="current")
    if not current_days:
        raise ValueError("No current mental-health daily records were provided")

    current_keys = set(current_days)
    combined = {key: value for key, value in history_days.items() if key not in current_keys}
    combined.update(current_days)

    by_person: dict[str, dict[date, dict[str, Any]]] = defaultdict(dict)
    for (person_id, day), record in combined.items():
        by_person[person_id][day] = record

    outputs: list[dict[str, Any]] = []
    for (person_id, current_date), current in sorted(
        current_days.items(),
        key=lambda item: (item[0][1], item[0][0]),
    ):
        person_days = by_person[person_id]
        result = _score_day(current, current_date, person_days, mental_config)
        persistence, evidence_start = _persistent_abnormal_days(
            current_date,
            person_days,
            mental_config,
            current_result=result,
        )
        result["persistent_abnormal_days"] = persistence
        result["evidence_window"] = _evidence_window(
            person_days[evidence_start] if persistence > 0 else current,
            current,
            evidence_start if persistence > 0 else current_date,
            current_date,
            mental_config,
        )
        outputs.append(result)
    return outputs


def build_personal_baselines(
    history_records: Iterable[Mapping[str, Any]],
    *,
    evaluation_date: str | date | None = None,
    config: MentalHealthConfig | None = None,
) -> dict[str, dict[str, Any]]:
    """Build deterministic per-person reference summaries for inspection or reuse."""
    mental_config = config or load_mental_health_config()
    cutoff = _coerce_date(evaluation_date, "evaluation_date") if evaluation_date is not None else None
    days = _coalesce_days(history_records, source="history")
    by_person: dict[str, list[tuple[date, dict[str, Any]]]] = defaultdict(list)
    for (person_id, day), record in days.items():
        if cutoff is None or day < cutoff:
            by_person[person_id].append((day, record))

    outputs: dict[str, dict[str, Any]] = {}
    for person_id, dated_records in sorted(by_person.items()):
        selected = _eligible_window(dated_records, mental_config)
        outputs[person_id] = _reference_summary(person_id, selected, mental_config)
    return outputs


def _score_day(
    current: Mapping[str, Any],
    current_date: date,
    person_days: Mapping[date, Mapping[str, Any]],
    config: MentalHealthConfig,
) -> dict[str, Any]:
    person_id = str(current["person_id"])
    prior = [(day, record) for day, record in person_days.items() if day < current_date]
    selected = _eligible_window(prior, config)
    reference = _reference_summary(person_id, selected, config)
    current_flags = set(_record_flags(current))
    current_state = _source_state(current, config, current_flags)

    score_details: dict[str, dict[str, Any]] = {}
    group_scores: dict[str, float | None] = {}
    for group_name, metric_names in _GROUP_METRICS.items():
        metric_details: dict[str, Any] = {}
        for metric_name in metric_names:
            _, source = _METRIC_SPECS[metric_name]
            if not current_state[f"{source}_qualified"]:
                continue
            current_value = _metric_number(current, metric_name)
            stats = reference["metric_references"][metric_name]
            if current_value is None or stats["count"] < config.baseline.initial_days:
                continue
            detail = _scalar_deviation(
                metric_name,
                current_value,
                stats,
                config.baseline,
            )
            metric_details[metric_name] = detail
        group_scores[group_name] = (
            _round(max(detail["score"] for detail in metric_details.values()))
            if metric_details
            else None
        )
        if metric_details:
            score_details[group_name] = metric_details

    optional_values = {
        name: _optional_score(current, name) if _optional_score(current, name) is not None else group_scores.get(name)
        for name in _OPTIONAL_FEATURES
    }
    manual_emergency_flag = _optional_manual_flag(current)
    feature_values = {**group_scores, **optional_values}
    feature_coverage = _feature_coverage(feature_values, config)

    if not current_state["qualified"]:
        current_flags.add("insufficient_current_day_quality")
    if reference["eligible_history_days"] < config.baseline.initial_days:
        current_flags.add("insufficient_baseline_history")
    elif reference["eligible_history_days"] < config.baseline.stable_days:
        current_flags.add("stable_baseline_not_ready")
    current_flags.update(reference["quality_flags"])

    timestamp = current.get("end_time") or current.get("start_time")
    if timestamp is None:
        timestamp = datetime.combine(
            current_date,
            time.min,
            tzinfo=config.aggregation.timezone_info,
        ).isoformat()

    output: dict[str, Any] = {
        "person_id": person_id,
        "date": current_date.isoformat(),
        "timestamp": timestamp,
        "start_time": current.get("start_time"),
        "end_time": current.get("end_time"),
        "device_id": current.get("device_id"),
        "scene_region": current.get("scene_region"),
        **group_scores,
        **optional_values,
        "manual_emergency_flag": manual_emergency_flag,
        "persistent_abnormal_days": 0,
        "baseline_quality": reference["baseline_quality"],
        "feature_coverage": feature_coverage,
        "initial_baseline_ready": (
            reference["eligible_history_days"] >= config.baseline.initial_days
        ),
        "stable_baseline_ready": (
            reference["eligible_history_days"] >= config.baseline.stable_days
        ),
        "risk_factor_details": score_details,
        "baseline_window": {
            "start_date": reference["start_date"],
            "end_date": reference["end_date"],
            "eligible_history_days": reference["eligible_history_days"],
        },
        "data_quality_flags": sorted(current_flags),
    }
    return output


def _reference_summary(
    person_id: str,
    selected: list[tuple[date, Mapping[str, Any]]],
    config: MentalHealthConfig,
) -> dict[str, Any]:
    metric_values: dict[str, list[float]] = {name: [] for name in _METRIC_SPECS}
    quality_values: list[float] = []
    quality_flags: set[str] = set()
    for _, record in selected:
        state_flags: set[str] = set()
        state = _source_state(record, config, state_flags)
        quality_values.append(state["quality"])
        quality_flags.update(state_flags)
        for metric_name, (_, source) in _METRIC_SPECS.items():
            if not state[f"{source}_qualified"]:
                continue
            value = _metric_number(record, metric_name)
            if value is not None:
                metric_values[metric_name].append(value)

    eligible_days = len(selected)
    mean_quality = fmean(quality_values) if quality_values else 0.0
    baseline_quality = _clamp(
        (eligible_days / config.baseline.stable_days) * mean_quality
    )
    return {
        "person_id": person_id,
        "eligible_history_days": eligible_days,
        "start_date": selected[0][0].isoformat() if selected else None,
        "end_date": selected[-1][0].isoformat() if selected else None,
        "baseline_quality": _round(baseline_quality),
        "mean_history_quality": _round(mean_quality),
        "quality_flags": sorted(quality_flags),
        "metric_references": {
            metric_name: _metric_stats(values, config.baseline)
            for metric_name, values in metric_values.items()
        },
    }


def _eligible_window(
    dated_records: Iterable[tuple[date, Mapping[str, Any]]],
    config: MentalHealthConfig,
) -> list[tuple[date, Mapping[str, Any]]]:
    eligible: list[tuple[date, Mapping[str, Any]]] = []
    for day, record in sorted(dated_records, key=lambda item: item[0]):
        if not _source_state(record, config, set())["qualified"]:
            continue
        if _exclude_from_baseline(record, eligible, config):
            continue
        eligible.append((day, record))
    return eligible[-config.baseline.max_window_days :]


def _exclude_from_baseline(
    record: Mapping[str, Any],
    accepted_history: list[tuple[date, Mapping[str, Any]]],
    config: MentalHealthConfig,
) -> bool:
    flags = set(_record_flags(record))
    if flags.intersection(
        {
            "baseline_excluded",
            "baseline_excluded_abnormal",
            "abnormal_day",
            "manual_exclude_from_baseline",
        }
    ):
        return True
    if record.get("exclude_from_baseline") is True:
        return True
    if _optional_manual_flag(record) is True:
        return True
    for field in _OPTIONAL_FEATURES:
        value = _optional_score(record, field)
        if value is not None and value >= config.baseline.abnormal_score_threshold:
            return True
    if len(accepted_history) < config.baseline.initial_days:
        return False

    state = _source_state(record, config, set())
    prior_records = [item[1] for item in accepted_history]
    for group_name, metric_names in _GROUP_METRICS.items():
        scores: list[float] = []
        for metric_name in metric_names:
            _, source = _METRIC_SPECS[metric_name]
            if not state[f"{source}_qualified"]:
                continue
            current_value = _metric_number(record, metric_name)
            if current_value is None:
                continue
            values = [
                value
                for prior in prior_records
                if _source_state(prior, config, set())[f"{source}_qualified"]
                for value in [_metric_number(prior, metric_name)]
                if value is not None
            ]
            if len(values) < config.baseline.initial_days:
                continue
            detail = _scalar_deviation(
                metric_name,
                current_value,
                _metric_stats(values, config.baseline),
                config.baseline,
            )
            scores.append(float(detail["score"]))
        if scores and max(scores) >= config.baseline.abnormal_score_threshold:
            return True
    return False


def _source_state(
    record: Mapping[str, Any],
    config: MentalHealthConfig,
    generated_flags: set[str],
) -> dict[str, Any]:
    flags = set(_record_flags(record))
    behavior_values = [
        _metric_number(record, name)
        for name, (_, source) in _METRIC_SPECS.items()
        if source == "behavior"
    ]
    coverage = _bounded_number(record, "observation_coverage", 0.0, 1.0)
    valid_seconds = _nonnegative_number(record, "valid_observation_seconds")
    valid_daytime_minutes = _nonnegative_number(record, "valid_daytime_detection_minutes")
    behavior_qualified = (
        any(value is not None for value in behavior_values)
        and (
            (coverage is not None and coverage > 0.0)
            or (valid_daytime_minutes is not None and valid_daytime_minutes > 0.0)
        )
        and (valid_seconds is None or valid_seconds > 0.0)
        and not flags.intersection(_REJECTING_DAY_FLAGS)
    )

    sleep_values = [
        _metric_number(record, name)
        for name, (_, source) in _METRIC_SPECS.items()
        if source == "sleep"
    ]
    sleep_qualified = (
        any(value is not None for value in sleep_values)
        and not flags.intersection(_REJECTING_SLEEP_FLAGS)
    )

    physiology_values = [
        _metric_number(record, name)
        for name, (_, source) in _METRIC_SPECS.items()
        if source == "physiology"
    ]
    physiology_qualified = (
        any(value is not None for value in physiology_values)
        and not flags.intersection(_REJECTING_SLEEP_FLAGS)
    )

    social_values = [
        _metric_number(record, name)
        for name, (_, source) in _METRIC_SPECS.items()
        if source == "social"
    ]
    social_qualified = (
        any(value is not None for value in social_values)
        and not flags.intersection(_REJECTING_SOCIAL_FLAGS)
    )

    movement_values = [
        _metric_number(record, name)
        for name, (_, source) in _METRIC_SPECS.items()
        if source == "movement"
    ]
    movement_quality = _bounded_number(record, "pose_quality_coverage", 0.0, 1.0)
    if movement_quality is None:
        movement_quality = _bounded_number(record, "pose_quality_score", 0.0, 1.0)
    movement_qualified = (
        any(value is not None for value in movement_values)
        and not flags.intersection(_REJECTING_DAY_FLAGS)
        and not flags.intersection({"movement_vitality_insufficient_data"})
    )

    if behavior_qualified:
        if coverage is None:
            quality = config.baseline.missing_quality_default
            generated_flags.add("missing_source_quality")
        else:
            quality = coverage
    elif movement_qualified:
        if movement_quality is None:
            quality = config.baseline.missing_quality_default
            generated_flags.add("missing_movement_quality")
        else:
            quality = movement_quality
    elif sleep_qualified:
        quality_score = _bounded_number(record, "quality_score", 0.0, 1.0)
        if quality_score is None:
            quality = config.baseline.missing_quality_default
            generated_flags.add("missing_source_quality")
        else:
            quality = quality_score
    elif physiology_qualified:
        quality_score = _bounded_number(record, "quality_score", 0.0, 1.0)
        if quality_score is None:
            quality = config.baseline.missing_quality_default
            generated_flags.add("missing_source_quality")
        else:
            quality = quality_score
    elif social_qualified:
        quality_score = _bounded_number(record, "quality_score", 0.0, 1.0)
        if quality_score is None:
            quality = config.baseline.missing_quality_default
            generated_flags.add("missing_source_quality")
        else:
            quality = quality_score
    else:
        quality = 0.0
    return {
        "behavior_qualified": behavior_qualified,
        "sleep_qualified": sleep_qualified,
        "physiology_qualified": physiology_qualified,
        "social_qualified": social_qualified,
        "movement_qualified": movement_qualified,
        "qualified": (
            behavior_qualified
            or sleep_qualified
            or physiology_qualified
            or social_qualified
            or movement_qualified
        ),
        "quality": float(quality),
    }


def _metric_stats(values: list[float], config: BaselineConfig) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "lower_quantile": None,
            "upper_quantile": None,
        }
    return {
        "count": len(values),
        "mean": fmean(values),
        "std": pstdev(values),
        "lower_quantile": _quantile(values, config.lower_quantile),
        "upper_quantile": _quantile(values, config.upper_quantile),
    }


def _scalar_deviation(
    metric_name: str,
    current: float,
    stats: Mapping[str, Any],
    config: BaselineConfig,
) -> dict[str, Any]:
    direction = _METRIC_SPECS[metric_name][0]
    mean = float(stats["mean"])
    std = float(stats["std"])
    lower = float(stats["lower_quantile"])
    upper = float(stats["upper_quantile"])
    if direction == "decrease":
        risk_delta = max(0.0, mean - current)
        quantile_distance = max(0.0, lower - current)
    elif direction == "increase":
        risk_delta = max(0.0, current - mean)
        quantile_distance = max(0.0, current - upper)
    else:
        risk_delta = abs(current - mean)
        quantile_distance = max(0.0, lower - current, current - upper)

    scale_floor = max(
        std,
        abs(mean) * config.zero_variance_relative_floor,
        config.zero_variance_absolute_floor,
    )
    standardized = _clamp(risk_delta / (scale_floor * config.z_score_full_scale))
    relative = _clamp(
        (risk_delta / max(abs(mean), config.zero_variance_absolute_floor))
        / config.relative_change_full_scale
    )
    quantile_component = _clamp(quantile_distance / scale_floor)
    score = max(standardized, relative, quantile_component)
    return {
        "direction": direction,
        "current_value": _round(current),
        "baseline_count": int(stats["count"]),
        "baseline_mean": _round(mean),
        "baseline_std": _round(std),
        "lower_quantile": _round(lower),
        "upper_quantile": _round(upper),
        "risk_delta": _round(risk_delta),
        "scale_floor": _round(scale_floor),
        "standardized_component": _round(standardized),
        "relative_component": _round(relative),
        "quantile_component": _round(quantile_component),
        "score": _round(score),
    }


def _persistent_abnormal_days(
    current_date: date,
    person_days: Mapping[date, Mapping[str, Any]],
    config: MentalHealthConfig,
    *,
    current_result: Mapping[str, Any],
) -> tuple[int, date]:
    count = 0
    cursor = current_date
    oldest = current_date
    while cursor in person_days:
        record = person_days[cursor]
        if not _source_state(record, config, set())["qualified"]:
            break
        result = (
            current_result
            if cursor == current_date
            else _score_day(record, cursor, person_days, config)
        )
        scores = [
            result.get(name)
            for name in _GROUP_METRICS
            if result.get(name) is not None
        ]
        if not scores or max(float(score) for score in scores) < config.baseline.abnormal_score_threshold:
            break
        count += 1
        oldest = cursor
        cursor -= timedelta(days=1)
    return count, oldest


def _feature_coverage(
    features: Mapping[str, float | None],
    config: MentalHealthConfig,
) -> float:
    expected = config.scoring.coverage_expected_features
    denominator = sum(config.scoring.weights[name] for name in expected)
    available = sum(
        config.scoring.weights[name]
        for name in expected
        if features.get(name) is not None
    )
    return _round(available / denominator)


def _evidence_window(
    start_record: Mapping[str, Any],
    end_record: Mapping[str, Any],
    start_date: date,
    end_date: date,
    config: MentalHealthConfig,
) -> dict[str, Any]:
    start = _record_boundary(start_record, start_date, "start_time", config, end=False)
    end = _record_boundary(end_record, end_date, "end_time", config, end=True)
    return {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "start_time": _round(start.timestamp()),
        "end_time": _round(end.timestamp()),
    }


def _record_boundary(
    record: Mapping[str, Any],
    day: date,
    field: str,
    config: MentalHealthConfig,
    *,
    end: bool,
) -> datetime:
    value = record.get(field)
    if value is not None:
        return _aware_datetime(value, field).astimezone(config.aggregation.timezone_info)
    boundary_time = time.max if end else time.min
    return datetime.combine(day, boundary_time, tzinfo=config.aggregation.timezone_info)


def _coalesce_days(
    records: Iterable[Mapping[str, Any]],
    *,
    source: str,
) -> dict[tuple[str, date], dict[str, Any]]:
    prepared: list[tuple[str, date, str, Mapping[str, Any]]] = []
    for index, record in enumerate(records, start=1):
        if not isinstance(record, Mapping):
            raise ValueError(f"{source} daily record {index}: expected an object")
        person_id = record.get("person_id")
        if not isinstance(person_id, str) or not person_id.strip():
            raise ValueError(
                f"{source} daily record {index}, field 'person_id': "
                "must be a non-empty stable business person ID"
            )
        day = _coerce_date(record.get("date"), f"{source} daily record {index}, field 'date'")
        canonical = json.dumps(
            record,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        prepared.append((person_id.strip(), day, canonical, record))

    groups: dict[tuple[str, date], list[tuple[str, Mapping[str, Any]]]] = defaultdict(list)
    for person_id, day, canonical, record in prepared:
        groups[(person_id, day)].append((canonical, record))

    outputs: dict[tuple[str, date], dict[str, Any]] = {}
    for key, items in groups.items():
        merged: dict[str, Any] = {
            "person_id": key[0],
            "date": key[1].isoformat(),
        }
        flags: set[str] = set()
        quality_flags: set[str] = set()
        for _, record in sorted(items, key=lambda item: item[0]):
            flags.update(_string_list(record.get("data_quality_flags")))
            quality_flags.update(_string_list(record.get("quality_flags")))
            for field, value in record.items():
                if field in {"person_id", "date", "data_quality_flags", "quality_flags"}:
                    continue
                if value is not None and merged.get(field) is None:
                    merged[field] = value
        merged["data_quality_flags"] = sorted(flags)
        merged["quality_flags"] = sorted(quality_flags)
        outputs[key] = merged
    return outputs


def _record_flags(record: Mapping[str, Any]) -> list[str]:
    return _string_list(record.get("data_quality_flags")) + _string_list(
        record.get("quality_flags")
    )


def _metric_number(record: Mapping[str, Any], field: str) -> float | None:
    ranges = {
        "activity_volume": (0.0, None),
        "active_ratio": (0.0, 1.0),
        "daytime_active_minutes": (0.0, 1440.0),
        "weighted_daytime_activity": (0.0, None),
        "bedroom_stay_ratio": (0.0, 1.0),
        "outdoor_event_count": (0.0, None),
        "outdoor_total_duration_minutes": (0.0, 1440.0),
        "sleep_onset_latency": (0.0, 720.0),
        "night_awakenings": (0.0, 100.0),
        "sleep_efficiency": (0.0, 1.0),
        "night_leave_bed_count": (0.0, 100.0),
        "night_leave_bed_minutes": (0.0, 1440.0),
        "sleep_midpoint_minute_of_day": (0.0, 1440.0),
        "sleep_midpoint_shift_minutes": (0.0, 720.0),
        "nighttime_activity_ratio": (0.0, 1.0),
        "scene_transition_count": (0.0, None),
        "call_count_7d": (0.0, None),
        "answered_call_count_7d": (0.0, None),
        "call_answer_rate_7d": (0.0, 1.0),
        "call_duration_minutes_7d": (0.0, None),
        "active_call_count_7d": (0.0, None),
        "missed_call_count_7d": (0.0, None),
        "mean_heart_rate": (20.0, 240.0),
        "mean_breath_rate": (4.0, 60.0),
        "heart_rate_std": (0.0, 120.0),
        "breath_rate_std": (0.0, 40.0),
        "heart_rate_range": (0.0, 220.0),
        "breath_rate_range": (0.0, 56.0),
        "heart_rate_outlier_ratio": (0.0, 1.0),
        "breath_rate_outlier_ratio": (0.0, 1.0),
        "abnormal_heart_rate_count": (0.0, None),
        "abnormal_breath_rate_count": (0.0, None),
        "gait_speed_norm_per_sec": (0.0, None),
        "sit_stand_duration_seconds": (0.0, 120.0),
        "turn_duration_seconds": (0.0, 120.0),
        "turn_stability_score": (0.0, 1.0),
        "gait_cycle_stability_score": (0.0, 1.0),
    }
    minimum, maximum = ranges[field]
    number = _bounded_number(record, field, minimum, maximum)
    if number is not None and field in {
        "night_awakenings",
        "night_leave_bed_count",
        "scene_transition_count",
        "outdoor_event_count",
        "call_count_7d",
        "answered_call_count_7d",
        "active_call_count_7d",
        "missed_call_count_7d",
        "abnormal_heart_rate_count",
        "abnormal_breath_rate_count",
    }:
        raw = record.get(field)
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise _record_field_error(record, field, "must be an integer")
    return number


def _optional_score(record: Mapping[str, Any], field: str) -> float | None:
    return _bounded_number(record, field, 0.0, 1.0)


def _optional_manual_flag(record: Mapping[str, Any]) -> bool | None:
    value = record.get("manual_emergency_flag")
    if value is None:
        return None
    if not isinstance(value, bool):
        raise _record_field_error(record, "manual_emergency_flag", "must be a boolean")
    return value


def _bounded_number(
    record: Mapping[str, Any],
    field: str,
    minimum: float,
    maximum: float | None,
) -> float | None:
    value = record.get(field)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise _record_field_error(record, field, "must be a finite number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise _record_field_error(record, field, "must be a finite number") from exc
    if not math.isfinite(number):
        raise _record_field_error(record, field, "must be a finite number")
    if number < minimum or (maximum is not None and number > maximum):
        interval = f"[{minimum}, {maximum}]" if maximum is not None else f">= {minimum}"
        raise _record_field_error(record, field, f"must be in {interval}")
    return number


def _nonnegative_number(record: Mapping[str, Any], field: str) -> float | None:
    return _bounded_number(record, field, 0.0, None)


def _record_field_error(record: Mapping[str, Any], field: str, detail: str) -> ValueError:
    return ValueError(
        f"mental-health daily record person={record.get('person_id')!r} "
        f"date={record.get('date')!r}, field '{field}': {detail}"
    )


def _coerce_date(value: Any, path: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not isinstance(value, str) or len(value) != 10:
        raise ValueError(f"{path}: must use YYYY-MM-DD format")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{path}: must be a valid YYYY-MM-DD date") from exc
    if parsed.isoformat() != value:
        raise ValueError(f"{path}: must use YYYY-MM-DD format")
    return parsed


def _aware_datetime(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"mental-health daily field '{field}' must be an ISO-8601 timestamp")
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(
            f"mental-health daily field '{field}' must be an ISO-8601 timestamp"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"mental-health daily field '{field}' must include a timezone")
    return parsed


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    fraction = position - lower_index
    return ordered[lower_index] + (ordered[upper_index] - ordered[lower_index]) * fraction


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _round(value: float) -> float:
    return round(float(value), 4)
