from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime
from numbers import Real
from statistics import fmean, pstdev
from typing import Any, Iterable, Mapping

from elderly_monitoring.modules.mental_health.config import (
    AggregationConfig,
    load_aggregation_config,
)


BASELINE_REJECTING_QUALITIES = {
    "invalid",
    "low_confidence",
    "low_quality",
    "missing_core_physiology_metrics",
    "rejected",
}


@dataclass(frozen=True)
class NightPhysiologyScore:
    night_physiology_score: float | None
    night_physiology_domain_score: float | None
    baseline_confidence: str
    baseline_quality: float
    persistent_abnormal_days: int
    factors: tuple[str, ...]
    risk_factor_details: dict[str, Any]
    initial_baseline_ready: bool
    stable_baseline_ready: bool


def normalize_night_physiology_daily(
    records: Iterable[Mapping[str, Any]],
    *,
    person_id: str,
    device_id: str | None = None,
    device_serial: str | None = None,
) -> list[dict[str, Any]]:
    """Normalize sleep-device heart-rate and respiration summaries to daily fields."""
    normalized = [
        _normalize_daily_feature(
            record,
            person_id=person_id,
            device_id=device_id,
            device_serial=device_serial,
        )
        for record in records
    ]
    return sorted(normalized, key=lambda item: (item["date"], item.get("device_serial") or ""))


def score_night_physiology_day(
    current: Mapping[str, Any],
    history: Iterable[Mapping[str, Any]],
    *,
    config: AggregationConfig | None = None,
) -> NightPhysiologyScore:
    """Score auxiliary nighttime heart-rate/respiration trends against personal history."""
    aggregation_config = config or load_aggregation_config()
    person_id = str(current.get("person_id") or "").strip()
    current_day = _parse_date(current.get("date"), "current.date")
    history_days = [
        _normalize_daily_feature(
            item,
            person_id=str(item.get("person_id") or person_id),
            device_id=None,
            device_serial=None,
        )
        for item in history
        if str(item.get("person_id") or "").strip() == person_id
        and _parse_date(item.get("date"), "history.date") < current_day
        and _baseline_eligible(item)
    ][-14:]
    baseline_quality = _baseline_quality(history_days)
    baseline_confidence = _baseline_confidence(history_days)
    initial_ready = len(history_days) >= 3
    stable_ready = len(history_days) >= 7
    if not initial_ready:
        return NightPhysiologyScore(
            night_physiology_score=None,
            night_physiology_domain_score=None,
            baseline_confidence=baseline_confidence,
            baseline_quality=baseline_quality,
            persistent_abnormal_days=0,
            factors=("insufficient_night_physiology_baseline",),
            risk_factor_details={},
            initial_baseline_ready=False,
            stable_baseline_ready=False,
        )

    normalized_current = _normalize_daily_feature(
        current,
        person_id=person_id,
        device_id=None,
        device_serial=None,
    )
    metric_specs = {
        "mean_heart_rate": "two_sided",
        "mean_breath_rate": "two_sided",
        "heart_rate_std": "increase",
        "breath_rate_std": "increase",
        "heart_rate_range": "increase",
        "breath_rate_range": "increase",
        "heart_rate_outlier_ratio": "increase",
        "breath_rate_outlier_ratio": "increase",
        "abnormal_heart_rate_count": "increase",
        "abnormal_breath_rate_count": "increase",
    }
    details: dict[str, Any] = {}
    for metric, direction in metric_specs.items():
        current_value = _optional_number(normalized_current.get(metric))
        values = [_optional_number(row.get(metric)) for row in history_days]
        values = [value for value in values if value is not None]
        if current_value is None or len(values) < 3:
            continue
        detail = _metric_deviation(current_value, values, direction)
        if detail["score"] > 0:
            details[metric] = detail

    score = max((detail["score"] for detail in details.values()), default=0.0)
    factors = _physiology_factors(details)
    if not factors:
        factors = ["night_physiology_within_personal_range"]

    persistence = 1 if score >= 0.6 else 0
    if persistence:
        previous = sorted(history_days, key=lambda item: str(item.get("date") or ""), reverse=True)
        for row in previous:
            row_day = _parse_date(row.get("date"), "history.date")
            previous_history = [
                item
                for item in history_days
                if _parse_date(item.get("date"), "history.date") < row_day
            ]
            previous_score = score_night_physiology_day(
                row,
                previous_history,
                config=aggregation_config,
            )
            if (
                previous_score.night_physiology_score is None
                or previous_score.night_physiology_score < 0.6
            ):
                break
            persistence += 1

    return NightPhysiologyScore(
        night_physiology_score=round(score, 4),
        night_physiology_domain_score=round(score * 100.0, 1),
        baseline_confidence=baseline_confidence,
        baseline_quality=baseline_quality,
        persistent_abnormal_days=persistence,
        factors=tuple(factors),
        risk_factor_details=details,
        initial_baseline_ready=initial_ready,
        stable_baseline_ready=stable_ready,
    )


def build_night_physiology_result(
    *,
    person_id: str,
    daily_features: Iterable[Mapping[str, Any]] = (),
    history_daily_features: Iterable[Mapping[str, Any]] = (),
    requested_date: date | None = None,
    device_id: str | None = None,
    device_serial: str | None = None,
    config: AggregationConfig | None = None,
) -> dict[str, Any]:
    aggregation_config = config or load_aggregation_config()
    daily = normalize_night_physiology_daily(
        daily_features,
        person_id=person_id,
        device_id=device_id,
        device_serial=device_serial,
    )
    if requested_date is not None:
        daily = [item for item in daily if item["date"] == requested_date.isoformat()]
    history = normalize_night_physiology_daily(
        history_daily_features,
        person_id=person_id,
        device_id=device_id,
        device_serial=device_serial,
    )

    scored: list[dict[str, Any]] = []
    for item in daily:
        score = score_night_physiology_day(item, history, config=aggregation_config)
        enriched = dict(item)
        enriched.update(
            {
                "night_physiology_score": score.night_physiology_score,
                "night_physiology_domain_score": score.night_physiology_domain_score,
                "baseline_confidence": score.baseline_confidence,
                "baseline_quality": score.baseline_quality,
                "persistent_abnormal_days": score.persistent_abnormal_days,
                "night_physiology_factors": list(score.factors),
                "night_physiology_details": score.risk_factor_details,
                "initial_baseline_ready": score.initial_baseline_ready,
                "stable_baseline_ready": score.stable_baseline_ready,
            }
        )
        scored.append(enriched)
        history.append(enriched)

    return {
        "schema_version": "night_physiology_service_v1",
        "model_version": "night-physiology-rulecard-v1",
        "person_id": person_id,
        "requested_date": requested_date.isoformat() if requested_date else None,
        "daily_features": scored,
        "quality_flags": _quality_flags(scored),
        "medical_disclaimer": "nighttime physiology trend only; not a medical diagnosis",
    }


def _normalize_daily_feature(
    record: Mapping[str, Any],
    *,
    person_id: str,
    device_id: str | None,
    device_serial: str | None,
) -> dict[str, Any]:
    item = dict(record)
    item["person_id"] = str(item.get("person_id") or person_id)
    item["date"] = _parse_date(item.get("date"), "daily.date").isoformat()
    item.setdefault("device_id", device_id)
    item.setdefault("device_serial", device_serial)

    _alias_number(item, "mean_heart_rate", "meanHeartRate", "heart_rate_mean", "avg_heart_rate", "night_heart_rate_mean")
    _alias_number(item, "mean_breath_rate", "meanBreathRate", "breath_rate_mean", "avg_breath_rate", "night_breath_rate_mean")
    _alias_number(item, "heart_rate_std", "heartRateStd", "heart_rate_sd", "night_heart_rate_std")
    _alias_number(item, "breath_rate_std", "breathRateStd", "breath_rate_sd", "night_breath_rate_std")
    _alias_number(item, "heart_rate_min", "min_heart_rate", "heartRateMin")
    _alias_number(item, "heart_rate_max", "max_heart_rate", "heartRateMax")
    _alias_number(item, "breath_rate_min", "min_breath_rate", "breathRateMin")
    _alias_number(item, "breath_rate_max", "max_breath_rate", "breathRateMax")
    _alias_number(item, "heart_rate_outlier_ratio", "heartRateOutlierRatio")
    _alias_number(item, "breath_rate_outlier_ratio", "breathRateOutlierRatio")

    high_count = _optional_number(_first_present(item, "highCount", "high_heart_rate_count"))
    low_count = _optional_number(_first_present(item, "lowCount", "low_heart_rate_count"))
    irregular_count = _optional_number(_first_present(item, "lowHighCount", "irregular_heart_rate_count"))
    if item.get("abnormal_heart_rate_count") is None:
        abnormal_heart = sum(value for value in (high_count, low_count, irregular_count) if value is not None)
        if abnormal_heart > 0:
            item["abnormal_heart_rate_count"] = int(abnormal_heart) if abnormal_heart.is_integer() else abnormal_heart
    _alias_number(item, "abnormal_heart_rate_count", "abnormalHeartRateCount", "heart_anomaly_count")
    _alias_number(item, "abnormal_breath_rate_count", "abnormalBreathRateCount", "breath_anomaly_count")

    _derive_range(item, "heart_rate")
    _derive_range(item, "breath_rate")
    _derive_outlier_ratio(item, "heart_rate")
    _derive_outlier_ratio(item, "breath_rate")

    item.setdefault("data_quality", "valid")
    flags = _string_list(item.get("quality_flags")) + _string_list(item.get("data_quality_flags"))
    if item.get("mean_heart_rate") is None and item.get("mean_breath_rate") is None:
        flags.append("missing_core_physiology_metrics")
    item["quality_flags"] = _dedupe(flags)
    item.setdefault("quality_score", 1.0 if item["data_quality"] == "valid" else 0.0)
    item["baseline_eligible"] = _baseline_eligible(item)
    return item


def _metric_deviation(current: float, values: list[float], direction: str) -> dict[str, Any]:
    mean = fmean(values)
    std = pstdev(values)
    scale = max(std, abs(mean) * 0.05, 0.05)
    if direction == "increase":
        delta = max(0.0, current - mean)
    elif direction == "decrease":
        delta = max(0.0, mean - current)
    else:
        delta = abs(current - mean)
    standardized = min(delta / (scale * 2.0), 1.0)
    relative = min((delta / max(abs(mean), 0.05)) / 0.50, 1.0)
    return {
        "direction": direction,
        "current_value": round(current, 4),
        "baseline_count": len(values),
        "baseline_mean": round(mean, 4),
        "baseline_std": round(std, 4),
        "risk_delta": round(delta, 4),
        "score": round(max(standardized, relative), 4),
    }


def _physiology_factors(details: Mapping[str, Mapping[str, Any]]) -> list[str]:
    labels = {
        "mean_heart_rate": "nighttime_heart_rate_shift",
        "mean_breath_rate": "nighttime_breath_rate_shift",
        "heart_rate_std": "nighttime_heart_rate_variability_increase",
        "breath_rate_std": "nighttime_breath_rate_variability_increase",
        "heart_rate_range": "nighttime_heart_rate_variability_increase",
        "breath_rate_range": "nighttime_breath_rate_variability_increase",
        "heart_rate_outlier_ratio": "nighttime_heart_rate_outlier_increase",
        "breath_rate_outlier_ratio": "nighttime_breath_rate_outlier_increase",
        "abnormal_heart_rate_count": "nighttime_heart_rate_abnormal_events_increase",
        "abnormal_breath_rate_count": "nighttime_breath_rate_abnormal_events_increase",
    }
    return _dedupe(
        labels[name]
        for name, detail in sorted(details.items(), key=lambda item: item[1]["score"], reverse=True)
        if detail["score"] >= 0.35
    )


def _baseline_eligible(record: Mapping[str, Any]) -> bool:
    if record.get("baseline_eligible") is False:
        return False
    quality = str(record.get("data_quality") or "valid")
    flags = set(str(flag) for flag in record.get("quality_flags") or [])
    return quality not in BASELINE_REJECTING_QUALITIES and not flags.intersection(BASELINE_REJECTING_QUALITIES)


def _baseline_quality(history: list[Mapping[str, Any]]) -> float:
    if not history:
        return 0.0
    qualities = [_optional_number(item.get("quality_score")) for item in history]
    quality = fmean([value for value in qualities if value is not None] or [1.0])
    return round(min(len(history) / 7.0, 1.0) * quality, 4)


def _baseline_confidence(history: list[Mapping[str, Any]]) -> str:
    if len(history) >= 14:
        return "high"
    if len(history) >= 7:
        return "medium"
    if len(history) >= 3:
        return "low"
    return "insufficient"


def _quality_flags(records: Iterable[Mapping[str, Any]]) -> list[str]:
    flags: list[str] = []
    for record in records:
        flags.extend(str(flag) for flag in record.get("quality_flags") or [])
        if record.get("night_physiology_score") is None:
            flags.append("night_physiology_score_unavailable")
    return _dedupe(flags)


def _alias_number(item: dict[str, Any], target: str, *aliases: str) -> None:
    if item.get(target) is not None:
        item[target] = _optional_number(item.get(target))
        return
    for alias in aliases:
        value = _optional_number(item.get(alias))
        if value is not None:
            item[target] = value
            return


def _derive_range(item: dict[str, Any], prefix: str) -> None:
    target = f"{prefix}_range"
    if item.get(target) is not None:
        return
    minimum = _optional_number(item.get(f"{prefix}_min"))
    maximum = _optional_number(item.get(f"{prefix}_max"))
    if minimum is not None and maximum is not None and maximum >= minimum:
        item[target] = round(maximum - minimum, 4)


def _derive_outlier_ratio(item: dict[str, Any], prefix: str) -> None:
    target = f"{prefix}_outlier_ratio"
    if item.get(target) is not None:
        return
    count = _optional_number(item.get(f"abnormal_{prefix}_count"))
    total = _optional_number(item.get(f"{prefix}_measurement_count"))
    if count is not None and total is not None and total > 0:
        item[target] = round(min(max(count / total, 0.0), 1.0), 4)


def _first_present(item: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in item and item[key] is not None:
            return item[key]
    return None


def _optional_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) else None


def _parse_date(value: Any, path: str) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if not isinstance(value, str) or len(value) != 10:
        raise ValueError(f"{path} must use YYYY-MM-DD format")
    parsed = date.fromisoformat(value)
    if parsed.isoformat() != value:
        raise ValueError(f"{path} must use YYYY-MM-DD format")
    return parsed


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _dedupe(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))
