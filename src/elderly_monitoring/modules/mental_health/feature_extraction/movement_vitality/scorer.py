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
    "insufficient_gait_quality",
    "insufficient_movement_vitality_baseline",
    "insufficient_sit_stand_quality",
    "insufficient_turn_quality",
    "low_quality",
    "movement_vitality_insufficient_data",
    "rejected",
}

METRIC_SPECS = {
    "gait_speed_norm_per_sec": ("decrease", 0.35),
    "sit_stand_duration_seconds": ("increase", 0.25),
    "turn_duration_seconds": ("increase", 0.10),
    "turn_stability_score": ("decrease", 0.10),
    "gait_cycle_stability_score": ("decrease", 0.20),
}


@dataclass(frozen=True)
class MovementVitalityScore:
    movement_vitality_score: float | None
    movement_vitality_domain_score: float | None
    baseline_confidence: str
    baseline_quality: float
    persistent_abnormal_days: int
    factors: tuple[str, ...]
    risk_factor_details: dict[str, Any]
    initial_baseline_ready: bool
    stable_baseline_ready: bool


def normalize_movement_vitality_daily(
    records: Iterable[Mapping[str, Any]],
    *,
    person_id: str | None = None,
    device_id: str | None = None,
) -> list[dict[str, Any]]:
    """Normalize day-level gait/transfer fields to movement-vitality fields."""
    normalized = [
        _normalize_daily_feature(record, person_id=person_id, device_id=device_id)
        for record in records
    ]
    return sorted(normalized, key=lambda item: (item["date"], item["person_id"], item.get("device_id") or ""))


def score_movement_vitality_day(
    current: Mapping[str, Any],
    history: Iterable[Mapping[str, Any]],
    *,
    config: AggregationConfig | None = None,
) -> MovementVitalityScore:
    """Score movement-vitality decline against the person's own prior days."""
    aggregation_config = config or load_aggregation_config()
    person_id = str(current.get("person_id") or "").strip()
    current_day = _parse_date(current.get("date"), "current.date")
    history_days = [
        _normalize_daily_feature(item, person_id=str(item.get("person_id") or person_id), device_id=None)
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
        return MovementVitalityScore(
            movement_vitality_score=None,
            movement_vitality_domain_score=None,
            baseline_confidence=baseline_confidence,
            baseline_quality=baseline_quality,
            persistent_abnormal_days=0,
            factors=("insufficient_movement_vitality_baseline",),
            risk_factor_details={},
            initial_baseline_ready=False,
            stable_baseline_ready=False,
        )

    normalized_current = _normalize_daily_feature(current, person_id=person_id, device_id=None)
    details: dict[str, Any] = {}
    weighted_scores: list[tuple[float, float]] = []
    for metric, (direction, weight) in METRIC_SPECS.items():
        current_value = _optional_number(normalized_current.get(metric))
        values = [_optional_number(row.get(metric)) for row in history_days]
        values = [value for value in values if value is not None]
        if current_value is None or len(values) < 3:
            continue
        detail = _metric_deviation(current_value, values, direction)
        if detail["score"] > 0:
            details[metric] = detail
        weighted_scores.append((detail["score"], weight))

    if not weighted_scores:
        return MovementVitalityScore(
            movement_vitality_score=None,
            movement_vitality_domain_score=None,
            baseline_confidence=baseline_confidence,
            baseline_quality=baseline_quality,
            persistent_abnormal_days=0,
            factors=("movement_vitality_insufficient_data",),
            risk_factor_details={},
            initial_baseline_ready=initial_ready,
            stable_baseline_ready=stable_ready,
        )

    denominator = sum(weight for _, weight in weighted_scores)
    score = sum(value * weight for value, weight in weighted_scores) / denominator
    quality = _movement_quality(normalized_current)
    if quality is not None and quality < 0.6:
        score *= 0.5
        details["quality_cap"] = {
            "pose_quality": round(quality, 4),
            "score_cap": 0.5,
        }
    score = min(max(score, 0.0), 1.0)
    factors = _movement_factors(details)
    if not factors:
        factors = ["movement_vitality_within_personal_range"]

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
            previous_score = score_movement_vitality_day(
                row,
                previous_history,
                config=aggregation_config,
            )
            if (
                previous_score.movement_vitality_score is None
                or previous_score.movement_vitality_score < 0.6
            ):
                break
            persistence += 1

    return MovementVitalityScore(
        movement_vitality_score=round(score, 4),
        movement_vitality_domain_score=round(score * 100.0, 1),
        baseline_confidence=baseline_confidence,
        baseline_quality=baseline_quality,
        persistent_abnormal_days=persistence,
        factors=tuple(factors),
        risk_factor_details=details,
        initial_baseline_ready=initial_ready,
        stable_baseline_ready=stable_ready,
    )


def build_movement_vitality_result(
    *,
    person_id: str,
    daily_features: Iterable[Mapping[str, Any]] = (),
    history_daily_features: Iterable[Mapping[str, Any]] = (),
    requested_date: date | None = None,
    device_id: str | None = None,
    config: AggregationConfig | None = None,
) -> dict[str, Any]:
    aggregation_config = config or load_aggregation_config()
    daily = normalize_movement_vitality_daily(
        daily_features,
        person_id=person_id,
        device_id=device_id,
    )
    if requested_date is not None:
        daily = [item for item in daily if item["date"] == requested_date.isoformat()]
    history = normalize_movement_vitality_daily(
        history_daily_features,
        person_id=person_id,
        device_id=device_id,
    )

    scored: list[dict[str, Any]] = []
    for item in daily:
        score = score_movement_vitality_day(item, history, config=aggregation_config)
        enriched = dict(item)
        enriched.update(
            {
                "movement_vitality_score": score.movement_vitality_score,
                "movement_vitality_domain_score": score.movement_vitality_domain_score,
                "baseline_confidence": score.baseline_confidence,
                "baseline_quality": score.baseline_quality,
                "persistent_abnormal_days": score.persistent_abnormal_days,
                "movement_vitality_factors": list(score.factors),
                "movement_vitality_details": score.risk_factor_details,
                "initial_baseline_ready": score.initial_baseline_ready,
                "stable_baseline_ready": score.stable_baseline_ready,
            }
        )
        scored.append(enriched)
        history.append(enriched)

    return {
        "schema_version": "movement_vitality_service_v1",
        "model_version": "movement-vitality-rulecard-v1",
        "person_id": person_id,
        "requested_date": requested_date.isoformat() if requested_date else None,
        "daily_features": scored,
        "quality_flags": _quality_flags(scored),
        "medical_disclaimer": "movement vitality trend only; not a medical diagnosis",
    }


def _normalize_daily_feature(
    record: Mapping[str, Any],
    *,
    person_id: str | None,
    device_id: str | None,
) -> dict[str, Any]:
    item = dict(record)
    resolved_person = str(item.get("person_id") or person_id or "").strip()
    if not resolved_person:
        raise ValueError("movement-vitality daily feature requires person_id")
    item["person_id"] = resolved_person
    item["date"] = _parse_date(item.get("date"), "daily.date").isoformat()
    item.setdefault("device_id", device_id)

    _alias_number(item, "gait_speed_norm_per_sec", "walking_speed_norm_per_sec", "walking_speed", "gait_speed")
    _alias_number(item, "gait_speed_mps", "walking_speed_mps")
    _alias_number(
        item,
        "sit_stand_duration_seconds",
        "sit_to_stand_seconds_median",
        "sit_to_stand_duration_seconds",
        "sit_stand_seconds",
    )
    _alias_number(
        item,
        "turn_duration_seconds",
        "turning_seconds_median",
        "turning_seconds",
    )
    _alias_number(item, "turn_stability_score", "turning_stability_score")
    _alias_number(item, "gait_cycle_stability_score", "gait_stability_score")
    _alias_number(item, "pose_quality_coverage", "pose_quality_score")

    flags = _string_list(item.get("quality_flags")) + _string_list(item.get("data_quality_flags"))
    if _usable_metric_count(item) == 0:
        flags.append("movement_vitality_insufficient_data")
    item["quality_flags"] = _dedupe(flags)
    quality = _movement_quality(item)
    item.setdefault("quality_score", quality if quality is not None else 1.0)
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


def _movement_factors(details: Mapping[str, Mapping[str, Any]]) -> list[str]:
    labels = {
        "gait_speed_norm_per_sec": "walking_speed_decline",
        "sit_stand_duration_seconds": "sit_to_stand_time_increase",
        "turn_duration_seconds": "turning_time_increase",
        "turn_stability_score": "turning_stability_decline",
        "gait_cycle_stability_score": "gait_stability_decline",
    }
    return _dedupe(
        labels[name]
        for name, detail in sorted(details.items(), key=lambda item: item[1].get("score", 0.0), reverse=True)
        if name in labels and detail.get("score", 0.0) >= 0.35
    )


def _usable_metric_count(record: Mapping[str, Any]) -> int:
    return sum(1 for name in METRIC_SPECS if _optional_number(record.get(name)) is not None)


def _movement_quality(record: Mapping[str, Any]) -> float | None:
    for field in ("pose_quality_coverage", "pose_quality_score", "quality_score"):
        value = _optional_number(record.get(field))
        if value is not None:
            return min(max(value, 0.0), 1.0)
    return None


def _baseline_eligible(record: Mapping[str, Any]) -> bool:
    if record.get("baseline_eligible") is False:
        return False
    quality = str(record.get("data_quality") or "valid")
    flags = set(_string_list(record.get("quality_flags"))) | set(_string_list(record.get("data_quality_flags")))
    return quality not in BASELINE_REJECTING_QUALITIES and not flags.intersection(BASELINE_REJECTING_QUALITIES)


def _baseline_quality(history: list[Mapping[str, Any]]) -> float:
    if not history:
        return 0.0
    qualities = [_movement_quality(item) for item in history]
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
        flags.extend(_string_list(record.get("quality_flags")))
        flags.extend(_string_list(record.get("data_quality_flags")))
        if record.get("movement_vitality_score") is None:
            flags.append("movement_vitality_score_unavailable")
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
