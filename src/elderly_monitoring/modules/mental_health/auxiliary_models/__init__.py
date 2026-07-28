"""Auxiliary model hooks that support, but do not replace, rule scorecards."""

from __future__ import annotations

from typing import Any, Mapping

from elderly_monitoring.modules.mental_health.auxiliary_models.change_point import (
    ChangePointEvidence,
    detect_trend_change,
)
from elderly_monitoring.modules.mental_health.auxiliary_models.isolation_forest import (
    IsolationForestEvidence,
    evaluate_multimetric_anomaly,
)


AUXILIARY_HISTORY_FIELDS = (
    "isolation_forest_history",
    "change_point_history",
    "auxiliary_history",
    "reference_feature_history",
    "baseline_feature_history",
    "recent_feature_history",
)


def evaluate_auxiliary_models(
    sample: Mapping[str, Any],
    *,
    features: Mapping[str, float | None],
    feature_names: tuple[str, ...],
    abnormal_score_threshold: float,
) -> dict[str, Any]:
    """Run optional auxiliary evidence models when personal history is supplied."""

    shared_history = _history_records(sample, preferred_field=None)
    isolation_history = _history_records(sample, preferred_field="isolation_forest_history") or shared_history
    change_history = _history_records(sample, preferred_field="change_point_history") or shared_history
    isolation = evaluate_multimetric_anomaly(
        isolation_history,
        features,
        feature_names=feature_names,
        abnormal_score_threshold=abnormal_score_threshold,
    )
    change_point = detect_trend_change(
        change_history,
        features,
        feature_names=feature_names,
    )
    return {
        "isolation_forest": isolation.to_dict(),
        "change_point": change_point.to_dict(),
    }


def _history_records(
    sample: Mapping[str, Any],
    *,
    preferred_field: str | None,
) -> tuple[Mapping[str, Any], ...]:
    fields = (preferred_field,) if preferred_field is not None else AUXILIARY_HISTORY_FIELDS
    for field in fields:
        if field is None:
            continue
        value = sample.get(field)
        if value is None:
            continue
        if not isinstance(value, list):
            raise ValueError(f"mental-health sample field '{field}' must be a list of objects")
        records: list[Mapping[str, Any]] = []
        for index, item in enumerate(value):
            if not isinstance(item, Mapping):
                raise ValueError(
                    f"mental-health sample field '{field}[{index}]' must be an object"
                )
            records.append(item)
        return tuple(records)
    return ()


__all__ = [
    "ChangePointEvidence",
    "IsolationForestEvidence",
    "detect_trend_change",
    "evaluate_auxiliary_models",
    "evaluate_multimetric_anomaly",
]
