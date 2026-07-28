"""Isolation Forest support for multimetric anomaly evidence."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from numbers import Real
from statistics import median
from typing import Any, Iterable, Mapping


try:  # pragma: no cover - exercised only when the optional ml extra is installed.
    from sklearn.ensemble import IsolationForest
except ImportError:  # pragma: no cover - the fallback keeps core tests dependency-light.
    IsolationForest = None  # type: ignore[assignment]


@dataclass(frozen=True)
class IsolationForestEvidence:
    """Auxiliary multimetric anomaly evidence; it never replaces the scorecard."""

    status: str
    anomaly_score: float
    is_anomaly: bool
    factors: list[str] = field(default_factory=list)
    feature_values: dict[str, float] = field(default_factory=dict)
    feature_contributions: dict[str, float] = field(default_factory=dict)
    reference_count: int = 0
    method: str = "isolation_forest"
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_multimetric_anomaly(
    history_records: Iterable[Mapping[str, Any]],
    current_features: Mapping[str, float | None],
    *,
    feature_names: Iterable[str],
    abnormal_score_threshold: float,
    min_reference_rows: int = 5,
    contamination: float = 0.15,
    random_state: int = 13,
) -> IsolationForestEvidence:
    """Fit a per-person reference model and score today's feature vector.

    The model consumes already-normalized risk scores in [0, 1]. A high
    auxiliary score means the current multi-feature pattern is unusual compared
    with the person's own recent baseline records.
    """

    names = [
        name
        for name in feature_names
        if _optional_unit(current_features.get(name)) is not None
    ]
    if len(names) < 2:
        return _insufficient("at_least_two_current_features_required")

    current_vector = [_optional_unit(current_features[name]) for name in names]
    if any(value is None for value in current_vector):
        return _insufficient("current_feature_vector_incomplete")
    current = [float(value) for value in current_vector if value is not None]

    reference_rows = _reference_matrix(history_records, names)
    if len(reference_rows) < min_reference_rows:
        return _insufficient("insufficient_reference_rows", len(reference_rows))

    if IsolationForest is not None:
        return _sklearn_isolation_forest(
            reference_rows,
            current,
            names=names,
            abnormal_score_threshold=abnormal_score_threshold,
            contamination=contamination,
            random_state=random_state,
        )
    return _robust_distance_fallback(
        reference_rows,
        current,
        names=names,
        abnormal_score_threshold=abnormal_score_threshold,
    )


def _sklearn_isolation_forest(
    reference_rows: list[list[float]],
    current: list[float],
    *,
    names: list[str],
    abnormal_score_threshold: float,
    contamination: float,
    random_state: int,
) -> IsolationForestEvidence:
    model = IsolationForest(
        n_estimators=100,
        contamination=max(0.01, min(0.5, contamination)),
        random_state=random_state,
    )
    model.fit(reference_rows)
    history_scores = [float(value) for value in model.decision_function(reference_rows)]
    current_score = float(model.decision_function([current])[0])
    anomaly_score = _percentile_anomaly_score(history_scores, current_score)
    prediction = int(model.predict([current])[0])
    contributions = _feature_contributions(reference_rows, current, names)
    factors = _factors(
        names,
        current,
        contributions,
        abnormal_score_threshold=abnormal_score_threshold,
    )
    return IsolationForestEvidence(
        status="available",
        anomaly_score=round(anomaly_score, 4),
        is_anomaly=prediction == -1 or anomaly_score >= 0.85,
        factors=factors,
        feature_values=dict(zip(names, current, strict=True)),
        feature_contributions=contributions,
        reference_count=len(reference_rows),
        method="isolation_forest",
    )


def _robust_distance_fallback(
    reference_rows: list[list[float]],
    current: list[float],
    *,
    names: list[str],
    abnormal_score_threshold: float,
) -> IsolationForestEvidence:
    contributions = _feature_contributions(reference_rows, current, names)
    if not contributions:
        return _insufficient("reference_variation_unavailable", len(reference_rows))
    mean_contribution = sum(contributions.values()) / len(contributions)
    anomaly_score = _clamp(mean_contribution / 3.0)
    factors = _factors(
        names,
        current,
        contributions,
        abnormal_score_threshold=abnormal_score_threshold,
    )
    return IsolationForestEvidence(
        status="available",
        anomaly_score=round(anomaly_score, 4),
        is_anomaly=anomaly_score >= 0.85,
        factors=factors,
        feature_values=dict(zip(names, current, strict=True)),
        feature_contributions=contributions,
        reference_count=len(reference_rows),
        method="robust_distance_fallback",
    )


def _reference_matrix(
    history_records: Iterable[Mapping[str, Any]],
    names: list[str],
) -> list[list[float]]:
    rows: list[list[float]] = []
    for record in history_records:
        row = [_optional_unit(record.get(name)) for name in names]
        if all(value is not None for value in row):
            rows.append([float(value) for value in row if value is not None])
    return rows


def _feature_contributions(
    reference_rows: list[list[float]],
    current: list[float],
    names: list[str],
) -> dict[str, float]:
    if not reference_rows:
        return {}
    columns = list(zip(*reference_rows, strict=True))
    output: dict[str, float] = {}
    for index, values in enumerate(columns):
        center = median(values)
        deviations = [abs(value - center) for value in values]
        mad = median(deviations)
        scale = max(1.4826 * mad, 0.05)
        output[names[index]] = round(abs(current[index] - center) / scale, 4)
    return output


def _factors(
    names: list[str],
    current: list[float],
    contributions: Mapping[str, float],
    *,
    abnormal_score_threshold: float,
) -> list[str]:
    selected = [
        name
        for name, value in zip(names, current, strict=True)
        if value >= abnormal_score_threshold or contributions.get(name, 0.0) >= 2.0
    ]
    selected.sort(key=lambda name: contributions.get(name, 0.0), reverse=True)
    return selected[:4]


def _percentile_anomaly_score(history_scores: list[float], current_score: float) -> float:
    if not history_scores:
        return 0.0
    lower_or_equal = sum(1 for score in history_scores if score >= current_score)
    return _clamp(lower_or_equal / len(history_scores))


def _optional_unit(value: Any) -> float | None:
    if value is None or isinstance(value, bool) or not isinstance(value, Real):
        return None
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        return None
    return number


def _insufficient(reason: str, reference_count: int = 0) -> IsolationForestEvidence:
    return IsolationForestEvidence(
        status="insufficient_data",
        anomaly_score=0.0,
        is_anomaly=False,
        reference_count=reference_count,
        reason=reason,
    )


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
