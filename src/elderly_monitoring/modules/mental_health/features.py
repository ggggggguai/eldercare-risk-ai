from __future__ import annotations

import math
from numbers import Real
from typing import Mapping

from elderly_monitoring.modules.mental_health.config import load_mental_health_config

MENTAL_HEALTH_FEATURES = (
    "activity_drop_score",
    "sleep_disturbance_score",
    "social_withdrawal_score",
    "routine_irregularity_score",
    "night_physiology_score",
    "movement_vitality_score",
    "negative_affect_score",
    "self_report_risk_score",
)


def weighted_mental_health_risk_score(
    features: Mapping[str, float | int | None],
    *,
    weights: Mapping[str, float] | None = None,
) -> float | None:
    """Return a weight-renormalized score over genuinely available features."""
    configured_weights = dict(weights or load_mental_health_config().scoring.weights)
    available: list[tuple[float, float]] = []
    for name, weight in configured_weights.items():
        value = features.get(name)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"mental-health feature '{name}' must be a finite number in [0, 1]")
        number = float(value)
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ValueError(f"mental-health feature '{name}' must be a finite number in [0, 1]")
        available.append((number, weight))
    if not available:
        return None
    denominator = sum(weight for _, weight in available)
    return round(sum(value * weight for value, weight in available) / denominator, 4)
