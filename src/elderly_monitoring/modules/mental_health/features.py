from __future__ import annotations

from typing import Mapping

MENTAL_HEALTH_FEATURES = (
    "activity_drop_score",
    "sleep_disturbance_score",
    "social_withdrawal_score",
    "routine_irregularity_score",
    "negative_affect_score",
    "self_report_risk_score",
)


def clamp_score(value: float | int | None) -> float:
    if value is None:
        return 0.0
    return max(0.0, min(1.0, float(value)))


def weighted_mental_health_risk_score(features: Mapping[str, float | int]) -> float:
    weights = {
        "activity_drop_score": 0.24,
        "sleep_disturbance_score": 0.22,
        "social_withdrawal_score": 0.18,
        "routine_irregularity_score": 0.16,
        "negative_affect_score": 0.10,
        "self_report_risk_score": 0.10,
    }
    score = 0.0
    for name, weight in weights.items():
        score += clamp_score(features.get(name)) * weight
    return round(score, 4)
