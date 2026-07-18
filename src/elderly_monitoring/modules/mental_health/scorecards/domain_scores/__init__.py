from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Mapping

from elderly_monitoring.modules.mental_health.features import weighted_mental_health_risk_score
from elderly_monitoring.modules.mental_health.scorecards.level_mapping import level_from_score


MOOD_SOCIAL_FEATURE_FACTORS = {
    "activity_drop_score": "activity_drop",
    "sleep_disturbance_score": "sleep_rhythm_disturbance",
    "social_withdrawal_score": "social_interaction_decline",
    "routine_irregularity_score": "daily_routine_irregularity",
    "night_physiology_score": "night_physiology_variation",
    "movement_vitality_score": "movement_vitality_decline",
    "negative_affect_score": "negative_affect_signal",
}
COGNITIVE_CLUE_FEATURE_FACTORS = {
    "movement_vitality_score": "motor_cognitive_clue",
    "routine_irregularity_score": "routine_or_wandering_clue",
    "self_report_risk_score": "active_or_self_report_clue",
    "active_cognitive_task_score": "active_cognitive_task_clue",
}
SUBMODULE_FUSION_WEIGHTS = {
    "mood_social_withdrawal": 0.65,
    "cognitive_change_clue": 0.35,
}
COGNITIVE_CLUE_WEIGHTS = {
    "movement_vitality_score": 0.40,
    "routine_irregularity_score": 0.20,
    "self_report_risk_score": 0.15,
    "active_cognitive_task_score": 0.25,
}


@dataclass(frozen=True)
class DomainScore:
    """A scorecard domain result; it is not a diagnosis label."""

    level: int
    score: int
    factors: list[str]
    confidence: float
    available_features: list[str] = field(default_factory=list)
    feature_scores: dict[str, float | None] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def build_submodule_scores(
    features: Mapping[str, float | None],
    *,
    weights: Mapping[str, float],
    thresholds: tuple[float, float, float],
    abnormal_score_threshold: float,
    confidence: float,
) -> dict[str, DomainScore]:
    mood_features = {
        name: features[name]
        for name in MOOD_SOCIAL_FEATURE_FACTORS
    }
    cognitive_features = {
        name: features[name]
        for name in COGNITIVE_CLUE_FEATURE_FACTORS
    }
    return {
        "mood_social_withdrawal": domain_score(
            feature_scores=mood_features,
            factors=MOOD_SOCIAL_FEATURE_FACTORS,
            weights=weights,
            thresholds=thresholds,
            abnormal_score_threshold=abnormal_score_threshold,
            confidence=confidence,
        ),
        "cognitive_change_clue": domain_score(
            feature_scores=cognitive_features,
            factors=COGNITIVE_CLUE_FEATURE_FACTORS,
            weights={**weights, **COGNITIVE_CLUE_WEIGHTS},
            thresholds=thresholds,
            abnormal_score_threshold=abnormal_score_threshold,
            confidence=confidence,
            primary_features=(
                "movement_vitality_score",
                "self_report_risk_score",
                "active_cognitive_task_score",
            ),
        ),
    }


def domain_score(
    *,
    feature_scores: Mapping[str, float | None],
    factors: Mapping[str, str],
    weights: Mapping[str, float],
    thresholds: tuple[float, float, float],
    abnormal_score_threshold: float,
    confidence: float,
    primary_features: tuple[str, ...] | None = None,
) -> DomainScore:
    raw_available_features = [
        name
        for name, value in feature_scores.items()
        if value is not None
    ]
    scoreable = bool(raw_available_features)
    if primary_features is not None:
        scoreable = any(name in raw_available_features for name in primary_features) or any(
            value is not None and value >= abnormal_score_threshold
            for name, value in feature_scores.items()
            if name not in primary_features
        )
    score = (
        weighted_mental_health_risk_score(
            feature_scores,
            weights={name: weights[name] for name in feature_scores},
        )
        if scoreable
        else None
    )
    numeric_score = 0.0 if score is None else score
    factor_names = [
        factors[name]
        for name, value in feature_scores.items()
        if value is not None and value >= abnormal_score_threshold
    ]
    available_features = raw_available_features if score is not None else []
    if score is None:
        factor_names = ["insufficient_data"]
    elif not factor_names and numeric_score > 0:
        available = {
            name: value
            for name, value in feature_scores.items()
            if value is not None
        }
        if available:
            factor_names = [factors[max(available, key=lambda name: available[name] or 0.0)]]
    elif not factor_names:
        factor_names = ["no_obvious_behavioral_risk"]
    return DomainScore(
        level=level_from_score(numeric_score, thresholds) if score is not None else 0,
        score=int(round(numeric_score * 100)),
        factors=factor_names,
        confidence=round(confidence, 4) if score is not None else 0.0,
        available_features=available_features,
        feature_scores=dict(feature_scores),
    )


def fused_submodule_score(submodules: Mapping[str, DomainScore]) -> float | None:
    available = [
        (name, result)
        for name, result in submodules.items()
        if result.available_features
    ]
    if not available:
        return None
    denominator = sum(SUBMODULE_FUSION_WEIGHTS[name] for name, _ in available)
    return round(
        sum(
            (result.score / 100.0) * SUBMODULE_FUSION_WEIGHTS[name]
            for name, result in available
        )
        / denominator,
        4,
    )
