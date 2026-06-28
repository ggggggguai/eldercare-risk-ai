from __future__ import annotations

from typing import Any, Mapping

from elderly_monitoring.common.schemas import AlgorithmEvent, action_for_level
from elderly_monitoring.modules.mental_health.features import (
    clamp_score,
    weighted_mental_health_risk_score,
)


class MentalHealthRiskPipeline:
    """Behavioral risk warning pipeline, not a diagnostic model."""

    model_version = "mental-health-risk-v0.1"

    def predict_from_features(self, sample: Mapping[str, Any]) -> AlgorithmEvent:
        risk_score = weighted_mental_health_risk_score(sample)
        self_report_risk = clamp_score(sample.get("self_report_risk_score"))

        if risk_score >= 0.9 or self_report_risk >= 0.9:
            risk_level = 4
            trigger_event = "urgent_manual_review"
        elif risk_score >= 0.65:
            risk_level = 3
            trigger_event = "persistent_behavioral_risk"
        elif risk_score >= 0.45:
            risk_level = 2
            trigger_event = "behavioral_rhythm_deviation"
        elif risk_score >= 0.25:
            risk_level = 1
            trigger_event = "mild_behavioral_change"
        else:
            risk_level = 0
            trigger_event = "normal"

        return AlgorithmEvent(
            module="mental_health",
            device_id=sample.get("device_id"),
            person_id=str(sample.get("person_id", "unknown")),
            timestamp=str(sample.get("timestamp", "")),
            scene_region=sample.get("scene_region"),
            risk_level=risk_level,
            risk_score=risk_score,
            confidence=self._confidence(sample, risk_score),
            trigger_event=trigger_event,
            risk_factors=self._risk_factors(sample, risk_level),
            recommended_action=action_for_level(risk_level, mental_health=True),
            evidence_window=None,
            model_version=self.model_version,
            metadata={"diagnosis": False},
        )

    def _risk_factors(self, sample: Mapping[str, Any], risk_level: int) -> list[str]:
        factors: list[str] = []
        if clamp_score(sample.get("activity_drop_score")) >= 0.5:
            factors.append("activity_drop")
        if clamp_score(sample.get("sleep_disturbance_score")) >= 0.5:
            factors.append("sleep_rhythm_disturbance")
        if clamp_score(sample.get("social_withdrawal_score")) >= 0.5:
            factors.append("social_interaction_decline")
        if clamp_score(sample.get("routine_irregularity_score")) >= 0.5:
            factors.append("daily_routine_irregularity")
        if clamp_score(sample.get("negative_affect_score")) >= 0.5:
            factors.append("negative_affect_signal")
        if clamp_score(sample.get("self_report_risk_score")) >= 0.5:
            factors.append("self_report_risk")
        if not factors and risk_level == 0:
            factors.append("no_obvious_behavioral_risk")
        return factors

    def _confidence(self, sample: Mapping[str, Any], risk_score: float) -> float:
        coverage = clamp_score(sample.get("feature_coverage", 0.7))
        baseline_quality = clamp_score(sample.get("baseline_quality", 0.7))
        review_support = clamp_score(sample.get("review_support", 0.0))
        confidence = 0.40 * coverage + 0.35 * baseline_quality + 0.15 * min(1.0, risk_score + 0.2) + 0.10 * review_support
        return round(confidence, 4)
