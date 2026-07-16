from __future__ import annotations

import math
from datetime import date, datetime, time
from numbers import Real
from typing import Any, Mapping

from elderly_monitoring.common.schemas import AlgorithmEvent, EvidenceWindow, action_for_level
from elderly_monitoring.modules.mental_health.config import (
    MENTAL_HEALTH_SCORE_FEATURES,
    MentalHealthConfig,
    load_mental_health_config,
)
from elderly_monitoring.modules.mental_health.features import weighted_mental_health_risk_score


class MentalHealthRiskPipeline:
    """Behavioral risk warning pipeline, not a diagnostic model."""

    def __init__(self, config: MentalHealthConfig | None = None) -> None:
        self.config = config or load_mental_health_config()
        self.model_version = self.config.version

    def predict_from_features(self, sample: Mapping[str, Any]) -> AlgorithmEvent:
        if not isinstance(sample, Mapping):
            raise ValueError("mental-health feature sample must be an object")
        person_id = sample.get("person_id")
        if not isinstance(person_id, str) or not person_id.strip():
            raise ValueError("mental-health sample field 'person_id' must be a non-empty stable ID")

        features = {
            name: _optional_score(sample, name)
            for name in MENTAL_HEALTH_SCORE_FEATURES
        }
        persistent_days = _nonnegative_integer(sample, "persistent_abnormal_days", default=0)
        baseline_quality = _optional_unit_score(sample, "baseline_quality", default=0.0)
        manual_flag = _manual_flag(sample)
        available_modalities = [name for name in MENTAL_HEALTH_SCORE_FEATURES if features[name] is not None]
        missing_modalities = [name for name in MENTAL_HEALTH_SCORE_FEATURES if features[name] is None]
        feature_coverage = self._feature_coverage(features)
        weighted_score = weighted_mental_health_risk_score(
            features,
            weights=self.config.scoring.weights,
        )
        risk_score = weighted_score if weighted_score is not None else 0.0

        initial_ready = _boolean_field(sample, "initial_baseline_ready", default=False)
        stable_ready = _boolean_field(sample, "stable_baseline_ready", default=False)
        applied_caps = self._applicable_caps(
            feature_coverage=feature_coverage,
            initial_ready=initial_ready,
            stable_ready=stable_ready,
            persistent_days=persistent_days,
        )
        candidate_level = self._candidate_level(risk_score) if weighted_score is not None else 0
        passive_level = min(
            candidate_level,
            self.config.scoring.passive_max_level,
            *(item["max_level"] for item in applied_caps),
        )

        self_report = features["self_report_risk_score"]
        strong_evidence_source: str | None = None
        if manual_flag is True:
            strong_evidence_source = "manual_emergency_flag"
        elif (
            self_report is not None
            and self_report >= self.config.scoring.self_report_emergency_threshold
        ):
            strong_evidence_source = "self_report_risk_score"

        if strong_evidence_source is not None:
            risk_level = 4
            score_status = "strong_evidence_override"
        elif weighted_score is None:
            risk_level = 0
            score_status = "unavailable"
        else:
            risk_level = passive_level
            score_status = "available"

        confidence = self._confidence(
            feature_coverage,
            baseline_quality,
            persistent_days,
        )
        if weighted_score is None and strong_evidence_source is None:
            confidence = 0.0
        trigger_event = _trigger_event(risk_level, score_status)
        risk_factors = self._risk_factors(
            features,
            risk_level,
            score_status,
            strong_evidence_source,
        )
        evidence_window = _evidence_window(sample, self.config)
        risk_factor_details = sample.get("risk_factor_details")
        if risk_factor_details is None:
            risk_factor_details = {}
        if not isinstance(risk_factor_details, Mapping):
            raise ValueError("mental-health sample field 'risk_factor_details' must be an object")
        baseline_window = sample.get("baseline_window")
        if baseline_window is None:
            baseline_window = {}
        if not isinstance(baseline_window, Mapping):
            raise ValueError("mental-health sample field 'baseline_window' must be an object")

        return AlgorithmEvent(
            module="mental_health",
            device_id=sample.get("device_id"),
            person_id=person_id.strip(),
            timestamp=_event_timestamp(sample),
            scene_region=sample.get("scene_region"),
            risk_level=risk_level,
            risk_score=round(risk_score, 4),
            confidence=round(confidence, 4),
            trigger_event=trigger_event,
            risk_factors=risk_factors,
            recommended_action=action_for_level(risk_level, mental_health=True),
            evidence_window=evidence_window,
            model_version=self.model_version,
            metadata={
                "diagnosis": False,
                "score_status": score_status,
                "feature_coverage": round(feature_coverage, 4),
                "baseline_quality": round(baseline_quality, 4),
                "persistent_abnormal_days": persistent_days,
                "available_modalities": available_modalities,
                "missing_modalities": missing_modalities,
                "risk_factor_details": dict(risk_factor_details),
                "baseline_window": dict(baseline_window),
                "initial_baseline_ready": initial_ready,
                "stable_baseline_ready": stable_ready,
                "applied_level_caps": applied_caps,
                "strong_evidence_source": strong_evidence_source,
            },
        )

    def _feature_coverage(self, features: Mapping[str, float | None]) -> float:
        expected = self.config.scoring.coverage_expected_features
        denominator = sum(self.config.scoring.weights[name] for name in expected)
        numerator = sum(
            self.config.scoring.weights[name]
            for name in expected
            if features.get(name) is not None
        )
        return numerator / denominator

    def _candidate_level(self, risk_score: float) -> int:
        level_1, level_2, level_3 = self.config.scoring.thresholds
        if risk_score >= level_3:
            return 3
        if risk_score >= level_2:
            return 2
        if risk_score >= level_1:
            return 1
        return 0

    def _applicable_caps(
        self,
        *,
        feature_coverage: float,
        initial_ready: bool,
        stable_ready: bool,
        persistent_days: int,
    ) -> list[dict[str, Any]]:
        caps = self.config.scoring.caps
        applied: list[dict[str, Any]] = []
        if feature_coverage < 0.40:
            applied.append({"reason": "coverage_below_0_40", "max_level": caps.coverage_below_0_40})
        elif feature_coverage < 0.60:
            applied.append({"reason": "coverage_below_0_60", "max_level": caps.coverage_below_0_60})
        if not initial_ready:
            applied.append(
                {"reason": "initial_baseline_not_ready", "max_level": caps.initial_baseline_not_ready}
            )
        elif not stable_ready:
            applied.append(
                {"reason": "stable_baseline_not_ready", "max_level": caps.stable_baseline_not_ready}
            )
        if persistent_days < self.config.scoring.min_persistent_days_for_level_3:
            applied.append(
                {
                    "reason": "persistent_days_below_minimum",
                    "max_level": caps.persistent_days_below_minimum,
                }
            )
        return applied

    def _confidence(
        self,
        feature_coverage: float,
        baseline_quality: float,
        persistent_days: int,
    ) -> float:
        confidence = self.config.scoring.confidence
        persistence_support = min(
            persistent_days / self.config.scoring.min_persistent_days_for_level_3,
            1.0,
        )
        return max(
            0.0,
            min(
                1.0,
                confidence.feature_coverage_weight * feature_coverage
                + confidence.baseline_quality_weight * baseline_quality
                + confidence.persistence_weight * persistence_support,
            ),
        )

    def _risk_factors(
        self,
        features: Mapping[str, float | None],
        risk_level: int,
        score_status: str,
        strong_evidence_source: str | None,
    ) -> list[str]:
        if score_status == "unavailable":
            return ["insufficient_data"]
        names = {
            "activity_drop_score": "activity_drop",
            "sleep_disturbance_score": "sleep_rhythm_disturbance",
            "social_withdrawal_score": "social_interaction_decline",
            "routine_irregularity_score": "daily_routine_irregularity",
            "negative_affect_score": "negative_affect_signal",
            "self_report_risk_score": "self_report_risk",
        }
        threshold = self.config.baseline.abnormal_score_threshold
        factors = [
            names[name]
            for name in MENTAL_HEALTH_SCORE_FEATURES
            if features[name] is not None and features[name] >= threshold
        ]
        if strong_evidence_source == "manual_emergency_flag":
            factors.append("manual_emergency_flag")
        if not factors and risk_level > 0:
            available = [name for name in MENTAL_HEALTH_SCORE_FEATURES if features[name] is not None]
            if available:
                factors.append(names[max(available, key=lambda name: features[name] or 0.0)])
        if not factors and risk_level == 0:
            factors.append("no_obvious_behavioral_risk")
        return factors


def _optional_score(sample: Mapping[str, Any], field: str) -> float | None:
    value = sample.get(field)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"mental-health sample field '{field}' must be a finite number in [0, 1]")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"mental-health sample field '{field}' must be a finite number in [0, 1]")
    return number


def _optional_unit_score(sample: Mapping[str, Any], field: str, *, default: float) -> float:
    value = sample.get(field)
    if value is None:
        return default
    score = _optional_score(sample, field)
    if score is None:
        return default
    return score


def _nonnegative_integer(sample: Mapping[str, Any], field: str, *, default: int) -> int:
    value = sample.get(field)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"mental-health sample field '{field}' must be a non-negative integer")
    return value


def _boolean_field(sample: Mapping[str, Any], field: str, *, default: bool) -> bool:
    value = sample.get(field)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"mental-health sample field '{field}' must be a boolean")
    return value


def _manual_flag(sample: Mapping[str, Any]) -> bool | None:
    value = sample.get("manual_emergency_flag")
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError("mental-health sample field 'manual_emergency_flag' must be a boolean")
    return value


def _trigger_event(risk_level: int, score_status: str) -> str:
    if score_status == "unavailable":
        return "insufficient_data"
    return {
        0: "normal",
        1: "mild_behavioral_change",
        2: "behavioral_rhythm_deviation",
        3: "persistent_behavioral_risk",
        4: "urgent_manual_review",
    }[risk_level]


def _event_timestamp(sample: Mapping[str, Any]) -> str:
    value = sample.get("timestamp") or sample.get("end_time") or sample.get("start_time")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("mental-health sample requires a non-empty event timestamp")
    return value.strip()


def _evidence_window(sample: Mapping[str, Any], config: MentalHealthConfig) -> EvidenceWindow:
    raw = sample.get("evidence_window")
    if isinstance(raw, EvidenceWindow):
        if raw.start_time is None and raw.end_time is None:
            raise ValueError("mental-health evidence_window must contain a boundary")
        return raw
    if raw is not None and not isinstance(raw, Mapping):
        raise ValueError("mental-health sample field 'evidence_window' must be an object")
    values = raw or {}
    start = _optional_epoch(values.get("start_time"), "evidence_window.start_time", config)
    end = _optional_epoch(values.get("end_time"), "evidence_window.end_time", config)
    if start is None:
        start = _optional_epoch(sample.get("start_time"), "start_time", config)
    if end is None:
        end = _optional_epoch(sample.get("end_time"), "end_time", config)
    if start is None and end is None:
        timestamp = _event_timestamp(sample)
        start = end = _optional_epoch(timestamp, "timestamp", config)
    elif start is None:
        start = end
    elif end is None:
        end = start
    if start is None or end is None:
        raise ValueError("mental-health evidence_window requires event-time boundaries")
    if end < start:
        raise ValueError("mental-health evidence_window.end_time must not precede start_time")
    return EvidenceWindow(start_time=round(start, 4), end_time=round(end, 4))


def _optional_epoch(value: Any, field: str, config: MentalHealthConfig) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"mental-health sample field '{field}' must be an event time")
    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"mental-health sample field '{field}' must be finite")
        return number
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"mental-health sample field '{field}' must be an event time")
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        try:
            parsed_date = date.fromisoformat(normalized)
        except ValueError as exc:
            raise ValueError(f"mental-health sample field '{field}' must be ISO-8601") from exc
        parsed = datetime.combine(parsed_date, time.min, tzinfo=config.aggregation.timezone_info)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"mental-health sample field '{field}' must include a timezone")
    return parsed.timestamp()
