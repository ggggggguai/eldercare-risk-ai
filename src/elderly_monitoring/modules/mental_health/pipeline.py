from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time
from numbers import Real
from typing import Any, Mapping

from elderly_monitoring.common.schemas import (
    AlgorithmEvent,
    EvidenceWindow,
    RecommendedAction,
    action_for_level,
)
from elderly_monitoring.modules.mental_health.config import (
    MENTAL_HEALTH_SCORE_FEATURES,
    MentalHealthConfig,
    load_mental_health_config,
)
from elderly_monitoring.modules.mental_health.auxiliary_models import evaluate_auxiliary_models
from elderly_monitoring.modules.mental_health.feature_extraction.cognitive_tasks import (
    score_active_cognitive_tasks,
)
from elderly_monitoring.modules.mental_health.scorecards.domain_scores import (
    SUBMODULE_FUSION_WEIGHTS,
    DomainScore as MentalSafetySubmoduleResult,
    build_submodule_scores,
    fused_submodule_score,
)
from elderly_monitoring.modules.mental_health.scorecards.level_mapping import (
    level_from_score,
    trigger_event_for_level,
)
from elderly_monitoring.modules.mental_health.scorecards.persistence_gates import (
    applicable_level_caps,
    baseline_confidence,
    cap_level,
)
from elderly_monitoring.modules.mental_health.scorecards.strong_rules import (
    apply_strong_rule_level,
    evaluate_strong_rules,
)


@dataclass(frozen=True)
class MentalSafetyResult:
    """V2 mental-safety result used by algorithm-side consumers."""

    person_id: str
    timestamp: str
    mental_safety_level: int
    mental_safety_score: int
    confidence: float
    baseline_confidence: str
    submodules: dict[str, MentalSafetySubmoduleResult]
    suggestion: str
    diagnosis: bool
    trigger_event: str
    risk_factors: list[str]
    recommended_action: RecommendedAction
    evidence_window: EvidenceWindow
    model_version: str
    device_id: str | None = None
    scene_region: str | None = None
    auxiliary_models: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["evidence_window"] = self.evidence_window.to_dict()
        payload["submodules"] = {
            name: result.to_dict()
            for name, result in self.submodules.items()
        }
        return payload

    def to_algorithm_event(self) -> AlgorithmEvent:
        return AlgorithmEvent(
            module="mental_health",
            device_id=self.device_id,
            person_id=self.person_id,
            timestamp=self.timestamp,
            scene_region=self.scene_region,
            risk_level=self.mental_safety_level,
            risk_score=round(self.mental_safety_score / 100.0, 4),
            confidence=self.confidence,
            trigger_event=self.trigger_event,
            risk_factors=self.risk_factors,
            recommended_action=self.recommended_action,
            evidence_window=self.evidence_window,
            model_version=self.model_version,
            metadata={
                **self.metadata,
                "diagnosis": self.diagnosis,
                "mental_safety_level": self.mental_safety_level,
                "mental_safety_score": self.mental_safety_score,
                "baseline_confidence": self.baseline_confidence,
                "submodules": {
                    name: result.to_dict()
                    for name, result in self.submodules.items()
                },
                "suggestion": self.suggestion,
                "auxiliary_models": dict(self.auxiliary_models),
            },
        )


class MentalHealthRiskPipeline:
    """Behavioral risk warning pipeline, not a diagnostic model."""

    def __init__(self, config: MentalHealthConfig | None = None) -> None:
        self.config = config or load_mental_health_config()
        self.model_version = self.config.version

    def predict_from_features(self, sample: Mapping[str, Any]) -> AlgorithmEvent:
        return self.predict_mental_safety(sample).to_algorithm_event()

    def predict_mental_safety(self, sample: Mapping[str, Any]) -> MentalSafetyResult:
        if not isinstance(sample, Mapping):
            raise ValueError("mental-health feature sample must be an object")
        person_id = sample.get("person_id")
        if not isinstance(person_id, str) or not person_id.strip():
            raise ValueError("mental-health sample field 'person_id' must be a non-empty stable ID")

        features = {
            name: _optional_score(sample, name)
            for name in MENTAL_HEALTH_SCORE_FEATURES
        }
        cognitive_task_result = score_active_cognitive_tasks(sample)
        active_cognitive_task_score = _optional_score(sample, "active_cognitive_task_score")
        if active_cognitive_task_score is None:
            active_cognitive_task_score = cognitive_task_result.active_cognitive_task_score
        scorecard_features: dict[str, float | None] = {
            **features,
            "active_cognitive_task_score": active_cognitive_task_score,
        }
        persistent_days = _nonnegative_integer(sample, "persistent_abnormal_days", default=0)
        baseline_quality = _optional_unit_score(sample, "baseline_quality", default=0.0)
        manual_flag = _manual_flag(sample)
        available_modalities = [name for name in MENTAL_HEALTH_SCORE_FEATURES if features[name] is not None]
        if active_cognitive_task_score is not None:
            available_modalities.append("active_cognitive_task_score")
        missing_modalities = [name for name in MENTAL_HEALTH_SCORE_FEATURES if features[name] is None]
        feature_coverage = self._feature_coverage(features)

        initial_ready = _boolean_field(sample, "initial_baseline_ready", default=False)
        stable_ready = _boolean_field(sample, "stable_baseline_ready", default=False)
        applied_caps = applicable_level_caps(
            feature_coverage=feature_coverage,
            initial_ready=initial_ready,
            stable_ready=stable_ready,
            persistent_days=persistent_days,
            min_persistent_days_for_level_3=self.config.scoring.min_persistent_days_for_level_3,
            caps=self.config.scoring.caps,
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

        confidence = self._confidence(
            feature_coverage,
            baseline_quality,
            persistent_days,
        )
        submodules = build_submodule_scores(
            features=scorecard_features,
            weights=self.config.scoring.weights,
            thresholds=self.config.scoring.thresholds,
            abnormal_score_threshold=self.config.baseline.abnormal_score_threshold,
            confidence=confidence,
        )
        weighted_score = fused_submodule_score(submodules)
        risk_score = weighted_score if weighted_score is not None else 0.0
        candidate_level = level_from_score(risk_score, self.config.scoring.thresholds) if weighted_score is not None else 0
        passive_level = cap_level(
            candidate_level,
            passive_max_level=self.config.scoring.passive_max_level,
            applied_caps=applied_caps,
        )
        strong_rule_sample = (
            sample
            if active_cognitive_task_score is None or sample.get("active_cognitive_task_score") is not None
            else {**sample, "active_cognitive_task_score": active_cognitive_task_score}
        )
        strong_rule_matches = evaluate_strong_rules(
            strong_rule_sample,
            features=scorecard_features,
            persistent_days=persistent_days,
            abnormal_score_threshold=self.config.baseline.abnormal_score_threshold,
        )

        if strong_evidence_source is not None:
            risk_level = 4
            score_status = "strong_evidence_override"
        elif strong_rule_matches:
            risk_level = apply_strong_rule_level(passive_level, strong_rule_matches)
            risk_score = max(risk_score, self.config.scoring.thresholds[2])
            score_status = "strong_rule_override" if risk_level > passive_level else "available"
        elif weighted_score is None:
            risk_level = 0
            score_status = "unavailable"
        else:
            risk_level = passive_level
            score_status = "available"

        if weighted_score is None and strong_evidence_source is None and not strong_rule_matches:
            confidence = 0.0
        trigger_event = trigger_event_for_level(risk_level, score_status)
        risk_factors = self._risk_factors(
            scorecard_features,
            risk_level,
            score_status,
            strong_evidence_source,
        )
        for match in strong_rule_matches:
            if match.factor not in risk_factors:
                risk_factors.append(match.factor)
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

        auxiliary_models = evaluate_auxiliary_models(
            sample,
            features=scorecard_features,
            feature_names=(*MENTAL_HEALTH_SCORE_FEATURES, "active_cognitive_task_score"),
            abnormal_score_threshold=self.config.baseline.abnormal_score_threshold,
        )
        auxiliary_models["active_cognitive_tasks"] = cognitive_task_result.to_dict()
        if auxiliary_models["isolation_forest"].get("is_anomaly"):
            risk_factors.append("multimetric_anomaly_auxiliary")
        if auxiliary_models["change_point"].get("has_change"):
            risk_factors.append("trend_change_auxiliary")

        suggestion = _suggestion(risk_level, risk_factors, score_status)
        metadata = {
            "score_status": score_status,
            "score_source": "submodule_fusion_v2",
            "submodule_fusion_weights": dict(SUBMODULE_FUSION_WEIGHTS),
            "strong_rule_matches": [match.to_dict() for match in strong_rule_matches],
            "auxiliary_model_status": {
                name: payload.get("status")
                for name, payload in auxiliary_models.items()
                if isinstance(payload, Mapping)
            },
            "active_cognitive_task_score": active_cognitive_task_score,
            "active_cognitive_task_details": cognitive_task_result.to_dict(),
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
        }

        return MentalSafetyResult(
            person_id=person_id.strip(),
            timestamp=_event_timestamp(sample),
            mental_safety_level=risk_level,
            mental_safety_score=int(round(risk_score * 100)),
            confidence=round(confidence, 4),
            baseline_confidence=baseline_confidence(
                baseline_quality=baseline_quality,
                initial_ready=initial_ready,
                stable_ready=stable_ready,
            ),
            submodules=submodules,
            suggestion=suggestion,
            diagnosis=False,
            trigger_event=trigger_event,
            risk_factors=risk_factors,
            recommended_action=action_for_level(risk_level, mental_health=True),
            evidence_window=evidence_window,
            model_version=self.model_version,
            device_id=sample.get("device_id"),
            scene_region=sample.get("scene_region"),
            auxiliary_models=auxiliary_models,
            metadata=metadata,
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
            "night_physiology_score": "night_physiology_variation",
            "movement_vitality_score": "movement_vitality_decline",
            "negative_affect_score": "negative_affect_signal",
            "self_report_risk_score": "self_report_risk",
            "active_cognitive_task_score": "active_cognitive_task_clue",
        }
        threshold = self.config.baseline.abnormal_score_threshold
        factors = [
            names[name]
            for name in names
            if features.get(name) is not None and features[name] >= threshold
        ]
        if strong_evidence_source == "manual_emergency_flag":
            factors.append("manual_emergency_flag")
        if not factors and risk_level > 0:
            available = [name for name in names if features.get(name) is not None]
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


def _suggestion(risk_level: int, risk_factors: list[str], score_status: str) -> str:
    if score_status == "unavailable":
        return "当前数据不足，建议继续观察。系统不作医学诊断。"
    if risk_level <= 0:
        return "近期行为和睡眠趋势未见明显持续偏离。系统不作医学诊断。"
    if risk_level >= 4:
        return "当前结果需要家属人工确认，建议尽快联系老人了解情况。系统不作医学诊断。"
    if "sleep_rhythm_disturbance" in risk_factors:
        return "近期行为和睡眠模式变化值得关注，建议家属主动沟通，了解是否存在身体不适或睡眠问题。系统不作医学诊断。"
    if "social_interaction_decline" in risk_factors:
        return "近期活动或社交连接较个人基线有所下降，建议家属主动联系并持续观察。系统不作医学诊断。"
    return "近期行为模式变化值得关注，建议家属主动沟通并持续观察。系统不作医学诊断。"


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
