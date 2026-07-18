from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Any, Mapping


@dataclass(frozen=True)
class StrongRuleMatch:
    rule_id: str
    level: int
    factor: str
    evidence: dict[str, Any]
    action: str = "manual_review"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_strong_rules(
    sample: Mapping[str, Any],
    *,
    features: Mapping[str, float | None],
    persistent_days: int,
    abnormal_score_threshold: float,
) -> list[StrongRuleMatch]:
    matches: list[StrongRuleMatch] = []
    activity = _score(features.get("activity_drop_score"))
    sleep = _score(features.get("sleep_disturbance_score"))
    social = _score(features.get("social_withdrawal_score"))
    movement = _score(features.get("movement_vitality_score"))
    routine = _score(features.get("routine_irregularity_score"))

    bedroom = _first_score(
        sample,
        "bedroom_stay_increase_score",
        "bedroom_stay_ratio_score",
        "bedroom_increase_score",
    )
    if (
        persistent_days >= 7
        and activity >= abnormal_score_threshold
        and social >= abnormal_score_threshold
        and bedroom >= abnormal_score_threshold
    ):
        matches.append(
            StrongRuleMatch(
                rule_id="activity_bedroom_social_7d",
                level=3,
                factor="activity_bedroom_social_strong_rule",
                evidence={
                    "persistent_abnormal_days": persistent_days,
                    "activity_drop_score": activity,
                    "bedroom_stay_increase_score": bedroom,
                    "social_withdrawal_score": social,
                },
            )
        )

    night_leave_bed = _night_leave_bed_score(sample)
    if sleep >= 0.75 and activity >= abnormal_score_threshold and night_leave_bed >= abnormal_score_threshold:
        matches.append(
            StrongRuleMatch(
                rule_id="sleep_leave_bed_activity_decline",
                level=3,
                factor="sleep_leave_bed_activity_strong_rule",
                evidence={
                    "sleep_disturbance_score": sleep,
                    "activity_drop_score": activity,
                    "night_leave_bed_score": night_leave_bed,
                },
            )
        )

    high_risk_wandering = _high_risk_wandering_score(sample)
    if high_risk_wandering >= abnormal_score_threshold:
        matches.append(
            StrongRuleMatch(
                rule_id="high_risk_wandering_2_nights",
                level=3,
                factor="high_risk_wandering_strong_rule",
                evidence={
                    "high_risk_wandering_score": high_risk_wandering,
                    "consecutive_nights_with_wandering": _integer(sample.get("consecutive_nights_with_wandering")),
                    "doorway_wandering_count": _integer(sample.get("doorway_wandering_count")),
                    "bathroom_entrance_wandering_count": _integer(sample.get("bathroom_entrance_wandering_count")),
                    "high_risk_roi_wandering_count": _integer(sample.get("high_risk_roi_wandering_count")),
                },
            )
        )

    active_cognitive = _first_score(
        sample,
        "active_cognitive_task_score",
        "cognitive_task_decline_score",
        "animal_fluency_decline_score",
        "countdown_task_decline_score",
    )
    if active_cognitive >= abnormal_score_threshold and (
        movement >= abnormal_score_threshold or routine >= abnormal_score_threshold
    ):
        matches.append(
            StrongRuleMatch(
                rule_id="active_cognitive_motor_decline",
                level=3,
                factor="active_cognitive_motor_strong_rule",
                evidence={
                    "active_cognitive_task_score": active_cognitive,
                    "movement_vitality_score": movement,
                    "routine_irregularity_score": routine,
                },
            )
        )

    return matches


def apply_strong_rule_level(base_level: int, matches: list[StrongRuleMatch]) -> int:
    if not matches:
        return base_level
    return max(base_level, max(match.level for match in matches))


def _night_leave_bed_score(sample: Mapping[str, Any]) -> float:
    explicit = _first_score(
        sample,
        "night_leave_bed_score",
        "night_out_of_bed_score",
        "leave_bed_frequency_score",
        "out_of_bed_frequency_score",
    )
    if explicit > 0.0:
        return explicit
    count = _first_number(
        sample,
        "night_leave_bed_count",
        "night_out_of_bed_count",
        "leave_bed_count",
        "out_of_bed_count",
    )
    if count is None:
        return 0.0
    return _clamp(count / 4.0)


def _high_risk_wandering_score(sample: Mapping[str, Any]) -> float:
    explicit = _first_score(
        sample,
        "high_risk_wandering_score",
        "wandering_high_risk_score",
    )
    if explicit > 0.0 and (
        sample.get("high_risk_wandering_verified") is True
        or sample.get("wandering_rule_verified") is True
    ):
        return explicit
    consecutive = _integer(sample.get("consecutive_nights_with_wandering"))
    high_risk_count = sum(
        _integer(sample.get(field))
        for field in (
            "doorway_wandering_count",
            "bathroom_entrance_wandering_count",
            "high_risk_roi_wandering_count",
        )
    )
    if consecutive >= 2 and high_risk_count >= 2:
        return 1.0
    return 0.0


def _first_score(sample: Mapping[str, Any], *fields: str) -> float:
    for field in fields:
        value = _score(sample.get(field))
        if value > 0.0:
            return value
    return 0.0


def _first_number(sample: Mapping[str, Any], *fields: str) -> float | None:
    for field in fields:
        value = sample.get(field)
        if isinstance(value, bool) or not isinstance(value, Real):
            continue
        number = float(value)
        if math.isfinite(number):
            return number
    return None


def _score(value: Any) -> float:
    if value is None or isinstance(value, bool) or not isinstance(value, Real):
        return 0.0
    number = float(value)
    if not math.isfinite(number):
        return 0.0
    return _clamp(number)


def _integer(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(value, 0)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
