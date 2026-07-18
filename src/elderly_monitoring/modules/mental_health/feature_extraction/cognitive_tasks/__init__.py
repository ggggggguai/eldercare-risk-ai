"""Optional active cognitive task scoring.

The active tasks are structured, opt-in clues for the cognitive-change module.
They are not diagnostic tests and should not be interpreted as screening
results without human review.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from numbers import Real
from typing import Any, Mapping


TASK_WEIGHTS = {
    "animal_fluency": 0.30,
    "countdown": 0.25,
    "recall": 0.25,
    "picture_description": 0.20,
}
DEFAULT_THRESHOLDS = (0.25, 0.45, 0.65)


@dataclass(frozen=True)
class ActiveCognitiveTaskScore:
    """Rule-score output for optional active cognitive tasks."""

    active_cognitive_task_score: float | None
    active_cognitive_task_level: int
    factors: tuple[str, ...] = ()
    confidence: float = 0.0
    available_tasks: tuple[str, ...] = ()
    task_scores: dict[str, float | None] = field(default_factory=dict)
    risk_factor_details: dict[str, Any] = field(default_factory=dict)
    diagnosis: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_feature_record(self) -> dict[str, Any]:
        return {
            "active_cognitive_task_score": self.active_cognitive_task_score,
            "active_cognitive_task_level": self.active_cognitive_task_level,
            "active_cognitive_task_factors": list(self.factors),
            "active_cognitive_task_confidence": self.confidence,
            "active_cognitive_task_details": dict(self.risk_factor_details),
            "diagnosis": False,
        }


def score_active_cognitive_tasks(record: Mapping[str, Any]) -> ActiveCognitiveTaskScore:
    """Convert optional active-task raw fields into a normalized risk score."""

    if not isinstance(record, Mapping):
        raise ValueError("active cognitive task record must be an object")

    task_scores = {
        "animal_fluency": _score_animal_fluency(record),
        "countdown": _score_countdown(record),
        "recall": _score_recall(record),
        "picture_description": _score_picture_description(record),
    }
    available = tuple(name for name, detail in task_scores.items() if detail is not None)
    if not available:
        return ActiveCognitiveTaskScore(
            active_cognitive_task_score=None,
            active_cognitive_task_level=0,
            factors=("active_cognitive_task_insufficient_data",),
            task_scores={name: None for name in TASK_WEIGHTS},
        )

    denominator = sum(TASK_WEIGHTS[name] for name in available)
    score = round(
        sum((task_scores[name]["score"] or 0.0) * TASK_WEIGHTS[name] for name in available)
        / denominator,
        4,
    )
    factors: list[str] = []
    for name in available:
        detail = task_scores[name]
        if detail is not None:
            factors.extend(detail["factors"])
    if not factors and score > 0.0:
        factors.append("active_cognitive_task_mild_variation")
    elif not factors:
        factors.append("active_cognitive_task_within_expected_range")

    confidence = _task_confidence(record, available)
    return ActiveCognitiveTaskScore(
        active_cognitive_task_score=score,
        active_cognitive_task_level=_level_from_score(score),
        factors=tuple(dict.fromkeys(factors)),
        confidence=confidence,
        available_tasks=available,
        task_scores={
            name: None if detail is None else detail["score"]
            for name, detail in task_scores.items()
        },
        risk_factor_details={
            name: detail
            for name, detail in task_scores.items()
            if detail is not None
        },
        diagnosis=False,
    )


def build_active_cognitive_task_features(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return feature fields that can be merged into a daily mental-health sample."""

    return score_active_cognitive_tasks(record).to_feature_record()


def _score_animal_fluency(record: Mapping[str, Any]) -> dict[str, Any] | None:
    section = _section(record, "animal_fluency", "animalFluency")
    valid_count = _optional_number(
        record,
        section,
        "animal_fluency_count",
        "valid_animal_count",
        "animal_count",
        minimum=0.0,
    )
    repetition_count = _optional_number(
        record,
        section,
        "animal_fluency_repetition_count",
        "repetition_count",
        "duplicate_count",
        minimum=0.0,
    )
    invalid_count = _optional_number(
        record,
        section,
        "animal_fluency_invalid_count",
        "invalid_count",
        minimum=0.0,
    )
    if valid_count is None and repetition_count is None and invalid_count is None:
        return None

    count_score = _clamp((16.0 - (valid_count or 16.0)) / 10.0)
    error_score = _clamp(((repetition_count or 0.0) + (invalid_count or 0.0)) / 4.0)
    score = round(max(count_score, error_score * 0.75), 4)
    factors = []
    if count_score >= 0.6:
        factors.append("animal_fluency_low_count")
    if error_score >= 0.6:
        factors.append("animal_fluency_repetition_or_invalid")
    return {
        "score": score,
        "factors": factors,
        "raw": {
            "valid_animal_count": valid_count,
            "repetition_count": repetition_count,
            "invalid_count": invalid_count,
        },
    }


def _score_countdown(record: Mapping[str, Any]) -> dict[str, Any] | None:
    section = _section(record, "countdown", "countdown_task")
    error_count = _optional_number(
        record,
        section,
        "countdown_error_count",
        "countdown_errors",
        "error_count",
        minimum=0.0,
    )
    duration = _optional_number(
        record,
        section,
        "countdown_duration_seconds",
        "duration_seconds",
        minimum=0.0,
    )
    completed = _optional_bool(record, section, "countdown_completed", "completed")
    if error_count is None and duration is None and completed is None:
        return None

    error_score = _clamp((error_count or 0.0) / 4.0)
    duration_score = _clamp(((duration or 45.0) - 45.0) / 45.0)
    incomplete_score = 0.8 if completed is False else 0.0
    score = round(max(error_score, duration_score, incomplete_score), 4)
    factors = []
    if error_score >= 0.6:
        factors.append("countdown_error_increase")
    if duration_score >= 0.6:
        factors.append("countdown_slow_response")
    if completed is False:
        factors.append("countdown_incomplete")
    return {
        "score": score,
        "factors": factors,
        "raw": {
            "error_count": error_count,
            "duration_seconds": duration,
            "completed": completed,
        },
    }


def _score_recall(record: Mapping[str, Any]) -> dict[str, Any] | None:
    section = _section(record, "recall", "repeat_task", "repetition_task")
    expected = _optional_number(
        record,
        section,
        "recall_expected_items",
        "expected_items",
        minimum=1.0,
    )
    recalled = _optional_number(
        record,
        section,
        "recall_recalled_items",
        "recalled_items",
        minimum=0.0,
    )
    omissions = _optional_number(
        record,
        section,
        "recall_omission_count",
        "omission_count",
        "missing_count",
        minimum=0.0,
    )
    if expected is None and recalled is None and omissions is None:
        return None
    expected_items = expected or 5.0
    if omissions is None and recalled is not None:
        omissions = max(expected_items - recalled, 0.0)
    omission_score = _clamp((omissions or 0.0) / expected_items)
    score = round(omission_score, 4)
    factors = ["recall_omission_increase"] if omission_score >= 0.6 else []
    return {
        "score": score,
        "factors": factors,
        "raw": {
            "expected_items": expected_items,
            "recalled_items": recalled,
            "omission_count": omissions,
        },
    }


def _score_picture_description(record: Mapping[str, Any]) -> dict[str, Any] | None:
    section = _section(record, "picture_description", "pictureDescription")
    expected = _optional_number(
        record,
        section,
        "picture_expected_information_points",
        "expected_information_points",
        minimum=1.0,
    )
    information_points = _optional_number(
        record,
        section,
        "picture_information_points",
        "information_points",
        "valid_information_points",
        minimum=0.0,
    )
    off_topic = _optional_number(
        record,
        section,
        "picture_off_topic_count",
        "off_topic_count",
        minimum=0.0,
    )
    latency = _optional_number(
        record,
        section,
        "picture_response_latency_seconds",
        "response_latency_seconds",
        minimum=0.0,
    )
    if information_points is None and expected is None and off_topic is None and latency is None:
        return None
    expected_points = expected or 6.0
    info_score = _clamp((expected_points - (information_points or expected_points)) / expected_points)
    off_topic_score = _clamp((off_topic or 0.0) / 3.0)
    latency_score = _clamp(((latency or 10.0) - 10.0) / 20.0)
    score = round(max(info_score, off_topic_score * 0.7, latency_score * 0.6), 4)
    factors = []
    if info_score >= 0.6:
        factors.append("picture_description_low_information")
    if off_topic_score >= 0.6:
        factors.append("picture_description_off_topic")
    if latency_score >= 0.6:
        factors.append("picture_description_response_delay")
    return {
        "score": score,
        "factors": factors,
        "raw": {
            "expected_information_points": expected_points,
            "information_points": information_points,
            "off_topic_count": off_topic,
            "response_latency_seconds": latency,
        },
    }


def _section(record: Mapping[str, Any], *names: str) -> Mapping[str, Any]:
    container = record.get("cognitive_tasks") or record.get("active_cognitive_tasks")
    if isinstance(container, Mapping):
        for name in names:
            value = container.get(name)
            if isinstance(value, Mapping):
                return value
    for name in names:
        value = record.get(name)
        if isinstance(value, Mapping):
            return value
    return {}


def _optional_number(
    record: Mapping[str, Any],
    section: Mapping[str, Any],
    *names: str,
    minimum: float,
) -> float | None:
    found = False
    value: Any = None
    for name in names:
        if name in section:
            value = section.get(name)
            found = True
            break
        if name in record:
            value = record.get(name)
            found = True
            break
    if not found or value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"active cognitive task field '{names[0]}' must be a finite number")
    number = float(value)
    if not math.isfinite(number) or number < minimum:
        raise ValueError(f"active cognitive task field '{names[0]}' must be at least {minimum:g}")
    return number


def _optional_bool(
    record: Mapping[str, Any],
    section: Mapping[str, Any],
    *names: str,
) -> bool | None:
    found = False
    value: Any = None
    for name in names:
        if name in section:
            value = section.get(name)
            found = True
            break
        if name in record:
            value = record.get(name)
            found = True
            break
    if not found or value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"active cognitive task field '{names[0]}' must be a boolean")
    return value


def _task_confidence(record: Mapping[str, Any], available: tuple[str, ...]) -> float:
    raw_quality = record.get("active_cognitive_task_quality")
    if raw_quality is None:
        raw_quality = record.get("cognitive_task_quality")
    quality = 1.0
    if raw_quality is not None:
        if isinstance(raw_quality, bool) or not isinstance(raw_quality, Real):
            raise ValueError("active cognitive task quality must be a finite number in [0, 1]")
        quality = float(raw_quality)
        if not math.isfinite(quality) or not 0.0 <= quality <= 1.0:
            raise ValueError("active cognitive task quality must be a finite number in [0, 1]")
    return round(_clamp((len(available) / len(TASK_WEIGHTS)) * quality), 4)


def _level_from_score(score: float) -> int:
    if score >= DEFAULT_THRESHOLDS[2]:
        return 3
    if score >= DEFAULT_THRESHOLDS[1]:
        return 2
    if score >= DEFAULT_THRESHOLDS[0]:
        return 1
    return 0


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


__all__ = [
    "ActiveCognitiveTaskScore",
    "build_active_cognitive_task_features",
    "score_active_cognitive_tasks",
]
