"""Sanitize and aggregate frozen-B0 output into bounded R8 support evidence."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from enum import IntEnum
from typing import Iterable

from .schemas import (
    B0_PACKAGE_SHA256,
    B0_RUN_LOCK_SHA256,
    FacialAffectObservation,
)


QUALITY_SCORE_MIN = 0.65
CONFIDENCE_MIN = 0.60
LOOKBACK_DAYS = 14


class FacialEvidenceLevel(IntEnum):
    UNAVAILABLE = -1
    OBSERVATION = 0
    F1 = 1
    F2 = 2


@dataclass(frozen=True)
class SessionAssessment:
    context: str
    session_id: str
    observed_date: str
    valid_task_count: int
    status: str
    provisional_negative: bool
    within_session_discordance: bool
    mean_quality: float | None
    mean_confidence: float | None
    task_keys: tuple[str, ...]


def _task_key(item: FacialAffectObservation) -> str:
    return str(item.task_slot) if item.task_slot is not None else str(item.segment_id)


def _is_lineage_allowed(item: FacialAffectObservation) -> bool:
    return (
        item.run_lock_sha256 == B0_RUN_LOCK_SHA256
        and item.package_sha256 == B0_PACKAGE_SHA256
    )


def _is_quality_eligible(item: FacialAffectObservation) -> bool:
    return bool(
        item.status == "completed"
        and item.quality_status == "valid"
        and item.quality_score >= QUALITY_SCORE_MIN
        and item.confidence is not None
        and item.confidence >= CONFIDENCE_MIN
        and not item.multiple_face_conflicts
        and item.prediction is not None
        and item.probabilities is not None
    )


def _assessment_payload(
    *,
    level: FacialEvidenceLevel,
    sessions: list[SessionAssessment],
    rejected: dict[str, int],
    duplicate_count: int,
    context_counts: dict[str, int],
) -> dict[str, object]:
    active = [item for item in sessions if item.context == "active_task_bundle"]
    provisional = [item for item in active if item.provisional_negative]
    observed_dates = sorted({item.observed_date for item in provisional})
    available = bool(active)
    label = {
        FacialEvidenceLevel.UNAVAILABLE: "unavailable",
        FacialEvidenceLevel.OBSERVATION: "observation_only",
        FacialEvidenceLevel.F1: "F1",
        FacialEvidenceLevel.F2: "F2",
    }[level]
    return {
        "domain": "facial_affect",
        "available": available,
        "evidence_level": int(level),
        "evidence_label": label,
        "supports_rule_vote": level in {FacialEvidenceLevel.F1, FacialEvidenceLevel.F2},
        "phq_probability_available": False,
        "valid_session_count": len(active),
        "provisional_negative_session_count": len(provisional),
        "provisional_negative_day_count": len(observed_dates),
        "context_counts": dict(sorted(context_counts.items())),
        "sessions": [asdict(item) for item in sessions],
        "rejected_counts": dict(sorted(rejected.items())),
        "deduplicated_observation_count": duplicate_count,
        "quality_thresholds": {
            "quality_score_min": QUALITY_SCORE_MIN,
            "confidence_min": CONFIDENCE_MIN,
            "lookback_days": LOOKBACK_DAYS,
        },
        "model_version": "facial-affect-causalnet-b0-opt-me-008-deploy-v1",
        "limitations": [
            "negative/positive/surprise are facial-affect classes, not PHQ probabilities",
            "no paired real-device PHQ-video training data are available",
            "facial evidence cannot independently produce L2 or L3",
        ],
    }


def aggregate_facial_affect(
    observations: Iterable[FacialAffectObservation],
    *,
    person_id: str,
    inference_cutoff: datetime,
    authorized: bool,
) -> dict[str, object]:
    """Return deterministic R8 evidence while rejecting leakage and weak inputs."""

    rejected: defaultdict[str, int] = defaultdict(int)
    context_counts: defaultdict[str, int] = defaultdict(int)
    if not authorized:
        rejected["authorization_missing"] += sum(1 for _ in observations)
        return _assessment_payload(
            level=FacialEvidenceLevel.UNAVAILABLE,
            sessions=[],
            rejected=dict(rejected),
            duplicate_count=0,
            context_counts={},
        )

    cutoff_date = inference_cutoff.date()
    earliest = cutoff_date - timedelta(days=LOOKBACK_DAYS - 1)
    deduplicated: dict[
        tuple[str, str, int | None, str | None], FacialAffectObservation
    ] = {}
    duplicate_count = 0
    for item in observations:
        if item.person_id != person_id:
            rejected["person_mismatch"] += 1
            continue
        if item.observed_at > inference_cutoff or item.known_at > inference_cutoff:
            rejected["after_inference_cutoff"] += 1
            continue
        if item.observed_at.date() < earliest:
            rejected["expired"] += 1
            continue
        if not _is_lineage_allowed(item):
            rejected["unapproved_b0_lineage"] += 1
            continue
        previous = deduplicated.get(item.observation_key)
        if previous is not None:
            if previous.model_dump(mode="json") != item.model_dump(mode="json"):
                rejected["conflicting_duplicate"] += 1
            else:
                duplicate_count += 1
            continue
        deduplicated[item.observation_key] = item

    grouped: defaultdict[tuple[str, str], list[FacialAffectObservation]] = defaultdict(list)
    for item in deduplicated.values():
        context_counts[item.observation_context] += 1
        if not _is_quality_eligible(item):
            if item.status != "completed":
                rejected["not_completed"] += 1
            elif item.quality_status != "valid" or item.quality_score < QUALITY_SCORE_MIN:
                rejected["low_quality"] += 1
            elif item.confidence is None or item.confidence < CONFIDENCE_MIN:
                rejected["low_confidence"] += 1
            elif item.multiple_face_conflicts:
                rejected["multiple_face_conflict"] += 1
            continue
        grouped[(item.observation_context, item.session_id)].append(item)

    sessions: list[SessionAssessment] = []
    for (context, session_id), items in sorted(grouped.items()):
        items = sorted(items, key=lambda item: _task_key(item))
        predictions = {str(item.prediction) for item in items}
        discordant = len(predictions) > 1
        provisional = bool(
            context == "active_task_bundle"
            and len(items) >= 2
            and predictions == {"negative"}
        )
        status = (
            "future_data_pending"
            if context == "phq9_questionnaire_session"
            else "provisional_negative"
            if provisional
            else "within_session_discordance"
            if discordant
            else "observation_only"
        )
        sessions.append(
            SessionAssessment(
                context=context,
                session_id=session_id,
                observed_date=max(item.observed_at for item in items).date().isoformat(),
                valid_task_count=len(items),
                status=status,
                provisional_negative=provisional,
                within_session_discordance=discordant,
                mean_quality=sum(item.quality_score for item in items) / len(items),
                mean_confidence=sum(float(item.confidence) for item in items) / len(items),
                task_keys=tuple(_task_key(item) for item in items),
            )
        )

    active_sessions = [item for item in sessions if item.context == "active_task_bundle"]
    negative_sessions = [item for item in active_sessions if item.provisional_negative]
    negative_days = {item.observed_date for item in negative_sessions}
    negative_ratio = (
        len(negative_sessions) / len(active_sessions) if active_sessions else 0.0
    )
    if (
        len(active_sessions) >= 3
        and len(negative_sessions) >= 2
        and len(negative_days) >= 2
        and negative_ratio >= 2.0 / 3.0
    ):
        level = FacialEvidenceLevel.F2
    elif len(negative_sessions) >= 2 and len(negative_days) >= 2:
        level = FacialEvidenceLevel.F1
    elif active_sessions:
        level = FacialEvidenceLevel.OBSERVATION
    else:
        level = FacialEvidenceLevel.UNAVAILABLE
    return _assessment_payload(
        level=level,
        sessions=sessions,
        rejected=dict(rejected),
        duplicate_count=duplicate_count,
        context_counts=dict(context_counts),
    )


class FacialAffectSupportAdapter:
    """Small injectable wrapper used by the R8 candidate runtime."""

    def assess(
        self,
        observations: Iterable[FacialAffectObservation],
        *,
        person_id: str,
        inference_cutoff: datetime,
        authorized: bool,
    ) -> dict[str, object]:
        return aggregate_facial_affect(
            observations,
            person_id=person_id,
            inference_cutoff=inference_cutoff,
            authorized=authorized,
        )


__all__ = [
    "CONFIDENCE_MIN",
    "FacialAffectSupportAdapter",
    "FacialEvidenceLevel",
    "LOOKBACK_DAYS",
    "QUALITY_SCORE_MIN",
    "SessionAssessment",
    "aggregate_facial_affect",
]
