"""Strict R8 request and structured facial-affect evidence contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from elderly_monitoring.modules.mental_health.mood_social.schemas import (
    MoodSocialInferRequest,
)


R8_REQUEST_SCHEMA = "mood_social_r8_candidate_request_v1"
R8_RESPONSE_SCHEMA = "mood_social_r8_candidate_response_v1"
B0_MODEL_VERSION = "facial-affect-causalnet-b0-opt-me-008-deploy-v1"
B0_RUN_LOCK_SHA256 = (
    "e1266924e54631a05ddc90a4d77d5457c8d9acacfce348ccbb49e3f00c536bb2"
)
B0_PACKAGE_SHA256 = (
    "f7ae5cc311b27b9725e43d2598d94b518f3b66b39ac9c191adb83cd0049ef298"
)


class FacialProbabilities(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    negative: float = Field(ge=0.0, le=1.0)
    positive: float = Field(ge=0.0, le=1.0)
    surprise: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_sum(self):
        if abs(self.negative + self.positive + self.surprise - 1.0) > 1e-5:
            raise ValueError("facial probabilities must sum to one")
        return self


class FacialAffectObservation(BaseModel):
    """One sanitized B0 task/segment result; raw media is deliberately absent."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    person_id: str = Field(min_length=1, max_length=128)
    session_id: str = Field(min_length=1, max_length=128)
    task_slot: int | None = Field(default=None, ge=1, le=9)
    segment_id: str | None = Field(default=None, min_length=1, max_length=128)
    observation_context: Literal[
        "active_task_bundle", "phq9_questionnaire_session"
    ]
    observed_at: datetime
    known_at: datetime
    status: Literal["completed", "degraded", "insufficient_input"]
    prediction: Literal["negative", "positive", "surprise"] | None = None
    probabilities: FacialProbabilities | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    quality_score: float = Field(ge=0.0, le=1.0)
    quality_status: Literal["valid", "degraded", "invalid"]
    face_coverage: float = Field(ge=0.0, le=1.0)
    track_continuity: float | None = Field(default=None, ge=0.0, le=1.0)
    multiple_face_conflicts: bool = False
    model_version: Literal[
        "facial-affect-causalnet-b0-opt-me-008-deploy-v1"
    ] = B0_MODEL_VERSION
    package_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    lineage_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("observed_at", "known_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("facial timestamps must include timezone offsets")
        return value

    @model_validator(mode="after")
    def validate_task_and_output(self):
        if self.task_slot is None and self.segment_id is None:
            raise ValueError("task_slot or segment_id is required")
        if self.known_at < self.observed_at:
            raise ValueError("known_at cannot precede observed_at")
        if self.status == "completed":
            if self.prediction is None or self.probabilities is None or self.confidence is None:
                raise ValueError("completed facial observation lacks classifier output")
            values = self.probabilities.model_dump()
            maximum = max(values.values())
            if abs(values[self.prediction] - maximum) > 1e-12:
                raise ValueError("prediction must be an argmax facial class")
        elif any(
            value is not None
            for value in (self.prediction, self.probabilities, self.confidence)
        ):
            raise ValueError("non-completed observation cannot expose classifier output")
        return self

    @property
    def observation_key(self) -> tuple[str, str, int | None, str | None]:
        return (
            self.observation_context,
            self.session_id,
            self.task_slot,
            self.segment_id,
        )


class R8CandidateRequest(MoodSocialInferRequest):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: Literal["mood_social_r8_candidate_request_v1"] = (
        R8_REQUEST_SCHEMA
    )
    inference_cutoff: datetime
    facial_affect_authorized: bool = False
    facial_affect_observations: list[FacialAffectObservation] = Field(
        default_factory=list, max_length=256
    )

    @field_validator("inference_cutoff")
    @classmethod
    def cutoff_requires_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("inference_cutoff must include a timezone offset")
        return value

    @model_validator(mode="after")
    def validate_facial_authorization(self):
        if self.facial_affect_observations and not self.facial_affect_authorized:
            raise ValueError("facial observations require explicit authorization")
        if any(
            item.person_id != self.person_id
            for item in self.facial_affect_observations
        ):
            raise ValueError("facial observation person_id mismatch")
        return self

    def r7_payload(self) -> dict[str, Any]:
        payload = self.model_dump(
            mode="python",
            exclude={
                "inference_cutoff",
                "facial_affect_authorized",
                "facial_affect_observations",
            },
        )
        payload["schema_version"] = "mood_social_infer_request_v3"
        return payload


__all__ = [
    "B0_MODEL_VERSION",
    "B0_PACKAGE_SHA256",
    "B0_RUN_LOCK_SHA256",
    "FacialAffectObservation",
    "FacialProbabilities",
    "R8CandidateRequest",
    "R8_REQUEST_SCHEMA",
    "R8_RESPONSE_SCHEMA",
]
