"""Strict request and response contracts for the explicit R9 candidate API."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from elderly_monitoring.modules.mental_health.mood_social.r8.schemas import R8CandidateRequest


R9_REQUEST_SCHEMA = "mood_social_r9_candidate_request_v1"
R9_RESPONSE_SCHEMA = "mood_social_r9_candidate_response_v1"


class R9CandidateRequest(R8CandidateRequest):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: Literal["mood_social_r9_candidate_request_v1"] = R9_REQUEST_SCHEMA

    def r8_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="python")
        payload["schema_version"] = "mood_social_r8_candidate_request_v1"
        return payload


class _ExpertBase(BaseModel):
    model_config = ConfigDict(extra="allow", allow_inf_nan=False)

    domain: str
    available: bool
    role: str


class DynamicProbabilityExpert(_ExpertBase):
    role: Literal["dynamic_probability"]
    probability_phq_ge5: float | None = Field(default=None, ge=0, le=1)
    probability_phq_ge10: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def probability_order(self):
        if self.probability_phq_ge5 is not None and self.probability_phq_ge10 is not None and self.probability_phq_ge10 > self.probability_phq_ge5:
            raise ValueError("PHQ dual-head monotonicity violation")
        return self


class LocalJointProbabilityExpert(DynamicProbabilityExpert):
    role: Literal["local_joint_probability"]


class HistoryProbabilityExpert(DynamicProbabilityExpert):
    role: Literal["history_probability"]


class AnomalyExpert(_ExpertBase):
    role: Literal["anomaly_only"]


class BackgroundExpert(_ExpertBase):
    role: Literal["background_only"]


class SupportExpert(_ExpertBase):
    role: Literal["support_only", "facial_support"]


ExpertAssessment = Annotated[
    Union[
        DynamicProbabilityExpert,
        LocalJointProbabilityExpert,
        HistoryProbabilityExpert,
        AnomalyExpert,
        BackgroundExpert,
        SupportExpert,
    ],
    Field(discriminator="role"),
]


class R9CandidateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str
    request_id: str
    person_id: str
    target_date: str
    model_version: str
    package_run_id: str
    core_package_run_id: str
    available: bool
    current_state_only: bool
    diagnosis: bool
    expert_assessments: dict[str, ExpertAssessment]
    joint_expert_assessments: dict[str, ExpertAssessment]
    passive_current_state: dict[str, Any]
    facial_affect_assessment: dict[str, Any]
    interaction_enhanced_current_state: dict[str, Any]
    history_informed_integrated_state: dict[str, Any]
    operational_attention_state: dict[str, Any]
    evidence_graph: dict[str, Any]
    rule_fusion: dict[str, Any]
    fusion_mode: str
    evidence_signature: str
    probability_used_sources: list[str]
    supporting_evidence_sources: list[str]
    observed_sources: list[str]
    quality: dict[str, Any]
    freshness: dict[str, Any]
    lineage: dict[str, Any]
    abstain_reason: str | None
    fallback_chain: dict[str, Any]
    fallback_model_version: str | None
    fallback_reason: str | None
    release_status: str
    limitations: list[str]


__all__ = ["ExpertAssessment", "R9CandidateRequest", "R9CandidateResponse", "R9_REQUEST_SCHEMA", "R9_RESPONSE_SCHEMA"]
