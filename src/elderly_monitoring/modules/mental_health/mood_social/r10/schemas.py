"""Strict contracts for the explicit R10 candidate API."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from elderly_monitoring.modules.mental_health.mood_social.r9.schemas import R9CandidateRequest


R10_REQUEST_SCHEMA = "mood_social_r10_candidate_request_v1"
R10_RESPONSE_SCHEMA = "mood_social_r10_candidate_response_v1"


class R10CandidateRequest(R9CandidateRequest):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    schema_version: Literal["mood_social_r10_candidate_request_v1"] = R10_REQUEST_SCHEMA

    def r9_payload(self) -> dict[str, Any]:
        value = self.model_dump(mode="python")
        value["schema_version"] = "mood_social_r9_candidate_request_v1"
        return value


class _ExpertBase(BaseModel):
    model_config = ConfigDict(extra="allow", allow_inf_nan=False)
    domain: str
    available: bool
    role: str


class ProbabilityExpert(_ExpertBase):
    role: Literal["dynamic_probability", "local_joint_probability", "history_probability"]
    probability_phq_ge5: float | None = Field(default=None, ge=0, le=1)
    probability_phq_ge10: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def order(self):
        if (
            self.probability_phq_ge5 is not None
            and self.probability_phq_ge10 is not None
            and self.probability_phq_ge10 > self.probability_phq_ge5
        ):
            raise ValueError("R10 PHQ dual-head monotonicity violation")
        return self


class AnomalyExpert(_ExpertBase):
    role: Literal["anomaly_only"]


class BackgroundExpert(_ExpertBase):
    role: Literal["background_only"]


class SupportExpert(_ExpertBase):
    role: Literal["support_only", "facial_support"]


ExpertAssessment = Annotated[
    Union[ProbabilityExpert, AnomalyExpert, BackgroundExpert, SupportExpert],
    Field(discriminator="role"),
]


class R10CandidateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    schema_version: Literal["mood_social_r10_candidate_response_v1"]
    request_id: str
    person_id: str
    target_date: str
    model_version: Literal["mood-social-current-state-v3.3.3-r10"]
    package_run_id: str
    core_package_run_id: str
    available: bool
    current_state_only: Literal[True]
    diagnosis: Literal[False]
    expert_assessments: dict[str, ExpertAssessment]
    joint_expert_assessments: dict[str, ExpertAssessment]
    passive_current_state: dict[str, Any]
    interaction_enhanced_current_state: dict[str, Any]
    history_informed_integrated_state: dict[str, Any]
    operational_attention_state: dict[str, Any]
    facial_affect_assessment: dict[str, Any]
    evidence_graph: dict[str, Any]
    rule_fusion: dict[str, Any]
    fusion_mode: Literal[
        "mutually_exclusive_sleep_history",
        "mutually_exclusive_activity_sleep",
        "independent_fallback",
    ]
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

    @model_validator(mode="after")
    def semantics(self):
        if self.available != bool(self.operational_attention_state.get("available")):
            raise ValueError("R10 top-level and operational availability differ")
        if any("facial" in value for value in self.probability_used_sources):
            raise ValueError("facial affect cannot be a PHQ probability source")
        if self.rule_fusion.get("attention_index_is_phq_probability") is not False:
            raise ValueError("R10 rule score must not be labelled as PHQ probability")
        selected = self.evidence_graph.get("selected_probability_node")
        if selected and selected not in self.probability_used_sources:
            raise ValueError("R10 selected joint node missing from probability sources")
        return self


__all__ = [
    "ExpertAssessment",
    "R10CandidateRequest",
    "R10CandidateResponse",
    "R10_REQUEST_SCHEMA",
    "R10_RESPONSE_SCHEMA",
]
