"""Strict request/response contracts for the explicit R11 engineering candidate."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator

from elderly_monitoring.modules.mental_health.mood_social.r9.schemas import (
    R9CandidateRequest,
)


R11_REQUEST_SCHEMA = "mood_social_r11_candidate_request_v1"
R11_RESPONSE_SCHEMA = "mood_social_r11_candidate_response_v1"


class R11CandidateRequest(R9CandidateRequest):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    schema_version: Literal["mood_social_r11_candidate_request_v1"] = R11_REQUEST_SCHEMA

    def r9_payload(self) -> dict[str, Any]:
        value = self.model_dump(mode="python")
        value["schema_version"] = "mood_social_r9_candidate_request_v1"
        return value


class R11CandidateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: Literal["mood_social_r11_candidate_response_v1"]
    request_id: str
    person_id: str
    target_date: str
    model_version: Literal["mood-social-current-state-v3.3.3-r11"]
    package_run_id: Literal["MH-20260814-R11-001"]
    core_package_run_id: str
    new_trained_model: Literal[False]
    model_evidence_source: Literal["R9/R7/R8 frozen packages"]
    available: bool
    current_state_only: Literal[True]
    diagnosis: Literal[False]
    expert_assessments: dict[str, dict[str, Any]]
    joint_expert_assessments: dict[str, dict[str, Any]]
    passive_current_state: dict[str, Any]
    history_informed_current_state: dict[str, Any]
    operational_attention_state: dict[str, Any]
    facial_affect_assessment: dict[str, Any]
    selected_probability_output: dict[str, Any] | None
    selected_probability_node: str | None
    probability_used_sources: list[str]
    supporting_evidence_sources: list[str]
    consumed_nodes: list[str]
    suppressed_nodes: list[str]
    reliability_status: Literal["usable", "degraded", "abstain"]
    domain_compatibility: dict[str, bool]
    quality_reasons: dict[str, list[str]]
    evidence_graph: dict[str, Any]
    rule_fusion: dict[str, Any]
    fusion_mode: Literal[
        "mutually_exclusive_sleep_history",
        "mutually_exclusive_activity_sleep",
        "independent_fallback",
    ]
    evidence_signature: str
    observed_sources: list[str]
    quality: dict[str, Any]
    freshness: dict[str, Any]
    lineage: dict[str, Any]
    fallback_chain: dict[str, Any]
    abstain_reason: str | None
    release_status: Literal[
        "engineering-validated/reused-model-evidence/integration-ready/device-validation-pending"
    ]
    limitations: list[str]

    @model_validator(mode="after")
    def semantic_guards(self):
        if self.available != bool(self.operational_attention_state.get("available")):
            raise ValueError("R11 top-level and operational availability differ")
        if self.rule_fusion.get("attention_index_is_phq_probability") is not False:
            raise ValueError("R11 operational score cannot be labelled a PHQ probability")
        if any("facial" in value for value in self.probability_used_sources):
            raise ValueError("facial affect cannot be a PHQ probability source")
        if self.selected_probability_node is None and self.selected_probability_output is not None:
            raise ValueError("selected probability output exists without a selected node")
        if self.selected_probability_output is not None:
            p5 = self.selected_probability_output.get("probability_phq_ge5")
            p10 = self.selected_probability_output.get("probability_phq_ge10")
            if p5 is not None and p10 is not None and float(p10) > float(p5):
                raise ValueError("R11 dual-head monotonicity violation")
        return self


__all__ = [
    "R11CandidateRequest",
    "R11CandidateResponse",
    "R11_REQUEST_SCHEMA",
    "R11_RESPONSE_SCHEMA",
]
