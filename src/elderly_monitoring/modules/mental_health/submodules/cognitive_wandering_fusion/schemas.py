"""Strict contracts for cognitive-change and wandering rule fusion."""

from __future__ import annotations

from datetime import date
from typing import Annotated, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.v35_schemas import (
    CognitiveV35InferResponse,
)


FUSION_REQUEST_SCHEMA_VERSION = "cognitive_wandering_fusion_request_v1"
FUSION_RESPONSE_SCHEMA_VERSION = "cognitive_wandering_fusion_response_v1"
FUSION_POLICY_VERSION = "cognitive-wandering-rule-fusion-v1"
FUSION_DEPLOYMENT_STATUS = "local_candidate_pending_home_validation"
COGNITIVE_WANDERING_FUSION_PATH = (
    "/v1/mental-health/cognitive-change/wandering-fusion/candidate-infer"
)

AttentionLevel = Literal["normal", "attention", "high_attention"]
FusionStatus = Literal["completed", "degraded", "insufficient_evidence"]
WanderingStatus = Literal["ready", "warming_up", "unavailable"]
Probability = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
NonNegativeFloat = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Token = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")]

SHAPE_NAMES = frozenset({"direct", "pacing", "lapping", "random", "wandering_like"})
TIER_NAMES = frozenset({"high_confidence", "uncertain", "unavailable", "error"})
CONTEXT_NAMES = frozenset(
    {
        "phone_call",
        "searching",
        "cleaning",
        "exercise",
        "social",
        "other",
        "unknown",
        "not_reviewed",
        "not_triggered",
    }
)
DEVIATION_METRICS = frozenset(
    {
        "tracking_coverage",
        "presence_hours",
        "direct_count_per_presence_hour",
        "pacing_count_per_presence_hour",
        "lapping_count_per_presence_hour",
        "random_count_per_presence_hour",
        "wandering_like_count_per_presence_hour",
        "wandering_like_duration_seconds_per_presence_hour",
        "uncertain_wandering_like_count_per_presence_hour",
        "night_wandering_like_ratio",
    }
)


class FusionSchemaModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class WanderingIdentity(FusionSchemaModel):
    model_id: str | None
    model_sha256: Sha256 | None
    config_id: Token
    config_sha256: Sha256
    policy_id: Token
    policy_sha256: Sha256 | None


class WanderingSourceRef(FusionSchemaModel):
    ref_type: str = Field(min_length=1)
    ref_id: str = Field(min_length=1)
    artifact_path: str | None
    sha256: Sha256 | None


class WanderingShapeCounts(FusionSchemaModel):
    direct: int = Field(ge=0)
    pacing: int = Field(ge=0)
    lapping: int = Field(ge=0)
    random: int = Field(ge=0)
    wandering_like: int = Field(ge=0)


class WanderingShapeDurations(FusionSchemaModel):
    direct: NonNegativeFloat
    pacing: NonNegativeFloat
    lapping: NonNegativeFloat
    random: NonNegativeFloat
    wandering_like: NonNegativeFloat


class WanderingConfidenceTier(FusionSchemaModel):
    count: int = Field(ge=0)
    duration_seconds: NonNegativeFloat


class WanderingConfidenceTiers(FusionSchemaModel):
    high_confidence: WanderingConfidenceTier
    uncertain: WanderingConfidenceTier
    unavailable: WanderingConfidenceTier
    error: WanderingConfidenceTier


class WanderingRates(FusionSchemaModel):
    wandering_like_count: NonNegativeFloat | None
    wandering_like_duration_seconds: NonNegativeFloat | None


class WanderingDailyReport(FusionSchemaModel):
    schema_version: Literal["wandering-handoff-daily-report-v2"]
    module: Literal["mental_health"]
    record_id: Token
    person_id: Token
    session_id: Token | None
    source_video_id: Token | None
    local_date: date
    timezone: str = Field(min_length=1)
    status: Literal["ready", "uncertain", "unavailable", "error"]
    presence_seconds: NonNegativeFloat | None
    tracking_coverage_seconds: NonNegativeFloat | None
    tracking_coverage: Probability | None
    qc_coverage: Probability | None
    episode_counts: WanderingShapeCounts
    episode_duration_seconds: WanderingShapeDurations
    confidence_tiers: WanderingConfidenceTiers
    rates_per_presence_hour: WanderingRates
    night_wandering_like_ratio: Probability | None
    context_counts: dict[str, int]
    unavailable_count: int = Field(ge=0)
    error_count: int = Field(ge=0)
    baseline_readiness: Literal[
        "warming_up", "initial_ready", "stable_ready", "unavailable"
    ]
    quality_flags: list[str]
    identity: WanderingIdentity
    source_refs: list[WanderingSourceRef] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_contract(self) -> "WanderingDailyReport":
        if set(self.context_counts) - CONTEXT_NAMES:
            raise ValueError("wandering context label drifted")
        if any(value < 0 for value in self.context_counts.values()):
            raise ValueError("wandering context counts must be non-negative")
        if len(self.quality_flags) != len(set(self.quality_flags)):
            raise ValueError("wandering quality flags must be unique")
        if self.unavailable_count != self.confidence_tiers.unavailable.count:
            raise ValueError("wandering unavailable count drifted")
        if self.error_count != self.confidence_tiers.error.count:
            raise ValueError("wandering error count drifted")
        if self.presence_seconds is None:
            if self.tracking_coverage_seconds is not None or self.tracking_coverage is not None:
                raise ValueError("wandering coverage requires presence")
        elif (
            self.tracking_coverage_seconds is not None
            and self.tracking_coverage_seconds > self.presence_seconds + 1e-6
        ):
            raise ValueError("wandering tracking seconds exceed presence")
        return self


class WanderingDeviationMetric(FusionSchemaModel):
    status: Literal[
        "ready", "warming_up", "observation_unavailable", "reference_unavailable"
    ]
    observation_value: float | None = Field(default=None, allow_inf_nan=False)
    reference_count: int = Field(ge=0)
    reference_median: float | None = Field(default=None, allow_inf_nan=False)
    reference_p90: float | None = Field(default=None, allow_inf_nan=False)
    delta_from_median: float | None = Field(default=None, allow_inf_nan=False)
    delta_from_p90: float | None = Field(default=None, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_ready_shape(self) -> "WanderingDeviationMetric":
        comparisons = (self.delta_from_median, self.delta_from_p90)
        if self.status == "ready":
            if (
                self.observation_value is None
                or self.reference_count < 1
                or self.reference_median is None
                or self.reference_p90 is None
                or any(value is None for value in comparisons)
            ):
                raise ValueError("ready wandering deviation metric is incomplete")
        elif any(value is not None for value in comparisons):
            raise ValueError("non-ready wandering deviation metric carries deltas")
        return self


class WanderingBaselineDeviation(FusionSchemaModel):
    schema_version: Literal["wandering-handoff-baseline-deviation-v1"]
    module: Literal["mental_health"]
    record_id: Token
    deviation_id: Token
    profile_id: Token | None
    person_id: Token
    session_id: Token | None
    source_video_id: Token | None
    local_date: date
    timezone: str = Field(min_length=1)
    status: Literal["ready", "uncertain", "unavailable", "error"]
    readiness_status: Literal[
        "warming_up", "initial_ready", "stable_ready", "unavailable"
    ]
    metrics: dict[str, WanderingDeviationMetric]
    quality_flags: list[str]
    identity: WanderingIdentity
    source_refs: list[WanderingSourceRef] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_contract(self) -> "WanderingBaselineDeviation":
        if self.record_id != self.deviation_id:
            raise ValueError("wandering record/deviation id mismatch")
        if set(self.metrics) != DEVIATION_METRICS:
            raise ValueError("wandering deviation metric set drifted")
        if len(self.quality_flags) != len(set(self.quality_flags)):
            raise ValueError("wandering deviation quality flags must be unique")
        return self


class CognitiveFusionEvidence(FusionSchemaModel):
    observed_at: AwareDatetime | None
    result: CognitiveV35InferResponse | None
    unavailable_reason: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_availability(self) -> "CognitiveFusionEvidence":
        if self.result is None:
            if self.observed_at is not None or self.unavailable_reason is None:
                raise ValueError("missing cognitive result requires only an unavailable reason")
        elif self.observed_at is None or self.unavailable_reason is not None:
            raise ValueError("available cognitive result requires observed_at only")
        return self


class CognitiveWanderingFusionRequest(FusionSchemaModel):
    schema_version: Literal["cognitive_wandering_fusion_request_v1"]
    request_id: Token
    person_id: Token
    evaluated_at: AwareDatetime
    cognitive: CognitiveFusionEvidence
    wandering_daily_reports: list[WanderingDailyReport] = Field(max_length=31)
    wandering_baseline_deviations: list[WanderingBaselineDeviation] = Field(max_length=31)
    policy_version: Literal["cognitive-wandering-rule-fusion-v1"]

    @model_validator(mode="after")
    def validate_identity_and_pairing(self) -> "CognitiveWanderingFusionRequest":
        daily_by_date: dict[date, WanderingDailyReport] = {}
        for row in self.wandering_daily_reports:
            if row.person_id != self.person_id:
                raise ValueError("wandering daily person_id does not match request")
            if row.local_date in daily_by_date:
                raise ValueError("duplicate wandering daily local_date")
            daily_by_date[row.local_date] = row
        deviation_dates: set[date] = set()
        for row in self.wandering_baseline_deviations:
            if row.person_id != self.person_id:
                raise ValueError("wandering deviation person_id does not match request")
            if row.local_date in deviation_dates:
                raise ValueError("duplicate wandering deviation local_date")
            deviation_dates.add(row.local_date)
            daily = daily_by_date.get(row.local_date)
            if daily is None:
                raise ValueError("wandering deviation has no matching daily report")
            if row.timezone != daily.timezone:
                raise ValueError("wandering daily/deviation timezone mismatch")
            if row.identity != daily.identity:
                raise ValueError("wandering daily/deviation identity mismatch")
            if row.readiness_status != daily.baseline_readiness:
                raise ValueError("wandering daily/deviation readiness mismatch")
        timezones = {row.timezone for row in self.wandering_daily_reports}
        if len(timezones) > 1:
            raise ValueError("one fusion request cannot mix wandering timezones")
        return self


class CognitiveComponentSummary(FusionSchemaModel):
    usable: bool
    source_status: str
    source_request_id: str | None
    observed_at: AwareDatetime | None
    age_days: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    score: float | None = Field(default=None, ge=0.0, le=100.0, allow_inf_nan=False)
    level: AttentionLevel | None
    confidence: Probability | None
    adaptation_mode: str | None
    model_version: str | None


class WanderingComponentSummary(FusionSchemaModel):
    status: WanderingStatus
    support_level: int | None = Field(default=None, ge=0, le=2)
    usable_day_count: int = Field(ge=0)
    baseline_ready_day_count: int = Field(ge=0)
    elevated_day_count: int = Field(ge=0)
    elevated_dates: list[date]
    ignored_dates: list[date]
    latest_usable_date: date | None
    source_model_ids: list[str]


class CognitiveWanderingFusionResponse(FusionSchemaModel):
    schema_version: Literal["cognitive_wandering_fusion_response_v1"]
    request_id: Token
    person_id: Token
    evaluated_at: AwareDatetime
    status: FusionStatus
    evidence_status: Literal[
        "cognitive_and_wandering",
        "cognitive_only",
        "wandering_only",
        "insufficient",
    ]
    attention_level: AttentionLevel | None
    attention_score: float | None = Field(default=None, ge=0.0, le=100.0)
    attention_score_semantics: Literal["rule-derived-not-diagnostic-probability"]
    cognitive_component: CognitiveComponentSummary
    wandering_component: WanderingComponentSummary
    decision_reasons: list[str]
    discordance_flags: list[str]
    warnings: list[str]
    recommendation: str
    diagnosis: Literal[False] = False
    research_outputs: None = None
    cognitive_model_version: str | None
    fusion_policy_version: Literal["cognitive-wandering-rule-fusion-v1"]
    fusion_policy_config_sha256: Sha256
    rule_contract_sha256: Sha256
    deployment_status: Literal["local_candidate_pending_home_validation"]

    @model_validator(mode="after")
    def validate_result_shape(self) -> "CognitiveWanderingFusionResponse":
        if self.status == "insufficient_evidence":
            if self.attention_level is not None or self.attention_score is not None:
                raise ValueError("insufficient fusion result cannot expose attention")
        elif self.attention_level is None or self.attention_score is None:
            raise ValueError("available fusion result requires attention")
        return self


__all__ = [
    "COGNITIVE_WANDERING_FUSION_PATH",
    "FUSION_DEPLOYMENT_STATUS",
    "FUSION_POLICY_VERSION",
    "FUSION_REQUEST_SCHEMA_VERSION",
    "FUSION_RESPONSE_SCHEMA_VERSION",
    "AttentionLevel",
    "CognitiveComponentSummary",
    "CognitiveFusionEvidence",
    "CognitiveWanderingFusionRequest",
    "CognitiveWanderingFusionResponse",
    "WanderingBaselineDeviation",
    "WanderingComponentSummary",
    "WanderingDailyReport",
]

