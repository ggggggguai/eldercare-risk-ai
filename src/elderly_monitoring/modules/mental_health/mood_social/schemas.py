from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from typing import Annotated, Literal
from zoneinfo import ZoneInfo

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError


REQUEST_SCHEMA_VERSION = "mood_social_infer_request_v3"
RESPONSE_SCHEMA_VERSION = "mood_social_infer_response_v3"
ERROR_SCHEMA_VERSION = "mood_social_error_v1"
MOOD_SOCIAL_MODULE = "mood_social_attention"
MOOD_SOCIAL_MODEL_VERSION = "mood-fusion-v3.3.3"
MOOD_SOCIAL_TIMEZONE = "Asia/Shanghai"

AvailableSource = Literal["camera", "sleep_device", "s10"]
UsedSource = Literal["camera", "sleep_device", "s10", "profile"]
CoveredDomain = Literal[
    "activity",
    "sleep",
    "physiology",
    "activity_sleep_joint",
    "social_context",
    "personal_change_activity",
    "personal_change_sleep",
    "personal_change_social",
]
EvidenceScope = Literal[
    "proxy_label_supported",
    "engineering_proxy",
    "insufficient_data",
]
Trend = Literal["rising", "falling", "stable", "unknown"]
MoodSocialErrorCode = Literal[
    "AUTHENTICATION_FAILED",
    "UNSUPPORTED_SCHEMA_VERSION",
    "FIELD_VALIDATION_ERROR",
    "DATE_VALIDATION_ERROR",
    "SOURCE_DECLARATION_MISMATCH",
    "MODEL_ARTIFACT_UNAVAILABLE",
    "INTERNAL_ERROR",
]

FACTOR_LABELS = {
    "observed_daytime_activity_low": "有效观测时段内白天活动偏低",
    "observed_low_activity_time_high": "有效观测时段内低活动时间偏高",
    "activity_rhythm_irregular": "日间活动节律不规则",
    "sleep_efficiency_low": "睡眠效率偏低",
    "sleep_fragmentation_high": "夜间睡眠较零散",
    "sleep_schedule_irregular": "睡眠时点不规则",
    "night_exit_frequent": "夜间离床较频繁",
    "night_physiology_attention": "夜间生理状态值得关注",
    "social_context_attention": "社会背景信息值得关注",
    "sedentary_time_increase": "低活动时间增加",
    "sleep_efficiency_drop": "睡眠效率下降",
    "sleep_fragmentation_increase": "夜间睡眠更零散",
    "sleep_schedule_change": "睡眠时点变化",
    "night_exit_increase": "夜间离床增加",
    "night_physiology_change": "夜间生理趋势变化",
    "contact_frequency_drop": "联系频次减少",
    "contact_duration_drop": "有效联系时长减少",
    "personal_activity_deviation": "活动状态偏离个人历史",
    "personal_sleep_deviation": "睡眠状态偏离个人历史",
    "personal_social_deviation": "社会联系偏离个人历史",
}

StrictBool = Annotated[bool, Field(strict=True)]
DiagnosisFalse = Annotated[
    bool,
    Field(strict=True, json_schema_extra={"const": False}),
]
StrictNonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
FiniteNumber = Annotated[
    float,
    Field(strict=True, allow_inf_nan=False),
]
Probability = Annotated[
    float,
    Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False),
]


def _require_non_blank_text(value: str) -> str:
    if not value.strip():
        raise ValueError("must contain non-whitespace text")
    return value


NonEmptyText = Annotated[
    str,
    StringConstraints(strict=True, min_length=1),
    AfterValidator(_require_non_blank_text),
]
StrictText = Annotated[str, StringConstraints(strict=True)]
PersonId = Annotated[
    str,
    StringConstraints(strict=True, min_length=1, max_length=64),
]
DeviceId = Annotated[
    str,
    StringConstraints(strict=True, min_length=1, max_length=64),
]
SceneVersion = Annotated[
    str,
    StringConstraints(strict=True, min_length=1, max_length=80),
]
RequestId = Annotated[
    str,
    StringConstraints(
        strict=True,
        pattern=r"^[A-Za-z0-9_.:-]{1,80}$",
    ),
]
StableCode = Annotated[
    str,
    StringConstraints(
        strict=True,
        pattern=r"^[a-z][a-z0-9_]*$",
    ),
]


def _parse_date_only(value: object) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if not isinstance(value, str) or re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is None:
        raise PydanticCustomError(
            "date_validation_error",
            "must use YYYY-MM-DD and must not include a time",
        )
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise PydanticCustomError(
            "date_validation_error",
            "must be a valid calendar date",
        ) from exc
    if parsed.isoformat() != value:
        raise PydanticCustomError(
            "date_validation_error",
            "must use zero-padded YYYY-MM-DD",
        )
    return parsed


DateOnly = Annotated[date, BeforeValidator(_parse_date_only)]


def _contract_error(
    error_type: str,
    reason: str,
    *,
    field: str,
) -> PydanticCustomError:
    return PydanticCustomError(
        error_type,
        "{reason}",
        {"reason": reason, "field": field},
    )


def _is_close(
    left: float | Decimal,
    right: float | Decimal,
    *,
    tolerance: float,
) -> bool:
    return (
        abs(Decimal(str(left)) - Decimal(str(right)))
        <= Decimal(str(tolerance))
    )


class MoodSocialSchemaModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MoodSocialProfile(MoodSocialSchemaModel):
    age_group: Literal["60_69", "70_79", "80_plus"] | None = None
    sex: Literal["female", "male"] | None = None
    living_arrangement: (
        Literal["alone", "with_family", "institution", "other"] | None
    ) = None
    marital_status: Literal["partnered", "not_partnered"] | None = None
    chronic_disease_count: StrictNonNegativeInt | None = None
    self_rated_health: Annotated[int, Field(strict=True, ge=1, le=5)] | None = None
    functional_limitation: Annotated[int, Field(strict=True, ge=0, le=3)] | None = None
    social_participation_days_per_week: (
        Annotated[int, Field(strict=True, ge=0, le=7)] | None
    ) = None
    education_level: (
        Literal["primary_or_less", "middle", "high_or_above"] | None
    ) = None
    economic_status: Literal["low", "middle", "high"] | None = None


class CameraGaitMetric(MoodSocialSchemaModel):
    camera_id: DeviceId
    scene_version: SceneVersion
    gait_speed_image_norm_per_sec: (
        Annotated[float, Field(strict=True, ge=0.0, allow_inf_nan=False)] | None
    ) = None
    sit_to_stand_duration_seconds: (
        Annotated[float, Field(strict=True, gt=0.0, allow_inf_nan=False)] | None
    ) = None
    turn_duration_seconds: (
        Annotated[float, Field(strict=True, gt=0.0, allow_inf_nan=False)] | None
    ) = None
    postural_stability: Probability | None = None

    @model_validator(mode="after")
    def require_observed_motion_value(self) -> "CameraGaitMetric":
        motion_values = (
            self.gait_speed_image_norm_per_sec,
            self.sit_to_stand_duration_seconds,
            self.turn_duration_seconds,
            self.postural_stability,
        )
        if all(value is None for value in motion_values):
            raise _contract_error(
                "field_validation_error",
                "at least one camera gait metric must be non-null",
                field="camera_gait_metrics",
            )
        return self


class MoodSocialActivity(MoodSocialSchemaModel):
    daytime_active_minutes: (
        Annotated[float, Field(strict=True, ge=0.0, le=720.0, allow_inf_nan=False)]
        | None
    ) = None
    weighted_daytime_activity: (
        Annotated[float, Field(strict=True, ge=0.0, le=720.0, allow_inf_nan=False)]
        | None
    ) = None
    valid_daytime_detection_minutes: Annotated[
        float,
        Field(strict=True, ge=0.0, le=720.0, allow_inf_nan=False),
    ]
    low_activity_minutes: (
        Annotated[float, Field(strict=True, ge=0.0, le=720.0, allow_inf_nan=False)]
        | None
    ) = None
    sedentary_bout_total_minutes: (
        Annotated[float, Field(strict=True, ge=0.0, le=720.0, allow_inf_nan=False)]
        | None
    ) = None
    longest_sedentary_bout_minutes: (
        Annotated[float, Field(strict=True, ge=0.0, le=720.0, allow_inf_nan=False)]
        | None
    ) = None
    activity_peak_minute_of_day: (
        Annotated[int, Field(strict=True, ge=360, le=1079)] | None
    ) = None
    hourly_activity_intensity: Annotated[
        list[Probability | None],
        Field(min_length=24, max_length=24),
    ]
    hourly_valid_detection_minutes: Annotated[
        list[
            Annotated[
                float,
                Field(strict=True, ge=0.0, le=60.0, allow_inf_nan=False),
            ]
        ],
        Field(min_length=24, max_length=24),
    ]
    camera_gait_metrics: list[CameraGaitMetric]

    @model_validator(mode="after")
    def validate_observation_semantics(self) -> "MoodSocialActivity":
        valid_minutes = self.valid_daytime_detection_minutes
        nullable_daily_fields = (
            self.daytime_active_minutes,
            self.weighted_daytime_activity,
            self.low_activity_minutes,
            self.sedentary_bout_total_minutes,
            self.longest_sedentary_bout_minutes,
            self.activity_peak_minute_of_day,
        )
        if valid_minutes == 0.0:
            if any(value is not None for value in nullable_daily_fields):
                raise _contract_error(
                    "field_validation_error",
                    "zero observation coverage requires null activity semantics",
                    field="valid_daytime_detection_minutes",
                )
            if any(minutes != 0.0 for minutes in self.hourly_valid_detection_minutes):
                raise _contract_error(
                    "field_validation_error",
                    "zero observation coverage requires 24 zero hourly coverage values",
                    field="hourly_valid_detection_minutes",
                )
            if any(value is not None for value in self.hourly_activity_intensity):
                raise _contract_error(
                    "field_validation_error",
                    "zero observation coverage requires 24 null hourly intensities",
                    field="hourly_activity_intensity",
                )
            if self.camera_gait_metrics:
                raise _contract_error(
                    "field_validation_error",
                    "zero observation coverage requires an empty gait metric array",
                    field="camera_gait_metrics",
                )
            return self

        for hour, (coverage, intensity) in enumerate(
            zip(
                self.hourly_valid_detection_minutes,
                self.hourly_activity_intensity,
                strict=True,
            )
        ):
            outside_daytime = hour < 6 or hour >= 18
            if outside_daytime and (coverage != 0.0 or intensity is not None):
                raise _contract_error(
                    "field_validation_error",
                    "hours outside 06:00-18:00 require zero coverage and null intensity",
                    field=f"hourly_valid_detection_minutes[{hour}]",
                )
            if coverage == 0.0 and intensity is not None:
                raise _contract_error(
                    "field_validation_error",
                    "zero hourly coverage requires null intensity",
                    field=f"hourly_activity_intensity[{hour}]",
                )
            if coverage > 0.0 and intensity is None:
                raise _contract_error(
                    "field_validation_error",
                    "positive hourly coverage requires an observed intensity",
                    field=f"hourly_activity_intensity[{hour}]",
                )

        hourly_valid_total = sum(
            (
                Decimal(str(minutes))
                for minutes in self.hourly_valid_detection_minutes
            ),
            start=Decimal("0"),
        )
        if hourly_valid_total <= Decimal("0"):
            raise _contract_error(
                "field_validation_error",
                "positive daily observation coverage requires positive hourly coverage",
                field="hourly_valid_detection_minutes",
            )
        if not _is_close(hourly_valid_total, valid_minutes, tolerance=0.1):
            raise _contract_error(
                "field_validation_error",
                "hourly coverage sum must match valid_daytime_detection_minutes within 0.1",
                field="hourly_valid_detection_minutes",
            )

        if (
            self.daytime_active_minutes is not None
            and self.daytime_active_minutes > valid_minutes
        ):
            raise _contract_error(
                "field_validation_error",
                "daytime_active_minutes must not exceed valid observation minutes",
                field="daytime_active_minutes",
            )
        if (
            self.weighted_daytime_activity is not None
            and self.weighted_daytime_activity > valid_minutes
        ):
            raise _contract_error(
                "field_validation_error",
                "weighted_daytime_activity must not exceed valid observation minutes",
                field="weighted_daytime_activity",
            )
        if self.low_activity_minutes is not None and self.low_activity_minutes > valid_minutes:
            raise _contract_error(
                "field_validation_error",
                "low_activity_minutes must not exceed valid observation minutes",
                field="low_activity_minutes",
            )

        longest = self.longest_sedentary_bout_minutes
        total = self.sedentary_bout_total_minutes
        low = self.low_activity_minutes
        if total is not None and 0.0 < total < 30.0:
            raise _contract_error(
                "field_validation_error",
                "positive sedentary bout total must be at least 30 minutes",
                field="sedentary_bout_total_minutes",
            )
        if longest is not None and 0.0 < longest < 30.0:
            raise _contract_error(
                "field_validation_error",
                "positive longest sedentary bout must be at least 30 minutes",
                field="longest_sedentary_bout_minutes",
            )
        if total is not None and total > valid_minutes:
            raise _contract_error(
                "field_validation_error",
                "total sedentary bout minutes must not exceed valid observation minutes",
                field="sedentary_bout_total_minutes",
            )
        if longest is not None and longest > valid_minutes:
            raise _contract_error(
                "field_validation_error",
                "longest sedentary bout must not exceed valid observation minutes",
                field="longest_sedentary_bout_minutes",
            )
        if longest is not None and total is not None and longest > total:
            raise _contract_error(
                "field_validation_error",
                "longest sedentary bout must not exceed total sedentary bout minutes",
                field="longest_sedentary_bout_minutes",
            )
        if total is not None and low is not None and total > low:
            raise _contract_error(
                "field_validation_error",
                "total sedentary bout minutes must not exceed low activity minutes",
                field="sedentary_bout_total_minutes",
            )
        if longest is not None and low is not None and longest > low:
            raise _contract_error(
                "field_validation_error",
                "longest sedentary bout must not exceed low activity minutes",
                field="longest_sedentary_bout_minutes",
            )

        if self.weighted_daytime_activity is not None:
            hourly_weighted = sum(
                (
                    Decimal(str(coverage))
                    * Decimal(
                        str(intensity if intensity is not None else 0.0)
                    )
                    for coverage, intensity in zip(
                        self.hourly_valid_detection_minutes,
                        self.hourly_activity_intensity,
                        strict=True,
                    )
                ),
                start=Decimal("0"),
            )
            if not _is_close(
                hourly_weighted,
                self.weighted_daytime_activity,
                tolerance=0.1,
            ):
                raise _contract_error(
                    "field_validation_error",
                    "weighted activity must match hourly intensity times coverage within 0.1",
                    field="weighted_daytime_activity",
                )

        scene_keys = [
            (metric.camera_id, metric.scene_version)
            for metric in self.camera_gait_metrics
        ]
        if len(scene_keys) != len(set(scene_keys)):
            raise _contract_error(
                "field_validation_error",
                "camera_id and scene_version pairs must be unique",
                field="camera_gait_metrics",
            )
        return self


class MoodSocialSleep(MoodSocialSchemaModel):
    in_bed_minutes: (
        Annotated[float, Field(strict=True, ge=0.0, le=1440.0, allow_inf_nan=False)]
        | None
    ) = None
    sleep_minutes: (
        Annotated[float, Field(strict=True, ge=0.0, le=1440.0, allow_inf_nan=False)]
        | None
    ) = None
    sleep_efficiency: Probability | None = None
    sleep_onset_minute_of_day: (
        Annotated[int, Field(strict=True, ge=0, le=1439)] | None
    ) = None
    wake_time_minute_of_day: (
        Annotated[int, Field(strict=True, ge=0, le=1439)] | None
    ) = None
    awake_minutes: (
        Annotated[float, Field(strict=True, ge=0.0, le=1440.0, allow_inf_nan=False)]
        | None
    ) = None
    awakening_count: StrictNonNegativeInt | None = None
    night_exit_count: StrictNonNegativeInt | None = None
    night_exit_minutes: (
        Annotated[float, Field(strict=True, ge=0.0, le=1440.0, allow_inf_nan=False)]
        | None
    ) = None

    @model_validator(mode="after")
    def validate_sleep_semantics(self) -> "MoodSocialSleep":
        values = (
            self.in_bed_minutes,
            self.sleep_minutes,
            self.sleep_efficiency,
            self.sleep_onset_minute_of_day,
            self.wake_time_minute_of_day,
            self.awake_minutes,
            self.awakening_count,
            self.night_exit_count,
            self.night_exit_minutes,
        )
        if all(value is None for value in values):
            raise _contract_error(
                "field_validation_error",
                "sleep object must contain at least one non-null semantic field",
                field="sleep",
            )
        if (
            self.sleep_minutes is not None
            and self.in_bed_minutes is not None
            and self.sleep_minutes > self.in_bed_minutes
        ):
            raise _contract_error(
                "field_validation_error",
                "sleep_minutes must not exceed in_bed_minutes",
                field="sleep_minutes",
            )
        if (
            self.awake_minutes is not None
            and self.in_bed_minutes is not None
            and self.awake_minutes > self.in_bed_minutes
        ):
            raise _contract_error(
                "field_validation_error",
                "awake_minutes must not exceed in_bed_minutes",
                field="awake_minutes",
            )
        if (
            self.in_bed_minutes is not None
            and self.in_bed_minutes > 0.0
            and self.sleep_minutes is not None
            and self.sleep_efficiency is not None
        ):
            expected = (
                Decimal(str(self.sleep_minutes))
                / Decimal(str(self.in_bed_minutes))
            )
            if not _is_close(self.sleep_efficiency, expected, tolerance=0.01):
                raise _contract_error(
                    "field_validation_error",
                    "sleep_efficiency must match sleep_minutes / in_bed_minutes within 0.01",
                    field="sleep_efficiency",
                )
        return self


class MoodSocialPhysiology(MoodSocialSchemaModel):
    heart_rate_mean_bpm: (
        Annotated[float, Field(strict=True, gt=0.0, allow_inf_nan=False)] | None
    ) = None
    heart_rate_min_bpm: (
        Annotated[float, Field(strict=True, gt=0.0, allow_inf_nan=False)] | None
    ) = None
    heart_rate_sd_bpm: (
        Annotated[float, Field(strict=True, ge=0.0, allow_inf_nan=False)] | None
    ) = None
    hrv_sdnn_ms: (
        Annotated[float, Field(strict=True, ge=0.0, allow_inf_nan=False)] | None
    ) = None
    respiration_rate_mean_bpm: (
        Annotated[float, Field(strict=True, gt=0.0, allow_inf_nan=False)] | None
    ) = None
    respiration_rate_sd_bpm: (
        Annotated[float, Field(strict=True, ge=0.0, allow_inf_nan=False)] | None
    ) = None
    respiratory_abnormal_ratio: Probability | None = None
    snoring_minutes: (
        Annotated[float, Field(strict=True, ge=0.0, le=1440.0, allow_inf_nan=False)]
        | None
    ) = None

    @model_validator(mode="after")
    def require_observed_value(self) -> "MoodSocialPhysiology":
        if all(
            value is None
            for value in (
                self.heart_rate_mean_bpm,
                self.heart_rate_min_bpm,
                self.heart_rate_sd_bpm,
                self.hrv_sdnn_ms,
                self.respiration_rate_mean_bpm,
                self.respiration_rate_sd_bpm,
                self.respiratory_abnormal_ratio,
                self.snoring_minutes,
            )
        ):
            raise _contract_error(
                "field_validation_error",
                "physiology object must contain at least one non-null semantic field",
                field="physiology",
            )
        return self


class MoodSocialSocial(MoodSocialSchemaModel):
    call_log_observed: StrictBool
    incoming_call_opportunities: StrictNonNegativeInt | None = None
    answered_call_count: StrictNonNegativeInt | None = None
    missed_call_count: StrictNonNegativeInt | None = None
    outgoing_call_count: StrictNonNegativeInt | None = None
    connected_duration_minutes: (
        Annotated[float, Field(strict=True, ge=0.0, le=1440.0, allow_inf_nan=False)]
        | None
    ) = None
    active_contact_count: StrictNonNegativeInt | None = None

    @model_validator(mode="after")
    def validate_call_log_semantics(self) -> "MoodSocialSocial":
        statistics = (
            self.incoming_call_opportunities,
            self.answered_call_count,
            self.missed_call_count,
            self.outgoing_call_count,
            self.connected_duration_minutes,
            self.active_contact_count,
        )
        if not self.call_log_observed:
            if any(value is not None for value in statistics):
                raise _contract_error(
                    "field_validation_error",
                    "unobserved call logs require all six statistics to be null or omitted",
                    field="call_log_observed",
                )
            return self

        if any(value is None for value in statistics):
            raise _contract_error(
                "field_validation_error",
                "observed call logs require all six raw statistics",
                field="call_log_observed",
            )

        incoming = self.incoming_call_opportunities
        answered = self.answered_call_count
        missed = self.missed_call_count
        outgoing = self.outgoing_call_count
        duration = self.connected_duration_minutes
        contacts = self.active_contact_count
        assert incoming is not None
        assert answered is not None
        assert missed is not None
        assert outgoing is not None
        assert duration is not None
        assert contacts is not None

        if incoming != answered + missed:
            raise _contract_error(
                "field_validation_error",
                "incoming_call_opportunities must equal answered_call_count + missed_call_count",
                field="incoming_call_opportunities",
            )
        connected = answered + outgoing
        if contacts > connected:
            raise _contract_error(
                "field_validation_error",
                "active_contact_count must not exceed connected call count",
                field="active_contact_count",
            )
        if connected == 0 and (duration != 0.0 or contacts != 0):
            raise _contract_error(
                "field_validation_error",
                "zero connected calls require zero duration and zero active contacts",
                field="connected_duration_minutes",
            )
        if connected > 0 and (duration <= 0.0 or contacts < 1):
            raise _contract_error(
                "field_validation_error",
                "positive connected calls require positive duration and at least one active contact",
                field="connected_duration_minutes",
            )
        return self


class MoodSocialDailyFeatures(MoodSocialSchemaModel):
    date: DateOnly
    activity: MoodSocialActivity | None
    sleep: MoodSocialSleep | None
    physiology: MoodSocialPhysiology | None
    social: MoodSocialSocial | None

    @property
    def has_domain_record(self) -> bool:
        return any(
            value is not None
            for value in (
                self.activity,
                self.sleep,
                self.physiology,
                self.social,
            )
        )


class MoodSocialHistoryAttentionIndex(MoodSocialSchemaModel):
    date: DateOnly
    attention_index: Probability
    model_version: NonEmptyText


class MoodSocialInferRequest(MoodSocialSchemaModel):
    schema_version: Literal["mood_social_infer_request_v3"]
    request_id: RequestId
    person_id: PersonId
    target_date: DateOnly
    timezone: Literal["Asia/Shanghai"]
    available_sources: Annotated[
        list[AvailableSource],
        Field(min_length=1, max_length=3),
    ]
    profile: MoodSocialProfile
    current_daily_features: MoodSocialDailyFeatures
    history_daily_features: Annotated[
        list[MoodSocialDailyFeatures],
        Field(max_length=28),
    ]
    history_attention_indices: Annotated[
        list[MoodSocialHistoryAttentionIndex],
        Field(max_length=6),
    ]

    @field_validator("available_sources")
    @classmethod
    def validate_available_source_uniqueness(
        cls,
        values: list[AvailableSource],
    ) -> list[AvailableSource]:
        if len(values) != len(set(values)):
            raise _contract_error(
                "field_validation_error",
                "available_sources must not contain duplicates",
                field="available_sources",
            )
        return values

    @model_validator(mode="after")
    def validate_request_relationships(self) -> "MoodSocialInferRequest":
        service_date = datetime.now(ZoneInfo(MOOD_SOCIAL_TIMEZONE)).date()
        if self.target_date >= service_date:
            raise _contract_error(
                "date_validation_error",
                "target_date must be an already completed natural day",
                field="target_date",
            )
        if self.current_daily_features.date != self.target_date:
            raise _contract_error(
                "date_validation_error",
                "current_daily_features.date must equal target_date",
                field="current_daily_features.date",
            )

        history_dates = [record.date for record in self.history_daily_features]
        if history_dates != sorted(history_dates):
            raise _contract_error(
                "date_validation_error",
                "history_daily_features must be ordered by ascending date",
                field="history_daily_features",
            )
        if len(history_dates) != len(set(history_dates)):
            raise _contract_error(
                "date_validation_error",
                "history_daily_features dates must not repeat",
                field="history_daily_features",
            )
        earliest_history_date = self.target_date - timedelta(days=28)
        latest_history_date = self.target_date - timedelta(days=1)
        for index, record in enumerate(self.history_daily_features):
            if not earliest_history_date <= record.date <= latest_history_date:
                raise _contract_error(
                    "date_validation_error",
                    "history date must be within D-28 through D-1",
                    field=f"history_daily_features[{index}].date",
                )
            if not record.has_domain_record:
                raise _contract_error(
                    "field_validation_error",
                    "history records must contain at least one non-null domain object",
                    field=f"history_daily_features[{index}]",
                )

        attention_dates = [
            record.date for record in self.history_attention_indices
        ]
        if attention_dates != sorted(attention_dates):
            raise _contract_error(
                "date_validation_error",
                "history_attention_indices must be ordered by ascending date",
                field="history_attention_indices",
            )
        if len(attention_dates) != len(set(attention_dates)):
            raise _contract_error(
                "date_validation_error",
                "history_attention_indices dates must not repeat",
                field="history_attention_indices",
            )
        if any(value >= self.target_date for value in attention_dates):
            raise _contract_error(
                "date_validation_error",
                "history attention dates must be earlier than target_date",
                field="history_attention_indices",
            )

        declared = set(self.available_sources)
        all_daily_records = [
            self.current_daily_features,
            *self.history_daily_features,
        ]
        for index, record in enumerate(all_daily_records):
            prefix = (
                "current_daily_features"
                if index == 0
                else f"history_daily_features[{index - 1}]"
            )
            mismatches: list[tuple[bool, str, str]] = [
                (record.activity is not None, "camera", "activity"),
                (record.sleep is not None, "sleep_device", "sleep"),
                (record.physiology is not None, "sleep_device", "physiology"),
                (record.social is not None, "s10", "social"),
            ]
            for is_present, required_source, domain in mismatches:
                if is_present and required_source not in declared:
                    raise _contract_error(
                        "source_declaration_mismatch",
                        f"{domain} requires available source {required_source}",
                        field=f"{prefix}.{domain}",
                    )
        return self


class MoodSocialPersonalChangeScores(MoodSocialSchemaModel):
    activity: Probability | None
    sleep: Probability | None
    social: Probability | None


class MoodSocialDomainScores(MoodSocialSchemaModel):
    activity: Probability | None
    sleep: Probability | None
    physiology: Probability | None
    activity_sleep_joint: Probability | None
    social_context: Probability | None
    personal_change: MoodSocialPersonalChangeScores


class MoodSocialModelContribution(MoodSocialSchemaModel):
    factor: StableCode
    label: NonEmptyText
    direction: Literal["up"]
    contribution: Annotated[
        float,
        Field(strict=True, gt=0.0, allow_inf_nan=False),
    ]

    @field_validator("factor")
    @classmethod
    def reject_non_risk_metadata_factors(cls, value: str) -> str:
        forbidden_tokens = ("bias", "mask", "confidence", "coverage")
        if any(token in value for token in forbidden_tokens):
            raise _contract_error(
                "field_validation_error",
                "contributions must not expose bias, mask, confidence, or coverage",
                field="factor",
            )
        return value

    @model_validator(mode="after")
    def validate_frozen_factor_label(self) -> "MoodSocialModelContribution":
        expected_label = FACTOR_LABELS.get(self.factor)
        if expected_label is not None and self.label != expected_label:
            raise _contract_error(
                "field_validation_error",
                "known factors must use their frozen Chinese label",
                field="label",
            )
        return self


class MoodSocialInferResponse(MoodSocialSchemaModel):
    schema_version: Literal["mood_social_infer_response_v3"]
    request_id: RequestId
    person_id: PersonId
    target_date: DateOnly
    module: Literal["mood_social_attention"]
    model_version: Literal["mood-fusion-v3.3.3"]
    available: StrictBool
    evidence_scope: EvidenceScope
    attention_index: Probability | None
    attention_score: Annotated[int, Field(strict=True, ge=0, le=100)] | None
    attention_level: Annotated[int, Field(strict=True, ge=0, le=3)] | None
    confidence: Probability
    used_sources: list[UsedSource]
    covered_domains: list[CoveredDomain]
    domain_scores: MoodSocialDomainScores | None
    model_contributions: Annotated[
        list[MoodSocialModelContribution],
        Field(max_length=3),
    ]
    trend: Trend
    summary: NonEmptyText
    limitations: list[NonEmptyText]
    diagnosis: DiagnosisFalse

    @model_validator(mode="after")
    def validate_response_relationships(self) -> "MoodSocialInferResponse":
        if self.diagnosis:
            raise _contract_error(
                "field_validation_error",
                "diagnosis must remain false",
                field="diagnosis",
            )
        if len(self.used_sources) != len(set(self.used_sources)):
            raise _contract_error(
                "field_validation_error",
                "used_sources must not contain duplicates",
                field="used_sources",
            )
        used_source_order = ["camera", "sleep_device", "s10", "profile"]
        if self.used_sources != sorted(
            self.used_sources,
            key=used_source_order.index,
        ):
            raise _contract_error(
                "field_validation_error",
                "used_sources must use the frozen source order",
                field="used_sources",
            )

        if len(self.covered_domains) != len(set(self.covered_domains)):
            raise _contract_error(
                "field_validation_error",
                "covered_domains must not contain duplicates",
                field="covered_domains",
            )
        covered_domain_order = [
            "activity",
            "sleep",
            "physiology",
            "activity_sleep_joint",
            "social_context",
            "personal_change_activity",
            "personal_change_sleep",
            "personal_change_social",
        ]
        if self.covered_domains != sorted(
            self.covered_domains,
            key=covered_domain_order.index,
        ):
            raise _contract_error(
                "field_validation_error",
                "covered_domains must use the frozen domain order",
                field="covered_domains",
            )
        expected_contributions = sorted(
            self.model_contributions,
            key=lambda item: (-item.contribution, item.factor),
        )
        if self.model_contributions != expected_contributions:
            raise _contract_error(
                "field_validation_error",
                "model_contributions must be sorted by contribution then factor",
                field="model_contributions",
            )

        if not self.available:
            if (
                self.evidence_scope != "insufficient_data"
                or self.attention_index is not None
                or self.attention_score is not None
                or self.attention_level is not None
                or self.confidence != 0.0
                or self.used_sources
                or self.covered_domains
                or self.domain_scores is not None
                or self.model_contributions
                or self.trend != "unknown"
                or self.summary != "当前未获得可用的情绪与社交关注证据。"
                or "insufficient_data" not in self.limitations
            ):
                raise _contract_error(
                    "field_validation_error",
                    "unavailable responses must use the frozen insufficient-data shape",
                    field="available",
                )
            return self

        if (
            self.attention_index is None
            or self.attention_score is None
            or self.attention_level is None
            or self.domain_scores is None
            or not self.used_sources
            or not self.covered_domains
        ):
            raise _contract_error(
                "field_validation_error",
                "available responses require attention values and effective branches",
                field="available",
            )

        if "insufficient_data" in self.limitations:
            raise _contract_error(
                "field_validation_error",
                "available responses must not use the insufficient_data limitation",
                field="limitations",
            )

        expected_scope = (
            "engineering_proxy"
            if "s10" in self.used_sources
            else "proxy_label_supported"
        )
        if self.evidence_scope != expected_scope:
            raise _contract_error(
                "field_validation_error",
                "evidence_scope must reflect actually used sources",
                field="evidence_scope",
            )

        expected_score = int(
            (Decimal(str(self.attention_index)) * Decimal("100")).quantize(
                Decimal("1"),
                rounding=ROUND_HALF_UP,
            )
        )
        if self.attention_score != expected_score:
            raise _contract_error(
                "field_validation_error",
                "attention_score must use half-up rounding",
                field="attention_score",
            )
        if self.attention_index < 0.25:
            expected_level = 0
        elif self.attention_index < 0.45:
            expected_level = 1
        elif self.attention_index < 0.65:
            expected_level = 2
        else:
            expected_level = 3
        if self.attention_level != expected_level:
            raise _contract_error(
                "field_validation_error",
                "attention_level must use the frozen unrounded thresholds",
                field="attention_level",
            )

        branch_values = {
            "activity": self.domain_scores.activity,
            "sleep": self.domain_scores.sleep,
            "physiology": self.domain_scores.physiology,
            "activity_sleep_joint": self.domain_scores.activity_sleep_joint,
            "social_context": self.domain_scores.social_context,
            "personal_change_activity": self.domain_scores.personal_change.activity,
            "personal_change_sleep": self.domain_scores.personal_change.sleep,
            "personal_change_social": self.domain_scores.personal_change.social,
        }
        covered = set(self.covered_domains)
        for branch, value in branch_values.items():
            if (branch in covered) != (value is not None):
                raise _contract_error(
                    "field_validation_error",
                    "domain scores must be null exactly when a branch is not covered",
                    field=f"domain_scores.{branch}",
                )

        source_by_branch = {
            "activity": {"camera"},
            "sleep": {"sleep_device"},
            "physiology": {"sleep_device"},
            "activity_sleep_joint": {"camera", "sleep_device"},
            "social_context": {"profile"},
            "personal_change_activity": {"camera"},
            "personal_change_sleep": {"sleep_device"},
            "personal_change_social": {"s10"},
        }
        derived_sources: set[str] = set()
        for branch in self.covered_domains:
            derived_sources.update(source_by_branch[branch])
        expected_used_sources = [
            source for source in used_source_order if source in derived_sources
        ]
        if self.used_sources != expected_used_sources:
            raise _contract_error(
                "field_validation_error",
                "used_sources must be derived from covered effective branches",
                field="used_sources",
            )

        expected_summary = _summary_for_contributions(self.model_contributions)
        if self.summary != expected_summary:
            raise _contract_error(
                "field_validation_error",
                "summary must use the frozen deterministic template",
                field="summary",
            )

        personal_factor_domains = {
            "sedentary_time_increase": "personal_change_activity",
            "personal_activity_deviation": "personal_change_activity",
            "sleep_efficiency_drop": "personal_change_sleep",
            "sleep_fragmentation_increase": "personal_change_sleep",
            "sleep_schedule_change": "personal_change_sleep",
            "night_exit_increase": "personal_change_sleep",
            "night_physiology_change": "personal_change_sleep",
            "personal_sleep_deviation": "personal_change_sleep",
            "contact_frequency_drop": "personal_change_social",
            "contact_duration_drop": "personal_change_social",
            "personal_social_deviation": "personal_change_social",
        }
        state_factor_domains = {
            "observed_daytime_activity_low": {
                "activity",
                "activity_sleep_joint",
            },
            "observed_low_activity_time_high": {
                "activity",
                "activity_sleep_joint",
            },
            "activity_rhythm_irregular": {
                "activity",
                "activity_sleep_joint",
            },
            "sleep_efficiency_low": {
                "sleep",
                "activity_sleep_joint",
            },
            "sleep_fragmentation_high": {
                "sleep",
                "activity_sleep_joint",
            },
            "sleep_schedule_irregular": {
                "sleep",
                "activity_sleep_joint",
            },
            "night_exit_frequent": {
                "sleep",
                "activity_sleep_joint",
            },
            "night_physiology_attention": {"physiology"},
            "social_context_attention": {"social_context"},
        }
        change_markers = (
            "下降",
            "增加",
            "减少",
            "变化",
            "偏离个人历史",
            "近期变差",
            "drop",
            "increase",
            "decrease",
        )
        prohibited_explanation_markers = (
            "抑郁症",
            "诊断",
            "因果",
            "phq",
            "psyche",
            "nhanes",
            "resilient",
        )
        for contribution in self.model_contributions:
            allowed_state_branches = state_factor_domains.get(
                contribution.factor
            )
            if (
                allowed_state_branches is not None
                and covered.isdisjoint(allowed_state_branches)
            ):
                raise _contract_error(
                    "field_validation_error",
                    "state factors require a corresponding covered expert branch",
                    field="model_contributions",
                )
            required_branch = personal_factor_domains.get(contribution.factor)
            if required_branch is not None and required_branch not in covered:
                raise _contract_error(
                    "field_validation_error",
                    "personal-change factors require the corresponding covered branch",
                    field="model_contributions",
                )
            explanation_text = (
                f"{contribution.factor} {contribution.label}".lower()
            )
            uses_change_language = any(
                marker in explanation_text
                for marker in change_markers
            )
            if uses_change_language and required_branch is None:
                raise _contract_error(
                    "field_validation_error",
                    "change language is reserved for frozen personal-change factors",
                    field="model_contributions",
                )
            if any(
                marker in explanation_text
                for marker in prohibited_explanation_markers
            ):
                raise _contract_error(
                    "field_validation_error",
                    "explanations must not contain diagnosis, causality, or dataset names",
                    field="model_contributions",
                )
        return self


def _summary_for_contributions(
    contributions: list[MoodSocialModelContribution],
) -> str:
    if not contributions:
        return "当前模型结果可用，暂未识别出明确的主要关注因素，建议继续观察近期趋势。"
    if len(contributions) == 1:
        return f"{contributions[0].label}，建议家属主动沟通并继续观察。"
    return (
        f"{contributions[0].label}、{contributions[1].label}"
        "，建议家属主动沟通并继续观察。"
    )


class MoodSocialErrorItem(MoodSocialSchemaModel):
    field: NonEmptyText
    reason: NonEmptyText


class MoodSocialErrorDetail(MoodSocialSchemaModel):
    schema_version: Literal["mood_social_error_v1"]
    request_id: StrictText | None
    code: MoodSocialErrorCode
    message: NonEmptyText
    errors: list[MoodSocialErrorItem]


class MoodSocialErrorResponse(MoodSocialSchemaModel):
    detail: MoodSocialErrorDetail


__all__ = [
    "ERROR_SCHEMA_VERSION",
    "MOOD_SOCIAL_MODEL_VERSION",
    "MOOD_SOCIAL_MODULE",
    "MOOD_SOCIAL_TIMEZONE",
    "REQUEST_SCHEMA_VERSION",
    "RESPONSE_SCHEMA_VERSION",
    "CameraGaitMetric",
    "MoodSocialActivity",
    "MoodSocialDailyFeatures",
    "MoodSocialDomainScores",
    "MoodSocialErrorCode",
    "MoodSocialErrorDetail",
    "MoodSocialErrorItem",
    "MoodSocialErrorResponse",
    "MoodSocialHistoryAttentionIndex",
    "MoodSocialInferRequest",
    "MoodSocialInferResponse",
    "MoodSocialModelContribution",
    "MoodSocialPersonalChangeScores",
    "MoodSocialPhysiology",
    "MoodSocialProfile",
    "MoodSocialSleep",
    "MoodSocialSocial",
]
