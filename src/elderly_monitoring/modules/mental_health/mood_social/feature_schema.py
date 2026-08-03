"""Frozen V3.3.3 feature schema for mood and social attention.

The HTTP request contains sparse, source-native daily observations.  This
module defines the deterministic, namespaced feature contract produced from
those observations before any fold-fitted preprocessing or model inference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum
from numbers import Real
from types import MappingProxyType
from typing import Any, Mapping


FEATURE_SCHEMA_VERSION = "mood_social_feature_schema_v3_3_3"
STATE_WINDOW_DAYS = 7
HISTORY_LOOKBACK_DAYS = 28
NO_EFFECTIVE_CONTACT_MAX_DAYS = HISTORY_LOOKBACK_DAYS + 1
CAMERA_DAYTIME_START = "06:00"
CAMERA_DAYTIME_END = "18:00"
CAMERA_DAYTIME_MINUTES = 720


class FeatureValueType(str, Enum):
    FLOAT = "float"
    INTEGER = "integer"
    CATEGORY = "category"


class RiskDirection(str, Enum):
    """How a raw feature is interpreted by personal-change logic.

    MODEL_LEARNED means that V3.3.3 does not impose a monotonic direction on
    the supervised model.  COVERAGE_ONLY fields may affect reliability but
    must never be treated as behavioral risk evidence.
    """

    DECREASE = "decrease"
    INCREASE = "increase"
    TWO_SIDED = "two_sided"
    CIRCULAR_TWO_SIDED = "circular_two_sided"
    MODEL_LEARNED = "model_learned"
    COVERAGE_ONLY = "coverage_only"


class FeatureRole(str, Enum):
    SUPERVISED = "supervised"
    PERSONAL_TREND = "personal_trend"
    COVERAGE = "coverage"


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    value_type: FeatureValueType
    risk_direction: RiskDirection
    role: FeatureRole
    minimum: float | int | None = None
    maximum: float | int | None = None
    exclusive_minimum: bool = False
    exclusive_maximum: bool = False
    categories: tuple[str, ...] = ()
    circular_group: str | None = None

    def __post_init__(self) -> None:
        if self.exclusive_minimum and self.minimum is None:
            raise ValueError("exclusive_minimum requires a minimum")
        if self.exclusive_maximum and self.maximum is None:
            raise ValueError("exclusive_maximum requires a maximum")
        if self.minimum is not None and self.maximum is not None:
            if self.minimum > self.maximum or (
                self.minimum == self.maximum
                and (self.exclusive_minimum or self.exclusive_maximum)
            ):
                raise ValueError("feature bounds do not contain any valid value")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value_type": self.value_type.value,
            "risk_direction": self.risk_direction.value,
            "role": self.role.value,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "exclusive_minimum": self.exclusive_minimum,
            "exclusive_maximum": self.exclusive_maximum,
            "categories": list(self.categories),
            "circular_group": self.circular_group,
        }


def _float_spec(
    name: str,
    direction: RiskDirection,
    role: FeatureRole,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    exclusive_minimum: bool = False,
    exclusive_maximum: bool = False,
    circular_group: str | None = None,
) -> FeatureSpec:
    return FeatureSpec(
        name=name,
        value_type=FeatureValueType.FLOAT,
        risk_direction=direction,
        role=role,
        minimum=minimum,
        maximum=maximum,
        exclusive_minimum=exclusive_minimum,
        exclusive_maximum=exclusive_maximum,
        circular_group=circular_group,
    )


def _integer_spec(
    name: str,
    direction: RiskDirection,
    role: FeatureRole,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> FeatureSpec:
    return FeatureSpec(
        name=name,
        value_type=FeatureValueType.INTEGER,
        risk_direction=direction,
        role=role,
        minimum=minimum,
        maximum=maximum,
    )


def _category_spec(name: str, categories: tuple[str, ...]) -> FeatureSpec:
    return FeatureSpec(
        name=name,
        value_type=FeatureValueType.CATEGORY,
        risk_direction=RiskDirection.MODEL_LEARNED,
        role=FeatureRole.SUPERVISED,
        categories=categories,
    )


# V3.3.2 section 4 remains the non-conflicting source for these group orders.
ACTIVITY_FEATURE_SPECS = (
    _float_spec(
        "activity_volume_norm",
        RiskDirection.DECREASE,
        FeatureRole.SUPERVISED,
        minimum=0.0,
        maximum=1.0,
    ),
    _float_spec(
        "active_ratio",
        RiskDirection.DECREASE,
        FeatureRole.SUPERVISED,
        minimum=0.0,
        maximum=1.0,
    ),
    _float_spec(
        "sedentary_ratio",
        RiskDirection.INCREASE,
        FeatureRole.SUPERVISED,
        minimum=0.0,
        maximum=1.0,
    ),
    _float_spec(
        "longest_inactive_bout_norm",
        RiskDirection.INCREASE,
        FeatureRole.SUPERVISED,
        minimum=0.0,
        maximum=1.0,
    ),
    _float_spec(
        "relative_amplitude",
        RiskDirection.DECREASE,
        FeatureRole.SUPERVISED,
        minimum=0.0,
        maximum=1.0,
    ),
    _float_spec(
        "interdaily_stability",
        RiskDirection.DECREASE,
        FeatureRole.SUPERVISED,
        minimum=0.0,
        maximum=1.0,
    ),
    _float_spec(
        "intradaily_variability",
        RiskDirection.INCREASE,
        FeatureRole.SUPERVISED,
        minimum=0.0,
    ),
    _float_spec(
        "activity_variability",
        RiskDirection.INCREASE,
        FeatureRole.SUPERVISED,
        minimum=0.0,
    ),
    _integer_spec(
        "valid_days",
        RiskDirection.COVERAGE_ONLY,
        FeatureRole.COVERAGE,
        minimum=0,
        maximum=STATE_WINDOW_DAYS,
    ),
    _float_spec(
        "feature_coverage",
        RiskDirection.COVERAGE_ONLY,
        FeatureRole.COVERAGE,
        minimum=0.0,
        maximum=1.0,
    ),
)


SLEEP_FEATURE_SPECS = (
    _float_spec(
        "sleep_duration_norm",
        RiskDirection.TWO_SIDED,
        FeatureRole.SUPERVISED,
        minimum=0.0,
        maximum=1.0,
    ),
    _float_spec(
        "time_in_bed_norm",
        RiskDirection.TWO_SIDED,
        FeatureRole.SUPERVISED,
        minimum=0.0,
        maximum=1.0,
    ),
    _float_spec(
        "sleep_efficiency",
        RiskDirection.DECREASE,
        FeatureRole.SUPERVISED,
        minimum=0.0,
        maximum=1.0,
    ),
    _float_spec(
        "sleep_onset_sin",
        RiskDirection.CIRCULAR_TWO_SIDED,
        FeatureRole.SUPERVISED,
        minimum=-1.0,
        maximum=1.0,
        circular_group="sleep_onset",
    ),
    _float_spec(
        "sleep_onset_cos",
        RiskDirection.CIRCULAR_TWO_SIDED,
        FeatureRole.SUPERVISED,
        minimum=-1.0,
        maximum=1.0,
        circular_group="sleep_onset",
    ),
    _float_spec(
        "wake_time_sin",
        RiskDirection.CIRCULAR_TWO_SIDED,
        FeatureRole.SUPERVISED,
        minimum=-1.0,
        maximum=1.0,
        circular_group="wake_time",
    ),
    _float_spec(
        "wake_time_cos",
        RiskDirection.CIRCULAR_TWO_SIDED,
        FeatureRole.SUPERVISED,
        minimum=-1.0,
        maximum=1.0,
        circular_group="wake_time",
    ),
    _float_spec(
        "sleep_midpoint_sin",
        RiskDirection.CIRCULAR_TWO_SIDED,
        FeatureRole.SUPERVISED,
        minimum=-1.0,
        maximum=1.0,
        circular_group="sleep_midpoint",
    ),
    _float_spec(
        "sleep_midpoint_cos",
        RiskDirection.CIRCULAR_TWO_SIDED,
        FeatureRole.SUPERVISED,
        minimum=-1.0,
        maximum=1.0,
        circular_group="sleep_midpoint",
    ),
    _float_spec(
        "sleep_fragmentation",
        RiskDirection.INCREASE,
        FeatureRole.SUPERVISED,
        minimum=0.0,
    ),
    _float_spec(
        "sleep_regularity",
        RiskDirection.DECREASE,
        FeatureRole.SUPERVISED,
        minimum=0.0,
        maximum=1.0,
    ),
    _float_spec(
        "night_exit_count_mean",
        RiskDirection.INCREASE,
        FeatureRole.SUPERVISED,
        minimum=0.0,
    ),
    _integer_spec(
        "valid_nights",
        RiskDirection.COVERAGE_ONLY,
        FeatureRole.COVERAGE,
        minimum=0,
        maximum=STATE_WINDOW_DAYS,
    ),
    _float_spec(
        "feature_coverage",
        RiskDirection.COVERAGE_ONLY,
        FeatureRole.COVERAGE,
        minimum=0.0,
        maximum=1.0,
    ),
)


PHYSIOLOGY_FEATURE_SPECS = (
    _float_spec(
        "heart_rate_mean_bpm",
        RiskDirection.MODEL_LEARNED,
        FeatureRole.SUPERVISED,
        minimum=0.0,
        exclusive_minimum=True,
    ),
    _float_spec(
        "heart_rate_min_bpm",
        RiskDirection.MODEL_LEARNED,
        FeatureRole.SUPERVISED,
        minimum=0.0,
        exclusive_minimum=True,
    ),
    _float_spec(
        "heart_rate_sd_bpm",
        RiskDirection.MODEL_LEARNED,
        FeatureRole.SUPERVISED,
        minimum=0.0,
    ),
    _float_spec(
        "hrv_sdnn_ms",
        RiskDirection.MODEL_LEARNED,
        FeatureRole.SUPERVISED,
        minimum=0.0,
    ),
    _float_spec(
        "respiration_rate_mean_bpm",
        RiskDirection.MODEL_LEARNED,
        FeatureRole.SUPERVISED,
        minimum=0.0,
        exclusive_minimum=True,
    ),
    _float_spec(
        "respiration_rate_sd_bpm",
        RiskDirection.MODEL_LEARNED,
        FeatureRole.SUPERVISED,
        minimum=0.0,
    ),
    _float_spec(
        "respiratory_abnormal_ratio",
        RiskDirection.MODEL_LEARNED,
        FeatureRole.SUPERVISED,
        minimum=0.0,
        maximum=1.0,
    ),
    _float_spec(
        "snoring_minutes_norm",
        RiskDirection.MODEL_LEARNED,
        FeatureRole.SUPERVISED,
        minimum=0.0,
        maximum=1.0,
    ),
    _integer_spec(
        "valid_nights",
        RiskDirection.COVERAGE_ONLY,
        FeatureRole.COVERAGE,
        minimum=0,
        maximum=STATE_WINDOW_DAYS,
    ),
    _float_spec(
        "feature_coverage",
        RiskDirection.COVERAGE_ONLY,
        FeatureRole.COVERAGE,
        minimum=0.0,
        maximum=1.0,
    ),
)


SOCIAL_CONTEXT_FEATURE_SPECS = (
    _category_spec("age_group", ("60_69", "70_79", "80_plus")),
    _category_spec("sex", ("female", "male")),
    _category_spec(
        "living_arrangement",
        ("alone", "with_family", "institution", "other"),
    ),
    _category_spec("marital_status", ("partnered", "not_partnered")),
    _integer_spec(
        "chronic_disease_count",
        RiskDirection.MODEL_LEARNED,
        FeatureRole.SUPERVISED,
        minimum=0,
    ),
    _integer_spec(
        "self_rated_health",
        RiskDirection.MODEL_LEARNED,
        FeatureRole.SUPERVISED,
        minimum=1,
        maximum=5,
    ),
    _integer_spec(
        "functional_limitation",
        RiskDirection.MODEL_LEARNED,
        FeatureRole.SUPERVISED,
        minimum=0,
        maximum=3,
    ),
    _integer_spec(
        "social_participation_days_per_week",
        RiskDirection.MODEL_LEARNED,
        FeatureRole.SUPERVISED,
        minimum=0,
        maximum=7,
    ),
    _category_spec(
        "education_level",
        ("primary_or_less", "middle", "high_or_above"),
    ),
    _category_spec("economic_status", ("low", "middle", "high")),
)


# V3.3.3 adds connected_call_count to the V3.3.2 social trend sequence.
# Windowed counts are per-observed-day means, so their mapped type is float.
SOCIAL_CONTACT_FEATURE_SPECS = (
    _float_spec(
        "incoming_call_opportunities",
        RiskDirection.MODEL_LEARNED,
        FeatureRole.PERSONAL_TREND,
        minimum=0.0,
    ),
    _float_spec(
        "outgoing_call_count",
        RiskDirection.DECREASE,
        FeatureRole.PERSONAL_TREND,
        minimum=0.0,
    ),
    _float_spec(
        "answered_call_count",
        RiskDirection.DECREASE,
        FeatureRole.PERSONAL_TREND,
        minimum=0.0,
    ),
    _float_spec(
        "missed_call_count",
        RiskDirection.INCREASE,
        FeatureRole.PERSONAL_TREND,
        minimum=0.0,
    ),
    _float_spec(
        "answer_rate",
        RiskDirection.DECREASE,
        FeatureRole.PERSONAL_TREND,
        minimum=0.0,
        maximum=1.0,
    ),
    _float_spec(
        "connected_call_count",
        RiskDirection.DECREASE,
        FeatureRole.PERSONAL_TREND,
        minimum=0.0,
    ),
    _float_spec(
        "connected_duration_minutes",
        RiskDirection.DECREASE,
        FeatureRole.PERSONAL_TREND,
        minimum=0.0,
        maximum=1440.0,
    ),
    _float_spec(
        "mean_connected_duration_minutes",
        RiskDirection.DECREASE,
        FeatureRole.PERSONAL_TREND,
        minimum=0.0,
        maximum=1440.0,
    ),
    _float_spec(
        "active_contact_count",
        RiskDirection.DECREASE,
        FeatureRole.PERSONAL_TREND,
        minimum=0.0,
    ),
    _integer_spec(
        "no_effective_contact_days",
        RiskDirection.INCREASE,
        FeatureRole.PERSONAL_TREND,
        minimum=0,
        maximum=NO_EFFECTIVE_CONTACT_MAX_DAYS,
    ),
)


ACTIVITY_DAILY_FEATURE_SPECS = (
    _float_spec(
        "observed_activity_intensity",
        RiskDirection.DECREASE,
        FeatureRole.PERSONAL_TREND,
        minimum=0.0,
        maximum=1.0,
    ),
    _float_spec(
        "active_ratio",
        RiskDirection.DECREASE,
        FeatureRole.PERSONAL_TREND,
        minimum=0.0,
        maximum=1.0,
    ),
    _float_spec(
        "sedentary_ratio",
        RiskDirection.INCREASE,
        FeatureRole.PERSONAL_TREND,
        minimum=0.0,
        maximum=1.0,
    ),
    _float_spec(
        "longest_inactive_bout_norm",
        RiskDirection.INCREASE,
        FeatureRole.PERSONAL_TREND,
        minimum=0.0,
        maximum=1.0,
    ),
)


SLEEP_DAILY_FEATURE_SPECS = (
    _float_spec(
        "sleep_duration_norm",
        RiskDirection.TWO_SIDED,
        FeatureRole.PERSONAL_TREND,
        minimum=0.0,
        maximum=1.0,
    ),
    _float_spec(
        "time_in_bed_norm",
        RiskDirection.TWO_SIDED,
        FeatureRole.PERSONAL_TREND,
        minimum=0.0,
        maximum=1.0,
    ),
    _float_spec(
        "sleep_efficiency",
        RiskDirection.DECREASE,
        FeatureRole.PERSONAL_TREND,
        minimum=0.0,
        maximum=1.0,
    ),
    _integer_spec(
        "sleep_onset_minute_of_day",
        RiskDirection.CIRCULAR_TWO_SIDED,
        FeatureRole.PERSONAL_TREND,
        minimum=0,
        maximum=1439,
    ),
    _integer_spec(
        "wake_time_minute_of_day",
        RiskDirection.CIRCULAR_TWO_SIDED,
        FeatureRole.PERSONAL_TREND,
        minimum=0,
        maximum=1439,
    ),
    _float_spec(
        "sleep_midpoint_minute_of_day",
        RiskDirection.CIRCULAR_TWO_SIDED,
        FeatureRole.PERSONAL_TREND,
        minimum=0.0,
        maximum=1440.0,
        exclusive_maximum=True,
    ),
    _float_spec(
        "sleep_fragmentation",
        RiskDirection.INCREASE,
        FeatureRole.PERSONAL_TREND,
        minimum=0.0,
    ),
    _integer_spec(
        "night_exit_count",
        RiskDirection.INCREASE,
        FeatureRole.PERSONAL_TREND,
        minimum=0,
    ),
    _float_spec(
        "night_exit_minutes_norm",
        RiskDirection.INCREASE,
        FeatureRole.PERSONAL_TREND,
        minimum=0.0,
        maximum=1.0,
    ),
)


PHYSIOLOGY_DAILY_FEATURE_SPECS = PHYSIOLOGY_FEATURE_SPECS[:8]


SOCIAL_CONTACT_DAILY_FEATURE_SPECS = (
    _integer_spec(
        "incoming_call_opportunities",
        RiskDirection.MODEL_LEARNED,
        FeatureRole.PERSONAL_TREND,
        minimum=0,
    ),
    _integer_spec(
        "outgoing_call_count",
        RiskDirection.DECREASE,
        FeatureRole.PERSONAL_TREND,
        minimum=0,
    ),
    _integer_spec(
        "answered_call_count",
        RiskDirection.DECREASE,
        FeatureRole.PERSONAL_TREND,
        minimum=0,
    ),
    _integer_spec(
        "missed_call_count",
        RiskDirection.INCREASE,
        FeatureRole.PERSONAL_TREND,
        minimum=0,
    ),
    SOCIAL_CONTACT_FEATURE_SPECS[4],
    _integer_spec(
        "connected_call_count",
        RiskDirection.DECREASE,
        FeatureRole.PERSONAL_TREND,
        minimum=0,
    ),
    SOCIAL_CONTACT_FEATURE_SPECS[6],
    SOCIAL_CONTACT_FEATURE_SPECS[7],
    _integer_spec(
        "active_contact_count",
        RiskDirection.DECREASE,
        FeatureRole.PERSONAL_TREND,
        minimum=0,
    ),
    SOCIAL_CONTACT_FEATURE_SPECS[9],
)


CAMERA_GAIT_KEY_ORDER = ("date", "camera_id", "scene_version")

CAMERA_GAIT_FEATURE_SPECS = (
    _float_spec(
        "gait_speed_image_norm_per_sec",
        RiskDirection.DECREASE,
        FeatureRole.PERSONAL_TREND,
        minimum=0.0,
    ),
    _float_spec(
        "sit_to_stand_duration_seconds",
        RiskDirection.INCREASE,
        FeatureRole.PERSONAL_TREND,
        minimum=0.0,
        exclusive_minimum=True,
    ),
    _float_spec(
        "turn_duration_seconds",
        RiskDirection.INCREASE,
        FeatureRole.PERSONAL_TREND,
        minimum=0.0,
        exclusive_minimum=True,
    ),
    _float_spec(
        "postural_stability",
        RiskDirection.DECREASE,
        FeatureRole.PERSONAL_TREND,
        minimum=0.0,
        maximum=1.0,
    ),
)


FEATURE_GROUP_ORDER = (
    "activity",
    "sleep",
    "physiology",
    "social_context",
    "social_contact",
)

FEATURE_GROUP_SPECS: Mapping[str, tuple[FeatureSpec, ...]] = MappingProxyType(
    {
        "activity": ACTIVITY_FEATURE_SPECS,
        "sleep": SLEEP_FEATURE_SPECS,
        "physiology": PHYSIOLOGY_FEATURE_SPECS,
        "social_context": SOCIAL_CONTEXT_FEATURE_SPECS,
        "social_contact": SOCIAL_CONTACT_FEATURE_SPECS,
    }
)

DAILY_FEATURE_GROUP_SPECS: Mapping[str, tuple[FeatureSpec, ...]] = MappingProxyType(
    {
        "activity": ACTIVITY_DAILY_FEATURE_SPECS,
        "sleep": SLEEP_DAILY_FEATURE_SPECS,
        "physiology": PHYSIOLOGY_DAILY_FEATURE_SPECS,
        "social": SOCIAL_CONTACT_DAILY_FEATURE_SPECS,
    }
)


def _names(specs: tuple[FeatureSpec, ...]) -> tuple[str, ...]:
    return tuple(spec.name for spec in specs)


ACTIVITY_FEATURE_ORDER = _names(ACTIVITY_FEATURE_SPECS)
SLEEP_FEATURE_ORDER = _names(SLEEP_FEATURE_SPECS)
PHYSIOLOGY_FEATURE_ORDER = _names(PHYSIOLOGY_FEATURE_SPECS)
SOCIAL_CONTEXT_FEATURE_ORDER = _names(SOCIAL_CONTEXT_FEATURE_SPECS)
SOCIAL_CONTACT_FEATURE_ORDER = _names(SOCIAL_CONTACT_FEATURE_SPECS)
CAMERA_GAIT_FEATURE_ORDER = _names(CAMERA_GAIT_FEATURE_SPECS)


def _supervised_expert_inputs(
    specs: tuple[FeatureSpec, ...],
) -> tuple[str, ...]:
    # Valid-day counts are frozen model inputs.  feature_coverage is reserved
    # for confidence calculation and must not become a learned risk shortcut.
    return tuple(spec.name for spec in specs if spec.name != "feature_coverage")


ACTIVITY_EXPERT_FEATURE_ORDER = _supervised_expert_inputs(ACTIVITY_FEATURE_SPECS)
SLEEP_EXPERT_FEATURE_ORDER = _supervised_expert_inputs(SLEEP_FEATURE_SPECS)
PHYSIOLOGY_EXPERT_FEATURE_ORDER = _supervised_expert_inputs(PHYSIOLOGY_FEATURE_SPECS)

SUPERVISED_EXPERT_FEATURE_ORDER: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "activity": ACTIVITY_EXPERT_FEATURE_ORDER,
        "sleep": SLEEP_EXPERT_FEATURE_ORDER,
        "activity_sleep_joint": (
            *(f"activity.{name}" for name in ACTIVITY_EXPERT_FEATURE_ORDER),
            *(f"sleep.{name}" for name in SLEEP_EXPERT_FEATURE_ORDER),
        ),
        "physiology": PHYSIOLOGY_EXPERT_FEATURE_ORDER,
        "social_context": SOCIAL_CONTEXT_FEATURE_ORDER,
    }
)


FeatureValue = float | int | str | None


@dataclass(frozen=True)
class DomainFeatureVector:
    domain: str
    specs: tuple[FeatureSpec, ...]
    values: tuple[FeatureValue, ...]
    feature_mask: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.specs) != len(self.values):
            raise ValueError("feature values must match the frozen schema length")
        if len(self.specs) != len(self.feature_mask):
            raise ValueError("feature_mask must match the frozen schema length")
        names = self.feature_names
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate feature names in domain {self.domain!r}")
        for spec, value, mask in zip(
            self.specs,
            self.values,
            self.feature_mask,
            strict=True,
        ):
            if mask not in (0, 1):
                raise ValueError(f"feature mask for {spec.name!r} must be 0 or 1")
            if mask == 1 and value is None:
                raise ValueError(f"available feature {spec.name!r} cannot be null")
            if value is not None:
                _validate_feature_value(spec, value)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return _names(self.specs)

    def value(self, name: str) -> FeatureValue:
        return self.values[self.feature_names.index(name)]

    def mask(self, name: str) -> int:
        return self.feature_mask[self.feature_names.index(name)]

    def values_dict(self) -> dict[str, FeatureValue]:
        return dict(zip(self.feature_names, self.values, strict=True))

    def mask_dict(self) -> dict[str, int]:
        return dict(zip(self.feature_names, self.feature_mask, strict=True))

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "values": self.values_dict(),
            "feature_mask": self.mask_dict(),
        }


@dataclass(frozen=True)
class DayMask:
    date: date
    activity: int
    sleep: int
    physiology: int
    social: int

    def __post_init__(self) -> None:
        if any(
            value not in (0, 1)
            for value in (self.activity, self.sleep, self.physiology, self.social)
        ):
            raise ValueError("day masks must be 0 or 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date.isoformat(),
            "activity": self.activity,
            "sleep": self.sleep,
            "physiology": self.physiology,
            "social": self.social,
        }


@dataclass(frozen=True)
class DailyMappedFeatures:
    date: date
    activity: DomainFeatureVector
    sleep: DomainFeatureVector
    physiology: DomainFeatureVector
    social: DomainFeatureVector
    day_mask: DayMask

    def __post_init__(self) -> None:
        if self.day_mask.date != self.date:
            raise ValueError("day_mask date must match the daily feature date")
        groups = (
            ("activity", self.activity),
            ("sleep", self.sleep),
            ("physiology", self.physiology),
            ("social", self.social),
        )
        for domain, vector in groups:
            if vector.domain != domain:
                raise ValueError(f"daily {domain} vector has the wrong domain")
            if vector.specs != DAILY_FEATURE_GROUP_SPECS[domain]:
                raise ValueError(f"daily {domain} vector does not match its schema")
            if getattr(self.day_mask, domain) == 0 and any(vector.feature_mask):
                raise ValueError(
                    f"daily {domain} vector cannot contain observed features "
                    "when its day_mask is 0"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date.isoformat(),
            "activity": self.activity.to_dict(),
            "sleep": self.sleep.to_dict(),
            "physiology": self.physiology.to_dict(),
            "social": self.social.to_dict(),
            "day_mask": self.day_mask.to_dict(),
        }


@dataclass(frozen=True)
class CameraGaitDay:
    date: date
    camera_id: str
    scene_version: str
    gait_speed_image_norm_per_sec: float | None
    sit_to_stand_duration_seconds: float | None
    turn_duration_seconds: float | None
    postural_stability: float | None

    def __post_init__(self) -> None:
        if not isinstance(self.camera_id, str) or not 1 <= len(self.camera_id) <= 64:
            raise ValueError("camera_id must contain 1 to 64 characters")
        if (
            not isinstance(self.scene_version, str)
            or not 1 <= len(self.scene_version) <= 80
        ):
            raise ValueError("scene_version must contain 1 to 80 characters")
        values = tuple(getattr(self, spec.name) for spec in CAMERA_GAIT_FEATURE_SPECS)
        if all(value is None for value in values):
            raise ValueError("camera gait context requires an observed motion value")
        for spec, value in zip(CAMERA_GAIT_FEATURE_SPECS, values, strict=True):
            if value is not None:
                _validate_feature_value(spec, value)

    def to_dict(self) -> dict[str, Any]:
        return {
            **dict(
                zip(
                    CAMERA_GAIT_KEY_ORDER,
                    (self.date.isoformat(), self.camera_id, self.scene_version),
                    strict=True,
                )
            ),
            **{name: getattr(self, name) for name in CAMERA_GAIT_FEATURE_ORDER},
        }


@dataclass(frozen=True)
class TrendContext:
    camera_gait_days: tuple[CameraGaitDay, ...]

    def __post_init__(self) -> None:
        keys = tuple(
            (item.date, item.camera_id, item.scene_version)
            for item in self.camera_gait_days
        )
        if keys != tuple(sorted(keys)):
            raise ValueError("camera gait context must use stable date/scene order")
        if len(keys) != len(set(keys)):
            raise ValueError("camera gait context must not repeat a date/scene key")

    def to_dict(self) -> dict[str, Any]:
        return {"camera_gait_days": [item.to_dict() for item in self.camera_gait_days]}


@dataclass(frozen=True)
class MappedMoodSocialFeatures:
    schema_version: str
    target_date: date
    window_start_date: date
    window_end_date: date
    daily_features: tuple[DailyMappedFeatures, ...]
    activity: DomainFeatureVector
    sleep: DomainFeatureVector
    physiology: DomainFeatureVector
    social_context: DomainFeatureVector
    social_contact: DomainFeatureVector
    trend_context: TrendContext

    def __post_init__(self) -> None:
        if self.schema_version != FEATURE_SCHEMA_VERSION:
            raise ValueError("unsupported feature schema version")
        actual_dates = tuple(item.date for item in self.daily_features)
        if len(actual_dates) != STATE_WINDOW_DAYS:
            raise ValueError("daily_features must contain exactly seven calendar days")
        expected_dates = tuple(
            self.target_date - timedelta(days=offset)
            for offset in range(STATE_WINDOW_DAYS - 1, -1, -1)
        )
        if actual_dates != expected_dates:
            raise ValueError(
                "daily_features must cover exact calendar days D-6 through D"
            )
        if actual_dates[0] != self.window_start_date:
            raise ValueError("window_start_date does not match daily_features")
        if actual_dates[-1] != self.window_end_date:
            raise ValueError("window_end_date does not match daily_features")
        if self.window_end_date != self.target_date:
            raise ValueError("the seven-day state window must end on target_date")
        groups = (
            ("activity", self.activity),
            ("sleep", self.sleep),
            ("physiology", self.physiology),
            ("social_context", self.social_context),
            ("social_contact", self.social_contact),
        )
        for domain, vector in groups:
            if vector.domain != domain:
                raise ValueError(f"windowed {domain} vector has the wrong domain")
            if vector.specs != FEATURE_GROUP_SPECS[domain]:
                raise ValueError(f"windowed {domain} vector does not match its schema")
        earliest_context_date = self.target_date - timedelta(days=HISTORY_LOOKBACK_DAYS)
        if any(
            not earliest_context_date <= item.date <= self.target_date
            for item in self.trend_context.camera_gait_days
        ):
            raise ValueError("camera gait context must stay within D-28 through D")

    @property
    def feature_groups(self) -> tuple[DomainFeatureVector, ...]:
        return (
            self.activity,
            self.sleep,
            self.physiology,
            self.social_context,
            self.social_contact,
        )

    def feature_mask_dict(self) -> dict[str, dict[str, int]]:
        return {group.domain: group.mask_dict() for group in self.feature_groups}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "target_date": self.target_date.isoformat(),
            "window_start_date": self.window_start_date.isoformat(),
            "window_end_date": self.window_end_date.isoformat(),
            "daily_features": [item.to_dict() for item in self.daily_features],
            "features": {
                group.domain: group.to_dict() for group in self.feature_groups
            },
            "feature_mask": self.feature_mask_dict(),
            "day_mask": [item.day_mask.to_dict() for item in self.daily_features],
            "trend_context": self.trend_context.to_dict(),
        }


def feature_schema_manifest() -> dict[str, Any]:
    """Return a JSON-ready manifest with deterministic group and field order."""

    return {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "state_window_days": STATE_WINDOW_DAYS,
        "history_lookback_days": HISTORY_LOOKBACK_DAYS,
        "group_order": list(FEATURE_GROUP_ORDER),
        "groups": {
            group: [spec.to_dict() for spec in FEATURE_GROUP_SPECS[group]]
            for group in FEATURE_GROUP_ORDER
        },
        "daily_groups": {
            group: [spec.to_dict() for spec in specs]
            for group, specs in DAILY_FEATURE_GROUP_SPECS.items()
        },
        "trend_context": {
            "camera_gait": {
                "key_order": list(CAMERA_GAIT_KEY_ORDER),
                "features": [spec.to_dict() for spec in CAMERA_GAIT_FEATURE_SPECS],
            }
        },
        "supervised_expert_feature_order": {
            expert: list(names)
            for expert, names in SUPERVISED_EXPERT_FEATURE_ORDER.items()
        },
    }


def _validate_feature_value(spec: FeatureSpec, value: float | int | str) -> None:
    numeric_value: float | int | None = None
    if spec.value_type == FeatureValueType.FLOAT:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"feature {spec.name!r} must be a finite float")
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            raise ValueError(f"feature {spec.name!r} must be finite")
    elif spec.value_type == FeatureValueType.INTEGER:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"feature {spec.name!r} must be an integer")
        numeric_value = value
    elif spec.value_type == FeatureValueType.CATEGORY:
        if not isinstance(value, str):
            raise TypeError(f"feature {spec.name!r} must be a category string")
        if value not in spec.categories:
            raise ValueError(f"feature {spec.name!r} has an unsupported category")
        return

    assert numeric_value is not None
    if spec.minimum is not None:
        below_minimum = (
            numeric_value <= spec.minimum
            if spec.exclusive_minimum
            else numeric_value < spec.minimum
        )
        if below_minimum:
            raise ValueError(f"feature {spec.name!r} is below its minimum")
    if spec.maximum is not None:
        above_maximum = (
            numeric_value >= spec.maximum
            if spec.exclusive_maximum
            else numeric_value > spec.maximum
        )
        if above_maximum:
            raise ValueError(f"feature {spec.name!r} is above its maximum")


__all__ = [
    "ACTIVITY_DAILY_FEATURE_SPECS",
    "ACTIVITY_EXPERT_FEATURE_ORDER",
    "ACTIVITY_FEATURE_ORDER",
    "ACTIVITY_FEATURE_SPECS",
    "CAMERA_GAIT_FEATURE_ORDER",
    "CAMERA_GAIT_FEATURE_SPECS",
    "CAMERA_GAIT_KEY_ORDER",
    "CAMERA_DAYTIME_END",
    "CAMERA_DAYTIME_MINUTES",
    "CAMERA_DAYTIME_START",
    "DAILY_FEATURE_GROUP_SPECS",
    "FEATURE_GROUP_ORDER",
    "FEATURE_GROUP_SPECS",
    "FEATURE_SCHEMA_VERSION",
    "HISTORY_LOOKBACK_DAYS",
    "NO_EFFECTIVE_CONTACT_MAX_DAYS",
    "PHYSIOLOGY_DAILY_FEATURE_SPECS",
    "PHYSIOLOGY_EXPERT_FEATURE_ORDER",
    "PHYSIOLOGY_FEATURE_ORDER",
    "PHYSIOLOGY_FEATURE_SPECS",
    "SLEEP_DAILY_FEATURE_SPECS",
    "SLEEP_EXPERT_FEATURE_ORDER",
    "SLEEP_FEATURE_ORDER",
    "SLEEP_FEATURE_SPECS",
    "SOCIAL_CONTACT_DAILY_FEATURE_SPECS",
    "SOCIAL_CONTACT_FEATURE_ORDER",
    "SOCIAL_CONTACT_FEATURE_SPECS",
    "SOCIAL_CONTEXT_FEATURE_ORDER",
    "SOCIAL_CONTEXT_FEATURE_SPECS",
    "STATE_WINDOW_DAYS",
    "SUPERVISED_EXPERT_FEATURE_ORDER",
    "CameraGaitDay",
    "DailyMappedFeatures",
    "DayMask",
    "DomainFeatureVector",
    "FeatureRole",
    "FeatureSpec",
    "FeatureValue",
    "FeatureValueType",
    "MappedMoodSocialFeatures",
    "RiskDirection",
    "TrendContext",
    "feature_schema_manifest",
]
