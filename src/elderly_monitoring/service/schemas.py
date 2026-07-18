from __future__ import annotations

from datetime import date as Date, datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
SessionStatus = Literal["starting", "running", "reconnecting", "stopping", "stopped", "failed"]
RoiType = Literal[
    "bed",
    "sofa",
    "dining_table",
    "doorway",
    "bathroom_entrance",
    "high_risk_passage",
    "activity_area",
    "ignore_area",
    "unknown",
]


def _validate_url(value: str, schemes: set[str]) -> str:
    value = value.strip()
    scheme, separator, remainder = value.partition("://")
    if not separator or scheme.lower() not in schemes or not remainder:
        raise ValueError(f"URL scheme must be one of: {', '.join(sorted(schemes))}")
    return value


class StartSessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: NonEmptyString
    stream_url: str
    device_id: NonEmptyString
    person_id: NonEmptyString
    scene_region: NonEmptyString
    callback_url: str

    @field_validator("stream_url")
    @classmethod
    def validate_stream_url(cls, value: str) -> str:
        return _validate_url(value, {"rtsp", "rtmp", "http", "https"})

    @field_validator("callback_url")
    @classmethod
    def validate_callback_url(cls, value: str) -> str:
        return _validate_url(value, {"http", "https"})


class StreamUrlUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stream_url: str

    @field_validator("stream_url")
    @classmethod
    def validate_stream_url(cls, value: str) -> str:
        return _validate_url(value, {"rtsp", "rtmp", "http", "https"})


class SessionAccepted(BaseModel):
    session_id: str
    status: SessionStatus


class SessionStatusResponse(BaseModel):
    session_id: str
    status: SessionStatus
    device_id: str
    person_id: str
    started_at: datetime
    last_frame_at: datetime | None = None
    last_error: str | None = None


class RoiAnnotateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_base64: NonEmptyString
    mime_type: Literal["image/jpeg", "image/png", "image/webp"] = "image/jpeg"
    image_width: int = Field(gt=0, le=10000)
    image_height: int = Field(gt=0, le=10000)
    scene_hint: str = Field(default="", max_length=500)
    expected_types: list[RoiType] = Field(default_factory=list)
    device_id: str | None = Field(default=None, max_length=64)


class RoiAnnotateResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    schema_version: str
    image: dict
    rois: list[dict]
    missing_expected: list[str]
    warnings: list[str]
    needs_human_review: bool = True
    model: dict


class DaytimeActivityFrameRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    timestamp: datetime | None = None
    observed_at: datetime | None = None
    person_id: str | None = Field(default=None, max_length=64)
    device_id: str | None = Field(default=None, max_length=64)
    camera_id: str | None = Field(default=None, max_length=64)
    bbox: list[float] | None = None
    bbox_format: Literal["xywh", "xyxy"] = "xywh"
    bbox_confidence: float | None = Field(default=None, ge=0, le=1)
    keypoints: list[Any] | None = None
    keypoint_confidence: float | None = Field(default=None, ge=0, le=1)
    tracking_confidence: float | None = Field(default=None, ge=0, le=1)
    zone: str | None = Field(default=None, max_length=64)
    room: str | None = Field(default=None, max_length=64)
    zone_id: str | None = Field(default=None, max_length=64)
    room_id: str | None = Field(default=None, max_length=64)
    posture: str | None = Field(default=None, max_length=64)
    data_quality: str | None = Field(default=None, max_length=64)
    image_width: int | None = Field(default=None, gt=0, le=10000)
    image_height: int | None = Field(default=None, gt=0, le=10000)


class DaytimeActivityWindowRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    window_start: datetime
    window_end: datetime
    person_id: str | None = Field(default=None, max_length=64)
    room: str | None = Field(default=None, max_length=64)
    zone: str | None = Field(default=None, max_length=64)
    room_id: str | None = Field(default=None, max_length=64)
    zone_id: str | None = Field(default=None, max_length=64)
    active_score: float | None = Field(default=None, ge=0, le=1)
    motion_state: str = "data_missing"
    posture: str | None = Field(default=None, max_length=64)
    valid_detection_ratio: float = Field(default=0.0, ge=0, le=1)
    data_quality: str = "valid"
    center_path_norm: float | None = Field(default=None, ge=0)
    pose_motion_norm: float | None = Field(default=None, ge=0)
    zone_transition_score: float = Field(default=0.0, ge=0, le=1)
    posture_change_score: float = Field(default=0.0, ge=0, le=1)
    quality_flags: list[str] = Field(default_factory=list)


class DaytimeActivityRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    person_id: NonEmptyString
    date: Date | None = None
    frames: list[DaytimeActivityFrameRequest] = Field(default_factory=list)
    windows: list[DaytimeActivityWindowRequest] = Field(default_factory=list)
    sleep_records: list[dict[str, Any]] = Field(default_factory=list)
    history_daily_features: list[dict[str, Any]] = Field(default_factory=list)
    roi_annotations: list[dict[str, Any]] = Field(default_factory=list)
    image_width: int | None = Field(default=None, gt=0, le=10000)
    image_height: int | None = Field(default=None, gt=0, le=10000)


class DaytimeActivityResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    schema_version: str = "daytime_activity_service_v1"
    model_version: str = "daytime-activity-v1"
    person_id: str
    requested_date: Date | None = None
    windows: list[dict[str, Any]]
    daily_features: list[dict[str, Any]]
    quality_flags: list[str] = Field(default_factory=list)


class SleepRhythmRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    person_id: NonEmptyString
    date: Date | None = None
    device_id: str | None = Field(default=None, max_length=64)
    device_serial: str | None = Field(default=None, max_length=64)
    reports: list[dict[str, Any]] = Field(default_factory=list)
    body_detect_messages: list[dict[str, Any]] = Field(default_factory=list)
    daily_features: list[dict[str, Any]] = Field(default_factory=list)
    history_daily_features: list[dict[str, Any]] = Field(default_factory=list)


class SleepRhythmResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    schema_version: str = "sleep_rhythm_service_v1"
    model_version: str = "sleep-rhythm-rulecard-v1"
    person_id: str
    requested_date: str | None = None
    daily_features: list[dict[str, Any]]
    quality_flags: list[str] = Field(default_factory=list)
    medical_disclaimer: str


class NightPhysiologyRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    person_id: NonEmptyString
    date: Date | None = None
    device_id: str | None = Field(default=None, max_length=64)
    device_serial: str | None = Field(default=None, max_length=64)
    daily_features: list[dict[str, Any]] = Field(default_factory=list)
    history_daily_features: list[dict[str, Any]] = Field(default_factory=list)


class NightPhysiologyResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    schema_version: str = "night_physiology_service_v1"
    model_version: str = "night-physiology-rulecard-v1"
    person_id: str
    requested_date: str | None = None
    daily_features: list[dict[str, Any]]
    quality_flags: list[str] = Field(default_factory=list)
    medical_disclaimer: str


class SocialConnectionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    person_id: NonEmptyString
    date: Date | None = None
    device_id: str | None = Field(default=None, max_length=64)
    call_events: list[dict[str, Any]] = Field(default_factory=list)
    daily_features: list[dict[str, Any]] = Field(default_factory=list)
    history_daily_features: list[dict[str, Any]] = Field(default_factory=list)


class SocialConnectionResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    schema_version: str = "social_connection_service_v1"
    model_version: str = "social-connection-rulecard-v1"
    person_id: str
    requested_date: str | None = None
    daily_features: list[dict[str, Any]]
    quality_flags: list[str] = Field(default_factory=list)
    privacy_boundary: str
    medical_disclaimer: str


class MentalHealthDailyRiskRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    person_id: NonEmptyString
    history_daily_features: list[dict[str, Any]] = Field(default_factory=list)
    current_daily_features: list[dict[str, Any]]


class MentalHealthDailyRiskResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    schema_version: str = "mental_health_daily_risk_service_v1"
    model_version: str
    person_id: str
    results: list[dict[str, Any]]
    medical_disclaimer: str
