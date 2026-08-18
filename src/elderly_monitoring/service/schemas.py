from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
SessionStatus = Literal["starting", "running", "reconnecting", "stopping", "stopped", "failed"]


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


class BaselinePeriodUpdate(BaseModel):
    """One completed upstream day/hour aggregate for personal baseline scoring."""

    model_config = ConfigDict(extra="forbid")

    period: dict[str, Any]

    @field_validator("period")
    @classmethod
    def validate_period(cls, value: dict[str, Any]) -> dict[str, Any]:
        if not value:
            raise ValueError("period must not be empty")
        return value


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
    stream_epoch: int = 0
    frame_diagnostics: dict[str, Any] = Field(default_factory=dict)
    runtime_diagnostics: dict[str, Any] = Field(default_factory=dict)
