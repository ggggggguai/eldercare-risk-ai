"""Versioned request and response models for the ASR boundary."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


ASRStatus = Literal[
    "completed",
    "completed_empty_speech",
    "unsupported_audio",
    "model_unavailable",
    "failed",
]


class AudioInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_type: Literal["url", "base64"]
    source: str = Field(min_length=1)
    format: Literal["wav", "mp3", "m4a"]
    sample_rate: int | None = Field(ge=8000, le=48000)
    channels: Literal[1, 2] | None

    @field_validator("source")
    @classmethod
    def source_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("audio source must not be blank")
        return value


class ASRRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["asr_request_v1"] = "asr_request_v1"
    request_id: str = Field(min_length=1, max_length=128)
    audio_input: AudioInput
    language: Literal["zh"] = "zh"
    enable_vad: Literal[True] = True
    enable_punctuation: Literal[True] = True
    return_timestamps: Literal[True] = True
    model_version: Literal["asr-paraformer-zh-v1.0"] = "asr-paraformer-zh-v1.0"

    @field_validator("request_id")
    @classmethod
    def request_id_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("request_id must not be blank")
        return value


class ASRSegment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    segment_id: str
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    text: str
    confidence: float | None = Field(default=None, ge=0, le=1)


class ASRQuality(BaseModel):
    model_config = ConfigDict(extra="forbid")

    audio_duration_ms: int = Field(ge=0)
    speech_duration_ms: int = Field(ge=0)
    speech_ratio: float = Field(ge=0, le=1)
    mean_confidence: float | None = Field(default=None, ge=0, le=1)
    decode_status: Literal["ok", "failed"]
    vad_status: Literal["ok", "disabled", "failed", "empty"]


class ASRModelInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "funasr-paraformer-zh"
    version: str = "asr-paraformer-zh-v1.0"


class ASRTranscript(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["asr_transcript_v1"] = "asr_transcript_v1"
    request_id: str
    status: ASRStatus
    text: str = ""
    segments: list[ASRSegment] = Field(default_factory=list)
    language: Literal["zh"] = "zh"
    quality: ASRQuality
    model: ASRModelInfo = Field(default_factory=ASRModelInfo)
    warnings: list[str] = Field(default_factory=list)
