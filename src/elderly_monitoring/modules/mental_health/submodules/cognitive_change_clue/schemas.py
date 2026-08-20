"""Strict HTTP contract for cognitive-change clue inference V3.3/V3.4."""

from __future__ import annotations

from typing import Annotated, Literal
from urllib.parse import urlparse

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)


REQUEST_SCHEMA_VERSION = "cognitive_infer_request_v1"
RESPONSE_SCHEMA_VERSION = "cognitive_infer_response_v1"
ERROR_SCHEMA_VERSION = "cognitive_error_response_v1"
DEFAULT_MODEL_VERSION = "cognitive-mm-v3.3.0"
V34_MODEL_VERSION = "cognitive-mm-v3.4.0"
SUPPORTED_MODEL_VERSIONS = (DEFAULT_MODEL_VERSION, V34_MODEL_VERSION)
# Backwards-compatible alias for callers that mean the stable default model.
MODEL_VERSION = DEFAULT_MODEL_VERSION
CognitiveModelVersion = Literal[
    "cognitive-mm-v3.3.0",
    "cognitive-mm-v3.4.0",
]

CognitiveStatus = Literal["completed", "degraded", "insufficient_input"]
CognitiveLevel = Literal["normal", "attention", "high_attention"]
ModalityName = Literal["audio", "text", "face"]
WarningModality = Literal["audio", "text", "face", "model"]
WarningCode = Literal[
    "missing_audio",
    "missing_text",
    "missing_face",
    "low_audio_quality",
    "low_text_quality",
    "low_face_quality",
    "asr_empty_speech",
    "asr_failed",
    "model_defaulted",
    "voice_prompt_adaptation",
]
CognitiveErrorCode = Literal[
    "AUTHENTICATION_FAILED",
    "UNSUPPORTED_SCHEMA_VERSION",
    "UNKNOWN_FIELD",
    "FIELD_VALIDATION_ERROR",
    "MEDIA_INVALID",
    "MEDIA_TOO_LARGE",
    "MEDIA_TOO_LONG",
    "MEDIA_MISALIGNED",
    "MEDIA_DECODE_FAILED",
    "MODEL_VERSION_UNSUPPORTED",
    "MEDIA_UNREACHABLE",
    "MODEL_ARTIFACT_UNAVAILABLE",
    "ALGORITHM_TIMEOUT",
    "INTERNAL_ERROR",
]
ASRStatus = Literal[
    "not_requested",
    "completed",
    "completed_empty_speech",
    "unsupported_audio",
    "model_unavailable",
    "failed",
]


class CognitiveSchemaModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _non_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("must not be blank")
    return value


RequestId = Annotated[
    str,
    StringConstraints(strict=True, min_length=1, max_length=128),
    AfterValidator(_non_blank),
]
SourceText = Annotated[
    str,
    StringConstraints(strict=True, min_length=1),
    AfterValidator(_non_blank),
]
Probability = Annotated[
    float,
    Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False),
]


class CaptureWindow(CognitiveSchemaModel):
    start_ms: Annotated[int, Field(strict=True, ge=0)]
    end_ms: Annotated[int, Field(strict=True, ge=0)]

    @model_validator(mode="after")
    def validate_duration(self) -> "CaptureWindow":
        duration = self.end_ms - self.start_ms
        if not 3_000 <= duration <= 60_000:
            raise ValueError("capture window duration must be between 3000 and 60000 ms")
        return self

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


class AudioMediaInput(CognitiveSchemaModel):
    source_type: Literal["url", "base64"]
    source: SourceText
    format: Literal["wav", "mp3", "m4a"]
    sample_rate: Annotated[int, Field(strict=True, ge=8_000, le=48_000)] | None
    channels: Literal[1, 2] | None

    @model_validator(mode="after")
    def validate_source(self) -> "AudioMediaInput":
        if self.source_type == "url":
            parsed = urlparse(self.source)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("URL source must be an absolute HTTP or HTTPS URL")
        return self


class FaceMediaInput(CognitiveSchemaModel):
    source_type: Literal["url", "base64"]
    source: SourceText
    format: Literal["mp4"]
    media_type: Literal["video"]
    fps: Annotated[int, Field(strict=True, ge=1, le=60)] | None

    @model_validator(mode="after")
    def validate_source(self) -> "FaceMediaInput":
        if self.source_type == "url":
            parsed = urlparse(self.source)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("URL source must be an absolute HTTP or HTTPS URL")
        return self


class CognitiveASROptions(CognitiveSchemaModel):
    language: Literal["zh"]
    return_timestamps: Literal[True]


class CognitiveInferRequest(CognitiveSchemaModel):
    schema_version: Literal["cognitive_infer_request_v1"]
    request_id: RequestId
    capture_window: CaptureWindow
    audio_input: AudioMediaInput | None
    face_input: FaceMediaInput | None
    asr_options: CognitiveASROptions
    model_version: str | None = None


class CognitiveResearchClassScores(CognitiveSchemaModel):
    hc: Probability
    mci: Probability
    ad: Probability


class CognitiveResearchOutputs(CognitiveSchemaModel):
    mci_hc_score: Probability
    ad_mci_hc_scores: CognitiveResearchClassScores
    moca_prediction: Annotated[
        float,
        Field(strict=True, ge=0.0, le=30.0, allow_inf_nan=False),
    ]


class CognitiveModalityQuality(CognitiveSchemaModel):
    audio: Probability
    text: Probability
    face: Probability


class CognitiveWarning(CognitiveSchemaModel):
    code: WarningCode
    modality: WarningModality
    message: Annotated[str, StringConstraints(strict=True, min_length=1)]


class CognitiveInferResponse(CognitiveSchemaModel):
    schema_version: Literal["cognitive_infer_response_v1"]
    request_id: RequestId
    status: CognitiveStatus
    cognitive_clue_score: Annotated[
        float,
        Field(strict=True, ge=0.0, le=100.0, allow_inf_nan=False),
    ] | None
    cognitive_clue_level: CognitiveLevel | None
    confidence: Probability | None
    research_outputs: CognitiveResearchOutputs | None
    used_modalities: list[ModalityName]
    modality_quality: CognitiveModalityQuality
    asr_status: ASRStatus
    model_version: CognitiveModelVersion
    warnings: list[CognitiveWarning]

    @model_validator(mode="after")
    def validate_response_shape(self) -> "CognitiveInferResponse":
        order = ["audio", "text", "face"]
        if self.used_modalities != sorted(set(self.used_modalities), key=order.index):
            raise ValueError("used_modalities must be unique and use audio/text/face order")
        warning_codes = [item.code for item in self.warnings]
        if len(warning_codes) != len(set(warning_codes)):
            raise ValueError("warning codes must not repeat")

        score_fields = (
            self.cognitive_clue_score,
            self.cognitive_clue_level,
            self.confidence,
        )
        if self.status == "insufficient_input":
            if (
                any(value is not None for value in (*score_fields, self.research_outputs))
                or self.used_modalities
            ):
                raise ValueError("insufficient_input must not expose model results")
            return self

        if any(value is None for value in score_fields) or "audio" not in self.used_modalities:
            raise ValueError("completed and degraded results require audio and model results")
        required_modalities = (
            order
            if self.model_version == DEFAULT_MODEL_VERSION
            else ["audio", "text"]
        )
        if self.status == "completed" and self.used_modalities != required_modalities:
            raise ValueError("completed requires every primary model modality")
        if self.status == "degraded" and self.used_modalities == required_modalities:
            raise ValueError("degraded must have at least one unavailable primary modality")
        if self.model_version == DEFAULT_MODEL_VERSION and self.research_outputs is None:
            raise ValueError("V3.3 available results require research outputs")
        if self.model_version == V34_MODEL_VERSION and self.research_outputs is not None:
            raise ValueError("V3.4 main-head model must not expose research outputs")
        return self


class CognitiveErrorItem(CognitiveSchemaModel):
    field: Annotated[str, StringConstraints(strict=True, min_length=1)]
    reason: Annotated[str, StringConstraints(strict=True, min_length=1)]


class CognitiveErrorDetail(CognitiveSchemaModel):
    schema_version: Literal["cognitive_error_response_v1"]
    request_id: str | None
    code: CognitiveErrorCode
    message: Annotated[str, StringConstraints(strict=True, min_length=1)]
    errors: list[CognitiveErrorItem]


class CognitiveErrorResponse(CognitiveSchemaModel):
    detail: CognitiveErrorDetail


__all__ = [
    "ASRStatus",
    "CognitiveModelVersion",
    "DEFAULT_MODEL_VERSION",
    "ERROR_SCHEMA_VERSION",
    "MODEL_VERSION",
    "REQUEST_SCHEMA_VERSION",
    "RESPONSE_SCHEMA_VERSION",
    "SUPPORTED_MODEL_VERSIONS",
    "V34_MODEL_VERSION",
    "AudioMediaInput",
    "CaptureWindow",
    "CognitiveASROptions",
    "CognitiveErrorCode",
    "CognitiveErrorDetail",
    "CognitiveErrorItem",
    "CognitiveErrorResponse",
    "CognitiveInferRequest",
    "CognitiveInferResponse",
    "CognitiveModalityQuality",
    "CognitiveResearchClassScores",
    "CognitiveResearchOutputs",
    "CognitiveWarning",
    "FaceMediaInput",
]
