from __future__ import annotations

from datetime import datetime
import hmac
from typing import Any, Literal, Protocol
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .deployment import (
    B0_DEPLOYMENT_MODEL_VERSION,
    B0_ENSEMBLE_RULE,
    B0_EXPECTED_RUN_LOCK_SHA256,
    B0DeploymentError,
)
from .video_pipeline import (
    FACIAL_AFFECT_VIDEO_INFER_PATH,
    FACIAL_AFFECT_VIDEO_REQUEST_SCHEMA,
    FACIAL_AFFECT_VIDEO_RESPONSE_SCHEMA,
    FacialAffectVideoError,
)


class FacialAffectVideoRuntimeProtocol(Protocol):
    def verify_package(self) -> dict[str, Any]: ...
    def infer(self, payload: dict[str, Any]) -> dict[str, Any]: ...


class FacialAffectTaskVideo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_slot: Literal[1, 2, 3]
    source_type: Literal["url"] = "url"
    source: str = Field(min_length=10, max_length=4096)
    format: Literal["mp4"] = "mp4"

    @field_validator("source")
    @classmethod
    def require_absolute_http_url(cls, value: str) -> str:
        try:
            parsed = urlsplit(value)
            _ = parsed.port
        except ValueError as exc:
            raise ValueError("source must be an absolute HTTP/HTTPS URL") from exc
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise ValueError("source must be an absolute HTTP/HTTPS URL without userinfo")
        return value


class FacialAffectVideoInferRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    schema_version: Literal["facial_affect_video_request_v1"] = (
        FACIAL_AFFECT_VIDEO_REQUEST_SCHEMA
    )
    request_id: str = Field(min_length=1, max_length=128)
    person_id: str = Field(min_length=1, max_length=128)
    session_id: str = Field(min_length=1, max_length=128)
    observed_at: datetime
    task_videos: list[FacialAffectTaskVideo] = Field(min_length=3, max_length=3)
    model_version: Literal[
        "facial-affect-causalnet-b0-opt-me-008-deploy-v1"
    ] = B0_DEPLOYMENT_MODEL_VERSION

    @field_validator("request_id", "person_id", "session_id")
    @classmethod
    def reject_control_or_whitespace_ids(cls, value: str) -> str:
        if value.strip() != value or any(ord(char) < 33 or ord(char) == 127 for char in value):
            raise ValueError("identifier must not contain whitespace or control characters")
        return value

    @field_validator("observed_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("observed_at must include a timezone offset")
        return value

    @model_validator(mode="after")
    def validate_task_slots(self):
        if [item.task_slot for item in self.task_videos] != [1, 2, 3]:
            raise ValueError("task_videos must contain ordered slots 1, 2, 3")
        return self


class FacialAffectVideoProbabilities(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    negative: float = Field(ge=0, le=1)
    positive: float = Field(ge=0, le=1)
    surprise: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def sum_to_one(self):
        if abs(self.negative + self.positive + self.surprise - 1.0) > 1e-6:
            raise ValueError("probabilities must sum to one")
        return self


class FacialAffectVideoTaskResult(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    task_slot: Literal[1, 2, 3]
    status: Literal["completed", "insufficient_input"]
    decoded_frame_count: int = Field(ge=0)
    effective_fps: float = Field(ge=0)
    container_duration_ms: int = Field(default=0, ge=0)
    decoded_duration_ms: int = Field(default=0, ge=0)
    invalid_frame_count: int = Field(default=0, ge=0)
    rotation_degrees: int = Field(default=0, ge=-360, le=360)
    face_coverage: float = Field(ge=0, le=1)
    candidate_count: int = Field(ge=0)
    track_continuity: float = Field(default=0, ge=0, le=1)
    face_scale_mean: float = Field(default=0, ge=0, le=1)
    face_scale_cv: float = Field(default=0, ge=0)
    maximum_track_displacement_ratio: float = Field(default=0, ge=0)
    multiple_face_conflicts: int = Field(default=0, ge=0)
    interpolated_face_frames: int = Field(default=0, ge=0)
    detector_version: str = Field(default="unknown", min_length=1, max_length=128)
    candidate_duration_ms: int | None = Field(default=None, ge=0)
    peak_score: float | None = Field(default=None, ge=0)
    head_motion_ratio: float | None = Field(default=None, ge=0)
    illumination_delta: float | None = Field(default=None, ge=0)
    onset_ms: int | None = Field(default=None, ge=0)
    apex_ms: int | None = Field(default=None, ge=0)
    offset_ms: int | None = Field(default=None, ge=0)
    onset_index: int | None = Field(default=None, ge=0)
    apex_index: int | None = Field(default=None, ge=0)
    offset_index: int | None = Field(default=None, ge=0)
    prediction: Literal["negative", "positive", "surprise"] | None = None
    probabilities: FacialAffectVideoProbabilities | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    quality_score: float = Field(ge=0, le=1)
    quality_status: Literal["valid", "warning", "invalid"]
    warnings: list[str]
    input_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    preprocessing_config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    spotting_config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    timings_ms: dict[str, int]

    @model_validator(mode="after")
    def validate_status_fields(self):
        keyframes = (
            self.onset_index,
            self.apex_index,
            self.offset_index,
            self.onset_ms,
            self.apex_ms,
            self.offset_ms,
        )
        prediction_fields = (
            self.prediction,
            self.probabilities,
            self.confidence,
            self.input_sha256,
        )
        if self.status == "completed":
            if any(value is None for value in (*keyframes, *prediction_fields)):
                raise ValueError("completed task must contain keyframes and prediction")
            if not (
                self.onset_index < self.apex_index < self.offset_index
                and self.onset_ms < self.apex_ms < self.offset_ms
            ):
                raise ValueError("completed task keyframes must be strictly ordered")
            values = self.probabilities.model_dump()
            if self.prediction != max(values, key=values.get):
                raise ValueError("prediction must match probability argmax")
            if abs(self.confidence - max(values.values())) > 1e-6:
                raise ValueError("confidence must equal maximum probability")
        elif any(value is not None for value in (*keyframes, *prediction_fields)):
            raise ValueError("insufficient task must not contain a prediction")
        return self


class FacialAffectVideoSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    source_task_slot: Literal[1, 2, 3]
    prediction: Literal["negative", "positive", "surprise"]
    probabilities: FacialAffectVideoProbabilities
    confidence: float = Field(ge=0, le=1)
    quality_score: float = Field(ge=0, le=1)


class FacialAffectVideoModelLineage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidate_id: Literal["B0"] = "B0"
    run_lock_sha256: Literal[
        "e1266924e54631a05ddc90a4d77d5457c8d9acacfce348ccbb49e3f00c536bb2"
    ] = B0_EXPECTED_RUN_LOCK_SHA256
    package_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_count: Literal[20] = 20
    ensemble_rule: Literal[
        "mean_of_fold_temperature_calibrated_probabilities"
    ] = B0_ENSEMBLE_RULE
    spotting_version: Literal["s10_aligned_roi_motion_strain_v1"]
    spotting_config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation_scope: Literal["CASME II exposed development estimate"]
    deployment_ensemble_independently_validated: Literal[False]


class FacialAffectVideoInferResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    schema_version: Literal["facial_affect_video_response_v1"] = (
        FACIAL_AFFECT_VIDEO_RESPONSE_SCHEMA
    )
    request_id: str
    person_id: str
    session_id: str
    observed_at: datetime
    status: Literal["completed", "degraded", "insufficient_input"]
    model_version: Literal[
        "facial-affect-causalnet-b0-opt-me-008-deploy-v1"
    ] = B0_DEPLOYMENT_MODEL_VERSION
    task_results: list[FacialAffectVideoTaskResult] = Field(min_length=3, max_length=3)
    summary_result: FacialAffectVideoSummary | None
    model_lineage: FacialAffectVideoModelLineage
    diagnosis: Literal[False] = False
    automatic_risk_level_mapping: Literal[False] = False
    medical_disclaimer: str
    warnings: list[str]

    @field_validator("request_id", "person_id", "session_id")
    @classmethod
    def reject_control_or_whitespace_ids(cls, value: str) -> str:
        if value.strip() != value or any(
            ord(char) < 33 or ord(char) == 127 for char in value
        ):
            raise ValueError(
                "identifier must not contain whitespace or control characters"
            )
        return value

    @field_validator("observed_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("observed_at must include a timezone offset")
        return value

    @model_validator(mode="after")
    def validate_aggregate(self):
        if [item.task_slot for item in self.task_results] != [1, 2, 3]:
            raise ValueError("task_results must contain ordered slots 1, 2, 3")
        completed = [item for item in self.task_results if item.status == "completed"]
        expected_status = (
            "completed" if len(completed) == 3 else "degraded" if completed else "insufficient_input"
        )
        if self.status != expected_status:
            raise ValueError("session status is inconsistent with task results")
        if not completed:
            if self.summary_result is not None:
                raise ValueError("insufficient session must not contain a summary")
            return self
        chosen = sorted(completed, key=lambda item: (-item.quality_score, item.task_slot))[0]
        if self.summary_result is None or self.summary_result.source_task_slot != chosen.task_slot:
            raise ValueError("summary must select the highest-quality task")
        expected_summary = {
            "source_task_slot": chosen.task_slot,
            "prediction": chosen.prediction,
            "probabilities": chosen.probabilities.model_dump(),
            "confidence": chosen.confidence,
            "quality_score": chosen.quality_score,
        }
        if self.summary_result.model_dump() != expected_summary:
            raise ValueError("summary must match the selected task result")
        return self


def video_validation_error_response(exception: RequestValidationError) -> dict[str, Any]:
    request_id = "unknown"
    try:
        body = exception.body if isinstance(exception.body, dict) else {}
        candidate = body.get("request_id")
        if (
            isinstance(candidate, str)
            and candidate
            and candidate.strip() == candidate
            and not any(ord(char) < 33 or ord(char) == 127 for char in candidate)
        ):
            request_id = candidate[:128]
    except Exception:
        pass
    return {
        "detail": {
            "code": "INVALID_REQUEST",
            "message": "facial-affect video request validation failed",
            "request_id": request_id,
            "errors": [
                {"location": list(error.get("loc", ())), "message": error.get("msg", "invalid")}
                for error in exception.errors()
            ],
        }
    }


def create_facial_affect_video_router(
    *, runtime: FacialAffectVideoRuntimeProtocol, api_token: str
) -> APIRouter:
    router = APIRouter(tags=["Facial affect clue"])
    bearer = HTTPBearer(auto_error=False)

    def require_token(
        request: Request,
        credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
    ) -> None:
        supplied = credentials.credentials if credentials is not None else ""
        valid_scheme = credentials is not None and credentials.scheme.lower() == "bearer"
        if not valid_scheme or not hmac.compare_digest(supplied, api_token):
            request_id = request.headers.get("x-request-id", "unknown")[:128]
            if (
                not request_id
                or request_id.strip() != request_id
                or any(ord(char) < 33 or ord(char) == 127 for char in request_id)
            ):
                request_id = "unknown"
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "code": "AUTHENTICATION_FAILED",
                    "message": "missing or invalid authorization",
                    "request_id": request_id,
                    "errors": [],
                },
            )

    @router.post(
        FACIAL_AFFECT_VIDEO_INFER_PATH,
        response_model=FacialAffectVideoInferResponse,
        dependencies=[Depends(require_token)],
        responses={401: {}, 413: {}, 422: {}, 500: {}, 503: {}, 504: {}},
    )
    def infer_video(
        payload: FacialAffectVideoInferRequest,
    ) -> FacialAffectVideoInferResponse:
        try:
            result = runtime.infer(payload.model_dump(mode="python"))
            return FacialAffectVideoInferResponse.model_validate(result)
        except (B0DeploymentError, FacialAffectVideoError) as exc:
            raise HTTPException(
                status_code=exc.status_code,
                detail={
                    "code": exc.code,
                    "message": str(exc),
                    "request_id": payload.request_id,
                    "errors": [],
                },
            ) from exc

    return router


__all__ = [
    "FacialAffectTaskVideo",
    "FacialAffectVideoInferRequest",
    "FacialAffectVideoInferResponse",
    "FacialAffectVideoRuntimeProtocol",
    "create_facial_affect_video_router",
    "video_validation_error_response",
]
