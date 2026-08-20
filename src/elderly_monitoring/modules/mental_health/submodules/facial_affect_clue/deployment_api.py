from __future__ import annotations

from datetime import datetime
import hmac
from typing import Any, Literal, Protocol

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .deployment import (
    B0_DEPLOYMENT_MODEL_VERSION,
    B0_ENSEMBLE_RULE,
    B0_EXPECTED_RUN_LOCK_SHA256,
    B0DeploymentError,
)


FACIAL_AFFECT_INFER_PATH = "/v1/mental-health/facial-affect/infer"


class FacialAffectRuntimeProtocol(Protocol):
    def verify_package(self) -> dict[str, Any]: ...

    def predict(self, routes: Any, *, input_sha256: str | None = None) -> dict[str, Any]: ...


class FacialAffectInferRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    schema_version: Literal["facial_affect_b0_request_v1"] = (
        "facial_affect_b0_request_v1"
    )
    request_id: str = Field(min_length=1, max_length=128)
    person_id: str = Field(min_length=1, max_length=128)
    observed_at: datetime
    routes: list[list[list[list[float]]]]
    input_sha256: str | None = Field(default=None, pattern=r"^[0-9a-fA-F]{64}$")

    @field_validator("observed_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("observed_at must include a timezone offset")
        return value


class FacialAffectProbabilities(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    negative: float = Field(ge=0, le=1)
    positive: float = Field(ge=0, le=1)
    surprise: float = Field(ge=0, le=1)


class FacialAffectModelLineage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidate_id: Literal["B0"] = "B0"
    run_lock_sha256: str
    package_sha256: str
    checkpoint_count: Literal[20] = 20
    ensemble_rule: Literal[
        "mean_of_fold_temperature_calibrated_probabilities"
    ] = B0_ENSEMBLE_RULE
    evaluation_scope: Literal["CASME II exposed development estimate"] = (
        "CASME II exposed development estimate"
    )
    deployment_ensemble_independently_validated: Literal[False] = False


class FacialAffectInferResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    schema_version: Literal["facial_affect_b0_response_v1"] = (
        "facial_affect_b0_response_v1"
    )
    status: Literal["completed"] = "completed"
    request_id: str
    person_id: str
    observed_at: datetime
    model_version: Literal[
        "facial-affect-causalnet-b0-opt-me-008-deploy-v1"
    ] = B0_DEPLOYMENT_MODEL_VERSION
    prediction: Literal["negative", "positive", "surprise"]
    probabilities: FacialAffectProbabilities
    confidence: float = Field(ge=0, le=1)
    input_sha256: str
    quality_status: Literal["valid"] = "valid"
    diagnosis: Literal[False] = False
    automatic_risk_level_mapping: Literal[False] = False
    medical_disclaimer: str = "仅作微表情情绪变化线索，不构成医学诊断。"
    warnings: list[str]
    model_lineage: FacialAffectModelLineage


def create_facial_affect_router(
    *, runtime: FacialAffectRuntimeProtocol, api_token: str
) -> APIRouter:
    router = APIRouter(tags=["Facial affect clue"])
    bearer = HTTPBearer(auto_error=False)

    def require_token(
        credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
    ) -> None:
        if (
            credentials is None
            or credentials.scheme.lower() != "bearer"
            or not hmac.compare_digest(credentials.credentials, api_token)
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="missing or invalid authorization",
            )

    @router.post(
        FACIAL_AFFECT_INFER_PATH,
        response_model=FacialAffectInferResponse,
        dependencies=[Depends(require_token)],
        responses={422: {}, 500: {}, 503: {}},
    )
    def infer_facial_affect(
        payload: FacialAffectInferRequest,
    ) -> FacialAffectInferResponse:
        try:
            result = runtime.predict(
                payload.routes,
                input_sha256=payload.input_sha256,
            )
        except B0DeploymentError as exc:
            raise HTTPException(
                status_code=exc.status_code,
                detail={"code": exc.code, "message": str(exc), "errors": []},
            ) from exc
        return FacialAffectInferResponse(
            request_id=payload.request_id,
            person_id=payload.person_id,
            observed_at=payload.observed_at,
            prediction=result["prediction"],
            probabilities=FacialAffectProbabilities(**result["probabilities"]),
            confidence=result["confidence"],
            input_sha256=result["input_sha256"],
            warnings=[
                "部署集成未在独立未见数据上验证，不能把开发集指标视为线上指标。"
            ],
            model_lineage=FacialAffectModelLineage(
                run_lock_sha256=B0_EXPECTED_RUN_LOCK_SHA256,
                package_sha256=result["package_sha256"],
                checkpoint_count=result["checkpoint_count"],
                ensemble_rule=result["ensemble_rule"],
            ),
        )

    return router


__all__ = [
    "FACIAL_AFFECT_INFER_PATH",
    "FacialAffectInferRequest",
    "FacialAffectInferResponse",
    "FacialAffectRuntimeProtocol",
    "create_facial_affect_router",
]
