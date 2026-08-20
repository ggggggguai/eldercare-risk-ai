"""HTTP endpoint for the local cognitive/wandering fusion candidate."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .policy import CognitiveWanderingFusionError, fuse_cognitive_wandering_attention
from .schemas import (
    COGNITIVE_WANDERING_FUSION_PATH,
    CognitiveWanderingFusionRequest,
    CognitiveWanderingFusionResponse,
)


def create_cognitive_wandering_fusion_router(*, api_token: str) -> APIRouter:
    router = APIRouter()
    bearer = HTTPBearer(auto_error=False)

    def require_token(
        credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
    ) -> None:
        if (
            credentials is None
            or credentials.scheme.lower() != "bearer"
            or credentials.credentials != api_token
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"code": "AUTHENTICATION_FAILED"},
            )

    @router.post(
        COGNITIVE_WANDERING_FUSION_PATH,
        response_model=CognitiveWanderingFusionResponse,
        dependencies=[Depends(require_token)],
        responses={
            401: {"description": "Authentication failed"},
            422: {"description": "Fusion contract or evidence failed validation"},
        },
    )
    def infer_cognitive_wandering_fusion(
        payload: CognitiveWanderingFusionRequest,
    ) -> CognitiveWanderingFusionResponse:
        try:
            return fuse_cognitive_wandering_attention(payload)
        except CognitiveWanderingFusionError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"code": "FUSION_EVIDENCE_INVALID", "message": str(exc)},
            ) from exc

    return router


__all__ = [
    "COGNITIVE_WANDERING_FUSION_PATH",
    "create_cognitive_wandering_fusion_router",
]

