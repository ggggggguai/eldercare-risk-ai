"""FastAPI router for subject-level cognitive V3.5 inference."""

from __future__ import annotations

import asyncio
from typing import Protocol

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.errors import (
    CognitiveAPIError,
)
from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.schemas import (
    CognitiveErrorResponse,
)
from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.v35_schemas import (
    CognitiveV35InferRequest,
    CognitiveV35InferResponse,
)


COGNITIVE_V35_INFER_PATH = "/v1/mental-health/cognitive-change/subject-infer"
COGNITIVE_V35_TIMEOUT_SECONDS = 180.0


class CognitiveV35RuntimeProtocol(Protocol):
    def infer(self, request: CognitiveV35InferRequest) -> CognitiveV35InferResponse: ...

    def verify_package(self) -> None: ...

    def close(self) -> None: ...


def create_cognitive_v35_router(
    *,
    runtime: CognitiveV35RuntimeProtocol,
    api_token: str,
) -> APIRouter:
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
            raise CognitiveAPIError(code="AUTHENTICATION_FAILED")

    @router.post(
        COGNITIVE_V35_INFER_PATH,
        response_model=CognitiveV35InferResponse,
        dependencies=[Depends(require_token)],
        responses={
            401: {"model": CognitiveErrorResponse},
            408: {"model": CognitiveErrorResponse},
            422: {"model": CognitiveErrorResponse},
            500: {"model": CognitiveErrorResponse},
            503: {"model": CognitiveErrorResponse},
            504: {"model": CognitiveErrorResponse},
        },
    )
    async def infer_cognitive_v35(
        payload: CognitiveV35InferRequest,
        http_request: Request,
    ) -> CognitiveV35InferResponse:
        http_request.state.cognitive_request_id = payload.request_id
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(runtime.infer, payload),
                timeout=COGNITIVE_V35_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError as exc:
            raise CognitiveAPIError(
                code="ALGORITHM_TIMEOUT",
                request_id=payload.request_id,
            ) from exc
        except CognitiveAPIError:
            raise
        except HTTPException:
            raise
        except Exception as exc:
            raise CognitiveAPIError(
                code="INTERNAL_ERROR",
                request_id=payload.request_id,
            ) from exc

    return router


__all__ = [
    "COGNITIVE_V35_INFER_PATH",
    "COGNITIVE_V35_TIMEOUT_SECONDS",
    "CognitiveV35RuntimeProtocol",
    "create_cognitive_v35_router",
]
