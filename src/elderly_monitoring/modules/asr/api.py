"""V3.3 Python entry and lightweight standalone ASR HTTP service."""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager, suppress

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from elderly_monitoring.modules.asr.engine import get_default_engine
from elderly_monitoring.modules.asr.facade import ASREngine, transcribe as _transcribe
from elderly_monitoring.modules.asr.runtime_integrity import verify_native_assets
from elderly_monitoring.modules.asr.schemas import ASRRequest, ASRTranscript
from elderly_monitoring.modules.asr.settings import ASRSettings


logger = logging.getLogger(__name__)


def transcribe(
    request: ASRRequest,
    *,
    engine: ASREngine | None = None,
    settings: ASRSettings | None = None,
) -> ASRTranscript:
    """Stable same-process entry named by the cognitive V3.3 design."""

    return _transcribe(request, engine=engine, settings=settings)


def create_asr_app(
    *,
    settings: ASRSettings | None = None,
    engine: ASREngine | None = None,
    api_token: str | None = None,
    prewarm: bool = True,
) -> FastAPI:
    runtime_settings = settings or ASRSettings.load()
    runtime_engine = engine or get_default_engine(runtime_settings)
    token = api_token or os.environ.get("ALGORITHM_API_TOKEN", "change-me")
    bearer = HTTPBearer(auto_error=False)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        verify_native_assets(runtime_settings)
        if prewarm:
            async def warmup() -> None:
                try:
                    await asyncio.to_thread(runtime_engine.warmup)
                except Exception as exc:
                    app.state.asr_warmup_error = exc
                    logger.exception("ASR model warmup failed")
                else:
                    logger.info("ASR model warmup completed")

            app.state.asr_warmup_task = asyncio.create_task(
                warmup(),
                name="asr-model-warmup",
            )
        yield
        warmup_task = app.state.asr_warmup_task
        if warmup_task is not None:
            warmup_task.cancel()
            with suppress(asyncio.CancelledError):
                await warmup_task

    app = FastAPI(
        title="Elderly Monitoring ASR Service",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.asr_warmup_error = None
    app.state.asr_warmup_task = None

    def require_token(
        credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
    ) -> None:
        if (
            credentials is None
            or credentials.scheme.lower() != "bearer"
            or credentials.credentials != token
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="missing or invalid authorization",
            )

    @app.get("/health/live")
    def live() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ready")
    def ready() -> dict[str, str]:
        if app.state.asr_warmup_error is not None:
            raise HTTPException(status_code=503, detail="ASR model warmup failed")
        if not bool(getattr(runtime_engine, "is_ready", False)):
            raise HTTPException(status_code=503, detail="ASR model is not ready")
        return {"status": "ready"}

    @app.post(
        "/v1/asr/transcribe",
        response_model=ASRTranscript,
        dependencies=[Depends(require_token)],
    )
    def transcribe_route(request: ASRRequest) -> ASRTranscript:
        if app.state.asr_warmup_error is not None:
            raise HTTPException(status_code=503, detail="ASR model warmup failed")
        if not bool(getattr(runtime_engine, "is_ready", False)):
            raise HTTPException(status_code=503, detail="ASR model is not ready")
        return transcribe(request, engine=runtime_engine, settings=runtime_settings)

    app.state.asr_engine = runtime_engine
    app.state.asr_settings = runtime_settings
    return app


app = create_asr_app()


__all__ = ["app", "create_asr_app", "transcribe"]
