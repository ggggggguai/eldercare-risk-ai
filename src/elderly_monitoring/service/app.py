from __future__ import annotations

from functools import partial
import shutil
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from elderly_monitoring.service.schemas import SessionAccepted, SessionStatusResponse, StartSessionRequest, StreamUrlUpdate
from elderly_monitoring.service.session import SessionManager, SessionStatus
from elderly_monitoring.service.settings import ServiceSettings
from elderly_monitoring.service.stream_reader import FFmpegStreamReader, StreamReader


def reader_factory_for(settings: ServiceSettings) -> Any:
    if settings.stream_reader_backend == "opencv":
        return StreamReader
    return partial(
        FFmpegStreamReader,
        scale_width=settings.ffmpeg_scale_width,
        max_fps=settings.max_inference_fps,
    )


def create_app(*, settings: ServiceSettings | None = None, session_manager: SessionManager | None = None) -> FastAPI:
    service_settings = settings or ServiceSettings.load()
    manager = session_manager or SessionManager(
        reader_factory=reader_factory_for(service_settings),
        model_path=str(service_settings.model_path),
        gait_model_path=service_settings.gait_model_path,
        gait_model_device=service_settings.gait_model_device,
        gait_model_window_frames=service_settings.gait_model_window_frames,
        reconnect_attempts=service_settings.reconnect_attempts,
        reconnect_delay_sec=service_settings.reconnect_delay_sec,
        reconnect_stable_after_sec=service_settings.reconnect_stable_after_sec,
        reconnect_stable_after_frames=service_settings.reconnect_stable_after_frames,
        callback_token=service_settings.callback_token,
        scene_risk_scores=service_settings.scene_risk_scores,
        branch_quality=service_settings.branch_quality,
        baseline_history_path=service_settings.baseline_history_path,
        pose_window_sec=service_settings.pose_window_sec,
        analysis_interval_sec=service_settings.analysis_interval_sec,
        fusion_interval_sec=service_settings.fusion_interval_sec,
        primary_lost_timeout_sec=service_settings.primary_lost_timeout_sec,
        event_cooldown_sec=service_settings.event_cooldown_sec,
        callback_timeout_sec=service_settings.callback_timeout_sec,
        callback_retry_delays_sec=service_settings.callback_retry_delays_sec,
        outbox_capacity=service_settings.outbox_capacity,
        outbox_drain_timeout_sec=service_settings.outbox_drain_timeout_sec,
        stop_budget_sec=service_settings.session_stop_timeout_sec,
        frame_queue_capacity=service_settings.frame_queue_capacity,
        open_timeout_ms=service_settings.stream_open_timeout_ms,
        read_timeout_ms=service_settings.stream_read_timeout_ms,
        max_inference_fps=service_settings.max_inference_fps,
        pose_inference_size=service_settings.pose_inference_size,
        fall_state=service_settings.fall_state,
    )
    app = FastAPI(title="Elderly Monitoring Fall Risk Service", version="0.2.0")
    bearer = HTTPBearer(auto_error=False)

    def require_token(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)) -> None:
        if credentials is None or credentials.scheme.lower() != "bearer" or credentials.credentials != service_settings.api_token:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="missing or invalid authorization")

    @app.exception_handler(Exception)
    async def internal_error(_: Request, __: Exception) -> JSONResponse:
        return JSONResponse(status_code=500, content={"detail": "internal server error"})

    @app.get("/health/live")
    def live() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ready")
    def ready() -> dict[str, str]:
        if not service_settings.model_path.exists():
            raise HTTPException(status_code=503, detail="model is not available")
        if (
            service_settings.gait_model_path is not None
            and not service_settings.gait_model_path.exists()
        ):
            raise HTTPException(status_code=503, detail="gait model is not available")
        if service_settings.stream_reader_backend == "ffmpeg" and (
            shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None
        ):
            raise HTTPException(
                status_code=503, detail="ffmpeg backend is not available"
            )
        return {"status": "ready"}

    @app.post("/v1/monitoring/sessions", response_model=SessionAccepted, status_code=202, dependencies=[Depends(require_token)])
    def start(request: StartSessionRequest) -> SessionAccepted:
        try:
            session = manager.start(**request.model_dump())
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return SessionAccepted(session_id=session.session_id, status=session.status.value)

    @app.get("/v1/monitoring/sessions/{session_id}", response_model=SessionStatusResponse, dependencies=[Depends(require_token)])
    def get_status(session_id: str) -> SessionStatusResponse:
        session = manager.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="session not found")
        return SessionStatusResponse(
            session_id=session.session_id, status=session.status.value, device_id=session.device_id,
            person_id=session.person_id, started_at=session.started_at, last_frame_at=session.last_frame_at,
            last_error=session.last_error,
            stream_epoch=int(getattr(session, "stream_epoch", 0)),
            frame_diagnostics=dict(getattr(session, "frame_diagnostics", {})),
            runtime_diagnostics=dict(getattr(session, "runtime_diagnostics", {})),
        )

    @app.put("/v1/monitoring/sessions/{session_id}/stream-url", response_model=SessionAccepted, dependencies=[Depends(require_token)])
    def update(session_id: str, request: StreamUrlUpdate) -> SessionAccepted:
        session = manager.update_url(session_id, request.stream_url)
        if session is None:
            raise HTTPException(status_code=404, detail="session not found")
        return SessionAccepted(session_id=session.session_id, status=session.status.value)

    @app.post("/v1/monitoring/sessions/{session_id}/stop", response_model=SessionAccepted, status_code=202, dependencies=[Depends(require_token)])
    def stop(session_id: str) -> SessionAccepted:
        session = manager.stop(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="session not found")
        return SessionAccepted(session_id=session.session_id, status=session.status.value)

    app.state.settings = service_settings
    app.state.session_manager = manager
    return app


app = create_app()
