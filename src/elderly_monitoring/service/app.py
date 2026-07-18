from __future__ import annotations

from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from elderly_monitoring.modules.roi_annotation.ezviz_client import EzvizVisionModelClient
from elderly_monitoring.modules.roi_annotation.service import RoiAnnotationError, annotate_roi_image
from elderly_monitoring.service.daytime_activity import build_daytime_activity_result
from elderly_monitoring.service.schemas import (
    DaytimeActivityRequest,
    DaytimeActivityResponse,
    NightPhysiologyRequest,
    NightPhysiologyResponse,
    MentalHealthDailyRiskRequest,
    MentalHealthDailyRiskResponse,
    RoiAnnotateRequest,
    RoiAnnotateResponse,
    SessionAccepted,
    SessionStatusResponse,
    SleepRhythmRequest,
    SleepRhythmResponse,
    SocialConnectionRequest,
    SocialConnectionResponse,
    StartSessionRequest,
    StreamUrlUpdate,
)
from elderly_monitoring.service.session import SessionManager, SessionStatus
from elderly_monitoring.service.settings import ServiceSettings
from elderly_monitoring.service.mental_health_daily import build_mental_health_daily_risk_result
from elderly_monitoring.service.night_physiology import build_night_physiology_service_result
from elderly_monitoring.service.sleep_rhythm import build_sleep_rhythm_service_result
from elderly_monitoring.service.social_connection import build_social_connection_service_result


def create_app(*, settings: ServiceSettings | None = None, session_manager: SessionManager | None = None) -> FastAPI:
    service_settings = settings or ServiceSettings.load()
    manager = session_manager or SessionManager(
        model_path=str(service_settings.model_path),
        reconnect_attempts=service_settings.reconnect_attempts,
        reconnect_delay_sec=service_settings.reconnect_delay_sec,
        callback_token=service_settings.callback_token,
        scene_risk_scores=service_settings.scene_risk_scores,
        baseline_history_path=service_settings.baseline_history_path,
        pose_window_sec=service_settings.pose_window_sec,
        analysis_interval_sec=service_settings.analysis_interval_sec,
        fusion_interval_sec=service_settings.fusion_interval_sec,
        event_cooldown_sec=service_settings.event_cooldown_sec,
        callback_timeout_sec=service_settings.callback_timeout_sec,
        callback_retry_delays_sec=service_settings.callback_retry_delays_sec,
        open_timeout_ms=service_settings.stream_open_timeout_ms,
        read_timeout_ms=service_settings.stream_read_timeout_ms,
        max_inference_fps=service_settings.max_inference_fps,
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

    @app.post("/v1/roi/annotate", response_model=RoiAnnotateResponse, dependencies=[Depends(require_token)])
    def annotate_roi(request: RoiAnnotateRequest) -> RoiAnnotateResponse:
        client = EzvizVisionModelClient(
            api_key=service_settings.ezviz_llm_api_key,
            base_url=service_settings.ezviz_llm_base_url,
            model=service_settings.ezviz_llm_model,
            timeout_seconds=service_settings.ezviz_llm_timeout_sec,
        )
        try:
            result = annotate_roi_image(
                image_base64=request.image_base64,
                mime_type=request.mime_type,
                image_width=request.image_width,
                image_height=request.image_height,
                scene_hint=request.scene_hint,
                expected_types=list(request.expected_types),
                client=client,
            )
        except RoiAnnotationError as exc:
            raise HTTPException(status_code=exc.status_code, detail={"category": exc.category, "message": str(exc)}) from exc
        return RoiAnnotateResponse(**result)

    @app.post(
        "/v1/mental-health/daytime-activity",
        response_model=DaytimeActivityResponse,
        dependencies=[Depends(require_token)],
    )
    def extract_daytime_activity(request: DaytimeActivityRequest) -> DaytimeActivityResponse:
        try:
            result = build_daytime_activity_result(request)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return DaytimeActivityResponse(**result)

    @app.post(
        "/v1/mental-health/sleep-rhythm",
        response_model=SleepRhythmResponse,
        dependencies=[Depends(require_token)],
    )
    def extract_sleep_rhythm(request: SleepRhythmRequest) -> SleepRhythmResponse:
        try:
            result = build_sleep_rhythm_service_result(request)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return SleepRhythmResponse(**result)

    @app.post(
        "/v1/mental-health/night-physiology",
        response_model=NightPhysiologyResponse,
        dependencies=[Depends(require_token)],
    )
    def extract_night_physiology(request: NightPhysiologyRequest) -> NightPhysiologyResponse:
        try:
            result = build_night_physiology_service_result(request)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return NightPhysiologyResponse(**result)

    @app.post(
        "/v1/mental-health/social-connection",
        response_model=SocialConnectionResponse,
        dependencies=[Depends(require_token)],
    )
    def extract_social_connection(request: SocialConnectionRequest) -> SocialConnectionResponse:
        try:
            result = build_social_connection_service_result(request)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return SocialConnectionResponse(**result)

    @app.post(
        "/v1/mental-health/daily-risk",
        response_model=MentalHealthDailyRiskResponse,
        dependencies=[Depends(require_token)],
    )
    def score_mental_health_daily(request: MentalHealthDailyRiskRequest) -> MentalHealthDailyRiskResponse:
        try:
            result = build_mental_health_daily_risk_result(request)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return MentalHealthDailyRiskResponse(**result)

    app.state.settings = service_settings
    app.state.session_manager = manager
    return app


app = create_app()
