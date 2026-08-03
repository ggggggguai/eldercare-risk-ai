from __future__ import annotations

from collections.abc import Callable

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from elderly_monitoring.modules.mental_health.mood_social import (
    MOOD_SOCIAL_INFER_PATH,
    MoodSocialAPIError,
    MoodSocialErrorResponse,
    MoodSocialInferRequest,
    MoodSocialInferResponse,
    infer_mood_social,
    validation_error_response,
)
from elderly_monitoring.modules.asr import ASRRequest, ASRTranscript
from elderly_monitoring.modules.asr.api import transcribe as transcribe_asr
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
from elderly_monitoring.service.session import SessionManager
from elderly_monitoring.service.settings import ServiceSettings
from elderly_monitoring.service.night_physiology import build_night_physiology_service_result
from elderly_monitoring.service.sleep_rhythm import build_sleep_rhythm_service_result
from elderly_monitoring.service.social_connection import build_social_connection_service_result


def create_app(
    *,
    settings: ServiceSettings | None = None,
    session_manager: SessionManager | None = None,
    asr_transcriber: Callable[[ASRRequest], ASRTranscript] | None = None,
) -> FastAPI:
    service_settings = settings or ServiceSettings.load()
    manager = session_manager or SessionManager(
        model_path=str(service_settings.model_path),
        gait_model_path=service_settings.gait_model_path,
        gait_model_device=service_settings.gait_model_device,
        gait_model_window_frames=service_settings.gait_model_window_frames,
        reconnect_attempts=service_settings.reconnect_attempts,
        reconnect_delay_sec=service_settings.reconnect_delay_sec,
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
        fall_state=service_settings.fall_state,
    )
    app = FastAPI(title="Elderly Monitoring Fall Risk Service", version="0.2.0")
    bearer = HTTPBearer(auto_error=False)
    run_asr = asr_transcriber or transcribe_asr

    def require_token(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)) -> None:
        if credentials is None or credentials.scheme.lower() != "bearer" or credentials.credentials != service_settings.api_token:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="missing or invalid authorization")

    def require_mood_social_token(
        credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
    ) -> None:
        if (
            credentials is None
            or credentials.scheme.lower() != "bearer"
            or credentials.credentials != service_settings.api_token
        ):
            raise MoodSocialAPIError(code="AUTHENTICATION_FAILED")

    @app.exception_handler(MoodSocialAPIError)
    async def mood_social_api_error(
        _: Request,
        exception: MoodSocialAPIError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exception.status_code,
            content=exception.as_response().model_dump(mode="json"),
        )

    @app.exception_handler(RequestValidationError)
    async def request_validation_error(
        request: Request,
        exception: RequestValidationError,
    ) -> JSONResponse:
        if request.url.path != MOOD_SOCIAL_INFER_PATH:
            return await request_validation_exception_handler(request, exception)
        response = validation_error_response(exception)
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=response.model_dump(mode="json"),
        )

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
        "/v1/asr/transcribe",
        response_model=ASRTranscript,
        dependencies=[Depends(require_token)],
    )
    def transcribe_audio(request: ASRRequest) -> ASRTranscript:
        return run_asr(request)

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
        from elderly_monitoring.service.mental_health_daily import (
            build_mental_health_daily_risk_result,
        )

        try:
            result = build_mental_health_daily_risk_result(request)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return MentalHealthDailyRiskResponse(**result)

    @app.post(
        MOOD_SOCIAL_INFER_PATH,
        response_model=MoodSocialInferResponse,
        dependencies=[Depends(require_mood_social_token)],
        responses={
            401: {"model": MoodSocialErrorResponse},
            422: {"model": MoodSocialErrorResponse},
            500: {"model": MoodSocialErrorResponse},
            503: {"model": MoodSocialErrorResponse},
        },
    )
    def infer_mood_social_attention(
        payload: MoodSocialInferRequest,
        http_request: Request,
    ) -> MoodSocialInferResponse:
        http_request.state.mood_social_request_id = payload.request_id
        try:
            return infer_mood_social(payload)
        except MoodSocialAPIError:
            raise
        except Exception as exc:
            raise MoodSocialAPIError(
                code="INTERNAL_ERROR",
                request_id=payload.request_id,
            ) from exc

    app.state.settings = service_settings
    app.state.session_manager = manager
    app.state.asr_transcriber = run_asr
    return app


app = create_app()
