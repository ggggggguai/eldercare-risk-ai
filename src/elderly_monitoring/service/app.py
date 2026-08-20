from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import Callable
import json

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
from elderly_monitoring.modules.mental_health.mood_social.r5_release import (
    MOOD_SOCIAL_R5_CANDIDATE_INFER_PATH,
    R5CandidateResponse,
    R5PackageError,
    infer_mood_social_r5_candidate,
)
from elderly_monitoring.modules.mental_health.mood_social.r6_release import (
    MOOD_SOCIAL_R6_CANDIDATE_INFER_PATH,
    R6CandidateResponse,
    R6PackageError,
    infer_mood_social_r6_candidate,
)
from elderly_monitoring.modules.mental_health.mood_social.r7.release import (
    MOOD_SOCIAL_R7_CANDIDATE_INFER_PATH,
    R7CandidateResponse,
    R7PackageError,
    infer_mood_social_r7_candidate,
)
from elderly_monitoring.modules.mental_health.mood_social.r8.release import (
    MOOD_SOCIAL_R8_CANDIDATE_INFER_PATH,
    R8CandidateRequest,
    R8CandidateResponse,
    R8PackageError,
    infer_mood_social_r8_candidate,
)
from elderly_monitoring.modules.mental_health.mood_social.r9.release import (
    MOOD_SOCIAL_R9_CANDIDATE_INFER_PATH,
    R9CandidateRequest,
    R9CandidateResponse,
    R9PackageError,
    infer_mood_social_r9_candidate,
)
from elderly_monitoring.modules.mental_health.mood_social.r10.release import (
    MOOD_SOCIAL_R10_CANDIDATE_INFER_PATH,
    R10CandidateRequest,
    R10CandidateResponse,
    R10PackageError,
    infer_mood_social_r10_candidate,
)
from elderly_monitoring.modules.mental_health.mood_social.r11.release import (
    MOOD_SOCIAL_R11_CANDIDATE_INFER_PATH,
    R11PackageError,
    infer_mood_social_r11_candidate,
)
from elderly_monitoring.modules.mental_health.mood_social.r11.production import (
    MOOD_SOCIAL_R11_PRODUCTION_INFER_PATH,
    infer_mood_social_r11_production,
)
from elderly_monitoring.modules.mental_health.mood_social.r11.schemas import (
    R11CandidateRequest,
    R11CandidateResponse,
)
from elderly_monitoring.modules.mental_health.mood_social.forecast_api import (
    MOOD_SOCIAL_FORECAST_INFER_PATH,
    infer_mood_social_forecast,
)
from elderly_monitoring.modules.mental_health.mood_social.schemas import (
    MoodSocialForecastResult,
)
from elderly_monitoring.modules.asr import ASRRequest, ASRTranscript
from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.api import (
    COGNITIVE_INFER_PATH,
    CognitiveRuntimeProtocol,
    create_cognitive_router,
)
from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.asr_client import (
    ASRHttpClient,
)
from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.errors import (
    CognitiveAPIError,
    validation_error_response as cognitive_validation_error_response,
)
from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.inference import (
    CognitiveInferenceRuntime,
)
from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.manifest import (
    CognitiveModelPackage,
)
from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.schemas import (
    DEFAULT_MODEL_VERSION,
    V34_MODEL_VERSION,
)
from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.v35_api import (
    COGNITIVE_V35_INFER_PATH,
    CognitiveV35RuntimeProtocol,
    create_cognitive_v35_router,
)
from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.v35_inference import (
    CognitiveV35InferenceRuntime,
)
from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.v35_package import (
    CognitiveV35ModelPackage,
)
from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.v35_schemas import (
    V35_MODEL_VERSION,
    V35_REQUEST_SCHEMA_VERSION,
)
from elderly_monitoring.modules.mental_health.submodules.cognitive_wandering_fusion import (
    create_cognitive_wandering_fusion_router,
)
from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.deployment import (
    B0DeploymentError,
    B0DeploymentRuntime,
)
from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.deployment_api import (
    FacialAffectRuntimeProtocol,
    create_facial_affect_router,
)
from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.video_api import (
    FacialAffectVideoRuntimeProtocol,
    create_facial_affect_video_router,
    video_validation_error_response,
)
from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.video_pipeline import (
    FACIAL_AFFECT_VIDEO_INFER_PATH,
    DeterministicMotionStrainSpotter,
    FacialAffectVideoRuntime,
    SpottingConfig,
    VideoPipelineConfig,
)
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
    cognitive_runtime: CognitiveRuntimeProtocol | None = None,
    cognitive_v35_runtime: CognitiveV35RuntimeProtocol | None = None,
    facial_affect_runtime: FacialAffectRuntimeProtocol | None = None,
    facial_affect_video_runtime: FacialAffectVideoRuntimeProtocol | None = None,
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
    bearer = HTTPBearer(auto_error=False)
    run_asr = asr_transcriber
    v33_package = CognitiveModelPackage(
        service_settings.cognitive_model_package_path,
        device=service_settings.cognitive_inference_device,
        expected_model_version=DEFAULT_MODEL_VERSION,
    )
    run_cognitive = cognitive_runtime or CognitiveInferenceRuntime(
        packages={
            DEFAULT_MODEL_VERSION: v33_package,
            V34_MODEL_VERSION: CognitiveModelPackage(
                service_settings.cognitive_model_package_v34_path,
                device=service_settings.cognitive_inference_device,
                expected_model_version=V34_MODEL_VERSION,
            ),
        },
        asr_client=ASRHttpClient(
            url=service_settings.cognitive_asr_url,
            api_token=service_settings.api_token,
        ),
    )
    run_cognitive_v35 = cognitive_v35_runtime or CognitiveV35InferenceRuntime(
        package=CognitiveV35ModelPackage(
            service_settings.cognitive_model_package_v35_path,
            encoder_package=v33_package,
            device=service_settings.cognitive_inference_device,
        ),
        asr_client=ASRHttpClient(
            url=service_settings.cognitive_asr_url,
            api_token=service_settings.api_token,
        ),
    )
    run_facial_affect = facial_affect_runtime or B0DeploymentRuntime(
        service_settings.facial_affect_b0_package_path,
        device=service_settings.facial_affect_inference_device,
    )
    try:
        spotting_payload = json.loads(
            service_settings.facial_affect_spotting_config_path.read_text(
                encoding="utf-8"
            )
        )
        spotting_config = SpottingConfig(**spotting_payload)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise RuntimeError("frozen facial-affect spotting config is unavailable") from exc
    run_facial_affect_video = (
        facial_affect_video_runtime
        or FacialAffectVideoRuntime(
            b0_runtime=run_facial_affect,
            config=VideoPipelineConfig(
                max_bytes=service_settings.facial_affect_video_max_bytes,
                download_timeout_seconds=(
                    service_settings.facial_affect_video_download_timeout_seconds
                ),
                allowed_origins=(
                    service_settings.facial_affect_video_url_allowed_origins
                ),
                temp_root=service_settings.facial_affect_video_temp_root,
                temp_retention_seconds=(
                    service_settings.facial_affect_video_temp_retention_seconds
                ),
            ),
            spotter=DeterministicMotionStrainSpotter(spotting_config),
        )
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            run_cognitive.verify_package()
            package_error = None
        except CognitiveAPIError as exc:
            package_error = exc
        app.state.cognitive_package_error = package_error
        try:
            run_cognitive_v35.verify_package()
            package_v35_error = None
        except CognitiveAPIError as exc:
            package_v35_error = exc
        app.state.cognitive_v35_package_error = package_v35_error
        try:
            run_facial_affect.verify_package()
            verify_facial_affect_probe = getattr(
                run_facial_affect, "verify_deterministic_probe", None
            )
            if callable(verify_facial_affect_probe):
                verify_facial_affect_probe()
            facial_affect_package_error = None
        except B0DeploymentError as exc:
            facial_affect_package_error = exc
        app.state.facial_affect_package_error = facial_affect_package_error
        try:
            yield
        finally:
            run_cognitive_v35.close()
            run_cognitive.close()
            close_facial_affect = getattr(run_facial_affect, "close", None)
            if callable(close_facial_affect):
                close_facial_affect()

    app = FastAPI(
        title="Elderly Monitoring Fall Risk Service",
        version="0.2.0",
        lifespan=lifespan,
    )
    app.state.cognitive_package_error = None
    app.state.cognitive_v35_package_error = None
    app.state.facial_affect_package_error = None

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

    @app.exception_handler(CognitiveAPIError)
    async def cognitive_api_error(
        _: Request,
        exception: CognitiveAPIError,
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
        if request.url.path == FACIAL_AFFECT_VIDEO_INFER_PATH:
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content=video_validation_error_response(exception),
            )
        if request.url.path in {COGNITIVE_INFER_PATH, COGNITIVE_V35_INFER_PATH}:
            response = (
                cognitive_validation_error_response(
                    exception,
                    request_schema_version=V35_REQUEST_SCHEMA_VERSION,
                    supported_model_versions=(V35_MODEL_VERSION,),
                )
                if request.url.path == COGNITIVE_V35_INFER_PATH
                else cognitive_validation_error_response(exception)
            )
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content=response.model_dump(mode="json"),
            )
        if request.url.path not in {
            MOOD_SOCIAL_INFER_PATH,
            MOOD_SOCIAL_R5_CANDIDATE_INFER_PATH,
        }:
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
        if app.state.cognitive_v35_package_error is not None:
            raise HTTPException(
                status_code=503,
                detail="final cognitive V3.5 model is not available",
            )
        if app.state.facial_affect_package_error is not None:
            raise HTTPException(
                status_code=503,
                detail="frozen facial-affect B0 package is not available",
            )
        return {
            "status": "ready",
            "facial_affect_b0_package": "ready",
            "facial_affect_b0_deterministic_probe": "passed",
            "facial_affect_spotting_config": "ready",
            "facial_affect_video_pipeline": (
                "ready"
                if service_settings.facial_affect_video_url_allowed_origins
                else "configuration_required"
            ),
        }

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

    if run_asr is not None:
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

    @app.post(
        MOOD_SOCIAL_R5_CANDIDATE_INFER_PATH,
        response_model=R5CandidateResponse,
        dependencies=[Depends(require_mood_social_token)],
        responses={422: {"model": MoodSocialErrorResponse}, 503: {"model": MoodSocialErrorResponse}},
    )
    def infer_mood_social_r5_integration_candidate(
        payload: MoodSocialInferRequest,
        http_request: Request,
    ) -> R5CandidateResponse:
        """Explicit immutable-r5 replay; production uses the audited adapter and fallback."""

        http_request.state.mood_social_request_id = payload.request_id
        try:
            return R5CandidateResponse.model_validate(
                infer_mood_social_r5_candidate(payload)
            )
        except R5PackageError as exc:
            raise MoodSocialAPIError(
                code="MODEL_ARTIFACT_UNAVAILABLE",
                request_id=payload.request_id,
            ) from exc

    @app.post(
        MOOD_SOCIAL_R6_CANDIDATE_INFER_PATH,
        response_model=R6CandidateResponse,
        dependencies=[Depends(require_mood_social_token)],
        responses={422: {"model": MoodSocialErrorResponse}, 503: {"model": MoodSocialErrorResponse}},
    )
    def infer_mood_social_r6_research_candidate(
        payload: MoodSocialInferRequest,
        http_request: Request,
    ) -> R6CandidateResponse:
        """Explicit r6 research replay; it does not affect the r5 production adapter."""

        http_request.state.mood_social_request_id = payload.request_id
        try:
            return R6CandidateResponse.model_validate(infer_mood_social_r6_candidate(payload))
        except R6PackageError as exc:
            raise MoodSocialAPIError(
                code="MODEL_ARTIFACT_UNAVAILABLE",
                request_id=payload.request_id,
            ) from exc

    @app.post(
        MOOD_SOCIAL_R7_CANDIDATE_INFER_PATH,
        response_model=R7CandidateResponse,
        dependencies=[Depends(require_mood_social_token)],
        responses={422: {"model": MoodSocialErrorResponse}, 503: {"model": MoodSocialErrorResponse}},
    )
    def infer_mood_social_r7_modular_candidate(
        payload: MoodSocialInferRequest,
        http_request: Request,
    ) -> R7CandidateResponse:
        """Explicit R7 modular expert replay; production R5/V3.4 stay unchanged."""

        http_request.state.mood_social_request_id = payload.request_id
        try:
            return R7CandidateResponse.model_validate(
                infer_mood_social_r7_candidate(payload)
            )
        except R7PackageError as exc:
            raise MoodSocialAPIError(
                code="MODEL_ARTIFACT_UNAVAILABLE",
                request_id=payload.request_id,
            ) from exc

    @app.post(
        MOOD_SOCIAL_R8_CANDIDATE_INFER_PATH,
        response_model=R8CandidateResponse,
        dependencies=[Depends(require_mood_social_token)],
        responses={422: {"model": MoodSocialErrorResponse}, 503: {"model": MoodSocialErrorResponse}},
    )
    def infer_mood_social_r8_integration_candidate(
        payload: R8CandidateRequest,
        http_request: Request,
    ) -> R8CandidateResponse:
        """Explicit R8 integration replay; R5 production and V3.4 stay unchanged."""

        http_request.state.mood_social_request_id = payload.request_id
        try:
            return R8CandidateResponse.model_validate(
                infer_mood_social_r8_candidate(payload)
            )
        except R8PackageError as exc:
            raise MoodSocialAPIError(
                code="MODEL_ARTIFACT_UNAVAILABLE",
                request_id=payload.request_id,
            ) from exc

    @app.post(
        MOOD_SOCIAL_R9_CANDIDATE_INFER_PATH,
        response_model=R9CandidateResponse,
        dependencies=[Depends(require_mood_social_token)],
        responses={422: {"model": MoodSocialErrorResponse}, 503: {"model": MoodSocialErrorResponse}},
    )
    def infer_mood_social_r9_local_joint_candidate(
        payload: R9CandidateRequest,
        http_request: Request,
    ) -> R9CandidateResponse:
        """Explicit R9 local-joint candidate; R5 production and V3.4 stay unchanged."""

        http_request.state.mood_social_request_id = payload.request_id
        try:
            return R9CandidateResponse.model_validate(
                infer_mood_social_r9_candidate(payload)
            )
        except R9PackageError as exc:
            raise MoodSocialAPIError(
                code="MODEL_ARTIFACT_UNAVAILABLE",
                request_id=payload.request_id,
            ) from exc

    @app.post(
        MOOD_SOCIAL_R10_CANDIDATE_INFER_PATH,
        response_model=R10CandidateResponse,
        dependencies=[Depends(require_mood_social_token)],
        responses={422: {"model": MoodSocialErrorResponse}, 503: {"model": MoodSocialErrorResponse}},
    )
    def infer_mood_social_r10_local_joint_candidate(
        payload: R10CandidateRequest,
        http_request: Request,
    ) -> R10CandidateResponse:
        """Explicit R10 candidate; fail closed when the honest No-Go has no package."""

        http_request.state.mood_social_request_id = payload.request_id
        try:
            return R10CandidateResponse.model_validate(
                infer_mood_social_r10_candidate(payload)
            )
        except R10PackageError as exc:
            raise MoodSocialAPIError(
                code="MODEL_ARTIFACT_UNAVAILABLE",
                request_id=payload.request_id,
            ) from exc

    @app.post(
        MOOD_SOCIAL_R11_CANDIDATE_INFER_PATH,
        response_model=R11CandidateResponse,
        dependencies=[Depends(require_mood_social_token)],
        responses={422: {"model": MoodSocialErrorResponse}, 503: {"model": MoodSocialErrorResponse}},
    )
    def infer_mood_social_r11_engineering_candidate(
        payload: R11CandidateRequest,
        http_request: Request,
    ) -> R11CandidateResponse:
        """Explicit R11 engineering candidate; the R5 production route is unchanged."""

        http_request.state.mood_social_request_id = payload.request_id
        try:
            return R11CandidateResponse.model_validate(
                infer_mood_social_r11_candidate(payload)
            )
        except R11PackageError as exc:
            raise MoodSocialAPIError(
                code="MODEL_ARTIFACT_UNAVAILABLE",
                request_id=payload.request_id,
            ) from exc

    @app.post(
        MOOD_SOCIAL_R11_PRODUCTION_INFER_PATH,
        response_model=R11CandidateResponse,
        dependencies=[Depends(require_mood_social_token)],
        responses={422: {"model": MoodSocialErrorResponse}, 503: {"model": MoodSocialErrorResponse}},
    )
    def infer_mood_social_r11_production_primary(
        payload: R11CandidateRequest,
        http_request: Request,
    ) -> R11CandidateResponse:
        """Production R11 evidence orchestration over frozen R9/R7/R8 nodes."""

        http_request.state.mood_social_request_id = payload.request_id
        try:
            return R11CandidateResponse.model_validate(
                infer_mood_social_r11_production(payload)
            )
        except R11PackageError as exc:
            raise MoodSocialAPIError(
                code="MODEL_ARTIFACT_UNAVAILABLE",
                request_id=payload.request_id,
            ) from exc

    @app.post(
        MOOD_SOCIAL_FORECAST_INFER_PATH,
        response_model=MoodSocialForecastResult,
        dependencies=[Depends(require_mood_social_token)],
        responses={422: {"model": MoodSocialErrorResponse}},
    )
    def infer_mood_social_v34_forecast(
        payload: MoodSocialInferRequest,
        http_request: Request,
    ) -> MoodSocialForecastResult:
        """Run V3.4 future-state forecasting independently of current state."""

        http_request.state.mood_social_request_id = payload.request_id
        return infer_mood_social_forecast(payload)

    app.include_router(
        create_cognitive_router(
            runtime=run_cognitive,
            api_token=service_settings.api_token,
        )
    )
    app.include_router(
        create_cognitive_v35_router(
            runtime=run_cognitive_v35,
            api_token=service_settings.api_token,
        )
    )
    app.include_router(
        create_cognitive_wandering_fusion_router(
            api_token=service_settings.api_token,
        )
    )
    app.include_router(
        create_facial_affect_router(
            runtime=run_facial_affect,
            api_token=service_settings.api_token,
        )
    )
    app.include_router(
        create_facial_affect_video_router(
            runtime=run_facial_affect_video,
            api_token=service_settings.api_token,
        )
    )

    app.state.settings = service_settings
    app.state.session_manager = manager
    app.state.asr_transcriber = run_asr
    app.state.cognitive_runtime = run_cognitive
    app.state.cognitive_v35_runtime = run_cognitive_v35
    app.state.facial_affect_runtime = run_facial_affect
    app.state.facial_affect_video_runtime = run_facial_affect_video
    return app


app = create_app()
