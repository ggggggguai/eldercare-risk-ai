"""Dedicated HTTP service for the complete psychological algorithm module.

This entrypoint intentionally excludes fall-risk monitoring sessions.  The
wandering implementation may reuse fall-risk tracking source code, but it is
deployed and scaled as part of this psychological service.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from elderly_monitoring.modules.mental_health.mood_social import (
    MoodSocialAPIError,
    MoodSocialErrorResponse,
    validation_error_response,
)
from elderly_monitoring.modules.mental_health.mood_social.forecast_api import (
    MOOD_SOCIAL_FORECAST_INFER_PATH,
    infer_mood_social_forecast,
)
from elderly_monitoring.modules.mental_health.mood_social.r11.production import (
    MOOD_SOCIAL_R11_PRODUCTION_INFER_PATH,
    infer_mood_social_r11_production,
)
from elderly_monitoring.modules.mental_health.mood_social.r11.release import (
    R11PackageError,
)
from elderly_monitoring.modules.mental_health.mood_social.r11.schemas import (
    R11CandidateRequest,
    R11CandidateResponse,
)
from elderly_monitoring.modules.mental_health.mood_social.schemas import (
    MoodSocialForecastResult,
    MoodSocialInferRequest,
)
from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.asr_client import (
    ASRHttpClient,
)
from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.errors import (
    CognitiveAPIError,
    validation_error_response as cognitive_validation_error_response,
)
from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.manifest import (
    CognitiveModelPackage,
)
from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.schemas import (
    DEFAULT_MODEL_VERSION,
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
from elderly_monitoring.service.settings import ServiceSettings


SERVICE_VERSION = "mental-health-deploy-v1"


def _asset_manifest_path() -> Path:
    return Path(
        os.getenv(
            "MENTAL_HEALTH_ASSET_MANIFEST",
            "deploy/mental_health/asset_manifest.json",
        )
    )


def _verify_assets(project_root: Path, manifest_path: Path) -> dict[str, Any]:
    from elderly_monitoring.service.mental_health_assets import verify_asset_manifest

    return verify_asset_manifest(project_root=project_root, manifest_path=manifest_path)


def _build_default_runtimes(
    settings: ServiceSettings,
) -> tuple[CognitiveV35RuntimeProtocol, FacialAffectRuntimeProtocol, FacialAffectVideoRuntimeProtocol]:
    encoder_package = CognitiveModelPackage(
        settings.cognitive_model_package_path,
        device=settings.cognitive_inference_device,
        expected_model_version=DEFAULT_MODEL_VERSION,
    )
    cognitive = CognitiveV35InferenceRuntime(
        package=CognitiveV35ModelPackage(
            settings.cognitive_model_package_v35_path,
            encoder_package=encoder_package,
            device=settings.cognitive_inference_device,
        ),
        asr_client=ASRHttpClient(
            url=settings.cognitive_asr_url,
            api_token=settings.api_token,
        ),
    )
    facial = B0DeploymentRuntime(
        settings.facial_affect_b0_package_path,
        device=settings.facial_affect_inference_device,
    )
    try:
        spotting_payload = json.loads(
            settings.facial_affect_spotting_config_path.read_text(encoding="utf-8")
        )
        spotting = SpottingConfig(**spotting_payload)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise RuntimeError("frozen facial-affect spotting config is unavailable") from exc
    facial_video = FacialAffectVideoRuntime(
        b0_runtime=facial,
        config=VideoPipelineConfig(
            max_bytes=settings.facial_affect_video_max_bytes,
            download_timeout_seconds=settings.facial_affect_video_download_timeout_seconds,
            allowed_origins=settings.facial_affect_video_url_allowed_origins,
            temp_root=settings.facial_affect_video_temp_root,
            temp_retention_seconds=settings.facial_affect_video_temp_retention_seconds,
        ),
        spotter=DeterministicMotionStrainSpotter(spotting),
    )
    return cognitive, facial, facial_video


def create_mental_health_app(
    *,
    settings: ServiceSettings | None = None,
    cognitive_v35_runtime: CognitiveV35RuntimeProtocol | None = None,
    facial_affect_runtime: FacialAffectRuntimeProtocol | None = None,
    facial_affect_video_runtime: FacialAffectVideoRuntimeProtocol | None = None,
    verify_assets: bool = True,
) -> FastAPI:
    """Build the psychological service without registering fall endpoints."""

    service_settings = settings or ServiceSettings.load()
    project_root = Path(__file__).resolve().parents[3]
    supplied = (
        cognitive_v35_runtime,
        facial_affect_runtime,
        facial_affect_video_runtime,
    )
    if any(item is None for item in supplied):
        defaults = _build_default_runtimes(service_settings)
        run_cognitive = cognitive_v35_runtime or defaults[0]
        run_facial = facial_affect_runtime or defaults[1]
        run_facial_video = facial_affect_video_runtime or defaults[2]
    else:
        run_cognitive = cognitive_v35_runtime
        run_facial = facial_affect_runtime
        run_facial_video = facial_affect_video_runtime

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        errors: dict[str, str] = {}
        if verify_assets:
            try:
                app.state.asset_audit = _verify_assets(project_root, _asset_manifest_path())
            except Exception as exc:  # readiness records the bounded reason
                errors["assets"] = f"{type(exc).__name__}: {exc}"
        for name, runtime in (
            ("cognitive_v35", run_cognitive),
            ("facial_affect_b0", run_facial),
        ):
            try:
                runtime.verify_package()
                probe = getattr(runtime, "verify_deterministic_probe", None)
                if callable(probe):
                    probe()
            except (CognitiveAPIError, B0DeploymentError, OSError, ValueError) as exc:
                errors[name] = f"{type(exc).__name__}: {exc}"
        app.state.readiness_errors = errors
        try:
            yield
        finally:
            close = getattr(run_cognitive, "close", None)
            if callable(close):
                close()
            close = getattr(run_facial, "close", None)
            if callable(close):
                close()

    app = FastAPI(
        title="YingLingAnJu Psychological Algorithm Service",
        version=SERVICE_VERSION,
        lifespan=lifespan,
    )
    app.state.readiness_errors = {}
    bearer = HTTPBearer(auto_error=False)

    def require_mood_token(
        credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
    ) -> None:
        if (
            credentials is None
            or credentials.scheme.lower() != "bearer"
            or credentials.credentials != service_settings.api_token
        ):
            raise MoodSocialAPIError(code="AUTHENTICATION_FAILED")

    @app.exception_handler(MoodSocialAPIError)
    async def mood_error(_: Request, exception: MoodSocialAPIError) -> JSONResponse:
        return JSONResponse(
            status_code=exception.status_code,
            content=exception.as_response().model_dump(mode="json"),
        )

    @app.exception_handler(CognitiveAPIError)
    async def cognitive_error(_: Request, exception: CognitiveAPIError) -> JSONResponse:
        return JSONResponse(
            status_code=exception.status_code,
            content=exception.as_response().model_dump(mode="json"),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exception: RequestValidationError) -> JSONResponse:
        if request.url.path == FACIAL_AFFECT_VIDEO_INFER_PATH:
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content=video_validation_error_response(exception),
            )
        if request.url.path == COGNITIVE_V35_INFER_PATH:
            response = cognitive_validation_error_response(
                exception,
                request_schema_version=V35_REQUEST_SCHEMA_VERSION,
                supported_model_versions=(V35_MODEL_VERSION,),
            )
            return JSONResponse(status_code=422, content=response.model_dump(mode="json"))
        if request.url.path in {
            MOOD_SOCIAL_R11_PRODUCTION_INFER_PATH,
            MOOD_SOCIAL_FORECAST_INFER_PATH,
        }:
            response = validation_error_response(exception)
            return JSONResponse(status_code=422, content=response.model_dump(mode="json"))
        return await request_validation_exception_handler(request, exception)

    @app.get("/health/live")
    def live() -> dict[str, str]:
        return {"status": "ok", "service": "psychological-algorithm"}

    @app.get("/health/ready")
    def ready() -> dict[str, Any]:
        if app.state.readiness_errors:
            raise HTTPException(
                status_code=503,
                detail={"status": "not_ready", "components": app.state.readiness_errors},
            )
        return {
            "status": "ready",
            "service": "psychological-algorithm",
            "product_branches": {
                "mood_social": "r11_current_plus_v3.4_forecast",
                "cognitive_change": "v3.5_plus_wandering_rule_fusion",
            },
            "fall_service_registered": False,
        }

    @app.post(
        MOOD_SOCIAL_R11_PRODUCTION_INFER_PATH,
        response_model=R11CandidateResponse,
        dependencies=[Depends(require_mood_token)],
        responses={422: {"model": MoodSocialErrorResponse}, 503: {"model": MoodSocialErrorResponse}},
    )
    def infer_r11(payload: R11CandidateRequest, request: Request) -> R11CandidateResponse:
        request.state.mood_social_request_id = payload.request_id
        try:
            return R11CandidateResponse.model_validate(infer_mood_social_r11_production(payload))
        except R11PackageError as exc:
            raise MoodSocialAPIError(
                code="MODEL_ARTIFACT_UNAVAILABLE",
                request_id=payload.request_id,
            ) from exc

    @app.post(
        MOOD_SOCIAL_FORECAST_INFER_PATH,
        response_model=MoodSocialForecastResult,
        dependencies=[Depends(require_mood_token)],
    )
    def infer_forecast(payload: MoodSocialInferRequest, request: Request) -> MoodSocialForecastResult:
        request.state.mood_social_request_id = payload.request_id
        return infer_mood_social_forecast(payload)

    app.include_router(
        create_cognitive_v35_router(runtime=run_cognitive, api_token=service_settings.api_token)
    )
    app.include_router(
        create_cognitive_wandering_fusion_router(api_token=service_settings.api_token)
    )
    app.include_router(
        create_facial_affect_router(runtime=run_facial, api_token=service_settings.api_token)
    )
    app.include_router(
        create_facial_affect_video_router(
            runtime=run_facial_video,
            api_token=service_settings.api_token,
        )
    )
    app.state.settings = service_settings
    app.state.cognitive_v35_runtime = run_cognitive
    app.state.facial_affect_runtime = run_facial
    app.state.facial_affect_video_runtime = run_facial_video
    return app


if os.getenv("MENTAL_HEALTH_DEFER_MODEL_LOAD", "0") == "1":
    # Build/test tooling can import this module without copying external model
    # assets.  The deployment container never sets this flag.
    app = FastAPI(
        title="YingLingAnJu Psychological Algorithm Service (deferred)",
        version=SERVICE_VERSION,
    )
else:
    app = create_mental_health_app()


__all__ = ["SERVICE_VERSION", "app", "create_mental_health_app"]
