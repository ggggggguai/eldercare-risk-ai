from __future__ import annotations

from functools import partial
import shutil
from pathlib import Path
import asyncio
from uuid import uuid4
from typing import Any, Mapping

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse, Response, StreamingResponse
import cv2
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from elderly_monitoring.service.schemas import (
    BaselinePeriodUpdate,
    EnvironmentReadingRequest,
    SessionAccepted,
    SessionStatusResponse,
    StartSessionRequest,
    StreamUrlUpdate,
)
from elderly_monitoring.service.session import SessionManager, SessionStatus
from elderly_monitoring.service.settings import ServiceSettings
from elderly_monitoring.service.stream_reader import FFmpegStreamReader, StreamReader
from elderly_monitoring.runtime.environment_store import (
    EnvironmentConflictError,
    EnvironmentStore,
    EnvironmentStoreError,
)


def reader_factory_for(settings: ServiceSettings) -> Any:
    if settings.stream_reader_backend == "opencv":
        return StreamReader
    return partial(
        FFmpegStreamReader,
        scale_width=settings.ffmpeg_scale_width,
        max_fps=settings.max_inference_fps,
    )


def _redact_demo_value(value: Any, *, key: str = "") -> Any:
    """Remove stream URLs and auth material before exposing local demo diagnostics."""
    sensitive = {"stream_url", "callback_url", "api_token", "access_token", "authorization", "token"}
    if key.lower() in sensitive:
        return "[redacted]"
    if isinstance(value, Mapping):
        return {
            str(item_key): _redact_demo_value(item_value, key=str(item_key))
            for item_key, item_value in value.items()
        }
    if isinstance(value, list):
        return [_redact_demo_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_demo_value(item) for item in value)
    if isinstance(value, str):
        redacted = value
        for scheme in ("rtsp://", "rtmp://", "http://", "https://"):
            marker = redacted.lower().find(scheme)
            if marker >= 0:
                redacted = redacted[:marker] + "[redacted]"
        return redacted
    return value


def _demo_event_trace(runtime_diagnostics: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return the latest transport trace without exposing callback configuration."""
    outbox = runtime_diagnostics.get("outbox")
    if not isinstance(outbox, Mapping):
        return None
    candidates = [
        item
        for collection_name in ("recent_terminal_items", "active_items")
        for item in (outbox.get(collection_name) or [])
        if isinstance(item, Mapping)
    ]
    if not candidates:
        return None
    latest = max(
        candidates,
        key=lambda item: str(item.get("first_generated_time") or ""),
    )
    allowed = (
        "event_id",
        "episode_id",
        "session_id",
        "stream_epoch",
        "lifecycle_state",
        "version_kind",
        "first_generated_time",
        "delivery_status",
        "attempt_count",
        "retry_count",
        "last_status_code",
        "last_error",
    )
    return {key: _redact_demo_value(latest.get(key), key=key) for key in allowed}


def _demo_model_ready(settings: ServiceSettings) -> bool:
    """Mirror the readiness gate without turning the demo status endpoint into an error."""
    if not settings.model_path.exists():
        return False
    if settings.gait_model_path is not None and not settings.gait_model_path.exists():
        return False
    if settings.sit_stand_runtime_mode == "experimental_tcn" and (
        settings.sit_stand_model_path is None
        or not settings.sit_stand_model_path.exists()
    ):
        return False
    if settings.near_fall_runtime_mode == "tabular_rescorer" and (
        settings.near_fall_model_path is None
        or not settings.near_fall_model_path.exists()
    ):
        return False
    if any(not path.exists() for path in settings.fall_event_shadow_checkpoint_paths):
        return False
    return not (
        settings.stream_reader_backend == "ffmpeg"
        and (shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None)
    )


def _demo_processing_stages(
    active: Any | None,
    runtime_diagnostics: Mapping[str, Any],
    event_trace: Mapping[str, Any] | None,
) -> list[dict[str, str]]:
    """Derive a conservative, evidence-backed processing chain for the demo UI."""
    frame = runtime_diagnostics.get("last_frame") or {}
    visual = frame.get("visual") or {}
    window = frame.get("window") or {}
    has_frame = bool(active and getattr(active, "last_frame_at", None))
    has_person = bool(visual.get("bbox"))
    has_window = bool(window)
    has_branches = bool(window.get("branches"))
    has_event = bool(runtime_diagnostics.get("latest_event") or event_trace)
    delivery_status = str((event_trace or {}).get("delivery_status") or "")
    stages = [
        ("session_created", bool(active), "会话尚未创建"),
        ("stream_connected", bool(active and active.status.value == "running"), "等待视频流连接"),
        ("first_frame_received", has_frame, "等待首帧"),
        ("person_tracking", has_person, "等待有效人体观测"),
        ("pose_window_ready", has_window, "等待姿态窗口"),
        ("branch_analysis", has_branches, "等待分支输入"),
        ("risk_fusion", has_event or bool(window.get("fusion_mask")), "等待有效融合结果"),
        (
            "event_sent",
            delivery_status == "delivered",
            "等待 AlgorithmEvent" if not has_event else "等待回调发送",
        ),
    ]
    if active and active.status.value in {"starting", "reconnecting"}:
        for index, (name, complete, reason) in enumerate(stages):
            if not complete:
                stages[index] = (name, False, "处理中")
                break
    if delivery_status == "failed":
        stages[-1] = ("event_sent", False, "回调发送失败")
    return [
        {"name": name, "status": "complete" if complete else "pending", "reason": reason}
        for name, complete, reason in stages
    ]


def create_app(*, settings: ServiceSettings | None = None, session_manager: SessionManager | None = None) -> FastAPI:
    service_settings = settings or ServiceSettings.load()
    environment_store = EnvironmentStore(
        capacity_per_device=service_settings.environment.store_capacity_per_device
    )
    manager = session_manager or SessionManager(
        reader_factory=reader_factory_for(service_settings),
        model_path=str(service_settings.model_path),
        release_id=service_settings.release_id,
        gait_model_path=service_settings.gait_model_path,
        gait_model_device=service_settings.gait_model_device,
        gait_model_window_frames=service_settings.gait_model_window_frames,
        sit_stand_runtime_mode=service_settings.sit_stand_runtime_mode,
        sit_stand_model_path=service_settings.sit_stand_model_path,
        sit_stand_model_device=service_settings.sit_stand_model_device,
        sit_stand_model_batch_size=service_settings.sit_stand_model_batch_size,
        near_fall_runtime_mode=service_settings.near_fall_runtime_mode,
        near_fall_model_path=service_settings.near_fall_model_path,
        near_fall_score_threshold=service_settings.near_fall_score_threshold,
        near_fall_alert_cooldown_sec=(
            service_settings.near_fall_alert_cooldown_sec
        ),
        fall_event_runtime_mode=service_settings.fall_event_runtime_mode,
        fall_event_shadow_checkpoint_paths=service_settings.fall_event_shadow_checkpoint_paths,
        fall_event_shadow_device=service_settings.fall_event_shadow_device,
        fall_event_shadow_threshold=service_settings.fall_event_shadow_threshold,
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
        environment_store=environment_store,
        environment_settings=service_settings.environment,
    )
    app = FastAPI(title="Elderly Monitoring Fall Risk Service", version="0.2.0")
    static_dir = Path(__file__).with_name("static")
    app.mount("/demo-assets", StaticFiles(directory=static_dir), name="demo-assets")
    demo_media_dir = Path(__file__).resolve().parents[3] / "reports" / "fall_risk" / "runtime" / "demo"
    if demo_media_dir.exists():
        app.mount("/demo-media", StaticFiles(directory=demo_media_dir), name="demo-media")
    @app.get("/demo/fall-risk", include_in_schema=False)
    def fall_risk_demo() -> FileResponse:
        return FileResponse(static_dir / "fall-risk.html")

    @app.get("/demo/live-view", include_in_schema=False)
    def live_view() -> FileResponse:
        return FileResponse(static_dir / "live-view.html")

    @app.get("/demo/live-status", include_in_schema=False)
    def demo_live_status() -> dict[str, Any]:
        """Redacted local-demo status so the page can discover an existing session."""
        sessions = getattr(manager, "sessions", {})
        active = next(
            (session for session in sessions.values() if session.status.value not in {"stopped", "failed"}),
            None,
        )
        if active is None:
            terminal = [
                session
                for session in sessions.values()
                if session.status.value in {"stopped", "failed"}
            ]
            if terminal:
                latest = max(
                    terminal,
                    key=lambda session: str(
                        getattr(session, "last_frame_at", None)
                        or getattr(session, "started_at", None)
                        or ""
                    ),
                )
                reason = _redact_demo_value(getattr(latest, "last_error", None))
                if not reason:
                    reason = "直播会话已停止。"
                return {
                    "active": False,
                    "model_ready": _demo_model_ready(service_settings),
                    "video_state": "not_connected",
                    "last_terminal": {
                        "session_id": latest.session_id,
                        "status": latest.status.value,
                        "reason": reason,
                    },
                }
            return {
                "active": False,
                "model_ready": _demo_model_ready(service_settings),
                "video_state": "not_connected",
                "processing": _demo_processing_stages(None, {}, None),
            }
        manager.get(active.session_id)
        runtime_diagnostics = dict(getattr(active, "runtime_diagnostics", {}))
        event_trace = _demo_event_trace(runtime_diagnostics)
        return {
            "active": True,
            "session_id": active.session_id,
            "request_id": active.request_id,
            "device_id": active.device_id,
            "scene_region": getattr(active, "scene_region", "unknown"),
            "started_at": active.started_at,
            "status": active.status.value,
            "model_ready": _demo_model_ready(service_settings),
            "video_state": (
                "connected"
                if getattr(active, "last_frame_at", None)
                else "connecting"
            ),
            "stream_epoch": int(getattr(active, "stream_epoch", 0)),
            "last_frame_at": getattr(active, "last_frame_at", None),
            "last_error": _redact_demo_value(getattr(active, "last_error", None)),
            "runtime_diagnostics": _redact_demo_value(
                runtime_diagnostics
            ),
            "visual": dict(runtime_diagnostics.get("last_frame", {}).get("visual", {})),
            "event_trace": event_trace,
            "processing": _demo_processing_stages(active, runtime_diagnostics, event_trace),
        }

    @app.get("/demo/config", include_in_schema=False)
    def demo_config() -> dict[str, bool]:
        """Expose availability only; the signed stream URL remains server-side."""
        return {"default_stream_configured": bool(service_settings.demo_stream_url)}

    @app.post("/demo/callback", include_in_schema=False, status_code=204, name="demo_callback")
    async def demo_callback(_: Request) -> Response:
        """Accept callbacks for the local demo without persisting event payloads."""
        return Response(status_code=204)

    @app.post("/demo/live-start", include_in_schema=False)
    def demo_live_start(
        request: Request, stream: StreamUrlUpdate | None = None
    ) -> dict[str, str]:
        """Start a real stream using service-side auth and runtime configuration."""
        stream_url = service_settings.demo_stream_url or (
            stream.stream_url if stream is not None else None
        )
        if not stream_url:
            raise HTTPException(
                status_code=422,
                detail="no default demo stream is configured",
            )
        request_id = f"FR-LIVE-{uuid4().hex[:12]}"
        callback_url = str(request.url_for("demo_callback"))
        try:
            session = manager.start(
                request_id=request_id,
                stream_url=stream_url,
                device_id="live-demo-camera",
                person_id="live-demo-person",
                scene_region="home",
                callback_url=callback_url,
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {
            "session_id": session.session_id,
            "status": session.status.value,
            "request_id": request_id,
        }

    @app.post("/demo/live-stop/{session_id}", include_in_schema=False)
    def demo_live_stop(session_id: str) -> dict[str, str]:
        session = manager.stop(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="session not found")
        return {"session_id": session.session_id, "status": session.status.value}

    @app.get("/demo/live.mjpeg", include_in_schema=False)
    async def demo_live_mjpeg() -> StreamingResponse:
        async def stream():
            while True:
                sessions = getattr(manager, "sessions", {})
                active = next((s for s in sessions.values() if s.status.value not in {"stopped", "failed"}), None)
                frame = getattr(active, "latest_frame", None) if active else None
                if frame is not None:
                    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
                    if ok:
                        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + encoded.tobytes() + b"\r\n"
                await asyncio.sleep(0.12)
        return StreamingResponse(stream(), media_type="multipart/x-mixed-replace; boundary=frame")
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
        if (
            service_settings.sit_stand_runtime_mode == "experimental_tcn"
            and (
                service_settings.sit_stand_model_path is None
                or not service_settings.sit_stand_model_path.exists()
            )
        ):
            raise HTTPException(status_code=503, detail="sit-stand model is not available")
        if (
            service_settings.near_fall_runtime_mode == "tabular_rescorer"
            and (
                service_settings.near_fall_model_path is None
                or not service_settings.near_fall_model_path.exists()
            )
        ):
            raise HTTPException(
                status_code=503,
                detail="near-fall model is not available",
            )
        missing_fall_event_models = [
            path.as_posix()
            for path in service_settings.fall_event_shadow_checkpoint_paths
            if not path.exists()
        ]
        if missing_fall_event_models:
            raise HTTPException(
                status_code=503,
                detail="fall-event TCN model is not available: "
                + ", ".join(missing_fall_event_models),
            )
        if service_settings.stream_reader_backend == "ffmpeg" and (
            shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None
        ):
            raise HTTPException(
                status_code=503, detail="ffmpeg backend is not available"
            )
        return {"status": "ready"}

    @app.post(
        "/v1/environment/readings",
        status_code=202,
        dependencies=[Depends(require_token)],
    )
    def ingest_environment_reading(
        reading: EnvironmentReadingRequest,
        request: Request,
    ) -> JSONResponse:
        if service_settings.environment.mode == "disabled":
            raise HTTPException(status_code=503, detail="environment input is disabled")
        configured_devices = {
            str(binding.get("environment_device_id"))
            for binding in service_settings.environment.camera_bindings.values()
            if isinstance(binding, Mapping) and binding.get("environment_device_id")
        }
        if reading.device_id not in configured_devices:
            raise HTTPException(status_code=422, detail="unknown environment device")
        try:
            _, result = environment_store.ingest(reading.model_dump(mode="json"))
        except EnvironmentConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except EnvironmentStoreError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return JSONResponse(
            status_code=200 if result == "duplicate" else 202,
            content={"status": result, "device_id": reading.device_id, "sequence": reading.sequence},
        )

    @app.get("/v1/environment/readings")
    def list_environment_readings(
        device_id: str | None = Query(default=None),
        limit: int = Query(default=100, ge=1, le=1000),
        order: str = Query(default="desc"),
    ) -> dict[str, Any]:
        if order not in {"asc", "desc"}:
            raise HTTPException(status_code=422, detail="order must be asc or desc")
        records = environment_store.readings(device_id=device_id, limit=limit, order=order)
        return {
            "items": [record.to_dict() for record in records],
            "total": len(records),
            "limit": limit,
            "order": order,
            "has_more": False,
        }

    @app.get("/v1/environment/readings/latest")
    def latest_environment_reading(device_id: str | None = Query(default=None)) -> dict[str, Any]:
        record = environment_store.latest(device_id=device_id)
        return {"item": record.to_dict() if record else None}

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

    @app.post(
        "/v1/monitoring/sessions/{session_id}/baseline-period",
        response_model=SessionAccepted,
        dependencies=[Depends(require_token)],
    )
    def update_baseline_period(
        session_id: str, request: BaselinePeriodUpdate
    ) -> SessionAccepted:
        try:
            session = manager.update_baseline_period(session_id, request.period)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
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
    app.state.environment_store = environment_store
    return app


app = create_app()
