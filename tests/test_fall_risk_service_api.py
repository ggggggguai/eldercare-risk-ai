import unittest
from pathlib import Path

from pydantic import ValidationError
from fastapi.testclient import TestClient

from elderly_monitoring.service.app import create_app
from elderly_monitoring.service.session import SessionStatus
from elderly_monitoring.service.schemas import StartSessionRequest, StreamUrlUpdate


class ServiceSchemaTest(unittest.TestCase):
    def test_accepts_valid_start_request(self) -> None:
        request = StartSessionRequest(
            request_id="request-1",
            stream_url="rtsp://camera.example/live",
            device_id="cam-1",
            person_id="elder-1",
            scene_region="living_room",
            callback_url="https://backend.example/events",
        )
        self.assertEqual(str(request.stream_url), "rtsp://camera.example/live")

    def test_rejects_empty_identifiers(self) -> None:
        for field in ("request_id", "device_id", "person_id", "scene_region"):
            payload = {
                "request_id": "request-1",
                "stream_url": "https://camera.example/live.m3u8",
                "device_id": "cam-1",
                "person_id": "elder-1",
                "scene_region": "living_room",
                "callback_url": "https://backend.example/events",
            }
            payload[field] = "   "
            with self.subTest(field=field), self.assertRaises(ValidationError):
                StartSessionRequest(**payload)

    def test_rejects_unsupported_stream_protocol(self) -> None:
        with self.assertRaises(ValidationError):
            StreamUrlUpdate(stream_url="file:///tmp/video.mp4")

    def test_rejects_non_http_callback(self) -> None:
        with self.assertRaises(ValidationError):
            StartSessionRequest(
                request_id="request-1",
                stream_url="https://camera.example/live.m3u8",
                device_id="cam-1",
                person_id="elder-1",
                scene_region="living_room",
                callback_url="ftp://backend.example/events",
            )


class _FakeSession:
    def __init__(self, session_id="s1", request_id="r1"):
        from datetime import datetime, timezone
        self.session_id = session_id
        self.request_id = request_id
        self.status = SessionStatus.RUNNING
        self.device_id = "cam-1"
        self.person_id = "elder-1"
        self.started_at = datetime.now(timezone.utc)
        self.last_frame_at = None
        self.last_error = None


class _FakeManager:
    def __init__(self):
        self.session = None

    def start(self, **kwargs):
        if self.session and self.session.request_id == kwargs["request_id"]:
            return self.session
        if self.session:
            raise ValueError("another session is active")
        self.session = _FakeSession(request_id=kwargs["request_id"])
        return self.session

    def get(self, session_id):
        if self.session and self.session.session_id == session_id:
            return self.session
        return None

    def update_url(self, session_id, stream_url):
        if not self.get(session_id):
            return None
        return self.session

    def stop(self, session_id):
        session = self.get(session_id)
        if session:
            session.status = SessionStatus.STOPPED
        return session


class ServiceApiTest(unittest.TestCase):
    def setUp(self):
        from elderly_monitoring.service.settings import ServiceSettings
        self.manager = _FakeManager()
        settings = ServiceSettings(api_token="api", model_path="missing.pt")
        self.client = TestClient(create_app(settings=settings, session_manager=self.manager))

    def headers(self):
        return {"Authorization": "Bearer api"}

    def payload(self, request_id="r1"):
        return {"request_id": request_id, "stream_url": "https://camera/live", "device_id": "cam-1", "person_id": "elder-1", "scene_region": "home", "callback_url": "https://backend/events"}

    def test_auth_start_query_update_stop_and_health(self):
        self.assertEqual(self.client.get("/health/live").status_code, 200)
        self.assertEqual(self.client.post("/v1/monitoring/sessions", json=self.payload()).status_code, 401)
        response = self.client.post("/v1/monitoring/sessions", json=self.payload(), headers=self.headers())
        self.assertEqual(response.status_code, 202)
        self.assertEqual(self.client.post("/v1/monitoring/sessions", json=self.payload(), headers=self.headers()).status_code, 202)
        self.assertEqual(self.client.post("/v1/monitoring/sessions", json=self.payload("r2"), headers=self.headers()).status_code, 409)
        status_response = self.client.get("/v1/monitoring/sessions/s1", headers=self.headers())
        self.assertEqual(status_response.status_code, 200)
        self.assertEqual(status_response.json()["stream_epoch"], 0)
        self.assertEqual(status_response.json()["frame_diagnostics"], {})
        self.assertEqual(self.client.put("/v1/monitoring/sessions/s1/stream-url", json={"stream_url": "https://camera/new"}, headers=self.headers()).status_code, 200)
        self.assertEqual(self.client.post("/v1/monitoring/sessions/s1/stop", headers=self.headers()).status_code, 202)
        self.assertEqual(self.client.post("/v1/monitoring/sessions/s1/stop", headers=self.headers()).status_code, 202)

    def test_ready_requires_model(self):
        self.assertEqual(self.client.get("/health/ready").status_code, 503)


class ServiceSettingsTest(unittest.TestCase):
    def test_loads_gait_model_runtime_overrides(self) -> None:
        from elderly_monitoring.service.settings import ServiceSettings

        settings = ServiceSettings.load(
            path=Path("/path/that/does/not/exist.yaml"),
            environ={
                "GAIT_MODEL_PATH": "models/gait.pt",
                "GAIT_MODEL_DEVICE": "cpu",
                "GAIT_MODEL_WINDOW_FRAMES": "96",
                "FRAME_QUEUE_CAPACITY": "4",
            },
        )

        self.assertEqual(settings.gait_model_path, Path("models/gait.pt"))
        self.assertEqual(settings.gait_model_device, "cpu")
        self.assertEqual(settings.gait_model_window_frames, 96)
        self.assertEqual(settings.frame_queue_capacity, 4)

    def test_repository_config_freezes_stage_two_runtime_gates(self) -> None:
        from elderly_monitoring.runtime.realtime_fall_risk import (
            _feature_assembly_config,
        )
        from elderly_monitoring.service.settings import ServiceSettings

        settings = ServiceSettings.load(
            path=Path("configs/modules/fall_risk_service.yaml"), environ={}
        )
        assembly = _feature_assembly_config(
            {
                "pose_window_sec": settings.pose_window_sec,
                "analysis_interval_sec": settings.analysis_interval_sec,
                "branch_quality": settings.branch_quality,
            }
        )

        self.assertEqual(settings.primary_lost_timeout_sec, 2.0)
        self.assertEqual(settings.fall_state["static_duration_sec"], 10.0)
        self.assertEqual(settings.fall_state["recovery_confirmation_sec"], 3.0)
        self.assertEqual(settings.fall_state["episode_ttl_sec"], 30.0)
        self.assertEqual(settings.fall_state["recovery_upright_angle_threshold"], 45.0)
        self.assertEqual(settings.fall_state["recovery_motion_threshold"], 0.05)
        self.assertEqual(settings.outbox_capacity, 32)
        self.assertEqual(settings.outbox_drain_timeout_sec, 3.0)
        self.assertEqual(assembly.gait_gate.min_frames, 8)
        self.assertEqual(assembly.near_fall_gate.min_frames, 5)
        self.assertEqual(assembly.near_fall_gate.min_effective_fps, 6.0)
        self.assertEqual(assembly.fall_state_gate.max_gap_sec, 0.75)


if __name__ == "__main__":
    unittest.main()
