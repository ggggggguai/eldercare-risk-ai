import unittest
from pathlib import Path
from unittest.mock import patch

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
        self.last_error = "stream failed at rtmp://secret.example/live?token=secret"
        self.stream_epoch = 0
        self.runtime_diagnostics = {
            "stream_url": "rtmp://secret.example/live?token=secret",
            "authorization": "Bearer secret",
            "latest_event": {
                "module": "fall_risk",
                "timestamp": "2026-08-27T10:00:00+00:00",
                "model_version": "fall-risk-test-v1",
            },
            "outbox": {
                "active_items": [],
                "recent_terminal_items": [
                    {
                        "event_id": "event-1",
                        "session_id": session_id,
                        "delivery_status": "delivered",
                        "last_status_code": 204,
                        "first_generated_time": "2026-08-27T10:00:01+00:00",
                    }
                ],
            },
        }


class _FakeManager:
    def __init__(self):
        self.session = None
        self.sessions = {}
        self.last_start_kwargs = None

    def start(self, **kwargs):
        self.last_start_kwargs = kwargs
        if self.session and self.session.request_id == kwargs["request_id"]:
            return self.session
        if self.session:
            raise ValueError("another session is active")
        self.session = _FakeSession(request_id=kwargs["request_id"])
        self.sessions[self.session.session_id] = self.session
        return self.session

    def get(self, session_id):
        if self.session and self.session.session_id == session_id:
            return self.session
        return None

    def update_url(self, session_id, stream_url):
        if not self.get(session_id):
            return None
        return self.session

    def update_baseline_period(self, session_id, period):
        if not self.get(session_id):
            return None
        self.session.baseline_period = period
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
        baseline_response = self.client.post(
            "/v1/monitoring/sessions/s1/baseline-period",
            json={"period": {"period_id": "2026-07-12"}},
            headers=self.headers(),
        )
        self.assertEqual(baseline_response.status_code, 200)
        self.assertEqual(self.client.post("/v1/monitoring/sessions/s1/stop", headers=self.headers()).status_code, 202)
        self.assertEqual(self.client.post("/v1/monitoring/sessions/s1/stop", headers=self.headers()).status_code, 202)

    def test_ready_requires_model(self):
        self.assertEqual(self.client.get("/health/ready").status_code, 503)

    def test_demo_live_adapter_starts_and_stops_without_browser_token(self):
        response = self.client.post(
            "/demo/live-start", json={"stream_url": "rtmp://camera/live"}
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "running")
        self.assertTrue(payload["request_id"].startswith("FR-LIVE-"))

        status_response = self.client.get("/demo/live-status")
        status_payload = status_response.json()
        self.assertTrue(status_payload["active"])
        self.assertEqual(status_payload["request_id"], payload["request_id"])
        self.assertEqual(status_payload["session_id"], payload["session_id"])
        self.assertEqual(status_payload["device_id"], "cam-1")
        self.assertIsNotNone(status_payload["started_at"])
        self.assertEqual(status_payload["event_trace"]["event_id"], "event-1")
        self.assertEqual(status_payload["event_trace"]["delivery_status"], "delivered")
        self.assertNotIn("stream_url", status_payload)
        status_text = status_response.text
        self.assertNotIn("secret.example", status_text)
        self.assertNotIn("Bearer secret", status_text)

        callback = self.client.post("/demo/callback", json={"module": "fall_risk"})
        self.assertEqual(callback.status_code, 204)
        stop_response = self.client.post(f"/demo/live-stop/{payload['session_id']}")
        self.assertEqual(stop_response.status_code, 200)
        self.assertEqual(stop_response.json()["status"], "stopped")

    def test_demo_uses_server_side_default_stream_without_exposing_url(self):
        from elderly_monitoring.service.settings import ServiceSettings

        secret_stream = "https://camera.example/live.flv?token=secret"
        settings = ServiceSettings(
            api_token="api",
            model_path=Path("missing.pt"),
            demo_stream_url=secret_stream,
        )
        client = TestClient(create_app(settings=settings, session_manager=self.manager))

        config_response = client.get("/demo/config")
        self.assertEqual(config_response.status_code, 200)
        self.assertEqual(config_response.json(), {"default_stream_configured": True})
        self.assertNotIn(secret_stream, config_response.text)

        start_response = client.post("/demo/live-start")
        self.assertEqual(start_response.status_code, 200)
        self.assertEqual(self.manager.last_start_kwargs["stream_url"], secret_stream)
        self.assertNotIn(secret_stream, start_response.text)
        client.close()

    def test_demo_live_status_exposes_redacted_terminal_failure(self):
        response = self.client.post(
            "/demo/live-start", json={"stream_url": "rtmp://camera/live"}
        )
        self.assertEqual(response.status_code, 200)
        self.manager.session.status = SessionStatus.FAILED
        status_response = self.client.get("/demo/live-status")
        payload = status_response.json()
        self.assertFalse(payload["active"])
        self.assertEqual(payload["last_terminal"]["status"], "failed")
        self.assertNotIn("secret.example", payload["last_terminal"]["reason"])
        self.assertNotIn("token=secret", payload["last_terminal"]["reason"])

    def test_demo_overlay_tracks_contained_video_content(self):
        response = self.client.get("/demo/fall-risk")

        self.assertEqual(response.status_code, 200)
        self.assertIn("syncVisualOverlay", response.text)
        self.assertIn("naturalWidth", response.text)
        self.assertIn("Math.min(frameWidth / sourceWidth", response.text)

    def test_demo_polls_visual_status_at_live_frame_rate_without_overlap(self):
        response = self.client.get("/demo/fall-risk")

        self.assertEqual(response.status_code, 200)
        self.assertIn("const LIVE_STATUS_POLL_MS = 150", response.text)
        self.assertIn("pollInFlight: false", response.text)
        self.assertIn("if (state.pollInFlight) return", response.text)
        self.assertIn("finally { state.pollInFlight = false; }", response.text)
        self.assertIn("}, LIVE_STATUS_POLL_MS)", response.text)

    def test_demo_start_rebinds_its_fixed_synthetic_baseline_period(self):
        response = self.client.get("/demo/fall-risk")

        self.assertEqual(response.status_code, 200)
        self.assertIn("DEMO_BASELINE_REQUEST_URL", response.text)
        self.assertIn("/baseline-period", response.text)
        self.assertIn("if (state.defaultStreamConfigured) await bindDemoBaseline", response.text)

    def test_demo_poll_rebinds_baseline_for_existing_live_session(self):
        response = self.client.get("/demo/fall-risk")

        self.assertEqual(response.status_code, 200)
        self.assertIn("baselineBindInFlight: false", response.text)
        self.assertIn("!payload.runtime_diagnostics?.baseline_comparison", response.text)
        self.assertIn("await bindDemoBaseline(payload.session_id)", response.text)

    def test_demo_timeline_click_renders_selected_event_evidence(self):
        response = self.client.get("/demo/fall-risk")

        self.assertEqual(response.status_code, 200)
        self.assertIn('id="eventDetail"', response.text)
        self.assertIn("renderTimelineEventDetail", response.text)
        self.assertIn("selectedTimelineEventKey", response.text)

    def test_demo_page_reload_does_not_stop_active_live_session(self):
        response = self.client.get("/demo/fall-risk")

        self.assertEqual(response.status_code, 200)
        self.assertNotIn("navigator.sendBeacon(`/demo/live-stop/", response.text)

    def test_demo_diagnostics_support_live_runtime_field_names(self):
        response = self.client.get("/demo/fall-risk")

        self.assertEqual(response.status_code, 200)
        self.assertIn("last_frame_age_sec", response.text)
        self.assertIn("max_gap_sec", response.text)
        self.assertIn("dropped_oldest", response.text)
        self.assertIn("正在积累有效观测", response.text)

    def test_demo_discloses_synthetic_personal_baseline(self):
        response = self.client.get("/demo/fall-risk")

        self.assertEqual(response.status_code, 200)
        self.assertIn("合成正常基线", response.text)
        self.assertIn("仅用于算法机制演示", response.text)
        self.assertIn("baseline_comparison", response.text)

    def test_ready_requires_ffmpeg_tools_for_ffmpeg_backend(self):
        from elderly_monitoring.service.settings import ServiceSettings

        settings = ServiceSettings(
            api_token="api",
            model_path=Path(__file__),
            stream_reader_backend="ffmpeg",
        )
        client = TestClient(create_app(settings=settings, session_manager=self.manager))
        with patch("elderly_monitoring.service.app.shutil.which", return_value=None):
            response = client.get("/health/ready")
        client.close()

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"], "ffmpeg backend is not available")


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
                "POSE_INFERENCE_SIZE": "512",
            },
        )

        self.assertEqual(settings.gait_model_path, Path("models/gait.pt"))
        self.assertEqual(settings.gait_model_device, "cpu")
        self.assertEqual(settings.gait_model_window_frames, 96)
        self.assertEqual(settings.frame_queue_capacity, 4)
        self.assertEqual(settings.pose_inference_size, 512)

    def test_empty_gait_model_override_disables_tcn(self) -> None:
        from elderly_monitoring.service.settings import ServiceSettings

        settings = ServiceSettings.load(
            path=Path("/path/that/does/not/exist.yaml"),
            environ={"GAIT_MODEL_PATH": ""},
        )

        self.assertIsNone(settings.gait_model_path)

    def test_loads_opt_in_sit_stand_tcn_runtime_overrides(self) -> None:
        from elderly_monitoring.service.settings import ServiceSettings

        settings = ServiceSettings.load(
            path=Path("/path/that/does/not/exist.yaml"),
            environ={
                "SIT_STAND_RUNTIME_MODE": "experimental_tcn",
                "SIT_STAND_MODEL_PATH": "reports/sit-stand.pt",
                "SIT_STAND_MODEL_DEVICE": "cpu",
                "SIT_STAND_MODEL_BATCH_SIZE": "32",
            },
        )

        self.assertEqual(settings.sit_stand_runtime_mode, "experimental_tcn")
        self.assertEqual(settings.sit_stand_model_path, Path("reports/sit-stand.pt"))
        self.assertEqual(settings.sit_stand_model_device, "cpu")
        self.assertEqual(settings.sit_stand_model_batch_size, 32)

    def test_loads_near_fall_tabular_runtime_overrides(self) -> None:
        from elderly_monitoring.service.settings import ServiceSettings

        settings = ServiceSettings.load(
            path=Path("/path/that/does/not/exist.yaml"),
            environ={
                "NEAR_FALL_RUNTIME_MODE": "tabular_rescorer",
                "NEAR_FALL_MODEL_PATH": "reports/near-fall.joblib",
                "NEAR_FALL_SCORE_THRESHOLD": "0.3815",
                "NEAR_FALL_ALERT_COOLDOWN_SEC": "15.0",
            },
        )

        self.assertEqual(settings.near_fall_runtime_mode, "tabular_rescorer")
        self.assertEqual(
            settings.near_fall_model_path,
            Path("reports/near-fall.joblib"),
        )
        self.assertEqual(settings.near_fall_score_threshold, 0.3815)
        self.assertEqual(settings.near_fall_alert_cooldown_sec, 15.0)

    def test_repository_config_freezes_stage_two_runtime_gates(self) -> None:
        from elderly_monitoring.runtime.realtime_fall_risk import (
            _feature_assembly_config,
        )
        from elderly_monitoring.service.settings import ServiceSettings

        settings = ServiceSettings.load(
            path=Path("configs/modules/fall_risk_service_v2.yaml"), environ={}
        )
        assembly = _feature_assembly_config(
            {
                "pose_window_sec": settings.pose_window_sec,
                "analysis_interval_sec": settings.analysis_interval_sec,
                "branch_quality": settings.branch_quality,
            }
        )

        self.assertEqual(settings.primary_lost_timeout_sec, 2.0)
        self.assertEqual(
            settings.release_id,
            "fall-risk-competition-v2-20260827",
        )
        self.assertEqual(settings.near_fall_runtime_mode, "tabular_rescorer")
        self.assertTrue(settings.near_fall_model_path.is_file())
        self.assertEqual(settings.near_fall_score_threshold, 0.3815)
        self.assertEqual(settings.near_fall_alert_cooldown_sec, 15.0)
        self.assertEqual(settings.fall_event_runtime_mode, "experimental_tcn")
        self.assertEqual(len(settings.fall_event_shadow_checkpoint_paths), 3)
        self.assertTrue(
            all(path.is_file() for path in settings.fall_event_shadow_checkpoint_paths)
        )
        self.assertEqual(settings.model_path, Path("models/yolov8n-pose.pt"))
        self.assertEqual(
            settings.gait_model_path,
            Path(
                "reports/fall_risk/gait_observable_context_v2/"
                "splitv3-3342705-scfaux-v1/pretrained-seed43/best_model.pt"
            ),
        )
        self.assertTrue(settings.gait_model_path.is_file())
        self.assertEqual(settings.gait_model_device, "cpu")
        self.assertEqual(settings.gait_model_window_frames, 16)
        self.assertEqual(settings.ffmpeg_scale_width, 640)
        self.assertEqual(settings.max_inference_fps, 10.0)
        self.assertEqual(settings.pose_inference_size, 640)
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
