from __future__ import annotations

import os

os.environ.setdefault("MENTAL_HEALTH_DEFER_MODEL_LOAD", "1")

from fastapi.testclient import TestClient

from elderly_monitoring.service.mental_health_app import create_mental_health_app
from elderly_monitoring.service.settings import ServiceSettings


class _Runtime:
    def verify_package(self):
        return {"status": "passed"}

    def verify_deterministic_probe(self):
        return {"status": "passed"}

    def close(self):
        return None

    def infer(self, request):  # pragma: no cover - route registration test only
        raise AssertionError("not called")

    def predict(self, routes, *, input_sha256=None):  # pragma: no cover
        raise AssertionError("not called")


def test_dedicated_app_exposes_complete_psychological_surface_without_fall_routes():
    runtime = _Runtime()
    app = create_mental_health_app(
        settings=ServiceSettings(api_token="secret-token"),
        cognitive_v35_runtime=runtime,
        facial_affect_runtime=runtime,
        facial_affect_video_runtime=runtime,
        verify_assets=False,
    )
    paths = {route.path for route in app.routes}
    assert "/v1/mental-health/mood-social/r11-production/infer" in paths
    assert "/v1/mental-health/mood-social/forecast/infer" in paths
    assert "/v1/mental-health/cognitive-change/subject-infer" in paths
    assert "/v1/mental-health/cognitive-change/wandering-fusion/candidate-infer" in paths
    assert "/v1/mental-health/facial-affect/infer" in paths
    assert "/v1/mental-health/facial-affect/video-infer" in paths
    assert not any(path.startswith("/v1/monitoring/sessions") for path in paths)

    with TestClient(app) as client:
        assert client.get("/health/live").status_code == 200
        ready = client.get("/health/ready")
        assert ready.status_code == 200
        assert ready.json()["fall_service_registered"] is False


def test_protected_routes_reject_missing_token():
    runtime = _Runtime()
    app = create_mental_health_app(
        settings=ServiceSettings(api_token="secret-token"),
        cognitive_v35_runtime=runtime,
        facial_affect_runtime=runtime,
        facial_affect_video_runtime=runtime,
        verify_assets=False,
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/mental-health/cognitive-change/wandering-fusion/candidate-infer",
            json={},
        )
        assert response.status_code == 401
