import base64
import io
import wave

from fastapi.testclient import TestClient

from elderly_monitoring.modules.asr import api
from elderly_monitoring.modules.asr.engine import EngineResult
from elderly_monitoring.modules.asr.settings import ASRSettings


class _FakeEngine:
    def __init__(self) -> None:
        self.warmed = False

    @property
    def is_ready(self) -> bool:
        return self.warmed

    def warmup(self) -> None:
        self.warmed = True

    def transcribe(self, audio):
        return EngineResult(text="部署接口转写。", timestamps=[[0, audio.duration_ms]])


def _wav_base64() -> str:
    stream = io.BytesIO()
    with wave.open(stream, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 16000 * 3)
    return base64.b64encode(stream.getvalue()).decode("ascii")


def test_dedicated_app_exposes_only_asr_routes(monkeypatch) -> None:
    monkeypatch.setattr(api, "verify_native_assets", lambda _: None)
    app = api.create_asr_app(
        settings=ASRSettings(),
        engine=_FakeEngine(),
        api_token="secret-token",
        prewarm=True,
    )
    with TestClient(app) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 200
        paths = {route.path for route in app.routes}
        assert "/v1/asr/transcribe" in paths
        assert not any(path.startswith("/v1/mental-health") for path in paths)
        assert "/v1/monitoring/sessions" not in paths
        payload = {
            "request_id": "asr-deploy-001",
            "audio_input": {
                "source_type": "base64",
                "source": _wav_base64(),
                "format": "wav",
                "sample_rate": 16000,
                "channels": 1,
            },
        }
        assert client.post("/v1/asr/transcribe", json=payload).status_code == 401
        response = client.post(
            "/v1/asr/transcribe",
            json=payload,
            headers={"Authorization": "Bearer secret-token"},
        )
        assert response.status_code == 200
        assert response.json()["text"] == "部署接口转写。"
