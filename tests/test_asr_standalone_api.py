from fastapi.testclient import TestClient

from elderly_monitoring.modules.asr.api import create_asr_app
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
        return EngineResult(text="独立接口转写。", timestamps=[[0, audio.duration_ms]])


def test_standalone_asr_service_prewarms_and_authenticates() -> None:
    engine = _FakeEngine()
    app = create_asr_app(
        settings=ASRSettings(),
        engine=engine,
        api_token="api",
        prewarm=True,
    )
    with TestClient(app) as client:
        assert engine.warmed is True
        assert client.get("/health/ready").status_code == 200
        payload = {
            "request_id": "asr-http-001",
            "audio_input": {
                "source_type": "base64",
                "source": "UklGRg==",
                "format": "wav",
                "sample_rate": None,
                "channels": None,
            },
        }
        assert client.post("/v1/asr/transcribe", json=payload).status_code == 401
