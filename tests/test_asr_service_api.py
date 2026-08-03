from fastapi.testclient import TestClient

from elderly_monitoring.modules.asr.schemas import ASRQuality, ASRTranscript
from elderly_monitoring.service.app import create_app
from elderly_monitoring.service.settings import ServiceSettings


class _FakeManager:
    pass


def _fake_transcriber(request):
    return ASRTranscript(
        request_id=request.request_id,
        status="completed",
        text="接口转写。",
        quality=ASRQuality(
            audio_duration_ms=1000,
            speech_duration_ms=800,
            speech_ratio=0.8,
            decode_status="ok",
            vad_status="ok",
        ),
        warnings=["confidence_unavailable"],
    )


def test_asr_http_adapter_reuses_algorithm_auth_and_contract() -> None:
    app = create_app(
        settings=ServiceSettings(api_token="api", model_path="missing.pt"),
        session_manager=_FakeManager(),
        asr_transcriber=_fake_transcriber,
    )
    client = TestClient(app)
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
    response = client.post(
        "/v1/asr/transcribe",
        json=payload,
        headers={"Authorization": "Bearer api"},
    )
    assert response.status_code == 200
    assert response.json()["schema_version"] == "asr_transcript_v1"
    assert response.json()["request_id"] == "asr-http-001"
