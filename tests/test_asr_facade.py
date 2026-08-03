import base64
import io
import wave

from elderly_monitoring.modules.asr.engine import (
    ASRModelUnavailableError,
    EngineResult,
    get_default_engine,
)
from elderly_monitoring.modules.asr.facade import transcribe
from elderly_monitoring.modules.asr.schemas import ASRRequest
from elderly_monitoring.modules.asr.settings import ASRSettings


def _silent_wav_base64(duration_ms: int = 3000) -> str:
    stream = io.BytesIO()
    with wave.open(stream, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * (16 * duration_ms))
    return base64.b64encode(stream.getvalue()).decode("ascii")


class _FakeEngine:
    def transcribe(self, audio):
        assert audio.sample_rate == 16000
        assert audio.duration_ms == 3000
        return EngineResult(
            text="测试语音。",
            sentence_info=[{"text": "测试语音。", "start": -30, "end": 3030}],
            timestamps=[[-30, 900], [900, 3030]],
        )


class _MissingEngine:
    def transcribe(self, audio):
        raise ASRModelUnavailableError("missing")


def _request(source: str) -> ASRRequest:
    return ASRRequest(
        request_id="asr-test-001",
        audio_input={
            "source_type": "base64",
            "source": source,
            "format": "wav",
            "sample_rate": 16000,
            "channels": 1,
        },
    )


def test_facade_returns_stable_transcript_with_fake_engine() -> None:
    result = transcribe(_request(_silent_wav_base64()), engine=_FakeEngine(), settings=ASRSettings())
    assert result.schema_version == "asr_transcript_v1"
    assert result.status == "completed"
    assert result.text == "测试语音。"
    assert result.segments[0].start_ms == 0
    assert result.segments[0].end_ms == 3000
    assert result.quality.mean_confidence is None


def test_facade_maps_decode_and_model_failures_to_contract_statuses() -> None:
    decode_failure = transcribe(_request("not-base64"), engine=_FakeEngine())
    assert decode_failure.status == "unsupported_audio"
    model_failure = transcribe(_request(_silent_wav_base64()), engine=_MissingEngine())
    assert model_failure.status == "model_unavailable"


def test_default_engine_is_reused_for_equal_runtime_settings() -> None:
    first = get_default_engine(ASRSettings())
    second = get_default_engine(ASRSettings())
    assert first is second
