from __future__ import annotations

import base64
import hashlib
import io
import json
import time
import wave
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from elderly_monitoring.modules.asr.schemas import ASRTranscript
from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.asr_client import (
    ASRClientError,
    ASRHttpClient,
)
from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.errors import (
    CognitiveAPIError,
)
from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.inference import (
    CognitiveInferenceRuntime,
    _level_for_score,
    _round_half_up,
)
from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.manifest import (
    CognitiveModelPackage,
    CognitivePackageError,
)
from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.media import (
    DecodedAudioMedia,
    MediaReadSettings,
    decode_audio_media,
    read_media_source,
    validate_media_alignment,
)
from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.schemas import (
    AudioMediaInput,
    CognitiveInferRequest,
    CognitiveInferResponse,
)
from elderly_monitoring.service.app import create_app
from elderly_monitoring.service.settings import ServiceSettings


def _payload(*, audio: bool = True, face: bool = True) -> dict[str, object]:
    return {
        "schema_version": "cognitive_infer_request_v1",
        "request_id": "cog-contract-001",
        "capture_window": {"start_ms": 0, "end_ms": 12_000},
        "audio_input": (
            {
                "source_type": "base64",
                "source": "AA==",
                "format": "wav",
                "sample_rate": 16_000,
                "channels": 1,
            }
            if audio
            else None
        ),
        "face_input": (
            {
                "source_type": "base64",
                "source": "AA==",
                "format": "mp4",
                "media_type": "video",
                "fps": 10,
            }
            if face
            else None
        ),
        "asr_options": {"language": "zh", "return_timestamps": True},
        "model_version": None,
    }


def _insufficient_response(request_id: str) -> CognitiveInferResponse:
    return CognitiveInferResponse.model_validate(
        {
            "schema_version": "cognitive_infer_response_v1",
            "request_id": request_id,
            "status": "insufficient_input",
            "cognitive_clue_score": None,
            "cognitive_clue_level": None,
            "confidence": None,
            "research_outputs": None,
            "used_modalities": [],
            "modality_quality": {"audio": 0.0, "text": 0.0, "face": 0.0},
            "asr_status": "not_requested",
            "model_version": "cognitive-mm-v3.3.0",
            "warnings": [],
        }
    )


class _RouteRuntime:
    def verify_package(self) -> None:
        return None

    def close(self) -> None:
        return None

    def infer(self, request: CognitiveInferRequest) -> CognitiveInferResponse:
        return _insufficient_response(request.request_id)


def _api_client() -> TestClient:
    settings = ServiceSettings(
        model_path=Path("unused.pt"),
        api_token="test-token",
    )
    return TestClient(
        create_app(
            settings=settings,
            session_manager=SimpleNamespace(),
            cognitive_runtime=_RouteRuntime(),
        )
    )


def test_cognitive_route_auth_and_strict_unknown_field_contract() -> None:
    client = _api_client()
    assert client.post(
        "/v1/mental-health/cognitive-change/infer", json=_payload()
    ).status_code == 401

    body = _payload()
    body["task_type"] = "animal_fluency"
    response = client.post(
        "/v1/mental-health/cognitive-change/infer",
        headers={"Authorization": "Bearer test-token"},
        json=body,
    )
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "UNKNOWN_FIELD"


def test_cognitive_route_maps_schema_and_normal_200_contract() -> None:
    client = _api_client()
    invalid = _payload()
    invalid["schema_version"] = "cognitive_infer_request_v2"
    response = client.post(
        "/v1/mental-health/cognitive-change/infer",
        headers={"Authorization": "Bearer test-token"},
        json=invalid,
    )
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "UNSUPPORTED_SCHEMA_VERSION"

    response = client.post(
        "/v1/mental-health/cognitive-change/infer",
        headers={"Authorization": "Bearer test-token"},
        json=_payload(audio=False, face=False),
    )
    assert response.status_code == 200
    assert response.json()["status"] == "insufficient_input"


@pytest.mark.parametrize(
    ("code", "status_code"),
    [
        ("MEDIA_UNREACHABLE", 408),
        ("MEDIA_DECODE_FAILED", 422),
        ("MODEL_ARTIFACT_UNAVAILABLE", 503),
    ],
)
def test_cognitive_route_preserves_structured_runtime_error_mapping(
    code: str,
    status_code: int,
) -> None:
    class ErrorRuntime(_RouteRuntime):
        def infer(self, request: CognitiveInferRequest) -> CognitiveInferResponse:
            raise CognitiveAPIError(code=code, request_id=request.request_id)

    app = create_app(
        settings=ServiceSettings(model_path=Path("unused.pt"), api_token="test-token"),
        session_manager=SimpleNamespace(),
        cognitive_runtime=ErrorRuntime(),
    )
    response = TestClient(app).post(
        "/v1/mental-health/cognitive-change/infer",
        headers={"Authorization": "Bearer test-token"},
        json=_payload(audio=False, face=False),
    )
    assert response.status_code == status_code
    assert response.json()["detail"]["code"] == code


def test_cognitive_route_maps_internal_error_and_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InternalRuntime(_RouteRuntime):
        def infer(self, _request: CognitiveInferRequest) -> CognitiveInferResponse:
            raise RuntimeError("hidden detail")

    def client_for(runtime: _RouteRuntime) -> TestClient:
        return TestClient(
            create_app(
                settings=ServiceSettings(model_path=Path("unused.pt"), api_token="test-token"),
                session_manager=SimpleNamespace(),
                cognitive_runtime=runtime,
            )
        )

    response = client_for(InternalRuntime()).post(
        "/v1/mental-health/cognitive-change/infer",
        headers={"Authorization": "Bearer test-token"},
        json=_payload(audio=False, face=False),
    )
    assert response.status_code == 500
    assert response.json()["detail"]["code"] == "INTERNAL_ERROR"
    assert "hidden detail" not in response.text

    from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue import api

    monkeypatch.setattr(api, "COGNITIVE_TIMEOUT_SECONDS", 0.001)

    class SlowRuntime(_RouteRuntime):
        def infer(self, request: CognitiveInferRequest) -> CognitiveInferResponse:
            time.sleep(0.05)
            return _insufficient_response(request.request_id)

    response = client_for(SlowRuntime()).post(
        "/v1/mental-health/cognitive-change/infer",
        headers={"Authorization": "Bearer test-token"},
        json=_payload(audio=False, face=False),
    )
    assert response.status_code == 504
    assert response.json()["detail"]["code"] == "ALGORITHM_TIMEOUT"


class _FakeEncoder:
    def __init__(self, size: int) -> None:
        self.size = size

    def encode(self, *_args, **_kwargs) -> np.ndarray:
        return np.full((self.size,), 0.1, dtype=np.float32)


class _FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.marker = torch.nn.Parameter(torch.zeros(()))

    def forward(self, _features, _quality, _missing_mask):
        return SimpleNamespace(
            hc_vs_non_hc_logit=torch.tensor([0.5], device=self.marker.device),
            mci_vs_hc_logit=torch.tensor([0.25], device=self.marker.device),
            ad_mci_hc_logits=torch.tensor([[0.1, 0.5, -0.2]], device=self.marker.device),
            moca_standardized=torch.tensor([0.2], device=self.marker.device),
        )


class _FakePackage:
    def __init__(self) -> None:
        self.assets = SimpleNamespace(
            model=_FakeModel(),
            audio_encoder=_FakeEncoder(768),
            text_encoder=_FakeEncoder(768),
            face_encoder=_FakeEncoder(512),
            face_detector=object(),
            metadata=SimpleNamespace(
                calibration_a=0.3690602100110895,
                calibration_b=0.4189044303368853,
                moca_mean=21.6866485013624,
                moca_std=5.423474201900755,
            ),
        )

    def verify(self):
        return self.assets.metadata

    def load_assets(self):
        return self.assets

    def close(self) -> None:
        return None


def _asr(text: str, status: str = "completed") -> ASRTranscript:
    return ASRTranscript.model_validate(
        {
            "schema_version": "asr_transcript_v1",
            "request_id": "cog-contract-001:asr",
            "status": status,
            "text": text,
            "segments": (
                [{"segment_id": "0", "start_ms": 0, "end_ms": 9000, "text": text, "confidence": None}]
                if text
                else []
            ),
            "language": "zh",
            "quality": {
                "audio_duration_ms": 12000,
                "speech_duration_ms": 9000 if text else 0,
                "speech_ratio": 0.75 if text else 0.0,
                "mean_confidence": None,
                "decode_status": "ok",
                "vad_status": "ok" if text else "empty",
            },
            "model": {"name": "funasr-paraformer-zh", "version": "asr-paraformer-zh-v1.0"},
            "warnings": [],
        }
    )


class _FakeASR:
    def __init__(self, transcript: ASRTranscript | Exception) -> None:
        self.transcript = transcript

    def transcribe(self, **_kwargs) -> ASRTranscript:
        if isinstance(self.transcript, Exception):
            raise self.transcript
        return self.transcript


@contextmanager
def _face_context(enabled: bool):
    yield (
        SimpleNamespace(duration_ms=12_000, frame_paths=(Path("frame.jpg"),))
        if enabled
        else None
    )


@pytest.mark.parametrize(
    ("audio", "face", "text", "expected_status", "expected_modalities"),
    [
        (True, True, True, "completed", ["audio", "text", "face"]),
        (True, False, True, "degraded", ["audio", "text"]),
        (True, True, False, "degraded", ["audio", "face"]),
        (True, False, False, "degraded", ["audio"]),
        (False, True, False, "insufficient_input", []),
        (False, False, False, "insufficient_input", []),
    ],
)
def test_frozen_modality_status_contracts(
    monkeypatch: pytest.MonkeyPatch,
    audio: bool,
    face: bool,
    text: bool,
    expected_status: str,
    expected_modalities: list[str],
) -> None:
    from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue import inference

    decoded = DecodedAudioMedia(
        decoded=SimpleNamespace(
            duration_ms=12_000,
            samples=np.zeros(12_000 * 16, dtype=np.float32),
        ),
        payload=b"wav",
    )
    monkeypatch.setattr(inference, "decode_audio_media", lambda *_args, **_kwargs: decoded)
    monkeypatch.setattr(
        inference,
        "decode_face_media",
        lambda *_args, **_kwargs: _face_context(face),
    )
    monkeypatch.setattr(
        inference,
        "preprocess_face_frames",
        lambda *_args, **_kwargs: SimpleNamespace(
            valid_count=16,
            mean_luma=0.5,
            tensors=(np.zeros((3, 224, 224), dtype=np.float32),),
        ),
    )
    transcript = _asr("老人正在完整描述图片中的人物和场景" * 4) if text else _asr("", "completed_empty_speech")
    runtime = CognitiveInferenceRuntime(
        package=_FakePackage(),
        asr_client=_FakeASR(transcript),
    )
    request = CognitiveInferRequest.model_validate(_payload(audio=audio, face=face))
    result = runtime.infer(request)
    assert result.status == expected_status
    assert result.used_modalities == expected_modalities
    if expected_status == "insufficient_input":
        assert result.cognitive_clue_score is None
        assert result.research_outputs is None
    else:
        assert result.cognitive_clue_score is not None
        assert result.research_outputs is not None
        if expected_status == "degraded":
            assert result.confidence <= 0.650


def test_asr_failure_preserves_decoded_audio_degraded_path(monkeypatch: pytest.MonkeyPatch) -> None:
    from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue import inference

    decoded = DecodedAudioMedia(
        decoded=SimpleNamespace(
            duration_ms=12_000,
            samples=np.zeros(12_000 * 16, dtype=np.float32),
        ),
        payload=b"wav",
    )
    monkeypatch.setattr(inference, "decode_audio_media", lambda *_args, **_kwargs: decoded)
    runtime = CognitiveInferenceRuntime(
        package=_FakePackage(),
        asr_client=_FakeASR(ASRClientError("timeout")),
    )
    result = runtime.infer(
        CognitiveInferRequest.model_validate(_payload(audio=True, face=False))
    )
    assert result.status == "degraded"
    assert result.used_modalities == ["audio"]
    assert result.asr_status == "failed"
    assert "asr_failed" in [warning.code for warning in result.warnings]


def test_repeated_runtime_input_keeps_core_results_identical(monkeypatch: pytest.MonkeyPatch) -> None:
    from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue import inference

    decoded = DecodedAudioMedia(
        decoded=SimpleNamespace(
            duration_ms=12_000,
            samples=np.zeros(12_000 * 16, dtype=np.float32),
        ),
        payload=b"wav",
    )
    monkeypatch.setattr(inference, "decode_audio_media", lambda *_args, **_kwargs: decoded)
    runtime = CognitiveInferenceRuntime(
        package=_FakePackage(),
        asr_client=_FakeASR(_asr("老人正在完整描述图片中的人物和场景" * 4)),
    )
    request = CognitiveInferRequest.model_validate(_payload(audio=True, face=False))
    first = runtime.infer(request)
    second = runtime.infer(request)
    assert first.cognitive_clue_score == second.cognitive_clue_score
    assert first.cognitive_clue_level == second.cognitive_clue_level
    assert first.confidence == second.confidence
    assert first.research_outputs == second.research_outputs


def test_asr_http_client_uses_fixed_contract_and_no_retry() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        body = json.loads(request.content)
        assert body["request_id"] == "request-1:asr"
        assert body["language"] == "zh"
        assert body["enable_vad"] is True
        assert body["enable_punctuation"] is True
        assert body["return_timestamps"] is True
        assert body["model_version"] == "asr-paraformer-zh-v1.0"
        return httpx.Response(503, json={"detail": "unavailable"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    asr = ASRHttpClient(
        url="http://asr.test/v1/asr/transcribe",
        api_token="token",
        client=client,
    )
    declaration = AudioMediaInput.model_validate(_payload()["audio_input"])
    audio = DecodedAudioMedia(
        decoded=SimpleNamespace(),
        payload=b"audio",
    )
    with pytest.raises(ASRClientError):
        asr.transcribe(request_id="request-1", audio=audio, declaration=declaration)
    assert len(requests) == 1


def _wav_base64(duration_ms: int) -> str:
    sample_count = int(duration_ms * 16_000 / 1000)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16_000)
        stream.writeframes(b"\0\0" * sample_count)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


@pytest.mark.parametrize("duration_ms", [2_999, 60_001])
def test_audio_duration_boundaries_map_to_media_too_long(duration_ms: int) -> None:
    media = AudioMediaInput(
        source_type="base64",
        source=_wav_base64(duration_ms),
        format="wav",
        sample_rate=16_000,
        channels=1,
    )
    with pytest.raises(CognitiveAPIError) as caught:
        decode_audio_media(media, request_id="duration-boundary")
    assert caught.value.code == "MEDIA_TOO_LONG"


def test_media_alignment_rejects_more_than_200_ms() -> None:
    with pytest.raises(CognitiveAPIError) as caught:
        validate_media_alignment(
            capture_duration_ms=12_000,
            audio_duration_ms=12_000,
            face_duration_ms=12_201,
            request_id="misaligned",
        )
    assert caught.value.code == "MEDIA_MISALIGNED"


def test_media_source_invalid_size_and_unreachable_codes(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = MediaReadSettings(connect_timeout_seconds=0.1, total_timeout_seconds=0.1)
    with pytest.raises(CognitiveAPIError) as invalid:
        read_media_source(
            source_type="base64",
            source="not-base64",
            max_bytes=8,
            request_id="invalid",
            field="audio_input.source",
            settings=settings,
        )
    assert invalid.value.code == "MEDIA_INVALID"

    with pytest.raises(CognitiveAPIError) as too_large:
        read_media_source(
            source_type="base64",
            source=base64.b64encode(b"12345").decode(),
            max_bytes=4,
            request_id="too-large",
            field="audio_input.source",
            settings=settings,
        )
    assert too_large.value.code == "MEDIA_TOO_LARGE"

    with pytest.raises(CognitiveAPIError) as unreachable:
        read_media_source(
            source_type="url",
            source="http://127.0.0.1:1/unreachable.wav",
            max_bytes=1024,
            request_id="unreachable",
            field="audio_input.source",
            settings=settings,
        )
    assert unreachable.value.code == "MEDIA_UNREACHABLE"


def test_score_boundaries_use_half_up_before_level() -> None:
    first = _round_half_up(39.95, 1)
    second = _round_half_up(69.95, 1)
    assert first == 40.0 and _level_for_score(first) == "attention"
    assert second == 70.0 and _level_for_score(second) == "high_attention"


def test_model_package_detects_checksum_mismatch(tmp_path: Path) -> None:
    checkpoint_hash = "b" * 64
    files = {
        "manifest.json": {
            "model_version": "cognitive-mm-v3.3.0",
            "network_access": "forbidden",
            "source_checkpoint_sha256": checkpoint_hash,
        },
        "calibration.json": {
            "checkpoint_sha256": checkpoint_hash,
            "a": 1.0,
            "b": 0.0,
        },
        "feature_stats.json": {
            "checkpoint_sha256": checkpoint_hash,
            "moca_train_subject_mean": 20.0,
            "moca_train_subject_std": 5.0,
        },
    }
    lines = []
    for name, value in sorted(files.items()):
        payload = json.dumps(value, sort_keys=True).encode("utf-8")
        (tmp_path / name).write_bytes(payload)
        lines.append(f"{hashlib.sha256(payload).hexdigest()}  {name}")
    (tmp_path / "sha256sums.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    CognitiveModelPackage(tmp_path).verify()
    (tmp_path / "calibration.json").write_text("{}", encoding="utf-8")
    with pytest.raises(CognitivePackageError):
        CognitiveModelPackage(tmp_path).verify()


def test_unsupported_model_version_is_rejected_before_loading() -> None:
    runtime = CognitiveInferenceRuntime(
        package=_FakePackage(),
        asr_client=_FakeASR(_asr("有效文本" * 20)),
    )
    payload = _payload(audio=False, face=False)
    payload["model_version"] = "cognitive-mm-v9"
    with pytest.raises(CognitiveAPIError) as caught:
        runtime.infer(CognitiveInferRequest.model_validate(payload))
    assert caught.value.code == "MODEL_VERSION_UNSUPPORTED"
