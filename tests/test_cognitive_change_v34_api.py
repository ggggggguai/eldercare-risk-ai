from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from elderly_monitoring.modules.asr.schemas import ASRTranscript
from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.errors import (
    CognitiveAPIError,
)
from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.inference import (
    CognitiveInferenceRuntime,
)
from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.manifest import (
    CognitiveModelPackage,
    CognitivePackageError,
)
from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.media import (
    DecodedAudioMedia,
)
from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.schemas import (
    DEFAULT_MODEL_VERSION,
    V34_MODEL_VERSION,
    CognitiveInferRequest,
    CognitiveInferResponse,
)
from elderly_monitoring.service.settings import PROJECT_ROOT, ServiceSettings
from elderly_monitoring.service.app import create_app


def _payload(
    *,
    model_version: str | None = V34_MODEL_VERSION,
    audio: bool = True,
    face: bool = True,
) -> dict[str, object]:
    return {
        "schema_version": "cognitive_infer_request_v1",
        "request_id": "cog-v34-contract-001",
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
        "model_version": model_version,
    }


def _asr(text_available: bool) -> ASRTranscript:
    text = "老人正在完整描述图片中的人物和场景" * 4 if text_available else ""
    return ASRTranscript.model_validate(
        {
            "schema_version": "asr_transcript_v1",
            "request_id": "cog-v34-contract-001:asr",
            "status": "completed" if text_available else "completed_empty_speech",
            "text": text,
            "segments": (
                [
                    {
                        "segment_id": "0",
                        "start_ms": 0,
                        "end_ms": 9_000,
                        "text": text,
                        "confidence": None,
                    }
                ]
                if text_available
                else []
            ),
            "language": "zh",
            "quality": {
                "audio_duration_ms": 12_000,
                "speech_duration_ms": 9_000 if text_available else 0,
                "speech_ratio": 0.75 if text_available else 0.0,
                "mean_confidence": None,
                "decode_status": "ok",
                "vad_status": "ok" if text_available else "empty",
            },
            "model": {
                "name": "funasr-paraformer-zh",
                "version": "asr-paraformer-zh-v1.0",
            },
            "warnings": [],
        }
    )


class _FakeASR:
    def __init__(self, text_available: bool = True) -> None:
        self.transcript = _asr(text_available)

    def transcribe(self, **_kwargs) -> ASRTranscript:
        return self.transcript


class _FakeEncoder:
    def __init__(self, size: int) -> None:
        self.size = size
        self.calls = 0

    def encode(self, *_args, **_kwargs) -> np.ndarray:
        self.calls += 1
        return np.full((self.size,), 0.1, dtype=np.float32)


class _FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.marker = torch.nn.Parameter(torch.zeros(()))
        self.last_missing_mask: torch.Tensor | None = None

    def forward(self, _features, _quality, missing_mask):
        self.last_missing_mask = missing_mask.detach().cpu().clone()
        return SimpleNamespace(
            hc_vs_non_hc_logit=torch.tensor([0.5], device=self.marker.device),
            mci_vs_hc_logit=torch.tensor([99.0], device=self.marker.device),
            ad_mci_hc_logits=torch.tensor([[99.0, -99.0, 0.0]], device=self.marker.device),
            moca_standardized=torch.tensor([99.0], device=self.marker.device),
        )


class _FakePackage:
    def __init__(self, model_version: str, *, fail: bool = False) -> None:
        self.model_version = model_version
        self.fail = fail
        self.load_calls = 0
        self.model = _FakeModel()
        self.face_encoder = _FakeEncoder(512)
        self.assets = SimpleNamespace(
            model=self.model,
            audio_encoder=_FakeEncoder(768),
            text_encoder=_FakeEncoder(768),
            face_encoder=self.face_encoder,
            face_detector=object(),
            metadata=SimpleNamespace(
                model_version=model_version,
                calibration_a=0.4,
                calibration_b=0.3,
                moca_mean=20.0,
                moca_std=5.0,
                primary_modalities=(
                    ("audio", "text")
                    if model_version == V34_MODEL_VERSION
                    else ("audio", "text", "face")
                ),
                research_outputs_available=model_version == DEFAULT_MODEL_VERSION,
            ),
        )

    def verify(self):
        if self.fail:
            raise CognitivePackageError("fixture package unavailable")
        return self.assets.metadata

    def load_assets(self):
        self.load_calls += 1
        if self.fail:
            raise CognitivePackageError("fixture package unavailable")
        return self.assets

    def close(self) -> None:
        return None


@contextmanager
def _face_context(enabled: bool):
    yield (
        SimpleNamespace(duration_ms=12_000, frame_paths=(Path("frame.jpg"),))
        if enabled
        else None
    )


def _patch_media(monkeypatch: pytest.MonkeyPatch, *, face: bool) -> None:
    from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue import (
        inference,
    )

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


def test_request_omission_and_null_keep_v33_default() -> None:
    omitted = _payload(model_version=None)
    omitted.pop("model_version")
    assert CognitiveInferRequest.model_validate(omitted).model_version is None
    assert CognitiveInferRequest.model_validate(_payload(model_version=None)).model_version is None


def test_cognitive_runtime_import_does_not_load_funasr() -> None:
    script = (
        "import sys; "
        "import elderly_monitoring.modules.mental_health.submodules."
        "cognitive_change_clue.inference; "
        "assert not any(name == 'funasr' or name.startswith('funasr.') "
        "for name in sys.modules)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_real_packages_verify_with_distinct_runtime_policies() -> None:
    v33 = CognitiveModelPackage(
        PROJECT_ROOT / "models/mental_health/cognitive_change_clue/v3.3.0",
        expected_model_version=DEFAULT_MODEL_VERSION,
    ).verify()
    v34_root = PROJECT_ROOT / "models/mental_health/cognitive_change_clue/v3.4.0"
    v34 = CognitiveModelPackage(
        v34_root,
        expected_model_version=V34_MODEL_VERSION,
    ).verify()
    assert v33.primary_modalities == ("audio", "text", "face")
    assert v33.research_outputs_available is True
    assert v34.primary_modalities == ("audio", "text")
    assert v34.research_outputs_available is False
    with pytest.raises(CognitivePackageError):
        CognitiveModelPackage(v34_root).verify()


def test_first_asset_load_rechecks_package_after_startup_verification(
    tmp_path: Path,
) -> None:
    checkpoint_hash = "a" * 64
    files = {
        "manifest.json": {
            "schema_version": "cognitive_model_manifest_v2",
            "model_version": V34_MODEL_VERSION,
            "network_access": "forbidden",
            "source_checkpoint_sha256": checkpoint_hash,
            "candidate": {
                "id": "V34-A2",
                "heads": "main_head",
                "primary_modalities": ["audio", "text"],
                "face_in_primary_score": False,
            },
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
    checksum_rows = []
    for name, value in sorted(files.items()):
        payload = json.dumps(value, sort_keys=True).encode("utf-8")
        (tmp_path / name).write_bytes(payload)
        checksum_rows.append(f"{hashlib.sha256(payload).hexdigest()}  {name}")
    (tmp_path / "sha256sums.txt").write_text(
        "\n".join(checksum_rows) + "\n",
        encoding="ascii",
    )
    package = CognitiveModelPackage(
        tmp_path,
        expected_model_version=V34_MODEL_VERSION,
    )
    package.verify()
    (tmp_path / "calibration.json").write_text("{}", encoding="utf-8")
    with pytest.raises(CognitivePackageError, match="checksum mismatch"):
        package.load_assets()


def test_service_settings_register_both_package_paths() -> None:
    settings = ServiceSettings.load(
        environ={
            "COGNITIVE_MODEL_PACKAGE_PATH": "C:/models/v33",
            "COGNITIVE_MODEL_PACKAGE_V34_PATH": "C:/models/v34",
        }
    )
    assert settings.cognitive_model_package_path == Path("C:/models/v33")
    assert settings.cognitive_model_package_v34_path == Path("C:/models/v34")

    app = create_app(settings=settings, session_manager=SimpleNamespace())
    assert set(app.state.cognitive_runtime.packages) == {
        DEFAULT_MODEL_VERSION,
        V34_MODEL_VERSION,
    }


def test_route_accepts_explicit_v34_and_preserves_response_version() -> None:
    class RouteRuntime:
        def verify_package(self) -> None:
            return None

        def close(self) -> None:
            return None

        def infer(self, request: CognitiveInferRequest) -> CognitiveInferResponse:
            assert request.model_version == V34_MODEL_VERSION
            return CognitiveInferResponse.model_validate(
                {
                    "schema_version": "cognitive_infer_response_v1",
                    "request_id": request.request_id,
                    "status": "insufficient_input",
                    "cognitive_clue_score": None,
                    "cognitive_clue_level": None,
                    "confidence": None,
                    "research_outputs": None,
                    "used_modalities": [],
                    "modality_quality": {"audio": 0.0, "text": 0.0, "face": 0.0},
                    "asr_status": "not_requested",
                    "model_version": V34_MODEL_VERSION,
                    "warnings": [],
                }
            )

    app = create_app(
        settings=ServiceSettings(model_path=Path("unused.pt"), api_token="test-token"),
        session_manager=SimpleNamespace(),
        cognitive_runtime=RouteRuntime(),
    )
    response = TestClient(app).post(
        "/v1/mental-health/cognitive-change/infer",
        headers={"Authorization": "Bearer test-token"},
        json=_payload(audio=False, face=False),
    )
    assert response.status_code == 200
    assert response.json()["model_version"] == V34_MODEL_VERSION


def test_unknown_and_missing_v34_fail_before_media_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue import (
        inference,
    )

    monkeypatch.setattr(
        inference,
        "decode_audio_media",
        lambda *_args, **_kwargs: pytest.fail("media must not be read"),
    )
    v33 = _FakePackage(DEFAULT_MODEL_VERSION)
    runtime = CognitiveInferenceRuntime(
        packages={DEFAULT_MODEL_VERSION: v33},
        asr_client=_FakeASR(),
    )
    unknown = _payload(model_version="cognitive-mm-v9")
    with pytest.raises(CognitiveAPIError) as unsupported:
        runtime.infer(CognitiveInferRequest.model_validate(unknown))
    assert unsupported.value.code == "MODEL_VERSION_UNSUPPORTED"

    with pytest.raises(CognitiveAPIError) as unavailable:
        runtime.infer(CognitiveInferRequest.model_validate(_payload()))
    assert unavailable.value.code == "MODEL_ARTIFACT_UNAVAILABLE"
    assert v33.load_calls == 0


def test_tampered_v34_does_not_fall_back_to_v33(monkeypatch: pytest.MonkeyPatch) -> None:
    from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue import (
        inference,
    )

    monkeypatch.setattr(
        inference,
        "decode_audio_media",
        lambda *_args, **_kwargs: pytest.fail("media must not be read"),
    )
    v33 = _FakePackage(DEFAULT_MODEL_VERSION)
    v34 = _FakePackage(V34_MODEL_VERSION, fail=True)
    runtime = CognitiveInferenceRuntime(
        packages={DEFAULT_MODEL_VERSION: v33, V34_MODEL_VERSION: v34},
        asr_client=_FakeASR(),
    )
    with pytest.raises(CognitiveAPIError) as caught:
        runtime.infer(CognitiveInferRequest.model_validate(_payload()))
    assert caught.value.code == "MODEL_ARTIFACT_UNAVAILABLE"
    assert v34.load_calls == 1
    assert v33.load_calls == 0


@pytest.mark.parametrize(
    ("audio", "face", "text", "status", "used"),
    [
        (True, True, True, "completed", ["audio", "text"]),
        (True, False, True, "completed", ["audio", "text"]),
        (True, True, False, "degraded", ["audio"]),
        (True, False, False, "degraded", ["audio"]),
        (False, True, False, "insufficient_input", []),
        (False, False, False, "insufficient_input", []),
    ],
)
def test_v34_modality_policy_masks_face_and_hides_untrained_research_heads(
    monkeypatch: pytest.MonkeyPatch,
    audio: bool,
    face: bool,
    text: bool,
    status: str,
    used: list[str],
) -> None:
    _patch_media(monkeypatch, face=face)
    package = _FakePackage(V34_MODEL_VERSION)
    runtime = CognitiveInferenceRuntime(
        packages={V34_MODEL_VERSION: package},
        asr_client=_FakeASR(text),
    )
    result = runtime.infer(
        CognitiveInferRequest.model_validate(
            _payload(audio=audio, face=face, model_version=V34_MODEL_VERSION)
        )
    )
    assert result.model_version == V34_MODEL_VERSION
    assert result.status == status
    assert result.used_modalities == used
    assert result.research_outputs is None
    assert package.face_encoder.calls == 0
    if audio:
        assert package.model.last_missing_mask is not None
        assert bool(package.model.last_missing_mask[0, 2]) is True
    if face:
        assert result.modality_quality.face > 0.0
    if status == "degraded":
        assert result.confidence is not None and result.confidence <= 0.650


def test_response_schema_rejects_cross_version_research_output_leakage() -> None:
    base = {
        "schema_version": "cognitive_infer_response_v1",
        "request_id": "cog-v34-schema",
        "status": "completed",
        "cognitive_clue_score": 50.0,
        "cognitive_clue_level": "attention",
        "confidence": 0.7,
        "research_outputs": None,
        "used_modalities": ["audio", "text"],
        "modality_quality": {"audio": 0.8, "text": 0.8, "face": 0.0},
        "asr_status": "completed",
        "model_version": V34_MODEL_VERSION,
        "warnings": [],
    }
    CognitiveInferResponse.model_validate(base)
    leaked = dict(base)
    leaked["research_outputs"] = {
        "mci_hc_score": 0.5,
        "ad_mci_hc_scores": {"hc": 0.3, "mci": 0.4, "ad": 0.3},
        "moca_prediction": 20.0,
    }
    with pytest.raises(ValueError):
        CognitiveInferResponse.model_validate(leaked)
