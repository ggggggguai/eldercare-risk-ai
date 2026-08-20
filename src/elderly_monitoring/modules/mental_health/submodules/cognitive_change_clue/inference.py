"""Runtime multimodal inference orchestration for cognitive clues V3.3/V3.4."""

from __future__ import annotations

import math
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Mapping

import numpy as np

from elderly_monitoring.modules.asr.schemas import ASRTranscript
from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.asr_client import (
    ASRClientError,
    ASRClientProtocol,
)
from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.errors import (
    CognitiveAPIError,
)
from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.manifest import (
    CognitiveModelPackage,
    CognitivePackageError,
)
from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.media import (
    DecodedAudioMedia,
    decode_audio_media,
    decode_face_media,
    validate_media_alignment,
)
from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.preprocess import (
    normalize_cognitive_text,
    preprocess_face_frames,
)
from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.quality import (
    ModalityQuality,
    compute_audio_quality,
    compute_face_quality,
    compute_text_quality,
)
from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.schemas import (
    DEFAULT_MODEL_VERSION,
    SUPPORTED_MODEL_VERSIONS,
    AudioMediaInput,
    CognitiveInferRequest,
    CognitiveInferResponse,
    CognitiveModalityQuality,
    CognitiveResearchClassScores,
    CognitiveResearchOutputs,
    CognitiveWarning,
)


class CognitiveInferenceRuntime:
    """One process-local runtime. Model and encoders are loaded lazily after integrity checks."""

    def __init__(
        self,
        *,
        package: CognitiveModelPackage | None = None,
        packages: Mapping[str, CognitiveModelPackage] | None = None,
        asr_client: ASRClientProtocol,
    ) -> None:
        if (package is None) == (packages is None):
            raise ValueError("provide either package or packages")
        registry = (
            {DEFAULT_MODEL_VERSION: package}
            if package is not None
            else dict(packages or {})
        )
        if any(version not in SUPPORTED_MODEL_VERSIONS for version in registry):
            raise ValueError("cognitive package registry contains an unsupported version")
        self.packages = registry
        self.asr_client = asr_client

    def verify_package(self, model_version: str = DEFAULT_MODEL_VERSION) -> None:
        package = self.packages.get(model_version)
        if package is None:
            raise CognitiveAPIError(code="MODEL_ARTIFACT_UNAVAILABLE", errors=())
        try:
            metadata = package.verify()
            if metadata.model_version != model_version:
                raise CognitivePackageError("registered model package version mismatch")
        except CognitivePackageError as exc:
            raise CognitiveAPIError(
                code="MODEL_ARTIFACT_UNAVAILABLE",
                errors=(),
            ) from exc

    def close(self) -> None:
        for package in self.packages.values():
            package.close()

    def infer(self, request: CognitiveInferRequest) -> CognitiveInferResponse:
        if request.model_version is not None and request.model_version not in SUPPORTED_MODEL_VERSIONS:
            raise CognitiveAPIError(
                code="MODEL_VERSION_UNSUPPORTED",
                request_id=request.request_id,
                errors=(
                    {
                        "field": "model_version",
                        "reason": _supported_version_reason(),
                    },
                ),
            )
        selected_model_version = request.model_version or DEFAULT_MODEL_VERSION
        defaulted = request.model_version is None
        package = self.packages.get(selected_model_version)
        if package is None:
            raise CognitiveAPIError(
                code="MODEL_ARTIFACT_UNAVAILABLE",
                request_id=request.request_id,
            )
        try:
            assets = package.load_assets()
            if (
                getattr(assets.metadata, "model_version", selected_model_version)
                != selected_model_version
            ):
                raise CognitivePackageError("registered model package version mismatch")
        except CognitivePackageError as exc:
            raise CognitiveAPIError(
                code="MODEL_ARTIFACT_UNAVAILABLE",
                request_id=request.request_id,
            ) from exc

        decoded_audio: DecodedAudioMedia | None = None
        if request.audio_input is not None:
            decoded_audio = decode_audio_media(
                request.audio_input,
                request_id=request.request_id,
            )

        face_context = (
            decode_face_media(
                request.face_input,
                request_id=request.request_id,
            )
            if request.face_input is not None
            else _null_context()
        )
        with face_context as decoded_face:
            validate_media_alignment(
                capture_duration_ms=request.capture_window.duration_ms,
                audio_duration_ms=(
                    decoded_audio.decoded.duration_ms if decoded_audio is not None else None
                ),
                face_duration_ms=(decoded_face.duration_ms if decoded_face is not None else None),
                request_id=request.request_id,
            )
            return self._infer_decoded(
                request=request,
                assets=assets,
                decoded_audio=decoded_audio,
                decoded_face=decoded_face,
                defaulted=defaulted,
                model_version=selected_model_version,
            )

    def _infer_decoded(
        self,
        *,
        request: CognitiveInferRequest,
        assets: Any,
        decoded_audio: DecodedAudioMedia | None,
        decoded_face: Any,
        defaulted: bool,
        model_version: str,
    ) -> CognitiveInferResponse:
        warnings: list[CognitiveWarning] = []
        audio_quality = _quality_for_missing_audio()
        text_quality = _quality_for_missing_text()
        face_quality = _quality_for_missing_face()
        asr_status = "not_requested"
        normalized_text = ""
        text_char_count = 0
        transcript: ASRTranscript | None = None

        if decoded_audio is None:
            _add_warning(warnings, "missing_audio", "audio", "audio input was not provided")
            _add_warning(warnings, "missing_text", "text", "text requires an audio input and ASR")
        else:
            asr_success = False
            try:
                transcript = self.asr_client.transcribe(
                    request_id=request.request_id,
                    audio=decoded_audio,
                    declaration=request.audio_input,
                )
                asr_status = transcript.status
                asr_success = transcript.status in {"completed", "completed_empty_speech"}
            except ASRClientError:
                asr_status = "failed"
                _add_warning(warnings, "asr_failed", "text", "ASR service failed or timed out")
            if asr_success and transcript is not None:
                speech_duration_ms = int(transcript.quality.speech_duration_ms)
                audio_quality = compute_audio_quality(
                    duration_ms=decoded_audio.decoded.duration_ms,
                    speech_duration_ms=speech_duration_ms,
                    decode_ok=True,
                )
                normalized_text, text_char_count = normalize_cognitive_text(transcript.text)
                text_quality = compute_text_quality(
                    char_count=text_char_count,
                    segments=transcript.segments,
                )
                if transcript.status == "completed_empty_speech" or not normalized_text:
                    _add_warning(warnings, "asr_empty_speech", "text", "ASR returned no usable speech")
                    _add_warning(warnings, "missing_text", "text", "usable ASR text was not available")
            else:
                # Keep duration/decode evidence without inventing a speech ratio.
                audio_quality = compute_audio_quality(
                    duration_ms=decoded_audio.decoded.duration_ms,
                    speech_duration_ms=0,
                    decode_ok=True,
                )
                _add_warning(warnings, "missing_text", "text", "usable ASR text was not available")

        if audio_quality.missing:
            _add_warning(warnings, "low_audio_quality", "audio", "audio quality is below the runtime threshold")

        audio_available = not bool(audio_quality.missing)
        text_available = audio_available and not bool(text_quality.missing)
        if decoded_audio is not None and not text_available and normalized_text:
            _add_warning(warnings, "low_text_quality", "text", "normalized ASR text quality is below the runtime threshold")

        primary_modalities = tuple(
            getattr(
                assets.metadata,
                "primary_modalities",
                ("audio", "text", "face"),
            )
        )
        primary_modality_set = set(primary_modalities)
        face_available = False
        if decoded_face is None:
            if "face" in primary_modality_set:
                _add_warning(warnings, "missing_face", "face", "face video input was not provided")
        else:
            try:
                face_preprocessed = preprocess_face_frames(
                    decoded_face.frame_paths,
                    assets.face_detector,
                )
                face_quality = compute_face_quality(
                    valid_count=face_preprocessed.valid_count,
                    mean_luma=face_preprocessed.mean_luma,
                )
                face_available = not bool(face_quality.missing)
                if face_available and "face" in primary_modality_set:
                    face_feature = assets.face_encoder.encode(face_preprocessed.tensors)
                else:
                    face_feature = np.zeros((512,), dtype=np.float32)
                    if not face_available:
                        _add_warning(warnings, "low_face_quality", "face", "face quality is below the runtime threshold")
            except Exception as exc:
                face_feature = np.zeros((512,), dtype=np.float32)
                face_quality = _quality_for_missing_face()
                _add_warning(warnings, "low_face_quality", "face", "face frames could not produce a usable feature")
                face_available = False

        if not audio_available:
            if decoded_face is not None and not face_available:
                _add_warning(warnings, "low_face_quality", "face", "face quality is below the runtime threshold")
            if defaulted:
                _add_warning(warnings, "model_defaulted", "model", "default model version was used")
            return _build_insufficient_response(
                request=request,
                asr_status=asr_status,
                qualities=(audio_quality, text_quality, face_quality),
                warnings=warnings,
                model_version=model_version,
            )

        try:
            audio_feature = assets.audio_encoder.encode(
                decoded_audio.decoded.samples,
                duration_ms=decoded_audio.decoded.duration_ms,
            )
            text_feature = (
                assets.text_encoder.encode(normalized_text)
                if text_available
                else np.zeros((768,), dtype=np.float32)
            )
            if not face_available:
                face_feature = np.zeros((512,), dtype=np.float32)
            features = {
                "audio": _tensor(audio_feature),
                "text": _tensor(text_feature),
                "face": _tensor(face_feature),
            }
            quality = _tensor(
                np.asarray(
                    [audio_quality.score, text_quality.score, face_quality.score],
                    dtype=np.float32,
                )
            ).reshape(1, 3)
            face_used = face_available and "face" in primary_modality_set
            missing_mask = _tensor(
                np.asarray(
                    [False, not text_available, not face_used],
                    dtype=np.bool_,
                )
            ).reshape(1, 3)
            import torch

            model_device = next(assets.model.parameters()).device
            features = {name: value.to(model_device) for name, value in features.items()}
            quality = quality.to(model_device)
            missing_mask = missing_mask.to(model_device)

            with torch.inference_mode():
                output = assets.model(features, quality, missing_mask)
            raw_logit = float(output.hc_vs_non_hc_logit[0].detach().cpu().item())
            probability = max(
                1e-7,
                min(
                    1.0 - 1e-7,
                    _sigmoid(
                        assets.metadata.calibration_a * raw_logit
                        + assets.metadata.calibration_b
                    ),
                ),
            )
            score = _round_half_up(probability * 100.0, 1)
            level = _level_for_score(score)
            certainty = 1.0 - _binary_entropy(probability)
            used = ["audio"]
            if text_available and "text" in primary_modality_set:
                used.append("text")
            if face_used:
                used.append("face")
            available_count = len(used)
            quality_values = [audio_quality.score]
            if "text" in used:
                quality_values.append(text_quality.score)
            if "face" in used:
                quality_values.append(face_quality.score)
            primary_count = len(primary_modalities)
            confidence = 0.5 * (sum(quality_values) / len(quality_values)) + 0.3 * certainty + 0.2 * (available_count / primary_count)
            if available_count < primary_count:
                confidence = min(confidence, 0.650)
            confidence = _round_half_up(max(0.0, min(1.0, confidence)), 3)
            research = None
            if getattr(assets.metadata, "research_outputs_available", True):
                mci_logit = float(output.mci_vs_hc_logit[0].detach().cpu().item())
                class_logits = output.ad_mci_hc_logits[0].detach().cpu()
                moca_standardized = float(output.moca_standardized[0].detach().cpu().item())
                research = CognitiveResearchOutputs(
                    mci_hc_score=_round_half_up(_sigmoid(mci_logit), 3),
                    ad_mci_hc_scores=CognitiveResearchClassScores(
                        hc=_round_half_up(float(torch.softmax(class_logits, dim=0)[0].item()), 3),
                        mci=_round_half_up(float(torch.softmax(class_logits, dim=0)[1].item()), 3),
                        ad=_round_half_up(float(torch.softmax(class_logits, dim=0)[2].item()), 3),
                    ),
                    moca_prediction=_round_half_up(
                        max(0.0, min(30.0, moca_standardized * assets.metadata.moca_std + assets.metadata.moca_mean)),
                        1,
                    ),
                )
        except CognitiveAPIError:
            raise
        except Exception as exc:
            raise CognitiveAPIError(
                code="INTERNAL_ERROR",
                request_id=request.request_id,
            ) from exc

        if not text_available:
            _add_warning(warnings, "missing_text", "text", "usable text feature was not available")
        if decoded_face is None:
            pass
        elif not face_available and not any(item.code == "low_face_quality" for item in warnings):
            _add_warning(warnings, "low_face_quality", "face", "face quality is below the runtime threshold")
        if defaulted:
            _add_warning(warnings, "model_defaulted", "model", "default model version was used")
        status = "completed" if used == list(primary_modalities) else "degraded"
        warnings = _ordered_warnings(warnings)
        return CognitiveInferResponse(
            schema_version="cognitive_infer_response_v1",
            request_id=request.request_id,
            status=status,
            cognitive_clue_score=score,
            cognitive_clue_level=level,
            confidence=confidence,
            research_outputs=research,
            used_modalities=used,
            modality_quality=CognitiveModalityQuality(
                audio=_round_half_up(audio_quality.score, 3),
                text=_round_half_up(text_quality.score if text_available else 0.0, 3),
                face=_round_half_up(face_quality.score if face_available else 0.0, 3),
            ),
            asr_status=asr_status,
            model_version=model_version,
            warnings=warnings,
        )


def _build_insufficient_response(
    *,
    request: CognitiveInferRequest,
    asr_status: str,
    qualities: tuple[ModalityQuality, ModalityQuality, ModalityQuality],
    warnings: list[CognitiveWarning],
    model_version: str,
) -> CognitiveInferResponse:
    warnings = _ordered_warnings(warnings)
    return CognitiveInferResponse(
        schema_version="cognitive_infer_response_v1",
        request_id=request.request_id,
        status="insufficient_input",
        cognitive_clue_score=None,
        cognitive_clue_level=None,
        confidence=None,
        research_outputs=None,
        used_modalities=[],
        modality_quality=CognitiveModalityQuality(
            audio=_round_half_up(qualities[0].score, 3),
            text=_round_half_up(qualities[1].score, 3),
            face=_round_half_up(qualities[2].score, 3),
        ),
        asr_status=asr_status,
        model_version=model_version,
        warnings=warnings,
    )


def _quality_for_missing_audio() -> ModalityQuality:
    return compute_audio_quality(duration_ms=0, speech_duration_ms=0, decode_ok=False)


def _quality_for_missing_text() -> ModalityQuality:
    return compute_text_quality(char_count=0)


def _quality_for_missing_face() -> ModalityQuality:
    return compute_face_quality(valid_count=0, mean_luma=0.0)


def _tensor(value: np.ndarray):
    import torch

    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    tensor = torch.from_numpy(array)
    if not torch.isfinite(tensor).all():
        raise ValueError("encoder produced non-finite values")
    return tensor


def _add_warning(
    warnings: list[CognitiveWarning],
    code: str,
    modality: str,
    message: str,
) -> None:
    if any(item.code == code for item in warnings):
        return
    warnings.append(CognitiveWarning(code=code, modality=modality, message=message))


def _ordered_warnings(
    warnings: list[CognitiveWarning],
) -> list[CognitiveWarning]:
    order = {
        "missing_audio": 0,
        "low_audio_quality": 1,
        "missing_text": 2,
        "low_text_quality": 3,
        "missing_face": 4,
        "low_face_quality": 5,
        "asr_empty_speech": 6,
        "asr_failed": 7,
        "model_defaulted": 8,
    }
    return sorted(warnings, key=lambda item: order[item.code])


def _sigmoid(value: float) -> float:
    if value >= 0:
        exponent = math.exp(-value)
        return 1.0 / (1.0 + exponent)
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def _binary_entropy(probability: float) -> float:
    p = max(1e-7, min(1.0 - 1e-7, probability))
    return -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p)) / math.log(2.0)


def _round_half_up(value: float, places: int) -> float:
    quantum = Decimal("1") if places == 0 else Decimal("1") / (Decimal(10) ** places)
    return float(Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP))


def _level_for_score(score: float) -> str:
    if score < 40.0:
        return "normal"
    if score < 70.0:
        return "attention"
    return "high_attention"


class _null_context:
    def __enter__(self):
        return None

    def __exit__(self, *args: Any) -> None:
        return None


def _supported_version_reason() -> str:
    return "must be null, " + ", or ".join(SUPPORTED_MODEL_VERSIONS)


__all__ = ["CognitiveInferenceRuntime"]
