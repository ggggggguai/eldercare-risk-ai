"""Three-task subject-level runtime for the frozen cognitive V3.5 model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

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
    CognitiveModalityQuality,
    CognitiveWarning,
)
from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.v35_model import (
    V35_INPUT_DIMS,
    V35_MODALITIES,
)
from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.v35_package import (
    CognitiveV35ModelPackage,
)
from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.v35_schemas import (
    V35_MODEL_VERSION,
    V35_VOICE_TASK_PROTOCOL,
    CognitiveV35InferRequest,
    CognitiveV35InferResponse,
    CognitiveV35TaskInput,
    CognitiveV35TaskQuality,
)


@dataclass(frozen=True)
class _EncodedTask:
    task_slot: int
    features: dict[str, np.ndarray]
    qualities: tuple[ModalityQuality, ModalityQuality, ModalityQuality]
    missing_mask: tuple[bool, bool, bool]
    used_modalities: tuple[str, ...]
    asr_status: str
    warnings: tuple[CognitiveWarning, ...]

    @property
    def status(self) -> str:
        if not self.used_modalities:
            return "insufficient_input"
        if self.used_modalities == V35_MODALITIES:
            return "completed"
        return "degraded"


class CognitiveV35InferenceRuntime:
    """Lazy V3.5 deployment runtime; all three task slots are inferred together."""

    def __init__(
        self,
        *,
        package: CognitiveV35ModelPackage,
        asr_client: ASRClientProtocol,
    ) -> None:
        self.package = package
        self.asr_client = asr_client

    def verify_package(self) -> None:
        try:
            metadata = self.package.verify()
            if metadata.model_version != V35_MODEL_VERSION:
                raise CognitivePackageError("registered V3.5 package version mismatch")
        except CognitivePackageError as exc:
            raise CognitiveAPIError(code="MODEL_ARTIFACT_UNAVAILABLE", errors=()) from exc

    def close(self) -> None:
        self.package.close()

    def infer(self, request: CognitiveV35InferRequest) -> CognitiveV35InferResponse:
        try:
            assets = self.package.load_assets()
            if assets.metadata.model_version != V35_MODEL_VERSION:
                raise CognitivePackageError("registered V3.5 package version mismatch")
        except CognitivePackageError as exc:
            raise CognitiveAPIError(
                code="MODEL_ARTIFACT_UNAVAILABLE",
                request_id=request.request_id,
            ) from exc

        encoded: list[_EncodedTask] = []
        for task in request.tasks:
            encoded.append(
                self._encode_task(
                    request=request,
                    task=task,
                    assets=assets,
                )
            )

        defaulted = "model_version" not in request.model_fields_set
        voice_adaptation = request.task_protocol == V35_VOICE_TASK_PROTOCOL
        adaptation_mode = (
            "voice_prompt_adaptation"
            if voice_adaptation
            else "standard_picture_description"
        )
        global_warnings = _aggregate_warnings(encoded)
        if defaulted:
            _add_warning(
                global_warnings,
                "model_defaulted",
                "model",
                "V3.5 subject model version was used by default",
            )
        if voice_adaptation:
            _add_warning(
                global_warnings,
                "voice_prompt_adaptation",
                "model",
                "Voice prompts replaced picture stimuli; picture-protocol validation metrics do not transfer to this adaptation",
            )
        task_quality = [_task_quality(item) for item in encoded]
        global_quality = _global_quality(encoded)
        aggregate_asr = _aggregate_asr_status(encoded)
        any_audio = any(not item.missing_mask[0] for item in encoded)
        if not any_audio:
            return CognitiveV35InferResponse(
                schema_version="cognitive_subject_infer_response_v1",
                request_id=request.request_id,
                task_protocol=request.task_protocol,
                stimulus_mode=request.stimulus_mode,
                adaptation_mode=adaptation_mode,
                status="insufficient_input",
                cognitive_clue_score=None,
                cognitive_clue_level=None,
                confidence=None,
                research_outputs=None,
                used_modalities=[],
                modality_quality=global_quality,
                task_quality=task_quality,
                asr_status=aggregate_asr,
                model_version=V35_MODEL_VERSION,
                warnings=_ordered_warnings(global_warnings),
            )

        try:
            import torch

            batch = build_v35_subject_batch(encoded)
            model_device = next(assets.models[0].parameters()).device
            batch = {
                "features": {
                    name: value.to(model_device)
                    for name, value in batch["features"].items()
                },
                "quality": batch["quality"].to(model_device),
                "missing_mask": batch["missing_mask"].to(model_device),
                "task_mask": batch["task_mask"].to(model_device),
            }
            raw_logits: list[float] = []
            with torch.inference_mode():
                for model in assets.models:
                    raw_logits.append(
                        float(model(batch)["subject_logit"][0].detach().cpu().item())
                    )
            raw_logit = sum(raw_logits) / len(raw_logits)
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
        except CognitiveAPIError:
            raise
        except Exception as exc:
            raise CognitiveAPIError(
                code="INTERNAL_ERROR",
                request_id=request.request_id,
            ) from exc

        score = _round_half_up(probability * 100.0, 1)
        certainty = 1.0 - _binary_entropy(probability)
        available_quality = [
            item.qualities[index].score
            for item in encoded
            for index in range(3)
            if not item.missing_mask[index]
        ]
        quality_mean = sum(available_quality) / len(available_quality)
        task_coverage = sum(bool(item.used_modalities) for item in encoded) / 3.0
        confidence = 0.5 * quality_mean + 0.3 * certainty + 0.2 * task_coverage
        all_complete = all(item.status == "completed" for item in encoded)
        if not all_complete:
            confidence = min(confidence, 0.650)
        if voice_adaptation:
            confidence = min(confidence, 0.600)
        confidence = _round_half_up(max(0.0, min(1.0, confidence)), 3)
        used = [
            modality
            for modality in V35_MODALITIES
            if any(modality in item.used_modalities for item in encoded)
        ]
        return CognitiveV35InferResponse(
            schema_version="cognitive_subject_infer_response_v1",
            request_id=request.request_id,
            task_protocol=request.task_protocol,
            stimulus_mode=request.stimulus_mode,
            adaptation_mode=adaptation_mode,
            status="completed" if all_complete else "degraded",
            cognitive_clue_score=score,
            cognitive_clue_level=_level_for_score(score),
            confidence=confidence,
            research_outputs=None,
            used_modalities=used,
            modality_quality=global_quality,
            task_quality=task_quality,
            asr_status=aggregate_asr,
            model_version=V35_MODEL_VERSION,
            warnings=_ordered_warnings(global_warnings),
        )

    def _encode_task(
        self,
        *,
        request: CognitiveV35InferRequest,
        task: CognitiveV35TaskInput,
        assets: Any,
    ) -> _EncodedTask:
        task_request_id = _task_request_id(request.request_id, task.task_slot)
        decoded_audio: DecodedAudioMedia | None = None
        if task.audio_input is not None:
            decoded_audio = decode_audio_media(
                task.audio_input,
                request_id=task_request_id,
            )
        face_context = (
            decode_face_media(task.face_input, request_id=task_request_id)
            if task.face_input is not None
            else _null_context()
        )
        with face_context as decoded_face:
            validate_media_alignment(
                capture_duration_ms=task.capture_window.duration_ms,
                audio_duration_ms=(
                    decoded_audio.decoded.duration_ms if decoded_audio is not None else None
                ),
                face_duration_ms=(
                    decoded_face.duration_ms if decoded_face is not None else None
                ),
                request_id=request.request_id,
            )
            return self._encode_decoded_task(
                task=task,
                task_request_id=task_request_id,
                assets=assets,
                decoded_audio=decoded_audio,
                decoded_face=decoded_face,
            )

    def _encode_decoded_task(
        self,
        *,
        task: CognitiveV35TaskInput,
        task_request_id: str,
        assets: Any,
        decoded_audio: DecodedAudioMedia | None,
        decoded_face: Any,
    ) -> _EncodedTask:
        warnings: list[CognitiveWarning] = []
        audio_quality = _missing_audio_quality()
        text_quality = _missing_text_quality()
        face_quality = _missing_face_quality()
        asr_status = "not_requested"
        normalized_text = ""
        transcript: ASRTranscript | None = None

        if decoded_audio is None:
            _add_warning(warnings, "missing_audio", "audio", "audio input was not provided")
            _add_warning(warnings, "missing_text", "text", "text requires audio and ASR")
        else:
            asr_success = False
            try:
                transcript = self.asr_client.transcribe(
                    request_id=task_request_id,
                    audio=decoded_audio,
                    declaration=task.audio_input,
                )
                asr_status = transcript.status
                asr_success = transcript.status in {
                    "completed",
                    "completed_empty_speech",
                }
            except ASRClientError:
                asr_status = "failed"
                _add_warning(
                    warnings,
                    "asr_failed",
                    "text",
                    "ASR service failed or timed out",
                )
            if asr_success and transcript is not None:
                audio_quality = compute_audio_quality(
                    duration_ms=decoded_audio.decoded.duration_ms,
                    speech_duration_ms=int(transcript.quality.speech_duration_ms),
                    decode_ok=True,
                )
                normalized_text, char_count = normalize_cognitive_text(transcript.text)
                text_quality = compute_text_quality(
                    char_count=char_count,
                    segments=transcript.segments,
                )
                if transcript.status == "completed_empty_speech" or not normalized_text:
                    _add_warning(
                        warnings,
                        "asr_empty_speech",
                        "text",
                        "ASR returned no usable speech",
                    )
                    _add_warning(
                        warnings,
                        "missing_text",
                        "text",
                        "usable ASR text was not available",
                    )
            else:
                audio_quality = compute_audio_quality(
                    duration_ms=decoded_audio.decoded.duration_ms,
                    speech_duration_ms=0,
                    decode_ok=True,
                )
                _add_warning(
                    warnings,
                    "missing_text",
                    "text",
                    "usable ASR text was not available",
                )
        if audio_quality.missing:
            _add_warning(
                warnings,
                "low_audio_quality",
                "audio",
                "audio quality is below the runtime threshold",
            )
        audio_available = not bool(audio_quality.missing)
        text_available = audio_available and not bool(text_quality.missing)
        if decoded_audio is not None and normalized_text and not text_available:
            _add_warning(
                warnings,
                "low_text_quality",
                "text",
                "normalized ASR text quality is below the runtime threshold",
            )

        face_available = False
        face_feature = np.zeros((V35_INPUT_DIMS["face"],), dtype=np.float32)
        if decoded_face is None:
            _add_warning(warnings, "missing_face", "face", "face video was not provided")
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
                if face_available:
                    face_feature = _feature_vector(
                        assets.face_encoder.encode(face_preprocessed.tensors),
                        V35_INPUT_DIMS["face"],
                    )
                else:
                    _add_warning(
                        warnings,
                        "low_face_quality",
                        "face",
                        "face quality is below the runtime threshold",
                    )
            except Exception:
                face_quality = _missing_face_quality()
                face_available = False
                _add_warning(
                    warnings,
                    "low_face_quality",
                    "face",
                    "face frames could not produce a usable feature",
                )

        audio_feature = np.zeros((V35_INPUT_DIMS["audio"],), dtype=np.float32)
        text_feature = np.zeros((V35_INPUT_DIMS["text"],), dtype=np.float32)
        try:
            if audio_available and decoded_audio is not None:
                audio_feature = _feature_vector(
                    assets.audio_encoder.encode(
                        decoded_audio.decoded.samples,
                        duration_ms=decoded_audio.decoded.duration_ms,
                    ),
                    V35_INPUT_DIMS["audio"],
                )
            if text_available:
                text_feature = _feature_vector(
                    assets.text_encoder.encode(normalized_text),
                    V35_INPUT_DIMS["text"],
                )
        except Exception as exc:
            raise CognitiveAPIError(
                code="INTERNAL_ERROR",
                request_id=task_request_id,
            ) from exc

        used = tuple(
            modality
            for modality, available in zip(
                V35_MODALITIES,
                (audio_available, text_available, face_available),
            )
            if available
        )
        return _EncodedTask(
            task_slot=task.task_slot,
            features={
                "audio": audio_feature,
                "text": text_feature,
                "face": face_feature,
            },
            qualities=(audio_quality, text_quality, face_quality),
            missing_mask=(not audio_available, not text_available, not face_available),
            used_modalities=used,
            asr_status=asr_status,
            warnings=tuple(_ordered_warnings(warnings)),
        )


def build_v35_subject_batch(encoded: list[_EncodedTask]) -> dict[str, Any]:
    if [item.task_slot for item in encoded] != [1, 2, 3]:
        raise ValueError("V3.5 encoded tasks must use slots 1, 2, 3 in order")
    import torch

    features = {
        modality: torch.from_numpy(
            np.stack([item.features[modality] for item in encoded], axis=0)
        ).unsqueeze(0)
        for modality in V35_MODALITIES
    }
    quality = torch.tensor(
        [
            [[value.score for value in item.qualities] for item in encoded]
        ],
        dtype=torch.float32,
    )
    missing_mask = torch.tensor(
        [[list(item.missing_mask) for item in encoded]],
        dtype=torch.bool,
    )
    task_mask = ~missing_mask.all(dim=-1)
    if not all(bool(torch.isfinite(value).all()) for value in features.values()):
        raise ValueError("V3.5 feature batch contains non-finite values")
    return {
        "features": features,
        "quality": quality,
        "missing_mask": missing_mask,
        "task_mask": task_mask,
    }


def _task_quality(item: _EncodedTask) -> CognitiveV35TaskQuality:
    return CognitiveV35TaskQuality(
        task_slot=item.task_slot,
        status=item.status,
        used_modalities=list(item.used_modalities),
        modality_quality=CognitiveModalityQuality(
            audio=_round_half_up(item.qualities[0].score, 3),
            text=_round_half_up(
                item.qualities[1].score if not item.missing_mask[1] else 0.0,
                3,
            ),
            face=_round_half_up(
                item.qualities[2].score if not item.missing_mask[2] else 0.0,
                3,
            ),
        ),
        asr_status=item.asr_status,
        warnings=list(item.warnings),
    )


def _global_quality(encoded: list[_EncodedTask]) -> CognitiveModalityQuality:
    values: list[float] = []
    for index in range(3):
        values.append(
            sum(
                item.qualities[index].score if not item.missing_mask[index] else 0.0
                for item in encoded
            )
            / len(encoded)
        )
    return CognitiveModalityQuality(
        audio=_round_half_up(values[0], 3),
        text=_round_half_up(values[1], 3),
        face=_round_half_up(values[2], 3),
    )


def _aggregate_warnings(encoded: list[_EncodedTask]) -> list[CognitiveWarning]:
    result: list[CognitiveWarning] = []
    for item in encoded:
        for warning in item.warnings:
            _add_warning(
                result,
                warning.code,
                warning.modality,
                f"one or more tasks: {warning.message}",
            )
    return result


def _aggregate_asr_status(encoded: list[_EncodedTask]) -> str:
    statuses = [item.asr_status for item in encoded]
    for candidate in (
        "failed",
        "model_unavailable",
        "unsupported_audio",
        "completed_empty_speech",
    ):
        if candidate in statuses:
            return candidate
    if statuses and all(value == "completed" for value in statuses):
        return "completed"
    if all(value == "not_requested" for value in statuses):
        return "not_requested"
    return "completed" if "completed" in statuses else "not_requested"


def _feature_vector(value: Any, expected_dim: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.shape != (expected_dim,) or not np.isfinite(array).all():
        raise ValueError(f"encoder output must be one finite {expected_dim}-D vector")
    return array


def _task_request_id(request_id: str, task_slot: int) -> str:
    return f"{request_id[:124]}-t{task_slot}"[:128]


def _missing_audio_quality() -> ModalityQuality:
    return compute_audio_quality(duration_ms=0, speech_duration_ms=0, decode_ok=False)


def _missing_text_quality() -> ModalityQuality:
    return compute_text_quality(char_count=0)


def _missing_face_quality() -> ModalityQuality:
    return compute_face_quality(valid_count=0, mean_luma=0.0)


def _add_warning(
    warnings: list[CognitiveWarning],
    code: str,
    modality: str,
    message: str,
) -> None:
    if any(item.code == code for item in warnings):
        return
    warnings.append(CognitiveWarning(code=code, modality=modality, message=message))


def _ordered_warnings(warnings: list[CognitiveWarning]) -> list[CognitiveWarning]:
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
        "voice_prompt_adaptation": 9,
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
    quantum = "1" if places == 0 else "1." + ("0" * places)
    return float(Decimal(str(value)).quantize(Decimal(quantum), rounding=ROUND_HALF_UP))


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


__all__ = ["CognitiveV35InferenceRuntime", "build_v35_subject_batch"]
