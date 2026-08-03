"""Stable Python entry points independent of the FunASR implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from elderly_monitoring.modules.asr.audio_decode import (
    AudioDecodeError,
    DecodedAudio,
    decode_audio_bytes,
    decode_audio_input,
    decode_audio_path,
)
from elderly_monitoring.modules.asr.engine import (
    ASRModelUnavailableError,
    EngineResult,
    get_default_engine,
)
from elderly_monitoring.modules.asr.quality import build_quality, build_segments
from elderly_monitoring.modules.asr.schemas import (
    ASRModelInfo,
    ASRQuality,
    ASRRequest,
    ASRTranscript,
)
from elderly_monitoring.modules.asr.settings import ASRSettings


class ASREngine(Protocol):
    def transcribe(self, audio: DecodedAudio) -> EngineResult: ...


def transcribe(
    request: ASRRequest,
    *,
    engine: ASREngine | None = None,
    settings: ASRSettings | None = None,
) -> ASRTranscript:
    runtime_settings = settings or ASRSettings.load()
    try:
        decoded = decode_audio_input(request.audio_input, runtime_settings)
    except AudioDecodeError:
        return _failure(request.request_id, "unsupported_audio", "audio_decode_failed")
    return _run(request, decoded, engine or get_default_engine(runtime_settings))


def transcribe_audio(
    audio_source: str | Path | bytes,
    *,
    request_id: str,
    engine: ASREngine | None = None,
    settings: ASRSettings | None = None,
) -> ASRTranscript:
    """Internal algorithm call accepting a local path or already available bytes."""

    runtime_settings = settings or ASRSettings.load()
    try:
        decoded = (
            decode_audio_bytes(audio_source, runtime_settings)
            if isinstance(audio_source, bytes)
            else decode_audio_path(audio_source, runtime_settings)
        )
    except AudioDecodeError:
        return _failure(request_id, "unsupported_audio", "audio_decode_failed")
    request = ASRRequest(
        request_id=request_id,
        audio_input={
            "source_type": "base64",
            "source": "aW50ZXJuYWw=",
            "format": "wav",
            "sample_rate": None,
            "channels": None,
        },
    )
    return _run(request, decoded, engine or get_default_engine(runtime_settings))


def _run(request: ASRRequest, decoded: DecodedAudio, engine: ASREngine) -> ASRTranscript:
    try:
        result = engine.transcribe(decoded)
    except ASRModelUnavailableError:
        return _failure(request.request_id, "model_unavailable", "asr_model_unavailable", decoded.duration_ms)
    except Exception:
        return _failure(request.request_id, "failed", "asr_inference_failed", decoded.duration_ms)
    segments, warnings = build_segments(
        result,
        decoded.duration_ms,
        return_timestamps=request.return_timestamps,
    )
    quality = build_quality(
        result,
        decoded.duration_ms,
        vad_enabled=request.enable_vad,
    )
    return ASRTranscript(
        request_id=request.request_id,
        status="completed" if result.text else "completed_empty_speech",
        text=result.text,
        segments=segments,
        quality=quality,
        warnings=warnings,
    )


def _failure(
    request_id: str,
    status: str,
    warning: str,
    duration_ms: int = 0,
) -> ASRTranscript:
    return ASRTranscript(
        request_id=request_id,
        status=status,
        quality=ASRQuality(
            audio_duration_ms=duration_ms,
            speech_duration_ms=0,
            speech_ratio=0.0,
            mean_confidence=None,
            decode_status="failed" if status == "unsupported_audio" else "ok",
            vad_status="failed",
        ),
        model=ASRModelInfo(),
        warnings=[warning],
    )
