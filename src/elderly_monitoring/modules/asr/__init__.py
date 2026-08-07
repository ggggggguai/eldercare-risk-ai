"""Stable ASR schemas with lazy engine imports.

Importing a boundary schema must not initialize the heavyweight FunASR runtime.
The standalone ASR service imports the facade explicitly when it is launched.
"""
from elderly_monitoring.modules.asr.schemas import (
    ASRModelInfo,
    ASRQuality,
    ASRRequest,
    ASRSegment,
    ASRTranscript,
    AudioInput,
)

__all__ = [
    "ASRModelInfo",
    "ASRQuality",
    "ASRRequest",
    "ASRSegment",
    "ASRTranscript",
    "AudioInput",
    "transcribe",
    "transcribe_audio",
    "prewarm_default_engine",
]


def __getattr__(name: str):
    if name in {"transcribe", "transcribe_audio"}:
        from elderly_monitoring.modules.asr.facade import transcribe, transcribe_audio

        return {"transcribe": transcribe, "transcribe_audio": transcribe_audio}[name]
    if name == "prewarm_default_engine":
        from elderly_monitoring.modules.asr.engine import prewarm_default_engine

        return prewarm_default_engine
    raise AttributeError(name)
