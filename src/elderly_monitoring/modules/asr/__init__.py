"""Stable ASR facade used by cognitive and other algorithm modules."""

from elderly_monitoring.modules.asr.facade import transcribe, transcribe_audio
from elderly_monitoring.modules.asr.engine import prewarm_default_engine
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
