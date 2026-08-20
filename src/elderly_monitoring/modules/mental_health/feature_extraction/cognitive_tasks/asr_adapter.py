"""Thin adapter from ASRTranscript v1 to the cognitive text branch."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from elderly_monitoring.modules.asr.schemas import ASRTranscript
from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.preprocess import (
    normalize_cognitive_text,
)
from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.quality import (
    compute_text_quality,
)


@dataclass(frozen=True)
class CognitiveTextInput:
    text: str | None
    normalized_text: str
    segments: tuple[dict[str, Any], ...]
    text_available: bool
    char_count: int
    mean_confidence: float
    confidence_available: bool
    quality_weight: float
    asr_status: str
    asr_model_version: str
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_cognitive_text_input(
    transcript: ASRTranscript | Mapping[str, Any],
) -> CognitiveTextInput:
    """Create text input and a missing-modality mask without task features."""

    parsed = (
        transcript
        if isinstance(transcript, ASRTranscript)
        else ASRTranscript.model_validate(transcript)
    )
    normalized_text, char_count = normalize_cognitive_text(parsed.text)
    quality = compute_text_quality(char_count=char_count, segments=parsed.segments)
    mean_confidence = float(quality.metrics["mean_confidence"])
    confidence_available = bool(quality.metrics["confidence_available"])
    quality_weight = round(quality.score, 4)
    available = (
        parsed.status == "completed"
        and char_count >= 5
        and quality_weight >= 0.30
    )
    return CognitiveTextInput(
        text=normalized_text if available else None,
        normalized_text=normalized_text,
        segments=tuple(segment.model_dump(mode="json") for segment in parsed.segments),
        text_available=available,
        char_count=char_count,
        mean_confidence=round(mean_confidence, 4),
        confidence_available=confidence_available,
        quality_weight=quality_weight,
        asr_status=parsed.status,
        asr_model_version=parsed.model.version,
        warnings=tuple(parsed.warnings),
    )


__all__ = [
    "CognitiveTextInput",
    "build_cognitive_text_input",
    "normalize_cognitive_text",
]
