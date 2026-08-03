"""Thin adapter from ASRTranscript v1 to the cognitive text branch."""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from elderly_monitoring.modules.asr.schemas import ASRTranscript


@dataclass(frozen=True)
class CognitiveTextInput:
    text: str | None
    normalized_text: str
    segments: tuple[dict[str, Any], ...]
    text_available: bool
    char_count: int
    mean_confidence: float
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
    confidences = [
        float(segment.confidence)
        for segment in parsed.segments
        if segment.text.strip()
        and segment.confidence is not None
        and math.isfinite(float(segment.confidence))
    ]
    mean_confidence = (
        sum(confidences) / len(confidences)
        if confidences
        else 0.0
    )
    length_score = _clip(char_count / 80.0)
    quality_weight = round(
        0.70 * length_score + 0.30 * _clip(mean_confidence),
        4,
    )
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
        quality_weight=quality_weight,
        asr_status=parsed.status,
        asr_model_version=parsed.model.version,
        warnings=tuple(parsed.warnings),
    )


def normalize_cognitive_text(text: str) -> tuple[str, int]:
    """Apply the frozen V3.3 text normalization and character counting rules."""

    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.translate(
        str.maketrans({",": "，", "!": "！", "?": "？", ";": "；", ":": "："})
    )
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = re.sub(r"[0-9]+", "<NUM>", normalized)
    kept: list[str] = []
    char_count = 0
    index = 0
    while index < len(normalized):
        if normalized.startswith("<NUM>", index):
            kept.append("<NUM>")
            char_count += 1
            index += len("<NUM>")
            continue
        character = normalized[index]
        if _is_chinese(character) or character.isascii() and character.isalpha():
            kept.append(character)
            char_count += 1
        elif character == " " or character in "，。！？；：":
            kept.append(character)
        index += 1
    cleaned = re.sub(r" +", " ", "".join(kept)).strip()
    return cleaned, char_count


def _is_chinese(character: str) -> bool:
    return "\u3400" <= character <= "\u4dbf" or "\u4e00" <= character <= "\u9fff"


def _clip(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


__all__ = [
    "CognitiveTextInput",
    "build_cognitive_text_input",
    "normalize_cognitive_text",
]
