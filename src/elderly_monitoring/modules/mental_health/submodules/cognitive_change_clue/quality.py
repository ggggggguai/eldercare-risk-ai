"""V3.3 modality quality formulas and missing masks."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence


AUDIO_MISSING_THRESHOLD = 0.30
TEXT_MISSING_THRESHOLD = 0.30
FACE_MISSING_THRESHOLD = 0.40
TEXT_QUALITY_VERSION = "cognitive_text_quality_v3.3.1"


@dataclass(frozen=True)
class ModalityQuality:
    score: float
    missing: int
    metrics: Mapping[str, float | int | bool | str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def merge_vad_intervals(
    intervals_ms: Iterable[Sequence[float | int]],
    duration_ms: int,
) -> tuple[tuple[int, int], ...]:
    """Clamp, sort, and merge VAD intervals within the decoded duration."""

    bounded: list[tuple[int, int]] = []
    upper = max(0, int(duration_ms))
    for interval in intervals_ms:
        if len(interval) < 2:
            continue
        try:
            start = int(round(float(interval[0])))
            end = int(round(float(interval[1])))
        except (TypeError, ValueError):
            continue
        start = max(0, min(start, upper))
        end = max(0, min(end, upper))
        if end < start:
            start, end = end, start
        if end > start:
            bounded.append((start, end))
    if not bounded:
        return ()
    bounded.sort()
    merged: list[tuple[int, int]] = [bounded[0]]
    for start, end in bounded[1:]:
        previous_start, previous_end = merged[-1]
        if start <= previous_end:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return tuple(merged)


def vad_union_ms(
    intervals_ms: Iterable[Sequence[float | int]],
    duration_ms: int,
) -> int:
    return sum(end - start for start, end in merge_vad_intervals(intervals_ms, duration_ms))


def compute_audio_quality(
    *,
    duration_ms: int,
    speech_duration_ms: int,
    decode_ok: bool = True,
) -> ModalityQuality:
    duration_ms = max(0, int(duration_ms))
    speech_duration_ms = max(0, min(int(speech_duration_ms), duration_ms))
    duration_sec = duration_ms / 1000.0
    duration_score = clip01((duration_sec - 3.0) / 7.0)
    speech_ratio = 0.0 if duration_ms == 0 else speech_duration_ms / duration_ms
    speech_score = clip01((speech_ratio - 0.10) / 0.50)
    score = (
        0.60 * speech_score
        + 0.20 * duration_score
        + 0.20 * float(bool(decode_ok))
    )
    return ModalityQuality(
        score=clip01(score),
        missing=int(not decode_ok or score < AUDIO_MISSING_THRESHOLD),
        metrics={
            "duration_ms": duration_ms,
            "speech_duration_ms": speech_duration_ms,
            "speech_ratio": speech_ratio,
            "duration_score": duration_score,
            "speech_score": speech_score,
            "decode_ok": bool(decode_ok),
        },
    )


def finite_segment_confidences(segments: Iterable[Any]) -> tuple[float, ...]:
    confidences: list[float] = []
    for segment in segments:
        if isinstance(segment, Mapping):
            text = str(segment.get("text") or "")
            value = segment.get("confidence")
        else:
            text = str(getattr(segment, "text", "") or "")
            value = getattr(segment, "confidence", None)
        if not text.strip() or value is None:
            continue
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(confidence) and 0.0 <= confidence <= 1.0:
            confidences.append(confidence)
    return tuple(confidences)


def compute_text_quality(
    *,
    char_count: int,
    segments: Iterable[Any] = (),
) -> ModalityQuality:
    char_count = max(0, int(char_count))
    confidences = finite_segment_confidences(segments)
    length_score = clip01(char_count / 80.0)
    confidence_available = bool(confidences)
    mean_confidence = sum(confidences) / len(confidences) if confidence_available else 0.0
    if confidence_available:
        score = 0.70 * length_score + 0.30 * clip01(mean_confidence)
    else:
        # Missing confidence is an unavailable measurement, not a zero-quality score.
        score = length_score
    return ModalityQuality(
        score=clip01(score),
        missing=int(char_count < 5 or score < TEXT_MISSING_THRESHOLD),
        metrics={
            "char_count": char_count,
            "length_score": length_score,
            "mean_confidence": mean_confidence,
            "confidence_count": len(confidences),
            "confidence_available": confidence_available,
            "quality_version": TEXT_QUALITY_VERSION,
        },
    )


def compute_face_quality(
    *,
    valid_count: int,
    mean_luma: float,
    sample_count: int = 16,
) -> ModalityQuality:
    sample_count = int(sample_count)
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    valid_count = max(0, min(int(valid_count), sample_count))
    mean_luma = clip01(mean_luma) if valid_count else 0.0
    valid_ratio = valid_count / sample_count
    brightness_score = clip01(1.0 - abs(mean_luma - 0.50) / 0.50)
    score = 0.70 * valid_ratio + 0.30 * brightness_score
    return ModalityQuality(
        score=clip01(score),
        missing=int(valid_count < 4 or score < FACE_MISSING_THRESHOLD),
        metrics={
            "valid_count": valid_count,
            "sample_count": sample_count,
            "valid_face_frame_ratio": valid_ratio,
            "mean_luma": mean_luma,
            "brightness_score": brightness_score,
        },
    )


__all__ = [
    "AUDIO_MISSING_THRESHOLD",
    "FACE_MISSING_THRESHOLD",
    "TEXT_MISSING_THRESHOLD",
    "TEXT_QUALITY_VERSION",
    "ModalityQuality",
    "clip01",
    "compute_audio_quality",
    "compute_face_quality",
    "compute_text_quality",
    "finite_segment_confidences",
    "merge_vad_intervals",
    "vad_union_ms",
]
