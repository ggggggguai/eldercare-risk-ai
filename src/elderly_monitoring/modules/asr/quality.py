"""Normalize FunASR timestamps and derive non-model quality metadata."""

from __future__ import annotations

import math
from typing import Any, Iterable

from elderly_monitoring.modules.asr.engine import EngineResult
from elderly_monitoring.modules.asr.schemas import ASRQuality, ASRSegment


def build_segments(
    result: EngineResult,
    audio_duration_ms: int,
    *,
    return_timestamps: bool = True,
) -> tuple[list[ASRSegment], list[str]]:
    warnings: list[str] = []
    segments: list[ASRSegment] = []
    for index, item in enumerate(result.sentence_info, start=1):
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        raw_start = _as_int(item.get("start"), 0)
        raw_end = _as_int(item.get("end"), audio_duration_ms)
        start_ms, end_ms = clamp_interval(raw_start, raw_end, audio_duration_ms)
        if (start_ms, end_ms) != (raw_start, raw_end):
            warnings.append("timestamp_clamped")
        segments.append(
            ASRSegment(
                segment_id=f"seg-{index:03d}",
                start_ms=start_ms if return_timestamps else 0,
                end_ms=end_ms if return_timestamps else audio_duration_ms,
                text=text,
                confidence=_as_confidence(item.get("confidence")),
            )
        )
    if not segments and result.text:
        start_ms, end_ms = interval_bounds(result.timestamps, audio_duration_ms)
        segments.append(
            ASRSegment(
                segment_id="seg-001",
                start_ms=start_ms if return_timestamps else 0,
                end_ms=end_ms if return_timestamps else audio_duration_ms,
                text=result.text,
                confidence=None,
            )
        )
    if result.text and not any(segment.confidence is not None for segment in segments):
        warnings.append("confidence_unavailable")
        if result.timestamps:
            warnings.append("speech_duration_estimated_from_timestamps")
    return segments, list(dict.fromkeys(warnings))


def build_quality(
    result: EngineResult,
    audio_duration_ms: int,
    *,
    vad_enabled: bool,
) -> ASRQuality:
    speech_duration_ms = union_duration(result.timestamps, audio_duration_ms)
    if not result.text:
        vad_status = "empty" if vad_enabled else "disabled"
    else:
        vad_status = "ok" if vad_enabled else "disabled"
    ratio = 0.0 if audio_duration_ms <= 0 else speech_duration_ms / audio_duration_ms
    confidences = [
        confidence
        for item in result.sentence_info
        if isinstance(item, dict)
        and str(item.get("text") or "").strip()
        and (confidence := _as_confidence(item.get("confidence"))) is not None
    ]
    return ASRQuality(
        audio_duration_ms=audio_duration_ms,
        speech_duration_ms=speech_duration_ms,
        speech_ratio=round(max(0.0, min(1.0, ratio)), 4),
        mean_confidence=(
            round(sum(confidences) / len(confidences), 4)
            if confidences
            else None
        ),
        decode_status="ok",
        vad_status=vad_status,
    )


def clamp_interval(start_ms: int, end_ms: int, duration_ms: int) -> tuple[int, int]:
    duration_ms = max(0, int(duration_ms))
    start = max(0, min(int(start_ms), duration_ms))
    end = max(0, min(int(end_ms), duration_ms))
    if end < start:
        start, end = end, start
    return start, end


def interval_bounds(timestamps: Iterable[Any], duration_ms: int) -> tuple[int, int]:
    intervals = _valid_intervals(timestamps, duration_ms)
    if not intervals:
        return 0, duration_ms
    return intervals[0][0], intervals[-1][1]


def union_duration(timestamps: Iterable[Any], duration_ms: int) -> int:
    intervals = _valid_intervals(timestamps, duration_ms)
    if not intervals:
        return 0
    total = 0
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return total + current_end - current_start


def _valid_intervals(timestamps: Iterable[Any], duration_ms: int) -> list[tuple[int, int]]:
    intervals = []
    for item in timestamps:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        start, end = clamp_interval(_as_int(item[0], 0), _as_int(item[1], 0), duration_ms)
        if end > start:
            intervals.append((start, end))
    return sorted(intervals)


def _as_int(value: Any, fallback: int) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return fallback


def _as_confidence(value: Any) -> float | None:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        return None
    return confidence
