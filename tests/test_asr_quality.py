from elderly_monitoring.modules.asr.engine import EngineResult
from elderly_monitoring.modules.asr.quality import build_quality, build_segments, union_duration


def test_timestamp_clamping_and_confidence_unavailable() -> None:
    result = EngineResult(
        text="第一句。第二句。",
        sentence_info=[
            {"text": "第一句。", "start": -30, "end": 500},
            {"text": "第二句。", "start": 500, "end": 1030},
        ],
        timestamps=[[-30, 100], [100, 500], [700, 1030]],
    )
    segments, warnings = build_segments(result, 1000)
    assert [(item.start_ms, item.end_ms) for item in segments] == [(0, 500), (500, 1000)]
    assert all(item.confidence is None for item in segments)
    assert warnings == [
        "timestamp_clamped",
        "confidence_unavailable",
        "speech_duration_estimated_from_timestamps",
    ]


def test_quality_uses_union_of_word_timestamps_as_speech_duration() -> None:
    result = EngineResult(text="有语音", timestamps=[[0, 200], [150, 400], [700, 900]])
    quality = build_quality(result, 1000, vad_enabled=True)
    assert union_duration(result.timestamps, 1000) == 600
    assert quality.speech_duration_ms == 600
    assert quality.speech_ratio == 0.6
    assert quality.mean_confidence is None


def test_explicit_finite_sentence_confidence_is_preserved() -> None:
    result = EngineResult(
        text="两句文本。",
        sentence_info=[
            {"text": "第一句。", "start": 0, "end": 400, "confidence": 0.8},
            {"text": "第二句。", "start": 400, "end": 800, "confidence": 0.6},
        ],
        timestamps=[[0, 800]],
    )
    segments, warnings = build_segments(result, 1000)
    quality = build_quality(result, 1000, vad_enabled=True)
    assert [segment.confidence for segment in segments] == [0.8, 0.6]
    assert quality.mean_confidence == 0.7
    assert "confidence_unavailable" not in warnings
