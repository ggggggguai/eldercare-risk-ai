from elderly_monitoring.modules.asr.schemas import ASRQuality, ASRSegment, ASRTranscript
from elderly_monitoring.modules.mental_health.feature_extraction.cognitive_tasks import (
    build_cognitive_text_input,
)


def _transcript(status: str, text: str, speech_ratio: float) -> ASRTranscript:
    return ASRTranscript(
        request_id="asr-cognitive-001",
        status=status,
        text=text,
        quality=ASRQuality(
            audio_duration_ms=1000,
            speech_duration_ms=int(1000 * speech_ratio),
            speech_ratio=speech_ratio,
            decode_status="ok",
            vad_status="ok" if text else "empty",
        ),
    )


def test_adapter_exposes_text_and_quality_mask_without_task_fields() -> None:
    result = build_cognitive_text_input(
        _transcript("completed", "老人回答了123个问题，状态保持稳定。" * 6, 0.72)
    )
    payload = result.to_dict()
    assert result.text_available is True
    assert "<NUM>" in result.normalized_text
    assert result.char_count >= 80
    assert result.mean_confidence == 0.0
    assert result.confidence_available is False
    assert result.quality_weight == 1.0
    assert "task_id" not in payload
    assert "prompt" not in payload


def test_adapter_marks_text_modality_missing_after_asr_failure() -> None:
    result = build_cognitive_text_input(_transcript("failed", "", 0.0))
    assert result.text is None
    assert result.text_available is False
    assert result.quality_weight == 0.0


def test_adapter_applies_v3_3_normalization_and_short_text_gate() -> None:
    transcript = _transcript("completed", "ＡＢＣ  123@@，测试。", 0.8)
    result = build_cognitive_text_input(transcript)
    assert result.text is None
    assert result.normalized_text == "ABC <NUM>测试。"
    assert result.char_count == 6
    assert result.quality_weight == 0.075
    assert result.text_available is False
    assert result.confidence_available is False


def test_adapter_uses_finite_segment_confidence_in_v3_3_q_text() -> None:
    transcript = _transcript("completed", "老人" * 20, 0.8)
    transcript.segments = [
        ASRSegment(
            segment_id="seg-001",
            start_ms=0,
            end_ms=1000,
            text="老人" * 20,
            confidence=0.8,
        )
    ]
    result = build_cognitive_text_input(transcript)
    assert result.char_count == 40
    assert result.mean_confidence == 0.8
    assert result.quality_weight == 0.59
    assert result.confidence_available is True
    assert result.text_available is True
