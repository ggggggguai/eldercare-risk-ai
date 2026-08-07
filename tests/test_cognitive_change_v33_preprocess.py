from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.preprocess import (
    audio_window_starts,
    expand_relative_box,
    make_audio_windows,
    normalize_cognitive_text,
    preprocess_face_frames,
    sample_frame_positions,
)
from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.quality import (
    compute_audio_quality,
    compute_face_quality,
    compute_text_quality,
    merge_vad_intervals,
    vad_union_ms,
)


def test_text_normalization_follows_frozen_order_and_drops_ascii_punctuation() -> None:
    normalized, char_count = normalize_cognitive_text(" ＡＢＣ\t123,测试!；保留。 ")
    assert normalized == "ABC <NUM>测试保留。"
    assert char_count == 8


def test_text_normalization_counts_each_number_run_as_one_literal_marker() -> None:
    normalized, char_count = normalize_cognitive_text("12年有345人")
    assert normalized == "<NUM>年有<NUM>人"
    assert char_count == 5


@pytest.mark.parametrize("source", ["", "   ", "，！？；：", "老人"])
def test_empty_punctuation_only_and_short_text_stay_missing(source: str) -> None:
    _normalized, char_count = normalize_cognitive_text(source)
    quality = compute_text_quality(char_count=char_count, segments=())
    assert char_count < 5
    assert quality.missing == 1


@pytest.mark.parametrize(
    ("duration_ms", "expected"),
    [
        (3_000, (0,)),
        (20_000, (0,)),
        (20_001, (0, 10_000)),
        (25_000, (0, 10_000)),
        (30_000, (0, 10_000)),
        (60_000, (0, 10_000, 20_000, 30_000, 40_000)),
    ],
)
def test_audio_window_starts_match_v3_3(duration_ms: int, expected: tuple[int, ...]) -> None:
    assert audio_window_starts(duration_ms) == expected


def test_audio_windows_pad_right_and_weights_cover_duration_once() -> None:
    samples = np.ones(25 * 16_000, dtype=np.float32)
    windows = make_audio_windows(samples, duration_ms=25_000)
    assert [window.valid_samples for window in windows] == [320_000, 240_000]
    assert [window.weight_ms for window in windows] == [10_000, 15_000]
    assert sum(window.weight_ms for window in windows) == 25_000
    assert windows[1].waveform.shape == (320_000,)
    assert np.all(windows[1].waveform[240_000:] == 0)


def test_vad_intervals_are_clamped_merged_and_unioned() -> None:
    intervals = ((-100, 1000), (900, 2000), (3000, 4000), (5000, 4500), (9000, 12000))
    assert merge_vad_intervals(intervals, 10_000) == (
        (0, 2000),
        (3000, 4000),
        (4500, 5000),
        (9000, 10_000),
    )
    assert vad_union_ms(intervals, 10_000) == 4500


def test_frozen_quality_formulas_and_missing_thresholds() -> None:
    audio = compute_audio_quality(duration_ms=10_000, speech_duration_ms=6_000)
    assert audio.score == pytest.approx(1.0)
    assert audio.missing == 0

    short_audio = compute_audio_quality(duration_ms=3_000, speech_duration_ms=300)
    assert short_audio.score == pytest.approx(0.2)
    assert short_audio.missing == 1

    text = compute_text_quality(
        char_count=80,
        segments=({"text": "有效", "confidence": 0.5}, {"text": "", "confidence": 1.0}),
    )
    assert text.score == pytest.approx(0.85)
    assert text.missing == 0
    no_confidence = compute_text_quality(char_count=5, segments=())
    assert no_confidence.score == pytest.approx(0.0625)
    assert no_confidence.missing == 1
    assert no_confidence.metrics["confidence_available"] is False

    length_only = compute_text_quality(char_count=24, segments=())
    assert length_only.score == pytest.approx(0.30)
    assert length_only.missing == 0

    face = compute_face_quality(valid_count=16, mean_luma=0.5)
    assert face.score == pytest.approx(1.0)
    assert face.missing == 0
    sparse_face = compute_face_quality(valid_count=3, mean_luma=0.5)
    assert sparse_face.score > 0.4
    assert sparse_face.missing == 1


def test_face_sampling_allows_duplicate_positions() -> None:
    assert sample_frame_positions(1) == (0,) * 16
    assert sample_frame_positions(16) == tuple(range(16))
    positions = sample_frame_positions(4)
    assert len(positions) == 16
    assert positions[0] == 0 and positions[-1] == 3
    assert len(set(positions)) == 4


def test_face_box_expansion_clips_to_image() -> None:
    assert expand_relative_box(
        x_min=-0.1,
        y_min=0.1,
        width=0.5,
        height=0.5,
        image_width=100,
        image_height=80,
    ) == (0, 0, 50, 56)


class _SingleFaceDetector:
    def process(self, _image: np.ndarray) -> SimpleNamespace:
        relative_box = SimpleNamespace(xmin=0.2, ymin=0.2, width=0.5, height=0.5)
        detection = SimpleNamespace(
            location_data=SimpleNamespace(relative_bounding_box=relative_box)
        )
        return SimpleNamespace(detections=[detection])


class _NoFaceDetector:
    def process(self, _image: np.ndarray) -> SimpleNamespace:
        return SimpleNamespace(detections=[])


class _MultipleFaceDetector:
    def process(self, _image: np.ndarray) -> SimpleNamespace:
        detection = _SingleFaceDetector().process(_image).detections[0]
        return SimpleNamespace(detections=[detection, detection])


class _ThreeValidDetections:
    def __init__(self) -> None:
        self.calls = 0

    def process(self, image: np.ndarray) -> SimpleNamespace:
        self.calls += 1
        if self.calls <= 3:
            return _SingleFaceDetector().process(image)
        return SimpleNamespace(detections=[])


def test_face_preprocess_processes_each_duplicate_sample_position(tmp_path) -> None:
    frame = np.full((64, 64, 3), 128, dtype=np.uint8)
    frame_path = tmp_path / "frame_1.jpg"
    assert cv2.imwrite(str(frame_path), frame)
    result = preprocess_face_frames([frame_path], _SingleFaceDetector())
    assert result.sampled_positions == (0,) * 16
    assert result.valid_count == 16
    assert all(tensor.shape == (3, 224, 224) for tensor in result.tensors)
    assert result.mean_luma == pytest.approx(128 / 255.0, abs=1e-3)


def test_face_preprocess_handles_zero_frames_zero_faces_and_multiple_faces(tmp_path) -> None:
    assert preprocess_face_frames([], _NoFaceDetector()).valid_count == 0
    frame = np.full((64, 64, 3), 128, dtype=np.uint8)
    frame_path = tmp_path / "frame_1.jpg"
    assert cv2.imwrite(str(frame_path), frame)
    assert preprocess_face_frames([frame_path], _NoFaceDetector()).valid_count == 0
    assert preprocess_face_frames([frame_path], _MultipleFaceDetector()).valid_count == 0


def test_face_preprocess_and_quality_mark_fewer_than_four_faces_missing(tmp_path) -> None:
    frame = np.full((64, 64, 3), 128, dtype=np.uint8)
    frame_path = tmp_path / "frame_1.jpg"
    assert cv2.imwrite(str(frame_path), frame)
    result = preprocess_face_frames([frame_path], _ThreeValidDetections())
    quality = compute_face_quality(valid_count=result.valid_count, mean_luma=result.mean_luma)
    assert result.valid_count == 3
    assert quality.missing == 1


def test_face_brightness_quality_is_symmetric_at_dark_and_bright_extremes() -> None:
    dark = compute_face_quality(valid_count=16, mean_luma=0.0)
    bright = compute_face_quality(valid_count=16, mean_luma=1.0)
    assert dark.score == pytest.approx(0.7)
    assert bright.score == pytest.approx(0.7)
    assert dark.missing == bright.missing == 0
