from __future__ import annotations

from scripts.evaluate.evaluate_fall_nearfall_v1 import (
    _grid,
    _merge_fall_windows,
    _near_fall_predictions,
    _select_single_subject_stream,
    _video_presence_metrics,
)
from scripts.prepare.prepare_fall_nearfall_adaptation import _partition_for_video


def test_grid_includes_exact_end_without_duplicate() -> None:
    assert _grid(4.0, 5.0, 0.5) == [4.0, 4.5, 5.0]
    assert _grid(4.0, 5.1, 0.5)[-1] == 5.1


def test_fall_positive_windows_are_merged_by_causal_stride() -> None:
    windows = [
        {"cutoff_time": 4.0, "window_start": 0.0, "score": 0.8, "onset_time": 3.0},
        {"cutoff_time": 4.5, "window_start": 0.5, "score": 0.7, "onset_time": 3.2},
        {"cutoff_time": 7.5, "window_start": 3.5, "score": 0.9, "onset_time": 6.5},
    ]
    events = _merge_fall_windows(
        windows,
        video_id="video",
        person_id="person",
        track_id="1",
    )
    assert len(events) == 2
    assert events[0]["window_count"] == 2
    assert events[0]["score"] == 0.8
    assert events[1]["start_time"] == 3.5


def test_near_fall_predictions_keep_only_emitted_threshold() -> None:
    predictions = _near_fall_predictions(
        [
            {
                "start_time": 1.0,
                "end_time": 2.0,
                "near_fall_event_score": 0.49,
                "model_version": "near-fall-rule-v0.1",
            },
            {
                "start_time": 2.0,
                "end_time": 3.0,
                "near_fall_event_score": 0.5,
                "model_version": "near-fall-rule-v0.1",
            },
        ],
        video_id="video",
        config_hash="a" * 64,
    )
    assert len(predictions) == 1
    assert predictions[0]["task_type"] == "near_fall_event"
    assert predictions[0]["status"] == "emitted"


def test_single_subject_stream_drops_low_confidence_duplicate_track() -> None:
    selected, diagnostics = _select_single_subject_stream(
        [
            {
                "frame_id": 0,
                "timestamp_sec": 0.0,
                "person_id": "video_001",
                "track_id": 1,
                "pose_confidence": 0.91,
                "core_keypoint_quality": 0.99,
                "bbox": [0.2, 0.1, 0.5, 0.8],
            },
            {
                "frame_id": 0,
                "timestamp_sec": 0.0,
                "person_id": "video_002",
                "track_id": 2,
                "pose_confidence": 0.41,
                "core_keypoint_quality": 0.86,
                "bbox": [0.9, 0.4, 1.0, 0.75],
            },
        ],
        video_id="video",
    )

    assert len(selected) == 1
    assert selected[0]["source_track_id"] == 1
    assert selected[0]["track_id"] == "primary"
    assert diagnostics["source_track_count"] == 2


def test_video_presence_reports_first_and_any_interval_hits() -> None:
    result = _video_presence_metrics(
        [
            {
                "video_id": "fall",
                "task_type": "fall_event",
                "event_type": "fall",
                "start_time": 2.0,
                "onset_time": 2.0,
                "end_time": 4.0,
            }
        ],
        [
            {"video_id": "fall", "alert_time": 1.0, "onset_time": 1.0},
            {"video_id": "fall", "alert_time": 3.0, "onset_time": 3.0},
        ],
        evaluated_video_ids={"fall", "negative"},
    )
    assert result["any_alert_inside_annotated_interval_count"] == 1
    assert result["first_alert_inside_annotated_interval_count"] == 0
    assert result["duplicate_prediction_count"] == 1


def test_adaptation_partition_keeps_holdout_video_isolated() -> None:
    assert _partition_for_video({"video_id": "fall_f05_dark_1", "scene_region": "dark"}) == "holdout"
    assert _partition_for_video({"video_id": "fall_n06_base_1", "scene_region": "base"}) == "validation"
    assert _partition_for_video({"video_id": "fall_f04_bed_1", "scene_region": "bed"}) == "validation"
    assert _partition_for_video({"video_id": "fall_f03_dark_1", "scene_region": "dark"}) == "train"
