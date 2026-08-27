from __future__ import annotations

from scripts.evaluate.evaluate_fall_nearfall_v1 import (
    _grid,
    _effective_evaluation_config,
    _merge_fall_windows,
    _near_fall_predictions,
    _select_near_fall_stream,
    _suppress_fall_event_realerts,
    _suppress_near_fall_on_fall,
    _suppress_near_fall_realerts,
    _select_single_subject_stream,
    _video_presence_metrics,
)
from scripts.evaluate.build_fall_self_eval_bundle import _operational_alarm_summary
from scripts.prepare.prepare_fall_nearfall_adaptation import (
    _build_governance_rows,
    _partition_for_video,
)


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


def test_fall_event_end_cap_limits_sustained_positive_tail() -> None:
    events = _merge_fall_windows(
        [
            {"cutoff_time": 4.0, "window_start": 0.0, "score": 0.8, "onset_time": 3.0},
            {"cutoff_time": 5.5, "window_start": 1.5, "score": 0.9, "onset_time": 4.5},
            {"cutoff_time": 7.0, "window_start": 3.0, "score": 0.85, "onset_time": 5.5},
        ],
        video_id="video",
        person_id="person",
        track_id="1",
        end_cap_sec=2.5,
    )
    assert len(events) == 1
    assert events[0]["alert_time"] == 4.0
    assert events[0]["end_time"] == 6.5


def test_fall_event_pre_alert_cap_keeps_predicted_onset_inside_interval() -> None:
    events = _merge_fall_windows(
        [{"cutoff_time": 4.0, "window_start": 0.0, "score": 0.8, "onset_time": 2.25}],
        video_id="video",
        person_id="person",
        track_id="1",
        end_cap_sec=2.5,
        pre_alert_sec=1.0,
    )
    assert events[0]["start_time"] == 2.25
    assert events[0]["onset_time"] == 2.25


def test_fall_event_cooldown_suppresses_repeat_alerts_but_allows_a_new_incident() -> None:
    events = _suppress_fall_event_realerts(
        [
            {"video_id": "video", "person_id": "person", "track_id": "1", "alert_time": 4.0},
            {"video_id": "video", "person_id": "person", "track_id": "1", "alert_time": 11.0},
            {"video_id": "video", "person_id": "person", "track_id": "1", "alert_time": 15.0},
        ],
        cooldown_sec=10.0,
    )

    assert [event["alert_time"] for event in events] == [4.0, 15.0]


def test_near_fall_cooldown_suppresses_repeat_alerts_per_video() -> None:
    predictions = _suppress_near_fall_realerts(
        [
            {"video_id": "a", "alert_time": 2.0},
            {"video_id": "a", "alert_time": 5.0},
            {"video_id": "a", "alert_time": 18.0},
            {"video_id": "b", "alert_time": 4.0},
        ],
        cooldown_sec=15.0,
    )

    assert [(row["video_id"], row["alert_time"]) for row in predictions] == [
        ("a", 2.0),
        ("a", 18.0),
        ("b", 4.0),
    ]


def test_near_fall_event_hold_extends_interval_without_delaying_alert() -> None:
    predictions = _near_fall_predictions(
        [
            {
                "start_time": 1.0,
                "end_time": 2.5,
                "near_fall_event_score": 0.8,
            }
        ],
        video_id="video",
        config_hash="0" * 64,
        threshold=0.5,
        hold_sec=0.75,
    )

    assert predictions[0]["alert_time"] == 2.5
    assert predictions[0]["end_time"] == 3.25


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


def test_effective_evaluation_config_uses_runtime_threshold() -> None:
    source = {"score_threshold": 0.5, "task_type": "near_fall_event"}
    effective = _effective_evaluation_config(source, 0.35)
    assert effective["score_threshold"] == 0.35
    assert source["score_threshold"] == 0.5


def test_near_fall_suppression_is_explicit_and_disabled_by_default() -> None:
    near = [{"prediction_id": "near-1"}]
    fall = [{"prediction_id": "fall-1"}]
    assert _suppress_near_fall_on_fall(near, fall, enabled=False) == near
    assert _suppress_near_fall_on_fall(near, fall, enabled=True) == []


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


def test_longest_near_fall_track_keeps_temporal_identity() -> None:
    selected = _select_near_fall_stream(
        [
            {"frame_id": 0, "timestamp_sec": 0.0, "person_id": "v", "track_id": 1},
            {"frame_id": 1, "timestamp_sec": 0.1, "person_id": "v", "track_id": 1},
            {"frame_id": 2, "timestamp_sec": 0.2, "person_id": "v", "track_id": 2},
        ],
        video_id="v",
        policy="longest_track",
    )
    assert [row["frame_id"] for row in selected] == [0, 1]
    assert {row["track_id"] for row in selected} == {"primary"}


def test_operational_alarm_summary_is_separate_from_event_matching() -> None:
    summary = _operational_alarm_summary(
        [
            {
                "label_id": "truth-1",
                "video_id": "video",
                "event_type": "fall",
                "start_time": 2.0,
                "end_time": 4.0,
            }
        ],
        [
            {
                "prediction_id": "pred-1",
                "video_id": "video",
                "status": "emitted",
                "alert_time": 3.0,
            }
        ],
    )
    assert summary["truth_event_hit_count"] == 1
    assert summary["truth_event_hit_recall"] == 1.0
    assert summary["definition"].endswith("diagnostic only")


def test_adaptation_partition_keeps_holdout_video_isolated() -> None:
    assert _partition_for_video({"video_id": "fall_f05_dark_1", "scene_region": "dark"}) == "holdout"
    assert _partition_for_video({"video_id": "fall_n06_base_1", "scene_region": "base"}) == "validation"
    assert _partition_for_video({"video_id": "fall_f04_bed_1", "scene_region": "bed"}) == "validation"
    assert _partition_for_video({"video_id": "fall_f03_dark_1", "scene_region": "dark"}) == "train"


def test_adaptation_onset_supervision_is_opt_in_for_fall_actions() -> None:
    actions = {
        "fall_f01_base_1": [
            {
                "action_id": "D01",
                "label_id": "label-1",
                "start_time": 1.0,
                "end_time": 3.0,
                "start_frame": 15,
                "end_frame": 44,
            },
            {
                "action_id": "A01",
                "label_id": "label-2",
                "start_time": 3.0,
                "end_time": 4.0,
                "start_frame": 45,
                "end_frame": 59,
            },
        ]
    }
    partitions = {"fall_f01_base_1": "train"}
    manifests = {"fall_f01_base_1": {"subject_id": "adult-1", "scene_region": "base"}}

    baseline = _build_governance_rows(actions, partitions, manifests)
    onset = _build_governance_rows(
        actions,
        partitions,
        manifests,
        onset_loss_weight=0.4,
    )

    assert baseline[0]["allowed_heads"] == ["presence"]
    assert baseline[0]["onset_loss_weight"] == 0.0
    assert onset[0]["allowed_heads"] == ["presence", "onset"]
    assert onset[0]["onset_loss_weight"] == 0.4
    assert onset[1]["allowed_heads"] == ["presence"]
