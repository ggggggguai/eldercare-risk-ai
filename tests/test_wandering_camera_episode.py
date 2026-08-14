from __future__ import annotations

import copy

import pytest

from elderly_monitoring.modules.mental_health.wandering.camera_episode import (
    CAMERA_EPISODE_CANDIDATE_SCHEMA_VERSION,
    CameraEpisodeError,
    aggregate_episode_candidates,
)


def _prediction(
    window_id: str,
    start: float,
    end: float,
    *,
    pattern: str = "pacing",
    parent_tracklet_id: str = "tracklet-1",
    source_video_id: str = "video-1",
    device_id: str = "camera-1",
    setup_id: str = "setup-1",
    stream_epoch: str = "epoch-1",
    track_id: int = 1,
    status: str = "ready",
) -> dict[str, object]:
    probabilities = {
        "direct": [0.8, 0.1, 0.05, 0.05],
        "pacing": [0.1, 0.7, 0.1, 0.1],
        "lapping": [0.1, 0.1, 0.7, 0.1],
        "random": [0.1, 0.1, 0.1, 0.7],
    }[pattern]
    row: dict[str, object] = {
        "schema_version": "wandering-camera-primary-prediction-v1",
        "window_id": window_id,
        "window_status": status,
        "parent_tracklet_id": parent_tracklet_id,
        "source_group_id": "group-1",
        "source_video_id": source_video_id,
        "device_id": device_id,
        "setup_id": setup_id,
        "stream_epoch": stream_epoch,
        "track_id": track_id,
        "window_start_sec": start,
        "window_end_sec": end,
        "reason_codes": [] if status == "ready" else ["synthetic_barrier"],
        "quality_flags": [],
        "probability_calibrated": False,
        "model_invocation_skipped": status == "unavailable",
        "binary": {
            "class_order": ["direct_or_non_wandering", "wandering_like"],
            "probabilities": [probabilities[0], sum(probabilities[1:])],
            "predicted_label": "direct_or_non_wandering" if pattern == "direct" else "wandering_like",
        },
        "subtype": {
            "class_order": ["pacing", "lapping", "random"],
            "probabilities": [0.7, 0.2, 0.1],
            "predicted_label": pattern if pattern != "direct" else "pacing",
        },
        "four_class": {
            "class_order": ["direct", "pacing", "lapping", "random"],
            "probabilities": probabilities,
            "predicted_label": pattern,
        },
    }
    if status != "ready":
        row["binary"] = row["subtype"] = row["four_class"] = None
    return row


def test_same_scope_parent_and_shape_merge_with_explicit_development_policy() -> None:
    episodes = aggregate_episode_candidates(
        [_prediction("w1", 0.0, 40.0), _prediction("w2", 40.0, 80.0)],
        merge_gap_seconds=0.0,
    )
    assert len(episodes) == 1
    episode = episodes[0]
    assert episode["schema_version"] == CAMERA_EPISODE_CANDIDATE_SCHEMA_VERSION
    assert episode["predicted_pattern"] == "pacing"
    assert episode["contributing_window_ids"] == ["w1", "w2"]
    assert episode["episode_start_sec"] == 0.0
    assert episode["episode_end_sec_exclusive"] == 80.0
    assert episode["prediction_status"] == "development_candidate"
    assert episode["policy_status"] == "development_unfrozen"
    assert episode["probability_calibrated"] is False
    assert episode["alert_decision"] is None
    assert "person_id" not in episode
    assert not any("risk" in key or "alert" in key and key != "alert_decision" for key in episode)


@pytest.mark.parametrize(
    "mutation",
    (
        {"source_video_id": "video-2"},
        {"device_id": "camera-2"},
        {"setup_id": "setup-2"},
        {"stream_epoch": "epoch-2"},
        {"track_id": 2},
        {"parent_tracklet_id": "tracklet-2"},
        {"pattern": "lapping"},
    ),
)
def test_scope_parent_or_shape_boundaries_never_merge(mutation: dict[str, object]) -> None:
    second = _prediction("w2", 40.0, 80.0, **mutation)
    episodes = aggregate_episode_candidates(
        [_prediction("w1", 0.0, 40.0), second],
        merge_gap_seconds=0.0,
    )
    assert len(episodes) == 2


def test_unavailable_and_inference_error_are_excluded_and_break_merging() -> None:
    rows = [
        _prediction("w1", 0.0, 40.0),
        _prediction("barrier", 20.0, 60.0, status="unavailable"),
        _prediction("w2", 40.0, 80.0),
        _prediction("error", 60.0, 100.0, status="inference_error"),
        _prediction("w3", 80.0, 120.0),
    ]
    episodes = aggregate_episode_candidates(rows, merge_gap_seconds=0.0)
    assert [episode["contributing_window_ids"] for episode in episodes] == [["w1"], ["w2"], ["w3"]]


def test_policy_is_not_implicit_or_frozen_and_bad_prediction_contract_fails_closed() -> None:
    with pytest.raises(TypeError):
        aggregate_episode_candidates([_prediction("w1", 0.0, 40.0)])  # type: ignore[call-arg]
    with pytest.raises(CameraEpisodeError):
        aggregate_episode_candidates([_prediction("w1", 0.0, 40.0)], merge_gap_seconds=-1.0)
    broken = copy.deepcopy(_prediction("w1", 0.0, 40.0))
    broken["four_class"]["class_order"] = ["pacing", "direct", "lapping", "random"]  # type: ignore[index]
    with pytest.raises(CameraEpisodeError):
        aggregate_episode_candidates([broken], merge_gap_seconds=0.0)


def test_binary_tie_and_below_threshold_follow_frozen_greater_than_or_equal_rule() -> None:
    tie = _prediction("tie", 0.0, 40.0)
    tie["binary"] = {
        "class_order": ["direct_or_non_wandering", "wandering_like"],
        "probabilities": [0.5, 0.5],
        "predicted_label": "wandering_like",
    }
    below = _prediction("below", 40.0, 80.0, pattern="direct")
    below["binary"] = {
        "class_order": ["direct_or_non_wandering", "wandering_like"],
        "probabilities": [0.51, 0.49],
        "predicted_label": "direct_or_non_wandering",
    }

    episodes = aggregate_episode_candidates([tie, below], merge_gap_seconds=0.0)
    assert len(episodes) == 2


@pytest.mark.parametrize(
    ("probabilities", "predicted_label"),
    (
        ([0.5, 0.5], "direct_or_non_wandering"),
        ([0.51, 0.49], "wandering_like"),
    ),
)
def test_binary_threshold_label_mismatch_fails_closed(
    probabilities: list[float],
    predicted_label: str,
) -> None:
    broken = _prediction("w1", 0.0, 40.0)
    broken["binary"] = {
        "class_order": ["direct_or_non_wandering", "wandering_like"],
        "probabilities": probabilities,
        "predicted_label": predicted_label,
    }
    with pytest.raises(CameraEpisodeError, match="binary predicted label is inconsistent"):
        aggregate_episode_candidates([broken], merge_gap_seconds=0.0)
