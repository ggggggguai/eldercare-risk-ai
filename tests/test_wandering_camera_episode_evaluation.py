from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from elderly_monitoring.modules.mental_health.wandering.camera_episode_evaluation import (
    CameraEpisodeEvaluationError,
    build_camera_episode_evaluation_bundle,
)
from elderly_monitoring.modules.mental_health.wandering.camera_episode_import import (
    CAMERA_EPISODE_TRUTH_SCHEMA_VERSION,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/modules/wandering_camera_episode_eval_v1.yaml"
CLASS_ORDER = ["direct", "pacing", "lapping", "random"]
CANDIDATE_ID = "topowander-m0s-seed20260731-epoch0005"
MANIFEST_SHA = "3a1e56c37b9b43e340dcd67a3163454da1d37f24b01d8677935f63a069d97ac7"
MODEL_SHA = "94c3c22d4caa38ece347d6a10f440b9fb3f7067b6c791ac1f259ce7efe69c031"


def _truth(
    episode_id: str,
    pattern: str,
    start: float,
    end: float,
    *,
    role: str = "wandering_like_positive",
    status: str = "accepted",
    purpose: str = "nonpurposeful",
    tracking_issue: str = "none",
) -> dict[str, object]:
    evidence = "scripted" if purpose != "unknown" else "unknown"
    return {
        "schema_version": CAMERA_EPISODE_TRUTH_SCHEMA_VERSION,
        "episode_id": episode_id,
        "source_video_id": "pilot-video-01",
        "target_track_id": 1,
        "cvat_track_id": None,
        "start_sec": start,
        "end_sec_exclusive": end,
        "observable_pattern": pattern,
        "purpose_context": purpose,
        "purpose_evidence": evidence,
        "evaluation_role": role,
        "script_type": "pilot",
        "visibility_quality": "good",
        "tracking_issue": tracking_issue,
        "annotation_status": status,
        "note": "",
    }


def _prediction(
    truth: dict[str, object],
    prediction_status: str,
    predicted_pattern: str | None,
) -> dict[str, object]:
    ready = prediction_status == "ready"
    qc_status = (
        prediction_status
        if prediction_status in {"unavailable", "boundary_uncertain"}
        else "ready"
    )
    predicted_binary_label = (
        "direct_or_non_wandering"
        if predicted_pattern == "direct"
        else "wandering_like"
    )
    binary_probabilities = (
        [0.7, 0.3] if predicted_pattern == "direct" else [0.3, 0.7]
    )
    four_class_probabilities = [0.3, 0.1, 0.1, 0.1]
    if predicted_pattern == "direct":
        four_class_probabilities = [0.7, 0.1, 0.1, 0.1]
    elif predicted_pattern is not None:
        four_class_probabilities[CLASS_ORDER.index(predicted_pattern)] = 0.5
    return {
        "schema_version": "wandering-camera-episode-prediction-v1",
        "episode_id": truth["episode_id"],
        "source_video_id": truth["source_video_id"],
        "track_id": truth["target_track_id"],
        "start_sec": truth["start_sec"],
        "end_sec_exclusive": truth["end_sec_exclusive"],
        "duration_sec": float(truth["end_sec_exclusive"]) - float(truth["start_sec"]),
        "prediction_status": prediction_status,
        "prediction_reason_codes": [] if ready else [prediction_status],
        "qc_status": qc_status,
        "qc_reason_codes": [] if qc_status == "ready" else [prediction_status],
        "predicted_pattern": predicted_pattern,
        "binary": (
            {
                "class_order": ["direct_or_non_wandering", "wandering_like"],
                "predicted_label": predicted_binary_label,
                "probabilities": binary_probabilities,
            }
            if ready
            else None
        ),
        "four_class": (
            {
                "class_order": CLASS_ORDER,
                "predicted_label": predicted_pattern,
                "probabilities": four_class_probabilities,
            }
            if ready
            else None
        ),
        "candidate_id": CANDIDATE_ID,
        "candidate_manifest_sha256": MANIFEST_SHA,
        "model_state_sha256": MODEL_SHA,
        "binary_decision_threshold": 0.5,
        "probability_calibrated": False,
    }


def _write_batch(
    root: Path,
    truths: list[dict[str, object]],
    predictions: list[dict[str, object]],
) -> Path:
    bundle = root / "prediction-bundle"
    bundle.mkdir()
    (bundle / "episode_predictions.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions),
        encoding="utf-8",
    )
    summary = {
        "schema_version": "wandering-camera-episode-run-summary-v1",
        "status": "wandering_m0cam_ep1a_oracle_boundary_ready",
        "evaluation_name": "oracle-boundary shape classification",
        "source_video_id": "pilot-video-01",
        "episode_count": len(predictions),
        "candidate_id": CANDIDATE_ID,
        "candidate_manifest_sha256": MANIFEST_SHA,
        "model_state_sha256": MODEL_SHA,
        "binary_decision_threshold": 0.5,
        "probability_calibrated": False,
        "models_retrained": False,
        "truth_labels_consumed_by_inference": False,
        "automatic_boundary_inference": False,
        "legacy_40_second_diagnostic_included": False,
    }
    (bundle / "summary.json").write_text(
        json.dumps(summary, sort_keys=True), encoding="utf-8"
    )
    truth_path = root / "truth.jsonl"
    truth_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in truths),
        encoding="utf-8",
    )
    index = root / "index.jsonl"
    index.write_text(
        json.dumps(
            {
                "bundle_id": "pilot-bundle-01",
                "prediction_bundle_dir": bundle.name,
                "truth_jsonl": truth_path.name,
                "participant_id": "P01",
                "session_id": "S01",
                "camera_setup_id": "C6C_OFFICE_01",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return index


def test_all_eligible_metrics_keep_pipeline_misses_and_shape_purpose_separate(
    tmp_path: Path,
) -> None:
    truths = [
        _truth("e1", "direct", 0.0, 10.0, role="ordinary_negative"),
        _truth(
            "e2",
            "pacing",
            10.0,
            25.0,
            role="purposeful_hard_negative",
            purpose="purposeful",
        ),
        _truth("e3", "lapping", 25.0, 64.0),
        _truth("e4", "random", 64.0, 104.0, role="uncertain", status="uncertain", purpose="unknown"),
        _truth("e5", "direct", 104.0, 154.0, role="ordinary_negative"),
        _truth("e6", "pacing", 154.0, 159.0),
        _truth("e7", "unknown", 159.0, 169.0, role="uncertain", status="uncertain", purpose="unknown"),
        _truth("e8", "random", 169.0, 179.0, role="excluded", status="excluded"),
    ]
    statuses = [
        ("ready", "direct"),
        ("ready", "direct"),
        ("unavailable", None),
        ("boundary_uncertain", None),
        ("inference_error", None),
        ("abstention", None),
        ("ready", "direct"),
        ("ready", "random"),
    ]
    predictions = [
        _prediction(truth, status, pattern)
        for truth, (status, pattern) in zip(truths, statuses, strict=True)
    ]
    index = _write_batch(tmp_path, truths, predictions)
    output = tmp_path / "evaluation"

    result = build_camera_episode_evaluation_bundle(
        config_path=CONFIG,
        batch_index_path=index,
        output_dir=output,
    )

    assert result.shape_eligible_count == 6
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    main = metrics["all_shape_eligible"]
    conditional = metrics["ready_only_conditional"]
    assert main["four_class"]["support"] == 6
    assert main["four_class"]["accuracy"] == pytest.approx(1 / 6)
    assert main["four_class"]["macro_f1"] == pytest.approx(0.125)
    assert conditional["four_class"]["support"] == 2
    assert conditional["four_class"]["macro_f1"] == "not_computable"
    assert main["shape_binary"]["support"] == 6
    assert main["shape_binary"]["macro_f1"] == pytest.approx(0.25)
    assert metrics["qc_coverage"]["definition"] == "qc_status_ready"
    assert metrics["qc_coverage"]["overall"]["qc_ready"] == 4
    assert metrics["qc_coverage"]["overall"]["coverage"] == pytest.approx(4 / 6)
    assert metrics["qc_coverage"]["overall"]["prediction_ready"] == 2
    assert metrics["qc_coverage"]["overall"]["prediction_ready_coverage"] == pytest.approx(
        2 / 6
    )
    assert metrics["duration_bands"]["short"]["support"] == 2
    assert metrics["duration_bands"]["medium"]["support"] == 2
    assert metrics["duration_bands"]["long"]["support"] == 2
    assert metrics["group_counts"] == {
        "participant": 1,
        "session": 1,
        "camera_setup": 1,
    }
    participant_support = metrics["group_support"]["participant"]["P01"]
    assert participant_support["eligible"] == 6
    assert participant_support["eligible_by_truth_class"] == {
        "direct": 2,
        "pacing": 2,
        "lapping": 1,
        "random": 1,
    }
    assert participant_support["prediction_status_counts_for_shape_eligible"] == {
        "ready": 2,
        "unavailable": 1,
        "boundary_uncertain": 1,
        "inference_error": 1,
        "abstention": 1,
    }
    purposeful = metrics["purposeful_hard_negative_diagnostics"]
    assert purposeful["eligible_count"] == 1
    assert purposeful["shape_prediction_distribution"]["direct"] == 1
    assert purposeful["alert_metrics_available"] is False
    assert metrics["probability_calibrated"] is False
    confusion = json.loads((output / "confusion.json").read_text(encoding="utf-8"))
    assert confusion["all_shape_eligible"]["four_class"]["lapping"]["pipeline_miss"] == 1
    assert confusion["all_shape_eligible"]["four_class"]["random"]["pipeline_miss"] == 1
    rows = [
        json.loads(line)
        for line in (output / "episode_results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    random_row = next(row for row in rows if row["episode_id"] == "e4")
    assert random_row["schema_version"] == "wandering-camera-episode-eval-result-v1"
    assert random_row["shape_truth_status"] == "eligible"
    assert random_row["annotation_status"] == "uncertain"
    assert random_row["evaluation_role"] == "uncertain"
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["prediction_status_counts_all_truth"] == {
        "ready": 4,
        "unavailable": 1,
        "boundary_uncertain": 1,
        "inference_error": 1,
        "abstention": 1,
    }
    assert summary["prediction_status_counts_for_shape_eligible"] == {
        "ready": 2,
        "unavailable": 1,
        "boundary_uncertain": 1,
        "inference_error": 1,
        "abstention": 1,
    }
    failures = [
        json.loads(line)
        for line in (output / "failures.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    purposeful_failure = next(row for row in failures if row["episode_id"] == "e2")
    assert purposeful_failure["participant_id"] == "P01"
    assert purposeful_failure["session_id"] == "S01"
    assert purposeful_failure["camera_setup_id"] == "C6C_OFFICE_01"
    assert purposeful_failure["evaluation_role"] == "purposeful_hard_negative"
    assert purposeful_failure["purpose_context"] == "purposeful"
    assert purposeful_failure["annotation_status"] == "accepted"
    assert purposeful_failure["duration_band"] == "medium"
    assert purposeful_failure["binary_truth"] == "wandering_like"
    assert purposeful_failure["predicted_binary_label"] == "direct_or_non_wandering"
    assert purposeful_failure["classification_error_types"] == [
        "four_class_misclassified",
        "shape_binary_misclassified",
    ]
    assert set(path.name for path in output.iterdir()) == {
        "README.md",
        "confusion.json",
        "episode_results.jsonl",
        "failures.jsonl",
        "metrics.json",
        "summary.json",
    }
    with pytest.raises(FileExistsError):
        build_camera_episode_evaluation_bundle(
            config_path=CONFIG,
            batch_index_path=index,
            output_dir=output,
        )


@pytest.mark.parametrize("field", ["episode_id", "source_video_id", "track_id", "start_sec", "end_sec_exclusive"])
def test_identity_drift_is_rejected(tmp_path: Path, field: str) -> None:
    truth = _truth("e1", "direct", 0.0, 10.0, role="ordinary_negative")
    prediction = _prediction(truth, "ready", "direct")
    replacements = {
        "episode_id": "different",
        "source_video_id": "different-video",
        "track_id": 2,
        "start_sec": 0.01,
        "end_sec_exclusive": 10.01,
    }
    prediction[field] = replacements[field]
    if field in {"start_sec", "end_sec_exclusive"}:
        prediction["duration_sec"] = (
            float(prediction["end_sec_exclusive"]) - float(prediction["start_sec"])
        )
    index = _write_batch(tmp_path, [truth], [prediction])

    with pytest.raises(CameraEpisodeEvaluationError, match="one-to-one|identity|missing"):
        build_camera_episode_evaluation_bundle(
            config_path=CONFIG,
            batch_index_path=index,
            output_dir=tmp_path / "evaluation",
        )


@pytest.mark.parametrize(
    "field",
    ["candidate_id", "candidate_manifest_sha256", "model_state_sha256"],
)
def test_prediction_candidate_identity_must_match_summary(
    tmp_path: Path, field: str
) -> None:
    truth = _truth("e1", "direct", 0.0, 10.0, role="ordinary_negative")
    prediction = _prediction(truth, "ready", "direct")
    prediction[field] = "different"
    index = _write_batch(tmp_path, [truth], [prediction])

    with pytest.raises(CameraEpisodeEvaluationError, match="identity differs from summary"):
        build_camera_episode_evaluation_bundle(
            config_path=CONFIG,
            batch_index_path=index,
            output_dir=tmp_path / "evaluation",
        )


@pytest.mark.parametrize(
    ("field", "mutation"),
    [
        ("models_retrained", "true"),
        ("models_retrained", "missing"),
        ("legacy_40_second_diagnostic_included", "true"),
        ("legacy_40_second_diagnostic_included", "missing"),
    ],
)
def test_prediction_summary_must_preserve_frozen_oracle_semantics(
    tmp_path: Path, field: str, mutation: str
) -> None:
    truth = _truth("e1", "direct", 0.0, 10.0, role="ordinary_negative")
    prediction = _prediction(truth, "ready", "direct")
    index = _write_batch(tmp_path, [truth], [prediction])
    summary_path = tmp_path / "prediction-bundle" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if mutation == "true":
        summary[field] = True
    else:
        del summary[field]
    summary_path.write_text(json.dumps(summary, sort_keys=True), encoding="utf-8")

    with pytest.raises(CameraEpisodeEvaluationError, match=field):
        build_camera_episode_evaluation_bundle(
            config_path=CONFIG,
            batch_index_path=index,
            output_dir=tmp_path / "evaluation",
        )


@pytest.mark.parametrize(
    ("prediction_status", "qc_status"),
    [
        ("ready", "unavailable"),
        ("unavailable", "ready"),
        ("boundary_uncertain", "ready"),
        ("inference_error", "unavailable"),
        ("abstention", "boundary_uncertain"),
    ],
)
def test_prediction_and_qc_status_matrix_is_validated(
    tmp_path: Path, prediction_status: str, qc_status: str
) -> None:
    truth = _truth("e1", "direct", 0.0, 10.0, role="ordinary_negative")
    predicted_pattern = "direct" if prediction_status == "ready" else None
    prediction = _prediction(truth, prediction_status, predicted_pattern)
    prediction["qc_status"] = qc_status
    prediction["qc_reason_codes"] = (
        [] if qc_status == "ready" else ["synthetic_qc_failure"]
    )
    index = _write_batch(tmp_path, [truth], [prediction])

    with pytest.raises(CameraEpisodeEvaluationError, match="QC/prediction status"):
        build_camera_episode_evaluation_bundle(
            config_path=CONFIG,
            batch_index_path=index,
            output_dir=tmp_path / "evaluation",
        )


@pytest.mark.parametrize(
    ("prediction_status", "reason_codes"),
    [
        ("ready", ["unexpected_failure_reason"]),
        ("inference_error", []),
    ],
)
def test_prediction_status_and_reason_codes_are_consistent(
    tmp_path: Path, prediction_status: str, reason_codes: list[str]
) -> None:
    truth = _truth("e1", "direct", 0.0, 10.0, role="ordinary_negative")
    predicted_pattern = "direct" if prediction_status == "ready" else None
    prediction = _prediction(truth, prediction_status, predicted_pattern)
    prediction["prediction_reason_codes"] = reason_codes
    index = _write_batch(tmp_path, [truth], [prediction])

    with pytest.raises(CameraEpisodeEvaluationError, match="prediction.*reasons"):
        build_camera_episode_evaluation_bundle(
            config_path=CONFIG,
            batch_index_path=index,
            output_dir=tmp_path / "evaluation",
        )


@pytest.mark.parametrize(
    "target", ["truth_pattern", "truth_annotation_status", "qc_status"]
)
def test_non_string_enums_are_rejected_as_domain_errors(
    tmp_path: Path, target: str
) -> None:
    truth = _truth("e1", "direct", 0.0, 10.0, role="ordinary_negative")
    prediction = _prediction(truth, "ready", "direct")
    if target == "truth_pattern":
        truth["observable_pattern"] = []
    elif target == "truth_annotation_status":
        truth["annotation_status"] = {}
    else:
        prediction["qc_status"] = {}
    index = _write_batch(tmp_path, [truth], [prediction])

    with pytest.raises(CameraEpisodeEvaluationError):
        build_camera_episode_evaluation_bundle(
            config_path=CONFIG,
            batch_index_path=index,
            output_dir=tmp_path / "evaluation",
        )


def test_shape_binary_metrics_use_the_independent_thresholded_binary_head(
    tmp_path: Path,
) -> None:
    truth = _truth("e1", "direct", 0.0, 10.0, role="ordinary_negative")
    prediction = _prediction(truth, "ready", "direct")
    prediction["binary"] = {
        "class_order": ["direct_or_non_wandering", "wandering_like"],
        "predicted_label": "wandering_like",
        "probabilities": [0.4, 0.6],
    }
    prediction["four_class"] = {
        "class_order": CLASS_ORDER,
        "predicted_label": "direct",
        "probabilities": [0.4, 0.2, 0.2, 0.2],
    }
    index = _write_batch(tmp_path, [truth], [prediction])
    output = tmp_path / "evaluation"

    build_camera_episode_evaluation_bundle(
        config_path=CONFIG,
        batch_index_path=index,
        output_dir=output,
    )

    confusion = json.loads((output / "confusion.json").read_text(encoding="utf-8"))
    assert confusion["all_shape_eligible"]["four_class"]["direct"]["direct"] == 1
    assert (
        confusion["all_shape_eligible"]["shape_binary"]
        ["direct_or_non_wandering"]["wandering_like"]
        == 1
    )
    result_row = json.loads(
        (output / "episode_results.jsonl").read_text(encoding="utf-8")
    )
    assert result_row["predicted_binary_label"] == "wandering_like"
    assert result_row["four_class_metric_disposition"] == "correct"
    assert result_row["shape_binary_metric_disposition"] == "misclassified"
    failure = json.loads((output / "failures.jsonl").read_text(encoding="utf-8"))
    assert failure["classification_error_types"] == ["shape_binary_misclassified"]


def test_binary_and_four_class_probabilities_preserve_hierarchical_relation(
    tmp_path: Path,
) -> None:
    truth = _truth("e1", "direct", 0.0, 10.0, role="ordinary_negative")
    prediction = _prediction(truth, "ready", "direct")
    prediction["binary"] = {
        "class_order": ["direct_or_non_wandering", "wandering_like"],
        "predicted_label": "direct_or_non_wandering",
        "probabilities": [0.6, 0.4],
    }
    index = _write_batch(tmp_path, [truth], [prediction])

    with pytest.raises(CameraEpisodeEvaluationError, match="hierarchical"):
        build_camera_episode_evaluation_bundle(
            config_path=CONFIG,
            batch_index_path=index,
            output_dir=tmp_path / "evaluation",
        )


def test_binary_label_must_follow_the_frozen_point_five_threshold(
    tmp_path: Path,
) -> None:
    truth = _truth("e1", "direct", 0.0, 10.0, role="ordinary_negative")
    prediction = _prediction(truth, "ready", "direct")
    prediction["binary"] = {
        "class_order": ["direct_or_non_wandering", "wandering_like"],
        "predicted_label": "direct_or_non_wandering",
        "probabilities": [0.5, 0.5],
    }
    index = _write_batch(tmp_path, [truth], [prediction])

    with pytest.raises(CameraEpisodeEvaluationError, match="binary.*0.5 threshold"):
        build_camera_episode_evaluation_bundle(
            config_path=CONFIG,
            batch_index_path=index,
            output_dir=tmp_path / "evaluation",
        )


@pytest.mark.parametrize(
    "probabilities",
    [
        [0.1, 0.7, 0.1, 0.1],
        [0.7, 0.2, 0.2, -0.1],
        [0.7, 0.1, 0.1],
    ],
)
def test_four_class_probabilities_and_label_must_be_consistent(
    tmp_path: Path, probabilities: list[float]
) -> None:
    truth = _truth("e1", "direct", 0.0, 10.0, role="ordinary_negative")
    prediction = _prediction(truth, "ready", "direct")
    prediction["four_class"] = {
        "class_order": CLASS_ORDER,
        "predicted_label": "direct",
        "probabilities": probabilities,
    }
    index = _write_batch(tmp_path, [truth], [prediction])

    with pytest.raises(CameraEpisodeEvaluationError, match="four-class"):
        build_camera_episode_evaluation_bundle(
            config_path=CONFIG,
            batch_index_path=index,
            output_dir=tmp_path / "evaluation",
        )


def test_purposeful_wandering_shape_is_diagnosed_even_when_role_is_misassigned(
    tmp_path: Path,
) -> None:
    truth = _truth(
        "e1",
        "lapping",
        0.0,
        20.0,
        role="wandering_like_positive",
        purpose="purposeful",
    )
    prediction = _prediction(truth, "ready", "lapping")
    index = _write_batch(tmp_path, [truth], [prediction])
    output = tmp_path / "evaluation"

    build_camera_episode_evaluation_bundle(
        config_path=CONFIG,
        batch_index_path=index,
        output_dir=output,
    )

    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    diagnostic = metrics["purposeful_hard_negative_diagnostics"]
    assert diagnostic["eligible_count"] == 1
    assert diagnostic["declared_role_count"] == 0
    assert diagnostic["role_mismatch_count"] == 1
    assert diagnostic["evaluation_role_distribution"] == {
        "wandering_like_positive": 1
    }


def test_shape_eligibility_distinguishes_clear_fragmented_and_unusable_truth(
    tmp_path: Path,
) -> None:
    truths = [
        _truth(
            "clear-fragmented",
            "pacing",
            0.0,
            10.0,
            purpose="unknown",
            tracking_issue="fragmented_track",
        ),
        _truth(
            "boundary-unjudgeable",
            "unknown",
            10.0,
            20.0,
            role="uncertain",
            status="uncertain",
            purpose="unknown",
        ),
        _truth(
            "whole-segment-unusable",
            "random",
            20.0,
            30.0,
            role="excluded",
            status="excluded",
        ),
        _truth(
            "wrong-person",
            "lapping",
            30.0,
            40.0,
            tracking_issue="wrong_target",
        ),
    ]
    predictions = [_prediction(truth, "ready", "direct") for truth in truths]
    index = _write_batch(tmp_path, truths, predictions)
    output = tmp_path / "evaluation"

    result = build_camera_episode_evaluation_bundle(
        config_path=CONFIG,
        batch_index_path=index,
        output_dir=output,
    )

    assert result.shape_eligible_count == 1
    rows = {
        row["episode_id"]: row
        for row in (
            json.loads(line)
            for line in (output / "episode_results.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
    }
    assert rows["clear-fragmented"]["shape_truth_status"] == "eligible"
    assert rows["boundary-unjudgeable"]["shape_truth_status"] == "unknown"
    assert rows["whole-segment-unusable"]["shape_truth_status"] == "excluded"
    assert rows["wrong-person"]["shape_truth_status"] == "excluded"


def test_duplicate_physical_episode_identity_is_rejected_even_with_new_ids(
    tmp_path: Path,
) -> None:
    first_truth = _truth("e1", "direct", 0.0, 10.0, role="ordinary_negative")
    second_truth = _truth("e2", "direct", 0.0, 10.0, role="ordinary_negative")
    first_prediction = _prediction(first_truth, "ready", "direct")
    second_prediction = _prediction(second_truth, "ready", "direct")
    index = _write_batch(
        tmp_path,
        [first_truth, second_truth],
        [first_prediction, second_prediction],
    )

    with pytest.raises(
        CameraEpisodeEvaluationError, match="duplicate physical episode identity"
    ):
        build_camera_episode_evaluation_bundle(
            config_path=CONFIG,
            batch_index_path=index,
            output_dir=tmp_path / "evaluation",
        )


def test_overlapping_physical_episode_identity_is_rejected_across_new_ids(
    tmp_path: Path,
) -> None:
    first_truth = _truth("e1", "direct", 0.0, 10.0, role="ordinary_negative")
    second_truth = _truth("e2", "pacing", 9.0, 15.0)
    index = _write_batch(
        tmp_path,
        [first_truth, second_truth],
        [
            _prediction(first_truth, "ready", "direct"),
            _prediction(second_truth, "ready", "pacing"),
        ],
    )

    with pytest.raises(
        CameraEpisodeEvaluationError, match="overlapping physical episode"
    ):
        build_camera_episode_evaluation_bundle(
            config_path=CONFIG,
            batch_index_path=index,
            output_dir=tmp_path / "evaluation",
        )


@pytest.mark.parametrize(
    "group_field", ["participant_id", "session_id", "camera_setup_id"]
)
def test_same_source_track_group_identity_must_be_consistent_across_bundles(
    tmp_path: Path, group_field: str
) -> None:
    first_truth = _truth("e1", "direct", 0.0, 10.0, role="ordinary_negative")
    second_truth = _truth("e2", "pacing", 10.0, 20.0)
    index = _write_batch(
        tmp_path,
        [first_truth],
        [_prediction(first_truth, "ready", "direct")],
    )
    second_root = tmp_path / "second"
    second_root.mkdir()
    second_index = _write_batch(
        second_root,
        [second_truth],
        [_prediction(second_truth, "ready", "pacing")],
    )
    second_row = json.loads(second_index.read_text(encoding="utf-8"))
    second_row.update(
        {
            "bundle_id": "pilot-bundle-02",
            "prediction_bundle_dir": "second/prediction-bundle",
            "truth_jsonl": "second/truth.jsonl",
            group_field: f"different-{group_field}",
        }
    )
    index.write_text(
        index.read_text(encoding="utf-8")
        + json.dumps(second_row, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(CameraEpisodeEvaluationError, match="group identity"):
        build_camera_episode_evaluation_bundle(
            config_path=CONFIG,
            batch_index_path=index,
            output_dir=tmp_path / "evaluation",
        )


def test_duplicate_episode_id_and_missing_class_macro_are_handled(tmp_path: Path) -> None:
    truth = _truth("e1", "direct", 0.0, 10.0, role="ordinary_negative")
    prediction = _prediction(truth, "ready", "direct")
    index = _write_batch(tmp_path, [truth], [prediction])
    output = tmp_path / "evaluation"
    build_camera_episode_evaluation_bundle(
        config_path=CONFIG,
        batch_index_path=index,
        output_dir=output,
    )
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["all_shape_eligible"]["four_class"]["macro_f1"] == "not_computable"

    duplicate_root = tmp_path / "duplicate"
    duplicate_root.mkdir()
    duplicate_index = _write_batch(duplicate_root, [truth, truth], [prediction, prediction])
    with pytest.raises(CameraEpisodeEvaluationError, match="duplicate episode_id"):
        build_camera_episode_evaluation_bundle(
            config_path=CONFIG,
            batch_index_path=duplicate_index,
            output_dir=duplicate_root / "evaluation",
        )


def test_evaluation_cli_help() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/wandering/run_camera_episode_evaluation.py"), "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--batch-index" in completed.stdout


def test_evaluation_cli_emits_structured_identity_rejection_without_success_bundle(
    tmp_path: Path,
) -> None:
    truth = _truth("e1", "direct", 0.0, 10.0, role="ordinary_negative")
    prediction = _prediction(truth, "ready", "direct")
    prediction["track_id"] = 2
    index = _write_batch(tmp_path, [truth], [prediction])
    output = tmp_path / "evaluation"

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/wandering/run_camera_episode_evaluation.py"),
            "--batch-index",
            str(index),
            "--output-dir",
            str(output),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    failure = json.loads(completed.stderr)
    assert failure["status"] == "rejected"
    assert failure["failure_type"] == "identity_or_input_error"
    assert failure["error_code"] == "track_identity_mismatch"
    assert failure["successful_evaluation_written"] is False
    assert not output.exists()


def test_evaluation_cli_emits_structured_missing_index_rejection(
    tmp_path: Path,
) -> None:
    output = tmp_path / "evaluation"

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/wandering/run_camera_episode_evaluation.py"),
            "--batch-index",
            str(tmp_path / "missing.jsonl"),
            "--output-dir",
            str(output),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    failure = json.loads(completed.stderr)
    assert failure["error_code"] == "batch_index_inaccessible"
    assert failure["successful_evaluation_written"] is False
    assert not output.exists()
