from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from elderly_monitoring.modules.mental_health.wandering.camera_development import (
    CameraDevelopmentError,
    build_coverage_counts,
    build_stage_timings,
    compute_eligible_negative_person_hours,
    evaluate_shape_counts,
    evaluate_purposeful_hard_negatives,
    evaluate_labeled_episodes,
    load_camera_development_config,
    match_episodes_one_to_one,
    run_authorized_camera_development,
    subtract_intervals,
    union_intervals,
)
from elderly_monitoring.modules.mental_health.wandering.release import CandidateRuntime


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/modules/wandering_camera_development_v1.yaml"
MANIFEST_SHA = "3a1e56c37b9b43e340dcd67a3163454da1d37f24b01d8677935f63a069d97ac7"


def _receipt(*, expires_at: str = "2027-01-01T00:00:00+08:00") -> dict:
    return {
        "schema_version": "wandering-camera-authorization-receipt-v1",
        "receipt_id": "receipt-development-0001",
        "approval_status": "approved",
        "active": True,
        "purpose": "camera_development",
        "dataset_role": "development",
        "valid_from": "2026-01-01T00:00:00+08:00",
        "expires_at": expires_at,
        "allowed_operations": ["run_development"],
        "participant_ids": ["participant-0001"],
        "session_ids": ["session-0001"],
        "camera_setup_ids": ["setup-0001"],
        "source_group_ids": ["group-0001"],
        "governance": {
            "consent_confirmed": True,
            "deidentified_storage": True,
            "access_control_confirmed": True,
            "retention_and_deletion_defined": True,
            "withdrawal_process_defined": True,
            "audio_policy": "not_collected",
        },
    }


def _collection(
    *,
    dataset_role: str = "development",
    receipt_id: str = "receipt-development-0001",
    participant_id: str = "participant-0001",
) -> dict:
    return {
        "schema_version": "wandering-camera-collection-v1",
        "collection_id": "collection-0001",
        "dataset_role": dataset_role,
        "authorization_receipt_id": receipt_id,
        "participants": [{"participant_id": participant_id}],
        "camera_setups": [{"camera_setup_id": "setup-0001", "setup_id": "adapter-setup-0001"}],
        "sessions": [
            {
                "session_id": "session-0001",
                "participant_id": participant_id,
                "source_group_id": "group-0001",
                "camera_setup_ids": ["setup-0001"],
            }
        ],
        "sources": [
            {
                "source_video_id": "video-0001",
                "session_id": "session-0001",
                "source_group_id": "group-0001",
                "camera_setup_id": "setup-0001",
                "device_id": "device-0001",
                "setup_id": "adapter-setup-0001",
                "stream_epoch": "epoch-0001",
                "tracking_ref": "inputs/tracking.jsonl",
                "media_sidecar_ref": "inputs/media.json",
            }
        ],
        "tracklet_participant_bindings": [],
        "participant_present_intervals": [],
        "clock_alignments": [],
    }


class _DeterministicCameraModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def forward(self, features, points, mask):
        batch = features.shape[0]
        zero = self.anchor.reshape(1, 1) * 0.0
        return {
            "binary_logit": zero.expand(batch, 1) + 3.0,
            "subtype_logits": torch.cat(
                (
                    zero.expand(batch, 1) + 3.0,
                    zero.expand(batch, 1),
                    zero.expand(batch, 1) - 1.0,
                ),
                dim=1,
            ),
        }


def _fake_candidate_loader(path: Path, *, expected_manifest_sha256: str) -> CandidateRuntime:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    model = _DeterministicCameraModel().eval()
    return CandidateRuntime(
        bundle_root=path.parent,
        manifest_path=path,
        manifest_sha256=expected_manifest_sha256,
        manifest=manifest,
        model=model,
    )


def _authorized_fixture(tmp_path: Path, *, labeled: bool) -> dict[str, Path | dict]:
    fixture_root = ROOT / "reports/mental_health/wandering_m0cam_engineering_v1/fixtures"
    tracking = tmp_path / "tracking.jsonl"
    tracking.write_bytes((fixture_root / "synthetic_tracking.jsonl").read_bytes())
    media = json.loads(
        (fixture_root / "synthetic_media_sidecar.json").read_text(encoding="utf-8")
    )
    media["authorization_status"] = (
        "authorized_camera_labeled_evaluation"
        if labeled
        else "authorized_camera_engineering_smoke"
    )
    sidecar = tmp_path / "sidecar.json"
    sidecar.write_text(json.dumps(media), encoding="utf-8")

    receipt = _receipt()
    receipt["expires_at"] = "2099-01-01T00:00:00+08:00"
    receipt["participant_ids"] = ["participant-0001"]
    receipt["session_ids"] = ["session-0001"]
    receipt["camera_setup_ids"] = ["camera-setup-0001"]
    receipt["source_group_ids"] = [media["source_group_id"]]
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    bindings = [
        {
            "source_group_id": media["source_group_id"],
            "source_video_id": media["source_video_id"],
            "device_id": media["device_id"],
            "setup_id": media["setup_id"],
            "stream_epoch": media["stream_epoch"],
            "track_id": track_id,
            "participant_id": "participant-0001",
            "session_id": "session-0001",
            "camera_setup_id": "camera-setup-0001",
            "clock_domain_id": "clock-0001",
        }
        for track_id in (1, 2, 3)
    ]
    collection = {
        "schema_version": "wandering-camera-collection-v1",
        "collection_id": "collection-0001",
        "dataset_role": "development",
        "authorization_receipt_id": receipt["receipt_id"],
        "participants": [{"participant_id": "participant-0001"}],
        "camera_setups": [
            {
                "camera_setup_id": "camera-setup-0001",
                "setup_id": media["setup_id"],
            }
        ],
        "sessions": [
            {
                "session_id": "session-0001",
                "participant_id": "participant-0001",
                "source_group_id": media["source_group_id"],
                "camera_setup_ids": ["camera-setup-0001"],
            }
        ],
        "sources": [
            {
                "source_video_id": media["source_video_id"],
                "session_id": "session-0001",
                "source_group_id": media["source_group_id"],
                "camera_setup_id": "camera-setup-0001",
                "device_id": media["device_id"],
                "setup_id": media["setup_id"],
                "stream_epoch": media["stream_epoch"],
                "tracking_ref": "inputs/tracking.jsonl",
                "media_sidecar_ref": "inputs/media.json",
            }
        ],
        "tracklet_participant_bindings": bindings,
        "participant_present_intervals": [
            {
                "participant_id": "participant-0001",
                "session_id": "session-0001",
                "clock_domain_id": "clock-0001",
                "start_sec": 0.0,
                "end_sec_exclusive": 40.0,
            }
        ],
        "clock_alignments": [
            {
                "clock_domain_id": "clock-0001",
                "session_id": "session-0001",
                "source_group_id": media["source_group_id"],
                "camera_setup_ids": ["camera-setup-0001"],
                "source_video_ids": [media["source_video_id"]],
                "status": "aligned",
            }
        ],
    }
    collection_path = tmp_path / "collection.json"
    collection_path.write_text(json.dumps(collection), encoding="utf-8")

    annotations_path = tmp_path / "annotations.jsonl"
    if labeled:
        annotation_rows = []
        for track_id, shape, role, purpose in (
            (1, "pacing", "purposeful_hard_negative", "purposeful"),
            (2, "direct", "ordinary_negative", "nonpurposeful"),
            (3, "pacing", "wandering_like_positive", "nonpurposeful"),
        ):
            annotation_rows.append(
                {
                    "annotation_id": f"annotation-{track_id}",
                    **bindings[track_id - 1],
                    "start_sec": 0.0,
                    "end_sec_exclusive": 40.0,
                    "observable_pattern": shape,
                    "purpose_context": purpose,
                    "purpose_evidence": "scripted",
                    "evaluation_role": role,
                    "script_type": "test_fixture",
                    "visibility_quality": "synthetic",
                    "tracking_issue": "none",
                    "annotation_status": "accepted",
                    "annotator_id": "fixture-annotator",
                    "reviewed_by": "fixture-reviewer",
                }
            )
        annotations_path.write_text(
            "".join(json.dumps(row) + "\n" for row in annotation_rows), encoding="utf-8"
        )

    policy = {
        "matching_policy": _matching_policy(),
        "uncertain_policy": {"policy_id": "uncertain-test-v1", "threshold": 0.5},
        "episode_merge_policy": {"policy_id": "merge-test-v1", "merge_gap_seconds": 0.0},
    }
    return {
        "receipt": receipt_path,
        "collection": collection_path,
        "tracking": tracking,
        "sidecar": sidecar,
        "annotations": annotations_path,
        "policy": policy,
    }


def test_development_config_has_fixed_identity_but_no_scientific_policy() -> None:
    config = load_camera_development_config(CONFIG)
    assert config["schema_version"] == "wandering-camera-development-config-v1"
    assert config["dataset_role"] == "development"
    assert config["candidate"]["manifest_sha256"] == (
        "3a1e56c37b9b43e340dcd67a3163454da1d37f24b01d8677935f63a069d97ac7"
    )
    assert config["matching_policy"] is None
    assert config["uncertain_policy"] is None
    assert config["episode_merge_policy"] is None
    assert config["inference"]["binary_decision_threshold"] == 0.5


def _scope(*, track_id: int = 1, session_id: str = "session-1") -> dict:
    return {
        "source_group_id": f"group-{session_id}",
        "source_video_id": f"video-{session_id}",
        "device_id": "device-1",
        "setup_id": "adapter-setup-1",
        "stream_epoch": "epoch-1",
        "track_id": track_id,
        "participant_id": "participant-1",
        "session_id": session_id,
        "camera_setup_id": "camera-setup-1",
        "clock_domain_id": f"clock-{session_id}",
    }


def _matching_policy() -> dict:
    return {
        "policy_id": "matching-test-v1",
        "minimum_temporal_iou": 0.5,
        "maximum_onset_delta_sec": 5.0,
    }


def test_episode_matching_is_one_to_one_and_has_stable_tie_break() -> None:
    truth = [
        {"annotation_id": "a1", **_scope(), "start_sec": 0.0, "end_sec_exclusive": 10.0},
        {"annotation_id": "a2", **_scope(), "start_sec": 0.0, "end_sec_exclusive": 10.0},
    ]
    predictions = [
        {"prediction_id": "p2", **_scope(), "start_sec": 0.0, "end_sec_exclusive": 10.0},
        {"prediction_id": "p1", **_scope(), "start_sec": 0.0, "end_sec_exclusive": 10.0},
        {"prediction_id": "p3", **_scope(), "start_sec": 1.0, "end_sec_exclusive": 9.0},
    ]
    result = match_episodes_one_to_one(
        predictions,
        truth,
        matching_policy={**_matching_policy(), "maximum_onset_delta_sec": 2.0},
    )
    assert result["matches"] == [
        {"prediction_id": "p1", "annotation_id": "a1", "temporal_iou": 1.0},
        {"prediction_id": "p2", "annotation_id": "a2", "temporal_iou": 1.0},
    ]
    assert result["unmatched_prediction_ids"] == ["p3"]
    assert result["unmatched_annotation_ids"] == []

    with pytest.raises(CameraDevelopmentError, match="explicit matching policy"):
        match_episodes_one_to_one(predictions, truth, matching_policy=None)
    separated = match_episodes_one_to_one(
        [{**predictions[0], "source_video_id": "video-a"}],
        [{**truth[0], "source_video_id": "video-b"}],
        matching_policy={**_matching_policy(), "maximum_onset_delta_sec": 2.0},
    )
    assert separated["matches"] == []


def test_episode_matching_maximizes_cardinality_before_local_iou() -> None:
    annotations = [
        {"annotation_id": "A1", **_scope(), "start_sec": 0.0, "end_sec_exclusive": 10.0},
        {"annotation_id": "A2", **_scope(), "start_sec": 5.0, "end_sec_exclusive": 15.0},
    ]
    predictions = [
        {"prediction_id": "P1", **_scope(), "start_sec": 0.0, "end_sec_exclusive": 15.0},
        {"prediction_id": "P2", **_scope(), "start_sec": 0.0, "end_sec_exclusive": 5.1},
    ]
    result = match_episodes_one_to_one(
        predictions, annotations, matching_policy=_matching_policy()
    )
    assert {(row["prediction_id"], row["annotation_id"]) for row in result["matches"]} == {
        ("P1", "A2"),
        ("P2", "A1"),
    }

    zero_overlap_policy = {
        **_matching_policy(),
        "minimum_temporal_iou": 0.0,
        "maximum_onset_delta_sec": 20.0,
    }
    separated = match_episodes_one_to_one(
        [{"prediction_id": "P", **_scope(), "start_sec": 0.0, "end_sec_exclusive": 1.0}],
        [{"annotation_id": "A", **_scope(), "start_sec": 1.0, "end_sec_exclusive": 2.0}],
        matching_policy=zero_overlap_policy,
    )
    assert separated["matches"] == []

    with pytest.raises(CameraDevelopmentError, match="policy_id"):
        match_episodes_one_to_one(
            predictions,
            annotations,
            matching_policy={
                "policy_id": "",
                "minimum_temporal_iou": 0.5,
                "maximum_onset_delta_sec": 5.0,
            },
        )


def test_shape_outputs_raw_four_class_and_binary_counts_only() -> None:
    pairs = [
        {"truth": "direct", "prediction": "direct"},
        {"truth": "pacing", "prediction": "direct"},
        {"truth": "lapping", "prediction": "lapping"},
        {"truth": "random", "prediction": "uncertain"},
    ]
    result = evaluate_shape_counts(pairs)
    assert result["four_class"]["confusion"]["pacing"]["direct"] == 1
    assert result["four_class"]["abstention_by_truth"]["random"] == 1
    assert result["binary"]["confusion"]["wandering_like"]["direct_or_non_wandering"] == 1
    assert "f1" not in json.dumps(result).lower()


def test_purposeful_hard_negative_is_diagnostic_not_shape_relabeling() -> None:
    result = evaluate_purposeful_hard_negatives(
        [
            {
                "annotation_id": "a1",
                "observable_pattern": "pacing",
                "purpose_context": "purposeful",
                "evaluation_role": "purposeful_hard_negative",
                "prediction": "pacing",
            }
        ]
    )
    assert result == {
        "eligible_count": 1,
        "shape_pacing_prediction_count": 1,
        "shape_wandering_like_prediction_count": 1,
        "alert_metrics_available": False,
    }


def test_coverage_keeps_runtime_statuses_distinct() -> None:
    result = build_coverage_counts(
        [
            {"prediction_status": "ready"},
            {"prediction_status": "unavailable"},
            {"prediction_status": "inference_error"},
            {"prediction_status": "uncertain"},
            {"prediction_status": "uncertain"},
        ]
    )
    assert result == {
        "total": 5,
        "ready": 1,
        "unavailable": 1,
        "inference_error": 1,
        "uncertain": 2,
    }


def test_interval_union_subtraction_and_negative_person_hours() -> None:
    assert union_intervals([(0, 10), (5, 15), (20, 30)]) == [(0.0, 15.0), (20.0, 30.0)]
    assert subtract_intervals([(0, 30)], [(5, 10), (20, 25)]) == [
        (0.0, 5.0),
        (10.0, 20.0),
        (25.0, 30.0),
    ]
    value = compute_eligible_negative_person_hours(
        participant_present=[{"participant_id": "u1", "session_id": "s1", "clock_domain_id": "c1", "start_sec": 0.0, "end_sec_exclusive": 3600.0}],
        ordinary_negative=[{"participant_id": "u1", "session_id": "s1", "clock_domain_id": "c1", "start_sec": 0.0, "end_sec_exclusive": 3600.0}],
        positive=[{"participant_id": "u1", "session_id": "s1", "clock_domain_id": "c1", "start_sec": 600.0, "end_sec_exclusive": 1200.0}],
        truth_uncertain_or_excluded=[],
        qc_unavailable=[{"participant_id": "u1", "session_id": "s1", "clock_domain_id": "c1", "start_sec": 1800.0, "end_sec_exclusive": 2400.0}],
        tracking_unavailable=[],
        inference_error=[],
        tracklet_participant_bindings=[{"track_id": 1, "participant_id": "u1", "session_id": "s1", "clock_domain_id": "c1"}],
        clock_alignments=[{"session_id": "s1", "clock_domain_id": "c1", "status": "aligned"}],
        prediction_uncertain=[{"participant_id": "u1", "session_id": "s1", "clock_domain_id": "c1", "start_sec": 2400.0, "end_sec_exclusive": 3000.0}],
    )
    assert value["status"] == "computed"
    assert value["eligible_negative_seconds"] == 2400.0
    assert value["eligible_negative_person_hours"] == pytest.approx(2 / 3)
    assert value["prediction_uncertain_subtracted"] is False

    missing = compute_eligible_negative_person_hours(
        participant_present=[],
        ordinary_negative=[],
        positive=[],
        truth_uncertain_or_excluded=[],
        qc_unavailable=[],
        tracking_unavailable=[],
        inference_error=[],
        tracklet_participant_bindings=[],
        clock_alignments=[],
        prediction_uncertain=[],
    )
    assert missing["status"] == "not_computable"


def test_negative_person_hours_sum_equal_relative_time_across_sessions() -> None:
    groups = [
        {"participant_id": "u1", "session_id": session, "clock_domain_id": f"clock-{session}"}
        for session in ("s1", "s2")
    ]
    intervals = [{**group, "start_sec": 0.0, "end_sec_exclusive": 3600.0} for group in groups]
    value = compute_eligible_negative_person_hours(
        participant_present=intervals,
        ordinary_negative=intervals,
        positive=[],
        truth_uncertain_or_excluded=[],
        qc_unavailable=[],
        tracking_unavailable=[],
        inference_error=[],
        tracklet_participant_bindings=[
            {**group, "track_id": index} for index, group in enumerate(groups, start=1)
        ],
        clock_alignments=[
            {"session_id": group["session_id"], "clock_domain_id": group["clock_domain_id"], "status": "aligned"}
            for group in groups
        ],
        prediction_uncertain=intervals,
    )
    assert value["status"] == "computed"
    assert value["eligible_negative_person_hours"] == pytest.approx(2.0)
    assert value["prediction_uncertain_subtracted"] is False


def test_evaluator_separates_truth_mask_abstention_unavailable_error_and_miss() -> None:
    scope = _scope()
    annotations = [
        {"annotation_id": "accepted-abstain", **scope, "start_sec": 0.0, "end_sec_exclusive": 10.0, "observable_pattern": "pacing", "evaluation_role": "wandering_like_positive", "annotation_status": "accepted", "purpose_context": "nonpurposeful"},
        {"annotation_id": "accepted-unavailable", **scope, "start_sec": 10.0, "end_sec_exclusive": 20.0, "observable_pattern": "direct", "evaluation_role": "ordinary_negative", "annotation_status": "accepted", "purpose_context": "nonpurposeful"},
        {"annotation_id": "accepted-error", **scope, "start_sec": 20.0, "end_sec_exclusive": 30.0, "observable_pattern": "direct", "evaluation_role": "ordinary_negative", "annotation_status": "accepted", "purpose_context": "nonpurposeful"},
        {"annotation_id": "accepted-miss", **scope, "start_sec": 30.0, "end_sec_exclusive": 40.0, "observable_pattern": "random", "evaluation_role": "wandering_like_positive", "annotation_status": "accepted", "purpose_context": "nonpurposeful"},
        {"annotation_id": "truth-mask", **scope, "start_sec": 40.0, "end_sec_exclusive": 50.0, "observable_pattern": "unknown", "evaluation_role": "uncertain", "annotation_status": "uncertain", "purpose_context": "unknown"},
    ]
    episodes = [
        {"episode_candidate_id": "episode-abstain", **scope, "episode_start_sec": 0.0, "episode_end_sec_exclusive": 10.0, "predicted_pattern": "pacing", "four_class_probability_summary": {"mean_probabilities": [0.1, 0.4, 0.3, 0.2]}},
        {"episode_candidate_id": "episode-masked", **scope, "episode_start_sec": 40.0, "episode_end_sec_exclusive": 50.0, "predicted_pattern": "pacing", "four_class_probability_summary": {"mean_probabilities": [0.1, 0.7, 0.1, 0.1]}},
    ]
    windows = [
        {"window_id": "unavailable", **scope, "window_start_sec": 10.0, "window_end_sec": 20.0, "window_status": "unavailable"},
        {"window_id": "error", **scope, "window_start_sec": 20.0, "window_end_sec": 30.0, "window_status": "inference_error"},
    ]
    result = evaluate_labeled_episodes(
        episodes,
        annotations,
        windows,
        {
            "matching_policy": _matching_policy(),
            "uncertain_policy": {"policy_id": "uncertain-test-v1", "threshold": 0.5},
            "episode_merge_policy": {"policy_id": "merge-test-v1", "merge_gap_seconds": 0.0},
        },
    )
    assert result["truth_masking"]["prediction_count"] == 1
    assert result["accepted_truth_dispositions"] == {
        "matched_scored": 0,
        "explicit_model_abstention": 1,
        "unavailable": 1,
        "inference_error": 1,
        "no_prediction_miss": 1,
    }


def test_stage_timing_fields_are_explicit_and_nonnegative() -> None:
    result = build_stage_timings(
        receipt_ms=1,
        collection_ms=2,
        source_preflight_ms=3,
        tracking_annotation_cohort_ms=4,
        candidate_load_ms=5,
        qc_preprocess_ms=6,
        forward_ms=7,
        evaluator_ms=8,
        artifact_write_ms=9,
    )
    assert result["total_ms"] == 45.0
    assert list(result["stages"]) == [
        "receipt",
        "collection",
        "source_preflight",
        "tracking_annotation_cohort",
        "candidate_load",
        "qc_preprocess",
        "forward",
        "evaluator",
        "artifact_write",
    ]


def test_receipt_failure_happens_before_all_data_and_runtime_calls(tmp_path: Path) -> None:
    calls: list[str] = []

    def called(name: str):
        def _inner(*args, **kwargs):
            calls.append(name)
            return {}

        return _inner

    with pytest.raises(CameraDevelopmentError, match="receipt"):
        run_authorized_camera_development(
            project_root=ROOT,
            config_path=CONFIG,
            receipt_path=tmp_path / "missing-receipt.json",
            collection_path=tmp_path / "missing-collection.json",
            tracking_path=tmp_path / "missing-tracking.jsonl",
            media_sidecar_path=tmp_path / "missing-sidecar.json",
            candidate_manifest_path=tmp_path / "missing-manifest.json",
            expected_manifest_sha256="0" * 64,
            output_dir=tmp_path / "final",
            _test_hooks={
                "read_collection": called("collection"),
                "read_tracking": called("tracking"),
                "read_annotations": called("annotations"),
                "load_candidate": called("loader"),
                "forward": called("forward"),
            },
        )
    assert calls == []
    assert not (tmp_path / "final").exists()
    assert not list(tmp_path.glob("final.staging-*"))


def test_expired_receipt_stops_before_collection_tracking_loader_and_output(tmp_path: Path) -> None:
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(
        json.dumps(_receipt(expires_at="2026-02-01T00:00:00+08:00")), encoding="utf-8"
    )
    calls: list[str] = []

    def called(*args, **kwargs):
        calls.append("called")
        return {}

    with pytest.raises(CameraDevelopmentError, match="receipt"):
        run_authorized_camera_development(
            project_root=ROOT,
            config_path=CONFIG,
            receipt_path=receipt_path,
            collection_path=tmp_path / "collection.json",
            tracking_path=tmp_path / "tracking.jsonl",
            media_sidecar_path=tmp_path / "sidecar.json",
            candidate_manifest_path=tmp_path / "manifest.json",
            expected_manifest_sha256=MANIFEST_SHA,
            output_dir=tmp_path / "final",
            _test_hooks={
                name: called
                for name in (
                    "read_collection",
                    "read_sidecar",
                    "source_preflight",
                    "read_tracking",
                    "load_candidate",
                    "forward",
                )
            },
        )
    assert calls == []
    assert not (tmp_path / "final").exists()
    assert not list(tmp_path.glob("final.staging-*"))


@pytest.mark.parametrize(
    ("collection", "message"),
    [
        (_collection(receipt_id="receipt-development-mismatch"), "receipt reference"),
        (_collection(participant_id="participant-0002"), "scope mismatch"),
        (_collection(dataset_role="sealed"), "sealed"),
    ],
)
def test_collection_receipt_id_and_sealed_role_stop_before_sidecar_tracking_and_loader(
    tmp_path: Path, collection: dict, message: str
) -> None:
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(_receipt()), encoding="utf-8")
    calls: list[str] = []

    def collection_read():
        calls.append("collection")
        return collection

    def forbidden(*args, **kwargs):
        calls.append("forbidden")
        return {}

    with pytest.raises(CameraDevelopmentError, match=message):
        run_authorized_camera_development(
            project_root=ROOT,
            config_path=CONFIG,
            receipt_path=receipt_path,
            collection_path=tmp_path / "collection.json",
            tracking_path=tmp_path / "tracking.jsonl",
            media_sidecar_path=tmp_path / "sidecar.json",
            candidate_manifest_path=tmp_path / "manifest.json",
            expected_manifest_sha256=MANIFEST_SHA,
            output_dir=tmp_path / "final",
            _test_hooks={
                "read_collection": collection_read,
                "read_sidecar": forbidden,
                "source_preflight": forbidden,
                "read_tracking": forbidden,
                "load_candidate": forbidden,
                "forward": forbidden,
            },
        )
    assert calls == ["collection"]
    assert not (tmp_path / "final").exists()
    assert not list(tmp_path.glob("final.staging-*"))


def test_labeled_entry_requires_all_policies_after_receipt_before_collection(tmp_path: Path) -> None:
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(_receipt()), encoding="utf-8")
    calls: list[str] = []
    with pytest.raises(CameraDevelopmentError, match="matching/uncertain/episode merge"):
        run_authorized_camera_development(
            project_root=ROOT,
            config_path=CONFIG,
            receipt_path=receipt_path,
            collection_path=tmp_path / "collection.json",
            tracking_path=tmp_path / "tracking.jsonl",
            media_sidecar_path=tmp_path / "sidecar.json",
            candidate_manifest_path=tmp_path / "manifest.json",
            expected_manifest_sha256=MANIFEST_SHA,
            output_dir=tmp_path / "final",
            mode="labeled_evaluation",
            _test_hooks={"read_collection": lambda: calls.append("collection")},
        )
    assert calls == []
    assert not (tmp_path / "final").exists()


def test_production_rejects_synthetic_before_candidate_or_output(tmp_path: Path) -> None:
    receipt = {
        "schema_version": "wandering-camera-authorization-receipt-v1",
        "receipt_id": "synthetic-fixture",
    }
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(CameraDevelopmentError):
        run_authorized_camera_development(
            project_root=ROOT,
            config_path=CONFIG,
            receipt_path=receipt_path,
            collection_path=tmp_path / "collection.json",
            tracking_path=tmp_path / "tracking.jsonl",
            media_sidecar_path=tmp_path / "sidecar.json",
            candidate_manifest_path=tmp_path / "manifest.json",
            expected_manifest_sha256="0" * 64,
            output_dir=tmp_path / "final",
        )
    assert not (tmp_path / "final").exists()


@pytest.mark.parametrize("labeled", [False, True])
def test_production_validators_support_authorized_success_as_test_fixture_only(
    tmp_path: Path, labeled: bool
) -> None:
    fixture = _authorized_fixture(tmp_path, labeled=labeled)
    output = tmp_path / "final"
    result = run_authorized_camera_development(
        project_root=ROOT,
        config_path=CONFIG,
        receipt_path=fixture["receipt"],
        collection_path=fixture["collection"],
        tracking_path=fixture["tracking"],
        media_sidecar_path=fixture["sidecar"],
        candidate_manifest_path=ROOT
        / "reports/mental_health/wandering_performance/m0r_score_entry_hardening_v1/artifacts/topowander_m0r_candidate_v3/candidate_manifest.json",
        expected_manifest_sha256=MANIFEST_SHA,
        output_dir=output,
        mode="labeled_evaluation" if labeled else "engineering_smoke",
        annotations_path=fixture["annotations"] if labeled else None,
        evaluation_policy=fixture["policy"] if labeled else None,
        _candidate_loader=_fake_candidate_loader,
        _test_fixture=True,
    )
    assert result["status"] == "wandering_m0cam_development_entry_hardened_waiting_c0_c1"
    assert result["evidence_scope"] == "test_fixture_only"
    execution = json.loads((output / "execution.json").read_text(encoding="utf-8"))
    assert execution["evidence_scope"] == "test_fixture_only"
    assert execution["test_fixture_only"] is True
    assert execution["authorization_summary"]["approved"] is True
    assert execution["authorized_camera_data_consumed"] is False
    expected_authorized_scope = (
        "authorized_labeled_development_evaluated"
        if labeled
        else "authorized_development_smoke"
    )
    assert execution["authorized_scope_if_non_fixture"] == expected_authorized_scope
    predictions = [
        json.loads(line)
        for line in (output / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert predictions
    assert {row["evidence_scope"] for row in predictions} == {"test_fixture_only"}
    episodes = [
        json.loads(line)
        for line in (output / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    if labeled:
        assert episodes
        assert {row["evidence_scope"] for row in episodes} == {"test_fixture_only"}
        evaluation = json.loads((output / "evaluation.json").read_text(encoding="utf-8"))
        assert evaluation["accepted_truth_dispositions"]["matched_scored"] == 2
        assert evaluation["accepted_truth_dispositions"]["unavailable"] == 1
        assert evaluation["eligible_negative_person_hours"]["status"] == "computed"
        assert evaluation["eligible_negative_person_hours"][
            "prediction_uncertain_subtracted"
        ] is False
    else:
        assert episodes == []
        assert not (output / "evaluation.json").exists()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["evidence_scope"] == "test_fixture_only"
    assert manifest["test_fixture_only"] is True
    assert not list(tmp_path.glob("final.staging-*"))
    with pytest.raises(CameraDevelopmentError, match="already exists"):
        run_authorized_camera_development(
            project_root=ROOT,
            config_path=CONFIG,
            receipt_path=fixture["receipt"],
            collection_path=fixture["collection"],
            tracking_path=fixture["tracking"],
            media_sidecar_path=fixture["sidecar"],
            candidate_manifest_path=ROOT
            / "reports/mental_health/wandering_performance/m0r_score_entry_hardening_v1/artifacts/topowander_m0r_candidate_v3/candidate_manifest.json",
            expected_manifest_sha256=MANIFEST_SHA,
            output_dir=output,
            _candidate_loader=_fake_candidate_loader,
            _test_fixture=True,
        )


@pytest.mark.parametrize(
    "mutation",
    ["unbound", "wrong_setup", "duplicate", "missing_clock"],
)
def test_labeled_production_path_rejects_incomplete_or_wrong_c3_before_loader(
    tmp_path: Path, mutation: str
) -> None:
    fixture = _authorized_fixture(tmp_path, labeled=True)
    collection_path = fixture["collection"]
    collection = json.loads(collection_path.read_text(encoding="utf-8"))
    if mutation == "unbound":
        collection["tracklet_participant_bindings"].pop()
    elif mutation == "wrong_setup":
        collection["tracklet_participant_bindings"][0]["setup_id"] = "wrong-setup"
    elif mutation == "duplicate":
        collection["tracklet_participant_bindings"].append(
            dict(collection["tracklet_participant_bindings"][0])
        )
    else:
        collection["clock_alignments"] = []
    collection_path.write_text(json.dumps(collection), encoding="utf-8")
    loader_calls: list[str] = []

    def forbidden_loader(*args, **kwargs):
        loader_calls.append("loader")
        raise AssertionError("candidate loader must not run")

    output = tmp_path / "rejected"
    with pytest.raises(CameraDevelopmentError):
        run_authorized_camera_development(
            project_root=ROOT,
            config_path=CONFIG,
            receipt_path=fixture["receipt"],
            collection_path=collection_path,
            tracking_path=fixture["tracking"],
            media_sidecar_path=fixture["sidecar"],
            candidate_manifest_path=ROOT
            / "reports/mental_health/wandering_performance/m0r_score_entry_hardening_v1/artifacts/topowander_m0r_candidate_v3/candidate_manifest.json",
            expected_manifest_sha256=MANIFEST_SHA,
            output_dir=output,
            mode="labeled_evaluation",
            annotations_path=fixture["annotations"],
            evaluation_policy=fixture["policy"],
            _candidate_loader=forbidden_loader,
            _test_fixture=True,
        )
    assert loader_calls == []
    assert not output.exists()
    assert not list(tmp_path.glob("rejected.staging-*"))


def test_labeled_authorization_is_maximum_use_not_automatic_evidence_upgrade(
    tmp_path: Path,
) -> None:
    fixture = _authorized_fixture(tmp_path, labeled=True)
    output = tmp_path / "smoke-with-labeled-authorization"
    result = run_authorized_camera_development(
        project_root=ROOT,
        config_path=CONFIG,
        receipt_path=fixture["receipt"],
        collection_path=fixture["collection"],
        tracking_path=fixture["tracking"],
        media_sidecar_path=fixture["sidecar"],
        candidate_manifest_path=ROOT
        / "reports/mental_health/wandering_performance/m0r_score_entry_hardening_v1/artifacts/topowander_m0r_candidate_v3/candidate_manifest.json",
        expected_manifest_sha256=MANIFEST_SHA,
        output_dir=output,
        mode="engineering_smoke",
        _candidate_loader=_fake_candidate_loader,
        _test_fixture=True,
    )
    assert result["evidence_scope"] == "test_fixture_only"
    execution = json.loads((output / "execution.json").read_text(encoding="utf-8"))
    assert execution["validation_scope"] == "authorized_camera_labeled_evaluation"
    assert execution["authorized_scope_if_non_fixture"] == "authorized_development_smoke"


def test_development_cli_has_no_synthetic_or_bypass_flags() -> None:
    text = (ROOT / "scripts/wandering/run_camera_development.py").read_text(encoding="utf-8")
    for forbidden in ("allow-synthetic", "fake-runtime", "public-loader", "bypass"):
        assert forbidden not in text
