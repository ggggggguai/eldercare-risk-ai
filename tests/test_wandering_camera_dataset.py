from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from elderly_monitoring.modules.mental_health.wandering.camera_dataset import (
    CameraDatasetError,
    build_camera_readiness,
    load_camera_collection_config,
    prepare_authorized_camera_session,
    prepare_camera_input_pair,
    validate_authorization_receipt,
    validate_collection_manifest,
    validate_episode_annotations,
    write_authorized_camera_annotations,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/data/wandering_camera_collection_v1.yaml"


def _config() -> dict:
    return load_camera_collection_config(CONFIG)


def _receipt() -> dict:
    return {
        "schema_version": "wandering-camera-authorization-receipt-v1",
        "receipt_id": "receipt-development-0001",
        "approval_status": "approved",
        "active": True,
        "purpose": "camera_development",
        "dataset_role": "development",
        "valid_from": "2026-08-01T00:00:00+08:00",
        "expires_at": "2026-09-01T00:00:00+08:00",
        "allowed_operations": ["prepare_session", "validate_annotations", "run_development"],
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


def _collection(*, dataset_role: str = "development") -> dict:
    return {
        "schema_version": "wandering-camera-collection-v1",
        "collection_id": "collection-0001",
        "dataset_role": dataset_role,
        "authorization_receipt_id": "receipt-development-0001",
        "participants": [{"participant_id": "participant-0001"}],
        "camera_setups": [
            {"camera_setup_id": "setup-0001", "setup_id": "adapter-setup-0001"}
        ],
        "sessions": [
            {
                "session_id": "session-0001",
                "participant_id": "participant-0001",
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
        "tracklet_participant_bindings": [
            {
                "source_group_id": "group-0001",
                "source_video_id": "video-0001",
                "device_id": "device-0001",
                "setup_id": "adapter-setup-0001",
                "stream_epoch": "epoch-0001",
                "track_id": 7,
                "participant_id": "participant-0001",
                "session_id": "session-0001",
                "camera_setup_id": "setup-0001",
                "clock_domain_id": "clock-0001",
            }
        ],
        "participant_present_intervals": [
            {
                "participant_id": "participant-0001",
                "session_id": "session-0001",
                "clock_domain_id": "clock-0001",
                "start_sec": 0.0,
                "end_sec_exclusive": 60.0,
            }
        ],
        "clock_alignments": [
            {
                "clock_domain_id": "clock-0001",
                "session_id": "session-0001",
                "source_group_id": "group-0001",
                "camera_setup_ids": ["setup-0001"],
                "source_video_ids": ["video-0001"],
                "status": "aligned",
            }
        ],
    }


def _annotation(**updates: object) -> dict:
    value = {
        "annotation_id": "annotation-0001",
        "source_group_id": "group-0001",
        "source_video_id": "video-0001",
        "device_id": "device-0001",
        "setup_id": "adapter-setup-0001",
        "stream_epoch": "epoch-0001",
        "track_id": 7,
        "participant_id": "participant-0001",
        "session_id": "session-0001",
        "camera_setup_id": "setup-0001",
        "clock_domain_id": "clock-0001",
        "start_sec": 10.0,
        "end_sec_exclusive": 20.0,
        "observable_pattern": "pacing",
        "purpose_context": "purposeful",
        "purpose_evidence": "participant_report",
        "evaluation_role": "purposeful_hard_negative",
        "script_type": "phone_call",
        "visibility_quality": "good",
        "tracking_issue": "none",
        "annotation_status": "accepted",
        "annotator_id": "annotator-0001",
        "reviewed_by": "reviewer-0001",
    }
    value.update(updates)
    return value


def test_fixed_collection_config_loads_exact_contract() -> None:
    config = _config()
    assert config["schema_version"] == "wandering-camera-collection-config-v1"
    assert config["dataset_roles"] == ["development", "sealed"]
    assert config["readiness_milestones"] == ["C0", "C1", "C2", "C3"]
    assert "matching_policy" not in config
    assert "uncertain_threshold" not in config


def test_receipt_requires_active_approved_development_scope() -> None:
    summary = validate_authorization_receipt(
        _receipt(),
        _config(),
        operation="run_development",
        participant_ids={"participant-0001"},
        session_ids={"session-0001"},
        camera_setup_ids={"setup-0001"},
        source_group_ids={"group-0001"},
        now="2026-08-13T12:00:00+08:00",
    )
    assert summary["receipt_id"] == "receipt-development-0001"
    assert summary["approved"] is True
    assert "participant_ids" not in summary
    assert len(summary["receipt_sha256"]) == 64

    for field, bad in (
        ("approval_status", "pending"),
        ("active", False),
        ("purpose", "camera_sealed_evaluation"),
        ("dataset_role", "sealed"),
    ):
        receipt = _receipt()
        receipt[field] = bad
        with pytest.raises(CameraDatasetError):
            validate_authorization_receipt(
                receipt,
                _config(),
                operation="run_development",
                now="2026-08-13T12:00:00+08:00",
            )

    with pytest.raises(CameraDatasetError, match="expired"):
        validate_authorization_receipt(
            _receipt(),
            _config(),
            operation="run_development",
            now="2026-09-02T00:00:00+08:00",
        )
    with pytest.raises(CameraDatasetError, match="scope"):
        validate_authorization_receipt(
            _receipt(),
            _config(),
            operation="run_development",
            participant_ids={"participant-9999"},
            now="2026-08-13T12:00:00+08:00",
        )


def test_collection_cross_references_group_isolation_and_privacy() -> None:
    value = validate_collection_manifest(_collection(), _config())
    assert value["sources"][0]["source_group_id"] == value["sessions"][0]["source_group_id"]

    bad = _collection()
    bad["sources"][0]["source_group_id"] = "group-crossed"
    with pytest.raises(CameraDatasetError, match="source_group"):
        validate_collection_manifest(bad, _config())

    bad = _collection()
    bad["participants"].append({"participant_id": "participant-0001"})
    with pytest.raises(CameraDatasetError, match="duplicate"):
        validate_collection_manifest(bad, _config())

    for forbidden in (
        "https://camera.example/live?token=secret",
        "C:/Users/operator/video.mp4",
        "/home/operator/video.mp4",
    ):
        bad = _collection()
        bad["sources"][0]["tracking_ref"] = forbidden
        with pytest.raises(CameraDatasetError):
            validate_collection_manifest(bad, _config())


def test_c3_binding_covers_full_tracking_scope_session_setup_and_clock() -> None:
    value = validate_collection_manifest(_collection(), _config())
    binding = value["tracklet_participant_bindings"][0]
    assert {
        "source_group_id",
        "source_video_id",
        "device_id",
        "setup_id",
        "stream_epoch",
        "track_id",
        "participant_id",
        "session_id",
        "camera_setup_id",
        "clock_domain_id",
    } == set(binding)

    for field, bad_value in (
        ("session_id", "session-wrong"),
        ("camera_setup_id", "setup-wrong"),
        ("setup_id", "adapter-setup-wrong"),
        ("clock_domain_id", "clock-wrong"),
    ):
        bad = _collection()
        bad["tracklet_participant_bindings"][0][field] = bad_value
        with pytest.raises(CameraDatasetError, match="binding"):
            validate_collection_manifest(bad, _config())

    duplicate = _collection()
    duplicate["tracklet_participant_bindings"].append(
        copy.deepcopy(duplicate["tracklet_participant_bindings"][0])
    )
    with pytest.raises(CameraDatasetError, match="duplicate tracklet binding"):
        validate_collection_manifest(duplicate, _config())


def test_annotations_keep_shape_and_purpose_separate_and_preserve_truth_states() -> None:
    collection = validate_collection_manifest(_collection(), _config())
    rows = validate_episode_annotations(
        [
            _annotation(),
            _annotation(
                annotation_id="annotation-0002",
                start_sec=20.0,
                end_sec_exclusive=25.0,
                observable_pattern="unknown",
                purpose_context="unknown",
                purpose_evidence="unknown",
                evaluation_role="uncertain",
                annotation_status="uncertain",
            ),
            _annotation(
                annotation_id="annotation-0003",
                start_sec=25.0,
                end_sec_exclusive=30.0,
                observable_pattern="unknown",
                purpose_context="unknown",
                purpose_evidence="unknown",
                evaluation_role="excluded",
                annotation_status="excluded",
            ),
        ],
        collection,
    )
    assert rows[0]["observable_pattern"] == "pacing"
    assert rows[0]["purpose_context"] == "purposeful"
    assert rows[0]["evaluation_role"] == "purposeful_hard_negative"
    assert [row["annotation_status"] for row in rows[1:]] == ["uncertain", "excluded"]

    bad = _annotation(purpose_context="unknown", purpose_evidence="participant_report")
    with pytest.raises(CameraDatasetError, match="purpose"):
        validate_episode_annotations([bad], collection)
    bad = _annotation(annotation_id="annotation-duplicate")
    with pytest.raises(CameraDatasetError, match="overlap"):
        validate_episode_annotations([bad, copy.deepcopy(bad)], collection)

    accepted_unknown = _annotation(
        observable_pattern="unknown",
        purpose_context="unknown",
        purpose_evidence="unknown",
        evaluation_role="ordinary_negative",
    )
    with pytest.raises(CameraDatasetError, match="accepted truth.*scoreable shape"):
        validate_episode_annotations([accepted_unknown], collection)


def test_readiness_is_not_ready_without_real_c0_c1_c2_c3() -> None:
    report = build_camera_readiness(config=_config())
    assert report["readiness_status"] == "not_ready"
    assert report["authorized_camera_data_consumed"] is False
    assert report["milestones"] == {"C0": False, "C1": False, "C2": False, "C3": False}
    assert "metrics" not in report
    assert "f1" not in json.dumps(report).lower()


def test_prepare_pair_only_validates_and_writes_canonical_inputs(tmp_path: Path) -> None:
    source = ROOT / "reports/mental_health/wandering_m0cam_engineering_v1/fixtures"
    tracking = source / "synthetic_tracking.jsonl"
    sidecar = source / "synthetic_media_sidecar.json"
    output = tmp_path / "canonical"
    result = prepare_camera_input_pair(
        tracking,
        sidecar,
        output,
        camera_config=ROOT / "configs/modules/wandering_camera_v1.yaml",
    )
    assert result["input_kind"] == "caller_supplied_tracking_and_sidecar"
    assert (output / "tracking.jsonl").is_file()
    assert (output / "media_sidecar.json").is_file()
    assert (output / "preparation_summary.json").is_file()
    assert hashlib.sha256((output / "tracking.jsonl").read_bytes()).hexdigest() == result[
        "normalized_tracking_sha256"
    ]
    with pytest.raises(CameraDatasetError, match="already exists"):
        prepare_camera_input_pair(
            tracking,
            sidecar,
            output,
            camera_config=ROOT / "configs/modules/wandering_camera_v1.yaml",
        )


def test_prepare_session_is_receipt_first_and_zero_reads_on_invalid_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt_path = tmp_path / "receipt.json"
    receipt = _receipt()
    receipt["allowed_operations"] = ["validate_annotations"]
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    calls: list[str] = []

    def forbidden(*args, **kwargs):
        calls.append("protected_input")
        raise AssertionError("protected input must not be read")

    monkeypatch.setattr(
        "elderly_monitoring.modules.mental_health.wandering.camera_dataset.prepare_camera_input_pair",
        forbidden,
    )
    with pytest.raises(CameraDatasetError, match="requested operation"):
        prepare_authorized_camera_session(
            receipt_path=receipt_path,
            collection_path=tmp_path / "missing-collection.json",
            tracking_jsonl_path=tmp_path / "missing-tracking.jsonl",
            media_sidecar_path=tmp_path / "missing-sidecar.json",
            output_dir=tmp_path / "output",
            collection_config_path=CONFIG,
            camera_config_path=ROOT / "configs/modules/wandering_camera_v1.yaml",
        )
    assert calls == []
    assert not (tmp_path / "output").exists()


def test_annotation_builder_is_receipt_first_and_zero_reads_on_invalid_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt_path = tmp_path / "receipt.json"
    receipt = _receipt()
    receipt["active"] = False
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    calls: list[str] = []

    def forbidden(*args, **kwargs):
        calls.append("annotations")
        raise AssertionError("annotations must not be read")

    monkeypatch.setattr(
        "elderly_monitoring.modules.mental_health.wandering.camera_dataset._read_jsonl_objects",
        forbidden,
    )
    with pytest.raises(CameraDatasetError, match="not active"):
        write_authorized_camera_annotations(
            receipt_path=receipt_path,
            collection_path=tmp_path / "missing-collection.json",
            annotations_path=tmp_path / "missing-annotations.jsonl",
            output_path=tmp_path / "annotations.jsonl",
            collection_config_path=CONFIG,
        )
    assert calls == []
    assert not (tmp_path / "annotations.jsonl").exists()


def test_auxiliary_clis_require_receipt_and_expose_no_bypass_flags() -> None:
    prepare_text = (ROOT / "scripts/wandering/prepare_camera_session.py").read_text(
        encoding="utf-8"
    )
    annotation_text = (ROOT / "scripts/wandering/build_camera_annotations.py").read_text(
        encoding="utf-8"
    )
    for text in (prepare_text, annotation_text):
        assert "--receipt" in text
        assert "--collection" in text
        for forbidden in ("allow-synthetic", "fake-runtime", "bypass"):
            assert forbidden not in text
