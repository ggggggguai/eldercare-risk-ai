from __future__ import annotations

import hashlib
import json
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pytest
import numpy as np

from elderly_monitoring.modules.fall_risk.self_collected_scf import (
    _manifest_candidate,
    build_scf_mvp_v1_candidates,
    evaluate_scf_near_fall_gate,
    replay_fall_rule_video,
    replay_gait_video,
    replay_near_fall_video,
)
from elderly_monitoring.modules.fall_risk.near_fall_self_collected import (
    EXPERIMENT_ACTIONS,
    _write_deterministic_npz,
    select_scf_near_fall_candidates,
)
from scripts.evaluate.evaluate_self_collected_near_fall_augmentation import _wilson


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_export(
    path: Path,
    *,
    source: str,
    task_size: int,
    width: int,
    height: int,
    label: str,
    bbox: tuple[float, float, float, float],
) -> None:
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"
    project = ET.SubElement(ET.SubElement(root, "meta"), "project")
    tasks = ET.SubElement(project, "tasks")
    task = ET.SubElement(tasks, "task")
    for name, value in (
        ("id", "1"),
        ("name", source),
        ("size", str(task_size)),
        ("mode", "interpolation"),
        ("start_frame", "0"),
        ("stop_frame", str(task_size - 1)),
        ("source", source),
    ):
        ET.SubElement(task, name).text = value
    original_size = ET.SubElement(task, "original_size")
    ET.SubElement(original_size, "width").text = str(width)
    ET.SubElement(original_size, "height").text = str(height)
    track = ET.SubElement(
        root,
        "track",
        {"id": "0", "task_id": "1", "label": label},
    )
    for frame, outside in ((0, "0"), (task_size - 1, "0")):
        box = ET.SubElement(
            track,
            "box",
            {
                "frame": str(frame),
                "outside": outside,
                "occluded": "0",
                "keyframe": "1",
                "xtl": str(bbox[0]),
                "ytl": str(bbox[1]),
                "xbr": str(bbox[2]),
                "ybr": str(bbox[3]),
            },
        )
        for name, value in (
            ("quality", "clear"),
            ("target_subject", "unknown"),
            ("note", ""),
        ):
            attribute = ET.SubElement(box, "attribute", {"name": name})
            attribute.text = value
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "annotations.xml",
            ET.tostring(root, encoding="utf-8", xml_declaration=True),
        )


def _fixture(root: Path, *, subject: str = "P02", frame_delta: int = 0) -> dict[str, Path]:
    filename = f"{subject}_S01_E001_N02_A02.mp4"
    video = root / "videos" / filename
    video.parent.mkdir(parents=True)
    video.write_bytes(b"fixture-video")
    export = root / f"{subject}_cvat_redacted.zip"
    cvat_width, cvat_height = ((640, 360) if subject == "P02" else (2560, 1440))
    _write_export(
        export,
        source=filename,
        task_size=10,
        width=cvat_width,
        height=cvat_height,
        label="A02_normal_turn",
        bbox=(10, 20, 30, 40),
    )
    inventory = root / "inventory.jsonl"
    _write_jsonl(
        inventory,
        [
            {
                "schema_version": "self-collected-delivery-inventory-v1",
                "batch_id": "SCF_MVP_V1",
                "file_name": filename,
                "canonical_relative_path": video.as_posix(),
                "sha256": _sha256(video),
                "subject": subject,
                "session": "S01",
                "episode": "E001",
                "script": "N02",
                "action": "A02",
                "fps_num": 15,
                "fps_den": 1,
                "fps": 15.0,
                "frame_count": 10 + frame_delta,
                "duration_sec": (10 + frame_delta) / 15,
                "width": 2560,
                "height": 1440,
                "rotation": 0,
                "cvat_task_id": "1",
                "cvat_task_size": 10,
                "cvat_width": cvat_width,
                "cvat_height": cvat_height,
                "cvat_frame_delta": frame_delta,
                "annotation_archive": f"{subject}.zip",
                "annotation_archive_sha256": "inner-source-hash",
                "disposition": "engineering_replay_only",
                "training_eligible": False,
                "training_blockers": [
                    "authorization_evidence_missing",
                    "subject_profile_missing",
                    "formal_label_review_missing",
                    "split_not_frozen",
                ],
                "target_subject_values": {"unknown": 2},
                "duplicate_content_files": [],
            }
        ],
    )
    decision = root / "decision.json"
    decision.write_text(
        json.dumps(
            {
                "schema_version": "self-collected-scf-mvp-v1-import-decision-v1",
                "batch_id": "SCF_MVP_V1",
                "inventory_sha256": _sha256(inventory),
                "annotation_exports": {
                    subject: {"path": export.as_posix(), "sha256": _sha256(export)}
                },
                "frame_mappings": {},
                "authorization_evidence": {},
                "subject_profiles": {},
                "double_reviewed_files": [],
                "frozen_split_id": None,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {"inventory": inventory, "decision": decision}


def test_candidate_import_scales_p02_and_stays_loss_ineligible() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        paths = _fixture(root)
        result = build_scf_mvp_v1_candidates(
            inventory_path=paths["inventory"],
            decision_path=paths["decision"],
            output_dir=root / "output",
        )
        action = json.loads(Path(result["action_labels_path"]).read_text().splitlines()[0])
        negative = json.loads(
            Path(result["near_fall_candidates_path"]).read_text().splitlines()[0]
        )
        report = json.loads(Path(result["report_path"]).read_text())

    assert action["bbox_start"] == [40.0, 80.0, 120.0, 160.0]
    assert action["coordinate_transform"] == "scale_xy_4_4"
    assert action["training_tier"] == "ignore"
    assert negative["hard_negative_type"] == "normal_turn"
    assert negative["loss_eligible"] is False
    assert report["root_artifacts_modified"] is False
    assert report["g0"]["status"] == "blocked"


def test_candidate_import_rejects_unmapped_frame_delta_and_hash_drift() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        paths = _fixture(root, subject="P03", frame_delta=3)
        with pytest.raises(ValueError, match="explicit frame mapping"):
            build_scf_mvp_v1_candidates(
                inventory_path=paths["inventory"],
                decision_path=paths["decision"],
                output_dir=root / "output",
            )

        decision = json.loads(paths["decision"].read_text())
        decision["frame_mappings"] = {
            "P03_S01_E001_N02_A02.mp4": {
                "strategy": "identity_with_trailing_media_frames",
                "trailing_media_frames": 3,
            }
        }
        paths["decision"].write_text(json.dumps(decision), encoding="utf-8")
        build_scf_mvp_v1_candidates(
            inventory_path=paths["inventory"],
            decision_path=paths["decision"],
            output_dir=root / "mapped",
        )

        paths["inventory"].write_text(paths["inventory"].read_text() + "\n")
        with pytest.raises(ValueError, match="inventory SHA-256 mismatch"):
            build_scf_mvp_v1_candidates(
                inventory_path=paths["inventory"],
                decision_path=paths["decision"],
                output_dir=root / "drifted",
            )


def test_duplicate_content_uses_one_protected_group() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        paths = _fixture(root)
        rows = [json.loads(paths["inventory"].read_text())]
        duplicate = dict(rows[0])
        duplicate["file_name"] = "P02_S02_E002_N02_A02.mp4"
        duplicate["session"] = "S02"
        duplicate["episode"] = "E002"
        decision = json.loads(paths["decision"].read_text())
        manifest = [
            _manifest_candidate(rows[0], decision),
            _manifest_candidate(duplicate, decision),
        ]

    assert len({row["sample_group_id"] for row in manifest}) == 1


def test_frozen_development_partition_and_experiment_inclusion() -> None:
    rows = []
    for subject in ("P01", "P02", "P03", "P04", "P05"):
        for action in ("A02", "A03", "A05", "A06", "A08", "A10", "C03", "C04", "C05"):
            rows.append(
                {
                    "candidate_id": f"{subject}-{action}",
                    "provisional_subject": subject,
                    "source_action_id": action,
                    "candidate_role": "positive_candidate" if action.startswith("C") else "hard_negative",
                    "sample_group_id": f"sha-{subject}-{action}",
                    "video_id": f"video-{subject}-{action}",
                    "start_frame": 0,
                    "loss_eligible": subject != "P03",
                    "disposition": "reviewed",
                }
            )

    for experiment, expected_actions in EXPERIMENT_ACTIONS.items():
        train = select_scf_near_fall_candidates(rows, experiment=experiment, partition="train")
        challenge = select_scf_near_fall_candidates(rows, experiment=experiment, partition="challenge")
        assert {row["provisional_subject"] for row in train} == {"P01", "P02", "P04"}
        assert {row["provisional_subject"] for row in challenge} == {"P05"}
        assert {row["source_action_id"] for row in train} == expected_actions
        assert not any(row["provisional_subject"] == "P03" for row in train + challenge)


def test_deterministic_npz_bytes_are_stable() -> None:
    arrays = {
        "features": np.arange(48, dtype=np.float32).reshape(2, 3, 2, 4),
        "labels": np.asarray([0, 1], dtype=np.int64),
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        first = Path(tmpdir) / "first.npz"
        second = Path(tmpdir) / "second.npz"
        _write_deterministic_npz(first, arrays)
        _write_deterministic_npz(second, arrays)
        assert _sha256(first) == _sha256(second)


def test_wilson_interval_contains_observed_rate() -> None:
    lower, upper = _wilson(3, 10)
    assert lower < 0.3 < upper
    assert _wilson(0, 0) == [0.0, 0.0]


def test_near_fall_gate_blocks_training_without_human_evidence() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        paths = _fixture(root)
        built = build_scf_mvp_v1_candidates(
            inventory_path=paths["inventory"],
            decision_path=paths["decision"],
            output_dir=root / "candidate",
        )
        baseline = root / "baseline.json"
        baseline.write_text(
            json.dumps({"status": "development_provisional", "seed": 42}),
            encoding="utf-8",
        )
        report = evaluate_scf_near_fall_gate(
            import_report_path=built["report_path"],
            near_fall_candidates_path=built["near_fall_candidates_path"],
            output_path=root / "gate.json",
            baseline_runs=[baseline],
        )

    assert report["status"] == "blocked"
    assert report["experiments"]["E0"]["status"] == "frozen_baseline_available"
    assert report["experiments"]["E1"]["status"] == "blocked"
    assert report["test_pose_read"] is False
    assert report["test_evaluated"] is False


def test_video_replay_reports_rule_and_frozen_model_without_fp_hour(monkeypatch) -> None:
    import elderly_monitoring.modules.fall_risk.near_fall_tcn as near_fall_tcn

    class _Predictor:
        def __init__(self, checkpoint_path: Path, *, device: str) -> None:
            assert device == "cpu"

        def predict_tensor(self, tensor):
            return {"status": "valid", "near_fall_event_score": 0.2}

    monkeypatch.setattr(near_fall_tcn, "NearFallTCNPredictor", _Predictor)
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        pose = root / "pose.jsonl"
        checkpoint = root / "checkpoint.pt"
        checkpoint.write_bytes(b"frozen")
        rows = []
        for frame in range(24):
            keypoints = []
            for index, name in enumerate(
                (
                    "left_shoulder", "right_shoulder", "left_wrist", "right_wrist",
                    "left_hip", "right_hip", "left_knee", "right_knee",
                    "left_ankle", "right_ankle",
                )
            ):
                keypoints.append(
                    {
                        "name": name,
                        "x": 0.4 + index * 0.01,
                        "y": 0.2 + index * 0.04,
                        "x_smooth": 0.4 + index * 0.01,
                        "y_smooth": 0.2 + index * 0.04,
                        "valid": True,
                        "quality_weight": 0.9,
                        "is_jump_outlier": False,
                    }
                )
            rows.append(
                {
                    "frame_id": frame,
                    "timestamp_sec": frame / 8,
                    "person_id": "p1",
                    "track_id": 1,
                    "pose_confidence": 0.9,
                    "keypoints": keypoints,
                    "window_quality": {"usable_for_near_fall": True},
                }
            )
        _write_jsonl(pose, rows)

        report = replay_near_fall_video(
            cleaned_pose_path=pose,
            checkpoint_paths=[checkpoint],
        )

    assert report["pose_record_count"] == 24
    assert report["tcn"][0]["valid_prediction_count"] == 1
    assert report["tcn"][0]["video_positive"] is False
    assert report["fp_hour_reported"] is False
    assert report["main_path_unchanged"] is True


def test_p03_rotation_maps_display_box_to_encoded_coordinates() -> None:
    from elderly_monitoring.modules.fall_risk.self_collected_scf import (
        _rotated_bbox_minus_90,
    )

    box = ET.Element(
        "box",
        {"xtl": "100", "ytl": "200", "xbr": "300", "ybr": "600"},
    )
    assert _rotated_bbox_minus_90(
        box,
        display_height=1920,
        inventory={"file_name": "P03.mp4", "width": 1920, "height": 1080},
    ) == [1320.0, 100.0, 1720.0, 300.0]


def test_gait_replay_keeps_rule_and_model_outputs_separate() -> None:
    class _Predictor:
        model_version = "gait-tcn-fixture"

        def predict_records(self, records):
            return 0.75

    rows = [_pose_row(frame, hip_y=0.4, trunk_horizontal=False) for frame in range(32)]
    report = replay_gait_video(
        records=rows, model_predictor=_Predictor(), model_threshold=0.4
    )

    assert report["rule"]["window_count"] == 1
    assert report["rule"]["threshold"] == 0.5
    assert report["tcn"]["valid_window_count"] == 1
    assert report["tcn"]["video_positive"] is True
    assert report["fp_hour_reported"] is False
    assert report["main_path_unchanged"] is True


def test_fall_rule_replay_uses_runtime_geometry_and_resets_on_gaps() -> None:
    upright = _pose_row(0, hip_y=0.35, trunk_horizontal=False)
    dropped = _pose_row(8, hip_y=0.65, trunk_horizontal=True)
    dropped["timestamp_sec"] = 0.5
    report = replay_fall_rule_video(records=[upright, dropped])

    assert report["track_count"] == 1
    assert report["analysis_interval_sec"] == 0.5
    assert report["trigger_count"] == 1
    assert report["video_positive"] is True
    assert report["test_pose_read"] is False

    after_gap = dict(dropped)
    after_gap["timestamp_sec"] = 2.0
    reset_report = replay_fall_rule_video(records=[upright, after_gap], max_gap_sec=0.75)
    assert reset_report["trigger_count"] == 0
    assert reset_report["streams"][0]["gap_reset_count"] == 1


def test_g1_action_stratum_uses_filename_action_not_first_cvat_track() -> None:
    from scripts.evaluate.replay_self_collected_g1 import _planned_actions

    rows = [
        {
            "video_id": "video-1",
            "file_name": "P03_S01_E001_N01_A01.mp4",
            "action_id": "U01",
        },
        {
            "video_id": "video-1",
            "file_name": "P03_S01_E001_N01_A01.mp4",
            "action_id": "A01",
        },
    ]

    assert _planned_actions(rows) == {"video-1": "A01"}


def _pose_row(frame: int, *, hip_y: float, trunk_horizontal: bool) -> dict[str, object]:
    shoulder_y = hip_y if trunk_horizontal else hip_y - 0.2
    points = []
    coordinates = {
        "left_shoulder": (0.3, shoulder_y),
        "right_shoulder": (0.5, shoulder_y),
        "left_hip": (0.4, hip_y),
        "right_hip": (0.6, hip_y),
        "left_knee": (0.4, hip_y + 0.15),
        "right_knee": (0.6, hip_y + 0.15),
        "left_ankle": (0.4, hip_y + 0.3),
        "right_ankle": (0.6, hip_y + 0.3),
    }
    for name, (x, y) in coordinates.items():
        points.append(
            {
                "name": name,
                "x": x,
                "y": y,
                "x_smooth": x,
                "y_smooth": y,
                "valid": True,
                "is_jump_outlier": False,
                "quality_weight": 0.95,
            }
        )
    return {
        "frame_id": frame,
        "timestamp_sec": frame / 8,
        "person_id": "p1",
        "track_id": 1,
        "keypoints": points,
        "bbox": [0.2, hip_y - 0.25, 0.7, hip_y + 0.35],
        "core_keypoint_quality": 0.95,
        "window_quality": {"usable_for_gait": True},
    }
