from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from elderly_monitoring.modules.fall_risk.sit_stand_training import (
    SIT_STAND_CHANNELS,
    SIT_STAND_JOINTS,
    SitStandDatasetConfig,
    audit_sit_stand_training_inputs,
    build_sit_stand_tensor,
    prepare_sit_stand_event_dataset,
)
from elderly_monitoring.modules.fall_risk.sit_stand_logistic import (
    SitStandLogisticConfig,
    evaluate_sit_stand_rule,
    train_sit_stand_logistic,
)
from elderly_monitoring.modules.fall_risk.sit_stand_tcn import (
    SitStandCandidateTCN,
    SitStandCandidateTCNConfig,
    build_sit_stand_direction_labels,
    compute_sit_stand_multitask_loss,
    evaluate_sit_stand_candidate_tcn,
    train_sit_stand_candidate_tcn,
)


BASE_JOINTS = SIT_STAND_JOINTS[:12]


def _pose_record(
    frame_id: int,
    *,
    direction: str = "static",
    usable: bool = True,
) -> dict[str, object]:
    progress = frame_id / 15.0
    if direction == "rise":
        hip_y = 0.70 - (0.24 * progress)
    elif direction == "sit":
        hip_y = 0.46 + (0.24 * progress)
    else:
        hip_y = 0.58
    shoulder_y = hip_y - 0.20
    ankle_y = 0.90
    knee_y = hip_y + ((ankle_y - hip_y) * 0.52)
    coordinates = {
        "left_shoulder": (0.45, shoulder_y),
        "right_shoulder": (0.55, shoulder_y),
        "left_elbow": (0.42, shoulder_y + 0.10),
        "right_elbow": (0.58, shoulder_y + 0.10),
        "left_wrist": (0.40, shoulder_y + 0.20),
        "right_wrist": (0.60, shoulder_y + 0.20),
        "left_hip": (0.46, hip_y),
        "right_hip": (0.54, hip_y),
        "left_knee": (0.46, knee_y),
        "right_knee": (0.54, knee_y),
        "left_ankle": (0.46, ankle_y),
        "right_ankle": (0.54, ankle_y),
    }
    quality = 0.95 if usable else 0.10
    return {
        "frame_id": frame_id,
        "timestamp_sec": frame_id / 4.0,
        "person_id": "person",
        "track_id": 1,
        "pose_confidence": quality,
        "core_keypoint_quality": quality,
        "window_quality": {
            "usable_for_sit_stand": usable,
            "mean_core_keypoint_quality": quality,
            "interpolated_point_ratio": 0.0,
            "jump_outlier_count": 0,
        },
        "keypoints": [
            {
                "name": name,
                "x": x,
                "y": y,
                "x_smooth": x if usable else None,
                "y_smooth": y if usable else None,
                "score": quality,
                "quality_weight": quality,
                "valid": usable,
                "source": "observed",
                "is_jump_outlier": False,
            }
            for name, (x, y) in coordinates.items()
        ],
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _label(
    index: int,
    *,
    action_id: str,
    partition: str,
    tier: str = "primary",
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    video_id = f"video_{index}"
    label_id = f"label_{index}"
    subject_id = f"subject_{index}"
    source_group_id = f"source_{index}"
    sample_group_id = f"sample_{index}"
    action_types = {
        "A03": "controlled_sit_down",
        "A04": "normal_sit_to_stand",
        "A05": "controlled_squat",
    }
    label = {
        "schema_version": "fall-risk-action-label-v3",
        "label_id": label_id,
        "asset_id": f"asset_{index}",
        "video_id": video_id,
        "subject_id": subject_id,
        "source_group_id": source_group_id,
        "sample_group_id": sample_group_id,
        "action_id": action_id,
        "action_type": action_types[action_id],
        "action_type_training_tier": tier,
        "training_tier": tier,
        "boundary_precision": "exact",
        "review_status": "single_annotated",
        "start_frame": 0,
        "end_frame_exclusive": 16,
        "start_time": 0.0,
        "end_time_exclusive": 4.0,
        "quality_flags": [],
        "linked_event_id": None,
    }
    manifest = {
        "asset_id": f"asset_{index}",
        "video_id": video_id,
        "path": f"unused/{video_id}.mp4",
        "sha256": f"source-sha-{index}",
        "dataset": "fixture",
        "scene_region": "room",
        "eligibility": True,
        "media_type": "video",
        "fps": 4.0,
        "frame_count": 16,
    }
    assignment = {
        "schema_version": "fall-risk-training-split-v3",
        "label_id": label_id,
        "label_kind": "action",
        "task_type": "action",
        "asset_id": f"asset_{index}",
        "video_id": video_id,
        "subject_id": subject_id,
        "source_group_id": source_group_id,
        "sample_group_id": sample_group_id,
        "split_group_id": f"split_{index}",
        "training_tier": tier,
        "partition": partition,
    }
    return label, manifest, assignment


def _fixture(root: Path, *, low_quality_index: int | None = None) -> dict[str, Path]:
    rows = [
        (1, "A04", "train"),
        (2, "A03", "train"),
        (3, "A05", "train"),
        (4, "A05", "train"),
        (5, "A04", "validation"),
        (6, "A03", "validation"),
        (7, "A05", "validation"),
        (8, "A05", "validation"),
        (9, "A04", "test"),
    ]
    labels: list[dict[str, object]] = []
    manifests: list[dict[str, object]] = []
    assignments: list[dict[str, object]] = []
    pose_dir = root / "poses"
    for index, action_id, partition in rows:
        label, manifest, assignment = _label(
            index, action_id=action_id, partition=partition
        )
        labels.append(label)
        manifests.append(manifest)
        assignments.append(assignment)
        if partition != "test":
            direction = "rise" if action_id == "A04" else "sit" if action_id == "A03" else "static"
            _write_jsonl(
                pose_dir / f"video_{index}.jsonl",
                [
                    _pose_record(
                        frame_id,
                        direction=direction,
                        usable=index != low_quality_index,
                    )
                    for frame_id in range(16)
                ],
            )
    labels_path = root / "labels.jsonl"
    manifest_path = root / "manifest.jsonl"
    assignments_path = root / "assignments.jsonl"
    split_path = root / "split.json"
    validation_path = root / "validation.json"
    _write_jsonl(labels_path, labels)
    _write_jsonl(manifest_path, manifests)
    _write_jsonl(assignments_path, assignments)
    split_path.write_text(
        json.dumps(
            {
                "schema_version": "fall-risk-training-split-v3",
                "split_id": "fixture-split",
                "assignments_sha256": _sha256(assignments_path),
                "input_sha256": {
                    "action_labels": _sha256(labels_path),
                    "manifest": _sha256(manifest_path),
                },
                "leakage_issues": [],
            }
        ),
        encoding="utf-8",
    )
    validation_path.write_text(
        json.dumps(
            {
                "valid": True,
                "training_ready": {"action_type": False},
                "input_sha256": {
                    "action_labels": _sha256(labels_path),
                    "manifest": _sha256(manifest_path),
                    "split_assignments": _sha256(assignments_path),
                    "split_report": _sha256(split_path),
                },
            }
        ),
        encoding="utf-8",
    )
    return {
        "labels": labels_path,
        "manifest": manifest_path,
        "assignments": assignments_path,
        "split": split_path,
        "validation": validation_path,
        "pose_dir": pose_dir,
    }


class SitStandTensorTest(unittest.TestCase):
    def test_tensor_has_14_joints_masks_and_preserved_image_height(self) -> None:
        records = [_pose_record(index, direction="rise") for index in range(16)]

        tensor = build_sit_stand_tensor(records, window_frames=16, target_fps=4.0)

        self.assertEqual(tensor.shape, (16, len(SIT_STAND_JOINTS), len(SIT_STAND_CHANNELS)))
        self.assertTrue(np.isfinite(tensor).all())
        self.assertTrue(np.all(tensor[..., -1] == 1.0))
        self.assertGreater(float(np.ptp(tensor[:, 12, 5])), 0.20)
        self.assertTrue(np.allclose(tensor[:, 12, :2], 0.0, atol=1e-6))

    def test_tensor_rejects_channel_contract_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "window_frames"):
            build_sit_stand_tensor([], window_frames=1, target_fps=4.0)


class SitStandPreparationTest(unittest.TestCase):
    def test_audit_blocks_localization_without_background_or_event_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))

            audit = audit_sit_stand_training_inputs(**paths)

        self.assertFalse(audit["training_gates"]["event_localization_ready"])
        self.assertTrue(audit["training_gates"]["event_presence_proxy_ready"])
        self.assertEqual(audit["evidence"]["explicit_background_label_count"], 0)
        self.assertEqual(audit["evidence"]["linked_event_label_count"], 0)
        self.assertFalse(audit["data_access"]["test_pose_read"])

    def test_builder_requires_explicit_provisional_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _fixture(root)
            with self.assertRaisesRegex(ValueError, "allow_provisional"):
                prepare_sit_stand_event_dataset(
                    **paths,
                    output_dir=root / "output",
                    config=SitStandDatasetConfig(),
                )

    def test_builder_never_requires_or_reads_test_pose(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _fixture(root)

            result = prepare_sit_stand_event_dataset(
                **paths,
                output_dir=root / "output",
                config=SitStandDatasetConfig(),
                allow_provisional=True,
            )
            metadata = json.loads(Path(result["metadata_path"]).read_text())
            with np.load(result["dataset_path"], allow_pickle=False) as archive:
                partitions = set(archive["partitions"].astype(str).tolist())
                segment_ids = archive["event_ids"].astype(str)
                weights = archive["sample_weights"].astype(float)

        self.assertEqual(partitions, {"train", "validation"})
        self.assertEqual(metadata["locked_test_label_count"], 1)
        self.assertFalse(metadata["test_pose_read"])
        for segment_id in set(segment_ids.tolist()):
            self.assertAlmostEqual(float(weights[segment_ids == segment_id].sum()), 1.0)

    def test_builder_rejects_missing_development_pose(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _fixture(root)
            (paths["pose_dir"] / "video_1.jsonl").unlink()

            with self.assertRaisesRegex(FileNotFoundError, "video_1"):
                prepare_sit_stand_event_dataset(
                    **paths,
                    output_dir=root / "output",
                    allow_provisional=True,
                )

    def test_audit_rejects_illegal_action_specific_tier(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _fixture(root)
            labels = [json.loads(line) for line in paths["labels"].read_text().splitlines()]
            labels[0]["action_type_training_tier"] = "made_up"
            _write_jsonl(paths["labels"], labels)

            with self.assertRaisesRegex(ValueError, "action_type_training_tier"):
                audit_sit_stand_training_inputs(**paths)

    def test_audit_rejects_subject_crossing_partitions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _fixture(root)
            labels = [json.loads(line) for line in paths["labels"].read_text().splitlines()]
            labels[4]["subject_id"] = labels[0]["subject_id"]
            _write_jsonl(paths["labels"], labels)

            with self.assertRaisesRegex(ValueError, "subject_id"):
                audit_sit_stand_training_inputs(**paths)

    def test_low_quality_windows_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _fixture(root, low_quality_index=1)

            result = prepare_sit_stand_event_dataset(
                **paths,
                output_dir=root / "output",
                allow_provisional=True,
            )
            metadata = json.loads(Path(result["metadata_path"]).read_text())

        self.assertGreater(metadata["rejected_window_counts"]["insufficient_quality"], 0)


class SitStandLogisticTrainingTest(unittest.TestCase):
    def test_provisional_training_checkpoint_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _fixture(root)
            prepared = prepare_sit_stand_event_dataset(
                **paths,
                output_dir=root / "prepared",
                allow_provisional=True,
            )
            with self.assertRaisesRegex(ValueError, "allow_provisional"):
                train_sit_stand_logistic(
                    prepared["dataset_path"],
                    root / "blocked-run",
                    metadata_path=prepared["metadata_path"],
                )

            first = train_sit_stand_logistic(
                prepared["dataset_path"],
                root / "run-1",
                metadata_path=prepared["metadata_path"],
                config=SitStandLogisticConfig(seed=42),
                allow_provisional=True,
            )
            resumed = train_sit_stand_logistic(
                prepared["dataset_path"],
                root / "run-2",
                metadata_path=prepared["metadata_path"],
                config=SitStandLogisticConfig(seed=42),
                allow_provisional=True,
                resume_checkpoint=first["checkpoint_path"],
            )
            metrics = json.loads(Path(first["metrics_path"]).read_text())
            config = json.loads((root / "run-1" / "config.json").read_text())
            self.assertTrue(Path(first["checkpoint_path"]).is_file())
            self.assertTrue(Path(first["failure_cases_path"]).is_file())
            self.assertFalse(metrics["test_evaluated"])
            self.assertEqual(metrics["status"], "provisional")
            self.assertEqual(
                config["data_provenance"]["input_sha256"]["labels"],
                _sha256(paths["labels"]),
            )
            self.assertTrue(resumed["resumed"])

    def test_rule_evaluation_persists_reproducibility_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _fixture(root)
            prepared = prepare_sit_stand_event_dataset(
                **paths,
                output_dir=root / "prepared",
                allow_provisional=True,
            )

            result = evaluate_sit_stand_rule(
                prepared["dataset_path"],
                root / "e0-run",
                metadata_path=prepared["metadata_path"],
                seed=42,
            )
            metrics = json.loads(Path(result["metrics_path"]).read_text())
            self.assertTrue(Path(result["log_path"]).is_file())

        self.assertEqual(metrics["run_id"], "e0-run")
        self.assertEqual(metrics["seed"], 42)
        self.assertFalse(metrics["test_evaluated"])
        self.assertIn("derived_split_sha256", metrics["data_provenance"])


class SitStandCandidateTCNTest(unittest.TestCase):
    def test_model_outputs_presence_and_masked_direction_heads(self) -> None:
        model = SitStandCandidateTCN(hidden_channels=8, dilations=(1, 2))
        features = torch.zeros(
            3,
            16,
            len(SIT_STAND_JOINTS),
            len(SIT_STAND_CHANNELS),
        )
        features[..., -1] = 1.0

        outputs = model(features)

        self.assertEqual(outputs["presence_logits"].shape, (3, 2))
        self.assertEqual(outputs["direction_logits"].shape, (3, 2))
        np.testing.assert_array_equal(
            build_sit_stand_direction_labels(
                np.asarray(["A03", "A04", "A05"])
            ),
            np.asarray([0, 1, -1]),
        )

    def test_direction_loss_ignores_hard_negative_rows(self) -> None:
        presence_logits = torch.tensor([[0.2, 0.8], [0.9, 0.1]])
        direction_logits = torch.tensor([[0.1, 0.9], [0.0, 0.0]])
        presence_labels = torch.tensor([1, 0])
        direction_labels = torch.tensor([1, -1])
        weights = torch.ones(2)

        first = compute_sit_stand_multitask_loss(
            presence_logits,
            direction_logits,
            presence_labels,
            direction_labels,
            weights,
            direction_loss_weight=1.0,
        )
        changed_negative = direction_logits.clone()
        changed_negative[1] = torch.tensor([-100.0, 100.0])
        second = compute_sit_stand_multitask_loss(
            presence_logits,
            changed_negative,
            presence_labels,
            direction_labels,
            weights,
            direction_loss_weight=1.0,
        )

        self.assertAlmostEqual(float(first["loss"]), float(second["loss"]), places=6)
        self.assertEqual(first["direction_count"], 1)

    def test_smoke_training_is_provisional_test_locked_and_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _fixture(root)
            prepared = prepare_sit_stand_event_dataset(
                **paths,
                output_dir=root / "prepared",
                allow_provisional=True,
            )
            config = SitStandCandidateTCNConfig(
                epochs=1,
                batch_size=4,
                hidden_channels=8,
                dilations=(1, 2),
                patience=2,
                seed=42,
                device="cpu",
                augment_mirror=False,
            )
            with self.assertRaisesRegex(ValueError, "allow_provisional"):
                train_sit_stand_candidate_tcn(
                    prepared["dataset_path"],
                    root / "blocked",
                    metadata_path=prepared["metadata_path"],
                    config=config,
                )

            first = train_sit_stand_candidate_tcn(
                prepared["dataset_path"],
                root / "run-1",
                metadata_path=prepared["metadata_path"],
                config=config,
                allow_provisional=True,
            )
            evaluation = evaluate_sit_stand_candidate_tcn(
                prepared["dataset_path"],
                first["checkpoint_path"],
                root / "run-1" / "independent-validation.json",
                metadata_path=prepared["metadata_path"],
                device="cpu",
                batch_size=4,
            )
            resumed = train_sit_stand_candidate_tcn(
                prepared["dataset_path"],
                root / "run-2",
                metadata_path=prepared["metadata_path"],
                config=SitStandCandidateTCNConfig(
                    **{**config.__dict__, "epochs": 2}
                ),
                allow_provisional=True,
                resume_checkpoint=first["last_checkpoint_path"],
            )
            metrics = json.loads(Path(first["metrics_path"]).read_text())

        self.assertTrue(resumed["resumed"])
        self.assertEqual(resumed["epochs_completed"], 2)
        self.assertEqual(metrics["task"], "sit_stand_candidate_clip_tcn_v1")
        self.assertFalse(metrics["test_evaluated"])
        self.assertIsNone(metrics["test"])
        self.assertIn("direction", metrics["validation"])
        self.assertFalse(evaluation["test_evaluated"])
        self.assertEqual(evaluation["partition"], "validation")


if __name__ == "__main__":
    unittest.main()
