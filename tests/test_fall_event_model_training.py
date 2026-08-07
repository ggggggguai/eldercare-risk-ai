from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
import torch

from elderly_monitoring.modules.fall_risk.fall_event_tcn import (
    FALL_SUBTYPE_LABELS,
    FallEventCandidateTCN,
    FallEventTCNConfig,
    FallEventTCNPredictor,
    build_fall_subtype_labels,
    compute_fall_multitask_loss,
    evaluate_fall_event_candidate_tcn,
    train_fall_event_candidate_tcn,
)
from elderly_monitoring.modules.fall_risk.fall_event_training import (
    FALL_EVENT_CHANNELS,
    FALL_EVENT_JOINTS,
    FallEventDatasetConfig,
    audit_fall_event_training_inputs,
    build_fall_event_tensor,
    prepare_fall_event_proxy_dataset,
)
from scripts.collect.run_fall_event_model import main as run_fall_event_model_main


BASE_JOINTS = FALL_EVENT_JOINTS[:12]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_model_checkpoint(path: Path) -> None:
    model_config = {
        "joint_count": len(FALL_EVENT_JOINTS),
        "input_channels": len(FALL_EVENT_CHANNELS),
        "hidden_channels": 8,
        "kernel_size": 3,
        "dilations": [1, 2],
        "dropout": 0.0,
    }
    model = FallEventCandidateTCN(**model_config)
    torch.save(
        {
            "schema_version": "fall-event-candidate-tcn-model-v1",
            "task": "fall_event_candidate_clip_tcn_v1",
            "target_task": "fall_event_v1",
            "status": "provisional",
            "model_version": "fall-event-test-model",
            "state_dict": model.state_dict(),
            "model_config": model_config,
            "joint_order": list(FALL_EVENT_JOINTS),
            "channel_order": list(FALL_EVENT_CHANNELS),
            "threshold": 0.5,
            "test_evaluated": False,
        },
        path,
    )


def _pose_record(
    frame_id: int,
    *,
    action_id: str,
    usable: bool = True,
) -> dict[str, object]:
    progress = frame_id / 31.0
    is_fall = action_id.startswith("D")
    hip_y = 0.42 + ((0.34 if is_fall else 0.12) * progress)
    if is_fall:
        shoulder_x = 0.50 + (0.18 * progress)
        shoulder_y = hip_y - (0.20 * (1.0 - progress))
    else:
        shoulder_x = 0.50
        shoulder_y = hip_y - 0.20
    ankle_y = 0.90
    knee_y = hip_y + ((ankle_y - hip_y) * 0.52)
    coordinates = {
        "left_shoulder": (shoulder_x - 0.05, shoulder_y),
        "right_shoulder": (shoulder_x + 0.05, shoulder_y),
        "left_elbow": (shoulder_x - 0.08, shoulder_y + 0.08),
        "right_elbow": (shoulder_x + 0.08, shoulder_y + 0.08),
        "left_wrist": (shoulder_x - 0.10, shoulder_y + 0.16),
        "right_wrist": (shoulder_x + 0.10, shoulder_y + 0.16),
        "left_hip": (0.46, hip_y),
        "right_hip": (0.54, hip_y),
        "left_knee": (0.46, knee_y),
        "right_knee": (0.54, knee_y),
        "left_ankle": (0.46, ankle_y),
        "right_ankle": (0.54, ankle_y),
    }
    quality = 0.95 if usable else 0.10
    bbox_height = 0.55 - ((0.20 if is_fall else 0.05) * progress)
    return {
        "frame_id": frame_id,
        "timestamp_sec": frame_id / 8.0,
        "person_id": "person",
        "track_id": 1,
        "pose_confidence": quality,
        "core_keypoint_quality": quality,
        "bbox": [0.30, hip_y - 0.30, 0.70, hip_y - 0.30 + bbox_height],
        "window_quality": {
            "usable_for_near_fall": usable,
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


def _label(
    index: int,
    *,
    action_id: str,
    partition: str,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    video_id = f"video_{index}"
    label_id = f"actionv3_{index:024x}"
    action_types = {
        "D01": "forward_fall",
        "D02": "lateral_fall",
        "D03": "backward_fall",
        "D05": "seated_fall",
        "A03": "controlled_sit_down",
        "A05": "controlled_squat",
        "A06": "controlled_bend",
        "A09": "kneel_or_floor_activity",
    }
    label = {
        "schema_version": "fall-risk-action-label-v3",
        "label_id": label_id,
        "asset_id": f"asset_{index}",
        "video_id": video_id,
        "subject_id": f"subject_{index}",
        "source_group_id": f"source_{index}",
        "sample_group_id": f"sample_{index}",
        "action_id": action_id,
        "action_type": action_types[action_id],
        "action_type_training_tier": "primary",
        "training_tier": "primary",
        "boundary_precision": "exact",
        "review_status": "single_annotated",
        "start_frame": 0,
        "end_frame_exclusive": 32,
        "start_time": 0.0,
        "end_time_exclusive": 4.0,
        "quality_flags": [],
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
        "fps": 8.0,
        "frame_count": 32,
    }
    assignment = {
        "schema_version": "fall-risk-training-split-v3",
        "label_id": label_id,
        "label_kind": "action",
        "task_type": "action",
        "asset_id": f"asset_{index}",
        "video_id": video_id,
        "subject_id": f"subject_{index}",
        "source_group_id": f"source_{index}",
        "sample_group_id": f"sample_{index}",
        "split_group_id": f"split_{index}",
        "training_tier": "primary",
        "partition": partition,
    }
    return label, manifest, assignment


def _fixture(root: Path, *, low_quality_index: int | None = None) -> dict[str, Path]:
    rows: list[tuple[int, str, str]] = []
    actions = ("D01", "D02", "D03", "D05", "A03", "A05", "A06", "A09")
    index = 1
    for partition in ("train", "validation"):
        for action_id in actions:
            rows.append((index, action_id, partition))
            index += 1
    rows.append((index, "D01", "test"))

    labels: list[dict[str, object]] = []
    manifests: list[dict[str, object]] = []
    assignments: list[dict[str, object]] = []
    pose_dir = root / "poses"
    for row_index, action_id, partition in rows:
        label, manifest, assignment = _label(
            row_index, action_id=action_id, partition=partition
        )
        labels.append(label)
        manifests.append(manifest)
        assignments.append(assignment)
        if partition != "test":
            _write_jsonl(
                pose_dir / f"video_{row_index}.jsonl",
                [
                    _pose_record(
                        frame_id,
                        action_id=action_id,
                        usable=row_index != low_quality_index,
                    )
                    for frame_id in range(32)
                ],
            )
    manifests.append(
        {
            "asset_id": "tabular_asset",
            "video_id": None,
            "path": "unused/table.csv",
            "dataset": "fixture",
            "eligibility": True,
            "media_type": "tabular",
        }
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
                "training_ready": {"action_type": False, "fall_event": False},
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


class FallEventPreparationTest(unittest.TestCase):
    def test_tensor_contract_preserves_vertical_motion_and_masks(self) -> None:
        records = [_pose_record(index, action_id="D01") for index in range(32)]

        tensor = build_fall_event_tensor(records, window_frames=32, target_fps=8.0)

        self.assertEqual(
            tensor.shape,
            (32, len(FALL_EVENT_JOINTS), len(FALL_EVENT_CHANNELS)),
        )
        self.assertTrue(np.isfinite(tensor).all())
        self.assertTrue(np.all(tensor[..., -1] == 1.0))
        self.assertGreater(float(np.ptp(tensor[:, 12, 5])), 0.25)

    def test_audit_marks_proxy_ready_but_formal_event_gate_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))

            audit = audit_fall_event_training_inputs(**paths)

        self.assertTrue(audit["training_gates"]["clip_proxy_ready"])
        self.assertFalse(audit["training_gates"]["formal_event_ready"])
        self.assertFalse(audit["data_access"]["test_pose_read"])
        self.assertEqual(
            set(audit["proxy_classes"]["positive_action_ids"]),
            {"D01", "D02", "D03", "D05"},
        )

    def test_builder_requires_provisional_gate_and_never_reads_test_pose(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _fixture(root)
            with self.assertRaisesRegex(ValueError, "allow_provisional"):
                prepare_fall_event_proxy_dataset(
                    **paths,
                    output_dir=root / "blocked",
                    config=FallEventDatasetConfig(),
                )

            prepared = prepare_fall_event_proxy_dataset(
                **paths,
                output_dir=root / "prepared",
                config=FallEventDatasetConfig(),
                allow_provisional=True,
            )
            metadata = json.loads(Path(prepared["metadata_path"]).read_text())
            with np.load(prepared["dataset_path"], allow_pickle=False) as archive:
                partitions = set(archive["partitions"].astype(str).tolist())
                labels = archive["labels"].astype(int)

        self.assertEqual(partitions, {"train", "validation"})
        self.assertEqual(set(labels.tolist()), {0, 1})
        self.assertEqual(metadata["locked_test_label_count"], 1)
        self.assertFalse(metadata["test_pose_read"])

    def test_low_quality_segment_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _fixture(root, low_quality_index=5)

            prepared = prepare_fall_event_proxy_dataset(
                **paths,
                output_dir=root / "prepared",
                allow_provisional=True,
            )
            metadata = json.loads(Path(prepared["metadata_path"]).read_text())

        self.assertGreater(metadata["rejected_segment_counts"]["insufficient_quality"], 0)


class FallEventCandidateTCNTest(unittest.TestCase):
    def test_model_has_presence_and_four_subtype_outputs(self) -> None:
        model = FallEventCandidateTCN(hidden_channels=8, dilations=(1, 2))
        features = torch.zeros(
            3,
            32,
            len(FALL_EVENT_JOINTS),
            len(FALL_EVENT_CHANNELS),
        )
        features[..., -1] = 1.0

        outputs = model(features)

        self.assertEqual(outputs["presence_logits"].shape, (3, 2))
        self.assertEqual(outputs["subtype_logits"].shape, (3, len(FALL_SUBTYPE_LABELS)))
        np.testing.assert_array_equal(
            build_fall_subtype_labels(np.asarray(["D01", "D02", "D03", "D05", "A03"])),
            np.asarray([0, 1, 2, 3, -1]),
        )

    def test_subtype_loss_ignores_negative_rows(self) -> None:
        presence_logits = torch.tensor([[0.2, 0.8], [0.9, 0.1]])
        subtype_logits = torch.tensor(
            [[0.1, 0.2, 0.3, 0.4], [0.0, 0.0, 0.0, 0.0]]
        )
        presence_labels = torch.tensor([1, 0])
        subtype_labels = torch.tensor([3, -1])
        weights = torch.ones(2)

        first = compute_fall_multitask_loss(
            presence_logits,
            subtype_logits,
            presence_labels,
            subtype_labels,
            weights,
            subtype_loss_weight=0.5,
        )
        changed = subtype_logits.clone()
        changed[1] = torch.tensor([-100.0, 100.0, -100.0, 100.0])
        second = compute_fall_multitask_loss(
            presence_logits,
            changed,
            presence_labels,
            subtype_labels,
            weights,
            subtype_loss_weight=0.5,
        )

        self.assertAlmostEqual(float(first["loss"]), float(second["loss"]), places=6)
        self.assertEqual(first["subtype_count"], 1)

    def test_smoke_training_is_test_locked_resumable_and_predictable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _fixture(root)
            prepared = prepare_fall_event_proxy_dataset(
                **paths,
                output_dir=root / "prepared",
                allow_provisional=True,
            )
            config = FallEventTCNConfig(
                epochs=1,
                batch_size=8,
                hidden_channels=8,
                dilations=(1, 2),
                patience=2,
                seed=42,
                device="cpu",
                augment_mirror=False,
                subtype_loss_weight=0.0,
            )
            with self.assertRaisesRegex(ValueError, "allow_provisional"):
                train_fall_event_candidate_tcn(
                    prepared["dataset_path"],
                    root / "blocked",
                    metadata_path=prepared["metadata_path"],
                    config=config,
                )

            first = train_fall_event_candidate_tcn(
                prepared["dataset_path"],
                root / "run-1",
                metadata_path=prepared["metadata_path"],
                config=config,
                allow_provisional=True,
            )
            evaluation = evaluate_fall_event_candidate_tcn(
                prepared["dataset_path"],
                first["checkpoint_path"],
                root / "run-1" / "independent-validation.json",
                metadata_path=prepared["metadata_path"],
                device="cpu",
                batch_size=8,
            )
            resumed = train_fall_event_candidate_tcn(
                prepared["dataset_path"],
                root / "run-2",
                metadata_path=prepared["metadata_path"],
                config=FallEventTCNConfig(**{**config.__dict__, "epochs": 2}),
                allow_provisional=True,
                resume_checkpoint=first["last_checkpoint_path"],
            )
            predictor = FallEventTCNPredictor(first["checkpoint_path"], device="cpu")
            with np.load(prepared["dataset_path"], allow_pickle=False) as archive:
                prediction = predictor.predict_tensor(archive["features"][0])
            metrics = json.loads(Path(first["metrics_path"]).read_text())

        self.assertTrue(resumed["resumed"])
        self.assertEqual(resumed["epochs_completed"], 2)
        self.assertFalse(metrics["test_evaluated"])
        self.assertIsNone(metrics["test"])
        self.assertFalse(evaluation["test_evaluated"])
        self.assertEqual(metrics["validation"]["subtype"]["status"], "not_trained")
        self.assertEqual(evaluation["subtype"]["status"], "not_trained")
        confusion = metrics["validation"]["presence"]["confusion_matrix"]
        self.assertEqual(metrics["failure_case_count"], confusion[0][1] + confusion[1][0])
        self.assertGreaterEqual(prediction["fall_event_score"], 0.0)
        self.assertLessEqual(prediction["fall_event_score"], 1.0)


class FallEventModelCLITest(unittest.TestCase):
    def _run(self, root: Path, records: list[dict[str, object]]) -> tuple[int, Path]:
        input_path = root / "poses.jsonl"
        output_path = root / "predictions.jsonl"
        checkpoint_path = root / "model.pt"
        _write_jsonl(input_path, records)
        _write_model_checkpoint(checkpoint_path)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            result = run_fall_event_model_main(
                [
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--checkpoint",
                    str(checkpoint_path),
                    "--device",
                    "cpu",
                ]
            )
        return result, output_path

    def test_empty_input_fails_closed_without_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, output_path = self._run(Path(tmp), [])

            self.assertEqual(result, 2)
            self.assertFalse(output_path.exists())

    def test_low_quality_window_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, output_path = self._run(
                Path(tmp),
                [_pose_record(index, action_id="D01", usable=False) for index in range(32)],
            )
            rows = [json.loads(line) for line in output_path.read_text().splitlines()]

        self.assertEqual(result, 0)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "unavailable")
        self.assertIsNone(rows[0]["fall_event_score"])
        self.assertFalse(rows[0]["fall_event_detected"])

    def test_normal_window_returns_provisional_shadow_score(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, output_path = self._run(
                Path(tmp),
                [_pose_record(index, action_id="D01") for index in range(32)],
            )
            rows = [json.loads(line) for line in output_path.read_text().splitlines()]

        self.assertEqual(result, 0)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "provisional_shadow")
        self.assertGreaterEqual(rows[0]["fall_event_score"], 0.0)
        self.assertLessEqual(rows[0]["fall_event_score"], 1.0)


if __name__ == "__main__":
    unittest.main()
