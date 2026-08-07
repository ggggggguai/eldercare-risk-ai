from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from elderly_monitoring.modules.fall_risk.near_fall_tcn import (
    NearFallTCNConfig,
    NearFallTCNPredictor,
    train_near_fall_tcn,
)
from elderly_monitoring.modules.fall_risk.near_fall_training import (
    NEAR_FALL_CHANNELS,
    NEAR_FALL_HARD_NEGATIVES,
    NEAR_FALL_JOINTS,
    NearFallDatasetConfig,
    build_near_fall_tensor,
    create_synthetic_near_fall_dataset,
    fit_normalization_statistics,
    prepare_near_fall_event_dataset,
    select_near_fall_training_labels,
    _manifest_index,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _pose_record(
    frame_id: int,
    *,
    positive: bool,
    usable: bool = True,
) -> dict[str, object]:
    progress = frame_id / 47.0
    recovery_bump = np.sin(np.pi * min(1.0, frame_id / 31.0)) if positive else 0.0
    hip_y = 0.58
    shoulder_shift = 0.16 * recovery_bump
    coordinates = {
        "left_shoulder": (0.44 + shoulder_shift, 0.36 + 0.03 * recovery_bump),
        "right_shoulder": (0.56 + shoulder_shift, 0.36 + 0.03 * recovery_bump),
        "left_wrist": (0.37 + 1.5 * shoulder_shift, 0.52 - 0.06 * recovery_bump),
        "right_wrist": (0.63 + 1.5 * shoulder_shift, 0.52 - 0.06 * recovery_bump),
        "left_hip": (0.46, hip_y),
        "right_hip": (0.54, hip_y),
        "left_knee": (0.46 + 0.01 * progress, 0.74),
        "right_knee": (0.54 + 0.01 * progress, 0.74),
        "left_ankle": (0.45, 0.91),
        "right_ankle": (0.55, 0.91),
    }
    quality = 0.95 if usable else 0.05
    return {
        "frame_id": frame_id,
        "timestamp_sec": frame_id / 8.0,
        "person_id": "person",
        "track_id": 1,
        "pose_confidence": quality,
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


def _event_label(
    index: int,
    *,
    partition: str,
    role: str,
    hard_negative_type: str | None = None,
    low_quality_pose: bool = False,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    video_id = f"video_{index}"
    label_id = f"eventv3_{index:024x}"
    positive = role == "positive"
    physical_event_id = f"physical_{index:024x}" if positive else None
    label = {
        "schema_version": "fall-risk-event-label-v3",
        "label_id": label_id,
        "asset_id": f"asset_{index}",
        "video_id": video_id,
        "content_sha256": f"{index:064x}"[-64:],
        "subject_id": f"subject_{index}",
        "source_group_id": f"source_{index}",
        "sample_group_id": f"samplegrp_{index:024x}",
        "physical_event_id": physical_event_id,
        "track_id": "1",
        "start_frame": 0,
        "end_frame_exclusive": 48,
        "start_time": 0.0,
        "end_time_exclusive": 6.0,
        "target_status": "confirmed",
        "boundary_precision": "exact",
        "quality_flags": [],
        "training_tier": "primary",
        "review_status": "double_reviewed",
        "reviewer_ids": ["reviewer_a", "reviewer_b"],
        "source_refs": [
            {
                "source_type": "manual_v3",
                "source_record_id": f"manual-{index}",
                "source_annotation_path": "synthetic/manual.jsonl",
                "source_annotation_sha256": f"{index:064x}"[-64:],
                "source_label_id": f"manual-{index}",
            }
        ],
        "task_type": "near_fall_event",
        "label_role": role,
        "event_type": "near_fall" if positive else None,
        "event_subtype": "stumble_recovery" if positive else None,
        "event_outcome": "recovered_without_fall" if positive else None,
        "hard_negative_type": hard_negative_type,
        "onset_frame": 8 if positive else None,
        "peak_frame": 20 if positive else None,
        "impact_frame": None,
        "recovery_frame": 31 if positive else None,
        "subtype_training_tier": "primary" if positive else "ignore",
        "_low_quality_pose": low_quality_pose,
    }
    manifest = {
        "asset_id": f"asset_{index}",
        "video_id": video_id,
        "path": f"unused/{video_id}.mp4",
        "sha256": label["content_sha256"],
        "dataset": "synthetic_fixture",
        "scene_region": "room",
        "eligibility": True,
        "media_type": "video",
        "fps": 8.0,
        "frame_count": 48,
    }
    assignment = {
        "schema_version": "fall-risk-training-split-v3",
        "label_id": label_id,
        "label_kind": "event",
        "task_type": "near_fall_event",
        "asset_id": label["asset_id"],
        "video_id": video_id,
        "content_sha256": label["content_sha256"],
        "subject_id": label["subject_id"],
        "source_group_id": label["source_group_id"],
        "sample_group_id": label["sample_group_id"],
        "physical_event_id": physical_event_id,
        "split_group_id": f"splitgrp_{index:024x}",
        "training_tier": "primary",
        "partition": partition,
    }
    return label, manifest, assignment


def _refresh_contract_hashes(paths: dict[str, Path]) -> None:
    assignments_sha = _sha256(paths["assignments"])
    split = json.loads(paths["split"].read_text(encoding="utf-8"))
    split["assignments_sha256"] = assignments_sha
    split["input_sha256"] = {
        "event_labels": _sha256(paths["labels"]),
        "manifest": _sha256(paths["manifest"]),
    }
    paths["split"].write_text(json.dumps(split), encoding="utf-8")
    validation = json.loads(paths["validation"].read_text(encoding="utf-8"))
    validation["input_sha256"] = {
        "event_labels": _sha256(paths["labels"]),
        "manifest": _sha256(paths["manifest"]),
        "split_assignments": assignments_sha,
        "split_report": _sha256(paths["split"]),
    }
    paths["validation"].write_text(json.dumps(validation), encoding="utf-8")


def _fixture(root: Path) -> dict[str, Path]:
    specifications: list[tuple[str, str, str | None, bool]] = []
    specifications.append(("train", "positive", None, False))
    specifications.extend(
        ("train", "negative", hard_negative, False)
        for hard_negative in NEAR_FALL_HARD_NEGATIVES
    )
    specifications.append(("train", "negative", "background", True))
    specifications.extend(
        [
            ("validation", "positive", None, False),
            ("validation", "negative", "background", False),
            ("test", "positive", None, False),
            ("test", "negative", "background", False),
        ]
    )
    labels: list[dict[str, object]] = []
    manifests: list[dict[str, object]] = []
    assignments: list[dict[str, object]] = []
    pose_dir = root / "poses"
    partition_counts = {
        partition: {"primary_positive": 0, "primary_negative": 0}
        for partition in ("train", "validation", "test")
    }
    for index, (partition, role, hard_negative, low_quality) in enumerate(
        specifications, start=1
    ):
        label, manifest, assignment = _event_label(
            index,
            partition=partition,
            role=role,
            hard_negative_type=hard_negative,
            low_quality_pose=low_quality,
        )
        labels.append(label)
        manifests.append(manifest)
        assignments.append(assignment)
        partition_counts[partition][f"primary_{role}"] += 1
        if partition != "test":
            _write_jsonl(
                pose_dir / f"video_{index}.jsonl",
                [
                    _pose_record(
                        frame_id,
                        positive=role == "positive",
                        usable=not low_quality,
                    )
                    for frame_id in range(48)
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
                "assignments_sha256": "pending",
                "input_sha256": {},
                "leakage_issues": [],
                "partition_supervision_counts": {
                    "near_fall_event": partition_counts
                },
            }
        ),
        encoding="utf-8",
    )
    validation_path.write_text(
        json.dumps(
            {
                "valid": True,
                "training_ready": {"near_fall_event": True},
                "counts": {
                    "primary_near_fall_positive": 3,
                    "primary_near_fall_negative": 11,
                },
                "hard_negative_coverage": {
                    "near_fall_event": {
                        "present": list(NEAR_FALL_HARD_NEGATIVES),
                        "missing": [],
                    }
                },
                "input_sha256": {},
            }
        ),
        encoding="utf-8",
    )
    paths = {
        "labels": labels_path,
        "manifest": manifest_path,
        "assignments": assignments_path,
        "split": split_path,
        "validation": validation_path,
        "pose_dir": pose_dir,
    }
    _refresh_contract_hashes(paths)
    return paths


class NearFallTensorContractTest(unittest.TestCase):
    def test_joint_channel_order_causal_derivatives_and_missing_mask(self) -> None:
        records = [_pose_record(index, positive=True) for index in range(8)]
        records[3]["keypoints"][2]["valid"] = False
        records[3]["keypoints"][2]["x_smooth"] = None
        records[3]["keypoints"][2]["y_smooth"] = None

        tensor = build_near_fall_tensor(records, window_frames=8, target_fps=8.0)
        changed = copy.deepcopy(records)
        changed[-1]["keypoints"][0]["x_smooth"] = 99.0
        changed_tensor = build_near_fall_tensor(
            changed, window_frames=8, target_fps=8.0
        )

        self.assertEqual(
            NEAR_FALL_JOINTS,
            (
                "left_shoulder",
                "right_shoulder",
                "left_wrist",
                "right_wrist",
                "left_hip",
                "right_hip",
                "left_knee",
                "right_knee",
                "left_ankle",
                "right_ankle",
            ),
        )
        self.assertEqual(
            NEAR_FALL_CHANNELS,
            ("x", "y", "dx", "dy", "ddx", "ddy", "quality", "valid_mask"),
        )
        self.assertEqual(tensor.shape, (8, 10, 8))
        self.assertEqual(float(tensor[3, 2, 7]), 0.0)
        self.assertTrue(np.allclose(tensor[3, 2, :7], 0.0))
        self.assertTrue(np.allclose(tensor[4, 2, 2:6], 0.0))
        self.assertTrue(np.allclose(tensor[:7], changed_tensor[:7]))
        self.assertTrue(np.isfinite(tensor).all())

    def test_normalization_statistics_use_train_only(self) -> None:
        features = np.zeros((4, 3, 10, 8), dtype=np.float32)
        features[..., 7] = 1.0
        features[:2, ..., 0] = np.asarray([1.0, 3.0])[:, None, None]
        features[2:, ..., 0] = 1000.0
        partitions = np.asarray(["train", "train", "validation", "validation"])

        first = fit_normalization_statistics(features, partitions)
        features[2:, ..., 0] = -5000.0
        second = fit_normalization_statistics(features, partitions)

        np.testing.assert_allclose(first["mean"], second["mean"])
        np.testing.assert_allclose(first["std"], second["std"])
        self.assertAlmostEqual(float(first["mean"][0]), 2.0)


class NearFallLabelAndGateTest(unittest.TestCase):
    def test_strict_selection_excludes_ignore_uncertain_and_low_quality(self) -> None:
        positive, _, _ = _event_label(1, partition="train", role="positive")
        progressed, _, _ = _event_label(
            2,
            partition="train",
            role="negative",
            hard_negative_type="progressed_to_fall",
        )
        ignored = {**positive, "label_id": "ignored", "label_role": "ignore"}
        uncertain = {**positive, "label_id": "uncertain", "target_status": "uncertain"}
        low_quality = {
            **positive,
            "label_id": "low-quality",
            "quality_flags": ["heavy_occlusion"],
        }
        auxiliary = {**positive, "label_id": "aux", "training_tier": "auxiliary"}
        wrong_task = {**positive, "label_id": "fall", "task_type": "fall_event"}

        selected = select_near_fall_training_labels(
            [positive, progressed, ignored, uncertain, low_quality, auxiliary, wrong_task]
        )

        self.assertEqual([row["label"] for row in selected], [1, 0])
        self.assertEqual(selected[1]["hard_negative_type"], "progressed_to_fall")

    def test_selection_accepts_single_person_assumed_primary_negatives(self) -> None:
        _, _, _ = _event_label(1, partition="train", role="positive")
        negative, _, _ = _event_label(
            2,
            partition="train",
            role="negative",
            hard_negative_type="progressed_to_fall",
        )
        negative["target_status"] = "single_person_assumed"

        selected = select_near_fall_training_labels([negative])

        self.assertEqual([row["label"] for row in selected], [0])

    def test_real_gate_fails_before_output_with_counts_and_missing_types(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _fixture(root)
            report = json.loads(paths["validation"].read_text(encoding="utf-8"))
            report["training_ready"]["near_fall_event"] = False
            report["counts"]["primary_near_fall_positive"] = 0
            report["counts"]["primary_near_fall_negative"] = 0
            report["hard_negative_coverage"]["near_fall_event"] = {
                "present": [],
                "missing": list(NEAR_FALL_HARD_NEGATIVES),
            }
            paths["validation"].write_text(json.dumps(report), encoding="utf-8")
            output = root / "blocked-output"

            with self.assertRaisesRegex(
                ValueError,
                "positive=0.*negative=0.*normal_turn.*progressed_to_fall",
            ):
                prepare_near_fall_event_dataset(
                    **paths,
                    output_dir=output,
                    config=NearFallDatasetConfig(),
                )

            self.assertFalse(output.exists())

    def test_hash_mismatch_and_protection_group_leakage_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _fixture(root)
            paths["labels"].write_text(
                paths["labels"].read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "event_labels SHA-256 mismatch"):
                prepare_near_fall_event_dataset(
                    **paths, output_dir=root / "bad-hash"
                )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _fixture(root)
            labels = [
                json.loads(line)
                for line in paths["labels"].read_text(encoding="utf-8").splitlines()
            ]
            assignments = [
                json.loads(line)
                for line in paths["assignments"].read_text(encoding="utf-8").splitlines()
            ]
            labels[-1]["subject_id"] = labels[0]["subject_id"]
            assignments[-1]["subject_id"] = assignments[0]["subject_id"]
            _write_jsonl(paths["labels"], labels)
            _write_jsonl(paths["assignments"], assignments)
            _refresh_contract_hashes(paths)
            with self.assertRaisesRegex(ValueError, "subject_id"):
                prepare_near_fall_event_dataset(
                    **paths, output_dir=root / "leaked"
                )


class NearFallDatasetPreparationTest(unittest.TestCase):
    def test_manifest_index_ignores_null_video_ids_but_rejects_real_duplicates(self) -> None:
        rows = [
            {"asset_id": "table", "video_id": None, "media_type": "tabular"},
            {"asset_id": "video", "video_id": "video_1", "media_type": "video"},
        ]

        index = _manifest_index(rows)

        self.assertEqual(set(index), {"video_1"})
        with self.assertRaisesRegex(
            ValueError, "duplicate near-fall manifest video_id: video_1"
        ):
            _manifest_index([rows[1], dict(rows[1])])

    def test_recovery_aligned_windows_test_lock_weights_and_quality_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _fixture(root)
            result = prepare_near_fall_event_dataset(
                **paths,
                output_dir=root / "prepared",
                config=NearFallDatasetConfig(
                    target_fps=4.0,
                    window_sec=4.0,
                    stride_sec=1.0,
                    min_observed_frames=8,
                ),
            )
            metadata = json.loads(Path(result["metadata_path"]).read_text())
            samples = [
                json.loads(line)
                for line in Path(result["samples_path"]).read_text().splitlines()
            ]
            with np.load(result["dataset_path"], allow_pickle=False) as archive:
                partitions = set(archive["partitions"].astype(str).tolist())
                event_ids = archive["event_ids"].astype(str)
                weights = archive["sample_weights"].astype(float)
                features = archive["features"]

        positives = [sample for sample in samples if sample["label"] == 1]
        self.assertEqual(partitions, {"train", "validation"})
        self.assertTrue(all(sample["anchor_reason"] == "recovery_frame" for sample in positives))
        self.assertTrue(all(sample["window_end_frame"] == 31 for sample in positives))
        self.assertTrue(all(sample["max_source_frame"] <= 31 for sample in positives))
        self.assertEqual(metadata["locked_test_label_count"], 2)
        self.assertFalse(metadata["test_pose_read"])
        self.assertGreater(metadata["rejected_window_counts"]["insufficient_quality"], 0)
        self.assertEqual(features.shape[1:], (16, 10, 8))
        for event_id in set(event_ids.tolist()):
            self.assertAlmostEqual(float(weights[event_ids == event_id].sum()), 1.0)
        self.assertTrue(any(np.sum(event_ids == event_id) > 1 for event_id in event_ids))


class NearFallTCNTrainingTest(unittest.TestCase):
    def test_fixed_seed_checkpoint_contract_rejection_and_synthetic_overfit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared = create_synthetic_near_fall_dataset(
                root / "synthetic", seed=7, sample_count_per_class=10
            )
            config = NearFallTCNConfig(
                epochs=45,
                batch_size=8,
                hidden_channels=12,
                dilations=(1, 2),
                dropout=0.0,
                learning_rate=0.01,
                patience=45,
                seed=19,
                device="cpu",
            )
            first = train_near_fall_tcn(
                prepared["dataset_path"],
                root / "run-1",
                metadata_path=prepared["metadata_path"],
                config=config,
                allow_synthetic=True,
            )
            second = train_near_fall_tcn(
                prepared["dataset_path"],
                root / "run-2",
                metadata_path=prepared["metadata_path"],
                config=config,
                allow_synthetic=True,
            )
            checkpoint = torch.load(first["checkpoint_path"], map_location="cpu")
            metrics = json.loads(Path(first["metrics_path"]).read_text())
            predictor = NearFallTCNPredictor(first["checkpoint_path"], device="cpu")
            with np.load(prepared["dataset_path"], allow_pickle=False) as archive:
                normalized = archive["features"]
                first_prediction = predictor.predict_tensor(
                    normalized[0], already_normalized=True
                )
            second_metrics = json.loads(Path(second["metrics_path"]).read_text())

        self.assertEqual(checkpoint["schema_version"], "near-fall-tcn-checkpoint-v1")
        self.assertEqual(checkpoint["task"], "near_fall_recovery_confirmation_v1")
        self.assertEqual(checkpoint["input_contract"]["joint_order"], list(NEAR_FALL_JOINTS))
        self.assertEqual(checkpoint["input_contract"]["channel_order"], list(NEAR_FALL_CHANNELS))
        self.assertNotIn("outcome_head", checkpoint)
        self.assertTrue(metrics["synthetic"])
        self.assertFalse(metrics["test_evaluated"])
        self.assertIsNone(metrics["test"])
        self.assertGreaterEqual(metrics["train"]["accuracy"], 0.95)
        self.assertEqual(metrics["history"], second_metrics["history"])
        self.assertEqual(first_prediction["status"], "valid")

        low_quality = np.zeros((16, 10, 8), dtype=np.float32)
        low_quality[:2, :, 6:] = 1.0
        rejected = predictor.predict_tensor(low_quality, already_normalized=False)
        self.assertEqual(rejected["status"], "unavailable")
        self.assertIn("quality", rejected["reason"])


if __name__ == "__main__":
    unittest.main()
