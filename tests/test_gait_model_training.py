from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from elderly_monitoring.modules.fall_risk.gait_training import (
    GaitWindowPreparationConfig,
    _select_labeled_track,
    build_context_gait_window,
    build_gait_tensor,
    enrich_gait_training_labels,
    prepare_gait_window_dataset,
    resample_pose_records,
    select_gait_training_labels,
)
from elderly_monitoring.modules.fall_risk.gait_tcn import (
    LightweightGaitTCN,
    _GaitWindowDataset,
    _augment_gait_window,
    _evaluate,
    compute_balanced_class_weights,
    select_balanced_accuracy_threshold,
)
from elderly_monitoring.modules.fall_risk.gait_tabular import (
    GaitTabularTrainingConfig,
    train_gait_tabular_baselines,
)
from scripts.prepare.prepare_gait_pose_quality import build_pose_jobs


JOINTS = (
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


def make_pose_record(frame_id: int, *, unstable: bool = False) -> dict[str, object]:
    center_x = 0.30 + (0.01 * frame_id)
    sway = (0.025 if unstable and frame_id % 2 else 0.0)
    coordinates = {
        "left_shoulder": (center_x - 0.04 + sway, 0.30),
        "right_shoulder": (center_x + 0.04 + sway, 0.30),
        "left_elbow": (center_x - 0.07 + sway, 0.42),
        "right_elbow": (center_x + 0.07 + sway, 0.42),
        "left_wrist": (center_x - 0.08 + sway, 0.54),
        "right_wrist": (center_x + 0.08 + sway, 0.54),
        "left_hip": (center_x - 0.035 + sway, 0.55),
        "right_hip": (center_x + 0.035 + sway, 0.55),
        "left_knee": (center_x - 0.04 + sway, 0.72),
        "right_knee": (center_x + 0.04 + sway, 0.72),
        "left_ankle": (center_x - 0.05 + sway, 0.90),
        "right_ankle": (center_x + 0.05 + sway, 0.90),
    }
    return {
        "frame_id": frame_id,
        "timestamp_sec": frame_id / 25.0,
        "person_id": "elder_001",
        "track_id": 1,
        "bbox": [100.0, 20.0, 220.0, 230.0],
        "core_keypoint_quality": 0.95,
        "window_quality": {"usable_for_gait": True},
        "keypoints": [
            {
                "name": name,
                "x": x,
                "y": y,
                "x_smooth": x,
                "y_smooth": y,
                "score": 0.95,
                "quality_weight": 0.95,
                "valid": True,
                "source": "observed",
                "is_jump_outlier": False,
            }
            for name, (x, y) in coordinates.items()
        ],
    }


def make_label(video_index: int, *, positive: bool) -> dict[str, object]:
    video_id = f"video_{video_index}"
    return {
        "label_id": f"label_{video_index}",
        "video_id": video_id,
        "file_path": f"unused/{video_id}.avi",
        "subject_id": "unknown",
        "scene": "test_room",
        "action_id": "B03" if positive else "A01",
        "action_name": "shuffling_walk" if positive else "normal_walk",
        "event_type": "gait_instability" if positive else "normal_activity",
        "start_frame": 0,
        "end_frame": 9,
        "start_time": 0.0,
        "end_time": 0.36,
        "quality": "clear",
        "bbox_start": [100.0, 20.0, 220.0, 230.0],
        "bbox_end": [100.0, 20.0, 220.0, 230.0],
    }


def make_v3_label(
    video_index: int,
    *,
    positive: bool,
    training_tier: str = "primary",
) -> dict[str, object]:
    video_id = f"video_{video_index}"
    return {
        "schema_version": "fall-risk-action-label-v3",
        "label_id": f"label_v3_{video_index}",
        "asset_id": f"asset_{video_index}",
        "video_id": video_id,
        "subject_id": f"subject_{video_index}",
        "source_group_id": f"source_{video_index}",
        "sample_group_id": f"sample_{video_index}",
        "action_id": "B03" if positive else "A01",
        "action_type": "shuffling_walk" if positive else "normal_walk",
        "start_frame": 0,
        "end_frame_exclusive": 10,
        "start_time": 0.0,
        "end_time_exclusive": 0.4,
        "training_tier": training_tier,
        "action_type_training_tier": training_tier,
        "quality_flags": [],
    }


def make_manifest(video_index: int) -> dict[str, object]:
    video_id = f"video_{video_index}"
    return {
        "asset_id": f"asset_{video_index}",
        "video_id": video_id,
        "path": f"unused/{video_id}.avi",
        "scene_region": "test_room",
        "dataset": "test_dataset",
        "fps": 25.0,
        "frame_count": 10,
        "eligibility": True,
        "source_group_id": f"source_{video_index}",
        "content_sha256": f"sha256_{video_index}",
    }


def make_assignment(
    label: dict[str, object], partition: str
) -> dict[str, object]:
    return {
        "schema_version": "fall-risk-training-split-v3",
        "label_id": label["label_id"],
        "label_kind": "action",
        "task_type": "action",
        "asset_id": label["asset_id"],
        "video_id": label["video_id"],
        "subject_id": label["subject_id"],
        "source_group_id": label["source_group_id"],
        "sample_group_id": label["sample_group_id"],
        "split_group_id": f"split_{label['source_group_id']}",
        "training_tier": label["training_tier"],
        "partition": partition,
    }


class GaitTensorTest(unittest.TestCase):
    def test_tcn_supervision_pooling_rejects_mask_without_observed_frames(self) -> None:
        model = LightweightGaitTCN(
            hidden_channels=8,
            dilations=(1,),
            dropout=0.0,
            use_quality_as_feature=False,
        )
        features = torch.zeros(1, 8, 14, 5)
        features[..., -1] = 1.0

        with self.assertRaisesRegex(ValueError, "no observed labeled frame"):
            model.forward_heads(features, torch.zeros(1, 8))

    def test_context_window_keeps_only_same_track_and_marks_label_span(self) -> None:
        label = make_v3_label(1, positive=True)
        label.update(
            {
                "start_frame": 25,
                "end_frame": 56,
                "end_frame_exclusive": 57,
                "start_time": 1.0,
                "end_time_exclusive": 2.25,
                "bbox_start": [100.0, 20.0, 220.0, 230.0],
                "bbox_end": [100.0, 20.0, 220.0, 230.0],
            }
        )
        target_track = [make_pose_record(frame_id) for frame_id in range(100)]
        other_track = [
            {
                **make_pose_record(frame_id),
                "track_id": 2,
                "bbox": [300.0, 20.0, 420.0, 230.0],
            }
            for frame_id in range(100)
        ]

        window = build_context_gait_window(
            label,
            other_track + target_track,
            window_frames=16,
            target_fps=4.0,
            max_gap_sec=0.13,
        )

        observed = [record for record in window["records"] if record is not None]
        self.assertEqual({record["track_id"] for record in observed}, {1})
        self.assertEqual(sum(window["valid_mask"]), 16)
        self.assertEqual(sum(window["label_span_mask"]), 5)
        self.assertTrue(
            all(
                not is_labeled or is_valid
                for is_labeled, is_valid in zip(
                    window["label_span_mask"], window["valid_mask"], strict=True
                )
            )
        )

    def test_hierarchical_validation_loss_is_batch_size_invariant(self) -> None:
        generator = np.random.default_rng(42)
        features = generator.normal(size=(6, 8, 14, 5)).astype(np.float32)
        features[..., 4] = 1.0
        labels = np.asarray([0, 0, 0, 1, 0, 1], dtype=np.int64)
        walking_targets = np.asarray([0, 0, 1, 1, 0, 1], dtype=np.int64)
        sample_weights = np.ones(6, dtype=np.float32)
        indices = np.arange(6, dtype=np.int64)
        dataset = _GaitWindowDataset(
            features,
            labels,
            sample_weights,
            walking_targets,
            indices,
            augment_mirror=False,
        )
        model = LightweightGaitTCN(
            hidden_channels=8,
            dilations=(1,),
            dropout=0.0,
            use_quality_as_feature=False,
            hierarchical_walking_gate=True,
        )
        criterion = torch.nn.CrossEntropyLoss(reduction="none")
        walking_criterion = torch.nn.CrossEntropyLoss(reduction="none")

        losses = []
        for batch_size in (1, 2, 6):
            evaluation = _evaluate(
                model,
                DataLoader(dataset, batch_size=batch_size, shuffle=False),
                criterion,
                torch.device("cpu"),
                aggregate_ids=np.asarray([f"g{index}" for index in indices]),
                sample_ids=np.asarray([f"s{index}" for index in indices]),
                aggregate_groups=False,
                aggregate_field="group_id",
                threshold=0.5,
                walking_criterion=walking_criterion,
                walking_gate_loss_weight=0.25,
            )
            losses.append(float(evaluation["loss"]))

        self.assertTrue(np.allclose(losses, losses[0], rtol=0.0, atol=1e-7), losses)

    def test_pose_augmentation_preserves_masks_and_delta_validity(self) -> None:
        features = torch.zeros(8, 14, 5)
        features[..., 0] = torch.linspace(0.0, 0.7, 8)[:, None]
        features[..., 1] = 0.5
        features[..., 4] = 1.0
        torch.manual_seed(7)

        augmented = _augment_gait_window(
            features,
            temporal_shift_frames=1,
            keypoint_dropout_probability=0.35,
            coordinate_jitter_std=0.01,
        )

        valid = augmented[..., 4] > 0
        self.assertTrue(torch.all(augmented[..., :4][~valid] == 0))
        valid_pairs = valid[1:] & valid[:-1]
        self.assertTrue(torch.all(augmented[1:, :, 2:4][~valid_pairs] == 0))

    def test_pose_augmentation_rebuilds_center_joints_and_recenters_pelvis(self) -> None:
        features = torch.zeros(8, 14, 5)
        features[..., 4] = 1.0
        features[..., 0, :2] = torch.tensor([-0.3, -1.0])
        features[..., 1, :2] = torch.tensor([0.3, -1.0])
        features[..., 6, :2] = torch.tensor([-0.2, 0.0])
        features[..., 7, :2] = torch.tensor([0.2, 0.0])
        features[..., 12, :2] = 0.0
        features[..., 13, :2] = torch.tensor([0.0, -1.0])
        torch.manual_seed(7)

        augmented = _augment_gait_window(
            features,
            temporal_shift_frames=0,
            keypoint_dropout_probability=0.10,
            coordinate_jitter_std=0.005,
        )

        valid = augmented[..., 4] > 0
        pelvis_expected = valid[:, 6] & valid[:, 7]
        shoulder_expected = valid[:, 0] & valid[:, 1]
        self.assertTrue(torch.equal(valid[:, 12], pelvis_expected))
        self.assertTrue(torch.equal(valid[:, 13], shoulder_expected))
        self.assertTrue(torch.all(augmented[~pelvis_expected] == 0))
        torch.testing.assert_close(
            augmented[pelvis_expected, 12, :2],
            torch.zeros_like(augmented[pelvis_expected, 12, :2]),
        )
        torch.testing.assert_close(
            augmented[shoulder_expected, 13, :2],
            (
                augmented[shoulder_expected, 0, :2]
                + augmented[shoulder_expected, 1, :2]
            )
            / 2.0,
        )

    def test_balanced_class_weights_use_effective_training_mass(self) -> None:
        labels = np.asarray([0, 0, 1, 1, 1], dtype=np.int64)
        prepared_weights = np.asarray([0.5, 0.5, 0.1, 0.1, 0.1], dtype=np.float32)
        indices = np.arange(len(labels), dtype=np.int64)

        class_weights = compute_balanced_class_weights(
            labels,
            prepared_weights,
            indices,
        )
        effective_weights = prepared_weights * class_weights[labels]

        self.assertAlmostEqual(
            float(effective_weights[labels == 0].sum()),
            float(effective_weights[labels == 1].sum()),
            places=6,
        )

    def test_threshold_is_selected_from_validation_scores(self) -> None:
        threshold = select_balanced_accuracy_threshold(
            labels=[0, 0, 1, 1],
            scores=[0.10, 0.20, 0.30, 0.40],
        )

        self.assertAlmostEqual(threshold, 0.25)

    def test_builds_centered_quality_aware_t_v_c_tensor(self) -> None:
        records = [make_pose_record(frame_id) for frame_id in range(6)]

        tensor = build_gait_tensor(records, window_frames=8)

        self.assertEqual(tensor.shape, (8, 14, 5))
        self.assertTrue(np.isfinite(tensor).all())
        self.assertTrue(np.allclose(tensor[6:], 0.0))
        self.assertTrue(np.all(tensor[:6, :, 4] > 0.0))
        self.assertTrue(np.allclose(tensor[:6, 12, :2], 0.0, atol=1e-6))

    def test_time_resampling_is_independent_of_source_fps(self) -> None:
        slow = [make_pose_record(frame_id * 5) for frame_id in range(9)]
        fast = [make_pose_record(frame_id * 2) for frame_id in range(21)]

        slow_resampled = resample_pose_records(
            slow,
            start_time_sec=0.0,
            window_frames=8,
            target_fps=5.0,
            max_gap_sec=0.25,
        )
        fast_resampled = resample_pose_records(
            fast,
            start_time_sec=0.0,
            window_frames=8,
            target_fps=5.0,
            max_gap_sec=0.25,
        )

        self.assertEqual(len(slow_resampled), 8)
        self.assertEqual(len(fast_resampled), 8)
        self.assertEqual(
            [row["timestamp_sec"] if row else None for row in slow_resampled],
            [row["timestamp_sec"] if row else None for row in fast_resampled],
        )

    def test_time_resampling_does_not_duplicate_one_source_observation(self) -> None:
        resampled = resample_pose_records(
            [make_pose_record(0)],
            start_time_sec=0.0,
            window_frames=4,
            target_fps=4.0,
            max_gap_sec=0.5,
        )

        self.assertEqual(sum(row is not None for row in resampled), 1)

    def test_training_label_filter_includes_non_walking_normal_hard_negatives(self) -> None:
        rows = [
            make_label(1, positive=False),
            make_label(2, positive=True),
            {**make_label(3, positive=False), "action_id": "A04", "action_name": "normal_stand"},
        ]

        selected = select_gait_training_labels(rows)

        self.assertEqual([row["label"] for row in selected], [0, 1, 0])
        self.assertEqual(
            [row["target_name"] for row in selected],
            ["normal_activity", "gait_instability", "normal_activity"],
        )

    def test_v3_label_filter_uses_half_open_boundaries_and_tiers(self) -> None:
        rows = [
            make_v3_label(1, positive=False),
            make_v3_label(2, positive=True, training_tier="auxiliary"),
            make_v3_label(3, positive=True, training_tier="ignore"),
        ]

        selected = select_gait_training_labels(rows)

        self.assertEqual([row["label"] for row in selected], [0, 1])
        self.assertEqual([row["end_frame"] for row in selected], [9, 9])
        self.assertEqual(
            [row["training_tier"] for row in selected],
            ["primary", "auxiliary"],
        )

    def test_v3_labels_are_joined_to_manifest(self) -> None:
        labels = [make_v3_label(1, positive=True)]

        [enriched] = enrich_gait_training_labels(
            labels,
            manifest_rows=[make_manifest(1)],
        )

        self.assertEqual(enriched["file_path"], "unused/video_1.avi")
        self.assertEqual(enriched["scene"], "test_room")
        self.assertEqual(enriched["dataset"], "test_dataset")
        self.assertEqual(enriched["fps"], 25.0)

    def test_v3_manifest_ignores_null_video_id_assets(self) -> None:
        non_video_asset = {
            "asset_id": "non_video_asset",
            "video_id": None,
            "path": "unused/depth.csv",
            "eligibility": False,
        }

        [enriched] = enrich_gait_training_labels(
            [make_v3_label(1, positive=True)],
            manifest_rows=[non_video_asset, make_manifest(1)],
        )

        self.assertEqual(enriched["video_id"], "video_1")
        self.assertEqual(enriched["file_path"], "unused/video_1.avi")

    def test_v3_manifest_rejects_duplicate_non_empty_video_id(self) -> None:
        manifest = make_manifest(1)

        with self.assertRaisesRegex(ValueError, "duplicate gait manifest video_id: video_1"):
            enrich_gait_training_labels(
                [make_v3_label(1, positive=True)],
                manifest_rows=[manifest, dict(manifest)],
            )

    def test_v3_manifest_join_keeps_strict_validation(self) -> None:
        label = make_v3_label(1, positive=True)
        invalid_cases = (
            ([], "missing manifest video"),
            ([{**make_manifest(1), "eligibility": False}], "ineligible manifest video"),
            ([{**make_manifest(1), "asset_id": "other_asset"}], "asset mismatch"),
        )

        for manifest_rows, error in invalid_cases:
            with self.subTest(error=error):
                with self.assertRaisesRegex(ValueError, error):
                    enrich_gait_training_labels(
                        [label],
                        manifest_rows=manifest_rows,
                    )

    def test_pose_jobs_stop_after_last_selected_gait_frame(self) -> None:
        rows = [
            make_label(1, positive=False),
            {**make_label(1, positive=True), "label_id": "second", "end_frame": 19},
            {**make_label(2, positive=False), "action_id": "A04", "action_name": "normal_stand"},
        ]

        jobs = build_pose_jobs(rows)

        self.assertEqual(len(jobs), 2)
        self.assertEqual(jobs[0]["video_id"], "video_1")
        self.assertEqual(jobs[0]["max_frames"], 20)

    def test_track_matching_prefers_annotation_bbox_over_background_coverage(self) -> None:
        label = make_label(1, positive=True)
        correct = [
            {**make_pose_record(frame_id), "track_id": 1}
            for frame_id in range(8)
        ]
        background = [
            {
                **make_pose_record(frame_id),
                "track_id": 2,
                "bbox": [300.0, 20.0, 420.0, 230.0],
            }
            for frame_id in range(10)
        ]

        selected = _select_labeled_track(label, background + correct)

        self.assertEqual({record["track_id"] for record in selected.values()}, {1})

    def test_track_matching_uses_pixel_bbox_when_pose_bbox_is_normalized(self) -> None:
        label = make_label(1, positive=True)
        correct = [
            {
                **make_pose_record(frame_id),
                "track_id": 1,
                "bbox": [0.25, 0.05, 0.55, 0.575],
                "bbox_pixels": [100.0, 20.0, 220.0, 230.0],
            }
            for frame_id in range(8)
        ]
        background = [
            {
                **make_pose_record(frame_id),
                "track_id": 2,
                "bbox": [0.75, 0.05, 1.05, 0.575],
                "bbox_pixels": [300.0, 20.0, 420.0, 230.0],
            }
            for frame_id in range(10)
        ]

        selected = _select_labeled_track(label, background + correct)

        self.assertEqual({record["track_id"] for record in selected.values()}, {1})


class GaitDatasetPreparationTest(unittest.TestCase):
    def test_context_protocol_retains_short_train_span_as_weak_supervision(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            labels_path = root / "labels_v3.jsonl"
            manifest_path = root / "manifest.jsonl"
            assignments_path = root / "assignments.jsonl"
            split_report_path = root / "split.json"
            pose_dir = root / "poses"
            output_dir = root / "prepared"
            pose_dir.mkdir()
            labels = [
                make_v3_label(index, positive=index % 2 == 1)
                for index in range(1, 5)
            ]
            for label in labels:
                label.update(
                    {
                        "start_frame": 25,
                        "end_frame_exclusive": 57,
                        "start_time": 1.0,
                        "end_time_exclusive": 2.25,
                    }
                )
            assignments = [
                make_assignment(label, "train" if index < 2 else "validation")
                for index, label in enumerate(labels)
            ]
            labels_path.write_text(
                "\n".join(json.dumps(row) for row in labels) + "\n",
                encoding="utf-8",
            )
            manifest_path.write_text(
                "\n".join(json.dumps(make_manifest(index)) for index in range(1, 5))
                + "\n",
                encoding="utf-8",
            )
            assignments_path.write_text(
                "\n".join(json.dumps(row) for row in assignments) + "\n",
                encoding="utf-8",
            )
            split_report_path.write_text(
                json.dumps(
                    {
                        "schema_version": "fall-risk-training-split-v3",
                        "split_id": "context_test_split",
                        "assignments_sha256": hashlib.sha256(
                            assignments_path.read_bytes()
                        ).hexdigest(),
                        "input_sha256": {
                            "action_labels": hashlib.sha256(
                                labels_path.read_bytes()
                            ).hexdigest(),
                            "manifest": hashlib.sha256(
                                manifest_path.read_bytes()
                            ).hexdigest(),
                        },
                        "leakage_issues": [],
                    }
                ),
                encoding="utf-8",
            )
            for index in range(1, 5):
                records = [
                    make_pose_record(frame_id, unstable=index % 2 == 1)
                    for frame_id in range(100)
                ]
                (pose_dir / f"video_{index}.jsonl").write_text(
                    "\n".join(json.dumps(row) for row in records) + "\n",
                    encoding="utf-8",
                )

            prepare_gait_window_dataset(
                labels_path,
                pose_dir,
                output_dir,
                manifest_path=manifest_path,
                assignments_path=assignments_path,
                split_report_path=split_report_path,
                config=GaitWindowPreparationConfig(
                    window_frames=16,
                    stride_frames=8,
                    min_observed_frames=10,
                    target_fps=4.0,
                    max_gap_sec=0.13,
                    context_expansion=True,
                    weak_min_labeled_observations=5,
                ),
            )
            with np.load(output_dir / "dataset.npz", allow_pickle=False) as dataset:
                label_span_masks = dataset["label_span_masks"]
                valid_masks = dataset["valid_masks"]
                evidence_tiers = dataset["evidence_tiers"].astype(str)
                sample_weights = dataset["sample_weights"]
                partitions = dataset["partitions"].astype(str)
            metadata = json.loads(
                (output_dir / "metadata.json").read_text(encoding="utf-8")
            )

        self.assertEqual(label_span_masks.shape, (4, 16))
        self.assertTrue(np.all(label_span_masks <= valid_masks))
        self.assertEqual(set(evidence_tiers), {"weak_context"})
        self.assertTrue(np.allclose(sample_weights[partitions == "train"], 0.35))
        self.assertEqual(metadata["context_protocol"]["validation_policy"], "primary_only")

    def test_formal_dataset_uses_frozen_assignments_and_tier_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            labels_path = root / "labels_v3.jsonl"
            manifest_path = root / "manifest.jsonl"
            assignments_path = root / "assignments.jsonl"
            split_report_path = root / "split.json"
            pose_dir = root / "poses"
            output_dir = root / "prepared"
            pose_dir.mkdir()
            labels = [
                make_v3_label(index, positive=index % 2 == 0)
                for index in range(1, 7)
            ]
            labels[1]["action_id"] = "B01"
            labels[1]["action_type"] = "slow_walk"
            excluded_label_id = str(labels[1]["label_id"])
            auxiliary = make_v3_label(7, positive=True, training_tier="auxiliary")
            labels.extend([auxiliary, make_v3_label(8, positive=False, training_tier="auxiliary")])
            labels[7]["action_id"] = "A04"
            labels[7]["action_type"] = "normal_sit_to_stand"
            for label in labels:
                label["end_frame_exclusive"] = 40
                label["end_time_exclusive"] = 1.6
            labels[0]["end_frame_exclusive"] = 80
            labels[0]["end_time_exclusive"] = 3.2
            partitions = ["train", "train", "validation", "validation", "test", "test"]
            assignments = [
                make_assignment(label, partition)
                for label, partition in zip(labels[:6], partitions, strict=True)
            ]
            assignments.extend(
                [
                    make_assignment(auxiliary, "validation"),
                    make_assignment(labels[7], "train"),
                ]
            )
            labels_path.write_text(
                "\n".join(json.dumps(row) for row in labels) + "\n",
                encoding="utf-8",
            )
            manifest_path.write_text(
                "\n".join(json.dumps(make_manifest(index)) for index in range(1, 9)) + "\n",
                encoding="utf-8",
            )
            assignments_path.write_text(
                "\n".join(json.dumps(row) for row in assignments) + "\n",
                encoding="utf-8",
            )
            split_report_path.write_text(
                json.dumps(
                    {
                        "schema_version": "fall-risk-training-split-v3",
                        "split_id": "test_split",
                        "assignments_sha256": hashlib.sha256(
                            assignments_path.read_bytes()
                        ).hexdigest(),
                        "input_sha256": {
                            "action_labels": hashlib.sha256(
                                labels_path.read_bytes()
                            ).hexdigest(),
                            "manifest": hashlib.sha256(
                                manifest_path.read_bytes()
                            ).hexdigest(),
                        },
                        "leakage_issues": [],
                    }
                ),
                encoding="utf-8",
            )
            for index in range(1, 9):
                if index in (5, 6):
                    continue
                records = [
                    make_pose_record(frame_id, unstable=index % 2 == 0)
                    for frame_id in range(80 if index == 1 else 40)
                ]
                (pose_dir / f"video_{index}.jsonl").write_text(
                    "\n".join(json.dumps(row) for row in records) + "\n",
                    encoding="utf-8",
                )

            prepare_gait_window_dataset(
                labels_path,
                pose_dir,
                output_dir,
                manifest_path=manifest_path,
                assignments_path=assignments_path,
                split_report_path=split_report_path,
                config=GaitWindowPreparationConfig(
                    window_frames=8,
                    stride_frames=4,
                    min_observed_frames=5,
                    target_fps=5.0,
                    max_gap_sec=0.25,
                    max_windows_per_segment=2,
                    auxiliary_weight=0.35,
                    target_profile="observable_instability_b02_b04",
                ),
            )
            with np.load(output_dir / "dataset.npz", allow_pickle=False) as dataset:
                label_ids = dataset["action_segment_ids"].astype(str)
                output_partitions = dataset["partitions"].astype(str)
                tiers = dataset["training_tiers"].astype(str)
                weights = dataset["sample_weights"]
                action_ids = dataset["action_ids"].astype(str)
                walking_targets = dataset["walking_targets"]
            sample_rows = [
                json.loads(line)
                for line in (output_dir / "samples.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            metadata = json.loads(
                (output_dir / "metadata.json").read_text(encoding="utf-8")
            )

        self.assertNotIn(str(auxiliary["label_id"]), set(label_ids))
        self.assertEqual(set(tiers[output_partitions == "validation"]), {"primary"})
        self.assertNotIn("test", set(output_partitions))
        self.assertNotIn(str(labels[4]["label_id"]), set(label_ids))
        self.assertNotIn(str(labels[5]["label_id"]), set(label_ids))
        self.assertNotIn(excluded_label_id, set(label_ids))
        self.assertEqual(
            set(walking_targets[np.isin(action_ids, ["A01", "B02", "B03", "B04"])]),
            {1},
        )
        self.assertEqual(set(walking_targets[action_ids == "A04"]), {0})
        self.assertEqual(
            metadata["target_contract"],
            {
                "functional_proxy_action_ids": ["B01"],
                "negative_action_ids": [
                    "A01", "A02", "A03", "A04", "A05", "A06",
                    "A07", "A08", "A09", "A10", "A11", "A12",
                ],
                "positive_action_ids": ["B02", "B03", "B04"],
                "profile": "observable_instability_b02_b04",
            },
        )
        self.assertTrue(metadata["split_is_provisional"])
        self.assertEqual(metadata["protocol_status"], "development_provisional")
        self.assertFalse(metadata["test_pose_read"])
        self.assertFalse(metadata["test_tensor_generated"])
        self.assertFalse(metadata["test_evaluated"])
        self.assertEqual(metadata["source_split_id"], "test_split")
        self.assertEqual(metadata["test_label_audit"]["label_count"], 2)
        self.assertNotIn("video_5", metadata["pose_inputs"])
        self.assertNotIn("video_6", metadata["pose_inputs"])
        self.assertEqual(
            [
                row["start_frame"]
                for row in sample_rows
                if row["label_id"] == str(labels[0]["label_id"])
            ],
            [0, 40],
        )
        for label_id in set(label_ids):
            self.assertLessEqual(int(np.sum(label_ids == label_id)), 2)
            expected = (
                0.35
                if label_id
                in {str(labels[6]["label_id"]), str(labels[7]["label_id"])}
                else 1.0
            )
            self.assertAlmostEqual(float(weights[label_ids == label_id].sum()), expected, places=6)

    def test_formal_dataset_rejects_missing_assignment(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            labels_path = root / "labels_v3.jsonl"
            manifest_path = root / "manifest.jsonl"
            assignments_path = root / "assignments.jsonl"
            split_report_path = root / "split.json"
            pose_dir = root / "poses"
            pose_dir.mkdir()
            label = make_v3_label(1, positive=True)
            labels_path.write_text(json.dumps(label) + "\n", encoding="utf-8")
            manifest_path.write_text(json.dumps(make_manifest(1)) + "\n", encoding="utf-8")
            assignments_path.write_text("", encoding="utf-8")
            split_report_path.write_text(
                json.dumps(
                    {
                        "schema_version": "fall-risk-training-split-v3",
                        "assignments_sha256": hashlib.sha256(b"").hexdigest(),
                        "input_sha256": {
                            "action_labels": hashlib.sha256(
                                labels_path.read_bytes()
                            ).hexdigest(),
                            "manifest": hashlib.sha256(
                                manifest_path.read_bytes()
                            ).hexdigest(),
                        },
                        "leakage_issues": [],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "missing frozen split assignment"):
                prepare_gait_window_dataset(
                    labels_path,
                    pose_dir,
                    root / "prepared",
                    manifest_path=manifest_path,
                    assignments_path=assignments_path,
                    split_report_path=split_report_path,
                )

    def test_formal_dataset_rejects_manifest_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            labels_path = root / "labels_v3.jsonl"
            manifest_path = root / "manifest.jsonl"
            assignments_path = root / "assignments.jsonl"
            split_report_path = root / "split.json"
            label = make_v3_label(1, positive=True)
            assignment = make_assignment(label, "train")
            labels_path.write_text(json.dumps(label) + "\n", encoding="utf-8")
            manifest_path.write_text(
                json.dumps(make_manifest(1)) + "\n", encoding="utf-8"
            )
            assignments_path.write_text(
                json.dumps(assignment) + "\n", encoding="utf-8"
            )
            split_report_path.write_text(
                json.dumps(
                    {
                        "schema_version": "fall-risk-training-split-v3",
                        "assignments_sha256": hashlib.sha256(
                            assignments_path.read_bytes()
                        ).hexdigest(),
                        "input_sha256": {
                            "action_labels": hashlib.sha256(
                                labels_path.read_bytes()
                            ).hexdigest(),
                            "manifest": "wrong-hash",
                        },
                        "leakage_issues": [],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "manifest SHA-256 mismatch"):
                prepare_gait_window_dataset(
                    labels_path,
                    root / "poses",
                    root / "prepared",
                    manifest_path=manifest_path,
                    assignments_path=assignments_path,
                    split_report_path=split_report_path,
                )

    def test_tabular_training_rejects_fold_that_reuses_locked_partitions(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            dataset_path = root / "dataset.npz"
            labels = np.asarray([0, 1, 0, 1], dtype=np.int64)
            np.savez_compressed(
                dataset_path,
                labels=labels,
                partitions=np.asarray(["train", "validation", "test", "test"]),
                fold_a_partitions=np.asarray(
                    ["train", "train", "validation", "validation"]
                ),
                sample_ids=np.asarray(["s0", "s1", "s2", "s3"]),
                action_segment_ids=np.asarray(["a0", "a1", "a2", "a3"]),
                action_ids=np.asarray(["A01", "B03", "A01", "B03"]),
                datasets=np.asarray(["source_a", "source_a", "source_b", "source_b"]),
                source_group_ids=np.asarray(["g0", "g1", "g2", "g3"]),
                segment_durations_sec=np.asarray([4.0, 4.0, 4.0, 4.0]),
                tabular_features=np.asarray(
                    [
                        [0.1, 0.30, 0.95],
                        [0.9, 0.30, 0.95],
                        [0.2, 0.70, 0.40],
                        [0.8, 0.70, 0.40],
                    ],
                    dtype=np.float32,
                ),
                rule_scores=np.asarray([0.1, 0.9, 0.2, 0.8], dtype=np.float32),
            )
            metadata_path = root / "metadata.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "task": "gait_instability_vs_normal_activity",
                        "dataset_sha256": hashlib.sha256(
                            dataset_path.read_bytes()
                        ).hexdigest(),
                        "tabular_feature_names": [
                            "feature",
                            "hip_width_mean",
                            "mean_core_keypoint_quality",
                        ],
                        "partition_schemes": {"fold_a": "fold_a_partitions"},
                        "split_protocol": "frozen_training_labels_v3",
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError, "reuses frozen assignments"
            ):
                train_gait_tabular_baselines(
                    dataset_path,
                    root / "output",
                    metadata_path=metadata_path,
                    config=GaitTabularTrainingConfig(
                        models=("rule",),
                        partition_scheme="fold_a",
                        feature_profile="exclude_quality_and_scale",
                    ),
                )
    def test_prepares_common_dataset_and_keeps_videos_in_one_partition(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            labels_path = root / "labels.jsonl"
            pose_dir = root / "poses"
            output_dir = root / "prepared"
            pose_dir.mkdir()
            labels = [
                make_label(index, positive=index >= 4)
                for index in range(1, 7)
            ]
            labels_path.write_text(
                "\n".join(json.dumps(row) for row in labels) + "\n",
                encoding="utf-8",
            )
            for index in range(1, 7):
                records = [
                    make_pose_record(frame_id, unstable=index >= 4)
                    for frame_id in range(10)
                ]
                (pose_dir / f"video_{index}.jsonl").write_text(
                    "\n".join(json.dumps(row) for row in records) + "\n",
                    encoding="utf-8",
                )

            summary = prepare_gait_window_dataset(
                labels_path,
                pose_dir,
                output_dir,
                config=GaitWindowPreparationConfig(
                    window_frames=8,
                    stride_frames=8,
                    min_observed_frames=5,
                    train_fraction=1 / 3,
                    validation_fraction=1 / 3,
                    split_search_iterations=128,
                ),
            )
            with np.load(output_dir / "dataset.npz", allow_pickle=False) as dataset:
                features = dataset["features"]
                labels_out = dataset["labels"]
                video_ids = dataset["split_group_ids"]
                partitions = dataset["partitions"]
                tabular_features = dataset["tabular_features"]
                rule_scores = dataset["rule_scores"]
            metadata = json.loads(
                (output_dir / "metadata.json").read_text(encoding="utf-8")
            )

        self.assertEqual(features.shape[1:], (8, 14, 5))
        self.assertEqual(len(features), len(tabular_features))
        self.assertEqual(len(features), len(rule_scores))
        self.assertEqual(set(labels_out.tolist()), {0, 1})
        self.assertEqual(summary["split_group"], "video_id")
        self.assertTrue(metadata["split_is_provisional"])
        partition_by_video: dict[str, set[str]] = {}
        for video_id, partition in zip(video_ids, partitions, strict=True):
            partition_by_video.setdefault(str(video_id), set()).add(str(partition))
        self.assertTrue(all(len(values) == 1 for values in partition_by_video.values()))
        for partition in ("train", "validation", "test"):
            partition_labels = {
                int(label)
                for label, assigned in zip(labels_out, partitions, strict=True)
                if assigned == partition
            }
            self.assertEqual(partition_labels, {0, 1})

    def test_tabular_cli_compares_rule_and_lightgbm_on_same_windows(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            labels_path = root / "labels.jsonl"
            pose_dir = root / "poses"
            prepared = root / "prepared"
            output = root / "tabular"
            pose_dir.mkdir()
            labels = [
                make_label(index, positive=index >= 7)
                for index in range(1, 13)
            ]
            labels_path.write_text(
                "\n".join(json.dumps(row) for row in labels) + "\n",
                encoding="utf-8",
            )
            for index in range(1, 13):
                records = [
                    make_pose_record(frame_id, unstable=index >= 7)
                    for frame_id in range(10)
                ]
                (pose_dir / f"video_{index}.jsonl").write_text(
                    "\n".join(json.dumps(row) for row in records) + "\n",
                    encoding="utf-8",
                )
            prepare_gait_window_dataset(
                labels_path,
                pose_dir,
                prepared,
                config=GaitWindowPreparationConfig(
                    window_frames=8,
                    stride_frames=8,
                    min_observed_frames=5,
                    train_fraction=0.5,
                    validation_fraction=0.25,
                    split_search_iterations=256,
                ),
            )
            repo = Path(__file__).resolve().parents[1]
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/train/train_gait_tabular_baselines.py",
                    "--data",
                    str(prepared / "dataset.npz"),
                    "--output-dir",
                    str(output),
                    "--models",
                    "rule,logistic,lightgbm,ebm",
                    "--lightgbm-estimators",
                    "4",
                    "--ebm-max-rounds",
                    "4",
                    "--ebm-outer-bags",
                    "1",
                ],
                cwd=repo,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            test_predictions_path = output / "lightgbm_test_window_predictions.jsonl"

        self.assertEqual(
            set(metrics["models"]), {"rule", "logistic", "lightgbm", "ebm"}
        )
        self.assertTrue(metrics["models"]["logistic"]["feature_importance"])
        self.assertFalse(metrics["test_evaluated"])
        self.assertFalse(test_predictions_path.exists())


if __name__ == "__main__":
    unittest.main()
