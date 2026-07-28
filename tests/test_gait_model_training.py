from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from elderly_monitoring.modules.fall_risk.gait_training import (
    GaitWindowPreparationConfig,
    _select_labeled_track,
    build_gait_tensor,
    prepare_gait_window_dataset,
    select_gait_training_labels,
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


class GaitTensorTest(unittest.TestCase):
    def test_builds_centered_quality_aware_t_v_c_tensor(self) -> None:
        records = [make_pose_record(frame_id) for frame_id in range(6)]

        tensor = build_gait_tensor(records, window_frames=8)

        self.assertEqual(tensor.shape, (8, 14, 5))
        self.assertTrue(np.isfinite(tensor).all())
        self.assertTrue(np.allclose(tensor[6:], 0.0))
        self.assertTrue(np.all(tensor[:6, :, 4] > 0.0))
        self.assertTrue(np.allclose(tensor[:6, 12, :2], 0.0, atol=1e-6))

    def test_training_label_filter_excludes_non_walking_normal_activity(self) -> None:
        rows = [
            make_label(1, positive=False),
            make_label(2, positive=True),
            {**make_label(3, positive=False), "action_id": "A04", "action_name": "normal_stand"},
        ]

        selected = select_gait_training_labels(rows)

        self.assertEqual([row["label"] for row in selected], [0, 1])
        self.assertEqual(
            [row["target_name"] for row in selected],
            ["normal_activity", "gait_instability"],
        )

    def test_pose_jobs_stop_after_last_selected_gait_frame(self) -> None:
        rows = [
            make_label(1, positive=False),
            {**make_label(1, positive=True), "label_id": "second", "end_frame": 19},
            {**make_label(2, positive=False), "action_id": "A04", "action_name": "normal_stand"},
        ]

        jobs = build_pose_jobs(rows)

        self.assertEqual(len(jobs), 1)
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
                    "rule,lightgbm,ebm",
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
            predictions = [
                json.loads(line)
                for line in (
                    output / "lightgbm_test_window_predictions.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(set(metrics["models"]), {"rule", "lightgbm", "ebm"})
        self.assertTrue(predictions)
        self.assertTrue(
            all(0.0 <= row["gait_risk_score"] <= 1.0 for row in predictions)
        )


if __name__ == "__main__":
    unittest.main()
