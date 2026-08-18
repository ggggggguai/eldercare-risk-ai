from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from elderly_monitoring.modules.fall_risk.fall_event_continuous_dataset import (
    ContinuousFallDatasetConfig,
    build_continuous_fall_dataset,
    clean_pose_records_for_causal_training,
)
from elderly_monitoring.modules.fall_risk.fall_event_continuous import (
    FALL_EVENT_CONTINUOUS_JOINTS,
)


def _pose(timestamp: float, *, interpolated: bool = False) -> dict:
    points = []
    for name in FALL_EVENT_CONTINUOUS_JOINTS:
        side = -1 if "left" in name else 1
        if "shoulder" in name:
            x, y = 0.5 + side * 0.08, 0.34
        elif "hip" in name:
            x, y = 0.5 + side * 0.05, 0.56
        elif "knee" in name:
            x, y = 0.5 + side * 0.05, 0.73
        elif "ankle" in name:
            x, y = 0.5 + side * 0.04, 0.90
        else:
            x, y = 0.5 + side * 0.02, 0.20
        points.append(
            {
                "name": name,
                "x": x,
                "y": y,
                "x_smooth": x,
                "y_smooth": y,
                "score": 0.9,
                "quality_weight": 0.9,
                "valid": True,
                "source": "interpolated" if interpolated else "observed",
                "is_jump_outlier": False,
            }
        )
    return {
        "timestamp_sec": timestamp,
        "frame_id": int(round(timestamp * 8)),
        "person_id": "person",
        "track_id": 1,
        "coordinate_system": "image_normalized_0_1",
        "bbox": [0.3, 0.1, 0.7, 0.95],
        "keypoints": points,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_interpolated_keypoints_are_masked_before_tensor_building() -> None:
    records = [_pose(0.0), _pose(0.125, interpolated=True)]
    cleaned = clean_pose_records_for_causal_training(records)

    assert cleaned[0]["keypoints"][0]["valid"] is True
    assert cleaned[1]["keypoints"][0]["valid"] is False
    assert cleaned[1]["keypoints"][0]["causal_masked"] is True
    assert records[1]["keypoints"][0]["valid"] is True


def test_builder_materializes_presence_and_onset_without_future_pose() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        pose_dir = root / "poses"
        pose_path = pose_dir / "video.jsonl"
        _write_jsonl(pose_path, [_pose(index * 0.125) for index in range(48)])
        manifest = [
            {
                "video_id": "video",
                "asset_id": "asset",
                "dataset": "fixture",
                "scene_region": "room",
                "eligibility": True,
                "duration_sec": 6.0,
            }
        ]
        governance = [
            {
                "sample_id": "sample",
                "source_label_id": "event1",
                "source_label_kind": "event",
                "video_id": "video",
                "asset_id": "asset",
                "subject_id": "subject",
                "source_group_id": "source",
                "sample_group_id": "sample_group",
                "split_group_id": "split",
                "partition": "train",
                "dataset": "fixture",
                "scene_region": "room",
                "start_frame": 16,
                "end_frame_exclusive": 32,
                "start_time_sec": 2.0,
                "end_time_exclusive_sec": 4.0,
                "target_presence": 1,
                "supervision_family": "primary_fall",
                "supervision_strength": "primary",
                "boundary_precision": "exact",
                "allowed_heads": ["presence", "onset"],
                "presence_loss_weight": 1.0,
                "onset_loss_weight": 1.0,
                "normalization_group_id": "event1",
                "sampling_weight": 1.0,
                "pose_path": pose_path.as_posix(),
            }
        ]
        result = build_continuous_fall_dataset(
            governance,
            manifest,
            pose_roots=[pose_dir],
            output_dir=root / "out",
            config=ContinuousFallDatasetConfig(
                window_sec=4.0,
                target_fps=8.0,
                min_observed_frames=16,
                min_valid_joint_ratio=0.5,
            ),
        )
        arrays = np.load(result["dataset_path"])
        samples = [
            json.loads(line)
            for line in Path(result["samples_path"]).read_text(encoding="utf-8").splitlines()
        ]

    assert arrays["features"].shape == (1, 32, 17, 20)
    assert arrays["presence_targets"].tolist() == [1]
    assert arrays["onset_masks"].tolist() == [1]
    assert samples[0]["future_observation_count"] == 0
    assert samples[0]["cutoff_time_sec"] == 4.0
    assert samples[0]["partial_context"] is False


def test_builder_left_pads_short_causal_history() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        pose_dir = root / "poses"
        pose_path = pose_dir / "short.jsonl"
        _write_jsonl(pose_path, [_pose(index * 0.125) for index in range(16)])
        governance = [
            {
                "sample_id": "short_sample",
                "source_label_id": "short_event",
                "source_label_kind": "event",
                "video_id": "short",
                "asset_id": "asset",
                "subject_id": "subject",
                "source_group_id": "source",
                "sample_group_id": "sample_group",
                "split_group_id": "split",
                "partition": "train",
                "dataset": "fixture",
                "scene_region": "room",
                "start_frame": 4,
                "end_frame_exclusive": 15,
                "start_time_sec": 0.5,
                "end_time_exclusive_sec": 1.875,
                "target_presence": 1,
                "supervision_family": "auxiliary_fall",
                "supervision_strength": "auxiliary",
                "boundary_precision": "approximate",
                "allowed_heads": ["presence"],
                "presence_loss_weight": 0.45,
                "onset_loss_weight": 0.0,
                "normalization_group_id": "short_event",
                "sampling_weight": 1.0,
                "pose_path": pose_path.as_posix(),
            }
        ]
        manifest = [
            {
                "video_id": "short",
                "asset_id": "asset",
                "eligibility": True,
                "duration_sec": 1.875,
            }
        ]
        result = build_continuous_fall_dataset(
            governance,
            manifest,
            pose_roots=[pose_dir],
            output_dir=root / "out",
            config=ContinuousFallDatasetConfig(min_partial_observed_frames=8),
        )
        arrays = np.load(result["dataset_path"])
        sample = json.loads(Path(result["samples_path"]).read_text().splitlines()[0])

    assert arrays["features"].shape == (1, 32, 17, 20)
    assert np.all(arrays["features"][0, :16, :, 14] == 0.0)
    assert sample["partial_context"] is True
    assert sample["required_observed_frame_count"] == 8
