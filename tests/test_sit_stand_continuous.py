from __future__ import annotations

import tempfile
import unittest
import hashlib
from pathlib import Path

import numpy as np

from scripts.prepare.prepare_sit_stand_continuous_dataset import build_parser
from elderly_monitoring.modules.fall_risk.sit_stand_continuous import (
    SIT_STAND_CONTINUOUS_CHANNELS,
    SitStandContinuousConfig,
    SitStandSamplingConfig,
    build_sit_stand_causal_window,
    prepare_sit_stand_continuous_dataset,
)


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


def _pose(timestamp: float, *, track_id: str = "track_1", future_shift: float = 0.0) -> dict:
    hip_y = 0.72 - (0.04 * timestamp) + future_shift
    coordinates = {
        "left_shoulder": (0.45, hip_y - 0.20),
        "right_shoulder": (0.55, hip_y - 0.20),
        "left_elbow": (0.42, hip_y - 0.10),
        "right_elbow": (0.58, hip_y - 0.10),
        "left_wrist": (0.40, hip_y),
        "right_wrist": (0.60, hip_y),
        "left_hip": (0.46, hip_y),
        "right_hip": (0.54, hip_y),
        "left_knee": (0.46, 0.80),
        "right_knee": (0.54, 0.80),
        "left_ankle": (0.46, 0.94),
        "right_ankle": (0.54, 0.94),
    }
    return {
        "timestamp_sec": timestamp,
        "person_id": "person_1",
        "track_id": track_id,
        "coordinate_system": "image_normalized_0_1",
        "keypoints": [
            {
                "name": name,
                "x": x,
                "y": y,
                "x_smooth": x,
                "y_smooth": y,
                "quality_weight": 0.95,
                "valid": True,
                "source": "observed",
                "is_jump_outlier": False,
            }
            for name, (x, y) in coordinates.items()
        ],
    }


class SitStandContinuousWindowTest(unittest.TestCase):
    def test_future_observations_do_not_change_the_causal_tensor(self) -> None:
        config = SitStandContinuousConfig(
            target_fps=2.0, context_sec=2.0, max_gap_sec=0.6, min_observed_frames=3
        )
        past = [_pose(value) for value in (0.5, 1.0, 1.5, 2.0)]
        baseline = build_sit_stand_causal_window(
            past, cutoff_time_sec=2.0, config=config
        )
        with_future = build_sit_stand_causal_window(
            past + [_pose(2.5, future_shift=100.0)],
            cutoff_time_sec=2.0,
            config=config,
        )

        np.testing.assert_array_equal(baseline.tensor, with_future.tensor)
        self.assertEqual(with_future.metadata["future_observation_count"], 1)

    def test_image_y_preserves_global_vertical_motion(self) -> None:
        config = SitStandContinuousConfig(
            target_fps=2.0, context_sec=2.0, max_gap_sec=0.6, min_observed_frames=3
        )
        window = build_sit_stand_causal_window(
            [_pose(value) for value in (0.5, 1.0, 1.5, 2.0)],
            cutoff_time_sec=2.0,
            config=config,
        )
        image_y = SIT_STAND_CONTINUOUS_CHANNELS.index("image_y")
        pelvis_index = 12

        self.assertGreater(float(np.ptp(window.tensor[:, pelvis_index, image_y])), 0.04)

    def test_duplicate_timestamp_and_track_switch_fail_closed(self) -> None:
        config = SitStandContinuousConfig(
            target_fps=2.0, context_sec=2.0, max_gap_sec=0.6, min_observed_frames=3
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            build_sit_stand_causal_window(
                [_pose(1.0), _pose(1.0)], cutoff_time_sec=2.0, config=config
            )
        with self.assertRaisesRegex(ValueError, "multiple target tracks"):
            build_sit_stand_causal_window(
                [_pose(1.0), _pose(1.5, track_id="track_2")],
                cutoff_time_sec=2.0,
                config=config,
            )

    def test_spatially_continuous_single_person_track_fragments_are_stitched(self) -> None:
        config = SitStandContinuousConfig(
            target_fps=2.0, context_sec=2.0, max_gap_sec=0.6, min_observed_frames=3
        )
        rows = [_pose(value) for value in (0.5, 1.0, 1.5, 2.0)]
        for row in rows:
            row["bbox"] = [0.4, 0.2, 0.6, 0.9]
        rows[-1]["track_id"] = "track_2"

        window = build_sit_stand_causal_window(
            rows, cutoff_time_sec=2.0, config=config
        )

        self.assertEqual(window.status, "valid")
        self.assertEqual(window.metadata["stitched_track_switch_count"], 1)

    def test_long_gap_is_unavailable_instead_of_fabricating_motion(self) -> None:
        config = SitStandContinuousConfig(
            target_fps=2.0, context_sec=2.0, max_gap_sec=0.2, min_observed_frames=3
        )
        window = build_sit_stand_causal_window(
            [_pose(0.5), _pose(2.0)], cutoff_time_sec=2.0, config=config
        )

        self.assertEqual(window.status, "unavailable")
        self.assertEqual(window.metadata["unavailable_reason"], "insufficient_observed_frames")

    def test_short_clip_is_left_padded_and_keeps_observation_threshold(self) -> None:
        config = SitStandContinuousConfig(
            target_fps=2.0, context_sec=8.0, max_gap_sec=0.6, min_observed_frames=3
        )

        window = build_sit_stand_causal_window(
            [_pose(value) for value in (0.5, 1.0, 1.5, 2.0)],
            cutoff_time_sec=2.0,
            config=config,
        )

        frame_mask = SIT_STAND_CONTINUOUS_CHANNELS.index("frame_mask")
        self.assertEqual(window.status, "valid")
        self.assertEqual(window.metadata["observed_frame_count"], 4)
        self.assertEqual(window.metadata["valid_joint_ratio"], 1.0)
        self.assertEqual(
            window.metadata["valid_joint_ratio_denominator"],
            "observed_frames_only",
        )
        self.assertEqual(float(window.tensor[0, 0, frame_mask]), 0.0)


class SitStandContinuousDatasetTest(unittest.TestCase):
    def test_prepare_cli_accepts_recall_balanced_sampling_parameters(self) -> None:
        args = build_parser().parse_args(
            [
                "--labels",
                "labels.jsonl",
                "--review-log",
                "review.jsonl",
                "--assignments",
                "assignments.jsonl",
                "--pose-dir",
                "pose",
                "--output-dir",
                "output",
                "--sampling-policy",
                "balanced_causal_v2",
                "--event-cutoff-fractions",
                "0.25",
                "0.5",
                "0.75",
                "1.0",
                "--post-event-offsets-sec",
                "0.25",
                "0.5",
                "--target-weight-shares",
                "0.4",
                "0.3",
                "0.3",
            ]
        )

        self.assertEqual(args.event_cutoff_fractions, [0.25, 0.5, 0.75, 1.0])
        self.assertEqual(args.post_event_offsets_sec, [0.25, 0.5])
        self.assertEqual(args.target_weight_shares, [0.4, 0.3, 0.3])

    def test_balanced_causal_sampling_uses_active_and_post_event_cutoffs(self) -> None:
        labels = [
            {
                "label_id": "event",
                "event_id": "event",
                "video_id": "video",
                "source_group_id": "source",
                "interval_type": "event",
                "eligibility": "eligible",
                "onset_time": 1.0,
                "offset_time": 2.0,
                "transition_type": "sit_to_stand",
            }
        ]
        assignments = [
            {"label_id": "event", "partition": "train", "split_group_id": "g1"}
        ]
        config = SitStandContinuousConfig(
            target_fps=4.0,
            context_sec=2.0,
            max_gap_sec=0.3,
            min_observed_frames=4,
            min_partial_observed_frames=2,
        )
        sampling = SitStandSamplingConfig(
            event_cutoff_fractions=(0.5, 1.0),
            post_event_offsets_sec=(0.5,),
            background_cutoff_fractions=(1.0,),
        )

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "dataset"
            result = prepare_sit_stand_continuous_dataset(
                labels,
                assignments,
                pose_reader=lambda _: [_pose(index / 4) for index in range(1, 11)],
                output_dir=output,
                config=config,
                sampling=sampling,
                manifest=[{"video_id": "video", "dataset": "fixture"}],
            )
            samples = [
                __import__("json").loads(line)
                for line in (output / "samples.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            with np.load(output / "dataset.npz", allow_pickle=False) as archive:
                targets = archive["targets"]
                boundaries = archive["boundary_targets"]

        self.assertEqual(result["sampling_policy"], "balanced_causal_v2")
        self.assertEqual(
            result["sampling_config"],
            {
                "event_cutoff_fractions": [0.5, 1.0],
                "post_event_offsets_sec": [0.5],
                "background_cutoff_fractions": [1.0],
                "auxiliary_weight": 0.5,
                "maximum_source_balance_factor": 4.0,
                "target_weight_shares": [0.5, 0.25, 0.25],
            },
        )
        self.assertEqual([row["cutoff_role"] for row in samples], ["active", "active", "post_event"])
        self.assertEqual(targets.tolist(), [1, 1, 0])
        self.assertEqual(float(boundaries[0, :, 1].sum()), 0.0)
        self.assertEqual(float(boundaries[1, :, 1].sum()), 1.0)
        self.assertAlmostEqual(sum(row["sample_weight"] for row in samples), 1.0, places=6)
        self.assertAlmostEqual(
            sum(row["sample_weight"] for row in samples if row["target"] == 0),
            2.0 / 3.0,
            places=6,
        )

    def test_dominant_track_is_selected_when_distractor_is_intermittent(self) -> None:
        labels = [
            {
                "label_id": "event",
                "event_id": "event",
                "video_id": "video",
                "interval_type": "event",
                "onset_time": 0.5,
                "offset_time": 2.0,
                "transition_type": "sit_to_stand",
            }
        ]
        assignments = [
            {"label_id": "event", "partition": "train", "split_group_id": "g1"}
        ]
        primary = [_pose(value) for value in (0.5, 1.0, 1.5, 2.0)]
        distractor = _pose(1.0, track_id="track_2")
        distractor["person_id"] = "person_2"
        for row in [*primary, distractor]:
            row["bbox"] = [0.4, 0.2, 0.6, 0.9]
        config = SitStandContinuousConfig(
            target_fps=2.0,
            context_sec=2.0,
            max_gap_sec=0.6,
            min_observed_frames=3,
            min_partial_observed_frames=2,
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = prepare_sit_stand_continuous_dataset(
                labels,
                assignments,
                pose_reader=lambda _: [*primary, distractor],
                output_dir=Path(tmp) / "dataset",
                config=config,
            )

        self.assertEqual(result["sample_count"], 1)
        self.assertEqual(result["target_selection_counts"], {"dominant_track": 1})

    def test_short_annotated_interval_uses_partial_context_threshold(self) -> None:
        labels = [
            {
                "label_id": "event",
                "event_id": "event",
                "video_id": "video",
                "interval_type": "event",
                "onset_time": 0.0,
                "offset_time": 1.0,
                "transition_type": "stand_to_sit",
            }
        ]
        assignments = [
            {"label_id": "event", "partition": "train", "split_group_id": "g1"}
        ]
        config = SitStandContinuousConfig(
            target_fps=2.0,
            context_sec=4.0,
            max_gap_sec=0.6,
            min_observed_frames=4,
            min_partial_observed_frames=2,
        )

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "dataset"
            result = prepare_sit_stand_continuous_dataset(
                labels,
                assignments,
                pose_reader=lambda _: [_pose(0.5), _pose(1.0)],
                output_dir=output,
                config=config,
            )
            samples = (output / "samples.jsonl").read_text(encoding="utf-8")
            with np.load(output / "dataset.npz", allow_pickle=False) as archive:
                frame_targets = archive["frame_targets"]
                boundary_targets = archive["boundary_targets"]
                supervision_masks = archive["supervision_masks"]

        self.assertEqual(result["sample_count"], 1)
        self.assertIn('"partial_context": true', samples)
        self.assertIn('"required_observed_frame_count": 2', samples)
        self.assertEqual(frame_targets.shape, (1, 8))
        self.assertEqual(boundary_targets.shape, (1, 8, 2))
        self.assertEqual(int(np.sum(supervision_masks)), 2)
        self.assertEqual(set(frame_targets[frame_targets >= 0].tolist()), {2})

    def test_bad_track_interval_is_rejected_without_aborting_other_samples(self) -> None:
        labels = [
            {
                "label_id": "bad",
                "event_id": "bad",
                "video_id": "bad_video",
                "interval_type": "event",
                "onset_time": 0.5,
                "offset_time": 2.0,
                "transition_type": "sit_to_stand",
            },
            {
                "label_id": "good",
                "event_id": "good",
                "video_id": "good_video",
                "interval_type": "event",
                "onset_time": 0.5,
                "offset_time": 2.0,
                "transition_type": "stand_to_sit",
            },
        ]
        assignments = [
            {"label_id": row["label_id"], "partition": "train", "split_group_id": row["label_id"]}
            for row in labels
        ]

        def reader(video_id: str) -> list[dict]:
            rows = [_pose(value) for value in (0.5, 1.0, 1.5, 2.0)]
            if video_id == "bad_video":
                rows[-1] = _pose(2.0, track_id="track_2")
            return rows

        config = SitStandContinuousConfig(
            target_fps=2.0, context_sec=2.0, max_gap_sec=0.6, min_observed_frames=3
        )
        with tempfile.TemporaryDirectory() as tmp:
            result = prepare_sit_stand_continuous_dataset(
                labels,
                assignments,
                pose_reader=reader,
                output_dir=Path(tmp) / "dataset",
                config=config,
            )

        self.assertEqual(result["sample_count"], 1)
        self.assertEqual(result["rejected"], {"multiple_target_tracks": 1})
        self.assertFalse(result["continuous_model_training_allowed"])

    def test_only_event_and_explicit_background_generate_windows_and_test_pose_is_not_read(self) -> None:
        labels = [
            {
                "label_id": "event_train",
                "event_id": "event_train",
                "video_id": "video_train",
                "interval_type": "event",
                "onset_time": 0.5,
                "offset_time": 2.0,
                "transition_type": "sit_to_stand",
            },
            {
                "label_id": "ignore_train",
                "event_id": "ignore_train",
                "video_id": "video_train",
                "interval_type": "ignore",
                "onset_time": 2.0,
                "offset_time": 3.0,
                "transition_type": None,
            },
            {
                "label_id": "event_test",
                "event_id": "event_test",
                "video_id": "video_test",
                "interval_type": "event",
                "onset_time": 0.5,
                "offset_time": 2.0,
                "transition_type": "stand_to_sit",
            },
        ]
        assignments = [
            {"label_id": "event_train", "partition": "train", "split_group_id": "g1"},
            {"label_id": "ignore_train", "partition": "train", "split_group_id": "g1"},
            {"label_id": "event_test", "partition": "test", "split_group_id": "g2"},
        ]
        calls: list[str] = []

        def reader(video_id: str) -> list[dict]:
            calls.append(video_id)
            if video_id == "video_test":
                raise AssertionError("test pose was opened")
            return [_pose(value) for value in (0.5, 1.0, 1.5, 2.0)]

        config = SitStandContinuousConfig(
            target_fps=2.0, context_sec=2.0, max_gap_sec=0.6, min_observed_frames=3
        )
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "dataset"
            result = prepare_sit_stand_continuous_dataset(
                labels,
                assignments,
                pose_reader=reader,
                output_dir=output_dir,
                config=config,
            )
            written_assignments = (output_dir / "assignments.jsonl").read_text()

        self.assertEqual(calls, ["video_train"])
        self.assertEqual(result["sample_count"], 1)
        self.assertEqual(result["locked_test_label_count"], 1)
        self.assertFalse(result["test_pose_read"])
        self.assertNotIn("event_test", written_assignments)

    def test_two_clean_builds_are_byte_deterministic(self) -> None:
        labels = [
            {
                "label_id": "event_train",
                "event_id": "event_train",
                "video_id": "video_train",
                "interval_type": "event",
                "onset_time": 0.5,
                "offset_time": 2.0,
                "transition_type": "sit_to_stand",
            }
        ]
        assignments = [
            {"label_id": "event_train", "partition": "train", "split_group_id": "g1"}
        ]
        config = SitStandContinuousConfig(
            target_fps=2.0, context_sec=2.0, max_gap_sec=0.6, min_observed_frames=3
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("first", "second"):
                prepare_sit_stand_continuous_dataset(
                    labels,
                    assignments,
                    pose_reader=lambda _: [_pose(value) for value in (0.5, 1.0, 1.5, 2.0)],
                    output_dir=root / name,
                    config=config,
                )
            for filename in (
                "dataset.npz",
                "samples.jsonl",
                "metadata.json",
                "assignments.jsonl",
                "split.json",
                "preparation_report.json",
            ):
                first_hash = hashlib.sha256((root / "first" / filename).read_bytes()).hexdigest()
                second_hash = hashlib.sha256((root / "second" / filename).read_bytes()).hexdigest()
                self.assertEqual(first_hash, second_hash, filename)


if __name__ == "__main__":
    unittest.main()
