from __future__ import annotations

import copy
import unittest

import numpy as np

from elderly_monitoring.modules.fall_risk.fall_event_continuous import (
    FALL_EVENT_CONTINUOUS_CHANNELS,
    FALL_EVENT_CONTINUOUS_JOINTS,
    build_fall_event_causal_window,
    build_fall_event_continuous_tensor,
    resample_causal_pose_records,
)


def _pose_record(
    timestamp_sec: float,
    *,
    descent: float = 0.0,
    person_id: str = "elder_1",
    track_id: int = 7,
) -> dict:
    points: list[dict] = []
    for index, name in enumerate(FALL_EVENT_CONTINUOUS_JOINTS):
        side = -1.0 if "left" in name else 1.0
        if name == "nose":
            x, y = 0.50, 0.18
        elif "eye" in name:
            x, y = 0.50 + side * 0.015, 0.16
        elif "ear" in name:
            x, y = 0.50 + side * 0.03, 0.18
        elif "shoulder" in name:
            x, y = 0.50 + side * 0.08, 0.34
        elif "elbow" in name:
            x, y = 0.50 + side * 0.11, 0.45
        elif "wrist" in name:
            x, y = 0.50 + side * 0.12, 0.56
        elif "hip" in name:
            x, y = 0.50 + side * 0.055, 0.56
        elif "knee" in name:
            x, y = 0.50 + side * 0.05, 0.73
        else:
            x, y = 0.50 + side * 0.045, 0.90
        y += descent
        points.append(
            {
                "name": name,
                "x": x,
                "y": y,
                "x_smooth": x,
                "y_smooth": y,
                "score": 0.9,
                "quality_weight": 0.85,
                "valid": True,
                "source": "observed",
                "is_jump_outlier": False,
            }
        )
    return {
        "timestamp_sec": timestamp_sec,
        "person_id": person_id,
        "track_id": track_id,
        "coordinate_system": "image_normalized_0_1",
        "bbox": [0.30, 0.10 + descent, 0.70, 0.95 + descent],
        "keypoints": points,
    }


class FallEventContinuousTensorTest(unittest.TestCase):
    def test_contract_contains_all_required_causal_streams(self) -> None:
        self.assertEqual(len(FALL_EVENT_CONTINUOUS_JOINTS), 17)
        self.assertEqual(
            FALL_EVENT_CONTINUOUS_CHANNELS,
            (
                "x_body",
                "y_body",
                "motion_x_per_sec",
                "motion_y_per_sec",
                "acceleration_x_per_sec2",
                "acceleration_y_per_sec2",
                "bone_x",
                "bone_y",
                "image_y",
                "bbox_height",
                "quality",
                "valid_mask",
                "interpolated_mask",
                "jump_outlier_mask",
                "frame_mask",
                "delta_t_sec",
                "gap_mask",
                "track_continuity_mask",
                "relative_time_sec",
                "effective_fps",
            ),
        )

    def test_tensor_is_prefix_causal_and_preserves_quality_flags(self) -> None:
        records = [_pose_record(index * 0.125) for index in range(6)]
        records[2]["keypoints"][9]["source"] = "interpolated"
        records[3]["keypoints"][9]["is_jump_outlier"] = True
        slots = [record["timestamp_sec"] for record in records]

        tensor = build_fall_event_continuous_tensor(
            records,
            slot_timestamps_sec=slots,
            cutoff_time_sec=slots[-1],
            max_gap_sec=0.25,
        )
        changed = copy.deepcopy(records)
        changed[-1]["keypoints"][0]["x"] = 0.99
        changed[-1]["keypoints"][0]["x_smooth"] = 0.99
        changed[-1]["keypoints"][5]["y"] = 0.99
        changed[-1]["keypoints"][5]["y_smooth"] = 0.99
        changed_tensor = build_fall_event_continuous_tensor(
            changed,
            slot_timestamps_sec=slots,
            cutoff_time_sec=slots[-1],
            max_gap_sec=0.25,
        )

        self.assertEqual(tensor.shape, (6, 17, 20))
        self.assertTrue(np.allclose(tensor[:-1], changed_tensor[:-1]))
        self.assertEqual(float(tensor[2, 9, 12]), 1.0)
        self.assertEqual(float(tensor[3, 9, 13]), 1.0)
        self.assertTrue(np.allclose(tensor[3:5, 9, 2:6], 0.0))
        self.assertTrue(np.isfinite(tensor).all())

    def test_tensor_rejects_future_observation_in_a_slot(self) -> None:
        with self.assertRaisesRegex(ValueError, "future pose observation"):
            build_fall_event_continuous_tensor(
                [_pose_record(0.2)],
                slot_timestamps_sec=[0.1],
                cutoff_time_sec=0.1,
                max_gap_sec=0.25,
            )

    def test_resampler_uses_latest_past_record_and_marks_long_gap(self) -> None:
        records = [_pose_record(0.0), _pose_record(0.3, descent=0.1)]
        slots, slot_times, source_times = resample_causal_pose_records(
            records,
            cutoff_time_sec=0.4,
            window_sec=0.5,
            target_fps=10.0,
            max_gap_sec=0.11,
        )
        tensor = build_fall_event_continuous_tensor(
            slots,
            slot_timestamps_sec=slot_times,
            cutoff_time_sec=0.4,
            max_gap_sec=0.11,
        )

        self.assertEqual(source_times, [0.0, 0.0, None, 0.3, 0.3])
        self.assertIsNone(slots[2])
        self.assertTrue(np.all(tensor[2, :, 14] == 0.0))
        self.assertTrue(np.all(tensor[2, :, 16] == 1.0))

    def test_window_summary_is_explicitly_infrastructure_only(self) -> None:
        records = [_pose_record(index * 0.125, descent=index * 0.01) for index in range(8)]
        window = build_fall_event_causal_window(
            records,
            cutoff_time_sec=0.875,
            window_sec=1.0,
            target_fps=8.0,
            max_gap_sec=0.2,
        )

        self.assertEqual(window.status, "synthetic_or_development_infrastructure_only")
        self.assertEqual(window.tensor.shape, (8, 17, 20))
        self.assertTrue(window.metadata["causal"])
        self.assertEqual(window.metadata["future_observation_count"], 0)
        self.assertGreater(window.metadata["valid_joint_ratio"], 0.9)

    def test_resampler_rejects_ambiguous_targets(self) -> None:
        with self.assertRaisesRegex(ValueError, "multiple target tracks"):
            resample_causal_pose_records(
                [_pose_record(0.0), _pose_record(0.1, track_id=8)],
                cutoff_time_sec=0.1,
                window_sec=0.2,
                target_fps=10.0,
                max_gap_sec=0.2,
            )

    def test_resampler_rejects_window_before_stream_origin(self) -> None:
        with self.assertRaisesRegex(ValueError, "insufficient causal history"):
            resample_causal_pose_records(
                [_pose_record(0.0)],
                cutoff_time_sec=0.1,
                window_sec=1.0,
                target_fps=8.0,
                max_gap_sec=0.2,
            )

    def test_timing_channels_use_source_time_not_repeated_slot_time(self) -> None:
        records = [_pose_record(0.0), _pose_record(0.3, descent=0.1)]
        slots, slot_times, _ = resample_causal_pose_records(
            records,
            cutoff_time_sec=0.4,
            window_sec=0.5,
            target_fps=10.0,
            max_gap_sec=0.4,
        )
        tensor = build_fall_event_continuous_tensor(
            slots,
            slot_timestamps_sec=slot_times,
            cutoff_time_sec=0.4,
            max_gap_sec=0.4,
        )

        delta_t = tensor[:, 0, FALL_EVENT_CONTINUOUS_CHANNELS.index("delta_t_sec")]
        relative_time = tensor[
            :, 0, FALL_EVENT_CONTINUOUS_CHANNELS.index("relative_time_sec")
        ]
        effective_fps = tensor[
            :, 0, FALL_EVENT_CONTINUOUS_CHANNELS.index("effective_fps")
        ]
        np.testing.assert_allclose(delta_t, [0.0, 0.0, 0.0, 0.3, 0.0])
        np.testing.assert_allclose(
            relative_time, [-0.4, -0.3, -0.2, -0.1, 0.0], atol=1e-6
        )
        np.testing.assert_allclose(
            effective_fps, [0.0, 0.0, 0.0, 1.0 / 0.3, 0.0], atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
