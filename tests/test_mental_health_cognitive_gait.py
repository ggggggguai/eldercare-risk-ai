from __future__ import annotations

import math
import unittest
from dataclasses import replace
from datetime import datetime, timedelta

from elderly_monitoring.modules.mental_health.feature_extraction.gait_transfer import (
    CognitiveGaitConfig,
    detect_turn_events,
    extract_cognitive_gait_features,
)


BASE_TIME = datetime.fromisoformat("2026-07-16T09:00:00+08:00")


def pose_record(
    frame_id: int,
    *,
    center_x: float,
    center_y: float = 0.55,
    second: float | None = None,
    person_id: str = "elder_001",
    track_id: int = 1,
    core_quality: float = 0.9,
    valid: bool = True,
    include_wrists: bool = True,
    support_wrist: bool = False,
) -> dict[str, object]:
    timestamp_sec = frame_id * 0.1 if second is None else second
    phase = math.sin(frame_id * math.pi / 2.0)
    shoulder_y = center_y - 0.22
    ankle_y = center_y + 0.34
    keypoint_values = {
        "left_shoulder": (center_x - 0.05, shoulder_y),
        "right_shoulder": (center_x + 0.05, shoulder_y),
        "left_hip": (center_x - 0.04, center_y),
        "right_hip": (center_x + 0.04, center_y),
        "left_knee": (center_x - 0.04, center_y + 0.17),
        "right_knee": (center_x + 0.04, center_y + 0.17),
        "left_ankle": (center_x - 0.05 + 0.02 * phase, ankle_y),
        "right_ankle": (center_x + 0.05 - 0.02 * phase, ankle_y),
    }
    if include_wrists:
        if support_wrist:
            keypoint_values["left_wrist"] = (center_x - 0.24, center_y + 0.02)
            keypoint_values["right_wrist"] = (center_x + 0.08, center_y + 0.02)
        else:
            keypoint_values["left_wrist"] = (center_x - 0.09, center_y + 0.02)
            keypoint_values["right_wrist"] = (center_x + 0.09, center_y + 0.02)

    observed_at = BASE_TIME + timedelta(seconds=timestamp_sec)
    return {
        "frame_id": frame_id,
        "timestamp_sec": round(timestamp_sec, 4),
        "observed_at": observed_at.isoformat(),
        "person_id": person_id,
        "track_id": track_id,
        "device_id": "cam_living",
        "scene_region": "living_room",
        "core_keypoint_quality": core_quality,
        "meters_per_norm_unit": 2.0,
        "keypoints": [
            {
                "name": name,
                "x": round(x, 4),
                "y": round(y, 4),
                "x_smooth": round(x, 4) if valid else None,
                "y_smooth": round(y, 4) if valid else None,
                "score": core_quality,
                "valid": valid,
                "source": "observed",
                "is_jump_outlier": False,
            }
            for name, (x, y) in keypoint_values.items()
        ],
        "window_quality": {
            "mean_core_keypoint_quality": core_quality,
            "low_quality_frame_ratio": 0.0 if valid else 1.0,
            "interpolated_point_ratio": 0.0,
            "jump_outlier_count": 0,
            "usable_for_gait": valid,
            "usable_for_sit_stand": valid,
            "usable_for_near_fall": valid,
        },
    }


def gait_sequence() -> list[dict[str, object]]:
    return [
        pose_record(frame_id, center_x=0.30 + 0.01 * frame_id, track_id=1)
        for frame_id in range(10)
    ]


def sit_to_stand_sequence(*, slow: bool = False) -> list[dict[str, object]]:
    hip_values = [0.70, 0.70, 0.68, 0.62, 0.55, 0.49, 0.45, 0.45, 0.45]
    step = 0.8 if slow else 0.35
    offset = 20
    return [
        pose_record(
            offset + index,
            center_x=0.50,
            center_y=hip_y,
            second=10.0 + index * step,
            track_id=2,
        )
        for index, hip_y in enumerate(hip_values)
    ]


def turn_sequence(*, unstable: bool = False) -> list[dict[str, object]]:
    centers = [
        (0.30, 0.55),
        (0.34, 0.55),
        (0.38, 0.55),
        (0.42, 0.57 if unstable else 0.55),
        (0.40 if unstable else 0.42, 0.60),
        (0.45 if unstable else 0.42, 0.64),
        (0.40 if unstable else 0.42, 0.68),
    ]
    offset = 50
    return [
        pose_record(
            offset + index,
            center_x=x,
            center_y=y,
            second=30.0 + index * (0.8 if unstable and index % 2 else 0.3),
            track_id=3,
        )
        for index, (x, y) in enumerate(centers)
    ]


class MentalHealthCognitiveGaitTest(unittest.TestCase):
    def test_daily_features_include_required_motor_cognitive_clues(self) -> None:
        config = replace(CognitiveGaitConfig(), min_turn_points=3)
        [daily] = extract_cognitive_gait_features(
            gait_sequence() + sit_to_stand_sequence() + turn_sequence(),
            config=config,
        )

        self.assertEqual(daily["person_id"], "elder_001")
        self.assertEqual(daily["date"], "2026-07-16")
        self.assertGreater(daily["gait_speed_norm_per_sec"], 0.0)
        self.assertAlmostEqual(daily["gait_speed_mps"], daily["gait_speed_norm_per_sec"] * 2.0, places=4)
        self.assertGreater(daily["sit_stand_duration_seconds"], 0.0)
        self.assertGreater(daily["turn_duration_seconds"], 0.0)
        self.assertGreaterEqual(daily["turn_stability_score"], 0.0)
        self.assertGreaterEqual(daily["gait_cycle_stability_score"], 0.0)
        self.assertFalse(daily["diagnosis"])

    def test_slow_sit_stand_increases_motor_clue_score(self) -> None:
        config = replace(CognitiveGaitConfig(), min_turn_points=3)
        [normal] = extract_cognitive_gait_features(
            gait_sequence() + sit_to_stand_sequence(slow=False) + turn_sequence(),
            config=config,
        )
        [slow] = extract_cognitive_gait_features(
            gait_sequence() + sit_to_stand_sequence(slow=True) + turn_sequence(),
            config=config,
        )

        self.assertGreater(slow["sit_stand_duration_seconds"], normal["sit_stand_duration_seconds"])
        self.assertGreater(slow["motor_cognitive_clue_score"], normal["motor_cognitive_clue_score"])

    def test_unstable_turn_has_lower_stability_than_smooth_turn(self) -> None:
        config = replace(CognitiveGaitConfig(), min_turn_points=3)
        [smooth] = detect_turn_events(turn_sequence(unstable=False), config=config)
        unstable = min(
            detect_turn_events(turn_sequence(unstable=True), config=config),
            key=lambda item: item["turn_stability_score"],
        )

        self.assertGreater(smooth["turn_stability_score"], unstable["turn_stability_score"])
        self.assertIn("turn_path_sway", unstable["risk_factors"])

    def test_low_quality_pose_does_not_become_false_high_risk(self) -> None:
        config = replace(CognitiveGaitConfig(), min_turn_points=3)
        low_quality = [
            pose_record(
                frame_id,
                center_x=0.30,
                track_id=1,
                core_quality=0.2,
                valid=False,
            )
            for frame_id in range(8)
        ]
        [daily] = extract_cognitive_gait_features(low_quality, config=config)

        self.assertIsNone(daily["gait_speed_norm_per_sec"])
        self.assertIsNone(daily["turn_stability_score"])
        self.assertIsNone(daily["motor_cognitive_clue_score"])
        self.assertIn("gait_speed_unavailable", daily["data_quality_flags"])
        self.assertFalse(daily["diagnosis"])


if __name__ == "__main__":
    unittest.main()
