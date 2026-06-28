import json
import tempfile
import unittest
from pathlib import Path

from elderly_monitoring.modules.fall_risk.near_fall import (
    NearFallDetectionConfig,
    extract_near_fall_events,
    run_near_fall_jsonl,
)


CORE_POINTS = (
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


def make_near_fall_record(
    frame_id: int,
    *,
    center_x: float = 0.50,
    hip_y: float = 0.55,
    timestamp_sec: float | None = None,
    person_id: str = "elder_001",
    track_id: int = 1,
    shoulder_offset_x: float = 0.0,
    knee_offset_y: float = 0.16,
    ankle_offset_y: float = 0.32,
    usable_for_near_fall: bool = True,
    valid: bool = True,
    core_quality: float = 0.9,
    include_wrists: bool = True,
    support_wrist: bool = False,
    wrist_swing: float = 0.0,
    jump_outlier: bool = False,
) -> dict[str, object]:
    shoulder_y = hip_y - 0.22
    point_values = {
        "left_shoulder": (center_x - 0.05 + shoulder_offset_x, shoulder_y),
        "right_shoulder": (center_x + 0.05 + shoulder_offset_x, shoulder_y),
        "left_hip": (center_x - 0.04, hip_y),
        "right_hip": (center_x + 0.04, hip_y),
        "left_knee": (center_x - 0.04, hip_y + knee_offset_y),
        "right_knee": (center_x + 0.04, hip_y + knee_offset_y),
        "left_ankle": (center_x - 0.04, hip_y + ankle_offset_y),
        "right_ankle": (center_x + 0.04, hip_y + ankle_offset_y),
    }
    if include_wrists:
        if support_wrist:
            point_values["left_wrist"] = (0.22 + wrist_swing, shoulder_y + 0.16)
            point_values["right_wrist"] = (center_x + 0.09, shoulder_y + 0.15)
        else:
            point_values["left_wrist"] = (center_x - 0.09 + wrist_swing, shoulder_y + 0.15)
            point_values["right_wrist"] = (center_x + 0.09, shoulder_y + 0.15)

    keypoints = [
        {
            "name": name,
            "x": round(x, 4) if valid else None,
            "y": round(y, 4) if valid else None,
            "x_smooth": round(x, 4) if valid else None,
            "y_smooth": round(y, 4) if valid else None,
            "score": core_quality,
            "valid": valid,
            "source": "observed",
            "is_jump_outlier": jump_outlier and name in CORE_POINTS,
        }
        for name, (x, y) in point_values.items()
    ]
    return {
        "frame_id": frame_id,
        "timestamp_sec": round(frame_id * 0.2 if timestamp_sec is None else timestamp_sec, 4),
        "person_id": person_id,
        "track_id": track_id,
        "scene_region": "home",
        "core_keypoint_quality": core_quality,
        "keypoints": keypoints,
        "window_quality": {
            "mean_core_keypoint_quality": core_quality,
            "low_quality_frame_ratio": 0.0 if usable_for_near_fall else 1.0,
            "interpolated_point_ratio": 0.0,
            "jump_outlier_count": len(CORE_POINTS) if jump_outlier else 0,
            "usable_for_gait": usable_for_near_fall,
            "usable_for_sit_stand": usable_for_near_fall,
            "usable_for_near_fall": usable_for_near_fall,
        },
    }


def make_sequence(
    center_values: list[float],
    *,
    hip_values: list[float] | None = None,
    step_sec: float = 0.2,
    person_id: str = "elder_001",
    track_id: int = 1,
    shoulder_offsets: list[float] | None = None,
    knee_offsets: list[float] | None = None,
    ankle_offsets: list[float] | None = None,
    usable_for_near_fall: bool = True,
    valid: bool = True,
    include_wrists: bool = True,
    support_wrist: bool = False,
    wrist_swings: list[float] | None = None,
    jump_outlier_indices: set[int] | None = None,
) -> list[dict[str, object]]:
    hips = hip_values or [0.55] * len(center_values)
    shoulder_offsets = shoulder_offsets or [0.0] * len(center_values)
    knee_offsets = knee_offsets or [0.16] * len(center_values)
    ankle_offsets = ankle_offsets or [0.32] * len(center_values)
    wrist_swings = wrist_swings or [0.0] * len(center_values)
    jump_outlier_indices = jump_outlier_indices or set()
    return [
        make_near_fall_record(
            frame_id=index,
            center_x=center,
            hip_y=hips[index],
            timestamp_sec=index * step_sec,
            person_id=person_id,
            track_id=track_id,
            shoulder_offset_x=shoulder_offsets[index],
            knee_offset_y=knee_offsets[index],
            ankle_offset_y=ankle_offsets[index],
            usable_for_near_fall=usable_for_near_fall,
            valid=valid,
            core_quality=0.9 if usable_for_near_fall else 0.2,
            include_wrists=include_wrists,
            support_wrist=support_wrist,
            wrist_swing=wrist_swings[index],
            jump_outlier=index in jump_outlier_indices,
        )
        for index, center in enumerate(center_values)
    ]


def fixed_window_config() -> NearFallDetectionConfig:
    return NearFallDetectionConfig(window_frames=8, step_frames=8, min_event_frames=5, min_output_score=0.0)


class FallRiskNearFallTest(unittest.TestCase):
    def test_stable_pose_window_has_low_score_and_feature_details(self) -> None:
        records = make_sequence([0.50, 0.505, 0.51, 0.515, 0.52, 0.525, 0.53, 0.535])

        [event] = extract_near_fall_events(records, config=fixed_window_config())

        self.assertLess(event["near_fall_event_score"], 0.25)
        self.assertEqual(event["event_type"], "unknown_near_fall")
        self.assertFalse(event["quality_coverage"]["insufficient_near_fall_quality"])
        self.assertIn("body_center_lateral_velocity_peak", event["near_fall_features"])
        self.assertEqual(event["risk_factors"], [])
        self.assertNotIn("risk_level", event)
        self.assertNotIn("recommended_action", event)

    def test_lateral_loss_of_balance_increases_score(self) -> None:
        records = make_sequence([0.50, 0.51, 0.58, 0.67, 0.59, 0.52, 0.51, 0.52])

        [event] = extract_near_fall_events(records, config=fixed_window_config())

        self.assertEqual(event["event_type"], "stumble_or_lateral_loss")
        self.assertGreaterEqual(event["near_fall_event_score"], 0.25)
        self.assertIn("body_center_lateral_acceleration_spike", event["risk_factors"])
        self.assertIn("body_center_path_deviation", event["risk_factors"])

    def test_rapid_descent_recovery_increases_score(self) -> None:
        hip_values = [0.55, 0.56, 0.64, 0.73, 0.66, 0.57, 0.55, 0.55]
        records = make_sequence([0.50] * len(hip_values), hip_values=hip_values)

        [event] = extract_near_fall_events(records, config=fixed_window_config())

        self.assertEqual(event["event_type"], "rapid_descent_recovery")
        self.assertGreaterEqual(event["near_fall_event_score"], 0.20)
        self.assertIn("hip_vertical_drop_recovery", event["risk_factors"])

    def test_sudden_stop_recovery_increases_score(self) -> None:
        centers = [0.30, 0.38, 0.46, 0.46, 0.461, 0.462, 0.54, 0.62]
        records = make_sequence(centers)

        [event] = extract_near_fall_events(records, config=fixed_window_config())

        self.assertEqual(event["event_type"], "sudden_stop_recovery")
        self.assertGreaterEqual(event["near_fall_event_score"], 0.35)
        self.assertIn("sudden_stop_recovery", event["risk_factors"])

    def test_support_proxy_increases_score_but_cannot_strong_trigger_alone(self) -> None:
        records = make_sequence(
            [0.50, 0.505, 0.51, 0.515, 0.52, 0.525, 0.53, 0.535],
            support_wrist=True,
            wrist_swings=[0.20, 0.12, 0.04, 0.0, 0.0, 0.0, 0.0, 0.0],
        )

        [event] = extract_near_fall_events(records, config=fixed_window_config())

        self.assertEqual(event["event_type"], "support_contact_proxy")
        self.assertGreater(event["near_fall_event_score"], 0.0)
        self.assertLess(event["near_fall_event_score"], 0.50)
        self.assertIn("support_contact_proxy", event["risk_factors"])

    def test_low_quality_window_is_marked_insufficient_without_high_risk(self) -> None:
        records = make_sequence(
            [0.50, 0.70, 0.30, 0.72, 0.31, 0.74, 0.32, 0.75],
            usable_for_near_fall=False,
            valid=False,
        )

        [event] = extract_near_fall_events(records, config=fixed_window_config())

        self.assertEqual(event["near_fall_event_score"], 0.0)
        self.assertEqual(event["event_type"], "unknown_near_fall")
        self.assertTrue(event["quality_coverage"]["insufficient_near_fall_quality"])
        self.assertIn("insufficient_near_fall_quality", event["risk_factors"])

    def test_multiple_tracks_are_analyzed_independently(self) -> None:
        first = make_sequence([0.50, 0.51, 0.58, 0.67, 0.59, 0.52, 0.51, 0.52])
        second = make_sequence(
            [0.40, 0.405, 0.41, 0.415, 0.42, 0.425, 0.43, 0.435],
            person_id="elder_002",
            track_id=2,
        )

        events = extract_near_fall_events(first + second, config=fixed_window_config())
        keys = {(event["person_id"], event["track_id"]) for event in events}

        self.assertEqual(keys, {("elder_001", 1), ("elder_002", 2)})
        self.assertEqual(events[0]["event_type"], "stumble_or_lateral_loss")
        self.assertLess(events[1]["near_fall_event_score"], 0.25)

    def test_missing_wrists_do_not_block_core_detection(self) -> None:
        records = make_sequence(
            [0.50, 0.51, 0.58, 0.67, 0.59, 0.52, 0.51, 0.52],
            include_wrists=False,
        )

        [event] = extract_near_fall_events(records, config=fixed_window_config())

        self.assertEqual(event["event_type"], "stumble_or_lateral_loss")
        self.assertFalse(event["near_fall_features"]["wrist_support_proxy"]["suspected"])

    def test_single_jump_outlier_does_not_create_high_score(self) -> None:
        records = make_sequence(
            [0.50, 0.505, 0.51, 0.85, 0.515, 0.52, 0.525, 0.53],
            jump_outlier_indices={3},
        )

        [event] = extract_near_fall_events(records, config=fixed_window_config())

        self.assertLess(event["near_fall_event_score"], 0.50)
        self.assertIn("jump_outlier_count", event["quality_coverage"])

    def test_run_near_fall_jsonl_reads_and_writes_event_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "poses_cleaned.jsonl"
            output_path = Path(tmpdir) / "near_fall.jsonl"
            input_path.write_text(
                "\n".join(
                    json.dumps(record, ensure_ascii=False)
                    for record in make_sequence([0.50, 0.51, 0.58, 0.67, 0.59, 0.52, 0.51, 0.52])
                ),
                encoding="utf-8",
            )

            count = run_near_fall_jsonl(
                input_path=input_path,
                output_path=output_path,
                config=fixed_window_config(),
            )
            lines = output_path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(count, 1)
        self.assertEqual(len(lines), 1)
        payload = json.loads(lines[0])
        self.assertIn("near_fall_event_score", payload)
        self.assertIn("near_fall_features", payload)
        self.assertIn("quality_coverage", payload)
        self.assertEqual(payload["model_version"], "near-fall-rule-v0.1")


if __name__ == "__main__":
    unittest.main()
