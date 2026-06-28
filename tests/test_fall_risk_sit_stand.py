import json
import tempfile
import unittest
from pathlib import Path

from elderly_monitoring.modules.fall_risk.sit_stand import (
    SitStandAnalysisConfig,
    extract_sit_stand_events,
    run_sit_stand_jsonl,
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


def make_sit_stand_record(
    frame_id: int,
    *,
    hip_y: float,
    center_x: float = 0.50,
    timestamp_sec: float | None = None,
    person_id: str = "elder_001",
    track_id: int = 1,
    shoulder_offset_x: float = 0.0,
    ankle_y: float = 0.88,
    usable_for_sit_stand: bool = True,
    valid: bool = True,
    core_quality: float = 0.9,
    include_wrists: bool = True,
    support_wrist: bool = False,
) -> dict[str, object]:
    shoulder_y = hip_y - 0.22
    knee_y = hip_y + ((ankle_y - hip_y) * 0.52)
    point_values = {
        "left_shoulder": (center_x - 0.05 + shoulder_offset_x, shoulder_y),
        "right_shoulder": (center_x + 0.05 + shoulder_offset_x, shoulder_y),
        "left_hip": (center_x - 0.04, hip_y),
        "right_hip": (center_x + 0.04, hip_y),
        "left_knee": (center_x - 0.04, knee_y),
        "right_knee": (center_x + 0.04, knee_y),
        "left_ankle": (center_x - 0.04, ankle_y),
        "right_ankle": (center_x + 0.04, ankle_y),
    }

    if include_wrists:
        if support_wrist:
            point_values["left_wrist"] = (0.26, 0.66)
            point_values["right_wrist"] = (center_x + 0.09, shoulder_y + 0.14)
        else:
            point_values["left_wrist"] = (center_x - 0.09, shoulder_y + 0.14)
            point_values["right_wrist"] = (center_x + 0.09, shoulder_y + 0.14)

    keypoints = [
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
        for name, (x, y) in point_values.items()
    ]
    return {
        "frame_id": frame_id,
        "timestamp_sec": round(frame_id * 0.4 if timestamp_sec is None else timestamp_sec, 4),
        "person_id": person_id,
        "track_id": track_id,
        "scene_region": "home",
        "core_keypoint_quality": core_quality,
        "keypoints": keypoints,
        "window_quality": {
            "mean_core_keypoint_quality": core_quality,
            "low_quality_frame_ratio": 0.0 if usable_for_sit_stand else 1.0,
            "interpolated_point_ratio": 0.0,
            "jump_outlier_count": 0,
            "usable_for_gait": usable_for_sit_stand,
            "usable_for_sit_stand": usable_for_sit_stand,
            "usable_for_near_fall": usable_for_sit_stand,
        },
    }


def make_sequence(
    hip_values: list[float],
    *,
    center_x_values: list[float] | None = None,
    step_sec: float = 0.4,
    person_id: str = "elder_001",
    track_id: int = 1,
    shoulder_offsets: list[float] | None = None,
    usable_for_sit_stand: bool = True,
    valid: bool = True,
    include_wrists: bool = True,
    support_wrist: bool = False,
) -> list[dict[str, object]]:
    center_values = center_x_values or [0.50] * len(hip_values)
    offsets = shoulder_offsets or [0.0] * len(hip_values)
    return [
        make_sit_stand_record(
            frame_id=index,
            hip_y=hip_y,
            center_x=center_values[index],
            timestamp_sec=index * step_sec,
            person_id=person_id,
            track_id=track_id,
            shoulder_offset_x=offsets[index],
            usable_for_sit_stand=usable_for_sit_stand,
            valid=valid,
            core_quality=0.9 if usable_for_sit_stand else 0.2,
            include_wrists=include_wrists,
            support_wrist=support_wrist,
        )
        for index, hip_y in enumerate(hip_values)
    ]


def normal_sit_to_stand_sequence() -> list[dict[str, object]]:
    return make_sequence([0.70, 0.70, 0.68, 0.62, 0.55, 0.49, 0.45, 0.45, 0.45, 0.45])


class FallRiskSitStandTest(unittest.TestCase):
    def test_normal_sit_to_stand_has_low_risk_and_event_features(self) -> None:
        [event] = extract_sit_stand_events(normal_sit_to_stand_sequence())

        self.assertEqual(event["transition_type"], "sit_to_stand")
        self.assertLess(event["sit_stand_risk_score"], 0.25)
        self.assertGreater(event["duration"], 0.0)
        self.assertLess(event["duration"], 3.0)
        self.assertEqual(event["failed_attempts"], 0)
        self.assertFalse(event["quality_coverage"]["insufficient_sit_stand_quality"])
        self.assertIn("hip_vertical_displacement", event["sit_stand_features"])
        self.assertEqual(event["risk_factors"], [])

    def test_normal_stand_to_sit_is_not_marked_as_failed_rise(self) -> None:
        records = make_sequence([0.45, 0.45, 0.49, 0.55, 0.62, 0.68, 0.70, 0.70])

        [event] = extract_sit_stand_events(records)

        self.assertEqual(event["transition_type"], "stand_to_sit")
        self.assertEqual(event["failed_attempts"], 0)
        self.assertNotIn("疑似多次起身失败", event["risk_factors"])

    def test_long_transition_duration_increases_risk(self) -> None:
        normal_score = extract_sit_stand_events(normal_sit_to_stand_sequence())[0]["sit_stand_risk_score"]
        slow_records = make_sequence(
            [0.70, 0.70, 0.69, 0.67, 0.65, 0.62, 0.59, 0.56, 0.53, 0.50, 0.47, 0.45, 0.45],
            step_sec=0.7,
        )

        [event] = extract_sit_stand_events(slow_records)

        self.assertGreater(event["duration"], 4.0)
        self.assertGreater(event["sit_stand_risk_score"], normal_score)
        self.assertIn("起身耗时较长", event["risk_factors"])

    def test_failed_attempt_before_successful_rise_is_reported(self) -> None:
        records = make_sequence([0.70, 0.68, 0.65, 0.69, 0.70, 0.67, 0.62, 0.56, 0.49, 0.45, 0.45])

        [event] = extract_sit_stand_events(records)

        self.assertEqual(event["transition_type"], "sit_to_stand")
        self.assertGreater(event["failed_attempts"], 0)
        self.assertIn("疑似多次起身失败", event["risk_factors"])

    def test_post_stand_sway_is_reported(self) -> None:
        hip_values = [0.70, 0.70, 0.66, 0.60, 0.53, 0.47, 0.45, 0.45, 0.45, 0.45, 0.45]
        center_values = [0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.54, 0.46, 0.55, 0.45, 0.53]

        [event] = extract_sit_stand_events(make_sequence(hip_values, center_x_values=center_values))

        self.assertIsNotNone(event["post_stand_sway"])
        self.assertGreater(event["post_stand_sway"], 0.1)
        self.assertIn("起身后存在明显摇晃", event["risk_factors"])

    def test_support_usage_proxy_uses_optional_wrist_points(self) -> None:
        [event] = extract_sit_stand_events(make_sequence(
            [0.70, 0.70, 0.67, 0.61, 0.54, 0.48, 0.45, 0.45, 0.45],
            support_wrist=True,
        ))

        self.assertTrue(event["support_usage"]["suspected"])
        self.assertIn("left_wrist_stays_near_side_surface", event["support_usage"]["evidence"])
        self.assertIn("疑似借助支撑", event["risk_factors"])

    def test_long_stabilization_time_is_reported(self) -> None:
        hip_values = [0.70, 0.70, 0.66, 0.60, 0.53, 0.47, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45]
        center_values = [0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.54, 0.46, 0.55, 0.50, 0.50, 0.50]
        config = SitStandAnalysisConfig(post_stand_window_sec=3.0)

        [event] = extract_sit_stand_events(make_sequence(hip_values, center_x_values=center_values), config=config)

        self.assertIsNotNone(event["stabilization_time"])
        self.assertGreater(event["stabilization_time"], 0.8)
        self.assertIn("站稳时间较长", event["risk_factors"])

    def test_low_quality_window_is_marked_insufficient_without_high_risk(self) -> None:
        records = make_sequence(
            [0.70, 0.70, 0.67, 0.61, 0.54, 0.48, 0.45, 0.45],
            usable_for_sit_stand=False,
            valid=False,
        )

        [event] = extract_sit_stand_events(records, config=SitStandAnalysisConfig(min_event_frames=5))

        self.assertEqual(event["sit_stand_risk_score"], 0.0)
        self.assertEqual(event["transition_type"], "unknown_transition")
        self.assertTrue(event["quality_coverage"]["insufficient_sit_stand_quality"])
        self.assertIn("insufficient_sit_stand_quality", event["risk_factors"])

    def test_missing_wrists_do_not_block_core_sit_stand_metrics(self) -> None:
        [event] = extract_sit_stand_events(make_sequence(
            [0.70, 0.70, 0.67, 0.61, 0.54, 0.48, 0.45, 0.45],
            include_wrists=False,
        ))

        self.assertEqual(event["transition_type"], "sit_to_stand")
        self.assertFalse(event["support_usage"]["suspected"])
        self.assertEqual(event["support_usage"]["evidence"], [])

    def test_multiple_tracks_are_analyzed_independently(self) -> None:
        records = normal_sit_to_stand_sequence() + make_sequence(
            [0.45, 0.45, 0.49, 0.55, 0.62, 0.68, 0.70, 0.70],
            person_id="elder_002",
            track_id=2,
        )

        events = extract_sit_stand_events(records)
        keys = {(event["person_id"], event["track_id"], event["transition_type"]) for event in events}

        self.assertEqual(keys, {("elder_001", 1, "sit_to_stand"), ("elder_002", 2, "stand_to_sit")})

    def test_run_sit_stand_jsonl_reads_and_writes_event_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "poses_cleaned.jsonl"
            output_path = Path(tmpdir) / "sit_stand.jsonl"
            input_path.write_text(
                "\n".join(json.dumps(record, ensure_ascii=False) for record in normal_sit_to_stand_sequence()),
                encoding="utf-8",
            )

            count = run_sit_stand_jsonl(
                input_path=input_path,
                output_path=output_path,
                config=SitStandAnalysisConfig(min_event_frames=5),
            )
            lines = output_path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(count, 1)
        self.assertEqual(len(lines), 1)
        payload = json.loads(lines[0])
        self.assertIn("sit_stand_risk_score", payload)
        self.assertIn("sit_stand_features", payload)
        self.assertIn("quality_coverage", payload)
        self.assertNotIn("risk_level", payload)
        self.assertNotIn("recommended_action", payload)


if __name__ == "__main__":
    unittest.main()
