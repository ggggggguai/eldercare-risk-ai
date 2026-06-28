import json
import tempfile
import unittest
from pathlib import Path

from elderly_monitoring.modules.fall_risk.pose_quality import (
    CORE_KEYPOINT_NAMES,
    PoseQualityConfig,
    process_pose_records,
    run_pose_quality_jsonl,
)


BASE_POINTS = {
    "left_shoulder": (0.40, 0.20),
    "right_shoulder": (0.60, 0.20),
    "left_hip": (0.42, 0.50),
    "right_hip": (0.58, 0.50),
    "left_knee": (0.43, 0.70),
    "right_knee": (0.57, 0.70),
    "left_ankle": (0.44, 0.90),
    "right_ankle": (0.56, 0.90),
}


def make_pose_record(
    frame_id: int,
    *,
    person_id: str = "elder_001",
    track_id: int = 1,
    timestamp_sec: float | None = None,
    overrides: dict[str, tuple[float, float, float]] | None = None,
    omitted: set[str] | None = None,
    default_score: float = 0.90,
) -> dict[str, object]:
    overrides = overrides or {}
    omitted = omitted or set()
    keypoints = []
    for name in CORE_KEYPOINT_NAMES:
        if name in omitted:
            continue
        x, y = BASE_POINTS[name]
        score = default_score
        if name in overrides:
            x, y, score = overrides[name]
        keypoints.append({"name": name, "x": x, "y": y, "score": score})
    return {
        "frame_id": frame_id,
        "timestamp_sec": frame_id * 0.1 if timestamp_sec is None else timestamp_sec,
        "person_id": person_id,
        "track_id": track_id,
        "scene_region": "home",
        "pose_confidence": 0.9,
        "keypoint_quality": 0.9,
        "keypoints": keypoints,
    }


def keypoint(record: dict[str, object], name: str) -> dict[str, object]:
    for item in record["keypoints"]:
        if item["name"] == name:
            return item
    raise AssertionError(f"missing keypoint {name}")


class FallRiskPoseQualityTest(unittest.TestCase):
    def test_low_confidence_core_keypoint_is_marked_invalid(self) -> None:
        records = [
            make_pose_record(
                0,
                overrides={"left_hip": (0.42, 0.50, 0.10)},
            )
        ]

        [cleaned] = process_pose_records(records)
        left_hip = keypoint(cleaned, "left_hip")

        self.assertFalse(left_hip["valid"])
        self.assertEqual(left_hip["source"], "low_confidence")
        self.assertIn("left_hip", cleaned["missing_core_names"])

    def test_core_quality_decreases_when_core_points_are_missing(self) -> None:
        good, bad = process_pose_records(
            [
                make_pose_record(0),
                make_pose_record(
                    1,
                    overrides={
                        "left_hip": (0.42, 0.50, 0.10),
                        "right_hip": (0.58, 0.50, 0.10),
                        "left_knee": (0.43, 0.70, 0.10),
                        "right_knee": (0.57, 0.70, 0.10),
                    },
                ),
            ]
        )

        self.assertGreater(good["core_keypoint_quality"], bad["core_keypoint_quality"])
        self.assertEqual(good["valid_core_count"], 8)
        self.assertEqual(bad["valid_core_count"], 4)

    def test_consecutive_low_quality_frames_become_low_quality_run(self) -> None:
        records = [
            make_pose_record(frame_id, default_score=0.10, overrides={"left_hip": (0.42, 0.50, 0.35)})
            for frame_id in range(3)
        ]

        cleaned = process_pose_records(records)

        self.assertEqual(cleaned[0]["quality_state"], "low_quality")
        self.assertEqual(cleaned[1]["quality_state"], "low_quality")
        self.assertEqual(cleaned[2]["quality_state"], "low_quality_run")
        self.assertEqual(cleaned[2]["low_quality_run_length"], 3)

    def test_short_missing_gap_is_linearly_interpolated(self) -> None:
        records = [
            make_pose_record(0, overrides={"left_ankle": (0.10, 0.90, 0.90)}),
            make_pose_record(1, omitted={"left_ankle"}),
            make_pose_record(2, omitted={"left_ankle"}),
            make_pose_record(3, overrides={"left_ankle": (0.40, 0.90, 0.90)}),
        ]

        cleaned = process_pose_records(records, config=PoseQualityConfig(jump_threshold_norm=1.0))
        frame_1_ankle = keypoint(cleaned[1], "left_ankle")
        frame_2_ankle = keypoint(cleaned[2], "left_ankle")

        self.assertTrue(frame_1_ankle["valid"])
        self.assertTrue(frame_2_ankle["valid"])
        self.assertEqual(frame_1_ankle["source"], "interpolated")
        self.assertEqual(frame_2_ankle["source"], "interpolated")
        self.assertAlmostEqual(frame_1_ankle["x"], 0.20)
        self.assertAlmostEqual(frame_2_ankle["x"], 0.30)

    def test_long_missing_gap_is_not_interpolated(self) -> None:
        records = [
            make_pose_record(0, overrides={"left_ankle": (0.10, 0.90, 0.90)}),
            make_pose_record(1, omitted={"left_ankle"}),
            make_pose_record(2, omitted={"left_ankle"}),
            make_pose_record(3, omitted={"left_ankle"}),
            make_pose_record(4, overrides={"left_ankle": (0.50, 0.90, 0.90)}),
        ]

        cleaned = process_pose_records(records)

        self.assertFalse(keypoint(cleaned[1], "left_ankle")["valid"])
        self.assertFalse(keypoint(cleaned[2], "left_ankle")["valid"])
        self.assertFalse(keypoint(cleaned[3], "left_ankle")["valid"])
        self.assertEqual(keypoint(cleaned[2], "left_ankle")["source"], "missing")

    def test_exponential_smoothing_reduces_jitter(self) -> None:
        records = [
            make_pose_record(0, overrides={"left_hip": (0.10, 0.50, 0.90)}),
            make_pose_record(1, overrides={"left_hip": (0.50, 0.50, 0.90)}),
            make_pose_record(2, overrides={"left_hip": (0.10, 0.50, 0.90)}),
        ]

        cleaned = process_pose_records(records, config=PoseQualityConfig(alpha=0.4, jump_threshold_norm=1.0))
        raw_delta = abs(keypoint(cleaned[1], "left_hip")["x"] - keypoint(cleaned[0], "left_hip")["x"])
        smooth_delta = abs(
            keypoint(cleaned[1], "left_hip")["x_smooth"] - keypoint(cleaned[0], "left_hip")["x_smooth"]
        )

        self.assertLess(smooth_delta, raw_delta)

    def test_smoothing_does_not_overwrite_original_coordinates(self) -> None:
        records = [
            make_pose_record(0, overrides={"left_hip": (0.10, 0.50, 0.90)}),
            make_pose_record(1, overrides={"left_hip": (0.50, 0.50, 0.90)}),
        ]

        cleaned = process_pose_records(records, config=PoseQualityConfig(alpha=0.4, jump_threshold_norm=1.0))
        left_hip = keypoint(cleaned[1], "left_hip")

        self.assertEqual(left_hip["x"], 0.50)
        self.assertNotEqual(left_hip["x"], left_hip["x_smooth"])

    def test_jump_outlier_is_flagged_without_deleting_point(self) -> None:
        records = [
            make_pose_record(0, overrides={"left_hip": (0.10, 0.50, 0.90)}),
            make_pose_record(1, overrides={"left_hip": (0.55, 0.50, 0.90)}),
        ]

        cleaned = process_pose_records(records, config=PoseQualityConfig(jump_threshold_norm=0.18))
        left_hip = keypoint(cleaned[1], "left_hip")

        self.assertTrue(left_hip["is_jump_outlier"])
        self.assertTrue(left_hip["valid"])

    def test_window_quality_summary_reports_ratios_and_usability(self) -> None:
        records = [
            make_pose_record(0),
            make_pose_record(1, omitted={"left_ankle"}),
            make_pose_record(2, overrides={"left_hip": (0.90, 0.50, 0.90)}),
            make_pose_record(3, default_score=0.10, overrides={"left_hip": (0.42, 0.50, 0.35)}),
        ]

        cleaned = process_pose_records(
            records,
            config=PoseQualityConfig(window_sec=1.0, max_interp_gap_frames=1, jump_threshold_norm=0.18),
        )
        summary = cleaned[0]["window_quality"]

        self.assertAlmostEqual(summary["low_quality_frame_ratio"], 0.25)
        self.assertGreater(summary["interpolated_point_ratio"], 0.0)
        self.assertGreaterEqual(summary["jump_outlier_count"], 1)
        self.assertIn("usable_for_gait", summary)
        self.assertIn("usable_for_sit_stand", summary)
        self.assertIn("usable_for_near_fall", summary)

    def test_multiple_tracks_do_not_share_interpolation_state(self) -> None:
        records = [
            make_pose_record(0, person_id="elder_a", track_id=1, overrides={"left_ankle": (0.10, 0.90, 0.90)}),
            make_pose_record(0, person_id="elder_b", track_id=2, overrides={"left_ankle": (0.80, 0.90, 0.90)}),
            make_pose_record(1, person_id="elder_a", track_id=1, omitted={"left_ankle"}),
            make_pose_record(1, person_id="elder_b", track_id=2, omitted={"left_ankle"}),
            make_pose_record(2, person_id="elder_a", track_id=1, overrides={"left_ankle": (0.30, 0.90, 0.90)}),
            make_pose_record(2, person_id="elder_b", track_id=2, overrides={"left_ankle": (0.60, 0.90, 0.90)}),
        ]

        cleaned = process_pose_records(records, config=PoseQualityConfig(jump_threshold_norm=1.0))
        by_person_frame = {(record["person_id"], record["frame_id"]): record for record in cleaned}

        self.assertAlmostEqual(keypoint(by_person_frame[("elder_a", 1)], "left_ankle")["x"], 0.20)
        self.assertAlmostEqual(keypoint(by_person_frame[("elder_b", 1)], "left_ankle")["x"], 0.70)

    def test_run_pose_quality_jsonl_reads_and_writes_cleaned_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "poses.jsonl"
            output_path = Path(tmpdir) / "poses_cleaned.jsonl"
            input_path.write_text(
                "\n".join(json.dumps(record, ensure_ascii=False) for record in [make_pose_record(0), make_pose_record(1)]),
                encoding="utf-8",
            )

            count = run_pose_quality_jsonl(input_path=input_path, output_path=output_path)
            lines = output_path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(count, 2)
        self.assertEqual(len(lines), 2)
        self.assertIn("core_keypoint_quality", json.loads(lines[0]))


if __name__ == "__main__":
    unittest.main()
