import json
import tempfile
import unittest
from pathlib import Path

from elderly_monitoring.modules.fall_risk.pose import (
    build_keypoints,
    build_pose_observation,
    keypoint_quality,
    write_jsonl,
)


class FallRiskPoseTest(unittest.TestCase):
    def test_build_keypoints_normalizes_coordinates(self) -> None:
        keypoints = build_keypoints(
            [[10, 20], [30, 40]],
            [0.9, 0.2],
            names=["left_hip", "right_hip"],
            normalize_by=(100, 200),
        )

        self.assertEqual(keypoints[0].name, "left_hip")
        self.assertEqual(keypoints[0].x, 0.1)
        self.assertEqual(keypoints[0].y, 0.1)
        self.assertEqual(keypoints[0].score, 0.9)
        self.assertEqual(keypoints[1].name, "right_hip")
        self.assertEqual(keypoints[1].x, 0.3)
        self.assertEqual(keypoints[1].y, 0.2)

    def test_keypoint_quality_combines_coverage_and_mean_score(self) -> None:
        keypoints = build_keypoints(
            [[0, 0], [1, 1], [2, 2]],
            [0.9, 0.6, 0.1],
            names=["nose", "left_shoulder", "left_ankle"],
        )

        self.assertEqual(keypoint_quality(keypoints), 0.6133)

    def test_build_pose_observation_outputs_expected_fields(self) -> None:
        keypoints = build_keypoints(
            [[10, 20], [30, 40]],
            [0.9, 0.7],
            names=["left_hip", "right_hip"],
        )

        observation = build_pose_observation(
            frame_id=12,
            person_id="elder_001",
            keypoints=keypoints,
            timestamp_sec=0.5,
            scene_region="living_room",
            track_id=1,
            bbox=[10, 20, 50, 80],
            pose_confidence=0.94321,
        )
        payload = observation.to_dict()

        self.assertEqual(payload["frame_id"], 12)
        self.assertEqual(payload["person_id"], "elder_001")
        self.assertEqual(payload["track_id"], 1)
        self.assertEqual(payload["bbox"], [10.0, 20.0, 50.0, 80.0])
        self.assertEqual(payload["scene_region"], "living_room")
        self.assertEqual(payload["pose_confidence"], 0.9432)
        self.assertEqual(payload["keypoint_quality"], 0.92)
        self.assertEqual(payload["keypoints"][0]["name"], "left_hip")

    def test_write_jsonl_persists_pose_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "poses.jsonl"
            count = write_jsonl(
                [
                    {"frame_id": 1, "person_id": "elder_001", "keypoints": []},
                    {"frame_id": 2, "person_id": "elder_001", "keypoints": []},
                ],
                output_path,
            )

            lines = output_path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(count, 2)
        self.assertEqual(len(lines), 2)
        self.assertEqual(json.loads(lines[0])["frame_id"], 1)
        self.assertEqual(json.loads(lines[1])["person_id"], "elder_001")


if __name__ == "__main__":
    unittest.main()
