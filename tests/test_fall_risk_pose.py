import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from elderly_monitoring.modules.fall_risk.pose import (
    COCO_KEYPOINT_NAMES,
    build_rtmpose_observations,
    build_keypoints,
    build_pose_observation,
    keypoint_quality,
    run_rtmpose_pose,
    write_jsonl,
)
from scripts.collect.run_fall_pose import build_parser


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

    def test_cli_default_backend_remains_yolov8_pose(self) -> None:
        parser = build_parser()

        args = parser.parse_args(["--input", "input.mp4", "--output", "poses.jsonl"])

        self.assertEqual(args.backend, "yolov8-pose")

    def test_cli_accepts_rtmpose_backend_options(self) -> None:
        parser = build_parser()

        args = parser.parse_args(
            [
                "--input",
                "input.mp4",
                "--output",
                "poses.jsonl",
                "--backend",
                "rtmpose",
                "--pose-config",
                "rtmpose.py",
                "--pose-checkpoint",
                "rtmpose.pth",
                "--device",
                "cpu",
            ]
        )

        self.assertEqual(args.backend, "rtmpose")
        self.assertEqual(args.pose_config, "rtmpose.py")
        self.assertEqual(args.pose_checkpoint, "rtmpose.pth")
        self.assertEqual(args.device, "cpu")

    def test_rtmpose_missing_dependency_raises_clear_runtime_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("importlib.import_module", side_effect=ImportError("missing mmpose")):
                with self.assertRaisesRegex(RuntimeError, "RTMPose.*MMPose.*MMCV.*MMEngine"):
                    run_rtmpose_pose(
                        video_path=Path("missing.mp4"),
                        output_path=Path(tmpdir) / "poses.jsonl",
                    )

    def test_build_rtmpose_observations_adapts_mock_coco17_prediction(self) -> None:
        points_xy = [[20 + index, 40 + index] for index in range(len(COCO_KEYPOINT_NAMES))]
        scores = [0.90 - (index * 0.01) for index in range(len(COCO_KEYPOINT_NAMES))]

        observations = build_rtmpose_observations(
            [
                {
                    "keypoints": points_xy,
                    "keypoint_scores": scores,
                    "bbox": [[10, 20, 110, 220]],
                    "bbox_score": [0.88],
                }
            ],
            frame_id=8,
            timestamp_sec=0.4,
            frame_size=(200, 400),
            scene_region="home",
            person_id_prefix="elder",
            normalize_coordinates=True,
        )

        self.assertEqual(len(observations), 1)
        payload = observations[0].to_dict()

        self.assertEqual(payload["frame_id"], 8)
        self.assertEqual(payload["person_id"], "elder_001")
        self.assertEqual(payload["track_id"], 1)
        self.assertEqual(payload["bbox"], [10.0, 20.0, 110.0, 220.0])
        self.assertEqual(payload["scene_region"], "home")
        self.assertEqual(payload["pose_confidence"], 0.88)
        self.assertEqual(payload["timestamp_sec"], 0.4)
        self.assertEqual(len(payload["keypoints"]), 17)
        self.assertEqual(payload["keypoints"][0]["name"], "nose")
        self.assertEqual(payload["keypoints"][-1]["name"], "right_ankle")
        self.assertEqual(payload["keypoints"][0]["x"], 0.1)
        self.assertEqual(payload["keypoints"][0]["y"], 0.1)

    def test_build_rtmpose_observations_adapts_batched_pred_instances(self) -> None:
        first_points = [[10 + index, 20 + index] for index in range(len(COCO_KEYPOINT_NAMES))]
        second_points = [[30 + index, 60 + index] for index in range(len(COCO_KEYPOINT_NAMES))]
        scores = [[0.8] * len(COCO_KEYPOINT_NAMES), [0.7] * len(COCO_KEYPOINT_NAMES)]

        observations = build_rtmpose_observations(
            {
                "pred_instances": {
                    "keypoints": [first_points, second_points],
                    "keypoint_scores": scores,
                    "bboxes": [[1, 2, 3, 4], [5, 6, 7, 8]],
                    "bbox_scores": [0.91, 0.82],
                    "track_ids": [10, 11],
                }
            },
            frame_id=3,
            timestamp_sec=0.15,
            frame_size=(100, 200),
            normalize_coordinates=False,
        )

        self.assertEqual(len(observations), 2)
        first_payload = observations[0].to_dict()
        second_payload = observations[1].to_dict()
        self.assertEqual(first_payload["person_id"], "elder_010")
        self.assertEqual(first_payload["track_id"], 10)
        self.assertEqual(first_payload["pose_confidence"], 0.91)
        self.assertEqual(second_payload["person_id"], "elder_011")
        self.assertEqual(second_payload["track_id"], 11)
        self.assertEqual(second_payload["bbox"], [5.0, 6.0, 7.0, 8.0])
        self.assertEqual(second_payload["keypoints"][0]["x"], 30.0)


if __name__ == "__main__":
    unittest.main()
