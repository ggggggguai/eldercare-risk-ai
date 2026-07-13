import unittest

from elderly_monitoring.runtime.feature_assembly import FeatureAssembler, FeatureAssemblyConfig


def _record(frame_id: int, timestamp: float, quality: float = 0.9):
    names = ["left_shoulder", "right_shoulder", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
    return {
        "frame_id": frame_id,
        "person_id": "elder-1",
        "track_id": 1,
        "scene_region": "bathroom",
        "timestamp_sec": timestamp,
        "bbox": [0.2, 0.1, 0.8, 0.9],
        "keypoint_quality": quality,
        "keypoints": [
            {"name": name, "x": 0.4 + (index % 2) * 0.1, "y": 0.2 + index * 0.08, "score": quality}
            for index, name in enumerate(names)
        ],
    }


class FeatureAssemblyTest(unittest.TestCase):
    def test_window_prunes_old_records_and_respects_interval(self) -> None:
        assembler = FeatureAssembler(
            person_id="elder-1",
            scene_region="bathroom",
            scene_risk_scores={"bathroom": 0.7},
            config=FeatureAssemblyConfig(window_sec=10.0, analysis_interval_sec=0.5),
        )
        self.assertIsNotNone(assembler.add_pose(_record(1, 0.0), monotonic_sec=0.0))
        self.assertIsNone(assembler.add_pose(_record(2, 0.2), monotonic_sec=0.2))
        snapshot = assembler.add_pose(_record(3, 0.6), monotonic_sec=0.6)
        self.assertIsNotNone(snapshot)
        assembler.add_pose(_record(4, 11.0), monotonic_sec=11.0)
        self.assertTrue(all(record["timestamp_sec"] >= 1.0 for record in assembler.records))

    def test_outputs_scene_and_insufficient_baseline_mark(self) -> None:
        assembler = FeatureAssembler(
            person_id="elder-1",
            scene_region="bathroom",
            scene_risk_scores={"bathroom": 0.7},
            config=FeatureAssemblyConfig(analysis_interval_sec=0.0),
        )
        snapshot = assembler.add_pose(_record(1, 0.0), monotonic_sec=0.0)
        self.assertEqual(snapshot.features["scene_risk_score"], 0.7)
        self.assertEqual(snapshot.features["baseline_deviation_score"], 0.0)
        self.assertLess(snapshot.features["feature_coverage"], 1.0)
        self.assertIn("insufficient_baseline_history", snapshot.quality_flags)

    def test_reset_clears_window(self) -> None:
        assembler = FeatureAssembler(person_id="elder-1", scene_region="home", scene_risk_scores={})
        assembler.add_pose(_record(1, 0.0), monotonic_sec=0.0)
        assembler.reset()
        self.assertEqual(list(assembler.records), [])


if __name__ == "__main__":
    unittest.main()
