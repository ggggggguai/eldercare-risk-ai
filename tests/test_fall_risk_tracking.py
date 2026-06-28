import json
import tempfile
import unittest
from pathlib import Path

from elderly_monitoring.modules.fall_risk.tracking import build_observation, write_jsonl


class FallRiskTrackingTest(unittest.TestCase):
    def test_build_observation_outputs_expected_tracking_fields(self) -> None:
        observation = build_observation(
            frame_id=12,
            track_id=1,
            bbox=[10, 20, 50, 80],
            confidence=0.94321,
            scene_region="living_room",
            person_id_prefix="elder",
            fps=24.0,
            previous_center=[25.0, 45.0],
        )

        self.assertEqual(observation.frame_id, 12)
        self.assertEqual(observation.person_id, "elder_001")
        self.assertEqual(observation.track_id, 1)
        self.assertEqual(observation.bbox, [10.0, 20.0, 50.0, 80.0])
        self.assertEqual(observation.scene_region, "living_room")
        self.assertEqual(observation.track_confidence, 0.9432)
        self.assertEqual(observation.center, [30.0, 50.0])
        self.assertEqual(observation.speed_px_per_sec, 169.706)
        self.assertEqual(observation.timestamp_sec, 0.5)

    def test_write_jsonl_persists_one_record_per_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "tracks.jsonl"
            count = write_jsonl(
                [
                    {"frame_id": 1, "person_id": "elder_001"},
                    {"frame_id": 2, "person_id": "elder_001"},
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
