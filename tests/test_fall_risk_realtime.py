import unittest

from elderly_monitoring.runtime.feature_assembly import FeatureSnapshot
from elderly_monitoring.runtime.realtime_fall_risk import RealtimeFallRiskEngine


class _Assembler:
    def __init__(self, snapshot):
        self.snapshot = snapshot
        self.reset_called = False

    def add_pose(self, record, monotonic_sec):
        return self.snapshot

    def reset(self):
        self.reset_called = True


class RealtimeEngineTest(unittest.TestCase):
    def test_low_quality_does_not_fuse(self) -> None:
        assembler = _Assembler(FeatureSnapshot(features={}, quality_flags=["insufficient_pose_quality"], usable=False))
        engine = RealtimeFallRiskEngine(assembler=assembler, fusion_interval_sec=0.0)
        self.assertIsNone(engine.process_pose({}, monotonic_sec=1.0))

    def test_valid_features_generate_algorithm_event(self) -> None:
        features = {
            "person_id": "elder-1", "device_id": "cam-1", "scene_region": "home",
            "timestamp": "2026-07-12T12:00:00+08:00", "keypoint_quality": 0.9,
            "feature_coverage": 1.0, "gait_risk_score": 0.0, "sit_stand_risk_score": 0.0,
            "near_fall_event_score": 0.9, "baseline_deviation_score": 0.0,
            "scene_risk_score": 0.0, "activity_rhythm_score": 0.0,
            "fall_event_score": 0.0, "long_static_score": 0.0,
        }
        assembler = _Assembler(FeatureSnapshot(features=features, quality_flags=[], usable=True, urgent=True))
        engine = RealtimeFallRiskEngine(assembler=assembler, fusion_interval_sec=100.0)
        event = engine.process_pose({}, monotonic_sec=1.0)
        self.assertEqual(event.risk_level, 3)

    def test_window_reset_is_forwarded(self) -> None:
        assembler = _Assembler(None)
        engine = RealtimeFallRiskEngine(assembler=assembler)
        engine.reset_window()
        self.assertTrue(assembler.reset_called)


if __name__ == "__main__":
    unittest.main()
