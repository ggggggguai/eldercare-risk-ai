import unittest

from elderly_monitoring.modules.fall_risk.features import feature_coverage
from elderly_monitoring.modules.fall_risk import FallRiskPipeline


class FallRiskPipelineTest(unittest.TestCase):
    def test_near_fall_produces_high_risk_event(self) -> None:
        event = FallRiskPipeline().predict_from_features(
            {
                "device_id": "cam_001",
                "person_id": "p01",
                "timestamp": "2026-06-26T14:20:00+08:00",
                "scene_region": "bedroom_bedside",
                "gait_risk_score": 0.6,
                "sit_stand_risk_score": 0.3,
                "near_fall_event_score": 0.8,
                "baseline_deviation_score": 0.5,
                "scene_risk_score": 0.4,
                "activity_rhythm_score": 0.2,
            }
        )

        self.assertEqual(event.module, "fall_risk")
        self.assertEqual(event.risk_level, 3)
        self.assertIsInstance(event.risk_level, int)
        self.assertGreaterEqual(event.risk_score, 0.0)
        self.assertLessEqual(event.risk_score, 1.0)
        self.assertEqual(event.recommended_action, "notify_guardian")
        self.assertIn("near_fall_event", event.risk_factors)
        self.assertEqual(event.model_version, "fall-risk-v0.1")

    def test_fall_or_long_static_produces_emergency_warning_with_explanation(self) -> None:
        event = FallRiskPipeline().predict_from_features(
            {
                "device_id": "cam_001",
                "person_id": "p01",
                "timestamp": "2026-06-26T14:21:00+08:00",
                "scene_region": "living_room",
                "fall_event_score": 0.91,
                "long_static_score": 0.82,
                "keypoint_quality": 0.72,
            }
        )

        self.assertEqual(event.risk_level, 4)
        self.assertEqual(event.trigger_event, "fall_or_long_static")
        self.assertEqual(event.recommended_action, "emergency_alert")
        self.assertIn("suspected_fall_event", event.risk_factors)
        self.assertIn("long_static_after_fall_risk", event.risk_factors)
        self.assertGreaterEqual(event.confidence, 0.0)
        self.assertLessEqual(event.confidence, 1.0)

    def test_feature_coverage_reflects_available_baseline_inputs(self) -> None:
        self.assertEqual(
            feature_coverage(
                {
                    "gait_risk_score": 0.2,
                    "near_fall_event_score": 0.0,
                    "scene_risk_score": 0.1,
                }
            ),
            0.5,
        )


if __name__ == "__main__":
    unittest.main()
