import unittest

from elderly_monitoring.modules.fall_risk.features import (
    feature_coverage,
    weighted_fall_risk_score,
)
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

    def test_specific_gait_explanations_are_preserved_in_fused_event(self) -> None:
        event = FallRiskPipeline().predict_from_features(
            {
                "person_id": "p01",
                "gait_risk_score": 0.72,
                "gait_risk_factors": [
                    "gait_speed_reduced",
                    "hip_lateral_sway",
                    "turn_instability",
                ],
            }
        )

        self.assertIn("gait_instability", event.risk_factors)
        self.assertIn("gait_speed_reduced", event.risk_factors)
        self.assertIn("hip_lateral_sway", event.risk_factors)
        self.assertIn("turn_instability", event.risk_factors)

    def test_fusion_mask_excludes_unavailable_values_and_renormalizes_weights(self) -> None:
        score = weighted_fall_risk_score(
            {
                "gait_risk_score": 0.6,
                "sit_stand_risk_score": 0.0,
                "near_fall_event_score": None,
                "scene_risk_score": 0.2,
                "fusion_mask": {
                    "gait_risk_score": True,
                    "sit_stand_risk_score": False,
                    "near_fall_event_score": False,
                    "baseline_deviation_score": False,
                    "scene_risk_score": True,
                    "activity_rhythm_score": False,
                },
            }
        )

        self.assertEqual(score, 0.4933)

    def test_initial_personal_baseline_weight_is_respected(self) -> None:
        full_weight = weighted_fall_risk_score(
            {
                "baseline_deviation_score": 0.8,
                "baseline_fusion_weight": 1.0,
                "fusion_mask": {"baseline_deviation_score": True},
            }
        )
        initial_weight = weighted_fall_risk_score(
            {
                "baseline_deviation_score": 0.8,
                "baseline_fusion_weight": 0.25,
                "fusion_mask": {"baseline_deviation_score": True},
            }
        )

        self.assertEqual(full_weight, 0.8)
        self.assertEqual(initial_weight, 0.8)

        mixed_full = weighted_fall_risk_score(
            {
                "gait_risk_score": 0.0,
                "baseline_deviation_score": 0.8,
                "baseline_fusion_weight": 1.0,
                "fusion_mask": {
                    "gait_risk_score": True,
                    "baseline_deviation_score": True,
                },
            }
        )
        mixed_initial = weighted_fall_risk_score(
            {
                "gait_risk_score": 0.0,
                "baseline_deviation_score": 0.8,
                "baseline_fusion_weight": 0.25,
                "fusion_mask": {
                    "gait_risk_score": True,
                    "baseline_deviation_score": True,
                },
            }
        )
        self.assertLess(mixed_initial, mixed_full)

    def test_event_metadata_preserves_epoch_and_branch_statuses(self) -> None:
        event = FallRiskPipeline().predict_from_features(
            {
                "person_id": "p01",
                "stream_epoch": 4,
                "gait_risk_score": 0.6,
                "fusion_mask": {"gait_risk_score": True},
                "branch_diagnostics": {
                    "gait": {"status": "valid"},
                    "near_fall": {"status": "unavailable"},
                },
            }
        )

        self.assertEqual(event.metadata["stream_epoch"], 4)
        self.assertEqual(event.metadata["branch_statuses"]["gait"], "valid")
        self.assertEqual(
            event.metadata["branch_statuses"]["near_fall"], "unavailable"
        )



if __name__ == "__main__":
    unittest.main()
