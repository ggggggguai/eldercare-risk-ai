import unittest

from elderly_monitoring.modules.mental_health import MentalHealthRiskPipeline


class MentalHealthRiskPipelineTest(unittest.TestCase):
    def test_behavioral_deviation_requires_manual_review_for_high_risk(self) -> None:
        event = MentalHealthRiskPipeline().predict_from_features(
            {
                "person_id": "p01",
                "timestamp": "2026-06-26",
                "activity_drop_score": 0.9,
                "sleep_disturbance_score": 0.9,
                "social_withdrawal_score": 0.8,
                "routine_irregularity_score": 0.8,
                "negative_affect_score": 0.7,
                "self_report_risk_score": 0.6,
                "feature_coverage": 0.8,
                "baseline_quality": 0.8,
            }
        )

        self.assertEqual(event.module, "mental_health")
        self.assertEqual(event.risk_level, 3)
        self.assertEqual(event.recommended_action, "manual_review")
        self.assertIs(event.metadata["diagnosis"], False)


if __name__ == "__main__":
    unittest.main()
