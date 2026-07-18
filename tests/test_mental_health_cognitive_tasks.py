from __future__ import annotations

import unittest

from elderly_monitoring.modules.mental_health.feature_extraction.cognitive_tasks import (
    build_active_cognitive_task_features,
    score_active_cognitive_tasks,
)
from elderly_monitoring.modules.mental_health.pipeline import MentalHealthRiskPipeline


class MentalHealthActiveCognitiveTasksTest(unittest.TestCase):
    def complete_sample(self, **overrides: object) -> dict[str, object]:
        sample: dict[str, object] = {
            "person_id": "p01",
            "timestamp": "2026-06-26T20:00:00+08:00",
            "start_time": "2026-06-26T08:00:00+08:00",
            "end_time": "2026-06-26T20:00:00+08:00",
            "activity_drop_score": None,
            "sleep_disturbance_score": None,
            "routine_irregularity_score": None,
            "baseline_quality": 1.0,
            "persistent_abnormal_days": 3,
            "initial_baseline_ready": True,
            "stable_baseline_ready": True,
        }
        sample.update(overrides)
        return sample

    def active_task_payload(self) -> dict[str, object]:
        return {
            "cognitive_tasks": {
                "animal_fluency": {
                    "valid_animal_count": 6,
                    "repetition_count": 2,
                },
                "countdown": {
                    "error_count": 3,
                    "duration_seconds": 80,
                    "completed": True,
                },
                "recall": {
                    "expected_items": 5,
                    "recalled_items": 2,
                },
                "picture_description": {
                    "expected_information_points": 6,
                    "information_points": 2,
                    "off_topic_count": 2,
                    "response_latency_seconds": 25,
                },
            },
            "active_cognitive_task_quality": 0.9,
        }

    def test_scores_structured_active_cognitive_tasks_without_diagnostic_output(self) -> None:
        score = score_active_cognitive_tasks(self.active_task_payload())

        self.assertGreaterEqual(score.active_cognitive_task_score or 0.0, 0.75)
        self.assertEqual(score.active_cognitive_task_level, 3)
        self.assertIn("animal_fluency_low_count", score.factors)
        self.assertIn("recall_omission_increase", score.factors)
        self.assertEqual(score.confidence, 0.9)
        self.assertFalse(score.diagnosis)

    def test_build_feature_record_can_be_merged_into_daily_features(self) -> None:
        features = build_active_cognitive_task_features(self.active_task_payload())

        self.assertGreaterEqual(features["active_cognitive_task_score"], 0.75)
        self.assertEqual(features["active_cognitive_task_level"], 3)
        self.assertIn("active_cognitive_task_details", features)
        self.assertFalse(features["diagnosis"])

    def test_pipeline_derives_active_task_score_for_cognitive_submodule_and_strong_rule(self) -> None:
        event = MentalHealthRiskPipeline().predict_from_features(
            self.complete_sample(
                movement_vitality_score=0.7,
                **self.active_task_payload(),
            )
        )

        active_detail = event.metadata["active_cognitive_task_details"]
        cognitive = event.metadata["submodules"]["cognitive_change_clue"]
        self.assertGreaterEqual(active_detail["active_cognitive_task_score"], 0.75)
        self.assertGreaterEqual(cognitive["feature_scores"]["active_cognitive_task_score"], 0.75)
        self.assertIn("active_cognitive_task_clue", cognitive["factors"])
        self.assertEqual(event.risk_level, 3)
        self.assertEqual(
            event.metadata["strong_rule_matches"][0]["rule_id"],
            "active_cognitive_motor_decline",
        )
        self.assertIn("active_cognitive_tasks", event.metadata["auxiliary_models"])

    def test_absent_active_tasks_do_not_change_existing_pipeline_surface(self) -> None:
        event = MentalHealthRiskPipeline().predict_from_features(
            self.complete_sample(movement_vitality_score=0.3)
        )

        self.assertIsNone(event.metadata["active_cognitive_task_score"])
        self.assertEqual(
            event.metadata["active_cognitive_task_details"]["factors"],
            ("active_cognitive_task_insufficient_data",),
        )
        self.assertNotIn("active_cognitive_task_score", event.metadata["available_modalities"])

    def test_invalid_active_task_fields_fail_with_context(self) -> None:
        with self.assertRaisesRegex(ValueError, "animal_fluency_count"):
            score_active_cognitive_tasks({"animal_fluency_count": -1})


if __name__ == "__main__":
    unittest.main()
