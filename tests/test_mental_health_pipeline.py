from __future__ import annotations

import copy
import unittest

from elderly_monitoring.common.config import load_yaml

from elderly_monitoring.modules.mental_health import MentalHealthRiskPipeline
from elderly_monitoring.modules.mental_health.config import (
    DEFAULT_CONFIG_PATH,
    load_mental_health_config,
    mental_health_config_from_mapping,
)


class MentalHealthRiskPipelineTest(unittest.TestCase):
    def complete_sample(self, **overrides: object) -> dict[str, object]:
        sample: dict[str, object] = {
            "person_id": "p01",
            "timestamp": "2026-06-26T20:00:00+08:00",
            "start_time": "2026-06-26T08:00:00+08:00",
            "end_time": "2026-06-26T20:00:00+08:00",
            "activity_drop_score": 0.9,
            "sleep_disturbance_score": 0.9,
            "routine_irregularity_score": 0.9,
            "baseline_quality": 1.0,
            "persistent_abnormal_days": 3,
            "initial_baseline_ready": True,
            "stable_baseline_ready": True,
            "baseline_window": {
                "start_date": "2026-06-19",
                "end_date": "2026-06-25",
                "eligible_history_days": 7,
            },
            "risk_factor_details": {},
        }
        sample.update(overrides)
        return sample

    def test_default_configuration_is_complete_and_validated(self) -> None:
        config = load_mental_health_config()

        self.assertEqual(config.baseline.initial_days, 3)
        self.assertEqual(config.baseline.stable_days, 7)
        self.assertEqual(config.baseline.max_window_days, 14)
        self.assertEqual(config.scoring.weights["activity_drop_score"], 0.24)
        self.assertEqual(config.scoring.thresholds, (0.25, 0.45, 0.65))
        self.assertEqual(config.scoring.coverage_expected_features, (
            "activity_drop_score",
            "sleep_disturbance_score",
            "routine_irregularity_score",
        ))

    def test_configuration_rejects_missing_invalid_and_non_increasing_values(self) -> None:
        valid = load_yaml(DEFAULT_CONFIG_PATH)
        invalid_cases: list[tuple[str, dict[str, object], str]] = []

        missing_weights = copy.deepcopy(valid)
        del missing_weights["scoring"]["weights"]
        invalid_cases.append(("missing weights", missing_weights, r"scoring\.weights"))

        nonpositive_weight = copy.deepcopy(valid)
        nonpositive_weight["scoring"]["weights"]["activity_drop_score"] = 0
        invalid_cases.append(("weight", nonpositive_weight, r"scoring\.weights\.activity_drop_score"))

        unordered_thresholds = copy.deepcopy(valid)
        unordered_thresholds["scoring"]["thresholds"]["level_2"] = 0.2
        invalid_cases.append(("thresholds", unordered_thresholds, r"scoring\.thresholds"))

        confidence_sum = copy.deepcopy(valid)
        confidence_sum["scoring"]["confidence"]["feature_coverage_weight"] = 0.5
        invalid_cases.append(("confidence", confidence_sum, r"scoring\.confidence"))

        invalid_days = copy.deepcopy(valid)
        invalid_days["baseline"]["initial_days"] = 8
        invalid_cases.append(("days", invalid_days, r"baseline.*days"))

        for label, values, pattern in invalid_cases:
            with self.subTest(label=label):
                with self.assertRaisesRegex(ValueError, pattern):
                    mental_health_config_from_mapping(values)

    def test_custom_yaml_weights_change_the_normalized_score(self) -> None:
        values = load_yaml(DEFAULT_CONFIG_PATH)
        values["scoring"]["weights"].update(
            {
                "activity_drop_score": 0.60,
                "sleep_disturbance_score": 0.10,
                "routine_irregularity_score": 0.10,
            }
        )
        pipeline = MentalHealthRiskPipeline(mental_health_config_from_mapping(values))

        event = pipeline.predict_from_features(
            self.complete_sample(
                activity_drop_score=1.0,
                sleep_disturbance_score=0.0,
                routine_irregularity_score=0.0,
            )
        )

        self.assertEqual(event.risk_score, 0.75)
        self.assertEqual(event.risk_level, 3)

    def test_behavioral_deviation_requires_manual_review_for_high_risk(self) -> None:
        event = MentalHealthRiskPipeline().predict_from_features(
            self.complete_sample()
        )

        self.assertEqual(event.module, "mental_health")
        self.assertEqual(event.risk_level, 3)
        self.assertEqual(event.recommended_action, "manual_review")
        self.assertIs(event.metadata["diagnosis"], False)
        self.assertIsNotNone(event.evidence_window)

    def test_missing_optional_features_do_not_enter_coverage_denominator(self) -> None:
        event = MentalHealthRiskPipeline().predict_from_features(
            self.complete_sample(
                activity_drop_score=0.8,
                sleep_disturbance_score=None,
                routine_irregularity_score=None,
            )
        )

        self.assertEqual(event.risk_score, 0.8)
        self.assertAlmostEqual(event.metadata["feature_coverage"], 0.24 / 0.62, places=4)
        self.assertEqual(event.risk_level, 1)
        self.assertEqual(event.metadata["available_modalities"], ["activity_drop_score"])
        self.assertIn("social_withdrawal_score", event.metadata["missing_modalities"])

    def test_no_scoreable_feature_is_insufficient_data_not_normal(self) -> None:
        event = MentalHealthRiskPipeline().predict_from_features(
            self.complete_sample(
                activity_drop_score=None,
                sleep_disturbance_score=None,
                routine_irregularity_score=None,
                baseline_quality=0.0,
                persistent_abnormal_days=0,
                initial_baseline_ready=False,
                stable_baseline_ready=False,
            )
        )

        self.assertEqual(event.risk_score, 0.0)
        self.assertEqual(event.risk_level, 0)
        self.assertEqual(event.confidence, 0.0)
        self.assertEqual(event.trigger_event, "insufficient_data")
        self.assertEqual(event.metadata["score_status"], "unavailable")
        self.assertIn("insufficient_data", event.risk_factors)

    def test_passive_features_cannot_trigger_level_four(self) -> None:
        event = MentalHealthRiskPipeline().predict_from_features(self.complete_sample())

        self.assertEqual(event.risk_score, 0.9)
        self.assertEqual(event.risk_level, 3)

    def test_baseline_and_persistence_caps_apply_in_fixed_order(self) -> None:
        cases = (
            (
                "initial",
                {"initial_baseline_ready": False, "stable_baseline_ready": False},
                1,
                "initial_baseline_not_ready",
            ),
            (
                "stable",
                {"initial_baseline_ready": True, "stable_baseline_ready": False},
                2,
                "stable_baseline_not_ready",
            ),
            (
                "persistence",
                {"persistent_abnormal_days": 2},
                2,
                "persistent_days_below_minimum",
            ),
        )
        for label, overrides, expected_level, expected_reason in cases:
            with self.subTest(label=label):
                event = MentalHealthRiskPipeline().predict_from_features(
                    self.complete_sample(**overrides)
                )
                self.assertEqual(event.risk_level, expected_level)
                self.assertIn(
                    expected_reason,
                    [item["reason"] for item in event.metadata["applied_level_caps"]],
                )

    def test_emergency_self_report_bypasses_passive_caps_but_not_confidence_formula(self) -> None:
        event = MentalHealthRiskPipeline().predict_from_features(
            self.complete_sample(
                activity_drop_score=None,
                sleep_disturbance_score=None,
                routine_irregularity_score=None,
                self_report_risk_score=0.9,
                baseline_quality=0.0,
                persistent_abnormal_days=0,
                initial_baseline_ready=False,
                stable_baseline_ready=False,
            )
        )

        self.assertEqual(event.risk_level, 4)
        self.assertEqual(event.recommended_action, "manual_review")
        self.assertEqual(event.confidence, 0.0)
        self.assertEqual(event.metadata["strong_evidence_source"], "self_report_risk_score")
        self.assertTrue(event.metadata["applied_level_caps"])

    def test_explicit_manual_emergency_flag_can_trigger_level_four_without_scores(self) -> None:
        event = MentalHealthRiskPipeline().predict_from_features(
            self.complete_sample(
                activity_drop_score=None,
                sleep_disturbance_score=None,
                routine_irregularity_score=None,
                manual_emergency_flag=True,
                baseline_quality=0.0,
                persistent_abnormal_days=0,
                initial_baseline_ready=False,
                stable_baseline_ready=False,
            )
        )

        self.assertEqual(event.risk_level, 4)
        self.assertEqual(event.risk_score, 0.0)
        self.assertEqual(event.metadata["score_status"], "strong_evidence_override")
        self.assertEqual(event.metadata["strong_evidence_source"], "manual_emergency_flag")

    def test_invalid_scores_and_manual_flag_fail_with_field_context(self) -> None:
        for field, value in (
            ("self_report_risk_score", 1.1),
            ("negative_affect_score", "0.8"),
            ("manual_emergency_flag", 1),
            ("persistent_abnormal_days", -1),
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, field):
                    MentalHealthRiskPipeline().predict_from_features(
                        self.complete_sample(**{field: value})
                    )

    def test_confidence_uses_fixed_coverage_baseline_and_persistence_formula(self) -> None:
        event = MentalHealthRiskPipeline().predict_from_features(
            self.complete_sample(
                activity_drop_score=0.5,
                sleep_disturbance_score=0.5,
                routine_irregularity_score=None,
                baseline_quality=0.5,
                persistent_abnormal_days=1,
            )
        )

        coverage = (0.24 + 0.22) / 0.62
        expected = 0.45 * coverage + 0.35 * 0.5 + 0.20 * (1 / 3)
        self.assertAlmostEqual(event.confidence, expected, places=4)


if __name__ == "__main__":
    unittest.main()
