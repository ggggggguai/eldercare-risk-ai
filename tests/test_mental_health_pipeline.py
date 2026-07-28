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
        self.assertEqual(config.scoring.weights["activity_drop_score"], 0.22)
        self.assertEqual(config.scoring.weights["night_physiology_score"], 0.10)
        self.assertEqual(config.scoring.weights["movement_vitality_score"], 0.20)
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

    def test_custom_yaml_weights_change_the_submodule_fused_score(self) -> None:
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
        self.assertEqual(event.metadata["score_source"], "submodule_fusion_v2")
        self.assertEqual(
            event.metadata["submodules"]["mood_social_withdrawal"]["score"],
            75,
        )
        self.assertEqual(
            event.metadata["submodules"]["cognitive_change_clue"]["score"],
            0,
        )
        self.assertEqual(
            event.metadata["submodules"]["cognitive_change_clue"]["available_features"],
            [],
        )

    def test_behavioral_deviation_requires_manual_review_for_high_risk(self) -> None:
        event = MentalHealthRiskPipeline().predict_from_features(
            self.complete_sample()
        )

        self.assertEqual(event.module, "mental_health")
        self.assertEqual(event.risk_level, 3)
        self.assertEqual(event.recommended_action, "manual_review")
        self.assertIs(event.metadata["diagnosis"], False)
        self.assertIsNotNone(event.evidence_window)

    def test_predict_mental_safety_returns_v2_result_and_algorithm_event_compatibility(self) -> None:
        result = MentalHealthRiskPipeline().predict_mental_safety(
            self.complete_sample(
                activity_drop_score=0.8,
                sleep_disturbance_score=0.7,
                routine_irregularity_score=0.4,
                movement_vitality_score=0.6,
            )
        )

        self.assertEqual(result.mental_safety_level, 2)
        self.assertEqual(result.mental_safety_score, 61)
        self.assertEqual(result.baseline_confidence, "high")
        self.assertIs(result.diagnosis, False)
        self.assertIn("系统不作医学诊断", result.suggestion)
        self.assertIn("mood_social_withdrawal", result.submodules)
        self.assertIn("cognitive_change_clue", result.submodules)
        self.assertGreaterEqual(result.submodules["mood_social_withdrawal"].score, 60)
        self.assertIn(
            "activity_drop",
            result.submodules["mood_social_withdrawal"].factors,
        )

        payload = result.to_dict()
        self.assertEqual(payload["mental_safety_level"], 2)
        self.assertEqual(
            payload["submodules"]["cognitive_change_clue"]["feature_scores"]["movement_vitality_score"],
            0.6,
        )

        event = result.to_algorithm_event()
        self.assertEqual(event.module, "mental_health")
        self.assertEqual(event.risk_level, result.mental_safety_level)
        self.assertEqual(event.risk_score, result.mental_safety_score / 100)
        self.assertEqual(event.metadata["mental_safety_score"], result.mental_safety_score)
        self.assertIn("submodules", event.metadata)

    def test_missing_optional_features_do_not_enter_coverage_denominator(self) -> None:
        event = MentalHealthRiskPipeline().predict_from_features(
            self.complete_sample(
                activity_drop_score=0.8,
                sleep_disturbance_score=None,
                routine_irregularity_score=None,
            )
        )

        self.assertEqual(event.risk_score, 0.8)
        self.assertAlmostEqual(event.metadata["feature_coverage"], 0.22 / 0.56, places=4)
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

        coverage = (0.22 + 0.22) / 0.56
        expected = 0.45 * coverage + 0.35 * 0.5 + 0.20 * (1 / 3)
        self.assertAlmostEqual(event.confidence, expected, places=4)

    def test_strong_rule_activity_bedroom_social_can_raise_level_three(self) -> None:
        event = MentalHealthRiskPipeline().predict_from_features(
            self.complete_sample(
                activity_drop_score=0.7,
                sleep_disturbance_score=None,
                routine_irregularity_score=None,
                social_withdrawal_score=0.7,
                bedroom_stay_increase_score=0.8,
                persistent_abnormal_days=7,
            )
        )

        self.assertEqual(event.risk_level, 3)
        self.assertEqual(event.metadata["score_status"], "strong_rule_override")
        self.assertIn("activity_bedroom_social_strong_rule", event.risk_factors)
        self.assertEqual(
            event.metadata["strong_rule_matches"][0]["rule_id"],
            "activity_bedroom_social_7d",
        )

    def test_strong_rule_sleep_leave_bed_activity_can_raise_level_three(self) -> None:
        event = MentalHealthRiskPipeline().predict_from_features(
            self.complete_sample(
                activity_drop_score=0.7,
                sleep_disturbance_score=0.8,
                routine_irregularity_score=None,
                night_leave_bed_count=4,
            )
        )

        self.assertEqual(event.risk_level, 3)
        self.assertIn("sleep_leave_bed_activity_strong_rule", event.risk_factors)
        self.assertEqual(
            event.metadata["strong_rule_matches"][0]["rule_id"],
            "sleep_leave_bed_activity_decline",
        )

    def test_strong_rule_high_risk_wandering_can_raise_level_three_without_score_features(self) -> None:
        event = MentalHealthRiskPipeline().predict_from_features(
            self.complete_sample(
                activity_drop_score=None,
                sleep_disturbance_score=None,
                routine_irregularity_score=None,
                consecutive_nights_with_wandering=2,
                doorway_wandering_count=1,
                bathroom_entrance_wandering_count=1,
            )
        )

        self.assertEqual(event.risk_level, 3)
        self.assertEqual(event.risk_score, 0.65)
        self.assertEqual(event.metadata["score_status"], "strong_rule_override")
        self.assertIn("high_risk_wandering_strong_rule", event.risk_factors)

    def test_unverified_high_risk_wandering_score_does_not_bypass_evidence_rule(self) -> None:
        event = MentalHealthRiskPipeline().predict_from_features(
            self.complete_sample(
                activity_drop_score=None,
                sleep_disturbance_score=None,
                routine_irregularity_score=None,
                high_risk_wandering_score=0.9,
                consecutive_nights_with_wandering=0,
                doorway_wandering_count=0,
                bathroom_entrance_wandering_count=0,
            )
        )

        self.assertNotIn("high_risk_wandering_strong_rule", event.risk_factors)
        self.assertFalse(event.metadata["strong_rule_matches"])

    def test_verified_high_risk_wandering_score_can_raise_level_three(self) -> None:
        event = MentalHealthRiskPipeline().predict_from_features(
            self.complete_sample(
                activity_drop_score=None,
                sleep_disturbance_score=None,
                routine_irregularity_score=None,
                high_risk_wandering_score=0.9,
                high_risk_wandering_verified=True,
            )
        )

        self.assertEqual(event.risk_level, 3)
        self.assertIn("high_risk_wandering_strong_rule", event.risk_factors)

    def test_strong_rule_active_cognitive_task_with_motor_change(self) -> None:
        event = MentalHealthRiskPipeline().predict_from_features(
            self.complete_sample(
                activity_drop_score=None,
                sleep_disturbance_score=None,
                routine_irregularity_score=None,
                movement_vitality_score=0.7,
                active_cognitive_task_score=0.8,
            )
        )

        self.assertEqual(event.risk_level, 3)
        self.assertIn("active_cognitive_motor_strong_rule", event.risk_factors)
        self.assertEqual(
            event.metadata["strong_rule_matches"][0]["rule_id"],
            "active_cognitive_motor_decline",
        )

    def test_auxiliary_isolation_forest_scores_multimetric_anomaly(self) -> None:
        history = [
            {
                "date": f"2026-06-{day:02d}",
                "activity_drop_score": 0.10 + day * 0.005,
                "sleep_disturbance_score": 0.12 + day * 0.004,
                "routine_irregularity_score": 0.08 + day * 0.003,
            }
            for day in range(10, 20)
        ]

        event = MentalHealthRiskPipeline().predict_from_features(
            self.complete_sample(
                activity_drop_score=0.86,
                sleep_disturbance_score=0.82,
                routine_irregularity_score=0.78,
                auxiliary_history=history,
            )
        )

        isolation = event.metadata["auxiliary_models"]["isolation_forest"]
        self.assertEqual(isolation["status"], "available")
        self.assertTrue(isolation["is_anomaly"])
        self.assertGreaterEqual(isolation["anomaly_score"], 0.8)
        self.assertIn("activity_drop_score", isolation["factors"])
        self.assertIn("multimetric_anomaly_auxiliary", event.risk_factors)
        self.assertEqual(event.metadata["auxiliary_model_status"]["isolation_forest"], "available")

    def test_auxiliary_change_point_detects_recent_upward_shift(self) -> None:
        history = [
            {"date": "2026-06-18", "activity_drop_score": 0.10, "sleep_disturbance_score": 0.12},
            {"date": "2026-06-19", "activity_drop_score": 0.11, "sleep_disturbance_score": 0.11},
            {"date": "2026-06-20", "activity_drop_score": 0.12, "sleep_disturbance_score": 0.10},
            {"date": "2026-06-21", "activity_drop_score": 0.13, "sleep_disturbance_score": 0.12},
            {"date": "2026-06-22", "activity_drop_score": 0.58, "sleep_disturbance_score": 0.14},
            {"date": "2026-06-23", "activity_drop_score": 0.62, "sleep_disturbance_score": 0.13},
            {"date": "2026-06-24", "activity_drop_score": 0.66, "sleep_disturbance_score": 0.12},
        ]

        event = MentalHealthRiskPipeline().predict_from_features(
            self.complete_sample(
                activity_drop_score=0.70,
                sleep_disturbance_score=0.13,
                routine_irregularity_score=None,
                auxiliary_history=history,
            )
        )

        change_point = event.metadata["auxiliary_models"]["change_point"]
        self.assertEqual(change_point["status"], "available")
        self.assertTrue(change_point["has_change"])
        self.assertEqual(change_point["changes"][0]["feature"], "activity_drop_score")
        self.assertEqual(change_point["changes"][0]["change_time"], "2026-06-22")
        self.assertIn("trend_change_auxiliary", event.risk_factors)

    def test_auxiliary_models_are_insufficient_without_history(self) -> None:
        event = MentalHealthRiskPipeline().predict_from_features(self.complete_sample())

        auxiliary = event.metadata["auxiliary_models"]
        self.assertEqual(auxiliary["isolation_forest"]["status"], "insufficient_data")
        self.assertEqual(auxiliary["change_point"]["status"], "insufficient_data")
        self.assertNotIn("multimetric_anomaly_auxiliary", event.risk_factors)
        self.assertNotIn("trend_change_auxiliary", event.risk_factors)


if __name__ == "__main__":
    unittest.main()
