from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from elderly_monitoring.modules.fall_risk.baseline import BaselineModelConfig
from elderly_monitoring.modules.fall_risk.baseline_longitudinal import (
    LongitudinalDataError,
    build_longitudinal_split,
)
from elderly_monitoring.modules.fall_risk.baseline_supervised import (
    FEATURE_NAMES,
    build_supervised_feature_rows,
    predict_longitudinal_logistic_candidate,
    train_longitudinal_logistic_candidate,
)

from test_fall_risk_baseline_longitudinal import dataset, protocol


class FallBaselineSupervisedTest(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = protocol(
            partitions={"train": 0.50, "validation": 0.50, "test": 0.0}
        )
        self.observations, self.labels, self.profiles, self.reviews = dataset(
            people=60, periods=6
        )
        self.artifact = build_longitudinal_split(
            self.observations, self.labels, self.profiles, self.reviews, self.protocol
        )
        self.assertEqual(self.artifact["metadata"]["status"], "ready")
        self.baseline_config = BaselineModelConfig(
            min_history_days=3,
            stable_history_days=3,
            min_history_records=3,
            min_metric_observations=3,
        )

    def _partition_observations(self, partition: str) -> list[dict]:
        identifiers = {
            row["observation_id"]
            for row in self.artifact["assignments"]
            if row["partition"] == partition
        }
        return [row for row in self.observations if row["observation_id"] in identifiers]

    def _development_inputs(self) -> tuple[list[dict], list[dict]]:
        observations = self._partition_observations("train") + self._partition_observations("validation")
        label_ids = {
            row["outcome_label_id"]
            for row in observations
            if row.get("outcome_label_id")
        }
        labels = [row for row in self.labels if row["label_id"] in label_ids]
        return observations, labels

    def test_feature_contract_separates_missing_values_from_masks(self) -> None:
        validation_scoring_ids = {
            row["observation_id"]
            for row in self.artifact["assignments"]
            if row["partition"] == "validation" and row["role"] == "scoring"
        }
        target = next(
            row
            for row in self.observations
            if row["observation_id"] in validation_scoring_ids
        )
        target["metric_quality"]["activity_volume"] = {
            "available": False,
            "observation_count": 0,
            "quality": 0.0,
            "coverage": 0.0,
            "exposure_hours": 0.0,
            "missing_reason": "not_observed",
        }
        target.pop("activity_volume", None)
        self.artifact = build_longitudinal_split(
            self.observations,
            self.labels,
            self.profiles,
            self.reviews,
            self.protocol,
        )
        observations = self._partition_observations("validation")

        rows = build_supervised_feature_rows(
            observations,
            self.artifact["assignments"],
            self.artifact["metadata"],
            self.protocol,
            partition="validation",
            baseline_config=self.baseline_config,
        )

        row = next(item for item in rows if item["observation_id"] == target["observation_id"])
        value_index = FEATURE_NAMES.index("activity_volume")
        available_index = FEATURE_NAMES.index("activity_volume_available")
        self.assertEqual(row["features"][value_index], 0.0)
        self.assertFalse(row["feature_mask"][value_index])
        self.assertEqual(row["features"][available_index], 0.0)
        self.assertTrue(row["feature_mask"][available_index])

    def test_training_refuses_test_observations(self) -> None:
        test_protocol = protocol(
            partitions={"train": 0.34, "validation": 0.33, "test": 0.33}
        )
        observations, labels, profiles, reviews = dataset(people=60, periods=6)
        artifact = build_longitudinal_split(
            observations, labels, profiles, reviews, test_protocol
        )
        test_ids = {
            row["observation_id"]
            for row in artifact["assignments"]
            if row["partition"] == "test"
        }
        self.assertTrue(test_ids)
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(LongitudinalDataError, "sealed test observations"):
                train_longitudinal_logistic_candidate(
                    observations,
                    labels,
                    artifact["assignments"],
                    artifact["metadata"],
                    test_protocol,
                    output_dir=Path(tmpdir) / "candidate",
                    baseline_config=self.baseline_config,
                )

    def test_logistic_artifact_binds_split_and_validation_calibration(self) -> None:
        observations, labels = self._development_inputs()
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "candidate"
            report = train_longitudinal_logistic_candidate(
                observations,
                labels,
                self.artifact["assignments"],
                self.artifact["metadata"],
                self.protocol,
                output_dir=output,
                baseline_config=self.baseline_config,
            )

            self.assertEqual(report["status"], "provisional")
            self.assertEqual(report["candidate_status"], "validation_only")
            self.assertFalse(report["test_access"])
            self.assertEqual(report["split_id"], self.artifact["metadata"]["split_id"])
            self.assertEqual(report["calibration"]["method"], "platt_validation")
            self.assertTrue((output / "model.joblib").is_file())
            prediction_rows = [
                json.loads(line)
                for line in (output / "validation_predictions.jsonl").read_text().splitlines()
                if line.strip()
            ]
            self.assertTrue(prediction_rows)

    def test_broken_model_and_low_quality_fall_back_to_guarded_score(self) -> None:
        feature = {
            "observation_id": "obs-fallback",
            "partition": "validation",
            "baseline_state": "stable",
            "quality_state": "unavailable",
            "features": [0.0] * len(FEATURE_NAMES),
            "feature_values": {"robust_ewma_cusum_guarded_score": 0.37},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            broken = Path(tmpdir) / "broken.joblib"
            broken.write_text("not a checkpoint", encoding="utf-8")
            [prediction] = predict_longitudinal_logistic_candidate(broken, [feature])

        self.assertTrue(prediction["used_fallback"])
        self.assertEqual(prediction["fallback_reason"], "model_load_failed")
        self.assertEqual(prediction["score"], 0.37)

    def test_empty_real_inputs_emit_blocked_report_without_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "blocked"
            report = train_longitudinal_logistic_candidate(
                [], [], [], {"status": "blocked", "split_id": None}, self.protocol,
                output_dir=output,
                baseline_config=self.baseline_config,
            )
            self.assertEqual(report["status"], "blocked")
            self.assertIn("no_observations", report["blockers"])
            self.assertFalse((output / "model.joblib").exists())


if __name__ == "__main__":
    unittest.main()
