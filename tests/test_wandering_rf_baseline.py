from __future__ import annotations

import hashlib
import inspect
import json
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path
from unittest import mock

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import yaml

from elderly_monitoring.modules.mental_health.wandering.handcrafted_features import FEATURE_NAMES
from elderly_monitoring.modules.mental_health.wandering.rf_baseline import (
    BINARY_CLASS_NAMES,
    FOUR_CLASS_NAMES,
    PRIMARY_SEED,
    RF_BASE_PARAMS,
    RF_SEEDS,
    TASK_BINARY,
    TASK_FOUR_CLASS,
    ModelTrustError,
    build_binary_sample_weights,
    build_frozen_wp_test_artifacts,
    create_rf_classifier,
    fit_rf_task,
    local_sensitivity,
    model_semantic_fingerprint,
    predict_preprocessed_record,
    safe_load_development_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


class WanderingRandomForestContractTest(unittest.TestCase):
    def test_config_and_frozen_evaluator_have_no_official_path_or_fit_call(self) -> None:
        config = yaml.safe_load(
            (PROJECT_ROOT / "configs/modules/wandering_rf_v1.yaml").read_text(encoding="utf-8")
        )
        self.assertNotIn("official", json.dumps(config, ensure_ascii=False).lower())
        evaluator_source = inspect.getsource(build_frozen_wp_test_artifacts)
        self.assertNotIn(".fit(", evaluator_source)
        self.assertIn("safe_load_development_model", evaluator_source)

    def test_classes_seeds_and_every_rf_parameter_are_frozen(self) -> None:
        self.assertEqual(FOUR_CLASS_NAMES, ("direct", "pacing", "lapping", "random"))
        self.assertEqual(BINARY_CLASS_NAMES, ("direct_or_non_wandering", "wandering_like"))
        self.assertEqual(RF_SEEDS, (20260731, 20260801, 20260802, 20260803, 20260804))
        self.assertEqual(PRIMARY_SEED, 20260731)
        self.assertEqual(
            RF_BASE_PARAMS,
            {
                "n_estimators": 500,
                "criterion": "gini",
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 2,
                "max_features": "sqrt",
                "bootstrap": True,
                "class_weight": None,
                "n_jobs": 1,
                "oob_score": False,
                "max_samples": None,
            },
        )
        classifier = create_rf_classifier(PRIMARY_SEED)
        for name, value in RF_BASE_PARAMS.items():
            self.assertEqual(classifier.get_params()[name], value)
        self.assertEqual(classifier.random_state, PRIMARY_SEED)

    def test_binary_source_class_weights_have_equal_group_totals(self) -> None:
        rows = []
        for source, label, count in (
            ("wandering_patterns", 0, 280),
            ("wandering_patterns", 1, 840),
            ("smartcare", 0, 63),
            ("smartcare", 1, 74),
        ):
            rows.extend({"source_dataset": source, "label": label, "split": "train"} for _ in range(count))

        weights = build_binary_sample_weights(rows)

        expected = {
            ("wandering_patterns", 0): 1.1223214285714285,
            ("wandering_patterns", 1): 0.37410714285714286,
            ("smartcare", 0): 4.988095238095238,
            ("smartcare", 1): 4.246621621621622,
        }
        totals = defaultdict(float)
        for row, weight in zip(rows, weights, strict=True):
            group = (row["source_dataset"], row["label"])
            self.assertAlmostEqual(weight, expected[group])
            totals[group] += float(weight)
        self.assertEqual(set(round(value, 10) for value in totals.values()), {round(1257 / 4, 10)})

    def test_fit_task_passes_only_the_fixed_train_rows_to_estimator(self) -> None:
        rows = []
        labels = ("direct", "pacing", "lapping", "random")
        for class_index, label in enumerate(labels):
            rows.extend(
                {
                    "sample_id": f"train-{class_index}-{index:04d}",
                    "source_dataset": "wandering_patterns",
                    "split": "train",
                    "binary_label": 0 if class_index == 0 else 1,
                    "pattern_label": label,
                    "binary_supervision_eligible": True,
                    "pattern_supervision_eligible": True,
                    "features": [float(class_index)] * 26,
                }
                for index in range(280)
            )
        rows.append(
            {
                **rows[0],
                "sample_id": "validation-sentinel",
                "split": "validation",
                "features": [9999.0] * 26,
            }
        )
        estimator = mock.Mock(spec=RandomForestClassifier)

        def fitted(x: np.ndarray, y: np.ndarray, *, sample_weight: np.ndarray) -> None:
            self.assertEqual(x.shape, (1120, 26))
            self.assertFalse(np.any(x == 9999.0))
            self.assertEqual(Counter(y.tolist()), Counter({0: 280, 1: 280, 2: 280, 3: 280}))
            np.testing.assert_array_equal(sample_weight, np.ones(1120))
            estimator.classes_ = np.asarray([0, 1, 2, 3])

        estimator.fit.side_effect = fitted
        with mock.patch(
            "elderly_monitoring.modules.mental_health.wandering.rf_baseline.create_rf_classifier",
            return_value=estimator,
        ):
            model, medians, weights = fit_rf_task(rows, task=TASK_FOUR_CLASS, seed=PRIMARY_SEED)
        self.assertIs(model, estimator)
        self.assertEqual(medians.shape, (26,))
        np.testing.assert_array_equal(weights, np.ones(1120))

    def test_unavailable_single_record_is_passed_through_without_prediction(self) -> None:
        model = mock.Mock()
        result = predict_preprocessed_record(
            model,
            {
                "sample_id": "short-trajectory",
                "preprocess_status": "unavailable",
                "reason_codes": ["too_few_valid_points"],
            },
            task=TASK_BINARY,
            train_medians=np.zeros(26),
        )
        self.assertEqual(
            result,
            {
                "sample_id": "short-trajectory",
                "prediction_status": "unavailable",
                "reason_codes": ["too_few_valid_points"],
            },
        )
        model.predict_proba.assert_not_called()

    def test_semantic_fingerprint_and_local_sensitivity_are_reproducible(self) -> None:
        x = np.asarray([[0.0] * 26, [1.0] * 26, [0.1] * 26, [0.9] * 26])
        y = np.asarray([0, 1, 0, 1])
        model = RandomForestClassifier(n_estimators=7, min_samples_leaf=1, random_state=17, n_jobs=1).fit(x, y)

        first = model_semantic_fingerprint(model)
        second = model_semantic_fingerprint(model)
        self.assertEqual(first, second)
        self.assertEqual(first["tree_count"], 7)
        sensitivity_a = local_sensitivity(model, x[1], np.median(x, axis=0), feature_names=FEATURE_NAMES)
        sensitivity_b = local_sensitivity(model, x[1], np.median(x, axis=0), feature_names=FEATURE_NAMES)
        self.assertEqual(sensitivity_a, sensitivity_b)
        self.assertEqual(len(sensitivity_a["perturbations"]), 26)
        self.assertEqual(sensitivity_a["explanation_type"], "local_sensitivity_not_shap_not_causal")

    def test_safe_loader_checks_external_manifest_hash_and_joblib_hash_before_deserializing(self) -> None:
        x = np.asarray([[0.0] * 26, [1.0] * 26, [0.2] * 26, [0.8] * 26])
        y = np.asarray([0, 1, 0, 1])
        model = create_rf_classifier(PRIMARY_SEED).fit(x, y)
        fingerprint = model_semantic_fingerprint(model)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_path = root / "models/binary/seed_20260731.joblib"
            model_path.parent.mkdir(parents=True)
            joblib.dump(model, model_path)
            model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
            import elderly_monitoring.modules.mental_health.wandering.rf_baseline as rf
            manifest = {
                "schema_version": "wandering-rf-development-manifest-v1",
                "environment": rf.current_environment_versions(),
                "artifacts": {
                    "models/binary/seed_20260731.joblib": {
                        "byte_count": model_path.stat().st_size,
                        "sha256": model_hash,
                    }
                },
                "models": {
                    "binary/20260731": {
                        "artifact": "models/binary/seed_20260731.joblib",
                        "classes": [0, 1],
                        "semantic_fingerprint": fingerprint,
                    }
                },
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_bytes(_canonical_bytes(manifest))
            manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

            loader = mock.Mock(wraps=joblib.load)
            with self.assertRaises(ModelTrustError):
                safe_load_development_model(root, task=TASK_BINARY, seed=PRIMARY_SEED, expected_manifest_sha256="0" * 64, joblib_loader=loader)
            loader.assert_not_called()

            model_path.write_bytes(model_path.read_bytes() + b"tamper")
            with self.assertRaises(ModelTrustError):
                safe_load_development_model(root, task=TASK_BINARY, seed=PRIMARY_SEED, expected_manifest_sha256=manifest_hash, joblib_loader=loader)
            loader.assert_not_called()

    def test_safe_loader_round_trip_matches_probabilities_and_semantic_fingerprint(self) -> None:
        x = np.asarray([[0.0] * 26, [1.0] * 26, [0.2] * 26, [0.8] * 26])
        y = np.asarray([0, 1, 0, 1])
        model = create_rf_classifier(PRIMARY_SEED).fit(x, y)
        fingerprint = model_semantic_fingerprint(model)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            path = root / "models/binary/seed_20260731.joblib"
            path.parent.mkdir(parents=True)
            joblib.dump(model, path)
            import elderly_monitoring.modules.mental_health.wandering.rf_baseline as rf
            manifest = {
                "schema_version": "wandering-rf-development-manifest-v1",
                "environment": rf.current_environment_versions(),
                "artifacts": {
                    "models/binary/seed_20260731.joblib": {
                        "byte_count": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                },
                "models": {
                    "binary/20260731": {
                        "artifact": "models/binary/seed_20260731.joblib",
                        "classes": [0, 1],
                        "semantic_fingerprint": fingerprint,
                    }
                },
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_bytes(_canonical_bytes(manifest))
            digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

            loaded = safe_load_development_model(root, task=TASK_BINARY, seed=PRIMARY_SEED, expected_manifest_sha256=digest)
            np.testing.assert_array_equal(loaded.classes_, [0, 1])
            np.testing.assert_allclose(loaded.predict_proba(x), model.predict_proba(x), atol=0.0, rtol=0.0)
            self.assertEqual(model_semantic_fingerprint(loaded), fingerprint)


if __name__ == "__main__":
    unittest.main()
