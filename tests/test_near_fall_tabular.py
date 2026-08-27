from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from elderly_monitoring.modules.fall_risk.near_fall_tabular import (
    extract_near_fall_tabular_features,
    rescore_near_fall_events,
)
from scripts.train.train_near_fall_tabular import _select_threshold


class NearFallTabularTest(unittest.TestCase):
    def test_feature_extractor_has_stable_finite_shape(self) -> None:
        tensor = np.zeros((24, 10, 8), dtype=np.float32)
        tensor[4:, :, 0] = np.linspace(0.0, 1.0, 20)[:, None]
        tensor[4:, :, 6] = 0.8
        tensor[4:, :, 7] = 1.0

        features = extract_near_fall_tabular_features(tensor)

        self.assertEqual(features.shape, (200,))
        self.assertTrue(np.isfinite(features).all())

    def test_feature_extractor_rejects_wrong_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape"):
            extract_near_fall_tabular_features(np.zeros((24, 8), dtype=np.float32))

    def test_threshold_selection_honors_recall_constraint(self) -> None:
        labels = np.asarray([1, 1, 0, 0], dtype=np.int64)
        scores = np.asarray([0.9, 0.4, 0.8, 0.2], dtype=np.float64)

        threshold, _ = _select_threshold(labels, scores, min_recall=1.0)

        self.assertEqual(threshold, 0.4)

    @patch(
        "elderly_monitoring.modules.fall_risk.near_fall_tabular.build_causal_near_fall_tensor",
        return_value=np.ones((24, 10, 8), dtype=np.float32),
    )
    def test_rescores_rule_candidate_with_tabular_model(self, _build_tensor) -> None:
        class Predictor:
            model_version = "near-fall-tabular-test"

            def predict_tensor(self, tensor):
                self.tensor = tensor
                return {"status": "valid", "near_fall_event_score": 0.82}

        result = rescore_near_fall_events(
            [{"start_time": 1.0, "end_time": 2.0, "near_fall_event_score": 0.6}],
            [{"timestamp_sec": 2.0}],
            predictor=Predictor(),
            fallback_to_rule=True,
        )

        self.assertEqual(result[0]["near_fall_event_score"], 0.82)
        self.assertEqual(result[0]["rule_near_fall_event_score"], 0.6)
        self.assertEqual(result[0]["score_source"], "tabular_rescorer")
        self.assertEqual(result[0]["model_version"], "near-fall-tabular-test")

    @patch(
        "elderly_monitoring.modules.fall_risk.near_fall_tabular.build_causal_near_fall_tensor",
        return_value=None,
    )
    def test_runtime_rescorer_falls_back_to_rule_when_tensor_unavailable(
        self, _build_tensor
    ) -> None:
        class Predictor:
            model_version = "near-fall-tabular-test"

        result = rescore_near_fall_events(
            [{"start_time": 1.0, "end_time": 2.0, "near_fall_event_score": 0.6}],
            [{"timestamp_sec": 2.0}],
            predictor=Predictor(),
            fallback_to_rule=True,
        )

        self.assertEqual(result[0]["near_fall_event_score"], 0.6)
        self.assertEqual(result[0]["score_source"], "rule_fallback")
        self.assertEqual(
            result[0]["fallback_reason"],
            "insufficient_causal_context_or_quality",
        )

    @patch(
        "elderly_monitoring.modules.fall_risk.near_fall_tabular.build_causal_near_fall_tensor",
        return_value=np.ones((24, 10, 8), dtype=np.float32),
    )
    def test_runtime_rescorer_falls_back_on_model_inference_failure(
        self, _build_tensor
    ) -> None:
        class Predictor:
            model_version = "near-fall-tabular-test"

            def predict_tensor(self, tensor):
                raise AssertionError("model inference failed")

        result = rescore_near_fall_events(
            [{"start_time": 1.0, "end_time": 2.0, "near_fall_event_score": 0.6}],
            [{"timestamp_sec": 2.0}],
            predictor=Predictor(),
            fallback_to_rule=True,
        )

        self.assertEqual(result[0]["near_fall_event_score"], 0.6)
        self.assertEqual(result[0]["score_source"], "rule_fallback")
        self.assertEqual(result[0]["fallback_reason"], "model inference failed")
