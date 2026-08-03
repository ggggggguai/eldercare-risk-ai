"""Focused protocol tests for OPT-FUSION-001."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from elderly_monitoring.modules.mental_health.mood_social.fusion.optimization import (
    CALIBRATION_METHODS,
    RUN_ID,
    TASK_ID,
    V2_VARIANTS,
    _fit_calibrator,
    _predict_calibrator,
    _prior_adjust,
    _v2_feature_names_for_test,
    _v2_features,
    load_optimization_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/training/mood_social_fusion_optimization_v3_3_3.yaml"


def _calibration_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset_id": ["a", "a", "a", "a", "b", "b", "b", "b"],
            "global_participant_id": [f"p{i}" for i in range(8)],
            "binary_target": [0, 0, 1, 1, 0, 0, 1, 1],
            "baseline_probability": [0.1, 0.3, 0.6, 0.9, 0.2, 0.4, 0.7, 0.8],
        }
    )


def test_config_freezes_offline_candidate_boundary() -> None:
    config = load_optimization_config(CONFIG, repository_root=ROOT)
    assert config.payload["task_id"] == TASK_ID
    assert config.payload["run_id"] == RUN_ID
    assert tuple(config.payload["calibration"]["methods"]) == CALIBRATION_METHODS
    assert tuple(config.payload["fusion_v2"]["variants"]) == V2_VARIANTS
    assert config.payload["production_boundary"]["offline_only"] is True
    assert config.payload["production_boundary"]["select_threshold"] is False
    assert "candidates" in config.model_path.parts


@pytest.mark.parametrize("method", CALIBRATION_METHODS)
def test_calibration_candidates_return_probabilities(method: str) -> None:
    frame = _calibration_frame()
    model = _fit_calibrator(frame, method)
    predicted = _predict_calibrator(
        model,
        frame["baseline_probability"].to_numpy(),
        method,
    )
    assert predicted.shape == (len(frame),)
    assert np.isfinite(predicted).all()
    assert np.all((predicted > 0) & (predicted < 1))


def test_v2_features_add_only_frozen_low_order_interactions() -> None:
    base = np.arange(16, dtype=float).reshape(2, 8)
    features = _v2_features(base)
    names = _v2_feature_names_for_test()
    assert features.shape == (2, 14)
    assert len(names) == features.shape[1]
    assert np.array_equal(features[:, :8], base)
    assert features[1, 8] == pytest.approx(base[1, 0] * base[1, 1])


def test_prior_adjustment_is_monotonic_and_offline_only_math() -> None:
    probability = np.array([0.1, 0.3, 0.6, 0.9])
    adjusted = _prior_adjust(probability, observed=0.2, prior=0.1)
    assert np.all(np.diff(adjusted) > 0)
    assert np.all(adjusted < probability)
