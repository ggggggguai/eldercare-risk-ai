"""Focused protocol tests for OPT-EXPERT-001."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from elderly_monitoring.modules.mental_health.mood_social.experts.optimization import (
    CANDIDATES,
    EXPERTS,
    _engineer_features,
    _fit_candidate,
    load_expert_optimization_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/training/mood_social_expert_optimization_v3_3_4.yaml"


def _frame(rows: int = 80) -> pd.DataFrame:
    target = np.array([0, 1] * (rows // 2), dtype=int)
    return pd.DataFrame(
        {
            "activity.activity_volume_norm": target * 0.8 + np.linspace(0.0, 0.2, rows),
            "activity.sedentary_ratio": 1.0 - target * 0.4,
            "sleep.sleep_efficiency": target * 0.7 + 0.2,
            "sleep.sleep_fragmentation": 1.0 - target * 0.3,
            "source__steps_rolling_6_median_recent": np.linspace(0.1, 0.9, rows),
            "source__sleep_asleep_mean_recent": np.linspace(0.3, 0.8, rows),
        }
    )


def test_config_freezes_expert_scope_and_candidate_grid() -> None:
    config = load_expert_optimization_config(CONFIG, repository_root=ROOT)
    assert tuple(config.payload["experts"]) == EXPERTS
    assert tuple(config.payload["candidates"]["ids"]) == CANDIDATES
    assert config.payload["production_boundary"]["overwrite_base_models"] is False
    assert config.payload["production_boundary"]["use_dataset_id_as_feature"] is False


def test_engineered_features_have_temporal_summaries_and_joint_interactions() -> None:
    features = _engineer_features("joint", _frame())
    assert "temporal_summary_mean" in features
    assert "temporal_summary_std" in features
    assert "activity_sleep_volume_efficiency" in features
    assert "sedentary_sleep_fragmentation" in features
    assert not any(
        "dataset" in name.lower() or "mask" in name.lower() for name in features.columns
    )


def test_available_model_candidates_fit_and_predict_finite_probabilities() -> None:
    frame = _frame()
    target = np.array([0, 1] * 40, dtype=int)
    features = _engineer_features("joint", frame)
    for candidate in ("lgbm_temporal", "catboost_temporal", "lgbm_temporal_bagging3"):
        model = _fit_candidate(
            features, target, candidate, seed=20260728, candidate_label="test"
        )
        predicted = model.predict(features)
        assert np.isfinite(predicted).all()
        assert np.all((predicted > 0.0) & (predicted < 1.0))


def test_candidate_grid_retains_xgboost_as_explicit_optional_failure() -> None:
    assert "xgboost_temporal" in CANDIDATES
    assert len(CANDIDATES) == 4
