from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import joblib
import pytest
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from elderly_monitoring.modules.mental_health.mood_social.fusion.optimization_v3_3_4 import (
    FusionV334Config,
    FusionV334OptimizationError,
    _weak_branch_ablation,
    load_v334_config,
    load_v334_inputs,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/training/mood_social_fusion_optimization_v3_3_4.yaml"


def test_current_online_baseline_binds_promoted_deployment_probability() -> None:
    config = load_v334_config(CONFIG, repository_root=ROOT)
    frame, _, protection = load_v334_inputs(config)

    target = frame["binary_target"].to_numpy(dtype=int)
    probability = frame["baseline_probability"].to_numpy(dtype=float)

    assert (
        protection["baseline_probability_column"] == "deployment_candidate_probability"
    )
    assert average_precision_score(target, probability) == pytest.approx(0.2387837904)
    assert roc_auc_score(target, probability) == pytest.approx(0.6519567616)
    assert brier_score_loss(target, probability) == pytest.approx(0.2399408103)


def test_legacy_fusion002_baseline_column_is_rejected() -> None:
    config = load_v334_config(CONFIG, repository_root=ROOT)
    payload = deepcopy(config.payload)
    payload["input"]["baseline_probability_column"] = "baseline_probability"
    invalid = FusionV334Config(config.repository_root, payload, config.config_path)

    with pytest.raises(
        FusionV334OptimizationError,
        match="must bind deployment_candidate_probability",
    ):
        load_v334_inputs(invalid)


def test_weak_branch_ablation_accepts_read_only_prediction_arrays() -> None:
    config = load_v334_config(CONFIG, repository_root=ROOT)
    frame, _, _ = load_v334_inputs(config)
    model = joblib.load(config.model_path)

    ablation = _weak_branch_ablation(frame.iloc[:256].copy(), model)

    assert set(ablation["removed_branch"]) == {
        "physiology",
        "social_context",
        "trend_social",
    }
