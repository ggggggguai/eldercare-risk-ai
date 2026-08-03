"""Focused protocol tests for OPT-FUSION-002."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from elderly_monitoring.modules.mental_health.mood_social.fusion.deployment_optimization import (
    CANDIDATE_IDS,
    RUN_ID,
    TASK_ID,
    _fit_candidate,
    _interaction_features,
    _promotion_decision,
    load_deployment_optimization_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "configs/training/mood_social_fusion_deployment_optimization_v3_3_3.yaml"
)


def _frame(rows: int = 80) -> pd.DataFrame:
    target = np.array([0, 1] * (rows // 2), dtype=int)
    frame = pd.DataFrame(
        {
            "dataset_id": ["source_a"] * rows,
            "global_participant_id": [f"participant_{index}" for index in range(rows)],
            "binary_target": target,
        }
    )
    for index, branch in enumerate(
        ("activity", "sleep", "joint", "physiology", "social_context")
    ):
        frame[f"expert_{branch}_expert_mask"] = 1
        frame[f"expert_{branch}_confidence"] = 0.5
        frame[f"expert_{branch}_current_probability"] = np.clip(
            0.25 + target * 0.45 + index * 0.005,
            0.01,
            0.99,
        )
    for branch in ("activity", "sleep", "social"):
        frame[f"trend_{branch}_personal_change_mask"] = 1
        frame[f"trend_{branch}_reliability"] = 0.5
        frame[f"trend_{branch}_personal_change_evidence"] = 0.2 + target * 0.6
    return frame


def test_config_freezes_deployment_safe_boundary_and_gate() -> None:
    config = load_deployment_optimization_config(CONFIG, repository_root=ROOT)
    assert config.payload["task_id"] == TASK_ID
    assert config.payload["run_id"] == RUN_ID
    assert tuple(config.payload["candidates"]["ids"]) == CANDIDATE_IDS
    assert config.payload["candidates"]["public_dataset_id_as_model_input"] is False
    assert config.payload["production_boundary"]["consume_public_dataset_id"] is False
    assert config.payload["production_boundary"]["select_production_threshold"] is False
    assert config.payload["promotion_gate"]["minimum_natural_auprc_delta"] == 0.01


def test_interaction_features_are_low_order_and_deterministic() -> None:
    base = np.arange(16, dtype=float).reshape(2, 8)
    transformed = _interaction_features(base)
    assert transformed.shape == (2, 14)
    assert np.array_equal(transformed[:, :8], base)
    assert transformed[1, 8] == base[1, 0] * base[1, 1]


def test_all_candidates_produce_finite_probabilities_without_source_mapping() -> None:
    training = _frame()
    test = training.iloc[:4].copy()
    for candidate_id in CANDIDATE_IDS:
        model = _fit_candidate(training, candidate_id)
        original = model.predict_frame(test)["fusion_probability"].to_numpy()
        changed = test.copy()
        changed["dataset_id"] = "unseen_production_home"
        remapped = model.predict_frame(changed)["fusion_probability"].to_numpy()
        assert np.isfinite(original).all()
        assert np.all((original > 0) & (original < 1))
        assert np.array_equal(original, remapped)


def test_promotion_gate_requires_every_frozen_condition() -> None:
    config = load_deployment_optimization_config(CONFIG, repository_root=ROOT)
    overall = pd.DataFrame(
        [
            {
                "model": "fusion_002_baseline",
                "auprc": 0.20,
                "auroc": 0.60,
                "brier": 0.24,
            },
            {
                "model": "deployment_candidate",
                "auprc": 0.22,
                "auroc": 0.61,
                "brier": 0.241,
            },
        ]
    )
    outer = pd.DataFrame({"auprc_delta": [0.01, 0.0, 0.02, -0.01, -0.02]})
    source = pd.DataFrame({"positive_rows": [20, 25], "auprc_delta": [-0.01, 0.0]})
    passed = _promotion_decision(config, overall, outer, source)
    assert passed["promotion_passed"] is True
    assert passed["art001_default"] == "deployment_candidate"

    source.loc[0, "auprc_delta"] = -0.03
    failed = _promotion_decision(config, overall, outer, source)
    assert failed["promotion_passed"] is False
    assert failed["art001_default"] == "fusion_002_baseline"
