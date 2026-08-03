"""Focused FUSION-002 protocol and gating tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from elderly_monitoring.modules.mental_health.mood_social.fusion.masked_stacking import (
    BUNDLE_VERSION,
    C_VALUES,
    INPUT_FEATURES,
    RUN_ID,
    TASK_ID,
    MaskedStackingModel,
    _calibration_payload,
    build_fusion_features,
    effective_evidence,
    evidence_combination,
    load_masked_stacking_config,
    masked_stacking_sample_weights,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/training/mood_social_masked_stacking_v3_3_3.yaml"


def _frame() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "global_participant_id": ["a", "b", "c", "d"],
            "dataset_id": ["d1", "d1", "d2", "d2"],
            "binary_target": [0, 1, 0, 1],
        }
    )
    for branch in ("activity", "sleep", "joint", "physiology", "social_context"):
        frame[f"expert_{branch}_current_probability"] = [0.2, 0.8, 0.4, 0.6]
        frame[f"expert_{branch}_expert_mask"] = [1, 0, 1, 0]
        frame[f"expert_{branch}_confidence"] = [0.5, 0.0, 0.25, 0.0]
    for branch in ("activity", "sleep", "social"):
        frame[f"trend_{branch}_personal_change_mask"] = [1, 0, 0, 0]
        frame[f"trend_{branch}_reliability"] = [0.4, 0.0, 0.0, 0.0]
        frame[f"trend_{branch}_personal_change_evidence"] = [
            0.3,
            np.nan,
            np.nan,
            np.nan,
        ]
    return frame


def test_config_freezes_fusion_protocol() -> None:
    config = load_masked_stacking_config(CONFIG, repository_root=ROOT)
    assert config.payload["task_id"] == TASK_ID
    assert config.payload["run_id"] == RUN_ID
    assert tuple(config.payload["input_features"]) == INPUT_FEATURES
    assert tuple(config.payload["logistic"]["c_values"]) == C_VALUES
    assert config.payload["probability"]["representation"] == "current_calibrated"
    assert config.payload["mask_policy"]["masks_as_risk_features"] is False
    assert config.payload["production_boundary"]["select_threshold"] is False


def test_formula_features_use_masks_and_confidence_without_mask_columns() -> None:
    frame = _frame()
    matrix = build_fusion_features(frame)
    assert matrix.shape == (4, 8)
    assert np.isfinite(matrix).all()
    assert matrix[1, 0] == 0.0
    assert matrix[0, 0] == pytest.approx(0.5 * np.log(0.2 / 0.8))
    assert effective_evidence(frame).tolist() == pytest.approx([3.7, 0.0, 1.25, 0.0])


def test_evidence_combination_is_report_only_c_s_t_dimension() -> None:
    frame = _frame().iloc[:1].copy()
    frame["expert_activity_expert_mask"] = 1
    frame["expert_sleep_expert_mask"] = 0
    frame["expert_physiology_expert_mask"] = 0
    frame["trend_activity_personal_change_mask"] = 0
    frame["trend_sleep_personal_change_mask"] = 0
    frame["trend_social_personal_change_mask"] = 0
    assert evidence_combination(frame).iloc[0] == "C"
    frame["expert_activity_expert_mask"] = 0
    frame["expert_joint_expert_mask"] = 0
    frame["expert_sleep_expert_mask"] = 1
    assert evidence_combination(frame).iloc[0] == "S"
    frame["expert_sleep_expert_mask"] = 0
    frame["trend_social_personal_change_mask"] = 1
    assert evidence_combination(frame).iloc[0] == "T"
    frame["trend_social_personal_change_mask"] = 0
    assert evidence_combination(frame).iloc[0] == "none"


def test_sample_weights_equalize_dataset_class_participant_and_repeat() -> None:
    frame = pd.DataFrame(
        {
            "dataset_id": ["a", "a", "a", "b", "b"],
            "global_participant_id": ["a1", "a1", "a2", "b1", "b2"],
            "binary_target": [0, 0, 1, 0, 1],
        }
    )
    weights = masked_stacking_sample_weights(frame)
    assert weights.mean() == pytest.approx(1.0)
    for dataset in ("a", "b"):
        selected = frame["dataset_id"].eq(dataset).to_numpy()
        assert weights[selected].sum() == pytest.approx(weights.sum() / 2)
        for target in (0, 1):
            group = selected & frame["binary_target"].eq(target).to_numpy()
            assert weights[group].sum() == pytest.approx(weights[selected].sum() / 2)


@pytest.mark.filterwarnings("ignore:.*penalty.*deprecated.*:FutureWarning")
def test_native_model_hard_gates_zero_effective_evidence() -> None:
    train = _frame().iloc[[0, 1]].copy()
    train["expert_activity_expert_mask"] = 1
    train["expert_activity_confidence"] = 1.0
    matrix = build_fusion_features(train)
    classifier = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs").fit(
        matrix, train["binary_target"]
    )
    model = MaskedStackingModel(
        bundle_version=BUNDLE_VERSION,
        training_version="mood-social-masked-stacking-v3.3.3-v1",
        task_id=TASK_ID,
        run_id=RUN_ID,
        input_features=INPUT_FEATURES,
        probability_representation="current_calibrated",
        selected_c=1.0,
        classifier=classifier,
    )
    result = model.predict_frame(_frame())
    assert not bool(result.loc[1, "available"])
    assert pd.isna(result.loc[1, "fusion_probability"])
    assert bool(result.loc[0, "available"])


def test_empty_calibration_payload_is_safe() -> None:
    frame = pd.DataFrame(
        {
            "fusion_available": pd.Series(dtype=bool),
            "fusion_probability": pd.Series(dtype=float),
            "binary_target": pd.Series(dtype=int),
        }
    )
    table, payload = _calibration_payload(frame)
    assert table.empty
    assert payload["ece"] == {"natural": None, "frozen_four_level": None}
