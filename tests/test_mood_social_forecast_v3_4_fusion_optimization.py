from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from elderly_monitoring.modules.mental_health.mood_social.forecast_fusion_optimization import (
    CALIBRATION_METHODS,
    COMPONENT_FAMILIES,
    ProbabilityCalibrator,
    compare_calibrators,
    freeze_calibration_and_threshold,
    search_blend_weights,
    select_family_components,
    simplex_weight_grid,
)


def _frame() -> pd.DataFrame:
    rows = []
    for participant_index in range(50):
        for month in range(2):
            target = int((participant_index + month) % 4 == 0)
            rows.append(
                {
                    "global_participant_id": f"person::{participant_index}",
                    "target_window_id": f"person::{participant_index}::m{month}",
                    "future_binary_target": target,
                    "outer_fold_id": 4,
                    "inner_fold_id": participant_index % 5,
                    "p_elasticnet_logistic": 0.75 if target else 0.25,
                    "p_lightgbm": 0.85 if target else 0.15,
                    "p_catboost": 0.65 if target else 0.35,
                }
            )
    return pd.DataFrame(rows)


def test_simplex_grid_is_frozen_nonnegative_and_complete() -> None:
    grid = simplex_weight_grid(0.1)
    assert len(grid) == 66
    assert all(sum(values) == pytest.approx(1.0) for values in grid)
    assert all(value >= 0.0 for values in grid for value in values)


def test_family_selection_uses_auprc_then_brier_then_order() -> None:
    rows = []
    for family_index, family in enumerate(COMPONENT_FAMILIES):
        rows.extend(
            [
                {
                    "model_family": family,
                    "candidate_id": f"{family_index}-a",
                    "candidate_index": family_index * 2,
                    "status": "passed",
                    "auprc": 0.7,
                    "brier": 0.2,
                },
                {
                    "model_family": family,
                    "candidate_id": f"{family_index}-b",
                    "candidate_index": family_index * 2 + 1,
                    "status": "passed",
                    "auprc": 0.7,
                    "brier": 0.3,
                },
            ]
        )
    selected = select_family_components(pd.DataFrame(rows))
    assert selected["candidate_id"].tolist() == ["0-a", "1-a", "2-a"]


def test_weight_calibration_and_threshold_use_complete_inner_oof() -> None:
    frame = _frame()
    selected_weights, search, blended = search_blend_weights(frame)
    assert len(search) == 66
    assert sum(selected_weights.values()) == pytest.approx(1.0)
    method, candidates, calibration_oof = compare_calibrators(frame, blended)
    assert set(candidates["method"]) == set(CALIBRATION_METHODS)
    assert calibration_oof["p_calibrated_selected"].notna().all()
    assert set(calibration_oof["inner_fold_id"]) == set(range(5))
    calibrator, threshold, audit = freeze_calibration_and_threshold(
        frame,
        blended,
        method,
        calibration_oof["p_calibrated_selected"],
    )
    assert isinstance(calibrator, ProbabilityCalibrator)
    assert 0.0 <= threshold <= 1.0
    assert audit["metric"] == "participant_equal_macro_f1"
    assert np.isfinite(calibrator.predict([0.1, 0.9])).all()


@pytest.mark.parametrize("method", CALIBRATION_METHODS)
def test_calibrators_are_finite_and_bounded(method: str) -> None:
    probability = np.linspace(0.01, 0.99, 100)
    target = (probability > 0.5).astype("int8")
    calibrator = ProbabilityCalibrator.fit(
        method, probability, target, np.ones(len(target))
    )
    predicted = calibrator.predict(probability)
    assert np.isfinite(predicted).all()
    assert ((predicted > 0.0) & (predicted < 1.0)).all()
