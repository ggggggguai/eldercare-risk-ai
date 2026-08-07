"""Focused protocol tests for OPT-CALIB-001."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from elderly_monitoring.modules.mental_health.mood_social.fusion.calibration_optimization import (
    AUDIT_VARIANTS,
    CALIBRATION_METHODS,
    SELECTABLE_VARIANTS,
    _fit_calibrator,
    _fit_variant,
    load_calibration_config,
    load_calibration_inputs,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/training/mood_social_calibration_optimization_v3_3_4.yaml"


def _frame(rows: int = 120) -> pd.DataFrame:
    target = np.array([0, 0, 1, 0, 1, 0] * (rows // 6), dtype=int)
    logit = np.linspace(-3.0, 3.0, rows) + target * 0.75
    return pd.DataFrame(
        {
            "fusion_logit": logit,
            "evidence_combination": np.where(
                np.arange(rows) % 3 == 0,
                "C+S",
                "T",
            ),
            "binary_target": target,
        }
    )


def test_config_freezes_candidate_grid_and_offline_boundary() -> None:
    config = load_calibration_config(CONFIG, repository_root=ROOT)
    assert tuple(config.payload["calibration"]["methods"]) == CALIBRATION_METHODS
    assert tuple(config.payload["candidates"]["selectable"]) == SELECTABLE_VARIANTS
    assert tuple(config.payload["candidates"]["audit_only"]) == AUDIT_VARIANTS
    assert config.payload["production_boundary"]["overwrite_online_package"] is False
    assert config.payload["production_boundary"]["change_http_behavior"] is False
    assert config.payload["production_boundary"]["consume_public_dataset_id"] is False


def test_all_calibrators_produce_finite_bounded_probabilities() -> None:
    probability = np.linspace(0.01, 0.99, 120)
    target = np.array([0, 1] * 60, dtype=int)
    for method in CALIBRATION_METHODS:
        calibrator = _fit_calibrator(probability, target, method)
        predicted = calibrator.predict(probability)
        assert np.isfinite(predicted).all()
        assert np.all((predicted >= 0.0) & (predicted <= 1.0))


def test_corrected_coverage_chain_fits_and_applies_global_probability() -> None:
    frame = _frame()
    model = _fit_variant(frame, "corrected_coverage_platt")
    assert model.fit_input_stage == "global"
    predicted = model.predict_frame(frame)
    primary = predicted["primary_probability"].to_numpy()
    global_probability = predicted["global_probability"].to_numpy()
    modes = frame["evidence_combination"].to_numpy()
    for mode, calibrator in model.coverage_calibrators.items():
        selected = modes == mode
        expected = calibrator.predict(global_probability[selected])
        assert np.allclose(predicted.loc[selected, "final_probability"], expected)
        assert not np.allclose(expected, calibrator.predict(primary[selected]))


def test_legacy_mismatch_is_audit_only_and_reproduced_explicitly() -> None:
    assert "legacy_mismatched_coverage_platt" not in SELECTABLE_VARIANTS
    model = _fit_variant(_frame(), "legacy_mismatched_coverage_platt")
    assert model.variant_id in AUDIT_VARIANTS
    assert model.fit_input_stage == "primary"
    predicted = model.predict_frame(_frame())
    assert np.isfinite(predicted["final_probability"]).all()


def test_calibration_model_joblib_round_trip(tmp_path: Path) -> None:
    frame = _frame()
    model = _fit_variant(frame, "corrected_coverage_platt")
    path = tmp_path / "candidate.joblib"
    joblib.dump(model, path)
    loaded = joblib.load(path)
    assert np.allclose(
        model.predict_frame(frame)["final_probability"],
        loaded.predict_frame(frame)["final_probability"],
    )


def test_data007_assignments_keep_outer_test_participants_out_of_inner_folds() -> None:
    config = load_calibration_config(CONFIG, repository_root=ROOT)
    frame, assignments, _ = load_calibration_inputs(config)
    sampled = frame.groupby("outer_fold", sort=True).head(20)
    for row in sampled.itertuples(index=False):
        mapping = assignments.loc[
            str(row.global_participant_id),
            "inner_validation_fold_by_outer_fold",
        ]
        assert mapping[str(int(row.outer_fold))] is None


def test_dataset_id_does_not_change_candidate_prediction() -> None:
    frame = _frame()
    frame["dataset_id"] = "public_source_a"
    model = _fit_variant(frame, "global_beta")
    original = model.predict_frame(frame)["final_probability"].to_numpy()
    changed = frame.copy()
    changed["dataset_id"] = "unknown_household"
    assert np.array_equal(
        original,
        model.predict_frame(changed)["final_probability"].to_numpy(),
    )
