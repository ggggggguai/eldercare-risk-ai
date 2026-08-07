import math

import pytest

from training.cognitive_change_clue.v34_metrics import (
    aggregate_subjects,
    apply_calibrator,
    expected_calibration_error,
    fit_calibrator,
    select_workpoint,
    temporary_platt_evaluation,
)


def _row(subject, label, logit, probability=None):
    row = {
        "subject_id": subject,
        "diagnosis": "HC" if label == 0 else "MCI",
        "label_hc_vs_non_hc": label,
        "raw_logit": logit,
    }
    if probability is not None:
        row["p"] = probability
    return row


def test_raw_and_calibrated_subject_aggregation_use_different_frozen_orders():
    rows = [_row("a", 0, -2.0, 0.1), _row("a", 0, 2.0, 0.8), _row("b", 1, 1.0, 0.7)]
    raw = aggregate_subjects(rows)
    calibrated = aggregate_subjects(rows, probability_field="p")
    assert raw[0]["probability"] == pytest.approx(0.5)
    assert calibrated[0]["probability"] == pytest.approx(0.45)
    assert raw[0]["aggregation"] == "mean_task_raw_logit_then_sigmoid"
    assert calibrated[0]["aggregation"] == "mean_task_calibrated_probability"


def test_ece_uses_ten_equal_width_bins_and_includes_probability_one():
    value = expected_calibration_error([0, 1], [0.0, 1.0])
    assert math.isfinite(value)
    assert value == pytest.approx(0.0)


def test_workpoint_chooses_highest_threshold_for_same_best_predictions():
    rows = [
        {"label_hc_vs_non_hc": 0, "probability": 0.10},
        {"label_hc_vs_non_hc": 0, "probability": 0.20},
        {"label_hc_vs_non_hc": 1, "probability": 0.80},
        {"label_hc_vs_non_hc": 1, "probability": 0.90},
    ]
    selected = select_workpoint(rows)
    assert selected["sensitivity"] == 1.0
    assert selected["specificity"] == 1.0
    assert selected["threshold"] == pytest.approx(0.8)


def test_temporary_platt_fits_calibration_and_scores_outer_only():
    calibration = [
        _row("hc1", 0, -2.0),
        _row("hc2", 0, -1.0),
        _row("mci1", 1, 1.0),
        _row("ad1", 1, 2.0),
    ]
    evaluation = [
        _row("hc3", 0, -1.5),
        _row("hc4", 0, -0.5),
        _row("mci2", 1, 0.5),
        _row("ad2", 1, 1.5),
    ]
    report = temporary_platt_evaluation(calibration, evaluation)
    assert report["fit_scope"] == "inner_calibration_task_raw_logits"
    assert report["outer_evaluation_subject_metrics"]["roc_auc"] == pytest.approx(1.0)


@pytest.mark.parametrize("method", ["platt", "temperature", "isotonic"])
def test_frozen_calibrators_are_serializable_and_bounded(method):
    logits = [-3.0, -1.0, 1.0, 3.0]
    labels = [0, 0, 1, 1]
    parameters = fit_calibrator(method, logits, labels)
    probabilities = apply_calibrator(method, [-10.0, 0.0, 10.0], parameters)
    assert len(probabilities) == 3
    assert all(0.0 <= value <= 1.0 for value in probabilities)
    assert probabilities == sorted(probabilities)


def test_temperature_respects_frozen_optimization_bounds():
    parameters = fit_calibrator("temperature", [-4.0, -1.0, 1.0, 4.0], [0, 0, 1, 1])
    assert 0.05 <= parameters["temperature"] <= 10.0
