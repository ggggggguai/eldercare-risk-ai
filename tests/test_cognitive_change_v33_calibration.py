from __future__ import annotations

import math

import pytest

from training.cognitive_change_clue.calibrate_and_test import (
    calibrated_probability,
    fit_platt_calibration,
)
from training.cognitive_change_clue.dataset import CognitiveDatasetError, CognitiveFeatureDataset


def _record(logit: float, label: int) -> dict:
    return {"hc_vs_non_hc": {"raw_logit": logit, "label": label}}


def test_platt_fit_is_finite_and_task_level() -> None:
    records = [_record(-2.0, 0), _record(-1.0, 0), _record(1.0, 1), _record(2.0, 1)]
    result = fit_platt_calibration(records, checkpoint_sha256="a" * 64)
    assert result["schema_version"] == "platt_calibration_v1"
    assert result["input"] == "task_raw_logit"
    assert result["validation_complete_task_count"] == 4
    assert math.isfinite(result["a"])
    assert math.isfinite(result["b"])
    assert calibrated_probability(2.0, result) > calibrated_probability(-2.0, result)


def test_platt_fit_rejects_single_class() -> None:
    with pytest.raises(Exception):
        fit_platt_calibration([_record(-1.0, 0), _record(1.0, 0)], checkpoint_sha256="a" * 64)


def test_training_dataset_still_rejects_test_split() -> None:
    with pytest.raises(CognitiveDatasetError, match="evaluation-only"):
        CognitiveFeatureDataset(split_name="test")
