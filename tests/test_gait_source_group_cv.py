from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "evaluate"
    / "run_gait_source_group_cv.py"
)
SPEC = importlib.util.spec_from_file_location("run_gait_source_group_cv", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_holdout_source_is_excluded_from_action_pretraining() -> None:
    arrays = {
        "partitions": np.asarray(
            ["train", "validation", "train", "validation", "excluded"]
        ),
        "source_group_ids": np.asarray(
            ["held", "held", "other", "other", "other"]
        ),
    }

    partitions = MODULE.derive_action_partitions(arrays, "held")

    assert partitions.tolist() == [
        "excluded",
        "excluded",
        "train",
        "validation",
        "excluded",
    ]


def test_gait_holdout_partition_is_source_disjoint() -> None:
    arrays = {"source_group_ids": np.asarray(["held", "other", "held"])}

    partitions = MODULE.derive_gait_partitions(arrays, "held")

    assert partitions.tolist() == ["validation", "train", "validation"]


def test_oof_metrics_use_fixed_threshold_and_include_pr_auc() -> None:
    rows = [
        {"label": 1, "probability": 0.9},
        {"label": 1, "probability": 0.4},
        {"label": 0, "probability": 0.6},
        {"label": 0, "probability": 0.1},
    ]

    metrics = MODULE.evaluation_metrics(rows, threshold=0.5)

    assert metrics["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert metrics["f1"] == pytest.approx(0.5)
    assert metrics["pr_auc"] == pytest.approx(5 / 6)
