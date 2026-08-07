"""Focused protocol tests for OPT-TREND-001."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from elderly_monitoring.modules.mental_health.mood_social.trend_optimization import (
    CANDIDATES,
    _attach_concordance,
    _candidate_score,
    compute_robust_trend_features,
    load_trend_optimization_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/training/mood_social_trend_optimization_v3_3_4.yaml"


def test_config_freezes_robust_runtime_and_proxy_boundaries() -> None:
    config = load_trend_optimization_config(CONFIG, repository_root=ROOT)
    assert config.payload["runtime_features"]["windows_days"] == [7, 28]
    assert config.payload["runtime_features"]["exclude_abnormal_days"] is True
    assert tuple(config.payload["candidates"]["ids"]) == CANDIDATES
    assert (
        config.payload["production_boundary"]["use_synthetic_calls_as_supervision"]
        is False
    )


def test_robust_runtime_excludes_abnormal_days_and_uses_dual_windows() -> None:
    today = date(2026, 8, 4)
    history = pd.DataFrame(
        {
            "date": [today - timedelta(days=value) for value in range(1, 29)],
            "value": np.linspace(0.2, 0.8, 28),
            "abnormal": [value in {2, 9} for value in range(1, 29)],
        }
    )
    result = compute_robust_trend_features(
        history, current_date=today, current_value=0.9
    )
    assert result.valid_short_days == 6
    assert result.valid_long_days == 26
    assert result.excluded_abnormal_days == 2
    assert 0.0 <= result.reliability <= 1.0
    assert result.change_point >= 0.0


def test_activity_sleep_concordance_boosts_only_strong_branches_and_caps_social() -> (
    None
):
    frame = pd.DataFrame(
        {
            "branch": ["activity", "sleep", "social"],
            "dataset_id": ["a", "a", "b"],
            "global_participant_id": ["p", "p", "s"],
            "canonical_row_index": [0, 0, 0],
            "outer_fold": [0, 0, 0],
            "robust_deviation": [0.8, 0.9, 0.9],
            "reliability": [0.4, 0.4, 0.8],
        }
    )
    result = _attach_concordance(frame)
    assert np.allclose(result.loc[:1, "optimized_reliability"], [0.55, 0.55])
    assert result.loc[2, "optimized_reliability"] == 0.25


def test_candidate_scores_do_not_require_dataset_id_as_risk_input() -> None:
    frame = pd.DataFrame(
        {
            "probability": [0.4, 0.6],
            "optimized_reliability": [0.5, 0.5],
            "activity_sleep_concordance": [0.0, 1.0],
            **{
                name: [0.2, 0.8]
                for name in (
                    "robust_deviation",
                    "risk_slope",
                    "isolation_forest",
                    "change_point",
                    "persistence",
                    "history_coverage",
                )
            },
        }
    )
    for candidate in CANDIDATES:
        first = _candidate_score(frame, "activity", candidate)
        changed = frame.copy()
        changed["dataset_id"] = ["x", "y"]
        assert np.array_equal(first, _candidate_score(changed, "activity", candidate))
