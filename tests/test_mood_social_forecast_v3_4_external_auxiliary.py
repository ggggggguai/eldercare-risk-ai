from __future__ import annotations

import numpy as np
import pandas as pd

from elderly_monitoring.modules.mental_health.mood_social.forecast_external_auxiliary import (
    apply_conscious_proxy_to_psyche,
    fit_final_ridge,
    nested_group_ridge_oof,
)
from elderly_monitoring.datasets.adapters.conscious_wearable import COMMON_FEATURES
from elderly_monitoring.datasets.adapters.obf_psychiatric import _rhythm_features


def _training_frame() -> pd.DataFrame:
    rows = []
    for participant in range(20):
        for window in range(2):
            row = {
                "global_participant_id": f"conscious::{participant}",
                "phq9_score": float((participant + window) % 12),
            }
            row.update(
                {
                    name: float(participant + window + feature_index / 10)
                    for feature_index, name in enumerate(COMMON_FEATURES)
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def test_nested_external_oof_is_participant_complete() -> None:
    frame = _training_frame()
    oof, audit = nested_group_ridge_oof(
        frame,
        COMMON_FEATURES,
        "phq9_score",
        "global_participant_id",
    )
    assert len(oof) == len(frame)
    assert oof["prediction"].notna().all()
    assert set(oof["outer_fold_id"]) == set(range(5))
    assert audit["participant_count"] == 20


def test_fixed_conscious_proxy_does_not_read_psyche_labels() -> None:
    training = _training_frame()
    model = fit_final_ridge(
        training,
        COMMON_FEATURES,
        "phq9_score",
        "global_participant_id",
    )
    source = pd.DataFrame(
        {
            "global_participant_id": ["psyche::1", "psyche::2"],
            "target_window_id": ["one", "two"],
            "outer_fold_id": [0, 1],
            "future_binary_target": [0, 1],
            **{
                f"{name}__anchor": [float(index), float(index + 1)]
                for index, name in enumerate(COMMON_FEATURES)
            },
        }
    )
    before = apply_conscious_proxy_to_psyche(model, {"forecast_1m": source})
    source["future_binary_target"] = 1 - source["future_binary_target"]
    after = apply_conscious_proxy_to_psyche(model, {"forecast_1m": source})
    pd.testing.assert_frame_equal(before, after)


def test_obf_rhythm_features_are_finite_for_complete_signal() -> None:
    timestamp = pd.date_range("2020-01-01", periods=3 * 1440, freq="min")
    activity = np.maximum(
        0.0, 100.0 + 80.0 * np.sin(np.arange(len(timestamp)) * 2 * np.pi / 1440)
    )
    frame = pd.DataFrame(
        {
            "timestamp": timestamp,
            "date": timestamp.normalize(),
            "activity": activity,
        }
    )
    features = _rhythm_features(frame)
    for name in (
        "relative_amplitude",
        "interdaily_stability",
        "intradaily_variability",
        "daily_total_mean",
    ):
        assert np.isfinite(features[name])
