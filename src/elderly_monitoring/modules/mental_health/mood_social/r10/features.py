"""Low-dimensional, runtime-equivalent R10 feature construction."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from elderly_monitoring.modules.mental_health.mood_social.r9.contract import (
    ACTIVITY_FEATURES,
    R9_HISTORY_FEATURES,
    SLEEP_FEATURES,
)


INTERACTION_FEATURES = (
    "r10.low_activity_x_sleep_fragmentation",
    "r10.sedentary_x_low_sleep_efficiency",
    "r10.activity_instability_x_sleep_irregularity",
    "r10.low_activity_x_night_exits",
    "r10.activity_volume_x_sleep_duration_deviation",
    "r10.activity_phase_x_sleep_phase",
)


def _number(frame: pd.DataFrame, name: str) -> pd.Series:
    return pd.to_numeric(frame[name], errors="coerce").astype(float)


def add_activity_sleep_interactions(frame: pd.DataFrame) -> pd.DataFrame:
    """Add only six frozen interactions that can be generated at runtime."""

    result = frame.copy()
    activity = _number(result, "activity.activity_volume_norm")
    sedentary = _number(result, "activity.sedentary_ratio")
    instability = _number(result, "activity.intradaily_variability")
    sleep_duration = _number(result, "sleep.sleep_duration_norm")
    sleep_efficiency = _number(result, "sleep.sleep_efficiency")
    fragmentation = _number(result, "sleep.sleep_fragmentation")
    irregularity = 1.0 - _number(result, "sleep.sleep_regularity")
    exits = _number(result, "sleep.night_exit_count_mean")
    activity_phase = np.arctan2(
        _number(result, "activity.relative_amplitude"),
        _number(result, "activity.interdaily_stability").replace(0.0, np.nan),
    )
    sleep_phase = np.arctan2(
        _number(result, "sleep.sleep_midpoint_sin"),
        _number(result, "sleep.sleep_midpoint_cos"),
    )
    result[INTERACTION_FEATURES[0]] = (1.0 - activity) * fragmentation
    result[INTERACTION_FEATURES[1]] = sedentary * (1.0 - sleep_efficiency)
    result[INTERACTION_FEATURES[2]] = instability * irregularity
    result[INTERACTION_FEATURES[3]] = (1.0 - activity) * exits
    result[INTERACTION_FEATURES[4]] = activity * np.abs(sleep_duration - 0.5)
    result[INTERACTION_FEATURES[5]] = np.cos(activity_phase - sleep_phase)
    return result


ACTIVITY_SLEEP_BASE_FEATURES = tuple(ACTIVITY_FEATURES) + tuple(SLEEP_FEATURES)
ACTIVITY_SLEEP_INTERACTION_FEATURES = ACTIVITY_SLEEP_BASE_FEATURES + INTERACTION_FEATURES
SLEEP_HISTORY_FEATURES = tuple(SLEEP_FEATURES) + tuple(R9_HISTORY_FEATURES)


def assert_risk_feature_contract(features: Iterable[str]) -> None:
    forbidden_tokens = (
        "participant",
        "dataset",
        "source",
        "path",
        "fold",
        "route",
        "mask",
        "coverage",
        "target",
        "current_phq",
        "future",
    )
    bad = sorted(
        feature
        for feature in features
        if any(token in feature.lower() for token in forbidden_tokens)
    )
    if bad:
        raise ValueError(f"R10 forbidden risk features: {bad}")


for _features in (
    ACTIVITY_SLEEP_BASE_FEATURES,
    ACTIVITY_SLEEP_INTERACTION_FEATURES,
    SLEEP_HISTORY_FEATURES,
):
    assert_risk_feature_contract(_features)


__all__ = [
    "ACTIVITY_SLEEP_BASE_FEATURES",
    "ACTIVITY_SLEEP_INTERACTION_FEATURES",
    "INTERACTION_FEATURES",
    "SLEEP_HISTORY_FEATURES",
    "add_activity_sleep_interactions",
    "assert_risk_feature_contract",
]
