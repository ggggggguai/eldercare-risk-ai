"""Research-only causal PSYCHE features and deployable route signatures."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from elderly_monitoring.modules.mental_health.mood_social.r3.contract import (
    ACTIVITY_MODEL_FEATURES,
    PROFILE_MODEL_FEATURES,
    SLEEP_MODEL_FEATURES,
)
from elderly_monitoring.modules.mental_health.mood_social.r4.contract import (
    PSYCHE_RESEARCH_SENSOR_FEATURES,
    R4_ALLOWED_FEATURES,
    assert_psyche_research_feature_names,
    assert_r4_feature_names,
)


DEPLOYABLE_ACTIVITY_FEATURES = tuple(
    name for name in ACTIVITY_MODEL_FEATURES if name in R4_ALLOWED_FEATURES
)
DEPLOYABLE_SLEEP_FEATURES = tuple(
    name for name in SLEEP_MODEL_FEATURES if name in R4_ALLOWED_FEATURES
)
DEPLOYABLE_PROFILE_FEATURES = tuple(
    name for name in PROFILE_MODEL_FEATURES if name in R4_ALLOWED_FEATURES
)
DEPLOYABLE_JOINT_FEATURES = (
    *DEPLOYABLE_ACTIVITY_FEATURES,
    *DEPLOYABLE_SLEEP_FEATURES,
    *DEPLOYABLE_PROFILE_FEATURES,
)

assert_r4_feature_names(DEPLOYABLE_ACTIVITY_FEATURES)
assert_r4_feature_names(DEPLOYABLE_SLEEP_FEATURES)
assert_r4_feature_names(DEPLOYABLE_PROFILE_FEATURES)
assert_r4_feature_names(DEPLOYABLE_JOINT_FEATURES)


def route_signature(route_pattern: str) -> str:
    route = str(route_pattern)
    if route == "001":
        return "profile"
    if route == "010":
        return "sleep"
    if route in {"101", "111"}:
        return "joint"
    return "unsupported"


def deployable_features_for_signature(signature: str) -> tuple[str, ...]:
    mapping = {
        "activity": DEPLOYABLE_ACTIVITY_FEATURES,
        "sleep": DEPLOYABLE_SLEEP_FEATURES,
        "profile": DEPLOYABLE_PROFILE_FEATURES,
        "joint": DEPLOYABLE_JOINT_FEATURES,
    }
    if signature not in mapping:
        raise ValueError(f"unsupported deployable feature signature: {signature}")
    return mapping[signature]


def _prior_slope(values: pd.Series) -> pd.Series:
    array = pd.to_numeric(values, errors="coerce").to_numpy(float)
    output = np.full(len(array), np.nan, dtype=float)
    for index in range(1, len(array)):
        prior = array[:index]
        valid = np.isfinite(prior)
        if valid.sum() >= 2:
            x = np.arange(index, dtype=float)[valid]
            output[index] = float(np.polyfit(x, prior[valid], 1)[0])
    return pd.Series(output, index=values.index)


def add_psyche_causal_features(
    frame: pd.DataFrame,
    *,
    source_features: Sequence[str] = PSYCHE_RESEARCH_SENSOR_FEATURES,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Derive D-and-prior features in natural month order without label history."""

    assert_psyche_research_feature_names(source_features)
    required = {"global_participant_id", "nominal_month", *source_features}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"PSYCHE causal feature frame missing columns: {missing}")
    result = frame.copy()
    month = pd.to_numeric(result["nominal_month"], errors="coerce")
    if month.isna().any():
        raise ValueError("PSYCHE nominal month is required for causal feature order")
    if result.duplicated(["global_participant_id", "nominal_month"]).any():
        raise ValueError("duplicate PSYCHE participant/month rows")
    result["psyche_causal.month_sin"] = np.sin(2.0 * np.pi * month / 12.0)
    result["psyche_causal.month_cos"] = np.cos(2.0 * np.pi * month / 12.0)
    generated = ["psyche_causal.month_sin", "psyche_causal.month_cos"]
    order = result.sort_values(
        ["global_participant_id", "nominal_month"], kind="stable"
    ).index
    ordered = result.loc[order]
    grouped = ordered.groupby("global_participant_id", sort=False)
    for feature in source_features:
        numeric = pd.to_numeric(ordered[feature], errors="coerce")
        by_person = numeric.groupby(ordered["global_participant_id"], sort=False)
        lag = by_person.shift(1)
        prior_sum = by_person.cumsum() - numeric.fillna(0.0)
        prior_count = numeric.notna().groupby(
            ordered["global_participant_id"], sort=False
        ).cumsum() - numeric.notna().astype(int)
        prior_mean = prior_sum / prior_count.replace(0, np.nan)
        slope = grouped[feature].apply(_prior_slope).reset_index(level=0, drop=True)
        names = {
            "lag1": lag,
            "prior_mean": prior_mean,
            "delta_prior_mean": numeric - prior_mean,
            "prior_slope": slope,
        }
        for suffix, values in names.items():
            name = f"psyche_causal.{feature}_{suffix}"
            result.loc[order, name] = values.reindex(order).to_numpy(float)
            generated.append(name)
    return result, tuple(generated)


__all__ = [
    "DEPLOYABLE_ACTIVITY_FEATURES",
    "DEPLOYABLE_JOINT_FEATURES",
    "DEPLOYABLE_PROFILE_FEATURES",
    "DEPLOYABLE_SLEEP_FEATURES",
    "add_psyche_causal_features",
    "deployable_features_for_signature",
    "route_signature",
]
