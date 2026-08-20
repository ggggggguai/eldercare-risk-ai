"""Historical PHQ-9 trajectory features for FORECAST-OPT-005D."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .forecast_state_aware import (
    HISTORY_FEATURES,
    build_history_features,
    prepare_source_index,
)

HALF_LIVES = (1, 2, 3, 6)


def build_trajectory_features(
    samples: pd.DataFrame, source: pd.DataFrame
) -> pd.DataFrame:
    """Build D0/D1 plus frozen exponential trajectory features."""

    base = build_history_features(samples, source)
    indexed = prepare_source_index(source)
    assessments = indexed.loc[
        indexed["phq9_score_end"].notna(),
        ["__participant_id", "__nominal_month", "phq9_score_end"],
    ].copy()
    assessments["__participant_id"] = assessments["__participant_id"].astype(str)
    history_map = {
        participant: rows.sort_values("__nominal_month")
        for participant, rows in assessments.groupby("__participant_id", sort=False)
    }
    extra_rows = []
    for sample in samples.itertuples(index=False):
        participant = str(sample.global_participant_id).split("::", 1)[-1]
        cutoff = int(sample.feature_month_slot)
        history = history_map.get(participant)
        eligible = (
            history[history["__nominal_month"] < cutoff]
            if history is not None
            else pd.DataFrame(columns=assessments.columns)
        )
        slots = eligible["__nominal_month"].to_numpy(dtype="float64")
        scores = eligible["phq9_score_end"].to_numpy(dtype="float64")
        row: dict[str, float | str] = {
            "global_participant_id": sample.global_participant_id,
            "target_window_id": sample.target_window_id,
        }
        for half_life in HALF_LIVES:
            prefix = f"trajectory_h{half_life}"
            if len(scores) == 0:
                row[f"{prefix}_weighted_mean"] = np.nan
                row[f"{prefix}_weighted_slope"] = np.nan
                row[f"{prefix}_effective_weight"] = 0.0
                row[f"{prefix}_recent3_trend"] = np.nan
                continue
            ages = cutoff - slots
            weights = np.exp(-np.log(2.0) * ages / float(half_life))
            total = float(weights.sum())
            row[f"{prefix}_weighted_mean"] = float(np.sum(weights * scores) / total)
            row[f"{prefix}_effective_weight"] = total
            if len(scores) >= 2 and np.unique(slots).size >= 2:
                center = float(np.sum(weights * slots) / total)
                score_center = float(np.sum(weights * scores) / total)
                denominator = float(np.sum(weights * np.square(slots - center)))
                row[f"{prefix}_weighted_slope"] = (
                    float(
                        np.sum(weights * (slots - center) * (scores - score_center))
                        / denominator
                    )
                    if denominator > 0
                    else 0.0
                )
            else:
                row[f"{prefix}_weighted_slope"] = 0.0
            recent_slots = slots[-3:]
            recent_scores = scores[-3:]
            row[f"{prefix}_recent3_trend"] = (
                float(np.polyfit(recent_slots, recent_scores, 1)[0])
                if len(recent_scores) >= 2 and np.unique(recent_slots).size >= 2
                else 0.0
            )
        extra_rows.append(row)
    extra = pd.DataFrame(extra_rows)
    return base.merge(
        extra,
        on=["global_participant_id", "target_window_id"],
        how="left",
        validate="one_to_one",
    )


def trajectory_feature_names(half_life: int) -> tuple[str, ...]:
    if half_life not in HALF_LIVES:
        raise ValueError(f"unsupported half-life: {half_life}")
    prefix = f"trajectory_h{half_life}"
    return (
        f"{prefix}_weighted_mean",
        f"{prefix}_weighted_slope",
        f"{prefix}_effective_weight",
        f"{prefix}_recent3_trend",
    )


def reliability(
    available: Iterable[float],
    count: Iterable[float],
    age_months: Iterable[float],
    half_life: float,
) -> np.ndarray:
    availability = np.nan_to_num(np.asarray(list(available), dtype="float64"))
    history_count = np.nan_to_num(np.asarray(list(count), dtype="float64"))
    age = np.nan_to_num(np.asarray(list(age_months), dtype="float64"), nan=np.inf)
    value = (
        availability
        * np.minimum(history_count / 3.0, 1.0)
        * np.exp(-np.log(2.0) * age / float(half_life))
    )
    return np.clip(np.nan_to_num(value), 0.0, 1.0)


def gated_probability(
    passive_probability: Iterable[float],
    state_probability: Iterable[float],
    confidence: Iterable[float],
    beta: float,
) -> np.ndarray:
    passive = np.clip(
        np.asarray(list(passive_probability), dtype="float64"), 1e-6, 1 - 1e-6
    )
    state = np.clip(
        np.asarray(list(state_probability), dtype="float64"), 1e-6, 1 - 1e-6
    )
    weight = np.clip(beta * np.asarray(list(confidence), dtype="float64"), 0.0, 1.0)
    passive_logit = np.log(passive) - np.log1p(-passive)
    state_logit = np.log(state) - np.log1p(-state)
    logit = passive_logit + weight * (state_logit - passive_logit)
    result = 1.0 / (1.0 + np.exp(-logit))
    return np.clip(result, 1e-6, 1 - 1e-6)


D0_HISTORY_FEATURES = (
    "historical_phq9_last_score",
    "historical_phq9_age_months",
    "historical_assessment_available",
)

__all__ = [
    "D0_HISTORY_FEATURES",
    "HALF_LIVES",
    "HISTORY_FEATURES",
    "build_trajectory_features",
    "gated_probability",
    "reliability",
    "trajectory_feature_names",
]
