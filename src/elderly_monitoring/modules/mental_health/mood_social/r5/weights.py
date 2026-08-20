"""Capped pre-registered training weights for OPT-V333-R5-003."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from elderly_monitoring.modules.mental_health.mood_social.r4.selection import (
    participant_equal_weights,
)


WeightScheme = Literal[
    "participant_equal",
    "threshold_near_mild_downweight",
    "extreme_severity_mild_strengthen",
]
R5_WEIGHT_SCHEMES: tuple[WeightScheme, ...] = (
    "participant_equal",
    "threshold_near_mild_downweight",
    "extreme_severity_mild_strengthen",
)


def r5_training_weights(frame: pd.DataFrame, scheme: WeightScheme) -> np.ndarray:
    """Return positive, participant-normalized weights with mild capped modifiers."""

    if scheme not in R5_WEIGHT_SCHEMES:
        raise ValueError(f"unregistered r5 weight scheme: {scheme}")
    if "global_participant_id" not in frame or "phq9_score_r3_target" not in frame:
        raise ValueError("r5 weights require participant and target PHQ score columns")
    base = participant_equal_weights(frame)
    score = pd.to_numeric(frame["phq9_score_r3_target"], errors="coerce").to_numpy(float)
    if not np.isfinite(score).all():
        raise ValueError("r5 training PHQ supervision must be finite")
    modifier = np.ones(len(frame), dtype=float)
    if scheme == "threshold_near_mild_downweight":
        modifier[np.isin(score, [4.0, 5.0, 9.0, 10.0])] = 0.85
    elif scheme == "extreme_severity_mild_strengthen":
        modifier[(score <= 3.0) | (score >= 15.0)] = 1.20
    weight = base * modifier
    # Re-normalize within participant so repeated PSYCHE windows never gain
    # more total mass solely because the participant has more observations.
    totals = pd.Series(weight, index=frame.index).groupby(
        frame["global_participant_id"], sort=False
    ).transform("sum")
    weight = weight / totals.to_numpy(float)
    weight = weight * len(weight) / weight.sum()
    if (
        weight.shape != (len(frame),)
        or not np.isfinite(weight).all()
        or (weight <= 0).any()
        or weight.max() / weight.min() > 20.0
    ):
        raise ValueError("r5 capped training weights are invalid")
    return weight


__all__ = ["R5_WEIGHT_SCHEMES", "WeightScheme", "r5_training_weights"]
