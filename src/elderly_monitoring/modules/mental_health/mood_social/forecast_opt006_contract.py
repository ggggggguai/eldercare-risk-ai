"""Frozen contracts and diagnostics for FORECAST-OPT-006.

The module is deliberately model-free.  It contains the field-level definitions
used by 006A and the inner-context aggregation rules used by 006B--006G.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .forecast_opt005_contract import (
    Opt005ContractError,
    assert_unique_keys,
    metric_bundle,
    participant_equal_weights,
)


class Opt006ContractError(Opt005ContractError):
    """Raised when an OPT006 frozen invariant is violated."""


def add_transition_slices(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the exact history-state and evaluation-only slice flags."""
    required = {
        "global_participant_id",
        "task_id",
        "target_window_id",
        "future_binary_target",
        "aux_phq9_score",
        "previous_phq9_score_target_only",
        "previous_phq9_available",
        "history_observed_month_count",
        "feature_nonmissing_count",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise Opt006ContractError(f"slice inputs missing: {missing}")
    result = frame.copy()
    assert_unique_keys(result, ["task_id", "target_window_id"])
    history = pd.to_numeric(
        result["previous_phq9_score_target_only"], errors="coerce"
    )
    future = pd.to_numeric(result["aux_phq9_score"], errors="coerce")
    available = result["previous_phq9_available"].eq(1) & history.notna()
    result["no_history"] = ~available
    result["stable_low"] = available & history.lt(10) & future.lt(10)
    result["new_onset"] = available & history.lt(10) & future.ge(10)
    result["persistent"] = available & history.ge(10) & future.ge(10)
    result["recovery"] = available & history.ge(10) & future.lt(10)
    result["short_history"] = result["history_observed_month_count"].le(2)
    result["low_completeness"] = result["feature_nonmissing_count"].le(19)
    result["boundary_phq"] = future.between(8, 11, inclusive="both")
    mixed = (
        result.groupby(["task_id", "global_participant_id"], sort=False)[
            "future_binary_target"
        ]
        .nunique()
        .ge(2)
        .rename("mixed_label_participant")
    )
    result = result.join(
        mixed, on=["task_id", "global_participant_id"], validate="many_to_one"
    )
    transition_sum = result[
        ["stable_low", "new_onset", "persistent", "recovery"]
    ].sum(axis=1)
    if not transition_sum[result["previous_phq9_available"].eq(1)].eq(1).all():
        raise Opt006ContractError("history-available transition is not exhaustive")
    return result


def weighted_rank_correlation(
    x: Sequence[float], y: Sequence[float], weights: Sequence[float]
) -> float:
    """Participant-equal Spearman: weighted Pearson correlation of average ranks."""
    x_values = np.asarray(x, dtype="float64")
    y_values = np.asarray(y, dtype="float64")
    w = np.asarray(weights, dtype="float64")
    valid = np.isfinite(x_values) & np.isfinite(y_values) & np.isfinite(w) & (w > 0)
    if valid.sum() < 2:
        return float("nan")
    rx = rankdata(x_values[valid], method="average")
    ry = rankdata(y_values[valid], method="average")
    w = w[valid] / w[valid].sum()
    mx = float(np.sum(w * rx))
    my = float(np.sum(w * ry))
    covariance = float(np.sum(w * (rx - mx) * (ry - my)))
    denominator = float(
        np.sqrt(np.sum(w * np.square(rx - mx)) * np.sum(w * np.square(ry - my)))
    )
    return float("nan") if denominator <= 0 else covariance / denominator


def score_change_diagnostics(frame: pd.DataFrame, score_column: str) -> dict[str, Any]:
    """Compute adjacent-window direction diagnostics within participant/horizon."""
    required = {
        "global_participant_id",
        "task_id",
        "feature_month_slot",
        "future_binary_target",
        "aux_phq9_score",
        score_column,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise Opt006ContractError(f"change inputs missing: {missing}")
    ordered = frame.sort_values(
        ["task_id", "global_participant_id", "feature_month_slot"]
    ).copy()
    groups = ordered.groupby(["task_id", "global_participant_id"], sort=False)
    ordered["previous_label"] = groups["future_binary_target"].shift(1)
    ordered["score_delta"] = groups[score_column].diff()
    ordered["phq_delta"] = groups["aux_phq9_score"].diff()
    adjacent = ordered[ordered["previous_label"].notna()].copy()
    adjacent["transition"] = (
        adjacent["previous_label"].astype(int).astype(str)
        + "->"
        + adjacent["future_binary_target"].astype(int).astype(str)
    )
    weights = participant_equal_weights(adjacent)
    return {
        "adjacent_rows": int(len(adjacent)),
        "mean_score_delta_0_to_1": float(
            adjacent.loc[adjacent["transition"].eq("0->1"), "score_delta"].mean()
        ),
        "mean_score_delta_1_to_0": float(
            adjacent.loc[adjacent["transition"].eq("1->0"), "score_delta"].mean()
        ),
        "score_delta_phq_delta_spearman_participant_equal": weighted_rank_correlation(
            adjacent["score_delta"], adjacent["phq_delta"], weights
        ),
    }


def inner_context_summary(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Apply the frozen five-context arithmetic aggregation and hard gates."""
    rows = list(records)
    if len(rows) != 5 or {int(row["evaluation_outer_fold"]) for row in rows} != set(
        range(5)
    ):
        raise Opt006ContractError("exactly five distinct inner contexts are required")
    deltas = np.asarray([row["auprc_delta"] for row in rows], dtype="float64")
    brier = np.asarray([row["brier_delta"] for row in rows], dtype="float64")
    ece = np.asarray([row["ece_delta"] for row in rows], dtype="float64")
    coverage = np.asarray([row.get("coverage_delta", 0.0) for row in rows])
    aggregate = float(deltas.mean())
    checks = {
        "aggregate_auprc_delta_ge_0_003": aggregate >= 0.003 - 1e-12,
        "contexts_non_decrease_ge_4": int((deltas >= -1e-12).sum()) >= 4,
        "worst_context_drop_within_0_015": float(deltas.min()) >= -0.015 - 1e-12,
        "brier_increase_within_0_002": float(brier.max()) <= 0.002 + 1e-12,
        "ece_increase_within_0_002": float(ece.max()) <= 0.002 + 1e-12,
        "coverage_not_reduced": bool((coverage >= -1e-12).all()),
    }
    return {
        "aggregate_auprc_delta": aggregate,
        "independent_outer_candidate": aggregate >= 0.005 - 1e-12
        and all(checks.values()),
        "exploration_gate_passed": all(checks.values()),
        "positive_or_equal_contexts": int((deltas >= -1e-12).sum()),
        "worst_context_delta": float(deltas.min()),
        "checks": checks,
        "all_passed": all(checks.values()),
    }


def metric_delta(
    frame: pd.DataFrame,
    candidate_probability: Sequence[float],
    reference_probability: Sequence[float],
    *,
    candidate_ranking: Sequence[float] | None = None,
    reference_ranking: Sequence[float] | None = None,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    candidate = metric_bundle(
        frame, candidate_probability, ranking_score=candidate_ranking
    )
    reference = metric_bundle(
        frame, reference_probability, ranking_score=reference_ranking
    )
    delta = {name: candidate[name] - reference[name] for name in candidate}
    return candidate, reference, delta


__all__ = [
    "Opt006ContractError",
    "add_transition_slices",
    "inner_context_summary",
    "metric_delta",
    "score_change_diagnostics",
    "weighted_rank_correlation",
]
