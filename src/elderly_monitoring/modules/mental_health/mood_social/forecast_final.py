"""Frozen evaluation helpers for FORECAST-OPT-004H.

These functions implement the exact participant-equal metrics, inner-only
operating-point selection, empirical rank mapping, and paired participant
bootstrap used by the final offline release.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score


def participant_equal_weights(frame: pd.DataFrame) -> np.ndarray:
    if frame.empty:
        return np.empty(0, dtype="float64")
    counts = (
        frame.groupby("global_participant_id", sort=False)["global_participant_id"]
        .transform("size")
        .to_numpy(dtype="float64")
    )
    return 1.0 / (float(frame["global_participant_id"].nunique()) * counts)


def empirical_percentile(
    reference: Sequence[float], values: Sequence[float]
) -> np.ndarray:
    """Apply pandas-compatible average percentile ranks using a frozen reference."""

    reference_array = np.sort(np.asarray(reference, dtype="float64"))
    values_array = np.asarray(values, dtype="float64")
    if reference_array.size == 0 or not np.isfinite(reference_array).all():
        raise ValueError("empirical percentile reference must be finite and nonempty")
    left = np.searchsorted(reference_array, values_array, side="left")
    right = np.searchsorted(reference_array, values_array, side="right")
    return (left + right + 1.0) / (2.0 * reference_array.size)


def cross_fitted_percentile(
    reference_values: Sequence[float],
    reference_fold: Sequence[int],
    values: Sequence[float],
) -> np.ndarray:
    """Average five frozen inner-validation empirical CDF mappings."""

    reference_values = np.asarray(reference_values, dtype="float64")
    reference_fold = np.asarray(reference_fold)
    predictions = []
    for fold in sorted(np.unique(reference_fold)):
        predictions.append(
            empirical_percentile(reference_values[reference_fold == fold], values)
        )
    if len(predictions) != 5:
        raise ValueError("cross-fitted percentile requires exactly five inner folds")
    return np.mean(predictions, axis=0)


def fit_platt(
    frame: pd.DataFrame, score_column: str
) -> tuple[LogisticRegression, dict[str, float]]:
    score = frame[score_column].to_numpy(dtype="float64")
    label = frame["future_binary_target"].to_numpy(dtype="int8")
    if np.unique(label).size != 2:
        raise ValueError("calibration data lacks a binary class")
    estimator = LogisticRegression(C=1_000_000.0, solver="lbfgs", max_iter=2000)
    estimator.fit(
        score.reshape(-1, 1),
        label,
        sample_weight=participant_equal_weights(frame),
    )
    return estimator, {
        "method": "platt_logistic_fixed",
        "coefficient": float(estimator.coef_[0, 0]),
        "intercept": float(estimator.intercept_[0]),
        "inner_rows": len(frame),
        "inner_participants": int(frame["global_participant_id"].nunique()),
    }


def apply_platt(estimator: LogisticRegression, score: Sequence[float]) -> np.ndarray:
    return estimator.predict_proba(np.asarray(score).reshape(-1, 1))[:, 1]


def weighted_ece(
    label: Sequence[int], probability: Sequence[float], weights: Sequence[float]
) -> float:
    label = np.asarray(label, dtype="int8")
    probability = np.clip(np.asarray(probability, dtype="float64"), 0.0, 1.0)
    weights = np.asarray(weights, dtype="float64")
    weights = weights / weights.sum()
    bins = np.minimum((probability * 10).astype(int), 9)
    result = 0.0
    for bin_id in range(10):
        mask = bins == bin_id
        if mask.any():
            mass = float(weights[mask].sum())
            result += mass * abs(
                float(np.average(probability[mask], weights=weights[mask]))
                - float(np.average(label[mask], weights=weights[mask]))
            )
    return float(result)


def confusion_metrics(
    label: Sequence[int], prediction: Sequence[bool], weights: Sequence[float]
) -> dict[str, float]:
    label = np.asarray(label, dtype="int8")
    prediction = np.asarray(prediction, dtype=bool)
    weights = np.asarray(weights, dtype="float64")
    weights = weights / weights.sum()
    tp = float(weights[(label == 1) & prediction].sum())
    fn = float(weights[(label == 1) & ~prediction].sum())
    tn = float(weights[(label == 0) & ~prediction].sum())
    fp = float(weights[(label == 0) & prediction].sum())
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    positive_f1 = 2.0 * tp / (2.0 * tp + fp + fn) if 2.0 * tp + fp + fn else 0.0
    negative_f1 = 2.0 * tn / (2.0 * tn + fp + fn) if 2.0 * tn + fp + fn else 0.0
    return {
        "macro_f1": 0.5 * (positive_f1 + negative_f1),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "tp_weight": tp,
        "fn_weight": fn,
        "tn_weight": tn,
        "fp_weight": fp,
    }


def _threshold_table(
    label: np.ndarray, probability: np.ndarray, weights: np.ndarray
) -> pd.DataFrame:
    order = np.argsort(probability, kind="mergesort")
    score = probability[order]
    y = label[order]
    w = weights[order]
    thresholds = np.unique(np.concatenate(([0.0, 1.0], score)))
    positive_weight = w * (y == 1)
    negative_weight = w * (y == 0)
    positive_prefix = np.concatenate(([0.0], np.cumsum(positive_weight)))
    negative_prefix = np.concatenate(([0.0], np.cumsum(negative_weight)))
    positions = np.searchsorted(score, thresholds, side="left")
    tp = positive_prefix[-1] - positive_prefix[positions]
    fp = negative_prefix[-1] - negative_prefix[positions]
    fn = positive_prefix[positions]
    tn = negative_prefix[positions]
    sensitivity = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=tp + fn > 0)
    specificity = np.divide(tn, tn + fp, out=np.zeros_like(tn), where=tn + fp > 0)
    f1_positive = np.divide(
        2 * tp,
        2 * tp + fp + fn,
        out=np.zeros_like(tp),
        where=2 * tp + fp + fn > 0,
    )
    f1_negative = np.divide(
        2 * tn,
        2 * tn + fp + fn,
        out=np.zeros_like(tn),
        where=2 * tn + fp + fn > 0,
    )
    return pd.DataFrame(
        {
            "threshold": thresholds,
            "macro_f1": 0.5 * (f1_positive + f1_negative),
            "sensitivity": sensitivity,
            "specificity": specificity,
        }
    )


def select_operating_points(
    frame: pd.DataFrame, probability_column: str
) -> dict[str, Any]:
    label = frame["future_binary_target"].to_numpy(dtype="int8")
    probability = frame[probability_column].to_numpy(dtype="float64")
    table = _threshold_table(label, probability, participant_equal_weights(frame))
    best_macro = float(table["macro_f1"].max())
    winner = table[np.isclose(table["macro_f1"], best_macro, atol=1e-15)]
    best_sensitivity = float(winner["sensitivity"].max())
    winner = winner[
        np.isclose(winner["sensitivity"], best_sensitivity, atol=1e-15)
    ].sort_values("threshold", ascending=True)
    macro = winner.iloc[0]
    fixed: dict[str, dict[str, float]] = {}
    for target in (0.8, 0.9):
        eligible = table[table["specificity"] >= target - 1e-15]
        best_recall = float(eligible["sensitivity"].max())
        eligible = eligible[
            np.isclose(eligible["sensitivity"], best_recall, atol=1e-15)
        ]
        best_specificity = float(eligible["specificity"].max())
        eligible = eligible[
            np.isclose(eligible["specificity"], best_specificity, atol=1e-15)
        ].sort_values("threshold", ascending=False)
        selected = eligible.iloc[0]
        fixed[f"specificity_{target:.1f}"] = {
            "target": target,
            "threshold": float(selected["threshold"]),
            "inner_recall": float(selected["sensitivity"]),
            "inner_specificity": float(selected["specificity"]),
        }
    return {
        "macro_f1": {
            "threshold": float(macro["threshold"]),
            "inner_macro_f1": float(macro["macro_f1"]),
            "inner_sensitivity": float(macro["sensitivity"]),
            "inner_specificity": float(macro["specificity"]),
        },
        "fixed_specificity": fixed,
    }


def binary_metrics(
    frame: pd.DataFrame,
    ranking_column: str,
    probability_column: str,
    threshold_column: str,
) -> dict[str, float]:
    label = frame["future_binary_target"].to_numpy(dtype="int8")
    ranking = frame[ranking_column].to_numpy(dtype="float64")
    probability = frame[probability_column].to_numpy(dtype="float64")
    threshold = frame[threshold_column].to_numpy(dtype="float64")
    weights = participant_equal_weights(frame)
    weights /= weights.sum()
    confusion = confusion_metrics(label, probability >= threshold, weights)
    return {
        "rows": len(frame),
        "participants": int(frame["global_participant_id"].nunique()),
        "positive_rate": float(np.sum(weights * label)),
        "auprc": float(average_precision_score(label, ranking, sample_weight=weights)),
        "auroc": float(roc_auc_score(label, ranking, sample_weight=weights)),
        "brier": float(np.sum(weights * (probability - label) ** 2)),
        "ece": weighted_ece(label, probability, weights),
        **{name: float(value) for name, value in confusion.items()},
    }


def fixed_specificity_metrics(
    frame: pd.DataFrame, probability_column: str, threshold_column: str
) -> dict[str, float]:
    return confusion_metrics(
        frame["future_binary_target"].to_numpy(dtype="int8"),
        frame[probability_column].to_numpy(dtype="float64")
        >= frame[threshold_column].to_numpy(dtype="float64"),
        participant_equal_weights(frame),
    )


def top_k_metrics(
    frame: pd.DataFrame, ranking_column: str, fractions: Sequence[float]
) -> list[dict[str, Any]]:
    weights = participant_equal_weights(frame)
    work = frame[
        ["label_month_slot", "target_window_id", "future_binary_target", ranking_column]
    ].copy()
    work["evaluation_weight"] = weights
    output = []
    for fraction in fractions:
        selected_parts = []
        monthly_counts = []
        for month, group in work.groupby("label_month_slot", sort=True):
            count = int(np.ceil(float(fraction) * len(group)))
            selected = group.sort_values(
                [ranking_column, "target_window_id"],
                ascending=[False, True],
                kind="mergesort",
            ).head(count)
            selected_parts.append(selected)
            monthly_counts.append(
                {"label_month_slot": int(month), "rows": len(group), "selected": count}
            )
        selected = pd.concat(selected_parts, ignore_index=True)
        selected_positive = float(
            (selected["evaluation_weight"] * selected["future_binary_target"]).sum()
        )
        output.append(
            {
                "q": float(fraction),
                "selected_rows": len(selected),
                "precision": selected_positive
                / float(selected["evaluation_weight"].sum()),
                "recall": selected_positive
                / float(
                    (work["evaluation_weight"] * work["future_binary_target"]).sum()
                ),
                "monthly_counts": monthly_counts,
            }
        )
    return output


def paired_participant_bootstrap(
    frame: pd.DataFrame,
    candidate_score: str,
    baseline_score: str,
    repetitions: int = 2000,
    seed: int = 20260728,
) -> pd.DataFrame:
    participants = np.asarray(
        sorted(frame["global_participant_id"].astype(str).unique())
    )
    participant_index = {
        participant: index for index, participant in enumerate(participants)
    }
    row_participant = (
        frame["global_participant_id"].astype(str).map(participant_index).to_numpy()
    )
    row_counts = np.bincount(row_participant, minlength=len(participants)).astype(
        "float64"
    )
    label = frame["future_binary_target"].to_numpy(dtype="int8")
    candidate = frame[candidate_score].to_numpy(dtype="float64")
    baseline = frame[baseline_score].to_numpy(dtype="float64")
    rng = np.random.default_rng(seed)
    rows = []
    for replicate in range(repetitions):
        drawn = rng.integers(0, len(participants), size=len(participants))
        multiplicity = np.bincount(drawn, minlength=len(participants)).astype("float64")
        weights = multiplicity[row_participant] / (
            float(len(participants)) * row_counts[row_participant]
        )
        if weights[label == 0].sum() <= 0 or weights[label == 1].sum() <= 0:
            rows.append({"replicate": replicate, "valid": False})
            continue
        candidate_ap = float(
            average_precision_score(label, candidate, sample_weight=weights)
        )
        baseline_ap = float(
            average_precision_score(label, baseline, sample_weight=weights)
        )
        rows.append(
            {
                "replicate": replicate,
                "valid": True,
                "candidate_auprc": candidate_ap,
                "baseline_auprc": baseline_ap,
                "auprc_delta": candidate_ap - baseline_ap,
            }
        )
    return pd.DataFrame(rows)


def bootstrap_summary(frame: pd.DataFrame) -> dict[str, Any]:
    valid = frame[frame["valid"]].copy()
    return {
        "repetitions": len(frame),
        "valid_repetitions": len(valid),
        "degenerate_repetitions": int((~frame["valid"]).sum()),
        "candidate_auprc_ci": [
            float(valid["candidate_auprc"].quantile(0.025)),
            float(valid["candidate_auprc"].quantile(0.975)),
        ],
        "baseline_auprc_ci": [
            float(valid["baseline_auprc"].quantile(0.025)),
            float(valid["baseline_auprc"].quantile(0.975)),
        ],
        "auprc_delta_median": float(valid["auprc_delta"].median()),
        "auprc_delta_ci": [
            float(valid["auprc_delta"].quantile(0.025)),
            float(valid["auprc_delta"].quantile(0.975)),
        ],
    }
