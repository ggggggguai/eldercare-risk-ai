"""Pairwise ranking and non-negative fusion helpers for OPT-004D."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score, roc_auc_score


PAIR_SCOPES = ("same_label_month", "global_inner_train")
MAX_OPPOSITE_PAIRS_PER_ANCHOR = 32
COMPONENTS = ("pointwise", "ordinal", "expert", "ranking")
FUSION_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


def participant_equal_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = (
        frame.groupby("global_participant_id", sort=False)["global_participant_id"]
        .transform("size")
        .to_numpy(dtype="float64")
    )
    return 1.0 / (float(frame["global_participant_id"].nunique()) * counts)


def select_ranking_features(columns: Sequence[str]) -> tuple[str, ...]:
    direct = {
        "feature_nonmissing_count",
        "history_observed_month_count",
        "history_month_span",
        "history_months_since_last_any_record",
        "baseline_available_count",
        "cold_start_field_count",
    }
    suffixes = (
        "__anchor",
        "__delta_1",
        "__local_slope",
        "__personal_z",
    )
    result = [
        name
        for name in columns
        if name in direct
        or (name.startswith("short__") and name.endswith(suffixes))
        or name.startswith("opt003__")
    ]
    prohibited = ("phq", "target", "future", "participant_id", "fold_id")
    result = [
        name
        for name in result
        if not any(token in name.lower() for token in prohibited)
    ]
    if not result:
        raise ValueError("no ranking features selected")
    return tuple(result)


@dataclass
class FoldStandardizer:
    feature_names: tuple[str, ...]
    medians: np.ndarray | None = None
    means: np.ndarray | None = None
    scales: np.ndarray | None = None

    def fit(self, frame: pd.DataFrame) -> "FoldStandardizer":
        matrix = (
            frame.loc[:, list(self.feature_names)]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype="float64")
        )
        matrix[~np.isfinite(matrix)] = np.nan
        medians = np.zeros(matrix.shape[1], dtype="float64")
        for column in range(matrix.shape[1]):
            finite = matrix[:, column][np.isfinite(matrix[:, column])]
            medians[column] = float(np.median(finite)) if finite.size else 0.0
        filled = np.where(np.isfinite(matrix), matrix, medians)
        means = filled.mean(axis=0)
        scales = filled.std(axis=0)
        scales[~np.isfinite(scales) | (scales == 0)] = 1.0
        self.medians = medians
        self.means = means
        self.scales = scales
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.medians is None or self.means is None or self.scales is None:
            raise RuntimeError("standardizer is not fitted")
        matrix = (
            frame.loc[:, list(self.feature_names)]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype="float64")
        )
        matrix[~np.isfinite(matrix)] = np.nan
        filled = np.where(np.isfinite(matrix), matrix, self.medians)
        return np.asarray((filled - self.means) / self.scales, dtype="float32")


def _group_indices(frame: pd.DataFrame, scope: str) -> list[np.ndarray]:
    if scope == "global_inner_train":
        return [np.arange(len(frame), dtype="int64")]
    if scope == "same_label_month":
        return [
            np.asarray(indices, dtype="int64")
            for indices in frame.groupby("label_month_slot", sort=True).indices.values()
        ]
    raise ValueError(f"unsupported pair scope: {scope}")


def build_pair_differences(
    frame: pd.DataFrame,
    matrix: np.ndarray,
    scope: str,
    seed: int,
    max_opposite_pairs_per_anchor: int = MAX_OPPOSITE_PAIRS_PER_ANCHOR,
) -> tuple[np.ndarray, np.ndarray, dict[str, int | str]]:
    """Sample bounded positive-anchor pairs and add symmetric reversals."""

    if len(frame) != len(matrix):
        raise ValueError("frame/matrix row mismatch")
    rng = np.random.default_rng(seed)
    labels = frame["future_binary_target"].to_numpy(dtype="int8")
    participants = frame["global_participant_id"].astype(str).to_numpy()
    differences = []
    semantic_pairs = 0
    anchors_used = 0
    anchors_without_opposite = 0
    max_sampled = 0
    for group in _group_indices(frame.reset_index(drop=True), scope):
        positives = group[labels[group] == 1]
        negatives = group[labels[group] == 0]
        for positive in positives:
            eligible = negatives[participants[negatives] != participants[positive]]
            if eligible.size == 0:
                anchors_without_opposite += 1
                continue
            size = min(max_opposite_pairs_per_anchor, int(eligible.size))
            chosen = rng.choice(eligible, size=size, replace=False)
            differences.append(matrix[positive] - matrix[chosen])
            semantic_pairs += size
            anchors_used += 1
            max_sampled = max(max_sampled, size)
    if not differences:
        raise ValueError("pair construction produced no pairs")
    positive_difference = np.concatenate(differences, axis=0).astype("float32")
    pair_matrix = np.concatenate([positive_difference, -positive_difference], axis=0)
    pair_label = np.concatenate(
        [
            np.ones(len(positive_difference), dtype="int8"),
            np.zeros(len(positive_difference), dtype="int8"),
        ]
    )
    return (
        pair_matrix,
        pair_label,
        {
            "scope": scope,
            "seed": int(seed),
            "semantic_pair_count": int(semantic_pairs),
            "symmetric_training_rows": int(len(pair_label)),
            "positive_anchors_used": int(anchors_used),
            "positive_anchors_without_opposite": int(anchors_without_opposite),
            "maximum_opposites_sampled_per_anchor": int(max_sampled),
        },
    )


def fit_predict_pairwise(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_names: Sequence[str],
    scope: str,
    seeds: Sequence[int],
    max_opposite_pairs_per_anchor: int = MAX_OPPOSITE_PAIRS_PER_ANCHOR,
) -> tuple[np.ndarray, list[dict[str, int | str]]]:
    standardizer = FoldStandardizer(tuple(feature_names)).fit(train)
    x_train = standardizer.transform(train)
    x_validation = standardizer.transform(validation)
    scores = []
    audits = []
    for seed in seeds:
        pair_matrix, pair_label, audit = build_pair_differences(
            train.reset_index(drop=True),
            x_train,
            scope,
            int(seed),
            max_opposite_pairs_per_anchor=max_opposite_pairs_per_anchor,
        )
        estimator = SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            alpha=0.0005,
            l1_ratio=0.05,
            max_iter=400,
            tol=1e-4,
            random_state=int(seed),
            average=True,
        )
        estimator.fit(pair_matrix, pair_label)
        scores.append(estimator.decision_function(x_validation))
        audits.append(audit)
    return np.mean(scores, axis=0), audits


def percentile_rank_by_fold(values: np.ndarray, inner_fold: np.ndarray) -> np.ndarray:
    result = np.empty(len(values), dtype="float64")
    for fold in sorted(np.unique(inner_fold)):
        mask = inner_fold == fold
        result[mask] = (
            pd.Series(values[mask]).rank(method="average", pct=True).to_numpy()
        )
    return result


def fusion_weight_candidates() -> list[dict[str, float]]:
    candidates = []
    for values in product(FUSION_GRID, repeat=len(COMPONENTS)):
        if abs(sum(values) - 1.0) > 1e-12:
            continue
        candidates.append(dict(zip(COMPONENTS, map(float, values), strict=True)))
    return candidates


def fuse(component_frame: pd.DataFrame, weights: dict[str, float]) -> np.ndarray:
    result = np.zeros(len(component_frame), dtype="float64")
    for name in COMPONENTS:
        result += weights[name] * component_frame[f"rank_{name}"].to_numpy(
            dtype="float64"
        )
    return result


def ranking_metrics(frame: pd.DataFrame, score: np.ndarray) -> dict[str, float]:
    y = frame["future_binary_target"].to_numpy(dtype="int8")
    weights = participant_equal_weights(frame)
    weights /= weights.sum()
    return {
        "auprc": float(average_precision_score(y, score, sample_weight=weights)),
        "auroc": float(roc_auc_score(y, score, sample_weight=weights)),
    }
