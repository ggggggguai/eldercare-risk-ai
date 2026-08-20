"""Leakage-safe ordinal PHQ-9 heads for FORECAST-OPT-004B.

This module is deliberately independent from the online mood-social package.
It only fits fold-local models and never knows about outer-test rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    cohen_kappa_score,
    mean_absolute_error,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

try:
    from catboost import CatBoostRegressor
except ImportError:  # pragma: no cover - guarded at runtime
    CatBoostRegressor = None  # type: ignore[assignment,misc]

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:  # pragma: no cover - guarded at runtime
    LGBMClassifier = None  # type: ignore[assignment,misc]
    LGBMRegressor = None  # type: ignore[assignment,misc]


ORDINAL_BOUNDS = ((0, 4), (5, 9), (10, 14), (15, 19), (20, 27))
CUMULATIVE_THRESHOLDS = (5.0, 10.0, 15.0, 20.0)
MODEL_FAMILIES = (
    "cumulative_logistic",
    "lightgbm_cumulative_thresholds",
    "catboost_regression",
    "lightgbm_regression_tail_mapping",
)


def ordinal_target(score: Sequence[float]) -> np.ndarray:
    """Map PHQ-9 score 0..27 to frozen five-level ordinal bins."""

    values = np.asarray(score, dtype="float64")
    if np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 27)):
        raise ValueError("PHQ-9 scores must be finite and in [0, 27]")
    return np.digitize(values, bins=[5.0, 10.0, 15.0, 20.0]).astype("int8")


def participant_equal_weights(frame: pd.DataFrame) -> np.ndarray:
    """Give every participant equal total mass."""

    if frame.empty:
        return np.empty(0, dtype="float64")
    counts = (
        frame.groupby("global_participant_id", sort=False)["global_participant_id"]
        .transform("size")
        .to_numpy(dtype="float64")
    )
    return 1.0 / (float(frame["global_participant_id"].nunique()) * counts)


def participant_threshold_weights(frame: pd.DataFrame, threshold: float) -> np.ndarray:
    """Participant- and class-balanced weights for a cumulative threshold."""

    labels = (frame["aux_phq9_score"].to_numpy(dtype="float64") >= threshold).astype(
        "int8"
    )
    if np.unique(labels).size != 2:
        raise ValueError(
            f"training fold lacks both classes for threshold {threshold:g}"
        )
    work = frame[["global_participant_id"]].copy()
    work["threshold_label"] = labels
    class_participants = work.groupby("threshold_label")[
        "global_participant_id"
    ].nunique()
    row_counts = (
        work.groupby(["global_participant_id", "threshold_label"], sort=False)[
            "threshold_label"
        ]
        .transform("size")
        .to_numpy(dtype="float64")
    )
    participant_counts = np.asarray(
        [float(class_participants[int(label)]) for label in labels], dtype="float64"
    )
    return 0.5 / (participant_counts * row_counts)


def numeric_matrix(frame: pd.DataFrame, feature_names: Sequence[str]) -> np.ndarray:
    values = frame.loc[:, list(feature_names)].apply(pd.to_numeric, errors="coerce")
    matrix = values.to_numpy(dtype="float64", copy=True)
    matrix[~np.isfinite(matrix)] = np.nan
    return matrix


@dataclass
class FoldPreprocessor:
    """Training-fold-only median fill and standardization."""

    feature_names: tuple[str, ...]
    medians: np.ndarray | None = None
    means: np.ndarray | None = None
    scales: np.ndarray | None = None

    def fit(self, frame: pd.DataFrame) -> "FoldPreprocessor":
        matrix = numeric_matrix(frame, self.feature_names)
        medians = np.zeros(matrix.shape[1], dtype="float64")
        for column in range(matrix.shape[1]):
            finite = matrix[:, column][np.isfinite(matrix[:, column])]
            medians[column] = float(np.median(finite)) if finite.size else 0.0
        filled = np.where(np.isfinite(matrix), matrix, medians)
        scaler = StandardScaler().fit(filled)
        self.medians = medians
        self.means = np.asarray(scaler.mean_, dtype="float64")
        self.scales = np.asarray(scaler.scale_, dtype="float64")
        self.scales[~np.isfinite(self.scales) | (self.scales == 0.0)] = 1.0
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.medians is None or self.means is None or self.scales is None:
            raise RuntimeError("preprocessor is not fitted")
        matrix = numeric_matrix(frame, self.feature_names)
        filled = np.where(np.isfinite(matrix), matrix, self.medians)
        return (filled - self.means) / self.scales


def enforce_monotone_cumulative(probability: np.ndarray) -> np.ndarray:
    """Project P(Y>=t) to the non-increasing cumulative probability cone."""

    values = np.asarray(probability, dtype="float64")
    if values.ndim != 2 or values.shape[1] != len(CUMULATIVE_THRESHOLDS):
        raise ValueError("cumulative probability must have shape (n, 4)")
    values = np.clip(values, 0.0, 1.0)
    return np.minimum.accumulate(values, axis=1)


def cumulative_to_class_probability(cumulative: np.ndarray) -> np.ndarray:
    """Convert four cumulative probabilities into five class probabilities."""

    survival = enforce_monotone_cumulative(cumulative)
    result = np.column_stack(
        [
            1.0 - survival[:, 0],
            survival[:, 0] - survival[:, 1],
            survival[:, 1] - survival[:, 2],
            survival[:, 2] - survival[:, 3],
            survival[:, 3],
        ]
    )
    result = np.clip(result, 0.0, 1.0)
    row_sum = result.sum(axis=1, keepdims=True)
    return np.divide(result, row_sum, out=np.zeros_like(result), where=row_sum > 0)


def select_compact_features(columns: Sequence[str]) -> tuple[str, ...]:
    """Frozen compact passive set used by the 004B ordinal comparison."""

    direct = {
        "feature_nonmissing_count",
        "history_observed_month_count",
        "history_month_span",
        "history_months_since_last_any_record",
        "baseline_available_count",
        "cold_start_field_count",
        "group_reference_available_count",
        "cold_start_flag",
    }
    suffixes = (
        "__anchor",
        "__anchor_missing",
        "__delta_1",
        "__delta_1_missing",
        "__delta_2",
        "__delta_2_missing",
        "__local_slope",
        "__local_slope_missing",
        "__personal_z",
        "__personal_z_missing",
        "__history_count",
        "__months_since_last_valid",
        "__months_since_last_valid_missing",
    )
    selected = [
        name
        for name in columns
        if name in direct
        or (name.startswith("short__") and any(name.endswith(s) for s in suffixes))
        or name.startswith("opt003__")
    ]
    prohibited_tokens = (
        "phq",
        "target",
        "future",
        "participant_id",
        "outer_fold",
        "inner_fold",
    )
    selected = [
        name
        for name in selected
        if not any(token in name.lower() for token in prohibited_tokens)
    ]
    if not selected:
        raise ValueError("compact feature selection returned no columns")
    return tuple(selected)


def _tree_matrix(frame: pd.DataFrame, feature_names: Sequence[str]) -> np.ndarray:
    matrix = numeric_matrix(frame, feature_names)
    # CatBoost and LightGBM accept NaN, but not +/-inf (already normalized above).
    return matrix


def _tail_probability(predicted_score: np.ndarray, scale: float) -> np.ndarray:
    logits = (
        predicted_score[:, None] - np.asarray(CUMULATIVE_THRESHOLDS)[None, :]
    ) / max(scale, 0.5)
    logits = np.clip(logits, -35.0, 35.0)
    return enforce_monotone_cumulative(1.0 / (1.0 + np.exp(-logits)))


def fit_predict_ordinal(
    family: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_names: Sequence[str],
    seed: int,
    max_rounds: int = 160,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit one fold and return monotone cumulative validation probabilities."""

    if family not in MODEL_FAMILIES:
        raise ValueError(f"unsupported ordinal family: {family}")
    train_score = train["aux_phq9_score"].to_numpy(dtype="float64")
    if family == "cumulative_logistic":
        preprocessor = FoldPreprocessor(tuple(feature_names)).fit(train)
        x_train = preprocessor.transform(train)
        x_validation = preprocessor.transform(validation)
        columns: list[np.ndarray] = []
        for threshold in CUMULATIVE_THRESHOLDS:
            estimator = LogisticRegression(
                C=0.15,
                solver="liblinear",
                max_iter=1500,
                random_state=seed,
            )
            labels = (train_score >= threshold).astype("int8")
            estimator.fit(
                x_train,
                labels,
                sample_weight=participant_threshold_weights(train, threshold),
            )
            columns.append(estimator.predict_proba(x_validation)[:, 1])
        raw = np.column_stack(columns)
        return enforce_monotone_cumulative(raw), {
            "family": family,
            "mapping": "four_cumulative_logistic_heads",
            "seed": seed,
        }

    x_train = _tree_matrix(train, feature_names)
    x_validation = _tree_matrix(validation, feature_names)
    if family == "lightgbm_cumulative_thresholds":
        if LGBMClassifier is None:
            raise RuntimeError("LightGBM is required; run in eldercare-ai")
        columns = []
        for threshold in CUMULATIVE_THRESHOLDS:
            estimator = LGBMClassifier(
                objective="binary",
                n_estimators=max_rounds,
                learning_rate=0.035,
                num_leaves=15,
                max_depth=5,
                min_child_samples=45,
                subsample=0.85,
                colsample_bytree=0.7,
                reg_alpha=0.15,
                reg_lambda=1.0,
                random_state=seed,
                n_jobs=1,
                deterministic=True,
                force_col_wise=True,
                verbosity=-1,
            )
            labels = (train_score >= threshold).astype("int8")
            estimator.fit(
                x_train,
                labels,
                sample_weight=participant_threshold_weights(train, threshold),
            )
            columns.append(estimator.predict_proba(x_validation)[:, 1])
        raw = np.column_stack(columns)
        return enforce_monotone_cumulative(raw), {
            "family": family,
            "mapping": "four_cumulative_lightgbm_heads",
            "seed": seed,
        }

    regression_weights = participant_equal_weights(train)
    if family == "catboost_regression":
        if CatBoostRegressor is None:
            raise RuntimeError("CatBoost is required; run in eldercare-ai")
        estimator = CatBoostRegressor(
            loss_function="RMSE",
            iterations=max_rounds,
            depth=5,
            learning_rate=0.04,
            l2_leaf_reg=5.0,
            random_seed=seed,
            thread_count=1,
            verbose=False,
            allow_writing_files=False,
        )
    else:
        if LGBMRegressor is None:
            raise RuntimeError("LightGBM is required; run in eldercare-ai")
        estimator = LGBMRegressor(
            objective="huber",
            n_estimators=max_rounds,
            learning_rate=0.035,
            num_leaves=15,
            max_depth=5,
            min_child_samples=45,
            subsample=0.85,
            colsample_bytree=0.7,
            reg_alpha=0.15,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=1,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
        )
    estimator.fit(x_train, train_score, sample_weight=regression_weights)
    train_prediction = np.clip(np.asarray(estimator.predict(x_train)), 0.0, 27.0)
    validation_prediction = np.clip(
        np.asarray(estimator.predict(x_validation)), 0.0, 27.0
    )
    residual = train_score - train_prediction
    residual_center = float(np.median(residual))
    residual_scale = float(np.median(np.abs(residual - residual_center)) * 1.4826)
    residual_scale = max(residual_scale, 1.0)
    cumulative = _tail_probability(
        validation_prediction + residual_center, residual_scale
    )
    return cumulative, {
        "family": family,
        "mapping": "fold_train_residual_logistic_tail",
        "residual_center": residual_center,
        "residual_scale": residual_scale,
        "seed": seed,
    }


def expected_score_from_classes(class_probability: np.ndarray) -> np.ndarray:
    midpoints = np.asarray([2.0, 7.0, 12.0, 17.0, 23.5], dtype="float64")
    return np.asarray(class_probability, dtype="float64") @ midpoints


def weighted_ece(
    y_true: np.ndarray, probability: np.ndarray, weights: np.ndarray
) -> float:
    weights = weights / weights.sum()
    bins = np.minimum((np.clip(probability, 0.0, 1.0) * 10).astype(int), 9)
    result = 0.0
    for bin_id in range(10):
        mask = bins == bin_id
        if not np.any(mask):
            continue
        mass = float(weights[mask].sum())
        result += mass * abs(
            float(np.average(probability[mask], weights=weights[mask]))
            - float(np.average(y_true[mask], weights=weights[mask]))
        )
    return float(result)


def ordinal_metrics(frame: pd.DataFrame, cumulative: np.ndarray) -> dict[str, float]:
    """Participant-equal binary, ordinal and calibration metrics."""

    survival = enforce_monotone_cumulative(cumulative)
    classes = cumulative_to_class_probability(survival)
    p_ge10 = survival[:, 1]
    y_binary = frame["future_binary_target"].to_numpy(dtype="int8")
    score = frame["aux_phq9_score"].to_numpy(dtype="float64")
    y_ordinal = ordinal_target(score)
    predicted_ordinal = classes.argmax(axis=1).astype("int8")
    predicted_score = expected_score_from_classes(classes)
    weights = participant_equal_weights(frame)
    weights = weights / weights.sum()
    return {
        "auprc": float(
            average_precision_score(y_binary, p_ge10, sample_weight=weights)
        ),
        "auroc": float(roc_auc_score(y_binary, p_ge10, sample_weight=weights)),
        "brier": float(np.sum(weights * (p_ge10 - y_binary) ** 2)),
        "ece": weighted_ece(y_binary, p_ge10, weights),
        "ordinal_mae": float(
            mean_absolute_error(y_ordinal, predicted_ordinal, sample_weight=weights)
        ),
        "score_mae": float(
            mean_absolute_error(score, predicted_score, sample_weight=weights)
        ),
        "weighted_kappa": float(
            cohen_kappa_score(
                y_ordinal,
                predicted_ordinal,
                weights="quadratic",
                sample_weight=weights,
            )
        ),
        "monotonic_violation_count": float(np.sum(np.diff(survival, axis=1) > 1e-12)),
        "p_ge10_reconstruction_max_abs_error": float(
            np.max(np.abs(p_ge10 - classes[:, 2:].sum(axis=1)))
        ),
    }
