"""Leakage-safe modelling helpers for the V3.4 forecast experiment.

The helpers in this module are intentionally offline-only.  They implement
the frozen participant-level weighting, preprocessing, nested search,
calibration and threshold rules used by the two forecast horizons.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score
from sklearn.preprocessing import StandardScaler

try:  # LightGBM is available in the locked training environment.
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover - base environment has no LightGBM.
    LGBMClassifier = None  # type: ignore[assignment,misc]

warnings.filterwarnings(
    "ignore",
    message="'penalty' was deprecated.*",
    category=FutureWarning,
    module=r"sklearn\.linear_model\._logistic",
)
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names.*",
    category=UserWarning,
    module=r"sklearn\.utils\.validation",
)


SEED = 20260728
MODEL_FAMILIES = ("dummy", "elasticnet_logistic", "lightgbm")


def participant_equal_weights(frame: pd.DataFrame) -> np.ndarray:
    """Give every participant total evaluation weight one, then normalize."""

    if frame.empty:
        return np.empty(0, dtype="float64")
    counts = (
        frame.groupby("global_participant_id", sort=False)["global_participant_id"]
        .transform("size")
        .to_numpy(dtype="float64")
    )
    participants = frame["global_participant_id"].nunique()
    return 1.0 / (float(participants) * counts)


def participant_class_weights(frame: pd.DataFrame) -> np.ndarray:
    """Frozen V3.4 participant-class training weights."""

    if frame.empty:
        raise ValueError("cannot compute training weights for an empty frame")
    labels = frame["future_binary_target"].to_numpy(dtype="int8")
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("training fold must contain both target classes")
    work = frame[["global_participant_id", "future_binary_target"]].copy()
    class_participants = work.groupby("future_binary_target")[
        "global_participant_id"
    ].nunique()
    counts = (
        work.groupby(["global_participant_id", "future_binary_target"], sort=False)[
            "future_binary_target"
        ]
        .transform("size")
        .to_numpy(dtype="float64")
    )
    p_count = np.array(
        [float(class_participants[int(label)]) for label in labels], dtype="float64"
    )
    return 0.5 / (p_count * counts)


def participant_equal_prior(frame: pd.DataFrame) -> float:
    """Positive-window rate after giving each participant equal weight."""

    weights = participant_equal_weights(frame)
    return float(
        np.sum(weights * frame["future_binary_target"].to_numpy()) / weights.sum()
    )


@dataclass
class ElasticNetPreprocessor:
    """Training-fold median fill and standardization for 27 + 27 fields."""

    feature_names: tuple[str, ...]
    medians: np.ndarray | None = None
    means: np.ndarray | None = None
    scales: np.ndarray | None = None
    all_missing: np.ndarray | None = None

    def fit(self, frame: pd.DataFrame) -> ElasticNetPreprocessor:
        values = _raw_matrix(frame, self.feature_names)
        medians = np.empty(values.shape[1], dtype="float64")
        all_missing = np.empty(values.shape[1], dtype="bool")
        for index in range(values.shape[1]):
            finite = values[:, index][np.isfinite(values[:, index])]
            all_missing[index] = finite.size == 0
            medians[index] = float(np.median(finite)) if finite.size else 0.0
        filled = np.where(np.isfinite(values), values, medians)
        scaler = StandardScaler(with_mean=True, with_std=True).fit(filled)
        scales = np.asarray(scaler.scale_, dtype="float64")
        scales[~np.isfinite(scales) | (scales == 0.0)] = 1.0
        self.medians = medians
        self.means = np.asarray(scaler.mean_, dtype="float64")
        self.scales = scales
        self.all_missing = all_missing
        return self

    @property
    def derived_feature_names(self) -> tuple[str, ...]:
        return self.feature_names + tuple(
            f"{name}__missing" for name in self.feature_names
        )

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if (
            self.medians is None
            or self.means is None
            or self.scales is None
            or self.all_missing is None
        ):
            raise RuntimeError("preprocessor has not been fitted")
        values = _raw_matrix(frame, self.feature_names)
        missing = (~np.isfinite(values)).astype("float64")
        filled = np.where(np.isfinite(values), values, self.medians)
        filled[:, self.all_missing] = 0.0
        missing[:, self.all_missing] = 1.0
        scaled = (filled - self.means) / self.scales
        return np.concatenate([scaled, missing], axis=1)

    def to_dict(self) -> dict[str, Any]:
        if (
            self.medians is None
            or self.means is None
            or self.scales is None
            or self.all_missing is None
        ):
            raise RuntimeError("preprocessor has not been fitted")
        return {
            "schema_version": "mood-social-forecast-elasticnet-preprocessor-v1",
            "raw_feature_names": list(self.feature_names),
            "derived_missing_indicator_names": [
                f"{name}__missing" for name in self.feature_names
            ],
            "medians": self.medians.tolist(),
            "means": self.means.tolist(),
            "scales": self.scales.tolist(),
            "all_missing": self.all_missing.tolist(),
            "all_missing_fill": 0.0,
            "fit_scope": "current_training_participants_only",
        }


def _raw_matrix(frame: pd.DataFrame, feature_names: Sequence[str]) -> np.ndarray:
    values = frame.loc[:, list(feature_names)].apply(pd.to_numeric, errors="coerce")
    matrix = values.to_numpy(dtype="float64", copy=True)
    matrix[~np.isfinite(matrix)] = np.nan
    return matrix


def build_estimator(model_family: str, params: Mapping[str, Any] | None = None) -> Any:
    params = dict(params or {})
    if model_family == "elasticnet_logistic":
        return LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            max_iter=5000,
            class_weight=None,
            random_state=SEED,
            **params,
        )
    if model_family == "lightgbm":
        if LGBMClassifier is None:
            raise RuntimeError("LightGBM is required; run in eldercare-ai")
        return LGBMClassifier(
            objective="binary",
            boosting_type="gbdt",
            class_weight=None,
            random_state=SEED,
            n_jobs=1,
            deterministic=True,
            force_col_wise=True,
            subsample_freq=1,
            bagging_seed=SEED,
            feature_fraction_seed=SEED,
            data_random_seed=SEED,
            extra_seed=SEED,
            verbosity=-1,
            **params,
        )
    raise ValueError(f"unsupported estimator family: {model_family}")


def fit_predict(
    model_family: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_names: Sequence[str],
    params: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, Any, ElasticNetPreprocessor | None]:
    """Fit one inner fold and return raw positive-class probabilities."""

    if model_family == "dummy":
        prior = participant_equal_prior(train)
        return np.full(len(validation), prior, dtype="float64"), {"prior": prior}, None
    y_train = train["future_binary_target"].to_numpy(dtype="int8")
    weights = participant_class_weights(train)
    if model_family == "elasticnet_logistic":
        preprocessor = ElasticNetPreprocessor(tuple(feature_names)).fit(train)
        x_train = preprocessor.transform(train)
        x_validation = preprocessor.transform(validation)
    else:
        preprocessor = None
        x_train = _raw_matrix(train, feature_names)
        x_validation = _raw_matrix(validation, feature_names)
    estimator = build_estimator(model_family, params)
    estimator.fit(x_train, y_train, sample_weight=weights)
    probability = np.asarray(
        estimator.predict_proba(x_validation)[:, 1], dtype="float64"
    )
    return probability, estimator, preprocessor


def predict_fitted(
    model_family: str,
    estimator: Any,
    preprocessor: ElasticNetPreprocessor | None,
    frame: pd.DataFrame,
    feature_names: Sequence[str],
) -> np.ndarray:
    if model_family == "dummy":
        return np.full(len(frame), float(estimator["prior"]), dtype="float64")
    if model_family == "elasticnet_logistic":
        matrix = preprocessor.transform(frame) if preprocessor is not None else None
    else:
        matrix = _raw_matrix(frame, feature_names)
    if matrix is None:
        raise RuntimeError("missing ElasticNet preprocessor")
    return np.asarray(estimator.predict_proba(matrix)[:, 1], dtype="float64")


def weighted_metrics(
    y_true: Sequence[int],
    p_raw: Sequence[float],
    p_calibrated: Sequence[float],
    threshold: float | Sequence[float],
    weights: Sequence[float] | None = None,
) -> dict[str, float]:
    y = np.asarray(y_true, dtype="int8")
    raw = np.asarray(p_raw, dtype="float64")
    calibrated = np.asarray(p_calibrated, dtype="float64")
    if weights is None:
        weights_array = np.full(len(y), 1.0 / max(len(y), 1), dtype="float64")
    else:
        weights_array = np.asarray(weights, dtype="float64")
        weights_array = weights_array / weights_array.sum()
    threshold_array = np.asarray(threshold, dtype="float64")
    if threshold_array.ndim == 0:
        threshold_array = np.full(len(y), float(threshold_array), dtype="float64")
    if threshold_array.shape != calibrated.shape:
        raise ValueError("threshold must be scalar or match the probability shape")
    prediction = (calibrated >= threshold_array).astype("int8")
    result: dict[str, float] = {}
    if np.unique(y).size == 2:
        result["auprc"] = float(
            average_precision_score(y, raw, sample_weight=weights_array)
        )
        from sklearn.metrics import roc_auc_score

        result["auroc"] = float(roc_auc_score(y, raw, sample_weight=weights_array))
    else:
        result["auprc"] = float("nan")
        result["auroc"] = float("nan")
    result["macro_f1"] = float(
        f1_score(
            y,
            prediction,
            labels=[0, 1],
            average="macro",
            sample_weight=weights_array,
            zero_division=0,
        )
    )
    result["sensitivity"] = float(_recall_for_class(y, prediction, 1, weights_array))
    result["specificity"] = float(_recall_for_class(y, prediction, 0, weights_array))
    result["brier"] = float(np.sum(weights_array * (calibrated - y) ** 2))
    result["ece"] = float(expected_calibration_error(y, calibrated, weights_array))
    result["positive_rate"] = float(np.sum(weights_array * y))
    unique_thresholds = np.unique(threshold_array)
    result["threshold"] = (
        float(unique_thresholds[0]) if len(unique_thresholds) == 1 else float("nan")
    )
    return result


def _recall_for_class(
    y: np.ndarray, prediction: np.ndarray, positive_class: int, weights: np.ndarray
) -> float:
    mask = y == positive_class
    denominator = float(weights[mask].sum())
    return (
        float(weights[mask & (prediction == positive_class)].sum() / denominator)
        if denominator
        else 0.0
    )


def expected_calibration_error(
    y_true: Sequence[int],
    probabilities: Sequence[float],
    weights: Sequence[float],
    bins: int = 10,
) -> float:
    """Tie-preserving participant-weighted equal-frequency ECE."""

    rows = calibration_bins(y_true, probabilities, weights, bins=bins)
    return float(sum(row["bin_weight"] * row["absolute_gap"] for row in rows))


def calibration_bins(
    y_true: Sequence[int],
    probabilities: Sequence[float],
    weights: Sequence[float],
    bins: int = 10,
) -> list[dict[str, float | int]]:
    """Return the frozen tie-preserving equal-frequency calibration bins."""

    y = np.asarray(y_true, dtype="float64")
    p = np.asarray(probabilities, dtype="float64")
    w = np.asarray(weights, dtype="float64")
    if len(y) == 0:
        return []
    order = np.argsort(p, kind="mergesort")
    y, p, w = y[order], p[order], w[order]
    total = float(w.sum())
    target_edges = np.linspace(total / bins, total, bins)
    groups: list[tuple[np.ndarray, float]] = []
    bin_start = 0
    tie_start = 0
    cumulative = 0.0
    edge_index = 0
    for index in range(len(y)):
        is_last = index == len(y) - 1 or p[index + 1] != p[index]
        if not is_last:
            continue
        cumulative += float(w[tie_start : index + 1].sum())
        tie_start = index + 1
        crossed_edge = False
        while edge_index < bins - 1 and cumulative >= target_edges[edge_index]:
            edge_index += 1
            crossed_edge = True
        if crossed_edge:
            groups.append(
                (
                    np.arange(bin_start, index + 1),
                    float(w[bin_start : index + 1].sum()),
                )
            )
            bin_start = index + 1
    if bin_start < len(y):
        groups.append((np.arange(bin_start, len(y)), float(w[bin_start:].sum())))
    result: list[dict[str, float | int]] = []
    for bin_index, (indices, group_weight) in enumerate(groups):
        if group_weight <= 0:
            continue
        mean_probability = float(np.sum(w[indices] * p[indices]) / group_weight)
        positive_rate = float(np.sum(w[indices] * y[indices]) / group_weight)
        result.append(
            {
                "bin_index": bin_index,
                "row_count": int(len(indices)),
                "bin_weight": float(group_weight / total),
                "mean_probability": mean_probability,
                "positive_rate": positive_rate,
                "absolute_gap": abs(mean_probability - positive_rate),
                "probability_min": float(p[indices[0]]),
                "probability_max": float(p[indices[-1]]),
            }
        )
    return result


def select_threshold(
    y_true: Sequence[int], probabilities: Sequence[float], weights: Sequence[float]
) -> tuple[float, dict[str, Any]]:
    p = np.asarray(probabilities, dtype="float64")
    candidates = sorted(set(float(value) for value in p.tolist()) | {0.0, 1.0})
    best_macro = -1.0
    best_sensitivity = -1.0
    selected = 0.0
    records: list[dict[str, float]] = []
    for threshold in candidates:
        metrics = weighted_metrics(y_true, p, p, threshold, weights)
        records.append(
            {
                "threshold": threshold,
                "macro_f1": metrics["macro_f1"],
                "sensitivity": metrics["sensitivity"],
            }
        )
        macro = metrics["macro_f1"]
        sensitivity = metrics["sensitivity"]
        if (
            macro > best_macro + 1e-12
            or (
                abs(macro - best_macro) <= 1e-12
                and sensitivity > best_sensitivity + 1e-12
            )
            or (
                abs(macro - best_macro) <= 1e-12
                and abs(sensitivity - best_sensitivity) <= 1e-12
                and threshold < selected
            )
        ):
            best_macro = macro
            best_sensitivity = sensitivity
            selected = threshold
    if best_macro < 0.0:
        raise ValueError("threshold selection had no candidates")
    return float(selected), {
        "metric": "participant_equal_macro_f1",
        "operator": "greater_than_or_equal",
        "selected_threshold": float(selected),
        "selected_macro_f1": float(best_macro),
        "selected_sensitivity": float(best_sensitivity),
        "candidate_count": len(candidates),
        "candidates": records,
    }


def crossfit_inner(
    model_family: str,
    frame: pd.DataFrame,
    inner_assignments: Mapping[str, int],
    outer_fold: int,
    feature_names: Sequence[str],
    params: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Generate strict participant-level inner OOF predictions."""

    parts: list[pd.DataFrame] = []
    for inner_fold in range(5):
        validation_ids = {
            pid for pid, fold in inner_assignments.items() if int(fold) == inner_fold
        }
        validation = frame[frame.global_participant_id.isin(validation_ids)].copy()
        train = frame[~frame.global_participant_id.isin(validation_ids)].copy()
        if train.empty or validation.empty:
            raise ValueError("inner fold has an empty train or validation set")
        p, _, _ = fit_predict(model_family, train, validation, feature_names, params)
        output = validation[
            [
                "global_participant_id",
                "target_window_id",
                "future_binary_target",
                "outer_fold_id",
            ]
        ].copy()
        output["inner_fold_id"] = inner_fold
        output["p_raw"] = p
        parts.append(output)
    result = pd.concat(parts, ignore_index=True)
    if result.target_window_id.duplicated().any():
        raise ValueError("inner OOF target windows are duplicated")
    return result.sort_values("target_window_id", kind="mergesort").reset_index(
        drop=True
    )


def isotonic_calibrate(
    y_true: Sequence[int], probabilities: Sequence[float], weights: Sequence[float]
) -> IsotonicRegression:
    calibrator = IsotonicRegression(
        increasing=True, out_of_bounds="clip", y_min=0.0, y_max=1.0
    )
    calibrator.fit(
        np.asarray(probabilities, dtype="float64"),
        np.asarray(y_true, dtype="int8"),
        sample_weight=np.asarray(weights, dtype="float64"),
    )
    return calibrator


def json_params(params: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(params), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


__all__ = [
    "ElasticNetPreprocessor",
    "MODEL_FAMILIES",
    "SEED",
    "build_estimator",
    "calibration_bins",
    "crossfit_inner",
    "expected_calibration_error",
    "fit_predict",
    "json_params",
    "isotonic_calibrate",
    "participant_class_weights",
    "participant_equal_prior",
    "participant_equal_weights",
    "predict_fitted",
    "select_threshold",
    "weighted_metrics",
]
