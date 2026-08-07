"""Leakage-safe history features and model selection for V3.4 Forecast.

This module is isolated from the V3.3 runtime and from the immutable V3.4
baseline artifacts. It only prepares and selects offline optimization
candidates under participant-level nested splits.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from elderly_monitoring.datasets.adapters.psyche_d import (
    LABEL_FIELDS,
    parse_sample_index,
)
from elderly_monitoring.modules.mental_health.mood_social.forecast_experiment import (
    FORECAST_TASKS,
    build_forecast_samples,
    load_feature_names,
)
from elderly_monitoring.modules.mental_health.mood_social.forecast_modeling import (
    participant_class_weights,
    participant_equal_weights,
)

try:  # Formal runs use the locked eldercare-ai environment.
    from catboost import CatBoostClassifier
except ImportError:  # pragma: no cover - optional outside formal training.
    CatBoostClassifier = None  # type: ignore[assignment,misc]

try:  # Formal runs use the locked eldercare-ai environment.
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover - optional outside formal training.
    LGBMClassifier = None  # type: ignore[assignment,misc]


SEED = 20260728
OPTIMIZATION_VERSION = "mood-social-forecast-v3.4-opt-v1"
FEATURE_GROUPS = ("anchor_only", "anchor_delta", "anchor_delta_rolling")
AUXILIARY_MODES = ("none", "phq9_score", "phq9_category")
MODEL_FAMILIES = ("elasticnet_logistic", "lightgbm", "catboost")
METADATA_COLUMNS = (
    "participant_id",
    "global_participant_id",
    "target_window_id",
    "feature_window_id",
    "label_month_slot",
    "feature_month_slot",
    "task_id",
    "nominal_gap_months",
    "timing_evidence",
    "natural_dates_available",
    "feature_start_time",
    "feature_end_time",
    "assessment_time",
    "future_binary_target",
    "aux_phq9_score_end",
    "aux_phq9_cat_end",
    "feature_nonmissing_count",
    "history_observed_month_count",
    "outer_fold_id",
)

warnings.filterwarnings(
    "ignore",
    message="'penalty' was deprecated.*",
    category=FutureWarning,
    module=r"sklearn\.linear_model\._logistic",
)


class ForecastOptimizationError(ValueError):
    """Raised when an optimization-only contract is violated."""


@dataclass
class HistoryPreprocessor:
    """Median-fill and scale history values without duplicating explicit masks."""

    feature_names: tuple[str, ...]
    medians: np.ndarray | None = None
    means: np.ndarray | None = None
    scales: np.ndarray | None = None

    def fit(self, frame: pd.DataFrame) -> HistoryPreprocessor:
        values = _raw_matrix(frame, self.feature_names)
        medians = np.empty(values.shape[1], dtype="float64")
        for index in range(values.shape[1]):
            finite = values[:, index][np.isfinite(values[:, index])]
            medians[index] = float(np.median(finite)) if finite.size else 0.0
        filled = np.where(np.isfinite(values), values, medians)
        scaler = StandardScaler().fit(filled)
        scales = np.asarray(scaler.scale_, dtype="float64")
        scales[~np.isfinite(scales) | (scales == 0.0)] = 1.0
        self.medians = medians
        self.means = np.asarray(scaler.mean_, dtype="float64")
        self.scales = scales
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.medians is None or self.means is None or self.scales is None:
            raise RuntimeError("history preprocessor has not been fitted")
        values = _raw_matrix(frame, self.feature_names)
        filled = np.where(np.isfinite(values), values, self.medians)
        return (filled - self.means) / self.scales


def _finite(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if np.isfinite(number) else float("nan")


def _source_frame(source: pd.DataFrame, source_fields: Sequence[str]) -> pd.DataFrame:
    required = set(source_fields) | set(LABEL_FIELDS)
    missing = sorted(required.difference(source.columns))
    if missing:
        raise ForecastOptimizationError(f"source fields are missing: {missing}")
    parsed = parse_sample_index(source.index)
    keys = pd.MultiIndex.from_arrays(
        [parsed["participant_id"].astype(str), parsed["nominal_month"].astype(int)]
    )
    if keys.has_duplicates:
        raise ForecastOptimizationError("participant-month keys must be unique")
    frame = source.copy()
    frame["__participant_id"] = parsed["participant_id"].astype(str).to_numpy()
    frame["__nominal_month"] = parsed["nominal_month"].astype(int).to_numpy()
    for name in (*source_fields, *LABEL_FIELDS):
        values = pd.to_numeric(frame[name], errors="coerce").astype("float64")
        frame[name] = values.mask(~np.isfinite(values), np.nan)
    return frame


def history_feature_groups(source_fields: Sequence[str]) -> dict[str, tuple[str, ...]]:
    """Return the exact, frozen derived-feature order for all ablations."""

    anchor: list[str] = []
    delta: list[str] = []
    rolling: list[str] = []
    for field in source_fields:
        anchor.extend((f"{field}__anchor", f"{field}__anchor_missing"))
        delta.extend(
            (
                f"{field}__delta_1",
                f"{field}__delta_1_missing",
                f"{field}__delta_2",
                f"{field}__delta_2_missing",
            )
        )
        rolling.extend(
            (
                f"{field}__rolling_mean_3",
                f"{field}__rolling_mean_3_missing",
                f"{field}__rolling_std_3",
                f"{field}__rolling_std_3_missing",
                f"{field}__rolling_slope_3",
                f"{field}__rolling_slope_3_missing",
                f"{field}__history_available_count",
                f"{field}__consecutive_missing_count",
                f"{field}__months_since_last_valid",
                f"{field}__months_since_last_valid_missing",
            )
        )
    return {
        "anchor_only": tuple(anchor),
        "anchor_delta": tuple(anchor + delta),
        "anchor_delta_rolling": tuple(anchor + delta + rolling),
    }


def _linear_slope(points: Sequence[tuple[int, float]]) -> float:
    if len(points) < 2:
        return float("nan")
    x = np.asarray([point[0] for point in points], dtype="float64")
    y = np.asarray([point[1] for point in points], dtype="float64")
    centered = x - x.mean()
    denominator = float(np.sum(centered**2))
    if denominator == 0.0:
        return float("nan")
    return float(np.sum(centered * (y - y.mean())) / denominator)


def _field_history_features(
    values_by_month: Mapping[int, float], cutoff_month: int, field: str
) -> dict[str, float | int]:
    anchor = _finite(values_by_month.get(cutoff_month))
    previous = _finite(values_by_month.get(cutoff_month - 1))
    previous_two = _finite(values_by_month.get(cutoff_month - 2))
    rolling_points = [
        (month, value)
        for month, value in (
            (cutoff_month - 2, previous_two),
            (cutoff_month - 1, previous),
            (cutoff_month, anchor),
        )
        if np.isfinite(value)
    ]
    rolling_values = np.asarray([value for _, value in rolling_points], dtype="float64")
    mean = float(np.mean(rolling_values)) if rolling_values.size else float("nan")
    std = float(np.std(rolling_values, ddof=0)) if rolling_values.size else float("nan")
    slope = _linear_slope(rolling_points)
    valid_history_months = sorted(
        month
        for month, value in values_by_month.items()
        if month <= cutoff_month and np.isfinite(_finite(value))
    )
    last_valid = valid_history_months[-1] if valid_history_months else None
    months_since = (
        float(cutoff_month - last_valid) if last_valid is not None else float("nan")
    )
    observed_months = [month for month in values_by_month if month <= cutoff_month]
    lower_bound = min(observed_months) if observed_months else cutoff_month
    consecutive_missing = 0
    for month in range(cutoff_month, lower_bound - 1, -1):
        if np.isfinite(_finite(values_by_month.get(month))):
            break
        consecutive_missing += 1
    return {
        f"{field}__anchor": anchor,
        f"{field}__anchor_missing": int(not np.isfinite(anchor)),
        f"{field}__delta_1": anchor - previous
        if np.isfinite(anchor) and np.isfinite(previous)
        else float("nan"),
        f"{field}__delta_1_missing": int(
            not (np.isfinite(anchor) and np.isfinite(previous))
        ),
        f"{field}__delta_2": anchor - previous_two
        if np.isfinite(anchor) and np.isfinite(previous_two)
        else float("nan"),
        f"{field}__delta_2_missing": int(
            not (np.isfinite(anchor) and np.isfinite(previous_two))
        ),
        f"{field}__rolling_mean_3": mean,
        f"{field}__rolling_mean_3_missing": int(not np.isfinite(mean)),
        f"{field}__rolling_std_3": std,
        f"{field}__rolling_std_3_missing": int(not np.isfinite(std)),
        f"{field}__rolling_slope_3": slope,
        f"{field}__rolling_slope_3_missing": int(not np.isfinite(slope)),
        f"{field}__history_available_count": len(valid_history_months),
        f"{field}__consecutive_missing_count": consecutive_missing,
        f"{field}__months_since_last_valid": months_since,
        f"{field}__months_since_last_valid_missing": int(not np.isfinite(months_since)),
    }


def build_history_forecast_samples(
    source: pd.DataFrame,
    *,
    task_id: str,
    mapping_path: Path,
    split_path: Path,
    expected_count: int | None = None,
) -> pd.DataFrame:
    """Build history features using only nominal months at or before ``c=t-h``."""

    if task_id not in FORECAST_TASKS:
        raise ForecastOptimizationError(f"unsupported task: {task_id}")
    source_fields = load_feature_names(mapping_path)
    baseline = build_forecast_samples(
        source,
        task_id=task_id,
        mapping_path=mapping_path,
        split_path=split_path,
        expected_count=expected_count,
    )
    frame = _source_frame(source, source_fields)
    participant_frames = {
        participant: rows.sort_values("__nominal_month", kind="mergesort")
        for participant, rows in frame.groupby("__participant_id", sort=False)
    }
    target_lookup = frame.set_index(["__participant_id", "__nominal_month"])
    result_rows: list[dict[str, Any]] = []
    for sample in baseline.to_dict(orient="records"):
        participant = str(sample["participant_id"])
        cutoff = int(sample["feature_month_slot"])
        label_month = int(sample["label_month_slot"])
        participant_frame = participant_frames[participant]
        past = participant_frame.loc[participant_frame["__nominal_month"] <= cutoff]
        if past.empty:
            raise ForecastOptimizationError("eligible sample has no past source rows")
        target_key = (participant, label_month)
        if target_key not in target_lookup.index:
            raise ForecastOptimizationError(
                "target row disappeared during history build"
            )
        target = target_lookup.loc[target_key]
        if isinstance(target, pd.DataFrame):
            raise ForecastOptimizationError(
                "target participant-month key is duplicated"
            )
        output = {name: sample[name] for name in METADATA_COLUMNS if name in sample}
        output["aux_phq9_score_end"] = int(target["phq9_score_end"])
        output["aux_phq9_cat_end"] = int(target["phq9_cat_end"])
        output["history_observed_month_count"] = int(past["__nominal_month"].nunique())
        for field in source_fields:
            values_by_month = dict(
                zip(
                    participant_frame["__nominal_month"].astype(int),
                    participant_frame[field],
                    strict=True,
                )
            )
            output.update(_field_history_features(values_by_month, cutoff, field))
        result_rows.append(output)
    groups = history_feature_groups(source_fields)
    columns = list(METADATA_COLUMNS) + list(groups["anchor_delta_rolling"])
    result = pd.DataFrame(result_rows).reindex(columns=columns)
    _validate_history_samples(result, task_id, source_fields, expected_count)
    return result


def _validate_history_samples(
    frame: pd.DataFrame,
    task_id: str,
    source_fields: Sequence[str],
    expected_count: int | None,
) -> None:
    gap = FORECAST_TASKS[task_id]
    if frame.empty:
        raise ForecastOptimizationError("history sample is empty")
    if expected_count is not None and len(frame) != expected_count:
        raise ForecastOptimizationError(
            f"{task_id} history count {len(frame)} != {expected_count}"
        )
    if frame["target_window_id"].duplicated().any():
        raise ForecastOptimizationError("history target windows are duplicated")
    if (frame["feature_month_slot"] != frame["label_month_slot"] - gap).any():
        raise ForecastOptimizationError("history cutoff is not c=t-h")
    groups = history_feature_groups(source_fields)
    derived = set(groups["anchor_delta_rolling"])
    prohibited = set(LABEL_FIELDS) | {
        "future_binary_target",
        "aux_phq9_score_end",
        "aux_phq9_cat_end",
        "label_month_slot",
    }
    if derived.intersection(prohibited):
        raise ForecastOptimizationError(
            "a label or target field entered model features"
        )
    if len(groups["anchor_only"]) != 54:
        raise ForecastOptimizationError("anchor feature count is not frozen at 54")
    if len(groups["anchor_delta"]) != 162:
        raise ForecastOptimizationError("anchor+delta count is not frozen at 162")
    if len(groups["anchor_delta_rolling"]) != 432:
        raise ForecastOptimizationError("full history count is not frozen at 432")


def feature_subset(frame: pd.DataFrame, feature_names: Sequence[str]) -> pd.DataFrame:
    missing = sorted(set(feature_names).difference(frame.columns))
    if missing:
        raise ForecastOptimizationError(f"model feature columns are missing: {missing}")
    return frame.loc[:, list(feature_names)].copy()


def _raw_matrix(frame: pd.DataFrame, feature_names: Sequence[str]) -> np.ndarray:
    matrix = (
        frame.loc[:, list(feature_names)]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype="float64", copy=True)
    )
    matrix[~np.isfinite(matrix)] = np.nan
    return matrix


def _fit_auxiliary(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_names: Sequence[str],
    mode: str,
) -> np.ndarray:
    preprocessor = HistoryPreprocessor(tuple(feature_names)).fit(train)
    x_train = preprocessor.transform(train)
    x_validation = preprocessor.transform(validation)
    weights = participant_equal_weights(train)
    if mode == "phq9_score":
        estimator = Ridge(alpha=10.0, solver="lsqr", tol=1e-4)
        estimator.fit(x_train, train["aux_phq9_score_end"], sample_weight=weights)
        prediction = estimator.predict(x_validation) / 27.0
        return np.clip(np.asarray(prediction, dtype="float64"), 0.0, 1.0)
    if mode == "phq9_category":
        estimator = Ridge(
            alpha=10.0,
            solver="lsqr",
            tol=1e-4,
        )
        estimator.fit(
            x_train,
            train["aux_phq9_cat_end"].astype("float64") / 4.0,
            sample_weight=weights,
        )
        return np.clip(
            np.asarray(estimator.predict(x_validation), dtype="float64"), 0.0, 1.0
        )
    raise ForecastOptimizationError(f"unsupported auxiliary mode: {mode}")


def strict_auxiliary_inner_views(
    frame: pd.DataFrame,
    assignments: Mapping[str, int],
    feature_names: Sequence[str],
    mode: str,
    validation_fold: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create auxiliary OOF train values and held-out validation values.

    The binary fit rows receive auxiliary predictions from models that did not
    see those rows. The current inner validation fold receives predictions from
    an auxiliary model fitted only on the binary fit rows.
    """

    if mode not in AUXILIARY_MODES[1:]:
        raise ForecastOptimizationError("strict auxiliary view needs an aux mode")
    fold_series = frame["global_participant_id"].map(assignments)
    if fold_series.isna().any():
        raise ForecastOptimizationError("inner split does not cover all participants")
    validation = frame.loc[fold_series.eq(validation_fold)].copy()
    fit = frame.loc[~fold_series.eq(validation_fold)].copy()
    if fit.empty or validation.empty:
        raise ForecastOptimizationError("inner auxiliary split is empty")
    fit["__auxiliary_prediction"] = np.nan
    remaining_folds = sorted(set(fold_series.loc[fit.index].astype(int).tolist()))
    for auxiliary_fold in remaining_folds:
        auxiliary_validation_mask = (
            fit["global_participant_id"].map(assignments).eq(auxiliary_fold)
        )
        auxiliary_train = fit.loc[~auxiliary_validation_mask]
        auxiliary_validation = fit.loc[auxiliary_validation_mask]
        if auxiliary_train.empty or auxiliary_validation.empty:
            raise ForecastOptimizationError("nested auxiliary split is empty")
        fit.loc[auxiliary_validation.index, "__auxiliary_prediction"] = _fit_auxiliary(
            auxiliary_train, auxiliary_validation, feature_names, mode
        )
    if fit["__auxiliary_prediction"].isna().any():
        raise ForecastOptimizationError("auxiliary OOF predictions are incomplete")
    validation["__auxiliary_prediction"] = _fit_auxiliary(
        fit, validation, feature_names, mode
    )
    return fit, validation


def strict_auxiliary_outer_views(
    outer_train: pd.DataFrame,
    outer_test: pd.DataFrame,
    assignments: Mapping[str, int],
    feature_names: Sequence[str],
    mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build strict OOF auxiliary values for an outer-fold final fit.

    Every outer-train row is predicted by an auxiliary model that excluded its
    participant. The untouched outer-test rows are predicted by one auxiliary
    model fitted on the complete outer-train partition.
    """

    if mode not in AUXILIARY_MODES[1:]:
        raise ForecastOptimizationError("strict auxiliary view needs an aux mode")
    train = outer_train.copy()
    test = outer_test.copy()
    fold_series = train["global_participant_id"].map(assignments)
    if fold_series.isna().any():
        raise ForecastOptimizationError("inner split does not cover outer-train")
    train["__auxiliary_prediction"] = np.nan
    for inner_fold in sorted(fold_series.astype(int).unique().tolist()):
        validation_mask = fold_series.eq(inner_fold)
        auxiliary_train = train.loc[~validation_mask]
        auxiliary_validation = train.loc[validation_mask]
        if auxiliary_train.empty or auxiliary_validation.empty:
            raise ForecastOptimizationError("outer auxiliary split is empty")
        train.loc[auxiliary_validation.index, "__auxiliary_prediction"] = (
            _fit_auxiliary(
                auxiliary_train,
                auxiliary_validation,
                feature_names,
                mode,
            )
        )
    if train["__auxiliary_prediction"].isna().any():
        raise ForecastOptimizationError("outer-train auxiliary OOF is incomplete")
    test["__auxiliary_prediction"] = _fit_auxiliary(
        outer_train,
        outer_test,
        feature_names,
        mode,
    )
    return train, test


def _binary_estimator(family: str, params: Mapping[str, Any]) -> Any:
    if family == "elasticnet_logistic":
        return LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            max_iter=1500,
            tol=1e-3,
            class_weight=None,
            random_state=SEED,
            **dict(params),
        )
    if family == "lightgbm":
        if LGBMClassifier is None:
            raise RuntimeError("LightGBM is required in eldercare-ai")
        return LGBMClassifier(
            objective="binary",
            boosting_type="gbdt",
            class_weight=None,
            random_state=SEED,
            n_jobs=1,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
            **dict(params),
        )
    if family == "catboost":
        if CatBoostClassifier is None:
            raise RuntimeError("CatBoost is required in eldercare-ai")
        return CatBoostClassifier(
            loss_function="Logloss",
            random_seed=SEED,
            thread_count=1,
            allow_writing_files=False,
            verbose=False,
            **dict(params),
        )
    raise ForecastOptimizationError(f"unsupported model family: {family}")


def fit_binary_candidate(
    family: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_names: Sequence[str],
    params: Mapping[str, Any],
) -> np.ndarray:
    weights = participant_class_weights(train)
    if family == "elasticnet_logistic":
        preprocessor = HistoryPreprocessor(tuple(feature_names)).fit(train)
        x_train = preprocessor.transform(train)
        x_validation = preprocessor.transform(validation)
    else:
        x_train = _raw_matrix(train, feature_names)
        x_validation = _raw_matrix(validation, feature_names)
    estimator = _binary_estimator(family, params)
    estimator.fit(
        x_train,
        train["future_binary_target"].astype("int8"),
        sample_weight=weights,
    )
    return np.asarray(estimator.predict_proba(x_validation)[:, 1], dtype="float64")


@dataclass(frozen=True)
class SelectionCandidate:
    feature_group: str
    model_family: str
    params: Mapping[str, Any]
    auxiliary_mode: str = "none"

    @property
    def candidate_id(self) -> str:
        payload = json.dumps(
            {
                "feature_group": self.feature_group,
                "model_family": self.model_family,
                "params": dict(self.params),
                "auxiliary_mode": self.auxiliary_mode,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def evaluate_inner_candidate(
    frame: pd.DataFrame,
    assignments: Mapping[str, int],
    feature_names: Sequence[str],
    candidate: SelectionCandidate,
    auxiliary_cache: dict[tuple[tuple[str, ...], str, int], tuple[pd.Series, pd.Series]]
    | None = None,
    auxiliary_feature_names: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    parts: list[pd.DataFrame] = []
    for inner_fold in range(5):
        if candidate.auxiliary_mode == "none":
            validation_ids = {
                participant
                for participant, fold in assignments.items()
                if int(fold) == inner_fold
            }
            validation = frame.loc[
                frame["global_participant_id"].isin(validation_ids)
            ].copy()
            train = frame.loc[
                ~frame["global_participant_id"].isin(validation_ids)
            ].copy()
            model_features = tuple(feature_names)
        else:
            auxiliary_features = tuple(auxiliary_feature_names or feature_names)
            cache_key = (
                auxiliary_features,
                candidate.auxiliary_mode,
                inner_fold,
            )
            cached = (
                auxiliary_cache.get(cache_key) if auxiliary_cache is not None else None
            )
            if cached is None:
                train, validation = strict_auxiliary_inner_views(
                    frame,
                    assignments,
                    auxiliary_features,
                    candidate.auxiliary_mode,
                    inner_fold,
                )
                if auxiliary_cache is not None:
                    auxiliary_cache[cache_key] = (
                        train["__auxiliary_prediction"].copy(),
                        validation["__auxiliary_prediction"].copy(),
                    )
            else:
                fold_series = frame["global_participant_id"].map(assignments)
                validation = frame.loc[fold_series.eq(inner_fold)].copy()
                train = frame.loc[~fold_series.eq(inner_fold)].copy()
                train["__auxiliary_prediction"] = cached[0].reindex(train.index)
                validation["__auxiliary_prediction"] = cached[1].reindex(
                    validation.index
                )
                if (
                    train["__auxiliary_prediction"].isna().any()
                    or validation["__auxiliary_prediction"].isna().any()
                ):
                    raise ForecastOptimizationError(
                        "cached auxiliary predictions do not match inner split"
                    )
            model_features = tuple(feature_names) + ("__auxiliary_prediction",)
        if train.empty or validation.empty:
            raise ForecastOptimizationError("inner candidate split is empty")
        probability = fit_binary_candidate(
            candidate.model_family,
            train,
            validation,
            model_features,
            candidate.params,
        )
        output = validation[
            [
                "global_participant_id",
                "target_window_id",
                "future_binary_target",
                "outer_fold_id",
            ]
        ].copy()
        output["inner_fold_id"] = inner_fold
        output["p_raw"] = probability
        output["auxiliary_prediction"] = (
            validation["__auxiliary_prediction"].to_numpy(dtype="float64")
            if candidate.auxiliary_mode != "none"
            else np.nan
        )
        parts.append(output)
    predictions = pd.concat(parts, ignore_index=True).sort_values(
        "target_window_id", kind="mergesort"
    )
    if predictions["target_window_id"].duplicated().any():
        raise ForecastOptimizationError("inner OOF contains duplicate windows")
    if len(predictions) != len(frame):
        raise ForecastOptimizationError("inner OOF does not cover outer train")
    weights = participant_equal_weights(predictions)
    y = predictions["future_binary_target"].to_numpy(dtype="int8")
    p = predictions["p_raw"].to_numpy(dtype="float64")
    auprc = float(average_precision_score(y, p, sample_weight=weights))
    brier = float(np.sum(weights * (p - y) ** 2) / weights.sum())
    return predictions.reset_index(drop=True), {"auprc": auprc, "brier": brier}


__all__ = [
    "AUXILIARY_MODES",
    "FEATURE_GROUPS",
    "METADATA_COLUMNS",
    "MODEL_FAMILIES",
    "OPTIMIZATION_VERSION",
    "ForecastOptimizationError",
    "HistoryPreprocessor",
    "SelectionCandidate",
    "build_history_forecast_samples",
    "evaluate_inner_candidate",
    "feature_subset",
    "fit_binary_candidate",
    "history_feature_groups",
    "strict_auxiliary_inner_views",
    "strict_auxiliary_outer_views",
]
