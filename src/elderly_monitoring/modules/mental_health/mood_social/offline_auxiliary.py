"""Strict nested-CV offline PHQ-9 auxiliary models for MODEL-006.

The models in this module are report-only. They never enter the five active
experts, PersonalTrend, MoodFusionModel, the HTTP response, or ART-001.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
from catboost import CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
import lightgbm
import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq
import scipy
from scipy.stats import spearmanr
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)
import yaml

from elderly_monitoring.datasets.adapters.nhanes import (
    TrainingFoldECDF as NhanesTrainingFoldECDF,
    canonical_column_order as nhanes_canonical_column_order,
)
from elderly_monitoring.datasets.adapters.psyche_d import (
    TrainingFoldECDF as PsycheDTrainingFoldECDF,
    canonical_column_order as psyche_d_canonical_column_order,
)
from elderly_monitoring.datasets.adapters.resilient import (
    TrainingFoldECDF as ResilientTrainingFoldECDF,
    canonical_column_order as resilient_canonical_column_order,
)
from elderly_monitoring.modules.mental_health.mood_social.evaluation import (
    load_evaluation_config,
    validate_baseline_protection,
)
from elderly_monitoring.modules.mental_health.mood_social.feature_schema import (
    ACTIVITY_EXPERT_FEATURE_ORDER,
    ACTIVITY_FEATURE_SPECS,
    FEATURE_SCHEMA_VERSION,
    PHYSIOLOGY_EXPERT_FEATURE_ORDER,
    PHYSIOLOGY_FEATURE_SPECS,
    SLEEP_EXPERT_FEATURE_ORDER,
    SLEEP_FEATURE_SPECS,
    SOCIAL_CONTEXT_FEATURE_ORDER,
    SOCIAL_CONTEXT_FEATURE_SPECS,
    FeatureValueType,
)


TRAINING_VERSION = "mood-social-offline-auxiliary-v3.3.3-v1"
BUNDLE_VERSION = "mood-social-offline-auxiliary-bundle-v1"
MANIFEST_VERSION = "mood-social-offline-auxiliary-manifest-v1"
OOF_VERSION = "mood-social-offline-auxiliary-oof-v1"
METRICS_VERSION = "mood-social-offline-auxiliary-metrics-v1"
SEARCH_VERSION = "mood-social-offline-auxiliary-search-v1"
TASK_ID = "MODEL-006"
RUN_ID = "MH-20260801-007"
RANDOM_SEED = 20260728
SPLIT_ID = "mood-social-v3.3.3-participant-nested-5x5-seed-20260728-v1"
SPLIT_SHA256 = "e9915dbc590a6ea454c34d26558ac5866c57ee8a8e78f44cdcc79c859df77ee3"
FEATURE_SCHEMA_SHA256 = (
    "d2ed18db72601ccb3b0cbb89907bc60ab336bd00f84f0b2c49dcea568bf339e9"
)
EVALUATION_RUN_ID = "MH-20260801-006"
EVALUATION_CORE_SHA256 = (
    "07bbcb88b7e41137d151e6c1775bcb805b0cd7e138476b802542a2b635e3a71e"
)
BASELINE_PROTECTION_SHA256 = (
    "ba403f4ed6f75e2bd612e198db8af5893a01afd09bfd5bfb99730b9f736428f6"
)
OUTER_FOLD_COUNT = 5
INNER_FOLD_COUNT = 5
PHQ9_MAX_SCORE = 27.0
GRADE_BANDS = ((0, 4), (5, 9), (10, 14), (15, 19), (20, 27))
GRADE_IDS = (0, 1, 2, 3, 4)
DATASET_ORDER = (
    "psyche_d",
    "resilient",
    "nhanes",
    "shenzhen_elderly",
    "nhanes_ssq_2005_2008",
)
REGRESSION_DATASETS = DATASET_ORDER
MULTICLASS_DATASETS = (
    "psyche_d",
    "nhanes",
    "nhanes_ssq_2005_2008",
)
MODEL_FAMILY_BY_TASK = {
    "psyche_d::regression": "lightgbm_regressor",
    "psyche_d::multiclass": "lightgbm_multiclass",
    "resilient::regression": "elasticnet_regression",
    "nhanes::regression": "lightgbm_regressor",
    "nhanes::multiclass": "lightgbm_multiclass",
    "shenzhen_elderly::regression": "catboost_regressor",
    "nhanes_ssq_2005_2008::regression": "lightgbm_regressor",
    "nhanes_ssq_2005_2008::multiclass": "lightgbm_multiclass",
}
SCORE_COLUMN_BY_DATASET = {
    "psyche_d": "phq9_score_end",
    "resilient": "phq9_score",
    "nhanes": "phq9_total",
    "shenzhen_elderly": "phq9_score",
    "nhanes_ssq_2005_2008": "phq9_total",
}
FEATURE_GROUPS_BY_DATASET = {
    "psyche_d": ("activity", "sleep"),
    "resilient": ("activity", "sleep", "physiology", "social_context"),
    "nhanes": ("activity", "sleep", "social_context"),
    "shenzhen_elderly": ("social_context",),
    "nhanes_ssq_2005_2008": ("social_context",),
}
ECDF_DATASETS = ("psyche_d", "resilient", "nhanes")
_ECDF_CLASS_BY_DATASET = {
    "psyche_d": PsycheDTrainingFoldECDF,
    "resilient": ResilientTrainingFoldECDF,
    "nhanes": NhanesTrainingFoldECDF,
}
_CANONICAL_ORDER_BY_DATASET = {
    "psyche_d": psyche_d_canonical_column_order,
    "resilient": resilient_canonical_column_order,
    "nhanes": nhanes_canonical_column_order,
}
_GROUP_FEATURE_SPECS = {
    "activity": tuple(
        (f"activity.{spec.name}", spec.value_type)
        for spec in ACTIVITY_FEATURE_SPECS
        if spec.name in ACTIVITY_EXPERT_FEATURE_ORDER
    ),
    "sleep": tuple(
        (f"sleep.{spec.name}", spec.value_type)
        for spec in SLEEP_FEATURE_SPECS
        if spec.name in SLEEP_EXPERT_FEATURE_ORDER
    ),
    "physiology": tuple(
        (f"physiology.{spec.name}", spec.value_type)
        for spec in PHYSIOLOGY_FEATURE_SPECS
        if spec.name in PHYSIOLOGY_EXPERT_FEATURE_ORDER
    ),
    "social_context": tuple(
        (f"social_context.{spec.name}", spec.value_type)
        for spec in SOCIAL_CONTEXT_FEATURE_SPECS
        if spec.name in SOCIAL_CONTEXT_FEATURE_ORDER
    ),
}


class OfflineAuxiliaryError(RuntimeError):
    """Raised when MODEL-006 inputs or invariants are invalid."""


@dataclass(frozen=True)
class OfflineDatasetSpec:
    dataset_id: str
    score_column: str
    feature_groups: tuple[str, ...]
    input_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    fit_activity_ecdf: bool
    regression_family: str
    train_multiclass: bool
    multiclass_family: str | None


@dataclass(frozen=True)
class OfflineAuxiliaryConfig:
    config_path: Path
    config_sha256: str
    payload: Mapping[str, Any]
    repository_root: Path
    report_directory: Path
    model_path: Path
    manifest_path: Path
    lightgbm_candidates: tuple[Mapping[str, int | float], ...]
    elasticnet_candidates: tuple[Mapping[str, int | float], ...]
    catboost_candidates: tuple[Mapping[str, int | float], ...]
    lightgbm_n_jobs: int
    catboost_thread_count: int
    elasticnet_max_iter: int
    elasticnet_tolerance: float


@dataclass(frozen=True)
class OfflineAuxiliaryInputs:
    repository_root: Path
    split_manifest: Mapping[str, Any]
    assignments: pd.DataFrame
    frames: Mapping[str, pd.DataFrame]
    input_bindings: Mapping[str, Mapping[str, Any]]
    upstream_protection: Mapping[str, Any]


@dataclass(frozen=True)
class OfflineFoldPreprocessor:
    input_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    selected_features: tuple[str, ...]
    category_vocabularies: Mapping[str, tuple[str, ...]]
    numeric_medians: Mapping[str, float]
    numeric_means: Mapping[str, float]
    numeric_scales: Mapping[str, float]
    output_columns: tuple[str, ...]
    standardize_numeric: bool

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        *,
        input_features: Sequence[str],
        categorical_features: Sequence[str],
        standardize_numeric: bool,
    ) -> OfflineFoldPreprocessor:
        features = tuple(input_features)
        categorical = tuple(categorical_features)
        if not features or len(set(features)) != len(features):
            raise OfflineAuxiliaryError("offline input feature order is invalid")
        if not set(categorical).issubset(features):
            raise OfflineAuxiliaryError("offline categorical feature set is invalid")
        missing = set(features) - set(frame.columns)
        if missing:
            raise OfflineAuxiliaryError(
                f"offline preprocessing columns missing: {missing}"
            )
        selected: list[str] = []
        vocabularies: dict[str, tuple[str, ...]] = {}
        medians: dict[str, float] = {}
        means: dict[str, float] = {}
        scales: dict[str, float] = {}
        output_columns: list[str] = []
        for feature in features:
            if feature in categorical:
                series = frame[feature]
                values = [
                    str(value)
                    for value in series.tolist()
                    if not _is_missing_scalar(value)
                ]
                vocabulary = tuple(
                    sorted(set(values), key=lambda value: value.encode("utf-8"))
                )
                observed_states = len(vocabulary) + int(series.isna().any())
                if observed_states < 2:
                    continue
                selected.append(feature)
                vocabularies[feature] = vocabulary
                for value in (*vocabulary, "__MISSING__", "__UNKNOWN__"):
                    output_columns.append(f"{feature}=={value}")
                continue
            values = pd.to_numeric(frame[feature], errors="coerce").to_numpy(
                dtype="float64"
            )
            finite = values[np.isfinite(values)]
            if finite.size < 2 or np.unique(finite).size < 2:
                continue
            median = float(np.median(finite))
            filled = np.where(np.isfinite(values), values, median)
            mean = float(np.mean(filled)) if standardize_numeric else 0.0
            scale = float(np.std(filled, ddof=0)) if standardize_numeric else 1.0
            if not math.isfinite(scale) or scale <= 0:
                scale = 1.0
            selected.append(feature)
            medians[feature] = median
            means[feature] = mean
            scales[feature] = scale
            output_columns.append(feature)
        if not selected or not output_columns:
            raise OfflineAuxiliaryError("offline preprocessing selected no features")
        return cls(
            input_features=features,
            categorical_features=categorical,
            selected_features=tuple(selected),
            category_vocabularies=vocabularies,
            numeric_medians=medians,
            numeric_means=means,
            numeric_scales=scales,
            output_columns=tuple(output_columns),
            standardize_numeric=bool(standardize_numeric),
        )

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if set(self.input_features) - set(frame.columns):
            raise OfflineAuxiliaryError("offline transform input is incomplete")
        columns: dict[str, np.ndarray] = {}
        for feature in self.selected_features:
            if feature in self.categorical_features:
                vocabulary = self.category_vocabularies[feature]
                encoded: list[str] = []
                allowed = set(vocabulary)
                for value in frame[feature].tolist():
                    if _is_missing_scalar(value):
                        encoded.append("__MISSING__")
                    else:
                        text = str(value)
                        encoded.append(text if text in allowed else "__UNKNOWN__")
                encoded_array = np.asarray(encoded, dtype=object)
                for value in (*vocabulary, "__MISSING__", "__UNKNOWN__"):
                    columns[f"{feature}=={value}"] = (encoded_array == value).astype(
                        "float64"
                    )
                continue
            values = np.array(
                pd.to_numeric(frame[feature], errors="coerce").to_numpy(
                    dtype="float64"
                ),
                dtype="float64",
                copy=True,
            )
            values[~np.isfinite(values)] = self.numeric_medians[feature]
            if self.standardize_numeric:
                values = (values - self.numeric_means[feature]) / self.numeric_scales[
                    feature
                ]
            columns[feature] = values
        result = pd.DataFrame(columns, index=frame.index)
        if tuple(result.columns) != self.output_columns:
            raise OfflineAuxiliaryError("offline preprocessing output order drifted")
        return result.astype("float64")

    def to_manifest(self) -> dict[str, Any]:
        return {
            "input_features": list(self.input_features),
            "categorical_features": list(self.categorical_features),
            "selected_features": list(self.selected_features),
            "category_vocabularies": {
                key: list(value) for key, value in self.category_vocabularies.items()
            },
            "numeric_medians": dict(self.numeric_medians),
            "numeric_means": dict(self.numeric_means),
            "numeric_scales": dict(self.numeric_scales),
            "output_columns": list(self.output_columns),
            "standardize_numeric": self.standardize_numeric,
        }


@dataclass(frozen=True)
class OfflineAuxiliarySubmodel:
    task_key: str
    dataset_id: str
    objective: str
    model_family: str
    input_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    preprocessor: OfflineFoldPreprocessor
    estimator: Any
    hyperparameters: Mapping[str, int | float]
    activity_ecdf: Any | None
    training_participant_count: int
    training_participant_sha256: str
    training_row_count: int

    def predict_frame(self, canonical_frame: pd.DataFrame) -> dict[str, np.ndarray]:
        spec = dataset_spec(self.dataset_id)
        materialized = materialize_features(
            canonical_frame,
            spec,
            activity_ecdf=self.activity_ecdf,
        )
        available = materialized.notna().any(axis=1).to_numpy(dtype=bool)
        if not available.any():
            if self.objective == "regression":
                return {
                    "available": available,
                    "prediction_normalized": np.full(len(materialized), np.nan),
                }
            return {
                "available": available,
                "probabilities": np.full((len(materialized), len(GRADE_IDS)), np.nan),
                "prediction_grade": np.full(len(materialized), -1, dtype="int64"),
            }
        matrix = self.preprocessor.transform(materialized.loc[available])
        if self.objective == "regression":
            prediction = np.full(len(materialized), np.nan, dtype="float64")
            prediction[available] = np.clip(
                np.asarray(self.estimator.predict(matrix), dtype="float64"), 0.0, 1.0
            )
            return {"available": available, "prediction_normalized": prediction}
        probabilities = np.full(
            (len(materialized), len(GRADE_IDS)), np.nan, dtype="float64"
        )
        raw = np.asarray(self.estimator.predict_proba(matrix), dtype="float64")
        probabilities[available] = _align_multiclass_probabilities(
            raw, tuple(int(value) for value in self.estimator.classes_)
        )
        grade = np.full(len(materialized), -1, dtype="int64")
        grade[available] = np.argmax(probabilities[available], axis=1)
        return {
            "available": available,
            "probabilities": probabilities,
            "prediction_grade": grade,
        }


@dataclass(frozen=True)
class OfflineAuxiliaryBundle:
    bundle_version: str
    training_version: str
    run_id: str
    split_id: str
    split_sha256: str
    feature_schema_version: str
    feature_schema_sha256: str
    status: str
    submodels: Mapping[str, OfflineAuxiliarySubmodel]

    def validate(self) -> None:
        if (
            self.bundle_version != BUNDLE_VERSION
            or self.training_version != TRAINING_VERSION
            or self.run_id != RUN_ID
            or self.split_id != SPLIT_ID
            or self.split_sha256 != SPLIT_SHA256
            or self.feature_schema_version != FEATURE_SCHEMA_VERSION
            or self.feature_schema_sha256 != FEATURE_SCHEMA_SHA256
            or self.status != "offline_auxiliary_not_active"
        ):
            raise OfflineAuxiliaryError("offline auxiliary bundle binding drifted")
        if tuple(self.submodels) != tuple(MODEL_FAMILY_BY_TASK):
            raise OfflineAuxiliaryError("offline auxiliary submodel order drifted")
        for task_key, submodel in self.submodels.items():
            if (
                submodel.task_key != task_key
                or submodel.model_family != MODEL_FAMILY_BY_TASK[task_key]
                or submodel.training_participant_count <= 0
                or submodel.training_row_count <= 0
                or not _is_sha256(submodel.training_participant_sha256)
            ):
                raise OfflineAuxiliaryError("offline auxiliary submodel is invalid")


@dataclass(frozen=True)
class _PreparedFold:
    train_frame: pd.DataFrame
    validation_frame: pd.DataFrame
    preprocessor: OfflineFoldPreprocessor
    train_matrix: pd.DataFrame
    validation_matrix: pd.DataFrame
    train_target: np.ndarray
    validation_target: np.ndarray
    train_weight: np.ndarray
    validation_weight: np.ndarray
    preprocessing_manifest: Mapping[str, Any]


def dataset_spec(dataset_id: str) -> OfflineDatasetSpec:
    if dataset_id not in DATASET_ORDER:
        raise OfflineAuxiliaryError(f"unsupported offline dataset: {dataset_id}")
    pairs = tuple(
        item
        for group in FEATURE_GROUPS_BY_DATASET[dataset_id]
        for item in _GROUP_FEATURE_SPECS[group]
    )
    input_features = tuple(name for name, _ in pairs)
    categorical = tuple(
        name for name, value_type in pairs if value_type == FeatureValueType.CATEGORY
    )
    regression_family = MODEL_FAMILY_BY_TASK[f"{dataset_id}::regression"]
    train_multiclass = dataset_id in MULTICLASS_DATASETS
    multiclass_family = (
        MODEL_FAMILY_BY_TASK[f"{dataset_id}::multiclass"] if train_multiclass else None
    )
    return OfflineDatasetSpec(
        dataset_id=dataset_id,
        score_column=SCORE_COLUMN_BY_DATASET[dataset_id],
        feature_groups=FEATURE_GROUPS_BY_DATASET[dataset_id],
        input_features=input_features,
        categorical_features=categorical,
        fit_activity_ecdf=dataset_id in ECDF_DATASETS,
        regression_family=regression_family,
        train_multiclass=train_multiclass,
        multiclass_family=multiclass_family,
    )


def phq9_grade(score: float | int) -> int:
    value = float(score)
    if not math.isfinite(value) or value < 0 or value > PHQ9_MAX_SCORE:
        raise OfflineAuxiliaryError("PHQ-9 score is outside 0..27")
    for index, (lower, upper) in enumerate(GRADE_BANDS):
        if lower <= value <= upper:
            return index
    raise OfflineAuxiliaryError("PHQ-9 grade mapping failed")


def regression_sample_weights(frame: pd.DataFrame) -> np.ndarray:
    if frame.empty or "global_participant_id" not in frame:
        raise OfflineAuxiliaryError("regression weight frame is invalid")
    participants = frame["global_participant_id"].astype(str)
    counts = participants.value_counts()
    weight = participants.map(lambda value: 1.0 / float(counts[value])).to_numpy(
        dtype="float64"
    )
    return _normalize_weights(weight)


def multiclass_sample_weights(frame: pd.DataFrame) -> np.ndarray:
    required = {"global_participant_id", "target_grade"}
    if frame.empty or not required.issubset(frame.columns):
        raise OfflineAuxiliaryError("multiclass weight frame is invalid")
    grades = pd.to_numeric(frame["target_grade"], errors="raise").astype("int64")
    present = sorted(set(grades.tolist()))
    if any(value not in GRADE_IDS for value in present):
        raise OfflineAuxiliaryError("multiclass target contains an invalid grade")
    participant = frame["global_participant_id"].astype(str)
    unit = participant + "::grade=" + grades.astype(str)
    class_total = 1.0 / len(present)
    weights = np.zeros(len(frame), dtype="float64")
    for grade in present:
        grade_mask = grades.to_numpy() == grade
        units = unit[grade_mask]
        unit_counts = units.value_counts()
        unit_total = class_total / len(unit_counts)
        indices = np.flatnonzero(grade_mask)
        for index, unit_id in zip(indices, units.tolist(), strict=True):
            weights[index] = unit_total / float(unit_counts[unit_id])
    return _normalize_weights(weights)


def regression_metrics(
    target: Sequence[float],
    prediction: Sequence[float],
    *,
    sample_weight: Sequence[float] | None = None,
) -> dict[str, float | None]:
    y = np.asarray(target, dtype="float64")
    p = np.asarray(prediction, dtype="float64")
    if y.shape != p.shape or y.ndim != 1 or y.size == 0:
        raise OfflineAuxiliaryError("regression metric inputs are invalid")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise OfflineAuxiliaryError("regression metric inputs are non-finite")
    weight = (
        np.ones(y.size, dtype="float64")
        if sample_weight is None
        else np.asarray(sample_weight, dtype="float64")
    )
    if weight.shape != y.shape or not np.isfinite(weight).all() or weight.sum() <= 0:
        raise OfflineAuxiliaryError("regression metric weights are invalid")
    weight = weight / weight.sum()
    error = p - y
    mae = float(np.sum(weight * np.abs(error)))
    rmse = float(np.sqrt(np.sum(weight * np.square(error))))
    correlation: float | None
    if np.unique(y).size < 2 or np.unique(p).size < 2:
        correlation = None
    else:
        result = spearmanr(y, p, nan_policy="raise")
        correlation = (
            float(result.statistic) if math.isfinite(result.statistic) else None
        )
    return {
        "mae_normalized": mae,
        "rmse_normalized": rmse,
        "mae_score": mae * PHQ9_MAX_SCORE,
        "rmse_score": rmse * PHQ9_MAX_SCORE,
        "spearman": correlation,
    }


def multiclass_metrics(
    target: Sequence[int],
    probabilities: np.ndarray,
    *,
    sample_weight: Sequence[float] | None = None,
) -> dict[str, Any]:
    y = np.asarray(target, dtype="int64")
    prob = np.asarray(probabilities, dtype="float64")
    if y.ndim != 1 or y.size == 0 or prob.shape != (y.size, len(GRADE_IDS)):
        raise OfflineAuxiliaryError("multiclass metric inputs are invalid")
    if not np.isfinite(prob).all() or np.any(prob < 0):
        raise OfflineAuxiliaryError("multiclass probabilities are invalid")
    row_sum = prob.sum(axis=1)
    if not np.allclose(row_sum, 1.0, atol=1e-8, rtol=0):
        raise OfflineAuxiliaryError("multiclass probabilities do not sum to one")
    weight = (
        np.ones(y.size, dtype="float64")
        if sample_weight is None
        else np.asarray(sample_weight, dtype="float64")
    )
    predicted = np.argmax(prob, axis=1).astype("int64")
    precision, recall, f1, support = precision_recall_fscore_support(
        y,
        predicted,
        labels=list(GRADE_IDS),
        sample_weight=weight,
        zero_division=0,
    )
    matrix = confusion_matrix(
        y, predicted, labels=list(GRADE_IDS), sample_weight=weight
    )
    return {
        "macro_f1": float(
            f1_score(
                y,
                predicted,
                labels=list(GRADE_IDS),
                average="macro",
                sample_weight=weight,
                zero_division=0,
            )
        ),
        "accuracy": float(accuracy_score(y, predicted, sample_weight=weight)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y, predicted, sample_weight=weight)
        ),
        "log_loss": float(
            log_loss(y, prob, labels=list(GRADE_IDS), sample_weight=weight)
        ),
        "ordinal_mae": float(np.average(np.abs(predicted - y), weights=weight)),
        "per_class": [
            {
                "grade": grade,
                "precision": float(precision[grade]),
                "recall": float(recall[grade]),
                "f1": float(f1[grade]),
                "weighted_support": float(support[grade]),
            }
            for grade in GRADE_IDS
        ],
        "confusion_matrix": matrix.tolist(),
    }


def load_offline_auxiliary_config(
    path: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> OfflineAuxiliaryConfig:
    config_path = Path(path).resolve()
    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else _find_repository_root(config_path)
    )
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise OfflineAuxiliaryError("MODEL-006 configuration is unreadable") from exc
    if not isinstance(payload, dict):
        raise OfflineAuxiliaryError("MODEL-006 configuration must be a mapping")
    if (
        payload.get("training_version") != TRAINING_VERSION
        or payload.get("task_id") != TASK_ID
        or payload.get("run_id") != RUN_ID
        or payload.get("frozen_document_version") != "V3.3.3"
        or int(payload.get("random_seed", -1)) != RANDOM_SEED
    ):
        raise OfflineAuxiliaryError("MODEL-006 top-level binding changed")
    split = payload.get("split")
    if not isinstance(split, dict) or (
        split.get("split_id") != SPLIT_ID
        or split.get("sha256") != SPLIT_SHA256
        or split.get("feature_schema_version") != FEATURE_SCHEMA_VERSION
        or split.get("feature_schema_sha256") != FEATURE_SCHEMA_SHA256
        or int(split.get("outer_fold_count", -1)) != OUTER_FOLD_COUNT
        or int(split.get("inner_fold_count", -1)) != INNER_FOLD_COUNT
    ):
        raise OfflineAuxiliaryError("MODEL-006 split binding changed")
    upstream = payload.get("upstream_evaluation")
    if not isinstance(upstream, dict) or (
        upstream.get("run_id") != EVALUATION_RUN_ID
        or upstream.get("report_core_sha256") != EVALUATION_CORE_SHA256
        or upstream.get("baseline_protection_sha256") != BASELINE_PROTECTION_SHA256
    ):
        raise OfflineAuxiliaryError("MODEL-006 upstream evaluation binding changed")
    targets = payload.get("targets")
    if not isinstance(targets, dict):
        raise OfflineAuxiliaryError("MODEL-006 target policy is missing")
    continuous = targets.get("continuous")
    five_level = targets.get("five_level")
    expected_bands = [list(band) for band in GRADE_BANDS]
    if (
        not isinstance(continuous, dict)
        or continuous.get("formula") != "phq9_total / 27"
        or float(continuous.get("minimum", -1)) != 0.0
        or float(continuous.get("maximum", -1)) != 1.0
        or not isinstance(five_level, dict)
        or five_level.get("bands") != expected_bands
        or five_level.get("class_ids") != list(GRADE_IDS)
    ):
        raise OfflineAuxiliaryError("MODEL-006 target definition changed")
    datasets = payload.get("datasets")
    if not isinstance(datasets, dict) or tuple(datasets) != DATASET_ORDER:
        raise OfflineAuxiliaryError("MODEL-006 dataset order changed")
    for dataset_id in DATASET_ORDER:
        item = datasets[dataset_id]
        spec = dataset_spec(dataset_id)
        expected_multiclass = spec.multiclass_family
        if not isinstance(item, dict) or (
            item.get("score_column") != spec.score_column
            or tuple(item.get("feature_groups", ())) != spec.feature_groups
            or bool(item.get("fit_activity_ecdf")) != spec.fit_activity_ecdf
            or item.get("regression_family") != spec.regression_family
            or bool(item.get("train_five_level")) != spec.train_multiclass
            or item.get("multiclass_family") != expected_multiclass
        ):
            raise OfflineAuxiliaryError(
                f"MODEL-006 dataset policy changed for {dataset_id}"
            )
    feature_policy = payload.get("feature_policy")
    required_true = (
        "use_only_frozen_deployment_compatible_schema_fields",
        "all_preprocessing_fit_inside_training_fold",
    )
    required_false = (
        "feature_coverage_is_model_input",
        "feature_mask_is_model_input",
        "missing_indicator_is_model_input",
        "scale_items_are_model_inputs",
        "publisher_labels_are_model_inputs",
        "ssq_specific_fields_are_model_inputs",
        "source_identifier_is_model_input",
    )
    if (
        not isinstance(feature_policy, dict)
        or any(not bool(feature_policy.get(field)) for field in required_true)
        or any(bool(feature_policy.get(field)) for field in required_false)
    ):
        raise OfflineAuxiliaryError("MODEL-006 feature boundary changed")
    weighting = payload.get("sample_weighting")
    if not isinstance(weighting, dict) or (
        weighting.get("regression") != "participant_equal_then_repeat_window_equal"
        or weighting.get("multiclass")
        != "class_equal_then_participant_class_equal_then_repeat_window_equal"
        or not bool(weighting.get("normalize_mean_to_one"))
        or bool(weighting.get("smote"))
    ):
        raise OfflineAuxiliaryError("MODEL-006 sample weighting changed")
    search = payload.get("search")
    if not isinstance(search, dict):
        raise OfflineAuxiliaryError("MODEL-006 search policy is missing")
    lightgbm_config = search.get("lightgbm")
    elastic_config = search.get("elasticnet_regression")
    catboost_config = search.get("catboost_regression")
    expected_lgbm_grid = {
        "n_estimators": [200, 400],
        "learning_rate": [0.03, 0.05],
        "num_leaves": [7, 15, 31],
        "max_depth": [3, 5, -1],
        "min_child_samples": [20, 50],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "reg_alpha": [0.0, 0.1],
        "reg_lambda": [0.1, 1.0],
    }
    expected_elastic_grid = {
        "alpha": [0.001, 0.01, 0.1, 1.0],
        "l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0],
    }
    expected_catboost_grid = {
        "iterations": [300, 600],
        "depth": [4, 6],
        "learning_rate": [0.03, 0.05],
        "l2_leaf_reg": [3, 10],
    }
    if not isinstance(lightgbm_config, dict) or (
        lightgbm_config.get("traversal") != "deterministic_hash_subset_of_frozen_grid"
        or int(lightgbm_config.get("candidate_count", -1)) != 8
        or int(lightgbm_config.get("n_jobs", -1)) != 1
        or lightgbm_config.get("grid") != expected_lgbm_grid
    ):
        raise OfflineAuxiliaryError("MODEL-006 LightGBM search changed")
    if not isinstance(elastic_config, dict) or (
        elastic_config.get("traversal") != "deterministic_full_grid"
        or int(elastic_config.get("max_iter", -1)) != 5000
        or float(elastic_config.get("tolerance", -1)) != 1.0e-6
        or elastic_config.get("grid") != expected_elastic_grid
    ):
        raise OfflineAuxiliaryError("MODEL-006 ElasticNet search changed")
    if not isinstance(catboost_config, dict) or (
        catboost_config.get("traversal") != "deterministic_full_grid"
        or int(catboost_config.get("thread_count", -1)) != 1
        or catboost_config.get("grid") != expected_catboost_grid
    ):
        raise OfflineAuxiliaryError("MODEL-006 CatBoost search changed")
    selection = payload.get("selection")
    if not isinstance(selection, dict) or (
        selection.get("regression")
        != "lowest_pooled_inner_weighted_rmse_then_mae_then_candidate_id"
        or selection.get("multiclass")
        != "highest_pooled_inner_weighted_macro_f1_then_log_loss_then_candidate_id"
        or selection.get("outer_test_fold_role") != "evaluation_only"
        or selection.get("probability_calibration") != "none_offline_auxiliary"
    ):
        raise OfflineAuxiliaryError("MODEL-006 selection policy changed")
    output = payload.get("output")
    if not isinstance(output, dict) or (
        output.get("model_status") != "offline_auxiliary_not_active"
        or bool(output.get("overwrite_active_experts"))
        or bool(output.get("enter_fusion"))
        or bool(output.get("enter_api"))
    ):
        raise OfflineAuxiliaryError("MODEL-006 output boundary changed")
    lgbm_candidates = deterministic_candidates(
        expected_lgbm_grid,
        candidate_count=8,
        seed=RANDOM_SEED,
        prefix="lgbm",
    )
    elastic_candidates = deterministic_candidates(
        expected_elastic_grid,
        candidate_count=math.prod(
            len(value) for value in expected_elastic_grid.values()
        ),
        seed=RANDOM_SEED,
        prefix="elasticnet",
    )
    catboost_candidates = deterministic_candidates(
        expected_catboost_grid,
        candidate_count=math.prod(
            len(value) for value in expected_catboost_grid.values()
        ),
        seed=RANDOM_SEED,
        prefix="catboost",
    )
    return OfflineAuxiliaryConfig(
        config_path=config_path,
        config_sha256=_sha256_file(config_path),
        payload=payload,
        repository_root=root,
        report_directory=root / str(output["report_directory"]),
        model_path=root / str(output["model_path"]),
        manifest_path=root / str(output["manifest_path"]),
        lightgbm_candidates=tuple(item["params"] for item in lgbm_candidates),
        elasticnet_candidates=tuple(item["params"] for item in elastic_candidates),
        catboost_candidates=tuple(item["params"] for item in catboost_candidates),
        lightgbm_n_jobs=int(lightgbm_config["n_jobs"]),
        catboost_thread_count=int(catboost_config["thread_count"]),
        elasticnet_max_iter=int(elastic_config["max_iter"]),
        elasticnet_tolerance=float(elastic_config["tolerance"]),
    )


def deterministic_candidates(
    grid: Mapping[str, Sequence[int | float]],
    *,
    candidate_count: int,
    seed: int,
    prefix: str,
) -> list[dict[str, Any]]:
    keys = tuple(grid)
    combinations = [
        dict(zip(keys, values, strict=True))
        for values in itertools.product(*(grid[key] for key in keys))
    ]
    ranked: list[tuple[str, dict[str, int | float]]] = []
    for params in combinations:
        payload = {"prefix": prefix, "seed": int(seed), "params": params}
        digest = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
        ranked.append((digest, params))
    ranked.sort(key=lambda item: item[0])
    if candidate_count <= 0 or candidate_count > len(ranked):
        raise OfflineAuxiliaryError("offline search candidate count is invalid")
    return [
        {"candidate_id": f"{prefix}-{digest[:16]}", "params": dict(params)}
        for digest, params in ranked[:candidate_count]
    ]


def load_offline_auxiliary_inputs(
    config: OfflineAuxiliaryConfig,
) -> OfflineAuxiliaryInputs:
    root = config.repository_root
    upstream_config_path = root / str(
        config.payload["upstream_evaluation"]["config_path"]
    )
    evaluation_config = load_evaluation_config(upstream_config_path)
    protection = validate_baseline_protection(root, evaluation_config)
    protection_sha = _canonical_payload_sha256(protection)
    if protection_sha != BASELINE_PROTECTION_SHA256:
        raise OfflineAuxiliaryError("five-expert protection summary drifted")
    report_path = root / str(config.payload["upstream_evaluation"]["report_path"])
    report_core = _report_core_sha256(report_path)
    if report_core != EVALUATION_CORE_SHA256:
        raise OfflineAuxiliaryError("EVAL-001 report core drifted")
    split_path = root / str(config.payload["split"]["relative_path"])
    if _sha256_file(split_path) != SPLIT_SHA256:
        raise OfflineAuxiliaryError("DATA-007 split hash drifted")
    try:
        split = json.loads(split_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OfflineAuxiliaryError("DATA-007 split is unreadable") from exc
    if (
        split.get("split_id") != SPLIT_ID
        or split.get("feature_schema_binding", {}).get("snapshot_sha256")
        != FEATURE_SCHEMA_SHA256
    ):
        raise OfflineAuxiliaryError("DATA-007 split binding drifted")
    assignments = pd.DataFrame(split.get("participant_assignments", []))
    required_assignment = {
        "dataset_id",
        "global_participant_id",
        "outer_fold",
        "inner_validation_fold_by_outer_fold",
    }
    if assignments.empty or not required_assignment.issubset(assignments.columns):
        raise OfflineAuxiliaryError("DATA-007 participant assignments are invalid")
    if assignments["global_participant_id"].duplicated().any():
        raise OfflineAuxiliaryError("DATA-007 participant assignments are duplicated")
    input_bindings: dict[str, Mapping[str, Any]] = {}
    frames: dict[str, pd.DataFrame] = {}
    for binding in split.get("inputs", []):
        dataset_id = str(binding["dataset_id"])
        if dataset_id not in DATASET_ORDER:
            raise OfflineAuxiliaryError("DATA-007 contains an unexpected source")
        for prefix in ("canonical", "mapping", "adapter_metadata", "artifact_manifest"):
            path = root / str(binding[f"{prefix}_relative_path"])
            if _sha256_file(path) != str(binding[f"{prefix}_sha256"]):
                raise OfflineAuxiliaryError(f"{dataset_id} {prefix} hash drifted")
        canonical_path = root / str(binding["canonical_relative_path"])
        frame = pd.read_parquet(canonical_path)
        if len(frame) != int(binding["row_count"]):
            raise OfflineAuxiliaryError(f"{dataset_id} canonical row count drifted")
        if set(frame["dataset_id"].astype(str)) != {dataset_id}:
            raise OfflineAuxiliaryError(f"{dataset_id} canonical identity drifted")
        frame = frame.reset_index(drop=True)
        frames[dataset_id] = frame
        input_bindings[dataset_id] = binding
    if tuple(frames) != DATASET_ORDER:
        raise OfflineAuxiliaryError("MODEL-006 canonical source order drifted")
    assignment_ids = set(assignments["global_participant_id"].astype(str))
    canonical_ids = {
        value
        for frame in frames.values()
        for value in frame["global_participant_id"].astype(str)
    }
    if assignment_ids != canonical_ids:
        raise OfflineAuxiliaryError("canonical participants do not match DATA-007")
    return OfflineAuxiliaryInputs(
        repository_root=root,
        split_manifest=split,
        assignments=assignments,
        frames=frames,
        input_bindings=input_bindings,
        upstream_protection=protection,
    )


def materialize_features(
    frame: pd.DataFrame,
    spec: OfflineDatasetSpec,
    *,
    activity_ecdf: Any | None,
) -> pd.DataFrame:
    source = frame
    if spec.fit_activity_ecdf:
        if activity_ecdf is None:
            raise OfflineAuxiliaryError("fold-fitted activity ECDF is required")
        canonical_columns = _CANONICAL_ORDER_BY_DATASET[spec.dataset_id]()
        missing_canonical = set(canonical_columns) - set(frame.columns)
        if missing_canonical:
            raise OfflineAuxiliaryError("ECDF canonical partition is incomplete")
        canonical = frame.loc[:, canonical_columns]
        source = activity_ecdf.transform(canonical)
    elif activity_ecdf is not None:
        raise OfflineAuxiliaryError(
            "unexpected activity ECDF for cross-sectional source"
        )
    missing: set[str] = set()
    result: dict[str, pd.Series] = {}
    for feature in spec.input_features:
        mask_column = f"feature_mask.{feature}"
        if feature not in source.columns or mask_column not in source.columns:
            missing.update({feature, mask_column} - set(source.columns))
            continue
        mask = pd.to_numeric(source[mask_column], errors="coerce").fillna(0).eq(1)
        values = source[feature].copy()
        values.loc[~mask] = np.nan
        result[feature] = values
    if missing:
        raise OfflineAuxiliaryError(
            f"offline canonical feature columns missing: {missing}"
        )
    return pd.DataFrame(result, index=frame.index)


def build_task_frame(
    inputs: OfflineAuxiliaryInputs,
    dataset_id: str,
) -> pd.DataFrame:
    spec = dataset_spec(dataset_id)
    frame = inputs.frames[dataset_id].copy()
    frame["canonical_row_index"] = np.arange(len(frame), dtype="int64")
    frame["prediction_id"] = (
        frame["dataset_id"].astype(str)
        + "::row="
        + frame["canonical_row_index"].astype(str)
    )
    score = pd.to_numeric(frame[spec.score_column], errors="raise").to_numpy(
        dtype="float64"
    )
    if not np.isfinite(score).all() or np.any(score < 0) or np.any(score > 27):
        raise OfflineAuxiliaryError(f"{dataset_id} PHQ-9 target is invalid")
    rounded = np.rint(score)
    if not np.allclose(score, rounded, atol=0, rtol=0):
        raise OfflineAuxiliaryError(f"{dataset_id} PHQ-9 target is not integral")
    frame["target_score"] = rounded.astype("int64")
    frame["target_normalized"] = frame["target_score"].astype("float64") / 27.0
    frame["target_grade"] = frame["target_score"].map(phq9_grade).astype("int64")
    assignment = inputs.assignments[
        inputs.assignments["dataset_id"].astype(str).eq(dataset_id)
    ][
        [
            "global_participant_id",
            "outer_fold",
            "inner_validation_fold_by_outer_fold",
        ]
    ]
    frame = frame.merge(
        assignment,
        on="global_participant_id",
        how="left",
        validate="many_to_one",
    )
    if frame["outer_fold"].isna().any():
        raise OfflineAuxiliaryError(f"{dataset_id} split assignment is incomplete")
    frame["outer_fold"] = frame["outer_fold"].astype("int64")
    return frame.sort_values("canonical_row_index", kind="stable").reset_index(
        drop=True
    )


def train_offline_auxiliary_models(
    repository_root: str | Path,
    config_path: str | Path,
    *,
    overwrite: bool = False,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Train and atomically publish all MODEL-006 report-only submodels."""

    started = datetime.now(timezone.utc)
    root = Path(repository_root).resolve()
    config = load_offline_auxiliary_config(
        config_path,
        repository_root=root,
    )
    _validate_output_paths(config)
    _assert_outputs_available(config, overwrite=overwrite)
    inputs = load_offline_auxiliary_inputs(config)
    protection_before = _canonical_payload_sha256(inputs.upstream_protection)
    oof_parts: list[pd.DataFrame] = []
    task_metrics: list[dict[str, Any]] = []
    search_tasks: list[dict[str, Any]] = []
    preprocessing_tasks: list[dict[str, Any]] = []
    warning_rows: list[dict[str, Any]] = []
    submodels: dict[str, OfflineAuxiliarySubmodel] = {}

    for task_key, model_family in MODEL_FAMILY_BY_TASK.items():
        dataset_id, objective = task_key.split("::", 1)
        spec = dataset_spec(dataset_id)
        task_frame = build_task_frame(inputs, dataset_id)
        outer_results: list[dict[str, Any]] = []
        task_oof_parts: list[pd.DataFrame] = []
        task_preprocessing: list[dict[str, Any]] = []
        for outer_fold in range(OUTER_FOLD_COUNT):
            result = _train_outer_fold(
                config,
                inputs,
                task_frame,
                spec,
                objective=objective,
                model_family=model_family,
                outer_fold=outer_fold,
            )
            outer_results.append(result["search"])
            task_oof_parts.append(result["oof"])
            task_preprocessing.append(result["preprocessing"])
            warning_rows.extend(result["warnings"])
        task_oof = pd.concat(task_oof_parts, ignore_index=True)
        task_oof = task_oof.sort_values(
            ["outer_fold", "canonical_row_index"], kind="stable"
        ).reset_index(drop=True)
        if len(task_oof) != len(task_frame):
            raise OfflineAuxiliaryError(f"{task_key} OOF row coverage changed")
        if task_oof["prediction_id"].duplicated().any():
            raise OfflineAuxiliaryError(f"{task_key} OOF prediction IDs are duplicated")
        final_selection = _select_production_candidate(
            outer_results,
            objective=objective,
        )
        production = _fit_production_submodel(
            config,
            inputs,
            task_frame,
            spec,
            objective=objective,
            model_family=model_family,
            params=final_selection["params"],
            task_key=task_key,
        )
        submodels[task_key] = production["submodel"]
        warning_rows.extend(production["warnings"])
        task_preprocessing.append(production["preprocessing"])
        metrics = _task_oof_metrics(task_oof, objective=objective)
        metrics.update(
            {
                "task_key": task_key,
                "dataset_id": dataset_id,
                "objective": objective,
                "model_family": model_family,
                "row_count": len(task_oof),
                "participant_count": int(task_oof["global_participant_id"].nunique()),
                "available_row_count": int(task_oof["available"].sum()),
                "unavailable_row_count": int((~task_oof["available"]).sum()),
                "final_candidate_id": final_selection["candidate_id"],
                "final_hyperparameters": dict(final_selection["params"]),
            }
        )
        task_metrics.append(metrics)
        search_tasks.append(
            {
                "task_key": task_key,
                "dataset_id": dataset_id,
                "objective": objective,
                "model_family": model_family,
                "outer_searches": outer_results,
                "production_selection": final_selection,
            }
        )
        preprocessing_tasks.append(
            {
                "task_key": task_key,
                "folds": task_preprocessing,
            }
        )
        oof_parts.append(task_oof)

    bundle = OfflineAuxiliaryBundle(
        bundle_version=BUNDLE_VERSION,
        training_version=TRAINING_VERSION,
        run_id=RUN_ID,
        split_id=SPLIT_ID,
        split_sha256=SPLIT_SHA256,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        feature_schema_sha256=FEATURE_SCHEMA_SHA256,
        status="offline_auxiliary_not_active",
        submodels=submodels,
    )
    bundle.validate()
    all_oof = pd.concat(oof_parts, ignore_index=True)
    all_oof["_task_order"] = all_oof["task_key"].map(
        {task: index for index, task in enumerate(MODEL_FAMILY_BY_TASK)}
    )
    all_oof = all_oof.sort_values(
        ["_task_order", "outer_fold", "canonical_row_index"], kind="stable"
    ).drop(columns="_task_order")
    all_oof = all_oof.reset_index(drop=True)
    protection_after = validate_baseline_protection(
        root,
        load_evaluation_config(
            root / str(config.payload["upstream_evaluation"]["config_path"])
        ),
    )
    if _canonical_payload_sha256(protection_after) != protection_before:
        raise OfflineAuxiliaryError("active expert protection changed during MODEL-006")
    ended = datetime.now(timezone.utc)
    return _publish_training_run(
        config,
        inputs,
        bundle,
        all_oof,
        task_metrics,
        search_tasks,
        preprocessing_tasks,
        warning_rows,
        protection_before,
        started=started,
        ended=ended,
        overwrite=overwrite,
        command=command or (),
    )


def _train_outer_fold(
    config: OfflineAuxiliaryConfig,
    inputs: OfflineAuxiliaryInputs,
    task_frame: pd.DataFrame,
    spec: OfflineDatasetSpec,
    *,
    objective: str,
    model_family: str,
    outer_fold: int,
) -> dict[str, Any]:
    outer_train = task_frame[task_frame["outer_fold"].ne(outer_fold)].copy()
    outer_test = task_frame[task_frame["outer_fold"].eq(outer_fold)].copy()
    if outer_train.empty or outer_test.empty:
        raise OfflineAuxiliaryError("outer fold partition is empty")
    prepared_inner: list[_PreparedFold] = []
    for inner_fold in range(INNER_FOLD_COUNT):
        inner_values = outer_train["inner_validation_fold_by_outer_fold"].map(
            lambda value: _inner_fold_value(value, outer_fold)
        )
        inner_train = outer_train[inner_values.ne(inner_fold)].copy()
        inner_validation = outer_train[inner_values.eq(inner_fold)].copy()
        prepared_inner.append(
            _prepare_fold(
                config,
                inputs,
                spec,
                inner_train,
                inner_validation,
                objective=objective,
                model_family=model_family,
                fold_identity=f"outer={outer_fold}/inner={inner_fold}",
            )
        )
    candidate_rows: list[dict[str, Any]] = []
    all_warnings: list[dict[str, Any]] = []
    for candidate in _candidate_records(config, model_family):
        predictions: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        validation_frames: list[pd.DataFrame] = []
        for inner_fold, prepared in enumerate(prepared_inner):
            estimator, fit_warnings = _fit_estimator(
                model_family,
                candidate["params"],
                prepared.train_matrix,
                prepared.train_target,
                prepared.train_weight,
                config,
                training_identity=(
                    f"{spec.dataset_id}/{objective}/outer={outer_fold}/"
                    f"inner={inner_fold}/{candidate['candidate_id']}"
                ),
            )
            all_warnings.extend(
                _warning_records(
                    fit_warnings,
                    task_key=f"{spec.dataset_id}::{objective}",
                    stage=f"outer={outer_fold}/inner={inner_fold}",
                    candidate_id=candidate["candidate_id"],
                )
            )
            prediction = _predict_estimator(
                estimator,
                prepared.validation_matrix,
                objective=objective,
            )
            predictions.append(prediction)
            targets.append(prepared.validation_target)
            validation_frames.append(prepared.validation_frame)
        pooled_frame = pd.concat(validation_frames, ignore_index=True)
        pooled_target = np.concatenate(targets)
        pooled_prediction = (
            np.concatenate(predictions)
            if objective == "regression"
            else np.vstack(predictions)
        )
        pooled_weight = _task_sample_weights(pooled_frame, objective)
        metrics = (
            regression_metrics(
                pooled_target,
                pooled_prediction,
                sample_weight=pooled_weight,
            )
            if objective == "regression"
            else multiclass_metrics(
                pooled_target.astype("int64"),
                pooled_prediction,
                sample_weight=pooled_weight,
            )
        )
        candidate_rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "params": dict(candidate["params"]),
                "pooled_inner_row_count": len(pooled_frame),
                "pooled_inner_participant_count": int(
                    pooled_frame["global_participant_id"].nunique()
                ),
                "metrics": metrics,
            }
        )
    selected = _rank_candidates(candidate_rows, objective=objective)[0]
    outer_prepared = _prepare_fold(
        config,
        inputs,
        spec,
        outer_train,
        outer_test,
        objective=objective,
        model_family=model_family,
        fold_identity=f"outer={outer_fold}/selected",
    )
    estimator, fit_warnings = _fit_estimator(
        model_family,
        selected["params"],
        outer_prepared.train_matrix,
        outer_prepared.train_target,
        outer_prepared.train_weight,
        config,
        training_identity=(
            f"{spec.dataset_id}/{objective}/outer={outer_fold}/"
            f"selected={selected['candidate_id']}"
        ),
    )
    all_warnings.extend(
        _warning_records(
            fit_warnings,
            task_key=f"{spec.dataset_id}::{objective}",
            stage=f"outer={outer_fold}/selected",
            candidate_id=selected["candidate_id"],
        )
    )
    prediction = _predict_estimator(
        estimator,
        outer_prepared.validation_matrix,
        objective=objective,
    )
    oof = _outer_oof_frame(
        outer_test,
        outer_prepared.validation_frame,
        prediction,
        dataset_id=spec.dataset_id,
        objective=objective,
        model_family=model_family,
        outer_fold=outer_fold,
    )
    return {
        "search": {
            "outer_fold": outer_fold,
            "outer_train_participant_count": int(
                outer_train["global_participant_id"].nunique()
            ),
            "outer_test_participant_count": int(
                outer_test["global_participant_id"].nunique()
            ),
            "candidate_count": len(candidate_rows),
            "candidates": candidate_rows,
            "selected_candidate_id": selected["candidate_id"],
            "selection_rule": _selection_rule(objective),
            "outer_test_metrics_not_used_for_selection": True,
        },
        "oof": oof,
        "preprocessing": {
            "scope": f"outer={outer_fold}",
            **outer_prepared.preprocessing_manifest,
        },
        "warnings": all_warnings,
    }


def _prepare_fold(
    config: OfflineAuxiliaryConfig,
    inputs: OfflineAuxiliaryInputs,
    spec: OfflineDatasetSpec,
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    *,
    objective: str,
    model_family: str,
    fold_identity: str,
) -> _PreparedFold:
    if train_frame.empty or validation_frame.empty:
        raise OfflineAuxiliaryError(f"{fold_identity} partition is empty")
    train_ids = set(train_frame["global_participant_id"].astype(str))
    validation_ids = set(validation_frame["global_participant_id"].astype(str))
    if train_ids & validation_ids:
        raise OfflineAuxiliaryError(f"{fold_identity} participant leakage detected")
    activity_ecdf = _fit_activity_ecdf(inputs, spec, train_ids)
    train_features = materialize_features(
        train_frame,
        spec,
        activity_ecdf=activity_ecdf,
    )
    validation_features = materialize_features(
        validation_frame,
        spec,
        activity_ecdf=activity_ecdf,
    )
    train_available = train_features.notna().any(axis=1)
    validation_available = validation_features.notna().any(axis=1)
    train_used = train_frame.loc[train_available].copy().reset_index(drop=True)
    validation_used = (
        validation_frame.loc[validation_available].copy().reset_index(drop=True)
    )
    train_features = train_features.loc[train_available].reset_index(drop=True)
    validation_features = validation_features.loc[validation_available].reset_index(
        drop=True
    )
    if train_used.empty or validation_used.empty:
        raise OfflineAuxiliaryError(f"{fold_identity} has no evaluable rows")
    standardize = model_family == "elasticnet_regression"
    preprocessor = OfflineFoldPreprocessor.fit(
        train_features,
        input_features=spec.input_features,
        categorical_features=spec.categorical_features,
        standardize_numeric=standardize,
    )
    train_matrix = preprocessor.transform(train_features)
    validation_matrix = preprocessor.transform(validation_features)
    target_column = "target_normalized" if objective == "regression" else "target_grade"
    train_target = pd.to_numeric(train_used[target_column], errors="raise").to_numpy(
        dtype="float64" if objective == "regression" else "int64"
    )
    validation_target = pd.to_numeric(
        validation_used[target_column], errors="raise"
    ).to_numpy(dtype="float64" if objective == "regression" else "int64")
    train_weight = _task_sample_weights(train_used, objective)
    validation_weight = _task_sample_weights(validation_used, objective)
    if objective == "multiclass" and len(np.unique(train_target)) < 2:
        raise OfflineAuxiliaryError(f"{fold_identity} multiclass train is single-class")
    return _PreparedFold(
        train_frame=train_used,
        validation_frame=validation_used,
        preprocessor=preprocessor,
        train_matrix=train_matrix,
        validation_matrix=validation_matrix,
        train_target=train_target,
        validation_target=validation_target,
        train_weight=train_weight,
        validation_weight=validation_weight,
        preprocessing_manifest={
            "fold_identity": fold_identity,
            "fit_participant_count": len(train_ids),
            "fit_participant_sha256": _string_set_sha256(train_ids),
            "validation_participant_count": len(validation_ids),
            "validation_participant_sha256": _string_set_sha256(validation_ids),
            "fit_row_count_before_availability": len(train_frame),
            "fit_row_count": len(train_used),
            "validation_row_count_before_availability": len(validation_frame),
            "validation_row_count": len(validation_used),
            "activity_ecdf": _ecdf_manifest(activity_ecdf),
            "preprocessor": preprocessor.to_manifest(),
        },
    )


def _fit_production_submodel(
    config: OfflineAuxiliaryConfig,
    inputs: OfflineAuxiliaryInputs,
    task_frame: pd.DataFrame,
    spec: OfflineDatasetSpec,
    *,
    objective: str,
    model_family: str,
    params: Mapping[str, int | float],
    task_key: str,
) -> dict[str, Any]:
    participant_ids = set(task_frame["global_participant_id"].astype(str))
    activity_ecdf = _fit_activity_ecdf(inputs, spec, participant_ids)
    features = materialize_features(
        task_frame,
        spec,
        activity_ecdf=activity_ecdf,
    )
    available = features.notna().any(axis=1)
    used = task_frame.loc[available].copy().reset_index(drop=True)
    feature_used = features.loc[available].reset_index(drop=True)
    preprocessor = OfflineFoldPreprocessor.fit(
        feature_used,
        input_features=spec.input_features,
        categorical_features=spec.categorical_features,
        standardize_numeric=model_family == "elasticnet_regression",
    )
    matrix = preprocessor.transform(feature_used)
    target_column = "target_normalized" if objective == "regression" else "target_grade"
    target = pd.to_numeric(used[target_column], errors="raise").to_numpy(
        dtype="float64" if objective == "regression" else "int64"
    )
    weight = _task_sample_weights(used, objective)
    candidate_id = _candidate_id(model_family, params)
    estimator, fit_warnings = _fit_estimator(
        model_family,
        params,
        matrix,
        target,
        weight,
        config,
        training_identity=f"{task_key}/production/{candidate_id}",
    )
    submodel = OfflineAuxiliarySubmodel(
        task_key=task_key,
        dataset_id=spec.dataset_id,
        objective=objective,
        model_family=model_family,
        input_features=spec.input_features,
        categorical_features=spec.categorical_features,
        preprocessor=preprocessor,
        estimator=estimator,
        hyperparameters=dict(params),
        activity_ecdf=activity_ecdf,
        training_participant_count=len(participant_ids),
        training_participant_sha256=_string_set_sha256(participant_ids),
        training_row_count=len(used),
    )
    return {
        "submodel": submodel,
        "warnings": _warning_records(
            fit_warnings,
            task_key=task_key,
            stage="production",
            candidate_id=candidate_id,
        ),
        "preprocessing": {
            "scope": "production",
            "fit_participant_count": len(participant_ids),
            "fit_participant_sha256": _string_set_sha256(participant_ids),
            "fit_row_count_before_availability": len(task_frame),
            "fit_row_count": len(used),
            "activity_ecdf": _ecdf_manifest(activity_ecdf),
            "preprocessor": preprocessor.to_manifest(),
        },
    }


def _fit_activity_ecdf(
    inputs: OfflineAuxiliaryInputs,
    spec: OfflineDatasetSpec,
    training_participant_ids: set[str],
) -> Any | None:
    if not spec.fit_activity_ecdf:
        return None
    binding = inputs.input_bindings[spec.dataset_id]
    participants = sorted(training_participant_ids, key=lambda value: value.encode())
    return _ECDF_CLASS_BY_DATASET[spec.dataset_id].fit(
        inputs.frames[spec.dataset_id],
        training_participant_ids=participants,
        split_id=SPLIT_ID,
        canonical_artifact_sha256=str(binding["canonical_sha256"]),
        expected_canonical_frame_sha256=str(binding["canonical_frame_sha256"]),
    )


def _fit_estimator(
    model_family: str,
    params: Mapping[str, int | float],
    matrix: pd.DataFrame,
    target: np.ndarray,
    sample_weight: np.ndarray,
    config: OfflineAuxiliaryConfig,
    *,
    training_identity: str,
) -> tuple[Any, list[warnings.WarningMessage]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if model_family == "lightgbm_regressor":
            estimator = LGBMRegressor(
                objective="regression_l2",
                random_state=RANDOM_SEED,
                n_jobs=config.lightgbm_n_jobs,
                deterministic=True,
                force_col_wise=True,
                verbosity=-1,
                subsample_freq=1,
                bagging_seed=RANDOM_SEED,
                feature_fraction_seed=RANDOM_SEED,
                data_random_seed=RANDOM_SEED,
                drop_seed=RANDOM_SEED,
                importance_type="gain",
                **dict(params),
            )
            estimator.fit(matrix, target, sample_weight=sample_weight)
        elif model_family == "lightgbm_multiclass":
            estimator = LGBMClassifier(
                objective="multiclass",
                num_class=len(GRADE_IDS),
                random_state=RANDOM_SEED,
                n_jobs=config.lightgbm_n_jobs,
                deterministic=True,
                force_col_wise=True,
                verbosity=-1,
                subsample_freq=1,
                bagging_seed=RANDOM_SEED,
                feature_fraction_seed=RANDOM_SEED,
                data_random_seed=RANDOM_SEED,
                drop_seed=RANDOM_SEED,
                importance_type="gain",
                **dict(params),
            )
            estimator.fit(matrix, target.astype("int64"), sample_weight=sample_weight)
        elif model_family == "elasticnet_regression":
            estimator = ElasticNet(
                alpha=float(params["alpha"]),
                l1_ratio=float(params["l1_ratio"]),
                fit_intercept=True,
                precompute=False,
                max_iter=config.elasticnet_max_iter,
                tol=config.elasticnet_tolerance,
                selection="cyclic",
                random_state=RANDOM_SEED,
            )
            estimator.fit(matrix, target, sample_weight=sample_weight)
        elif model_family == "catboost_regressor":
            estimator = CatBoostRegressor(
                loss_function="RMSE",
                eval_metric="RMSE",
                random_seed=RANDOM_SEED,
                thread_count=config.catboost_thread_count,
                allow_writing_files=False,
                verbose=False,
                random_strength=0.0,
                bootstrap_type="No",
                leaf_estimation_method="Newton",
                metadata=_deterministic_catboost_metadata(
                    config,
                    params,
                    training_identity=training_identity,
                ),
                **dict(params),
            )
            estimator.fit(matrix, target, sample_weight=sample_weight, verbose=False)
        else:
            raise OfflineAuxiliaryError(
                f"unsupported offline model family: {model_family}"
            )
    return estimator, list(caught)


def _predict_estimator(
    estimator: Any,
    matrix: pd.DataFrame,
    *,
    objective: str,
) -> np.ndarray:
    if objective == "regression":
        return np.clip(np.asarray(estimator.predict(matrix), dtype="float64"), 0.0, 1.0)
    raw = np.asarray(estimator.predict_proba(matrix), dtype="float64")
    return _align_multiclass_probabilities(
        raw,
        tuple(int(value) for value in estimator.classes_),
    )


def _align_multiclass_probabilities(
    probabilities: np.ndarray,
    classes: Sequence[int],
) -> np.ndarray:
    raw = np.asarray(probabilities, dtype="float64")
    if raw.ndim != 2 or raw.shape[1] != len(classes):
        raise OfflineAuxiliaryError("multiclass probability shape is invalid")
    result = np.zeros((raw.shape[0], len(GRADE_IDS)), dtype="float64")
    for index, class_id in enumerate(classes):
        if class_id not in GRADE_IDS:
            raise OfflineAuxiliaryError("multiclass estimator class is invalid")
        result[:, class_id] = raw[:, index]
    total = result.sum(axis=1)
    if np.any(total <= 0) or not np.isfinite(result).all():
        raise OfflineAuxiliaryError("multiclass estimator probabilities are invalid")
    return result / total[:, None]


def _candidate_records(
    config: OfflineAuxiliaryConfig,
    model_family: str,
) -> list[dict[str, Any]]:
    if model_family.startswith("lightgbm_"):
        params = config.lightgbm_candidates
    elif model_family == "elasticnet_regression":
        params = config.elasticnet_candidates
    elif model_family == "catboost_regressor":
        params = config.catboost_candidates
    else:
        raise OfflineAuxiliaryError("unknown offline candidate family")
    return [
        {
            "candidate_id": _candidate_id(model_family, item),
            "params": dict(item),
        }
        for item in params
    ]


def _candidate_id(
    model_family: str,
    params: Mapping[str, int | float],
) -> str:
    digest = hashlib.sha256(
        _canonical_json_bytes({"model_family": model_family, "params": dict(params)})
    ).hexdigest()
    return f"{model_family}-{digest[:16]}"


def _rank_candidates(
    rows: Sequence[Mapping[str, Any]],
    *,
    objective: str,
) -> list[dict[str, Any]]:
    copied = [dict(row) for row in rows]
    if objective == "regression":
        return sorted(
            copied,
            key=lambda row: (
                float(row["metrics"]["rmse_normalized"]),
                float(row["metrics"]["mae_normalized"]),
                str(row["candidate_id"]),
            ),
        )
    return sorted(
        copied,
        key=lambda row: (
            -float(row["metrics"]["macro_f1"]),
            float(row["metrics"]["log_loss"]),
            str(row["candidate_id"]),
        ),
    )


def _select_production_candidate(
    outer_results: Sequence[Mapping[str, Any]],
    *,
    objective: str,
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for outer in outer_results:
        for candidate in outer["candidates"]:
            grouped[str(candidate["candidate_id"])].append(candidate)
    aggregate: list[dict[str, Any]] = []
    for candidate_id, rows in grouped.items():
        if len(rows) != OUTER_FOLD_COUNT:
            raise OfflineAuxiliaryError("production candidate lacks outer coverage")
        params = dict(rows[0]["params"])
        if any(dict(row["params"]) != params for row in rows):
            raise OfflineAuxiliaryError("production candidate parameters drifted")
        if objective == "regression":
            metrics: dict[str, Any] = {
                "rmse_normalized": float(
                    np.mean([row["metrics"]["rmse_normalized"] for row in rows])
                ),
                "mae_normalized": float(
                    np.mean([row["metrics"]["mae_normalized"] for row in rows])
                ),
            }
        else:
            metrics = {
                "macro_f1": float(
                    np.mean([row["metrics"]["macro_f1"] for row in rows])
                ),
                "log_loss": float(
                    np.mean([row["metrics"]["log_loss"] for row in rows])
                ),
            }
        aggregate.append(
            {
                "candidate_id": candidate_id,
                "params": params,
                "metrics": metrics,
            }
        )
    selected = _rank_candidates(aggregate, objective=objective)[0]
    return {
        **selected,
        "selection_rule": _selection_rule(objective),
        "outer_test_metrics_not_used_for_selection": True,
        "all_candidates": aggregate,
    }


def _selection_rule(objective: str) -> str:
    if objective == "regression":
        return "lowest_pooled_inner_weighted_rmse_then_mae_then_candidate_id"
    return "highest_pooled_inner_weighted_macro_f1_then_log_loss_then_candidate_id"


def _outer_oof_frame(
    all_test: pd.DataFrame,
    available_test: pd.DataFrame,
    prediction: np.ndarray,
    *,
    dataset_id: str,
    objective: str,
    model_family: str,
    outer_fold: int,
) -> pd.DataFrame:
    output = (
        all_test[
            [
                "prediction_id",
                "canonical_row_index",
                "global_participant_id",
                "target_score",
                "target_normalized",
                "target_grade",
            ]
        ]
        .copy()
        .reset_index(drop=True)
    )
    output.insert(0, "oof_version", OOF_VERSION)
    output.insert(1, "task_key", f"{dataset_id}::{objective}")
    output.insert(2, "dataset_id", dataset_id)
    output.insert(3, "objective", objective)
    output.insert(4, "model_family", model_family)
    output.insert(8, "outer_fold", outer_fold)
    available_ids = set(available_test["prediction_id"].astype(str))
    output["available"] = output["prediction_id"].astype(str).isin(available_ids)
    output["prediction_normalized"] = np.nan
    output["prediction_score"] = np.nan
    output["prediction_grade"] = pd.array([pd.NA] * len(output), dtype="Int64")
    for grade in GRADE_IDS:
        output[f"probability_grade_{grade}"] = np.nan
    row_lookup = {
        value: index
        for index, value in enumerate(available_test["prediction_id"].astype(str))
    }
    available_indices = [
        index
        for index, value in enumerate(output["prediction_id"].astype(str))
        if value in row_lookup
    ]
    prediction_order = np.asarray(
        [
            row_lookup[output.iloc[index]["prediction_id"]]
            for index in available_indices
        ],
        dtype="int64",
    )
    if objective == "regression":
        values = np.asarray(prediction, dtype="float64")[prediction_order]
        output.loc[available_indices, "prediction_normalized"] = values
        output.loc[available_indices, "prediction_score"] = values * PHQ9_MAX_SCORE
        grades = [
            phq9_grade(min(27, math.floor(value * PHQ9_MAX_SCORE + 0.5)))
            for value in values
        ]
        output.loc[available_indices, "prediction_grade"] = grades
    else:
        probabilities = np.asarray(prediction, dtype="float64")[prediction_order]
        grades = np.argmax(probabilities, axis=1).astype("int64")
        output.loc[available_indices, "prediction_grade"] = grades
        expected_grade = probabilities @ np.asarray(GRADE_IDS, dtype="float64")
        output.loc[available_indices, "prediction_normalized"] = expected_grade / 4.0
        for grade in GRADE_IDS:
            output.loc[available_indices, f"probability_grade_{grade}"] = probabilities[
                :, grade
            ]
    return output.sort_values("canonical_row_index", kind="stable").reset_index(
        drop=True
    )


def _task_oof_metrics(oof: pd.DataFrame, *, objective: str) -> dict[str, Any]:
    available = oof[oof["available"]].copy()
    if available.empty:
        raise OfflineAuxiliaryError("offline OOF contains no evaluable rows")
    weights = _task_sample_weights(available, objective)
    if objective == "regression":
        overall = regression_metrics(
            available["target_normalized"],
            available["prediction_normalized"],
            sample_weight=weights,
        )
    else:
        probabilities = available[
            [f"probability_grade_{grade}" for grade in GRADE_IDS]
        ].to_numpy(dtype="float64")
        overall = multiclass_metrics(
            available["target_grade"].to_numpy(dtype="int64"),
            probabilities,
            sample_weight=weights,
        )
    fold_rows: list[dict[str, Any]] = []
    for outer_fold in range(OUTER_FOLD_COUNT):
        fold = available[available["outer_fold"].eq(outer_fold)].copy()
        fold_weight = _task_sample_weights(fold, objective)
        metrics = (
            regression_metrics(
                fold["target_normalized"],
                fold["prediction_normalized"],
                sample_weight=fold_weight,
            )
            if objective == "regression"
            else multiclass_metrics(
                fold["target_grade"].to_numpy(dtype="int64"),
                fold[[f"probability_grade_{grade}" for grade in GRADE_IDS]].to_numpy(
                    dtype="float64"
                ),
                sample_weight=fold_weight,
            )
        )
        fold_rows.append(
            {
                "outer_fold": outer_fold,
                "row_count": len(fold),
                "participant_count": int(fold["global_participant_id"].nunique()),
                "metrics": metrics,
            }
        )
    return {"overall": overall, "outer_folds": fold_rows}


def _publish_training_run(
    config: OfflineAuxiliaryConfig,
    inputs: OfflineAuxiliaryInputs,
    bundle: OfflineAuxiliaryBundle,
    oof: pd.DataFrame,
    task_metrics: Sequence[Mapping[str, Any]],
    search_tasks: Sequence[Mapping[str, Any]],
    preprocessing_tasks: Sequence[Mapping[str, Any]],
    warning_rows: Sequence[Mapping[str, Any]],
    protection_sha256: str,
    *,
    started: datetime,
    ended: datetime,
    overwrite: bool,
    command: Sequence[str],
) -> dict[str, Any]:
    root = config.repository_root
    stage_root = Path(tempfile.mkdtemp(prefix="model-006-stage-", dir=root))
    try:
        stage_report = stage_root / "report"
        stage_report.mkdir(parents=True)
        stage_model = stage_root / config.model_path.name
        stage_manifest = stage_root / config.manifest_path.name
        shutil.copyfile(config.config_path, stage_report / "training_config.yaml")
        joblib.dump(bundle, stage_model, compress=0, protocol=5)
        model_sha = _sha256_file(stage_model)
        submodel_manifest = {
            task_key: _submodel_manifest(submodel)
            for task_key, submodel in bundle.submodels.items()
        }
        manifest_payload = {
            "manifest_version": MANIFEST_VERSION,
            "training_version": TRAINING_VERSION,
            "task_id": TASK_ID,
            "run_id": RUN_ID,
            "status": "offline_auxiliary_not_active",
            "frozen_document_version": "V3.3.3",
            "split_id": SPLIT_ID,
            "split_sha256": SPLIT_SHA256,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
            "upstream_evaluation_run_id": EVALUATION_RUN_ID,
            "upstream_evaluation_core_sha256": EVALUATION_CORE_SHA256,
            "baseline_protection_sha256": protection_sha256,
            "training_config_sha256": config.config_sha256,
            "model_path": config.model_path.relative_to(root).as_posix(),
            "model_sha256": model_sha,
            "model_bytes": stage_model.stat().st_size,
            "submodel_count": len(submodel_manifest),
            "submodels": submodel_manifest,
            "production_boundary": {
                "active": False,
                "enters_fusion": False,
                "enters_api": False,
                "contains_scale_input_features": False,
                "contains_publisher_label_input_features": False,
                "contains_feature_masks_as_inputs": False,
            },
        }
        _write_json(stage_manifest, manifest_payload)
        manifest_sha = _sha256_file(stage_manifest)
        predictions_path = stage_report / "predictions.parquet"
        _write_deterministic_parquet(oof, predictions_path)
        metrics_payload = {
            "metrics_version": METRICS_VERSION,
            "task_id": TASK_ID,
            "run_id": RUN_ID,
            "target_policy": {
                "continuous": "phq9_total / 27 clipped predictions to [0,1]",
                "five_level_bands": [list(band) for band in GRADE_BANDS],
            },
            "task_count": len(task_metrics),
            "tasks": list(task_metrics),
            "summary": {
                "oof_row_count": len(oof),
                "available_row_count": int(oof["available"].sum()),
                "unavailable_row_count": int((~oof["available"]).sum()),
                "regression_task_count": len(REGRESSION_DATASETS),
                "multiclass_task_count": len(MULTICLASS_DATASETS),
            },
        }
        _write_json(stage_report / "metrics.json", metrics_payload)
        _write_json(
            stage_report / "search_results.json",
            {
                "search_version": SEARCH_VERSION,
                "task_id": TASK_ID,
                "run_id": RUN_ID,
                "outer_test_metrics_not_used_for_selection": True,
                "task_count": len(search_tasks),
                "tasks": list(search_tasks),
            },
        )
        _write_json(
            stage_report / "preprocessing.json",
            {
                "preprocessing_version": "mood-social-offline-fold-preprocessing-v1",
                "all_fit_inside_training_fold": True,
                "feature_mask_is_input": False,
                "feature_coverage_is_input": False,
                "task_count": len(preprocessing_tasks),
                "tasks": list(preprocessing_tasks),
            },
        )
        warning_counter = Counter(str(row["category"]) for row in warning_rows)
        warnings_payload = {
            "warning_version": "mood-social-offline-auxiliary-warnings-v1",
            "warning_count": len(warning_rows),
            "counts_by_category": dict(sorted(warning_counter.items())),
            "warnings": list(warning_rows),
            "limitations": [
                "All models are offline-only and are excluded from fusion and HTTP output.",
                "Public wearable and cross-sectional profile fields are deployment-compatible proxies, not diagnostic evidence.",
                "RESILIENT is a single small source; its regression uncertainty is not evidence of transportability.",
                "Rare PHQ-9 grade 4 samples make multiclass fold metrics unstable, especially in NHANES sources.",
                "No production threshold or probability calibrator is selected in MODEL-006.",
            ],
        }
        _write_json(stage_report / "warnings.json", warnings_payload)
        _write_json(
            stage_report / "model_card.json",
            _model_card_payload(bundle, task_metrics, submodel_manifest),
        )
        run_payload = {
            "run_version": "mood-social-offline-auxiliary-run-v1",
            "task_id": TASK_ID,
            "run_id": RUN_ID,
            "status": "completed",
            "started_at": started.isoformat(),
            "ended_at": ended.isoformat(),
            "duration_seconds": (ended - started).total_seconds(),
            "command": list(command),
            "random_seed": RANDOM_SEED,
            "code_version": _git_state(root),
            "python": sys.version,
            "platform": platform.platform(),
            "dependencies": _dependency_versions(),
            "training_config_sha256": config.config_sha256,
            "model_sha256": model_sha,
            "manifest_sha256": manifest_sha,
            "baseline_protection_sha256": protection_sha256,
            "upstream_evaluation_core_sha256": EVALUATION_CORE_SHA256,
        }
        _write_json(stage_report / "run.json", run_payload)
        artifact_rows = _tree_hash_rows(stage_report)
        _write_json(
            stage_report / "artifacts.json",
            {
                "artifact_manifest_version": "mood-social-offline-artifacts-v1",
                "task_id": TASK_ID,
                "run_id": RUN_ID,
                "artifact_count": len(artifact_rows),
                "artifacts": artifact_rows,
                "report_core_sha256": _rows_sha256(
                    [row for row in artifact_rows if row["path"] not in {"run.json"}]
                ),
                "external_model": {
                    "path": config.model_path.relative_to(root).as_posix(),
                    "sha256": model_sha,
                    "bytes": stage_model.stat().st_size,
                },
                "external_manifest": {
                    "path": config.manifest_path.relative_to(root).as_posix(),
                    "sha256": manifest_sha,
                    "bytes": stage_manifest.stat().st_size,
                },
            },
        )
        report_core_sha = _deterministic_report_core_sha256(stage_report)
        _atomic_publish_file(stage_model, config.model_path, overwrite=overwrite)
        _atomic_publish_file(stage_manifest, config.manifest_path, overwrite=overwrite)
        _atomic_publish_directory(
            stage_report,
            config.report_directory,
            overwrite=overwrite,
        )
    finally:
        if stage_root.exists():
            shutil.rmtree(stage_root)
    return {
        "task_id": TASK_ID,
        "run_id": RUN_ID,
        "status": "completed",
        "model_path": config.model_path,
        "manifest_path": config.manifest_path,
        "report_directory": config.report_directory,
        "model_sha256": model_sha,
        "manifest_sha256": manifest_sha,
        "report_core_sha256": report_core_sha,
        "oof_row_count": len(oof),
        "available_row_count": int(oof["available"].sum()),
        "warning_count": len(warning_rows),
        "metrics": list(task_metrics),
        "baseline_protection_sha256": protection_sha256,
    }


def _submodel_manifest(submodel: OfflineAuxiliarySubmodel) -> dict[str, Any]:
    estimator = submodel.estimator
    native_hash: str
    if submodel.model_family.startswith("lightgbm_"):
        native_hash = hashlib.sha256(
            estimator.booster_.model_to_string().encode("utf-8")
        ).hexdigest()
    elif submodel.model_family == "catboost_regressor":
        with tempfile.TemporaryDirectory(prefix="model-006-catboost-") as directory:
            path = Path(directory) / "model.cbm"
            estimator.save_model(path, format="cbm")
            native_hash = _sha256_file(path)
    elif submodel.model_family == "elasticnet_regression":
        native_hash = _canonical_payload_sha256(
            {
                "intercept": float(estimator.intercept_),
                "coef": [float(value) for value in estimator.coef_],
            }
        )
    else:
        raise OfflineAuxiliaryError("unknown offline submodel family")
    return {
        "dataset_id": submodel.dataset_id,
        "objective": submodel.objective,
        "model_family": submodel.model_family,
        "hyperparameters": dict(submodel.hyperparameters),
        "input_features": list(submodel.input_features),
        "categorical_features": list(submodel.categorical_features),
        "selected_features": list(submodel.preprocessor.selected_features),
        "output_columns": list(submodel.preprocessor.output_columns),
        "training_participant_count": submodel.training_participant_count,
        "training_participant_sha256": submodel.training_participant_sha256,
        "training_row_count": submodel.training_row_count,
        "activity_ecdf": _ecdf_manifest(submodel.activity_ecdf),
        "native_model_sha256": native_hash,
        "offline_only": True,
    }


def _model_card_payload(
    bundle: OfflineAuxiliaryBundle,
    task_metrics: Sequence[Mapping[str, Any]],
    submodel_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "model_card_version": "mood-social-offline-auxiliary-model-card-v1",
        "task_id": TASK_ID,
        "run_id": RUN_ID,
        "name": "PHQ-9 offline continuous and five-level auxiliary models",
        "intended_use": "offline model analysis and reporting only",
        "prohibited_uses": [
            "MoodFusionModel input",
            "PersonalTrend input",
            "HTTP response",
            "medical diagnosis",
            "replacement of the five active standalone experts",
        ],
        "target_definition": {
            "continuous": "PHQ-9 total divided by 27",
            "five_level": [list(band) for band in GRADE_BANDS],
        },
        "feature_policy": (
            "Only frozen deployment-compatible unified Schema fields with real "
            "feature masks are materialized. Masks, feature coverage, scale items, "
            "publisher labels, SSQ-only research fields and source IDs are not model inputs."
        ),
        "split_policy": (
            "DATA-007 participant-level outer 5-fold evaluation with inner 5-fold "
            "selection; preprocessing and activity ECDF are fit inside each training fold."
        ),
        "submodels": dict(submodel_manifest),
        "metrics": list(task_metrics),
        "bundle_status": bundle.status,
    }


def _task_sample_weights(frame: pd.DataFrame, objective: str) -> np.ndarray:
    return (
        regression_sample_weights(frame)
        if objective == "regression"
        else multiclass_sample_weights(frame)
    )


def _normalize_weights(weight: np.ndarray) -> np.ndarray:
    result = np.asarray(weight, dtype="float64")
    if (
        result.ndim != 1
        or result.size == 0
        or not np.isfinite(result).all()
        or np.any(result <= 0)
        or result.mean() <= 0
    ):
        raise OfflineAuxiliaryError("offline sample weights are invalid")
    return result / result.mean()


def _inner_fold_value(value: Any, outer_fold: int) -> int:
    if not isinstance(value, Mapping):
        raise OfflineAuxiliaryError("inner fold assignment is malformed")
    selected = value.get(str(outer_fold), value.get(outer_fold))
    if selected is None:
        raise OfflineAuxiliaryError("outer-training participant lacks inner fold")
    result = int(selected)
    if result not in range(INNER_FOLD_COUNT):
        raise OfflineAuxiliaryError("inner fold assignment is outside 0..4")
    return result


def _ecdf_manifest(ecdf: Any | None) -> dict[str, Any] | None:
    if ecdf is None:
        return None
    payload = dict(ecdf.to_dict())
    payload.pop("sorted_training_values", None)
    payload["sha256"] = str(ecdf.sha256)
    return payload


def _warning_records(
    caught: Sequence[warnings.WarningMessage],
    *,
    task_key: str,
    stage: str,
    candidate_id: str,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in caught:
        category = item.category.__name__
        message = str(item.message)
        result.append(
            {
                "task_key": task_key,
                "stage": stage,
                "candidate_id": candidate_id,
                "category": category,
                "message": message,
                "convergence_warning": issubclass(item.category, ConvergenceWarning),
            }
        )
    return result


def _deterministic_catboost_metadata(
    config: OfflineAuxiliaryConfig,
    hyperparameters: Mapping[str, int | float],
    *,
    training_identity: str,
) -> dict[str, str]:
    if not training_identity or "\0" in training_identity:
        raise OfflineAuxiliaryError("CatBoost training identity is invalid")
    payload = {
        "task_id": TASK_ID,
        "run_id": RUN_ID,
        "training_config_sha256": config.config_sha256,
        "training_identity": training_identity,
        "hyperparameters": dict(hyperparameters),
    }
    digest = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    model_guid = "-".join(digest[index : index + 8] for index in range(0, 32, 8))
    return {
        "model_guid": model_guid,
        "train_finish_time": "1970-01-01T00:00:00Z",
        "eldercare_training_identity": training_identity,
        "eldercare_training_sha256": digest,
    }


def _report_core_sha256(report_path: Path) -> str:
    artifact_path = report_path / "artifacts.json"
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OfflineAuxiliaryError("upstream report artifacts are unreadable") from exc
    rows = payload.get("artifacts")
    if not isinstance(rows, list) or int(payload.get("artifact_count", -1)) != len(
        rows
    ):
        raise OfflineAuxiliaryError("upstream report artifact manifest is invalid")
    for row in rows:
        path = report_path / str(row["path"])
        if (
            not path.is_file()
            or path.stat().st_size != int(row["bytes"])
            or _sha256_file(path) != str(row["sha256"])
        ):
            raise OfflineAuxiliaryError("upstream report artifact drifted")
    core = str(payload.get("core_sha256", ""))
    if not _is_sha256(core):
        raise OfflineAuxiliaryError("upstream report core hash is invalid")
    return core


def _validate_output_paths(config: OfflineAuxiliaryConfig) -> None:
    root = config.repository_root
    for path in (config.report_directory, config.model_path, config.manifest_path):
        try:
            path.resolve().relative_to(root)
        except ValueError as exc:
            raise OfflineAuxiliaryError(
                "MODEL-006 output escapes repository root"
            ) from exc
    active_names = {
        "activity_expert.joblib",
        "sleep_expert.joblib",
        "activity_sleep_joint_expert.joblib",
        "physiology_expert.joblib",
        "social_context_expert.joblib",
    }
    if config.model_path.name in active_names or config.report_directory.name != RUN_ID:
        raise OfflineAuxiliaryError("MODEL-006 output collides with an active artifact")


def _assert_outputs_available(
    config: OfflineAuxiliaryConfig,
    *,
    overwrite: bool,
) -> None:
    existing = [
        path
        for path in (config.report_directory, config.model_path, config.manifest_path)
        if path.exists()
    ]
    if existing and not overwrite:
        raise OfflineAuxiliaryError(
            "MODEL-006 output already exists; pass overwrite=True for deterministic rebuild"
        )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_json_bytes(payload))


def _write_deterministic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pyarrow.Table.from_pandas(frame, preserve_index=False)
    pq.write_table(
        table,
        path,
        compression="zstd",
        compression_level=9,
        use_dictionary=False,
        write_statistics=True,
        row_group_size=65536,
        data_page_version="1.0",
    )


def _atomic_publish_file(stage: Path, destination: Path, *, overwrite: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise OfflineAuxiliaryError(f"output already exists: {destination}")
    backup: Path | None = None
    if destination.exists():
        backup_root = Path(
            tempfile.mkdtemp(prefix="model-006-backup-", dir=destination.parent)
        )
        backup = backup_root / destination.name
        os.replace(destination, backup)
    try:
        os.replace(stage, destination)
    except Exception:
        if backup is not None and backup.exists():
            os.replace(backup, destination)
        raise
    finally:
        if backup is not None:
            shutil.rmtree(backup.parent, ignore_errors=True)


def _atomic_publish_directory(
    stage: Path,
    destination: Path,
    *,
    overwrite: bool,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise OfflineAuxiliaryError(f"output already exists: {destination}")
    backup: Path | None = None
    if destination.exists():
        backup_root = Path(
            tempfile.mkdtemp(prefix="model-006-backup-", dir=destination.parent)
        )
        backup = backup_root / destination.name
        os.replace(destination, backup)
    try:
        os.replace(stage, destination)
    except Exception:
        if backup is not None and backup.exists():
            os.replace(backup, destination)
        raise
    finally:
        if backup is not None:
            shutil.rmtree(backup.parent, ignore_errors=True)


def _tree_hash_rows(directory: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(directory).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(
            (item for item in directory.rglob("*") if item.is_file()),
            key=lambda item: item.relative_to(directory).as_posix().encode("utf-8"),
        )
    ]


def _deterministic_report_core_sha256(report: Path) -> str:
    rows = [
        row
        for row in _tree_hash_rows(report)
        if row["path"] not in {"run.json", "artifacts.json"}
    ]
    return _rows_sha256(rows)


def _rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = "".join(
        f"{row['path']}\0{row['bytes']}\0{row['sha256']}\n" for row in rows
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            default=_json_default,
        )
        + "\n"
    ).encode("utf-8")


def _canonical_payload_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload).rstrip(b"\n")).hexdigest()


def _string_set_sha256(values: Sequence[str] | set[str]) -> str:
    ordered = sorted(
        set(str(value) for value in values), key=lambda value: value.encode()
    )
    return hashlib.sha256("\n".join(ordered).encode("utf-8")).hexdigest()


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _is_missing_scalar(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def _find_repository_root(path: Path) -> Path:
    for candidate in (path.parent, *path.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src").is_dir():
            return candidate
    raise OfflineAuxiliaryError("could not locate algorithm repository root")


def _git_state(root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


def _dependency_versions() -> dict[str, str]:
    return {
        "catboost": _package_version("catboost"),
        "joblib": joblib.__version__,
        "lightgbm": lightgbm.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyarrow": pyarrow.__version__,
        "scikit_learn": sklearn.__version__,
        "scipy": scipy.__version__,
    }


def _package_version(name: str) -> str:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return "unknown"


__all__ = [
    "BASELINE_PROTECTION_SHA256",
    "BUNDLE_VERSION",
    "DATASET_ORDER",
    "EVALUATION_CORE_SHA256",
    "GRADE_BANDS",
    "GRADE_IDS",
    "MANIFEST_VERSION",
    "MODEL_FAMILY_BY_TASK",
    "MULTICLASS_DATASETS",
    "OfflineAuxiliaryBundle",
    "OfflineAuxiliaryConfig",
    "OfflineAuxiliaryError",
    "OfflineAuxiliarySubmodel",
    "OfflineDatasetSpec",
    "OfflineFoldPreprocessor",
    "REGRESSION_DATASETS",
    "RUN_ID",
    "TASK_ID",
    "TRAINING_VERSION",
    "build_task_frame",
    "dataset_spec",
    "deterministic_candidates",
    "load_offline_auxiliary_config",
    "load_offline_auxiliary_inputs",
    "materialize_features",
    "multiclass_metrics",
    "multiclass_sample_weights",
    "phq9_grade",
    "regression_metrics",
    "regression_sample_weights",
    "train_offline_auxiliary_models",
]
