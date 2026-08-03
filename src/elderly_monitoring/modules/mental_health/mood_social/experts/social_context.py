"""Leakage-safe V3.3.3 SocialContextExpert training and inference artifacts.

The expert is deliberately independent from the activity, sleep, physiology and
social-contact branches.  It consumes only the ten frozen ``social_context``
fields and their masks from the canonical tables.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

import joblib
from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
import pyarrow
import sklearn
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
import yaml

from elderly_monitoring.modules.mental_health.mood_social.feature_schema import (
    FEATURE_SCHEMA_VERSION,
    SOCIAL_CONTEXT_FEATURE_ORDER,
    SOCIAL_CONTEXT_FEATURE_SPECS,
)
from elderly_monitoring.modules.mental_health.mood_social.splits import (
    DEFAULT_SPLIT_RELATIVE,
    SPLIT_ID,
    build_split_manifest,
)


MODEL_ID = "M-SOC-001"
MODEL_VERSION = "social-context-expert-v3.3.3"
TRAINING_CONFIG_VERSION = "mood-social-social-context-training-config-v1"
MODEL_BUNDLE_VERSION = "mood-social-social-context-expert-bundle-v1"
PREPROCESSOR_VERSION = "mood-social-social-context-fold-preprocessor-v1"
CALIBRATOR_VERSION = "mood-social-probability-calibrator-v1"
METRICS_VERSION = "mood-social-binary-metrics-v1"
PREDICTIONS_VERSION = "mood-social-social-context-oof-predictions-v1"
RUN_MANIFEST_VERSION = "mood-social-social-context-training-run-v1"
MODEL_MANIFEST_VERSION = "mood-social-social-context-model-manifest-v1"
EXPECTED_SPLIT_SHA256 = (
    "e9915dbc590a6ea454c34d26558ac5866c57ee8a8e78f44cdcc79c859df77ee3"
)
EXPECTED_FEATURE_SCHEMA_SHA256 = (
    "d2ed18db72601ccb3b0cbb89907bc60ab336bd00f84f0b2c49dcea568bf339e9"
)

# Preserve the canonical input order from DATA-007, omitting PSYCHE-D, which
# has no deployment-compatible profile fields.
SOCIAL_CONTEXT_DATASET_IDS = (
    "resilient",
    "nhanes",
    "shenzhen_elderly",
    "nhanes_ssq_2005_2008",
)
SOCIAL_DATASET_IDS = SOCIAL_CONTEXT_DATASET_IDS
EXCLUDED_DATASET_IDS = ("psyche_d",)
SOCIAL_CONTEXT_FEATURE_COLUMNS = tuple(
    f"social_context.{name}" for name in SOCIAL_CONTEXT_FEATURE_ORDER
)
SOCIAL_FEATURE_COLUMNS = SOCIAL_CONTEXT_FEATURE_COLUMNS
SOCIAL_CONTEXT_MASK_COLUMNS = tuple(
    f"feature_mask.{name}" for name in SOCIAL_CONTEXT_FEATURE_COLUMNS
)
SOCIAL_MASK_COLUMNS = SOCIAL_CONTEXT_MASK_COLUMNS
SOCIAL_CONTEXT_CATEGORICAL_COLUMNS = tuple(
    feature
    for feature, spec in zip(
        SOCIAL_CONTEXT_FEATURE_COLUMNS, SOCIAL_CONTEXT_FEATURE_SPECS, strict=True
    )
    if spec.value_type.value == "category"
)
SOCIAL_CONTEXT_NUMERIC_COLUMNS = tuple(
    feature
    for feature in SOCIAL_CONTEXT_FEATURE_COLUMNS
    if feature not in SOCIAL_CONTEXT_CATEGORICAL_COLUMNS
)
OUTER_FOLD_COUNT = 5
INNER_FOLD_COUNT = 5
MISSING_CATEGORY = "__MISSING__"

UPSTREAM_EXPERT_GUARDS: Mapping[str, Mapping[str, str]] = {
    "activity": {
        "model_path": "models/mental_health/mood_social/v3.3.3/activity_expert.joblib",
        "model_sha256": "906a3cc2b74cdd3beeba446ca32885c7d2f0388be230ab793a2728bfd711bc3a",
        "manifest_path": "models/mental_health/mood_social/v3.3.3/activity_expert_manifest.json",
        "manifest_sha256": "eb2521a217083c784ecef1b93deef33aebec5dd4bcb614db5890ed47d5700b1b",
    },
    "sleep": {
        "model_path": "models/mental_health/mood_social/v3.3.3/sleep_expert.joblib",
        "model_sha256": "966ba9ecd32ac18ac63ed9a6733a2e84f859a657cdb664bc2e034f0857056ade",
        "manifest_path": "models/mental_health/mood_social/v3.3.3/sleep_expert_manifest.json",
        "manifest_sha256": "11096d9916e822be23d8c9acdc86bb9c3c2700bf7c0ea2bcad8abf315ce30459",
    },
    "joint": {
        "model_path": "models/mental_health/mood_social/v3.3.3/activity_sleep_joint_expert.joblib",
        "model_sha256": "2152e7a62ba6382d64ddb1ca865410fa99e0acf9b73db54ed4c25622c46e70c8",
        "manifest_path": "models/mental_health/mood_social/v3.3.3/activity_sleep_joint_expert_manifest.json",
        "manifest_sha256": "f54149fc0e504d052ec763e56441fdc638e7b00a8c7801ded497e24943755e17",
    },
    "physiology": {
        "model_path": "models/mental_health/mood_social/v3.3.3/physiology_expert.joblib",
        "model_sha256": "1596e4ecae67319818c672f0cfc1ca7177fb4481f21ca66363154e8290c69a9e",
        "manifest_path": "models/mental_health/mood_social/v3.3.3/physiology_expert_manifest.json",
        "manifest_sha256": "2afdb2320f748932ff25d017ff60dd61e54e6e55e5104c72a3e7fec1d483192f",
    },
}
_UPSTREAM_IDENTITIES = {
    "activity": ("M-ACT-001", "activity-expert-v3.3.3"),
    "sleep": ("M-SLP-001", "sleep-expert-v3.3.3"),
    "joint": ("M-JNT-001", "activity-sleep-joint-expert-v3.3.3"),
    "physiology": ("M-PHY-001", "physiology-expert-v3.3.3"),
}


class SocialContextExpertError(RuntimeError):
    """Raised when MODEL-005 input, training, or artifact checks fail."""


@dataclass(frozen=True)
class SocialContextTrainingConfig:
    run_id: str
    random_seed: int
    split_id: str
    split_sha256: str
    feature_schema_version: str
    feature_schema_sha256: str
    dataset_ids: tuple[str, ...]
    search_space: Mapping[str, tuple[int | float, ...]]
    search_candidate_count: int
    search_traversal: str
    auprc_tie_tolerance: float
    calibration_positive_participant_threshold: int
    decision_threshold: float
    ece_bin_count: int
    catboost_thread_count: int
    config_path: Path
    config_sha256: str


@dataclass(frozen=True)
class SocialContextTrainingInputs:
    repository_root: Path
    split_manifest: Mapping[str, Any]
    split_manifest_sha256: str
    assignments: pd.DataFrame
    frames: Mapping[str, pd.DataFrame]
    input_bindings: Mapping[str, Mapping[str, Any]]
    upstream_expert_bindings: Mapping[str, Mapping[str, Any]]


def _spec_categories(feature: str) -> tuple[str, ...]:
    name = feature.split(".", 1)[1]
    for spec in SOCIAL_CONTEXT_FEATURE_SPECS:
        if spec.name == name:
            return tuple(spec.categories or ())
    raise SocialContextExpertError("unknown SocialContext feature")


@dataclass(frozen=True)
class SocialContextFoldPreprocessor:
    """Fold-local category vocabulary, numeric medians and feature selection."""

    input_features: tuple[str, ...]
    selected_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    numeric_features: tuple[str, ...]
    medians: tuple[float | None, ...]
    category_levels: Mapping[str, tuple[str, ...]]
    observed_counts: tuple[int, ...]
    unique_counts: tuple[int, ...]
    version: str = PREPROCESSOR_VERSION
    categorical_encoding: str = "training_fold_category_levels_with_missing_sentinel"
    numeric_imputation: str = "training_fold_median"
    feature_selection: str = "training_fold_observed_nonconstant"
    add_missing_indicators: bool = False

    def __post_init__(self) -> None:
        if self.version != PREPROCESSOR_VERSION:
            raise SocialContextExpertError("unsupported SocialContext preprocessor")
        if self.input_features != SOCIAL_CONTEXT_FEATURE_COLUMNS:
            raise SocialContextExpertError("SocialContext input feature order changed")
        if not self.selected_features:
            raise SocialContextExpertError("SocialContext selected no usable features")
        if not set(self.selected_features).issubset(self.input_features):
            raise SocialContextExpertError("SocialContext selected an unknown feature")
        if set(self.categorical_features) | set(self.numeric_features) != set(
            self.selected_features
        ):
            raise SocialContextExpertError(
                "SocialContext feature type audit is invalid"
            )
        if set(self.categorical_features) & set(self.numeric_features):
            raise SocialContextExpertError("SocialContext feature type overlap")
        if len(self.medians) != len(self.selected_features):
            raise SocialContextExpertError("SocialContext median count does not match")
        if len(self.observed_counts) != len(self.input_features) or len(
            self.unique_counts
        ) != len(self.input_features):
            raise SocialContextExpertError(
                "SocialContext selection audit does not match"
            )
        if self.add_missing_indicators:
            raise SocialContextExpertError("feature masks cannot become risk inputs")
        for feature in self.categorical_features:
            levels = self.category_levels.get(feature)
            if not levels or MISSING_CATEGORY in levels:
                raise SocialContextExpertError("categorical vocabulary is invalid")
        for feature, median in zip(self.selected_features, self.medians, strict=True):
            if feature in self.numeric_features and (
                median is None or not math.isfinite(float(median))
            ):
                raise SocialContextExpertError("numeric median is invalid")

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> SocialContextFoldPreprocessor:
        values, masks = _masked_social_values(frame)
        selected: list[str] = []
        categorical: list[str] = []
        numeric: list[str] = []
        medians: list[float | None] = []
        levels: dict[str, tuple[str, ...]] = {}
        observed_counts: list[int] = []
        unique_counts: list[int] = []
        for index, feature in enumerate(SOCIAL_CONTEXT_FEATURE_COLUMNS):
            observed = masks[:, index].astype(bool)
            if feature in SOCIAL_CONTEXT_CATEGORICAL_COLUMNS:
                raw = values[feature][observed]
                finite_values = [str(item) for item in raw if item is not None]
                unique = sorted(
                    set(finite_values), key=lambda item: item.encode("utf-8")
                )
                observed_count = len(finite_values)
                unique_count = len(unique)
                observed_counts.append(observed_count)
                unique_counts.append(unique_count)
                if observed_count >= 2 and unique_count >= 2:
                    selected.append(feature)
                    categorical.append(feature)
                    levels[feature] = tuple(unique)
                    medians.append(None)
            else:
                raw = pd.to_numeric(
                    pd.Series(values[feature][observed]), errors="coerce"
                ).to_numpy(dtype="float64")
                raw = raw[np.isfinite(raw)]
                observed_count = len(raw)
                unique_count = len(np.unique(raw)) if observed_count else 0
                observed_counts.append(observed_count)
                unique_counts.append(unique_count)
                if observed_count >= 2 and unique_count >= 2:
                    selected.append(feature)
                    numeric.append(feature)
                    medians.append(float(np.median(raw)))
        return cls(
            input_features=SOCIAL_CONTEXT_FEATURE_COLUMNS,
            selected_features=tuple(selected),
            categorical_features=tuple(categorical),
            numeric_features=tuple(numeric),
            medians=tuple(medians),
            category_levels=levels,
            observed_counts=tuple(observed_counts),
            unique_counts=tuple(unique_counts),
        )

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        values, _ = _masked_social_values(frame)
        output: dict[str, Any] = {}
        for feature, median in zip(self.selected_features, self.medians, strict=True):
            raw = values[feature]
            if feature in self.categorical_features:
                known = set(self.category_levels[feature])
                output[feature] = [
                    str(item)
                    if item is not None and str(item) in known
                    else MISSING_CATEGORY
                    for item in raw
                ]
            else:
                numeric = (
                    pd.to_numeric(pd.Series(raw), errors="coerce")
                    .to_numpy(dtype="float64")
                    .copy()
                )
                numeric[~np.isfinite(numeric)] = float(median)  # type: ignore[arg-type]
                output[feature] = numeric
        result = pd.DataFrame(output, index=frame.index)
        if not result.empty:
            for feature in self.categorical_features:
                result[feature] = result[feature].astype(str)
            for feature in self.numeric_features:
                result[feature] = pd.to_numeric(result[feature], errors="raise").astype(
                    "float64"
                )
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "input_features": list(self.input_features),
            "selected_features": list(self.selected_features),
            "categorical_features": list(self.categorical_features),
            "numeric_features": list(self.numeric_features),
            "medians": list(self.medians),
            "category_levels": {
                name: list(levels)
                for name, levels in sorted(self.category_levels.items())
            },
            "observed_counts": dict(
                zip(self.input_features, self.observed_counts, strict=True)
            ),
            "unique_counts": dict(
                zip(self.input_features, self.unique_counts, strict=True)
            ),
            "categorical_encoding": self.categorical_encoding,
            "numeric_imputation": self.numeric_imputation,
            "feature_selection": self.feature_selection,
            "missing_category": MISSING_CATEGORY,
            "add_missing_indicators": self.add_missing_indicators,
        }


# Descriptive aliases keep the public API easy to discover for downstream tasks.
SocialContextCategoricalFoldPreprocessor = SocialContextFoldPreprocessor
CategoricalFoldPreprocessor = SocialContextFoldPreprocessor


@dataclass(frozen=True)
class SocialContextProbabilityCalibrator:
    method: str
    estimator: IsotonicRegression | LogisticRegression
    positive_participant_count: int
    fit_row_count: int
    version: str = CALIBRATOR_VERSION
    probability_clip_epsilon: float = 1.0e-6

    def __post_init__(self) -> None:
        if self.version != CALIBRATOR_VERSION:
            raise SocialContextExpertError("unsupported probability calibrator")
        if self.method not in {"isotonic", "platt"}:
            raise SocialContextExpertError("unsupported calibration method")
        if self.positive_participant_count <= 0 or self.fit_row_count <= 0:
            raise SocialContextExpertError("calibrator training scope is empty")

    @classmethod
    def fit(
        cls,
        raw_probability: Sequence[float],
        target: Sequence[int],
        sample_weight: Sequence[float],
        *,
        positive_participant_count: int,
        isotonic_min_positive_participants: int,
    ) -> SocialContextProbabilityCalibrator:
        raw = np.asarray(raw_probability, dtype="float64")
        y = np.asarray(target, dtype="int64")
        weight = np.asarray(sample_weight, dtype="float64")
        if (
            raw.ndim != 1
            or len(raw) != len(y)
            or len(raw) != len(weight)
            or len(raw) == 0
            or not np.isfinite(raw).all()
            or not np.isfinite(weight).all()
            or np.any(weight <= 0)
            or set(np.unique(y)) != {0, 1}
        ):
            raise SocialContextExpertError("calibrator inputs are invalid")
        method = (
            "isotonic"
            if positive_participant_count >= isotonic_min_positive_participants
            else "platt"
        )
        if method == "isotonic":
            estimator: IsotonicRegression | LogisticRegression = IsotonicRegression(
                y_min=0.0,
                y_max=1.0,
                increasing=True,
                out_of_bounds="clip",
            )
            estimator.fit(raw, y, sample_weight=weight)
        else:
            estimator = LogisticRegression(
                C=np.inf,
                solver="lbfgs",
                max_iter=5000,
                random_state=20260728,
            )
            estimator.fit(_probability_logit(raw)[:, None], y, sample_weight=weight)
        return cls(
            method=method,
            estimator=estimator,
            positive_participant_count=positive_participant_count,
            fit_row_count=len(raw),
        )

    def predict(self, raw_probability: Sequence[float]) -> np.ndarray:
        raw = np.asarray(raw_probability, dtype="float64")
        if raw.ndim != 1 or not np.isfinite(raw).all():
            raise SocialContextExpertError("calibrator prediction input is invalid")
        if self.method == "isotonic":
            calibrated = self.estimator.predict(raw)
        else:
            calibrated = self.estimator.predict_proba(
                _probability_logit(raw, self.probability_clip_epsilon)[:, None]
            )[:, 1]
        return np.clip(np.asarray(calibrated, dtype="float64"), 0.0, 1.0)

    def to_dict(self) -> dict[str, Any]:
        common = {
            "version": self.version,
            "method": self.method,
            "positive_participant_count": self.positive_participant_count,
            "fit_row_count": self.fit_row_count,
            "probability_clip_epsilon": self.probability_clip_epsilon,
        }
        if self.method == "isotonic":
            estimator = self.estimator
            return {
                **common,
                "x_thresholds": [float(value) for value in estimator.X_thresholds_],
                "y_thresholds": [float(value) for value in estimator.y_thresholds_],
            }
        estimator = self.estimator
        return {
            **common,
            "classes": [int(value) for value in estimator.classes_],
            "coefficient": [float(value) for value in estimator.coef_[0]],
            "intercept": [float(value) for value in estimator.intercept_],
        }


SocialContextCalibrator = SocialContextProbabilityCalibrator


@dataclass(frozen=True)
class SocialContextExpertBundle:
    """Serialized aggregate CatBoost model used by API/fusion follow-up tasks."""

    model_id: str
    model_version: str
    bundle_version: str
    feature_schema_version: str
    feature_schema_sha256: str
    split_id: str
    split_manifest_sha256: str
    dataset_ids: tuple[str, ...]
    input_feature_names: tuple[str, ...]
    preprocessor: SocialContextFoldPreprocessor
    classifier: CatBoostClassifier
    calibrator: SocialContextProbabilityCalibrator
    hyperparameters: Mapping[str, int | float]
    training_scope: str
    training_participant_sha256: str
    training_participant_count: int
    training_row_count: int
    run_id: str

    def __post_init__(self) -> None:
        if (
            self.model_id != MODEL_ID
            or self.model_version != MODEL_VERSION
            or self.bundle_version != MODEL_BUNDLE_VERSION
        ):
            raise SocialContextExpertError("SocialContext bundle identity changed")
        if self.feature_schema_version != FEATURE_SCHEMA_VERSION:
            raise SocialContextExpertError("SocialContext feature schema changed")
        if self.feature_schema_sha256 != EXPECTED_FEATURE_SCHEMA_SHA256:
            raise SocialContextExpertError("SocialContext feature schema hash changed")
        if self.split_id != SPLIT_ID:
            raise SocialContextExpertError("SocialContext split ID changed")
        if self.split_manifest_sha256 != EXPECTED_SPLIT_SHA256:
            raise SocialContextExpertError("SocialContext split hash changed")
        if self.dataset_ids != SOCIAL_CONTEXT_DATASET_IDS:
            raise SocialContextExpertError("SocialContext dataset membership changed")
        if self.input_feature_names != SOCIAL_CONTEXT_FEATURE_COLUMNS:
            raise SocialContextExpertError("SocialContext feature order changed")
        if self.classifier.get_params().get("random_seed") != 20260728:
            raise SocialContextExpertError("SocialContext CatBoost seed changed")
        if self.training_participant_count <= 0 or self.training_row_count <= 0:
            raise SocialContextExpertError("SocialContext training scope is empty")
        if not _is_sha256(self.training_participant_sha256):
            raise SocialContextExpertError("SocialContext participant hash is invalid")

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        _, masks = _masked_social_values(frame)
        matrix = self.preprocessor.transform(frame)
        raw = np.asarray(
            self.classifier.predict_proba(matrix)[:, 1],
            dtype="float64",
        )
        calibrated = self.calibrator.predict(raw)
        selected_indices = [
            SOCIAL_CONTEXT_FEATURE_COLUMNS.index(feature)
            for feature in self.preprocessor.selected_features
        ]
        observed = masks[:, selected_indices].sum(axis=1).astype("int64")
        expert_mask = (observed > 0).astype("int8")
        raw[expert_mask == 0] = np.nan
        calibrated[expert_mask == 0] = np.nan
        return pd.DataFrame(
            {
                "raw_probability": raw,
                "calibrated_probability": calibrated,
                "expert_mask": expert_mask,
                "observed_feature_count": observed,
                "observed_feature_fraction": observed / len(self.input_feature_names),
            },
            index=frame.index,
        )

    def audit_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_version": self.model_version,
            "bundle_version": self.bundle_version,
            "feature_schema_version": self.feature_schema_version,
            "feature_schema_sha256": self.feature_schema_sha256,
            "split_id": self.split_id,
            "split_manifest_sha256": self.split_manifest_sha256,
            "dataset_ids": list(self.dataset_ids),
            "input_feature_names": list(self.input_feature_names),
            "preprocessor": self.preprocessor.to_dict(),
            "calibrator": self.calibrator.to_dict(),
            "source_transform": "canonical_same_semantics_identity",
            "hyperparameters": dict(self.hyperparameters),
            "training_scope": self.training_scope,
            "training_participant_sha256": self.training_participant_sha256,
            "training_participant_count": self.training_participant_count,
            "training_row_count": self.training_row_count,
            "run_id": self.run_id,
        }


@dataclass(frozen=True)
class _PreparedInnerFold:
    inner_fold: int
    train_table: pd.DataFrame
    validation_table: pd.DataFrame
    train_matrix: pd.DataFrame
    validation_matrix: pd.DataFrame
    train_target: np.ndarray
    validation_target: np.ndarray
    train_weight: np.ndarray
    preprocessor: SocialContextFoldPreprocessor


def load_social_context_training_config(
    path: str | Path,
) -> SocialContextTrainingConfig:
    config_path = Path(path).resolve()
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise SocialContextExpertError(
            "SocialContext training config is unreadable"
        ) from exc
    if not isinstance(payload, dict):
        raise SocialContextExpertError(
            "SocialContext training config must be a mapping"
        )
    try:
        search_space = {
            str(name): tuple(values)
            for name, values in payload["search"]["space"].items()
        }
        config = SocialContextTrainingConfig(
            run_id=str(payload["run"]["run_id"]),
            random_seed=int(payload["runtime"]["random_seed"]),
            split_id=str(payload["data"]["split_id"]),
            split_sha256=str(payload["data"]["split_sha256"]),
            feature_schema_version=str(payload["data"]["feature_schema_version"]),
            feature_schema_sha256=str(payload["data"]["feature_schema_sha256"]),
            dataset_ids=tuple(str(value) for value in payload["data"]["datasets"]),
            search_space=search_space,
            search_candidate_count=int(payload["search"]["candidate_count"]),
            search_traversal=str(payload["search"]["traversal"]),
            auprc_tie_tolerance=float(payload["search"]["auprc_tie_tolerance"]),
            calibration_positive_participant_threshold=int(
                payload["calibration"]["isotonic_min_positive_participants"]
            ),
            decision_threshold=float(payload["evaluation"]["decision_threshold"]),
            ece_bin_count=int(payload["evaluation"]["ece_bin_count"]),
            catboost_thread_count=int(payload["runtime"]["catboost_thread_count"]),
            config_path=config_path,
            config_sha256=_sha256_file(config_path),
        )
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise SocialContextExpertError(
            "SocialContext training config is malformed"
        ) from exc
    _validate_training_config(payload, config)
    return config


def _validate_training_config(
    payload: Mapping[str, Any],
    config: SocialContextTrainingConfig,
) -> None:
    if payload.get("config_version") != TRAINING_CONFIG_VERSION:
        raise SocialContextExpertError("SocialContext config version changed")
    if payload.get("task_id") != "MODEL-005" or payload.get("model_id") != MODEL_ID:
        raise SocialContextExpertError("SocialContext config task identity changed")
    if payload.get("model_version") != MODEL_VERSION:
        raise SocialContextExpertError("SocialContext config model version changed")
    if (
        config.random_seed != 20260728
        or config.split_id != SPLIT_ID
        or config.split_sha256 != EXPECTED_SPLIT_SHA256
        or config.feature_schema_version != FEATURE_SCHEMA_VERSION
        or config.feature_schema_sha256 != EXPECTED_FEATURE_SCHEMA_SHA256
        or config.dataset_ids != SOCIAL_CONTEXT_DATASET_IDS
    ):
        raise SocialContextExpertError("SocialContext frozen input binding changed")
    expected_space = {
        "iterations": (300, 600),
        "depth": (4, 6),
        "learning_rate": (0.03, 0.05),
        "l2_leaf_reg": (3, 10),
    }
    if dict(config.search_space) != expected_space:
        raise SocialContextExpertError("SocialContext CatBoost search space changed")
    if (
        config.search_traversal != "deterministic_full_grid"
        or config.search_candidate_count
        != math.prod(len(v) for v in expected_space.values())
        or config.auprc_tie_tolerance < 0
        or config.calibration_positive_participant_threshold != 200
        or not 0 < config.decision_threshold < 1
        or config.ece_bin_count < 2
        or config.catboost_thread_count != 1
    ):
        raise SocialContextExpertError(
            "SocialContext implementation settings are invalid"
        )
    if (
        tuple(payload.get("data", {}).get("excluded_datasets", ()))
        != EXCLUDED_DATASET_IDS
    ):
        raise SocialContextExpertError("SocialContext excluded-source policy changed")
    preprocessing = payload.get("preprocessing")
    expected_preprocessing = {
        "source_transform": "canonical_same_semantics_identity",
        "categorical_encoding": "training_fold_category_levels_with_missing_sentinel",
        "numeric_imputation": "training_fold_median",
        "feature_selection": "training_fold_observed_nonconstant",
        "add_missing_indicators": False,
        "use_feature_masks_as_risk_inputs": False,
        "use_feature_coverage_as_risk_input": False,
        "use_activity_sleep_or_physiology_as_risk_inputs": False,
        "use_social_contact_as_risk_inputs": False,
    }
    if preprocessing != expected_preprocessing:
        raise SocialContextExpertError("SocialContext preprocessing policy changed")
    calibration = payload.get("calibration")
    expected_calibration = {
        "method": "isotonic_if_positive_participants_at_least_200_else_platt",
        "isotonic_min_positive_participants": 200,
        "fit_input": "strict_cross_fitted_raw_probability",
        "fit_weighting": "frozen_training_sample_weight",
        "probability_clip_epsilon": 1.0e-6,
    }
    if calibration != expected_calibration:
        raise SocialContextExpertError("SocialContext calibration policy changed")


def deterministic_search_candidates(
    search_space: Mapping[str, Sequence[int | float]],
    *,
    random_seed: int,
    candidate_count: int,
) -> list[dict[str, Any]]:
    names = tuple(search_space)
    candidates: list[dict[str, Any]] = []
    for values in itertools.product(*(search_space[name] for name in names)):
        params = dict(zip(names, values, strict=True))
        digest = _sha256_bytes(
            f"{random_seed}\0".encode("ascii") + _canonical_json_bytes(params)
        )
        candidates.append(
            {
                "candidate_id": f"catboost-{digest[:16]}",
                "selection_digest": digest,
                "params": params,
            }
        )
    candidates.sort(key=lambda item: _canonical_json_bytes(item["params"]))
    if len(candidates) != candidate_count:
        raise SocialContextExpertError("SocialContext must search the complete grid")
    return candidates


def compute_training_sample_weights(frame: pd.DataFrame) -> np.ndarray:
    required = {"dataset_id", "global_participant_id", "binary_target"}
    if not required.issubset(frame.columns) or len(frame) == 0:
        raise SocialContextExpertError("sample-weight frame is incomplete")
    datasets = frame["dataset_id"].astype(str)
    participants = frame["global_participant_id"].astype(str)
    target = pd.to_numeric(frame["binary_target"], errors="raise").astype("int64")
    if not target.isin([0, 1]).all():
        raise SocialContextExpertError("sample-weight target must be binary")
    weights = np.zeros(len(frame), dtype="float64")
    for dataset_id in sorted(set(datasets), key=lambda value: value.encode("utf-8")):
        dataset_mask = datasets.eq(dataset_id).to_numpy()
        classes = sorted(set(target[dataset_mask].tolist()))
        class_mass = 1.0 / len(classes)
        for class_value in classes:
            class_mask = dataset_mask & target.eq(class_value).to_numpy()
            unit_counts = participants[class_mask].value_counts(sort=False)
            unit_mass = class_mass / len(unit_counts)
            for row_index in np.flatnonzero(class_mask):
                weights[row_index] = unit_mass / int(
                    unit_counts[participants.iloc[row_index]]
                )
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise SocialContextExpertError("sample-weight calculation failed")
    weights *= len(weights) / weights.sum()
    return weights


def sample_weight_audit(
    frame: pd.DataFrame, sample_weight: Sequence[float]
) -> dict[str, Any]:
    weight = np.asarray(sample_weight, dtype="float64")
    if len(weight) != len(frame):
        raise SocialContextExpertError("sample-weight audit length does not match")
    by_dataset: dict[str, Any] = {}
    for dataset_id in sorted(set(frame["dataset_id"].astype(str))):
        selected = frame["dataset_id"].astype(str).eq(dataset_id).to_numpy()
        by_dataset[dataset_id] = {
            "total": float(weight[selected].sum()),
            "by_class": {
                str(class_value): float(
                    weight[
                        selected
                        & pd.to_numeric(frame["binary_target"])
                        .eq(class_value)
                        .to_numpy()
                    ].sum()
                )
                for class_value in sorted(
                    set(pd.to_numeric(frame.loc[selected, "binary_target"]).astype(int))
                )
            },
        }
    return {
        "definition": (
            "dataset_equal_then_class_equal_then_participant_class_equal_"
            "then_repeat_window_equal"
        ),
        "normalized_total": float(weight.sum()),
        "by_dataset": by_dataset,
    }


def binary_metrics(
    target: Sequence[int],
    probability: Sequence[float],
    *,
    decision_threshold: float = 0.5,
    ece_bin_count: int = 10,
) -> dict[str, Any]:
    y = np.asarray(target, dtype="int64")
    score = np.asarray(probability, dtype="float64")
    if (
        y.ndim != 1
        or score.ndim != 1
        or len(y) != len(score)
        or len(y) == 0
        or not np.isfinite(score).all()
        or not np.all((0 <= score) & (score <= 1))
        or not set(np.unique(y)).issubset({0, 1})
    ):
        raise SocialContextExpertError("binary metric inputs are invalid")
    predicted = (score >= decision_threshold).astype("int64")
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    curve, ece = _calibration_curve(y, score, ece_bin_count)
    return {
        "metric_version": METRICS_VERSION,
        "row_count": int(len(y)),
        "positive_row_count": int(y.sum()),
        "negative_row_count": int(len(y) - y.sum()),
        "auprc": float(average_precision_score(y, score)) if np.any(y == 1) else None,
        "auroc": float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else None,
        "macro_f1": float(f1_score(y, predicted, average="macro", labels=[0, 1])),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else None,
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "brier_score": float(brier_score_loss(y, score)),
        "ece": ece,
        "decision_threshold": decision_threshold,
        "confusion": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "calibration_curve": curve,
    }


def load_social_context_training_inputs(
    repository_root: str | Path,
    *,
    expected_split_sha256: str = EXPECTED_SPLIT_SHA256,
) -> SocialContextTrainingInputs:
    root = Path(repository_root).resolve()
    split_path = root / DEFAULT_SPLIT_RELATIVE
    if _sha256_file(split_path) != expected_split_sha256:
        raise SocialContextExpertError("frozen DATA-007 split file hash changed")
    try:
        manifest = json.loads(split_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SocialContextExpertError("frozen DATA-007 split is unreadable") from exc
    rebuilt = build_split_manifest(repository_root=root)
    if manifest != rebuilt:
        raise SocialContextExpertError("split, five inputs, or MH-003 schema drifted")
    if (
        manifest.get("split_id") != SPLIT_ID
        or manifest.get("feature_schema_binding", {}).get("snapshot_sha256")
        != EXPECTED_FEATURE_SCHEMA_SHA256
    ):
        raise SocialContextExpertError("frozen DATA-007 binding changed")
    input_rows = manifest.get("inputs")
    if not isinstance(input_rows, list):
        raise SocialContextExpertError("frozen DATA-007 inputs are malformed")
    input_bindings = {
        str(item["dataset_id"]): item for item in input_rows if isinstance(item, dict)
    }
    expected_ids = set(SOCIAL_CONTEXT_DATASET_IDS + EXCLUDED_DATASET_IDS)
    if set(input_bindings) != expected_ids:
        raise SocialContextExpertError("SocialContext dataset membership changed")
    frames: dict[str, pd.DataFrame] = {}
    for dataset_id in SOCIAL_CONTEXT_DATASET_IDS:
        binding = input_bindings[dataset_id]
        path = root / str(binding["canonical_relative_path"])
        if _sha256_file(path) != binding["canonical_sha256"]:
            raise SocialContextExpertError("SocialContext canonical hash changed")
        try:
            frame = pd.read_parquet(path)
        except (OSError, TypeError, ValueError) as exc:
            raise SocialContextExpertError(
                "SocialContext canonical is unreadable"
            ) from exc
        if not isinstance(frame.index, pd.RangeIndex):
            frame = frame.reset_index(drop=True)
        _validate_base_social_frame(frame, dataset_id, binding)
        frames[dataset_id] = frame
    assignments = pd.DataFrame(manifest["participant_assignments"])
    assignments = assignments[
        assignments["dataset_id"].astype(str).isin(SOCIAL_CONTEXT_DATASET_IDS)
    ].copy()
    assignments = assignments.sort_values(
        "global_participant_id", kind="stable"
    ).reset_index(drop=True)
    expected_keys = set().union(
        *(set(frame["global_participant_id"].astype(str)) for frame in frames.values())
    )
    if set(assignments["global_participant_id"].astype(str)) != expected_keys:
        raise SocialContextExpertError(
            "SocialContext split does not cover participants"
        )
    upstream_bindings = _validate_upstream_expert_guards(root)
    return SocialContextTrainingInputs(
        repository_root=root,
        split_manifest=manifest,
        split_manifest_sha256=expected_split_sha256,
        assignments=assignments,
        frames=frames,
        input_bindings={
            dataset_id: input_bindings[dataset_id]
            for dataset_id in SOCIAL_CONTEXT_DATASET_IDS
        },
        upstream_expert_bindings=upstream_bindings,
    )


def train_social_context_expert(
    repository_root: str | Path,
    config_path: str | Path,
    *,
    report_dir: str | Path | None = None,
    model_path: str | Path | None = None,
    model_manifest_path: str | Path | None = None,
    overwrite: bool = False,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Train nested OOF predictions and publish one aggregate production bundle."""

    started = datetime.now(timezone.utc)
    config = load_social_context_training_config(config_path)
    root = Path(repository_root).resolve()
    inputs = load_social_context_training_inputs(
        root, expected_split_sha256=config.split_sha256
    )
    destination = (
        Path(report_dir).resolve()
        if report_dir is not None
        else root / "reports" / "mental_health" / "mood_social" / config.run_id
    )
    production_model = (
        Path(model_path).resolve()
        if model_path is not None
        else root
        / "models"
        / "mental_health"
        / "mood_social"
        / "v3.3.3"
        / "social_context_expert.joblib"
    )
    production_manifest = (
        Path(model_manifest_path).resolve()
        if model_manifest_path is not None
        else production_model.with_name("social_context_expert_manifest.json")
    )
    _check_output_targets(
        destination, production_model, production_manifest, overwrite=overwrite
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    production_model.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{config.run_id}.", dir=destination.parent))
    staged_model = stage / "_publish" / production_model.name
    staged_manifest = stage / "_publish" / production_manifest.name
    try:
        summary = _train_into_stage(
            inputs,
            config,
            stage,
            staged_model,
            staged_manifest,
            started=started,
            command=tuple(command or ()),
        )
        _replace_file(staged_model, production_model, overwrite=overwrite)
        _replace_file(staged_manifest, production_manifest, overwrite=overwrite)
        shutil.rmtree(stage / "_publish")
        if destination.exists():
            if not overwrite:
                raise FileExistsError(destination)
            shutil.rmtree(destination)
        os.replace(stage, destination)
        return {
            **summary,
            "report_dir": destination.as_posix(),
            "model_path": production_model.as_posix(),
            "model_manifest_path": production_manifest.as_posix(),
        }
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def _train_into_stage(
    inputs: SocialContextTrainingInputs,
    config: SocialContextTrainingConfig,
    stage: Path,
    staged_model: Path,
    staged_manifest: Path,
    *,
    started: datetime,
    command: Sequence[str],
) -> dict[str, Any]:
    for relative in (
        "calibration",
        "explanations",
        "failures",
        "folds",
        "preprocessing",
        "_publish",
    ):
        (stage / relative).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config.config_path, stage / "training_config.yaml")
    candidates = deterministic_search_candidates(
        config.search_space,
        random_seed=config.random_seed,
        candidate_count=config.search_candidate_count,
    )
    assignments = _assignment_lookup(inputs.assignments)
    outer_predictions: list[pd.DataFrame] = []
    outer_search_results: list[dict[str, Any]] = []
    fold_audits: list[dict[str, Any]] = []
    raw_production_oof: list[pd.DataFrame] = []
    all_participants = set(assignments)

    for outer_fold in range(OUTER_FOLD_COUNT):
        test_keys = {
            key
            for key, assignment in assignments.items()
            if assignment["outer_fold"] == outer_fold
        }
        train_keys = all_participants - test_keys
        prepared_folds = [
            _prepare_inner_fold(
                inputs,
                assignments,
                outer_fold=outer_fold,
                inner_fold=inner_fold,
                outer_train_keys=train_keys,
            )
            for inner_fold in range(INNER_FOLD_COUNT)
        ]
        search_result = _search_outer_fold(
            candidates, prepared_folds, config=config, outer_fold=outer_fold
        )
        outer_search_results.append(search_result)
        best = search_result["best_candidate"]
        best_oof = search_result.pop("_best_oof")
        fold_bundle, fold_prediction, fold_audit = _fit_outer_fold(
            inputs,
            train_keys=train_keys,
            test_keys=test_keys,
            best_oof=best_oof,
            hyperparameters=best["params"],
            outer_fold=outer_fold,
            config=config,
        )
        outer_predictions.append(fold_prediction)
        raw_production_oof.append(
            fold_prediction[
                [
                    "prediction_id",
                    "dataset_id",
                    "global_participant_id",
                    "canonical_row_index",
                    "binary_target",
                    "outer_fold",
                    "raw_probability",
                    "expert_mask",
                ]
            ].copy()
        )
        fold_audits.append(fold_audit)
        fold_model_path = stage / "folds" / f"outer_fold_{outer_fold}.joblib"
        _write_joblib_atomic(fold_model_path, fold_bundle)
        persisted_fold_bundle = joblib.load(fold_model_path)
        if not isinstance(persisted_fold_bundle, SocialContextExpertBundle):
            raise SocialContextExpertError("persisted outer fold bundle is invalid")
        fold_audit["catboost_model_sha256"] = _catboost_model_sha256(
            persisted_fold_bundle.classifier
        )
        _write_json(stage / "folds" / f"outer_fold_{outer_fold}.json", fold_audit)
        _write_preprocessing_audit(
            stage / "preprocessing" / f"outer_fold_{outer_fold}", fold_bundle
        )
        _write_json(
            stage / "calibration" / f"outer_fold_{outer_fold}.json",
            fold_bundle.calibrator.to_dict(),
        )

    oof = (
        pd.concat(outer_predictions, ignore_index=True)
        .sort_values(["dataset_id", "canonical_row_index"], kind="stable")
        .reset_index(drop=True)
    )
    _validate_oof_predictions(oof, inputs, assignments)
    metrics = _build_oof_metrics(oof, config)
    final_candidate = _select_final_candidate(outer_search_results, config)
    raw_oof = (
        pd.concat(raw_production_oof, ignore_index=True)
        .sort_values(["dataset_id", "canonical_row_index"], kind="stable")
        .reset_index(drop=True)
    )
    final_bundle, final_audit = _fit_final_bundle(
        inputs,
        raw_oof=raw_oof,
        hyperparameters=final_candidate["params"],
        config=config,
    )
    _assert_bundle_mask_diagnostic(final_bundle)
    _write_joblib_atomic(staged_model, final_bundle)
    persisted_final_bundle = joblib.load(staged_model)
    if not isinstance(persisted_final_bundle, SocialContextExpertBundle):
        raise SocialContextExpertError("persisted production bundle is invalid")
    final_bundle = persisted_final_bundle
    _assert_bundle_mask_diagnostic(final_bundle)
    final_audit["catboost_model_sha256"] = _catboost_model_sha256(
        final_bundle.classifier
    )
    model_sha256 = _sha256_file(staged_model)
    _write_preprocessing_audit(stage / "preprocessing" / "final", final_bundle)
    _write_json(stage / "preprocessing" / "final" / "bundle.json", final_audit)
    _write_json(
        stage / "calibration" / "production.json", final_bundle.calibrator.to_dict()
    )
    _write_json(
        stage / "search_results.json",
        {
            "traversal": config.search_traversal,
            "candidate_count": len(candidates),
            "full_grid_count": math.prod(
                len(values) for values in config.search_space.values()
            ),
            "candidate_ids": [item["candidate_id"] for item in candidates],
            "outer_folds": outer_search_results,
            "production_selection": final_candidate,
        },
    )
    _write_predictions(stage / "predictions.parquet", oof)
    predictions_sha256 = _sha256_file(stage / "predictions.parquet")
    _write_json(stage / "metrics.json", metrics)
    metrics_sha256 = _sha256_file(stage / "metrics.json")
    _write_json(
        stage / "explanations" / "feature_importance.json",
        _feature_importance(final_bundle),
    )
    warnings = _training_warnings(final_bundle, inputs)
    _write_json(
        stage / "failures" / "warnings.json", {"failures": [], "warnings": warnings}
    )
    ended = datetime.now(timezone.utc)
    code_bindings = _code_bindings(inputs.repository_root)
    run_manifest = _run_manifest(
        inputs,
        config,
        metrics,
        final_bundle,
        final_candidate,
        fold_audits,
        model_sha256=model_sha256,
        predictions_sha256=predictions_sha256,
        metrics_sha256=metrics_sha256,
        started=started,
        ended=ended,
        warnings=warnings,
        command=command,
        code_bindings=code_bindings,
    )
    _write_json(stage / "run.json", run_manifest)
    model_manifest = {
        "manifest_version": MODEL_MANIFEST_VERSION,
        "status": "active",
        "model_id": MODEL_ID,
        "model_version": MODEL_VERSION,
        "run_id": config.run_id,
        "model_file": staged_model.name,
        "model_sha256": model_sha256,
        "training_config_sha256": config.config_sha256,
        "split_id": SPLIT_ID,
        "split_manifest_sha256": EXPECTED_SPLIT_SHA256,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_schema_sha256": EXPECTED_FEATURE_SCHEMA_SHA256,
        "dataset_ids": list(SOCIAL_CONTEXT_DATASET_IDS),
        "excluded_dataset_ids": list(EXCLUDED_DATASET_IDS),
        "input_feature_names": list(SOCIAL_CONTEXT_FEATURE_COLUMNS),
        "selected_feature_names": list(final_bundle.preprocessor.selected_features),
        "categorical_feature_names": list(
            final_bundle.preprocessor.categorical_features
        ),
        "numeric_feature_names": list(final_bundle.preprocessor.numeric_features),
        "hyperparameters": dict(final_bundle.hyperparameters),
        "calibrator": final_bundle.calibrator.to_dict(),
        "source_transform": "canonical_same_semantics_identity",
        "training_scope": final_bundle.training_scope,
        "training_participant_count": final_bundle.training_participant_count,
        "training_participant_sha256": final_bundle.training_participant_sha256,
        "training_row_count": final_bundle.training_row_count,
        "metrics_sha256": metrics_sha256,
        "predictions_sha256": predictions_sha256,
        "code_bindings": code_bindings,
        "dependencies": _dependency_versions(),
        "upstream_expert_guards": dict(inputs.upstream_expert_bindings),
        "expert_mask_rule": "at_least_one_selected_social_context_input",
        "forbidden_inputs": [
            "questionnaire_items",
            "publisher_labels",
            "ssq_specific_fields",
            "activity",
            "sleep",
            "physiology",
            "social_contact",
            "s10_call_fields",
        ],
    }
    _write_json(staged_manifest, model_manifest)
    _write_json(
        stage / "model_artifact.json",
        {
            **model_manifest,
            "model_file": "models/mental_health/mood_social/v3.3.3/social_context_expert.joblib",
            "model_manifest_file": "models/mental_health/mood_social/v3.3.3/social_context_expert_manifest.json",
        },
    )
    _write_json(stage / "artifacts.json", _report_artifact_hashes(stage))
    return {
        "run_id": config.run_id,
        "status": "completed",
        "model_id": MODEL_ID,
        "model_sha256": model_sha256,
        "predictions_sha256": predictions_sha256,
        "metrics_sha256": metrics_sha256,
        "row_count": int(len(oof)),
        "positive_row_count": int(oof["binary_target"].sum()),
        "participant_count": int(oof["global_participant_id"].nunique()),
        "calibration_method": final_bundle.calibrator.method,
        "selected_features": list(final_bundle.preprocessor.selected_features),
        "final_hyperparameters": dict(final_bundle.hyperparameters),
        "overall_metrics": metrics["overall"]["calibrated"],
    }


def _prepare_inner_fold(
    inputs: SocialContextTrainingInputs,
    assignments: Mapping[str, Mapping[str, Any]],
    *,
    outer_fold: int,
    inner_fold: int,
    outer_train_keys: set[str],
) -> _PreparedInnerFold:
    validation_keys = {
        key
        for key in outer_train_keys
        if assignments[key]["inner_validation_fold_by_outer_fold"][str(outer_fold)]
        == inner_fold
    }
    train_keys = outer_train_keys - validation_keys
    if not train_keys or not validation_keys or train_keys & validation_keys:
        raise SocialContextExpertError("inner fold participant relation is invalid")
    train_table = _available_social_rows(_transform_partition(inputs, train_keys))
    validation_table = _available_social_rows(
        _transform_partition(inputs, validation_keys)
    )
    preprocessor = SocialContextFoldPreprocessor.fit(train_table)
    return _PreparedInnerFold(
        inner_fold=inner_fold,
        train_table=train_table,
        validation_table=validation_table,
        train_matrix=preprocessor.transform(train_table),
        validation_matrix=preprocessor.transform(validation_table),
        train_target=train_table["binary_target"].to_numpy(dtype="int64"),
        validation_target=validation_table["binary_target"].to_numpy(dtype="int64"),
        train_weight=compute_training_sample_weights(train_table),
        preprocessor=preprocessor,
    )


def _search_outer_fold(
    candidates: Sequence[Mapping[str, Any]],
    prepared_folds: Sequence[_PreparedInnerFold],
    *,
    config: SocialContextTrainingConfig,
    outer_fold: int,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    predictions_by_candidate: dict[str, pd.DataFrame] = {}
    for candidate in candidates:
        fold_metrics: list[dict[str, Any]] = []
        oof_parts: list[pd.DataFrame] = []
        for prepared in prepared_folds:
            classifier = _fit_classifier(
                prepared.train_matrix,
                prepared.train_target,
                prepared.train_weight,
                candidate["params"],
                config,
                prepared.preprocessor,
                training_identity=(
                    f"search_outer_{outer_fold}_inner_{prepared.inner_fold}_"
                    f"{candidate['candidate_id']}"
                ),
            )
            probability = np.asarray(
                classifier.predict_proba(prepared.validation_matrix)[:, 1],
                dtype="float64",
            )
            metric = binary_metrics(
                prepared.validation_target,
                probability,
                decision_threshold=config.decision_threshold,
                ece_bin_count=config.ece_bin_count,
            )
            fold_metrics.append(
                {
                    "inner_fold": prepared.inner_fold,
                    "auprc": metric["auprc"],
                    "brier_score": metric["brier_score"],
                    "row_count": metric["row_count"],
                    "positive_row_count": metric["positive_row_count"],
                    "selected_features": list(prepared.preprocessor.selected_features),
                }
            )
            part = prepared.validation_table[
                [
                    "prediction_id",
                    "dataset_id",
                    "global_participant_id",
                    "canonical_row_index",
                    "binary_target",
                ]
            ].copy()
            part["raw_probability"] = probability
            oof_parts.append(part)
        mean_auprc = float(np.mean([item["auprc"] for item in fold_metrics]))
        mean_brier = float(np.mean([item["brier_score"] for item in fold_metrics]))
        result = {
            "candidate_id": candidate["candidate_id"],
            "params": dict(candidate["params"]),
            "mean_inner_auprc": mean_auprc,
            "mean_inner_brier_score": mean_brier,
            "inner_folds": fold_metrics,
        }
        results.append(result)
        predictions_by_candidate[str(candidate["candidate_id"])] = pd.concat(
            oof_parts, ignore_index=True
        ).sort_values(["dataset_id", "canonical_row_index"], kind="stable")
    ranked = sorted(
        results,
        key=lambda item: (
            -_round_for_tie(
                float(item["mean_inner_auprc"]), config.auprc_tie_tolerance
            ),
            float(item["mean_inner_brier_score"]),
            str(item["candidate_id"]),
        ),
    )
    best = ranked[0]
    return {
        "outer_fold": outer_fold,
        "selection_rule": (
            "highest_mean_inner_auprc_then_lowest_mean_inner_brier_then_candidate_id"
        ),
        "best_candidate": best,
        "candidates": results,
        "_best_oof": predictions_by_candidate[str(best["candidate_id"])],
    }


def _fit_outer_fold(
    inputs: SocialContextTrainingInputs,
    *,
    train_keys: set[str],
    test_keys: set[str],
    best_oof: pd.DataFrame,
    hyperparameters: Mapping[str, int | float],
    outer_fold: int,
    config: SocialContextTrainingConfig,
) -> tuple[SocialContextExpertBundle, pd.DataFrame, dict[str, Any]]:
    train_partition = _transform_partition(inputs, train_keys)
    test_table = _transform_partition(inputs, test_keys)
    train_table = _available_social_rows(train_partition)
    preprocessor = SocialContextFoldPreprocessor.fit(train_table)
    train_weight = compute_training_sample_weights(train_table)
    classifier = _fit_classifier(
        preprocessor.transform(train_table),
        train_table["binary_target"].to_numpy(dtype="int64"),
        train_weight,
        hyperparameters,
        config,
        preprocessor,
        training_identity=f"outer_fold_{outer_fold}_aggregate",
    )
    best_oof = best_oof.sort_values(
        ["dataset_id", "canonical_row_index"], kind="stable"
    ).reset_index(drop=True)
    calibration_weight = compute_training_sample_weights(best_oof)
    positive_participants = int(
        best_oof.loc[best_oof["binary_target"].eq(1), "global_participant_id"].nunique()
    )
    calibrator = SocialContextProbabilityCalibrator.fit(
        best_oof["raw_probability"],
        best_oof["binary_target"],
        calibration_weight,
        positive_participant_count=positive_participants,
        isotonic_min_positive_participants=(
            config.calibration_positive_participant_threshold
        ),
    )
    bundle = SocialContextExpertBundle(
        model_id=MODEL_ID,
        model_version=MODEL_VERSION,
        bundle_version=MODEL_BUNDLE_VERSION,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        feature_schema_sha256=EXPECTED_FEATURE_SCHEMA_SHA256,
        split_id=SPLIT_ID,
        split_manifest_sha256=EXPECTED_SPLIT_SHA256,
        dataset_ids=SOCIAL_CONTEXT_DATASET_IDS,
        input_feature_names=SOCIAL_CONTEXT_FEATURE_COLUMNS,
        preprocessor=preprocessor,
        classifier=classifier,
        calibrator=calibrator,
        hyperparameters=dict(hyperparameters),
        training_scope=f"outer_fold_{outer_fold}_train",
        training_participant_sha256=participant_set_sha256(
            set(train_table["global_participant_id"].astype(str))
        ),
        training_participant_count=int(train_table["global_participant_id"].nunique()),
        training_row_count=len(train_table),
        run_id=config.run_id,
    )
    predicted = bundle.predict(test_table)
    output = test_table[
        [
            "prediction_id",
            "dataset_id",
            "global_participant_id",
            "canonical_row_index",
            "binary_target",
        ]
    ].copy()
    output["outer_fold"] = outer_fold
    output = pd.concat(
        [output.reset_index(drop=True), predicted.reset_index(drop=True)], axis=1
    )
    audit = {
        **bundle.audit_dict(),
        "outer_fold": outer_fold,
        "test_participant_count": len(test_keys),
        "test_participant_sha256": participant_set_sha256(test_keys),
        "test_row_count": len(test_table),
        "preprocessing_participant_count": len(train_keys),
        "preprocessing_participant_sha256": participant_set_sha256(train_keys),
        "preprocessing_row_count": len(train_partition),
        "training_sample_weights": sample_weight_audit(train_table, train_weight),
        "calibration_sample_weights": sample_weight_audit(best_oof, calibration_weight),
        "raw_test_metrics": binary_metrics(
            output.loc[output["expert_mask"].eq(1), "binary_target"],
            output.loc[output["expert_mask"].eq(1), "raw_probability"],
            decision_threshold=config.decision_threshold,
            ece_bin_count=config.ece_bin_count,
        ),
        "calibrated_test_metrics": binary_metrics(
            output.loc[output["expert_mask"].eq(1), "binary_target"],
            output.loc[output["expert_mask"].eq(1), "calibrated_probability"],
            decision_threshold=config.decision_threshold,
            ece_bin_count=config.ece_bin_count,
        ),
        "catboost_model_sha256": _catboost_model_sha256(classifier),
    }
    return bundle, output, audit


def _fit_final_bundle(
    inputs: SocialContextTrainingInputs,
    *,
    raw_oof: pd.DataFrame,
    hyperparameters: Mapping[str, int | float],
    config: SocialContextTrainingConfig,
) -> tuple[SocialContextExpertBundle, dict[str, Any]]:
    assignments = _assignment_lookup(inputs.assignments)
    all_keys = set(assignments)
    if (
        len(raw_oof) != sum(len(frame) for frame in inputs.frames.values())
        or raw_oof["prediction_id"].duplicated().any()
        or set(raw_oof["global_participant_id"].astype(str)) != all_keys
    ):
        raise SocialContextExpertError("production calibrator OOF coverage is invalid")
    partition = _transform_partition(inputs, all_keys)
    table = _available_social_rows(partition)
    preprocessor = SocialContextFoldPreprocessor.fit(table)
    weight = compute_training_sample_weights(table)
    classifier = _fit_classifier(
        preprocessor.transform(table),
        table["binary_target"].to_numpy(dtype="int64"),
        weight,
        hyperparameters,
        config,
        preprocessor,
        training_identity="production_aggregate",
    )
    calibration_table = raw_oof[raw_oof["expert_mask"].eq(1)].reset_index(drop=True)
    calibration_weight = compute_training_sample_weights(calibration_table)
    positive_participants = int(
        calibration_table.loc[
            calibration_table["binary_target"].eq(1), "global_participant_id"
        ].nunique()
    )
    calibrator = SocialContextProbabilityCalibrator.fit(
        calibration_table["raw_probability"],
        calibration_table["binary_target"],
        calibration_weight,
        positive_participant_count=positive_participants,
        isotonic_min_positive_participants=(
            config.calibration_positive_participant_threshold
        ),
    )
    bundle = SocialContextExpertBundle(
        model_id=MODEL_ID,
        model_version=MODEL_VERSION,
        bundle_version=MODEL_BUNDLE_VERSION,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        feature_schema_sha256=EXPECTED_FEATURE_SCHEMA_SHA256,
        split_id=SPLIT_ID,
        split_manifest_sha256=EXPECTED_SPLIT_SHA256,
        dataset_ids=SOCIAL_CONTEXT_DATASET_IDS,
        input_feature_names=SOCIAL_CONTEXT_FEATURE_COLUMNS,
        preprocessor=preprocessor,
        classifier=classifier,
        calibrator=calibrator,
        hyperparameters=dict(hyperparameters),
        training_scope="all_social_context_sources_for_production",
        training_participant_sha256=participant_set_sha256(
            set(table["global_participant_id"].astype(str))
        ),
        training_participant_count=int(table["global_participant_id"].nunique()),
        training_row_count=len(table),
        run_id=config.run_id,
    )
    return bundle, {
        **bundle.audit_dict(),
        "preprocessing_participant_count": len(all_keys),
        "preprocessing_participant_sha256": participant_set_sha256(all_keys),
        "preprocessing_row_count": len(partition),
        "training_sample_weights": sample_weight_audit(table, weight),
        "calibration_sample_weights": sample_weight_audit(
            calibration_table, calibration_weight
        ),
        "catboost_model_sha256": _catboost_model_sha256(classifier),
    }


def _transform_partition(
    inputs: SocialContextTrainingInputs,
    participant_keys: set[str],
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for dataset_order, dataset_id in enumerate(SOCIAL_CONTEXT_DATASET_IDS):
        base = inputs.frames[dataset_id]
        selected = base[
            base["global_participant_id"].astype(str).isin(participant_keys)
        ]
        if selected.empty:
            raise SocialContextExpertError("training partition omitted a social source")
        part = selected[
            [
                "dataset_id",
                "global_participant_id",
                "binary_target",
                *SOCIAL_CONTEXT_FEATURE_COLUMNS,
                *SOCIAL_CONTEXT_MASK_COLUMNS,
            ]
        ].copy()
        part["canonical_row_index"] = selected.index.to_numpy(dtype="int64")
        part["prediction_id"] = (
            part["dataset_id"].astype(str)
            + "::row="
            + part["canonical_row_index"].astype(str)
        )
        part["_dataset_order"] = dataset_order
        parts.append(part)
    result = (
        pd.concat(parts, ignore_index=True)
        .sort_values(["_dataset_order", "canonical_row_index"], kind="stable")
        .drop(columns="_dataset_order")
    )
    result = result.reset_index(drop=True)
    _masked_social_values(result)
    return result


def _available_social_rows(frame: pd.DataFrame) -> pd.DataFrame:
    _, masks = _masked_social_values(frame)
    available = masks.sum(axis=1) > 0
    result = frame.loc[available].reset_index(drop=True)
    if result.empty:
        raise SocialContextExpertError(
            "SocialContext fold has no available feature rows"
        )
    return result


def _fit_classifier(
    matrix: pd.DataFrame,
    target: np.ndarray,
    sample_weight: np.ndarray,
    hyperparameters: Mapping[str, int | float],
    config: SocialContextTrainingConfig,
    preprocessor: SocialContextFoldPreprocessor,
    *,
    training_identity: str,
) -> CatBoostClassifier:
    categorical_indices = [
        index
        for index, feature in enumerate(matrix.columns)
        if feature in preprocessor.categorical_features
    ]
    classifier = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Logloss",
        random_seed=config.random_seed,
        thread_count=config.catboost_thread_count,
        allow_writing_files=False,
        verbose=False,
        random_strength=0.0,
        bootstrap_type="No",
        leaf_estimation_method="Newton",
        metadata=_deterministic_catboost_metadata(
            config,
            hyperparameters,
            training_identity=training_identity,
        ),
        **dict(hyperparameters),
    )
    classifier.fit(
        matrix,
        target,
        sample_weight=sample_weight,
        cat_features=categorical_indices,
        verbose=False,
    )
    return classifier


def _deterministic_catboost_metadata(
    config: SocialContextTrainingConfig,
    hyperparameters: Mapping[str, int | float],
    *,
    training_identity: str,
) -> dict[str, str]:
    if not training_identity or "\0" in training_identity:
        raise SocialContextExpertError("CatBoost training identity is invalid")
    payload = {
        "model_id": MODEL_ID,
        "model_version": MODEL_VERSION,
        "run_id": config.run_id,
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


def _select_final_candidate(
    outer_search_results: Sequence[Mapping[str, Any]],
    config: SocialContextTrainingConfig,
) -> dict[str, Any]:
    by_candidate: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for outer in outer_search_results:
        for candidate in outer["candidates"]:
            by_candidate[str(candidate["candidate_id"])].append(candidate)
    aggregated: list[dict[str, Any]] = []
    for candidate_id, rows in by_candidate.items():
        if len(rows) != OUTER_FOLD_COUNT:
            raise SocialContextExpertError("production candidate lacks search coverage")
        aggregated.append(
            {
                "candidate_id": candidate_id,
                "params": dict(rows[0]["params"]),
                "mean_nested_inner_auprc": float(
                    np.mean([row["mean_inner_auprc"] for row in rows])
                ),
                "mean_nested_inner_brier_score": float(
                    np.mean([row["mean_inner_brier_score"] for row in rows])
                ),
            }
        )
    ranked = sorted(
        aggregated,
        key=lambda item: (
            -_round_for_tie(
                item["mean_nested_inner_auprc"], config.auprc_tie_tolerance
            ),
            item["mean_nested_inner_brier_score"],
            item["candidate_id"],
        ),
    )
    return {
        **ranked[0],
        "selection_rule": (
            "highest_mean_nested_inner_auprc_then_lowest_mean_nested_inner_"
            "brier_then_candidate_id_without_outer_test_metrics"
        ),
        "all_candidates": aggregated,
    }


def _build_oof_metrics(
    oof: pd.DataFrame,
    config: SocialContextTrainingConfig,
) -> dict[str, Any]:
    def pair(frame: pd.DataFrame) -> dict[str, Any]:
        available = frame[frame["expert_mask"].eq(1)]
        return {
            "raw": binary_metrics(
                available["binary_target"],
                available["raw_probability"],
                decision_threshold=config.decision_threshold,
                ece_bin_count=config.ece_bin_count,
            ),
            "calibrated": binary_metrics(
                available["binary_target"],
                available["calibrated_probability"],
                decision_threshold=config.decision_threshold,
                ece_bin_count=config.ece_bin_count,
            ),
        }

    return {
        "metrics_version": METRICS_VERSION,
        "model_id": MODEL_ID,
        "model_version": MODEL_VERSION,
        "run_id": config.run_id,
        "prediction_scope": "strict_outer_fold_oof",
        "primary_metric": "auprc",
        "hyperparameter_tie_break": "lower_brier_score",
        "decision_threshold": config.decision_threshold,
        "ece": {
            "binning": "equal_width",
            "bin_count": config.ece_bin_count,
            "empty_bins": "omitted",
        },
        "availability": {
            "row_count": int(len(oof)),
            "available_row_count": int(oof["expert_mask"].sum()),
            "unavailable_row_count": int(oof["expert_mask"].eq(0).sum()),
        },
        "overall": pair(oof),
        "by_dataset": {
            dataset_id: pair(oof[oof["dataset_id"].astype(str).eq(dataset_id)])
            for dataset_id in SOCIAL_CONTEXT_DATASET_IDS
        },
        "by_outer_fold": {
            str(outer_fold): pair(oof[oof["outer_fold"].eq(outer_fold)])
            for outer_fold in range(OUTER_FOLD_COUNT)
        },
    }


def _validate_oof_predictions(
    oof: pd.DataFrame,
    inputs: SocialContextTrainingInputs,
    assignments: Mapping[str, Mapping[str, Any]],
) -> None:
    expected_rows = sum(len(frame) for frame in inputs.frames.values())
    expected_positive = sum(
        int(pd.to_numeric(frame["binary_target"]).sum())
        for frame in inputs.frames.values()
    )
    expected_ids = {
        f"{dataset_id}::row={row_index}"
        for dataset_id, frame in inputs.frames.items()
        for row_index in range(len(frame))
    }
    if (
        len(oof) != expected_rows
        or oof["prediction_id"].duplicated().any()
        or set(oof["prediction_id"].astype(str)) != expected_ids
        or int(oof["binary_target"].sum()) != expected_positive
        or set(oof["dataset_id"].astype(str)) != set(SOCIAL_CONTEXT_DATASET_IDS)
    ):
        raise SocialContextExpertError("outer OOF prediction coverage is invalid")
    if not oof["expert_mask"].isin([0, 1]).all():
        raise SocialContextExpertError("outer OOF expert mask is invalid")
    available = oof["expert_mask"].eq(1)
    if (
        oof.loc[available, ["raw_probability", "calibrated_probability"]]
        .isna()
        .any()
        .any()
        or oof.loc[~available, ["raw_probability", "calibrated_probability"]]
        .notna()
        .any()
        .any()
    ):
        raise SocialContextExpertError("outer OOF probability-mask relation is invalid")
    expected_outer = (
        oof["global_participant_id"]
        .astype(str)
        .map({key: value["outer_fold"] for key, value in assignments.items()})
    )
    if not expected_outer.astype("int64").equals(oof["outer_fold"].astype("int64")):
        raise SocialContextExpertError("outer OOF fold assignment drifted")


def _validate_base_social_frame(
    frame: pd.DataFrame,
    dataset_id: str,
    binding: Mapping[str, Any],
) -> None:
    required = {
        "dataset_id",
        "global_participant_id",
        "binary_target",
        *SOCIAL_CONTEXT_FEATURE_COLUMNS,
        *SOCIAL_CONTEXT_MASK_COLUMNS,
    }
    if not required.issubset(frame.columns):
        raise SocialContextExpertError("SocialContext canonical columns changed")
    if (
        set(frame["dataset_id"].astype(str)) != {dataset_id}
        or len(frame) != int(binding["row_count"])
        or int(pd.to_numeric(frame["binary_target"]).sum())
        != int(binding["positive_row_count"])
        or frame["global_participant_id"].astype(str).nunique()
        != int(binding["participant_count"])
    ):
        raise SocialContextExpertError("SocialContext canonical summary changed")
    _, masks = _masked_social_values(frame)
    if not np.any(masks):
        raise SocialContextExpertError("SocialContext source has no profile evidence")


def _masked_social_values(
    frame: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    missing_columns = set(
        SOCIAL_CONTEXT_FEATURE_COLUMNS + SOCIAL_CONTEXT_MASK_COLUMNS
    ) - set(frame.columns)
    if missing_columns:
        raise SocialContextExpertError(
            "SocialContext unified feature columns are missing"
        )
    values: dict[str, np.ndarray] = {}
    masks = np.empty((len(frame), len(SOCIAL_CONTEXT_FEATURE_COLUMNS)), dtype="int8")
    for index, (feature, mask_name) in enumerate(
        zip(SOCIAL_CONTEXT_FEATURE_COLUMNS, SOCIAL_CONTEXT_MASK_COLUMNS, strict=True)
    ):
        try:
            mask_float = pd.to_numeric(frame[mask_name], errors="raise").to_numpy(
                dtype="float64", na_value=np.nan
            )
        except (TypeError, ValueError) as exc:
            raise SocialContextExpertError(
                "SocialContext feature masks are invalid"
            ) from exc
        if (
            not np.isfinite(mask_float).all()
            or not np.isin(mask_float, [0.0, 1.0]).all()
        ):
            raise SocialContextExpertError("SocialContext feature masks are invalid")
        mask = mask_float.astype("int8")
        if feature in SOCIAL_CONTEXT_CATEGORICAL_COLUMNS:
            raw = frame[feature].astype("object").to_numpy()
            normalized = np.empty(len(raw), dtype="object")
            missing = pd.isna(raw)
            for row_index, item in enumerate(raw):
                normalized[row_index] = None if missing[row_index] else str(item)
            finite = ~missing
            if np.any(mask.astype(bool) & ~finite) or np.any(
                ~mask.astype(bool) & finite
            ):
                raise SocialContextExpertError(
                    "SocialContext value-mask relation is invalid"
                )
            allowed = set(_spec_categories(feature))
            observed = {str(item) for item in normalized[mask.astype(bool)]}
            if not observed.issubset(allowed):
                raise SocialContextExpertError(
                    "SocialContext category value is invalid"
                )
            values[feature] = normalized
        else:
            try:
                numeric = pd.to_numeric(frame[feature], errors="raise").to_numpy(
                    dtype="float64", na_value=np.nan
                )
            except (TypeError, ValueError) as exc:
                raise SocialContextExpertError(
                    "SocialContext numeric value is invalid"
                ) from exc
            finite = np.isfinite(numeric)
            if np.any(mask.astype(bool) & ~finite) or np.any(
                ~mask.astype(bool) & finite
            ):
                raise SocialContextExpertError(
                    "SocialContext value-mask relation is invalid"
                )
            values[feature] = numeric.astype("object")
        masks[:, index] = mask
    return values, masks


def _validate_upstream_expert_guards(root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name in ("activity", "sleep", "joint", "physiology"):
        guard = UPSTREAM_EXPERT_GUARDS[name]
        model_path = root / guard["model_path"]
        manifest_path = root / guard["manifest_path"]
        if (
            _sha256_file(model_path) != guard["model_sha256"]
            or _sha256_file(manifest_path) != guard["manifest_sha256"]
        ):
            raise SocialContextExpertError(
                f"upstream {name} expert model or manifest drifted"
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise SocialContextExpertError(
                f"upstream {name} expert manifest is unreadable"
            ) from exc
        expected_id, expected_version = _UPSTREAM_IDENTITIES[name]
        if (
            manifest.get("model_id") != expected_id
            or manifest.get("model_version") != expected_version
            or manifest.get("model_sha256") != guard["model_sha256"]
            or manifest.get("split_manifest_sha256") != EXPECTED_SPLIT_SHA256
            or manifest.get("feature_schema_sha256") != EXPECTED_FEATURE_SCHEMA_SHA256
        ):
            raise SocialContextExpertError(
                f"upstream {name} expert identity or frozen binding changed"
            )
        result[name] = {
            **dict(guard),
            "model_bytes": model_path.stat().st_size,
            "manifest_bytes": manifest_path.stat().st_size,
            "model_id": expected_id,
            "model_version": expected_version,
        }
    return result


def _calibration_curve(
    target: np.ndarray,
    probability: np.ndarray,
    bin_count: int,
) -> tuple[list[dict[str, Any]], float]:
    edges = np.linspace(0.0, 1.0, bin_count + 1)
    indices = np.minimum(
        np.searchsorted(edges, probability, side="right") - 1, bin_count - 1
    )
    indices = np.maximum(indices, 0)
    curve: list[dict[str, Any]] = []
    ece = 0.0
    for bin_index in range(bin_count):
        selected = indices == bin_index
        if not selected.any():
            continue
        mean_probability = float(probability[selected].mean())
        observed_rate = float(target[selected].mean())
        count = int(selected.sum())
        ece += (count / len(target)) * abs(mean_probability - observed_rate)
        curve.append(
            {
                "bin_index": bin_index,
                "lower": float(edges[bin_index]),
                "upper": float(edges[bin_index + 1]),
                "row_count": count,
                "mean_probability": mean_probability,
                "observed_positive_rate": observed_rate,
            }
        )
    return curve, float(ece)


def _probability_logit(
    probability: np.ndarray,
    epsilon: float = 1.0e-6,
) -> np.ndarray:
    clipped = np.clip(probability, epsilon, 1.0 - epsilon)
    return np.log(clipped / (1.0 - clipped))


def _assignment_lookup(
    assignments: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    return {
        str(row["global_participant_id"]): {
            "outer_fold": int(row["outer_fold"]),
            "inner_validation_fold_by_outer_fold": dict(
                row["inner_validation_fold_by_outer_fold"]
            ),
        }
        for row in assignments.to_dict(orient="records")
    }


def participant_set_sha256(participant_ids: Sequence[str] | set[str]) -> str:
    digest = hashlib.sha256()
    values = list(participant_ids)
    if not values or len(values) != len(set(values)):
        raise SocialContextExpertError(
            "participant hash input must be unique and non-empty"
        )
    for participant_id in sorted(values, key=lambda value: value.encode("utf-8")):
        if (
            not isinstance(participant_id, str)
            or not participant_id
            or "\0" in participant_id
        ):
            raise SocialContextExpertError("participant hash input is invalid")
        digest.update(participant_id.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _round_for_tie(value: float, tolerance: float) -> float:
    if tolerance <= 0:
        return value
    return round(value / tolerance) * tolerance


def _catboost_model_sha256(classifier: CatBoostClassifier) -> str:
    with tempfile.TemporaryDirectory(prefix="mood-social-catboost-") as directory:
        path = Path(directory) / "model.cbm"
        classifier.save_model(path, format="cbm")
        return _sha256_file(path)


def _feature_importance(bundle: SocialContextExpertBundle) -> dict[str, Any]:
    importance = bundle.classifier.get_feature_importance(type="FeatureImportance")
    rows = [
        {"feature": feature, "importance": float(value)}
        for feature, value in zip(
            bundle.preprocessor.selected_features, importance, strict=True
        )
    ]
    rows.sort(key=lambda item: (-item["importance"], item["feature"]))
    return {
        "method": "catboost_prediction_values_change_for_training_audit",
        "production_attribution_method": "shap_deferred_to_inference_explanation_task",
        "features": rows,
    }


def _assert_bundle_mask_diagnostic(bundle: SocialContextExpertBundle) -> None:
    frame = pd.DataFrame(
        {
            **{name: [np.nan] for name in SOCIAL_CONTEXT_FEATURE_COLUMNS},
            **{name: [0] for name in SOCIAL_CONTEXT_MASK_COLUMNS},
        }
    )
    predicted = bundle.predict(frame)
    if (
        int(predicted.loc[0, "expert_mask"]) != 0
        or not pd.isna(predicted.loc[0, "raw_probability"])
        or not pd.isna(predicted.loc[0, "calibrated_probability"])
    ):
        raise SocialContextExpertError(
            "SocialContext all-missing mask diagnostic failed"
        )


def _training_warnings(
    bundle: SocialContextExpertBundle,
    inputs: SocialContextTrainingInputs,
) -> list[dict[str, str]]:
    unselected = sorted(
        set(SOCIAL_CONTEXT_FEATURE_COLUMNS) - set(bundle.preprocessor.selected_features)
    )
    return [
        {
            "code": "CROSS_SOURCE_PROFILE_PROXY",
            "message": (
                "Strictly synonymous profile fields are combined across four survey "
                "sources; this is state association, not prospective diagnosis."
            ),
        },
        {
            "code": "SSQ_FIELDS_EXCLUDED",
            "message": (
                "NHANES 2005-2008 SSQ-specific support fields remain controlled "
                "source columns and are not SocialContextExpert risk inputs."
            ),
        },
        {
            "code": "SMALL_RESILIENT_SOURCE",
            "message": (
                "RESILIENT contributes 73 participants and 10 positive rows; "
                "source-specific metrics should be interpreted with that sample size."
            ),
        },
        {
            "code": "STRUCTURAL_FEATURE_ABSENCE",
            "message": "Training-fold selection excluded: " + ", ".join(unselected),
        },
        {
            "code": "NO_PERFORMANCE_GATE",
            "message": (
                "Metrics are reported as observed; MODEL-005 has no minimum "
                "performance promotion threshold."
            ),
        },
    ]


def _run_manifest(
    inputs: SocialContextTrainingInputs,
    config: SocialContextTrainingConfig,
    metrics: Mapping[str, Any],
    bundle: SocialContextExpertBundle,
    final_candidate: Mapping[str, Any],
    fold_audits: Sequence[Mapping[str, Any]],
    *,
    model_sha256: str,
    predictions_sha256: str,
    metrics_sha256: str,
    started: datetime,
    ended: datetime,
    warnings: Sequence[Mapping[str, str]],
    command: Sequence[str],
    code_bindings: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "run_manifest_version": RUN_MANIFEST_VERSION,
        "run_id": config.run_id,
        "task_id": "MODEL-005",
        "plan_id": "PLAN-SOC-001",
        "status": "completed",
        "started_at_utc": started.isoformat(),
        "ended_at_utc": ended.isoformat(),
        "duration_seconds": (ended - started).total_seconds(),
        "command": list(command),
        "frozen_document_version": "V3.3.3",
        "model_id": MODEL_ID,
        "model_version": MODEL_VERSION,
        "algorithm": "CatBoost",
        "code_version": _git_state(inputs.repository_root),
        "code_bindings": dict(code_bindings),
        "python": sys.version,
        "platform": platform.platform(),
        "dependencies": _dependency_versions(),
        "training_config": {
            "path": config.config_path.as_posix(),
            "sha256": config.config_sha256,
        },
        "feature_schema": {
            "version": FEATURE_SCHEMA_VERSION,
            "sha256": EXPECTED_FEATURE_SCHEMA_SHA256,
        },
        "split": {
            "split_id": SPLIT_ID,
            "sha256": EXPECTED_SPLIT_SHA256,
            "integrity": inputs.split_manifest["integrity"],
        },
        "input_bindings": [
            {
                "dataset_id": dataset_id,
                "canonical_relative_path": binding["canonical_relative_path"],
                "canonical_sha256": binding["canonical_sha256"],
                "canonical_frame_sha256": binding["canonical_frame_sha256"],
                "artifact_manifest_sha256": binding["artifact_manifest_sha256"],
                "mapping_sha256": binding["mapping_sha256"],
                "row_count": binding["row_count"],
                "participant_count": binding["participant_count"],
                "positive_row_count": binding["positive_row_count"],
            }
            for dataset_id, binding in inputs.input_bindings.items()
        ],
        "excluded_sources": list(EXCLUDED_DATASET_IDS),
        "upstream_expert_guards": dict(inputs.upstream_expert_bindings),
        "outer_fold_audits": [
            {
                "outer_fold": audit["outer_fold"],
                "training_participant_count": audit["training_participant_count"],
                "training_participant_sha256": audit["training_participant_sha256"],
                "training_row_count": audit["training_row_count"],
                "test_participant_count": audit["test_participant_count"],
                "test_participant_sha256": audit["test_participant_sha256"],
                "test_row_count": audit["test_row_count"],
                "calibration_method": audit["calibrator"]["method"],
                "selected_features": audit["preprocessor"]["selected_features"],
            }
            for audit in fold_audits
        ],
        "production_training": bundle.audit_dict(),
        "production_candidate": dict(final_candidate),
        "metrics": {
            "overall": metrics["overall"],
            "by_dataset": metrics["by_dataset"],
            "metrics_sha256": metrics_sha256,
        },
        "mask_only_diagnostic": {
            "status": "pass",
            "result": "zero_observed_features_returns_expert_mask_0_and_null_probability",
        },
        "forbidden_input_audit": {
            "status": "pass",
            "result": "only_frozen_social_context_columns_and_masks_loaded",
        },
        "all_mask_fusion_hard_gate": "not_applicable_to_MODEL_005",
        "failures": [],
        "warnings": list(warnings),
        "artifacts": {
            "model_sha256": model_sha256,
            "predictions_sha256": predictions_sha256,
            "metrics_sha256": metrics_sha256,
        },
        "production_package_status": (
            "standalone_social_context_expert_active_full_v3_3_3_package_pending_ART_001"
        ),
        "conclusion": (
            "SocialContextExpert completed with strict nested participant OOF "
            "evaluation and a production aggregate bundle."
        ),
    }


def _dependency_versions() -> dict[str, str]:
    import catboost

    return {
        "catboost": catboost.__version__,
        "joblib": joblib.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyarrow": pyarrow.__version__,
        "scikit_learn": sklearn.__version__,
    }


def _git_state(root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "working_tree_dirty": None}
    return {"commit": commit, "working_tree_dirty": bool(status.strip())}


def _code_bindings(root: Path) -> dict[str, str]:
    relative_paths = (
        "src/elderly_monitoring/modules/mental_health/mood_social/experts/social_context.py",
        "scripts/train_mood_social_social_context_expert_v3_3_3.py",
        "scripts/validate_mood_social_social_context_expert_v3_3_3.py",
        "configs/training/mood_social_social_context_expert_v3_3_3.yaml",
    )
    return {
        relative: _sha256_file(root / relative)
        for relative in relative_paths
        if (root / relative).is_file()
    }


def _write_preprocessing_audit(
    directory: Path,
    bundle: SocialContextExpertBundle,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    _write_json(directory / "preprocessor.json", bundle.preprocessor.to_dict())


def _write_predictions(path: Path, frame: pd.DataFrame) -> None:
    output = frame.copy()
    output.insert(0, "prediction_schema_version", PREDICTIONS_VERSION)
    temporary = path.with_name(f".{path.name}.tmp")
    output.to_parquet(temporary, index=False, engine="pyarrow", compression="zstd")
    os.replace(temporary, path)


def _write_joblib_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    joblib.dump(payload, temporary, compress=3)
    os.replace(temporary, path)


def _write_json(path: Path, payload: Any) -> None:
    _write_bytes(path, _canonical_json_bytes(payload))


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _report_artifact_hashes(stage: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(
        (
            item
            for item in stage.rglob("*")
            if item.is_file() and "_publish" not in item.parts
        ),
        key=lambda item: item.relative_to(stage).as_posix().encode("utf-8"),
    ):
        relative = path.relative_to(stage).as_posix()
        if relative == "artifacts.json":
            continue
        rows.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return {
        "artifact_manifest_version": "mood-social-social-context-report-artifacts-v1",
        "artifact_count": len(rows),
        "artifacts": rows,
    }


def _check_output_targets(
    report_dir: Path,
    model_path: Path,
    manifest_path: Path,
    *,
    overwrite: bool,
) -> None:
    existing = [
        path for path in (report_dir, model_path, manifest_path) if path.exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(existing[0])


def _replace_file(source: Path, destination: Path, *, overwrite: bool) -> None:
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, destination)


def _canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(chunk_size), b""):
                digest.update(chunk)
    except OSError as exc:
        raise SocialContextExpertError(
            "required SocialContext file is unreadable"
        ) from exc
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


__all__ = [
    "CALIBRATOR_VERSION",
    "CategoricalFoldPreprocessor",
    "EXCLUDED_DATASET_IDS",
    "EXPECTED_FEATURE_SCHEMA_SHA256",
    "EXPECTED_SPLIT_SHA256",
    "MISSING_CATEGORY",
    "MODEL_BUNDLE_VERSION",
    "MODEL_ID",
    "MODEL_VERSION",
    "SOCIAL_CONTEXT_CATEGORICAL_COLUMNS",
    "SOCIAL_CONTEXT_DATASET_IDS",
    "SOCIAL_CONTEXT_FEATURE_COLUMNS",
    "SOCIAL_CONTEXT_MASK_COLUMNS",
    "SOCIAL_CONTEXT_NUMERIC_COLUMNS",
    "SOCIAL_DATASET_IDS",
    "SOCIAL_FEATURE_COLUMNS",
    "SOCIAL_MASK_COLUMNS",
    "SocialContextCalibrator",
    "SocialContextCategoricalFoldPreprocessor",
    "SocialContextExpertBundle",
    "SocialContextExpertError",
    "SocialContextFoldPreprocessor",
    "SocialContextProbabilityCalibrator",
    "SocialContextTrainingConfig",
    "SocialContextTrainingInputs",
    "UPSTREAM_EXPERT_GUARDS",
    "binary_metrics",
    "compute_training_sample_weights",
    "deterministic_search_candidates",
    "load_social_context_training_config",
    "load_social_context_training_inputs",
    "participant_set_sha256",
    "sample_weight_audit",
    "train_social_context_expert",
]
