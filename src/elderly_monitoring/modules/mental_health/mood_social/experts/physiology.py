"""Leakage-safe V3.3.3 PhysiologyExpert training and inference artifacts."""

from __future__ import annotations

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
import warnings

import joblib
import numpy as np
import pandas as pd
import pyarrow
import sklearn
from sklearn.exceptions import ConvergenceWarning
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
    PHYSIOLOGY_EXPERT_FEATURE_ORDER,
)
from elderly_monitoring.modules.mental_health.mood_social.splits import (
    DEFAULT_SPLIT_RELATIVE,
    SPLIT_ID,
    build_split_manifest,
)


MODEL_ID = "M-PHY-001"
MODEL_VERSION = "physiology-expert-v3.3.3"
TRAINING_CONFIG_VERSION = "mood-social-physiology-training-config-v1"
MODEL_BUNDLE_VERSION = "mood-social-physiology-expert-bundle-v1"
PREPROCESSOR_VERSION = "mood-social-numeric-fold-preprocessor-v1"
CALIBRATOR_VERSION = "mood-social-probability-calibrator-v1"
METRICS_VERSION = "mood-social-binary-metrics-v1"
PREDICTIONS_VERSION = "mood-social-physiology-oof-predictions-v1"
RUN_MANIFEST_VERSION = "mood-social-physiology-training-run-v1"
MODEL_MANIFEST_VERSION = "mood-social-physiology-model-manifest-v1"
EXPECTED_SPLIT_SHA256 = (
    "e9915dbc590a6ea454c34d26558ac5866c57ee8a8e78f44cdcc79c859df77ee3"
)
EXPECTED_FEATURE_SCHEMA_SHA256 = (
    "d2ed18db72601ccb3b0cbb89907bc60ab336bd00f84f0b2c49dcea568bf339e9"
)
PHYSIOLOGY_DATASET_IDS = ("resilient",)
EXCLUDED_DATASET_IDS = (
    "psyche_d",
    "nhanes",
    "shenzhen_elderly",
    "nhanes_ssq_2005_2008",
)
ALL_FROZEN_DATASET_IDS = (
    "psyche_d",
    "resilient",
    "nhanes",
    "shenzhen_elderly",
    "nhanes_ssq_2005_2008",
)
PHYSIOLOGY_FEATURE_COLUMNS = tuple(
    f"physiology.{name}" for name in PHYSIOLOGY_EXPERT_FEATURE_ORDER
)
PHYSIOLOGY_MASK_COLUMNS = tuple(
    f"feature_mask.{name}" for name in PHYSIOLOGY_FEATURE_COLUMNS
)
OUTER_FOLD_COUNT = 5
INNER_FOLD_COUNT = 5

UPSTREAM_EXPERT_GUARDS: Mapping[str, Mapping[str, str]] = {
    "activity": {
        "model_path": (
            "models/mental_health/mood_social/v3.3.3/activity_expert.joblib"
        ),
        "model_sha256": (
            "906a3cc2b74cdd3beeba446ca32885c7d2f0388be230ab793a2728bfd711bc3a"
        ),
        "manifest_path": (
            "models/mental_health/mood_social/v3.3.3/activity_expert_manifest.json"
        ),
        "manifest_sha256": (
            "eb2521a217083c784ecef1b93deef33aebec5dd4bcb614db5890ed47d5700b1b"
        ),
    },
    "sleep": {
        "model_path": "models/mental_health/mood_social/v3.3.3/sleep_expert.joblib",
        "model_sha256": (
            "966ba9ecd32ac18ac63ed9a6733a2e84f859a657cdb664bc2e034f0857056ade"
        ),
        "manifest_path": (
            "models/mental_health/mood_social/v3.3.3/sleep_expert_manifest.json"
        ),
        "manifest_sha256": (
            "11096d9916e822be23d8c9acdc86bb9c3c2700bf7c0ea2bcad8abf315ce30459"
        ),
    },
    "joint": {
        "model_path": (
            "models/mental_health/mood_social/v3.3.3/activity_sleep_joint_expert.joblib"
        ),
        "model_sha256": (
            "2152e7a62ba6382d64ddb1ca865410fa99e0acf9b73db54ed4c25622c46e70c8"
        ),
        "manifest_path": (
            "models/mental_health/mood_social/v3.3.3/"
            "activity_sleep_joint_expert_manifest.json"
        ),
        "manifest_sha256": (
            "f54149fc0e504d052ec763e56441fdc638e7b00a8c7801ded497e24943755e17"
        ),
    },
}
_UPSTREAM_IDENTITIES = {
    "activity": ("M-ACT-001", "activity-expert-v3.3.3"),
    "sleep": ("M-SLP-001", "sleep-expert-v3.3.3"),
    "joint": ("M-JNT-001", "activity-sleep-joint-expert-v3.3.3"),
}


class PhysiologyExpertError(RuntimeError):
    """Raised when MODEL-004 input, training, or artifact checks fail."""


@dataclass(frozen=True)
class PhysiologyTrainingConfig:
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
    calibration_method: str
    decision_threshold: float
    ece_bin_count: int
    logistic_solver: str
    logistic_max_iter: int
    config_path: Path
    config_sha256: str


@dataclass(frozen=True)
class PhysiologyTrainingInputs:
    repository_root: Path
    split_manifest: Mapping[str, Any]
    split_manifest_sha256: str
    assignments: pd.DataFrame
    frames: Mapping[str, pd.DataFrame]
    input_bindings: Mapping[str, Mapping[str, Any]]
    upstream_expert_bindings: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class PhysiologyNumericFoldPreprocessor:
    """Training-fold-only selection, median imputation, and standardization."""

    input_features: tuple[str, ...]
    selected_features: tuple[str, ...]
    medians: tuple[float, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    observed_counts: tuple[int, ...]
    unique_counts: tuple[int, ...]
    version: str = PREPROCESSOR_VERSION
    imputation: str = "training_fold_median"
    encoding: str = "numeric_identity_no_categorical_inputs"
    standardization: str = "training_fold_mean_std_after_median_imputation"
    feature_selection: str = "training_fold_finite_nonconstant"
    add_missing_indicators: bool = False

    def __post_init__(self) -> None:
        if self.version != PREPROCESSOR_VERSION:
            raise PhysiologyExpertError("unsupported PhysiologyExpert preprocessor")
        if self.input_features != PHYSIOLOGY_FEATURE_COLUMNS:
            raise PhysiologyExpertError("PhysiologyExpert input feature order changed")
        if not self.selected_features:
            raise PhysiologyExpertError("PhysiologyExpert selected no usable features")
        if not set(self.selected_features).issubset(self.input_features):
            raise PhysiologyExpertError("PhysiologyExpert selected an unknown feature")
        selected_count = len(self.selected_features)
        if not (
            selected_count == len(self.medians) == len(self.means) == len(self.scales)
        ):
            raise PhysiologyExpertError(
                "PhysiologyExpert preprocessing parameter count does not match"
            )
        if len(self.input_features) != len(self.observed_counts) or len(
            self.input_features
        ) != len(self.unique_counts):
            raise PhysiologyExpertError(
                "PhysiologyExpert selection audit does not match"
            )
        if any(not math.isfinite(value) for value in self.medians):
            raise PhysiologyExpertError("PhysiologyExpert medians must be finite")
        if any(not math.isfinite(value) for value in self.means):
            raise PhysiologyExpertError("PhysiologyExpert means must be finite")
        if any(not math.isfinite(value) or value <= 0 for value in self.scales):
            raise PhysiologyExpertError("PhysiologyExpert scales must be positive")
        if self.add_missing_indicators:
            raise PhysiologyExpertError(
                "feature masks cannot become learned risk inputs"
            )

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> PhysiologyNumericFoldPreprocessor:
        values, _ = _masked_physiology_values(frame)
        selected: list[str] = []
        medians: list[float] = []
        observed_counts: list[int] = []
        unique_counts: list[int] = []
        for index, feature in enumerate(PHYSIOLOGY_FEATURE_COLUMNS):
            finite = values[:, index][np.isfinite(values[:, index])]
            observed_count = int(len(finite))
            unique_count = int(len(np.unique(finite))) if observed_count else 0
            observed_counts.append(observed_count)
            unique_counts.append(unique_count)
            if observed_count >= 2 and unique_count >= 2:
                selected.append(feature)
                medians.append(float(np.median(finite)))
        if not selected:
            raise PhysiologyExpertError("PhysiologyExpert selected no usable features")
        indices = [PHYSIOLOGY_FEATURE_COLUMNS.index(name) for name in selected]
        filled = values[:, indices].copy()
        for column, median in enumerate(medians):
            missing = ~np.isfinite(filled[:, column])
            filled[missing, column] = median
        means = tuple(float(value) for value in filled.mean(axis=0))
        scales = tuple(float(value) for value in filled.std(axis=0, ddof=0))
        return cls(
            input_features=PHYSIOLOGY_FEATURE_COLUMNS,
            selected_features=tuple(selected),
            medians=tuple(medians),
            means=means,
            scales=scales,
            observed_counts=tuple(observed_counts),
            unique_counts=tuple(unique_counts),
        )

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        values, _ = _masked_physiology_values(frame)
        indices = [self.input_features.index(name) for name in self.selected_features]
        selected = values[:, indices].copy()
        for column, median in enumerate(self.medians):
            missing = ~np.isfinite(selected[:, column])
            selected[missing, column] = median
        selected = (selected - np.asarray(self.means)) / np.asarray(self.scales)
        if not np.isfinite(selected).all():
            raise PhysiologyExpertError(
                "PhysiologyExpert preprocessing left non-finite data"
            )
        return pd.DataFrame(
            selected,
            columns=self.selected_features,
            index=frame.index,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "input_features": list(self.input_features),
            "selected_features": list(self.selected_features),
            "medians": list(self.medians),
            "means": list(self.means),
            "scales": list(self.scales),
            "observed_counts": dict(
                zip(self.input_features, self.observed_counts, strict=True)
            ),
            "unique_counts": dict(
                zip(self.input_features, self.unique_counts, strict=True)
            ),
            "imputation": self.imputation,
            "encoding": self.encoding,
            "standardization": self.standardization,
            "feature_selection": self.feature_selection,
            "add_missing_indicators": self.add_missing_indicators,
        }


@dataclass(frozen=True)
class PhysiologyProbabilityCalibrator:
    method: str
    estimator: LogisticRegression
    positive_participant_count: int
    fit_row_count: int
    version: str = CALIBRATOR_VERSION
    probability_clip_epsilon: float = 1.0e-6

    def __post_init__(self) -> None:
        if self.version != CALIBRATOR_VERSION:
            raise PhysiologyExpertError("unsupported probability calibrator")
        if self.method != "platt":
            raise PhysiologyExpertError("PhysiologyExpert calibration must use Platt")
        if self.positive_participant_count <= 0 or self.fit_row_count <= 0:
            raise PhysiologyExpertError("calibrator training scope is empty")

    @classmethod
    def fit(
        cls,
        raw_probability: Sequence[float],
        target: Sequence[int],
        sample_weight: Sequence[float],
        *,
        positive_participant_count: int,
    ) -> PhysiologyProbabilityCalibrator:
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
            raise PhysiologyExpertError("calibrator inputs are invalid")
        estimator = LogisticRegression(
            C=np.inf,
            solver="lbfgs",
            max_iter=5000,
            random_state=20260728,
        )
        estimator.fit(
            _probability_logit(raw)[:, None],
            y,
            sample_weight=weight,
        )
        return cls(
            method="platt",
            estimator=estimator,
            positive_participant_count=positive_participant_count,
            fit_row_count=len(raw),
        )

    def predict(self, raw_probability: Sequence[float]) -> np.ndarray:
        raw = np.asarray(raw_probability, dtype="float64")
        if raw.ndim != 1 or not np.isfinite(raw).all():
            raise PhysiologyExpertError("calibrator prediction input is invalid")
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
        estimator = self.estimator
        return {
            **common,
            "classes": [int(value) for value in estimator.classes_],
            "coefficient": [float(value) for value in estimator.coef_[0]],
            "intercept": [float(value) for value in estimator.intercept_],
        }


@dataclass(frozen=True)
class PhysiologyExpertBundle:
    """Serialized aggregate model used by later API and fusion tasks."""

    model_id: str
    model_version: str
    bundle_version: str
    feature_schema_version: str
    feature_schema_sha256: str
    split_id: str
    split_manifest_sha256: str
    dataset_ids: tuple[str, ...]
    input_feature_names: tuple[str, ...]
    preprocessor: PhysiologyNumericFoldPreprocessor
    classifier: LogisticRegression
    calibrator: PhysiologyProbabilityCalibrator
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
            raise PhysiologyExpertError("PhysiologyExpert bundle identity changed")
        if self.feature_schema_version != FEATURE_SCHEMA_VERSION:
            raise PhysiologyExpertError("PhysiologyExpert feature schema changed")
        if self.feature_schema_sha256 != EXPECTED_FEATURE_SCHEMA_SHA256:
            raise PhysiologyExpertError("PhysiologyExpert feature schema hash changed")
        if self.split_id != SPLIT_ID:
            raise PhysiologyExpertError("PhysiologyExpert split ID changed")
        if self.split_manifest_sha256 != EXPECTED_SPLIT_SHA256:
            raise PhysiologyExpertError("PhysiologyExpert split hash changed")
        if self.dataset_ids != PHYSIOLOGY_DATASET_IDS:
            raise PhysiologyExpertError("PhysiologyExpert dataset membership changed")
        if self.input_feature_names != PHYSIOLOGY_FEATURE_COLUMNS:
            raise PhysiologyExpertError("PhysiologyExpert feature order changed")
        if (
            self.classifier.solver != "saga"
            or self.classifier.max_iter != 5000
            or self.classifier.l1_ratio is None
            or not 0 <= self.classifier.l1_ratio <= 1
        ):
            raise PhysiologyExpertError("PhysiologyExpert classifier contract changed")
        if self.training_participant_count <= 0 or self.training_row_count <= 0:
            raise PhysiologyExpertError("PhysiologyExpert training scope is empty")
        if not _is_sha256(self.training_participant_sha256):
            raise PhysiologyExpertError("PhysiologyExpert participant hash is invalid")

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        _, masks = _masked_physiology_values(frame)
        matrix = self.preprocessor.transform(frame)
        raw = np.asarray(
            self.classifier.predict_proba(matrix)[:, 1],
            dtype="float64",
        )
        calibrated = self.calibrator.predict(raw)
        observed = masks.sum(axis=1).astype("int64")
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
            "source_transform": "resilient_canonical_same_semantics_identity",
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
    preprocessor: PhysiologyNumericFoldPreprocessor


def load_physiology_training_config(path: str | Path) -> PhysiologyTrainingConfig:
    config_path = Path(path).resolve()
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise PhysiologyExpertError(
            "PhysiologyExpert training config is unreadable"
        ) from exc
    if not isinstance(payload, dict):
        raise PhysiologyExpertError(
            "PhysiologyExpert training config must be a mapping"
        )
    try:
        search_space_payload = payload["search"]["space"]
        search_space = {
            str(name): tuple(values) for name, values in search_space_payload.items()
        }
        config = PhysiologyTrainingConfig(
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
            calibration_method=str(payload["calibration"]["method"]),
            decision_threshold=float(payload["evaluation"]["decision_threshold"]),
            ece_bin_count=int(payload["evaluation"]["ece_bin_count"]),
            logistic_solver=str(payload["runtime"]["logistic_solver"]),
            logistic_max_iter=int(payload["runtime"]["logistic_max_iter"]),
            config_path=config_path,
            config_sha256=_sha256_file(config_path),
        )
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise PhysiologyExpertError(
            "PhysiologyExpert training config is malformed"
        ) from exc
    _validate_training_config(payload, config)
    return config


def _validate_training_config(
    payload: Mapping[str, Any],
    config: PhysiologyTrainingConfig,
) -> None:
    if payload.get("config_version") != TRAINING_CONFIG_VERSION:
        raise PhysiologyExpertError("PhysiologyExpert config version changed")
    if payload.get("task_id") != "MODEL-004" or payload.get("model_id") != MODEL_ID:
        raise PhysiologyExpertError("PhysiologyExpert config task identity changed")
    if payload.get("model_version") != MODEL_VERSION:
        raise PhysiologyExpertError("PhysiologyExpert config model version changed")
    if (
        config.random_seed != 20260728
        or config.split_id != SPLIT_ID
        or config.split_sha256 != EXPECTED_SPLIT_SHA256
        or config.feature_schema_version != FEATURE_SCHEMA_VERSION
        or config.feature_schema_sha256 != EXPECTED_FEATURE_SCHEMA_SHA256
        or config.dataset_ids != PHYSIOLOGY_DATASET_IDS
    ):
        raise PhysiologyExpertError("PhysiologyExpert frozen input binding changed")
    expected_space = {
        "C": (0.01, 0.1, 1, 10),
        "l1_ratio": (0, 0.25, 0.5, 0.75, 1),
    }
    if dict(config.search_space) != expected_space:
        raise PhysiologyExpertError("PhysiologyExpert ElasticNet search space changed")
    full_count = math.prod(len(values) for values in expected_space.values())
    if (
        config.search_traversal != "deterministic_full_grid"
        or config.search_candidate_count != full_count
        or config.auprc_tie_tolerance < 0
        or config.calibration_method != "platt"
        or not 0 < config.decision_threshold < 1
        or config.ece_bin_count < 2
        or config.logistic_solver != "saga"
        or config.logistic_max_iter != 5000
    ):
        raise PhysiologyExpertError(
            "PhysiologyExpert implementation settings are invalid"
        )
    if tuple(payload.get("data", {}).get("excluded_datasets", ())) != (
        EXCLUDED_DATASET_IDS
    ):
        raise PhysiologyExpertError("PhysiologyExpert excluded-source policy changed")
    preprocessing = payload.get("preprocessing")
    if preprocessing != {
        "source_transform": "resilient_canonical_same_semantics_identity",
        "imputation": "training_fold_median",
        "categorical_encoding": "numeric_identity_no_categorical_inputs",
        "standardization": "training_fold_mean_std_after_median_imputation",
        "feature_selection": "training_fold_finite_nonconstant",
        "add_missing_indicators": False,
        "use_feature_masks_as_risk_inputs": False,
        "use_feature_coverage_as_risk_input": False,
        "use_activity_or_sleep_as_risk_inputs": False,
    }:
        raise PhysiologyExpertError("PhysiologyExpert preprocessing policy changed")
    if payload.get("calibration") != {
        "method": "platt",
        "fit_input": "strict_cross_fitted_raw_probability",
        "fit_weighting": "frozen_training_sample_weight",
        "probability_clip_epsilon": 1.0e-6,
    }:
        raise PhysiologyExpertError("PhysiologyExpert calibration policy changed")


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
        identity = _sha256_bytes(
            f"{random_seed}\0".encode("ascii") + _canonical_json_bytes(params)
        )
        candidates.append(
            {
                "candidate_id": f"elasticnet-{identity[:16]}",
                "selection_digest": identity,
                "params": params,
            }
        )
    candidates.sort(key=lambda item: _canonical_json_bytes(item["params"]))
    if candidate_count != len(candidates):
        raise PhysiologyExpertError("PhysiologyExpert must search the complete grid")
    for candidate in candidates:
        candidate["candidate_id"] = f"elasticnet-{candidate['selection_digest'][:16]}"
    return candidates


def compute_training_sample_weights(frame: pd.DataFrame) -> np.ndarray:
    required = {"dataset_id", "global_participant_id", "binary_target"}
    if not required.issubset(frame.columns) or len(frame) == 0:
        raise PhysiologyExpertError("sample-weight frame is incomplete")
    datasets = frame["dataset_id"].astype(str)
    participants = frame["global_participant_id"].astype(str)
    target = pd.to_numeric(frame["binary_target"], errors="raise").astype("int64")
    if not target.isin([0, 1]).all():
        raise PhysiologyExpertError("sample-weight target must be binary")
    weights = np.zeros(len(frame), dtype="float64")
    for dataset_id in sorted(set(datasets), key=lambda value: value.encode("utf-8")):
        dataset_mask = datasets.eq(dataset_id).to_numpy()
        classes = sorted(set(target[dataset_mask].tolist()))
        class_mass = 1.0 / len(classes)
        for class_value in classes:
            class_mask = dataset_mask & target.eq(class_value).to_numpy()
            units = participants[class_mask]
            unit_counts = units.value_counts(sort=False)
            unit_mass = class_mass / len(unit_counts)
            row_indices = np.flatnonzero(class_mask)
            for row_index in row_indices:
                weights[row_index] = unit_mass / int(
                    unit_counts[participants.iloc[row_index]]
                )
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise PhysiologyExpertError("sample-weight calculation failed")
    weights *= len(weights) / weights.sum()
    return weights


def sample_weight_audit(
    frame: pd.DataFrame,
    sample_weight: Sequence[float],
) -> dict[str, Any]:
    weight = np.asarray(sample_weight, dtype="float64")
    if len(weight) != len(frame):
        raise PhysiologyExpertError("sample-weight audit length does not match")
    rows: dict[str, Any] = {}
    for dataset_id in sorted(set(frame["dataset_id"].astype(str))):
        selected = frame["dataset_id"].astype(str).eq(dataset_id).to_numpy()
        by_class = {
            str(class_value): float(
                weight[
                    selected
                    & pd.to_numeric(frame["binary_target"]).eq(class_value).to_numpy()
                ].sum()
            )
            for class_value in sorted(
                set(pd.to_numeric(frame.loc[selected, "binary_target"]).astype(int))
            )
        }
        rows[dataset_id] = {
            "total": float(weight[selected].sum()),
            "by_class": by_class,
        }
    return {
        "definition": (
            "dataset_equal_then_class_equal_then_participant_class_equal_"
            "then_repeat_window_equal"
        ),
        "normalized_total": float(weight.sum()),
        "by_dataset": rows,
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
        raise PhysiologyExpertError("binary metric inputs are invalid")
    predicted = (score >= decision_threshold).astype("int64")
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    curve, ece = _calibration_curve(y, score, ece_bin_count)
    return {
        "metric_version": METRICS_VERSION,
        "row_count": int(len(y)),
        "positive_row_count": int(y.sum()),
        "negative_row_count": int(len(y) - y.sum()),
        "auprc": (float(average_precision_score(y, score)) if np.any(y == 1) else None),
        "auroc": (float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else None),
        "macro_f1": float(
            f1_score(y, predicted, average="macro", labels=[0, 1], zero_division=0)
        ),
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


def load_physiology_training_inputs(
    repository_root: str | Path,
    *,
    expected_split_sha256: str = EXPECTED_SPLIT_SHA256,
) -> PhysiologyTrainingInputs:
    root = Path(repository_root).resolve()
    split_path = root / DEFAULT_SPLIT_RELATIVE
    if _sha256_file(split_path) != expected_split_sha256:
        raise PhysiologyExpertError("frozen DATA-007 split file hash changed")
    try:
        manifest = json.loads(split_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PhysiologyExpertError("frozen DATA-007 split is unreadable") from exc
    rebuilt = build_split_manifest(repository_root=root)
    if manifest != rebuilt:
        raise PhysiologyExpertError("split, five inputs, or MH-003 schema drifted")
    if (
        manifest.get("split_id") != SPLIT_ID
        or manifest.get("feature_schema_binding", {}).get("snapshot_sha256")
        != EXPECTED_FEATURE_SCHEMA_SHA256
    ):
        raise PhysiologyExpertError("frozen DATA-007 binding changed")
    upstream_bindings = _validate_upstream_expert_guards(root)
    inputs = manifest.get("inputs")
    if not isinstance(inputs, list):
        raise PhysiologyExpertError("frozen DATA-007 inputs are malformed")
    input_bindings = {
        str(item["dataset_id"]): item for item in inputs if isinstance(item, dict)
    }
    if tuple(input_bindings) != ALL_FROZEN_DATASET_IDS:
        raise PhysiologyExpertError("frozen DATA-007 dataset membership changed")
    for dataset_id in ALL_FROZEN_DATASET_IDS:
        binding = input_bindings[dataset_id]
        path = root / str(binding["canonical_relative_path"])
        if _sha256_file(path) != binding["canonical_sha256"]:
            raise PhysiologyExpertError("frozen canonical input hash changed")
    frames: dict[str, pd.DataFrame] = {}
    for dataset_id in PHYSIOLOGY_DATASET_IDS:
        binding = input_bindings[dataset_id]
        path = root / str(binding["canonical_relative_path"])
        try:
            frame = pd.read_parquet(path)
        except (OSError, TypeError, ValueError) as exc:
            raise PhysiologyExpertError(
                "PhysiologyExpert canonical is unreadable"
            ) from exc
        if not isinstance(frame.index, pd.RangeIndex):
            frame = frame.reset_index(drop=True)
        _validate_base_physiology_frame(frame, dataset_id, binding)
        frames[dataset_id] = frame
    assignments = pd.DataFrame(manifest["participant_assignments"])
    assignments = assignments[
        assignments["dataset_id"].astype(str).isin(PHYSIOLOGY_DATASET_IDS)
    ].copy()
    assignments = assignments.sort_values(
        "global_participant_id", kind="stable"
    ).reset_index(drop=True)
    expected_keys = set().union(
        *(set(frame["global_participant_id"].astype(str)) for frame in frames.values())
    )
    if set(assignments["global_participant_id"].astype(str)) != expected_keys:
        raise PhysiologyExpertError(
            "PhysiologyExpert split does not cover its participants"
        )
    return PhysiologyTrainingInputs(
        repository_root=root,
        split_manifest=manifest,
        split_manifest_sha256=expected_split_sha256,
        assignments=assignments,
        frames=frames,
        input_bindings=input_bindings,
        upstream_expert_bindings=upstream_bindings,
    )


def _validate_upstream_expert_guards(root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name in ("activity", "sleep", "joint"):
        guard = UPSTREAM_EXPERT_GUARDS[name]
        model_path = root / guard["model_path"]
        manifest_path = root / guard["manifest_path"]
        if (
            _sha256_file(model_path) != guard["model_sha256"]
            or _sha256_file(manifest_path) != guard["manifest_sha256"]
        ):
            raise PhysiologyExpertError(
                f"upstream {name} expert model or manifest drifted"
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise PhysiologyExpertError(
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
            raise PhysiologyExpertError(
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


def train_physiology_expert(
    repository_root: str | Path,
    config_path: str | Path,
    *,
    report_dir: str | Path | None = None,
    model_path: str | Path | None = None,
    model_manifest_path: str | Path | None = None,
    overwrite: bool = False,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    config = load_physiology_training_config(config_path)
    root = Path(repository_root).resolve()
    inputs = load_physiology_training_inputs(
        root,
        expected_split_sha256=config.split_sha256,
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
        / "physiology_expert.joblib"
    )
    production_manifest = (
        Path(model_manifest_path).resolve()
        if model_manifest_path is not None
        else production_model.with_name("physiology_expert_manifest.json")
    )
    _check_output_targets(
        destination,
        production_model,
        production_manifest,
        overwrite=overwrite,
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
        production_model.parent.mkdir(parents=True, exist_ok=True)
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
    inputs: PhysiologyTrainingInputs,
    config: PhysiologyTrainingConfig,
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
    outer_candidate_oof: list[Mapping[str, pd.DataFrame]] = []
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
            candidates,
            prepared_folds,
            config=config,
            outer_fold=outer_fold,
        )
        candidate_oof = search_result.pop("_candidate_oof_by_id")
        outer_candidate_oof.append(candidate_oof)
        outer_search_results.append(search_result)
        best = search_result["best_candidate"]
        best_oof = candidate_oof[str(best["candidate_id"])]
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
        _write_joblib_atomic(
            stage / "folds" / f"outer_fold_{outer_fold}.joblib",
            fold_bundle,
        )
        _write_json(
            stage / "folds" / f"outer_fold_{outer_fold}.json",
            fold_audit,
        )
        _write_preprocessing_audit(
            stage / "preprocessing" / f"outer_fold_{outer_fold}",
            fold_bundle,
        )
        _write_json(
            stage / "calibration" / f"outer_fold_{outer_fold}.json",
            fold_bundle.calibrator.to_dict(),
        )

    oof = pd.concat(outer_predictions, ignore_index=True)
    oof = oof.sort_values(
        ["dataset_id", "canonical_row_index"],
        kind="stable",
    ).reset_index(drop=True)
    _validate_oof_predictions(oof, inputs, assignments)
    metrics = _build_oof_metrics(oof, config)
    final_candidate = _select_final_candidate(
        outer_search_results,
        outer_candidate_oof,
        config,
    )
    raw_oof = pd.concat(raw_production_oof, ignore_index=True)
    raw_oof = raw_oof.sort_values(
        ["dataset_id", "canonical_row_index"],
        kind="stable",
    ).reset_index(drop=True)
    final_bundle, final_audit = _fit_final_bundle(
        inputs,
        raw_oof=raw_oof,
        hyperparameters=final_candidate["params"],
        config=config,
    )
    _assert_bundle_mask_diagnostic(final_bundle)
    _write_joblib_atomic(staged_model, final_bundle)
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
    feature_importance = _feature_importance(final_bundle)
    _write_json(stage / "explanations" / "feature_importance.json", feature_importance)
    warnings = _training_warnings(
        final_bundle,
        inputs,
        metrics,
        outer_search_results,
        fold_audits,
    )
    _write_json(
        stage / "failures" / "warnings.json",
        {"failures": [], "warnings": warnings},
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
        "algorithm": "ElasticNet Logistic",
        "model_file": production_model_relative_name(staged_model),
        "model_sha256": model_sha256,
        "training_config_sha256": config.config_sha256,
        "split_id": SPLIT_ID,
        "split_manifest_sha256": EXPECTED_SPLIT_SHA256,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_schema_sha256": EXPECTED_FEATURE_SCHEMA_SHA256,
        "dataset_ids": list(PHYSIOLOGY_DATASET_IDS),
        "excluded_dataset_ids": list(EXCLUDED_DATASET_IDS),
        "input_feature_names": list(PHYSIOLOGY_FEATURE_COLUMNS),
        "selected_feature_names": list(final_bundle.preprocessor.selected_features),
        "hyperparameters": dict(final_bundle.hyperparameters),
        "calibrator": final_bundle.calibrator.to_dict(),
        "source_transform": "resilient_canonical_same_semantics_identity",
        "row_eligibility": "at_least_one_real_physiology_model_input",
        "upstream_expert_guards": {
            name: dict(binding)
            for name, binding in inputs.upstream_expert_bindings.items()
        },
        "training_scope": final_bundle.training_scope,
        "training_participant_count": final_bundle.training_participant_count,
        "training_participant_sha256": (final_bundle.training_participant_sha256),
        "training_row_count": final_bundle.training_row_count,
        "metrics_sha256": metrics_sha256,
        "predictions_sha256": predictions_sha256,
        "code_bindings": code_bindings,
        "dependencies": _dependency_versions(),
    }
    _write_json(staged_manifest, model_manifest)
    _write_json(
        stage / "model_artifact.json",
        {
            **model_manifest,
            "model_file": (
                "models/mental_health/mood_social/v3.3.3/physiology_expert.joblib"
            ),
            "model_manifest_file": (
                "models/mental_health/mood_social/v3.3.3/physiology_expert_manifest.json"
            ),
        },
    )
    artifact_hashes = _report_artifact_hashes(stage)
    _write_json(stage / "artifacts.json", artifact_hashes)
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
    inputs: PhysiologyTrainingInputs,
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
        raise PhysiologyExpertError("inner fold participant relation is invalid")
    train_table = _transform_partition(inputs, train_keys)
    validation_table = _transform_partition(inputs, validation_keys)
    train_table = _available_physiology_rows(train_table)
    validation_table = _available_physiology_rows(validation_table)
    preprocessor = PhysiologyNumericFoldPreprocessor.fit(train_table)
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
    config: PhysiologyTrainingConfig,
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
            )
            probability = classifier.predict_proba(prepared.validation_matrix)[:, 1]
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
                    "classifier_convergence": _classifier_convergence(classifier),
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
            part["inner_fold"] = prepared.inner_fold
            oof_parts.append(part)
        pooled_oof = pd.concat(oof_parts, ignore_index=True).sort_values(
            ["dataset_id", "canonical_row_index"], kind="stable"
        )
        pooled_metrics = binary_metrics(
            pooled_oof["binary_target"],
            pooled_oof["raw_probability"],
            decision_threshold=config.decision_threshold,
            ece_bin_count=config.ece_bin_count,
        )
        if pooled_metrics["auprc"] is None:
            raise PhysiologyExpertError("pooled inner OOF has no positive rows")
        result = {
            "candidate_id": candidate["candidate_id"],
            "params": dict(candidate["params"]),
            "pooled_inner_auprc": pooled_metrics["auprc"],
            "pooled_inner_brier_score": pooled_metrics["brier_score"],
            "pooled_inner_row_count": pooled_metrics["row_count"],
            "pooled_inner_positive_row_count": pooled_metrics["positive_row_count"],
            "inner_folds": fold_metrics,
        }
        results.append(result)
        predictions_by_candidate[str(candidate["candidate_id"])] = pooled_oof
    ranked = sorted(
        results,
        key=lambda item: (
            -_round_for_tie(
                float(item["pooled_inner_auprc"]),
                config.auprc_tie_tolerance,
            ),
            float(item["pooled_inner_brier_score"]),
            str(item["candidate_id"]),
        ),
    )
    best = ranked[0]
    return {
        "outer_fold": outer_fold,
        "selection_rule": (
            "highest_pooled_inner_oof_auprc_then_lowest_pooled_inner_oof_"
            "brier_then_candidate_id"
        ),
        "best_candidate": best,
        "candidates": results,
        "_candidate_oof_by_id": predictions_by_candidate,
    }


def _fit_outer_fold(
    inputs: PhysiologyTrainingInputs,
    *,
    train_keys: set[str],
    test_keys: set[str],
    best_oof: pd.DataFrame,
    hyperparameters: Mapping[str, int | float],
    outer_fold: int,
    config: PhysiologyTrainingConfig,
) -> tuple[PhysiologyExpertBundle, pd.DataFrame, dict[str, Any]]:
    train_partition = _transform_partition(inputs, train_keys)
    test_table = _transform_partition(inputs, test_keys)
    train_table = _available_physiology_rows(train_partition)
    preprocessor = PhysiologyNumericFoldPreprocessor.fit(train_table)
    train_weight = compute_training_sample_weights(train_table)
    classifier = _fit_classifier(
        preprocessor.transform(train_table),
        train_table["binary_target"].to_numpy(dtype="int64"),
        train_weight,
        hyperparameters,
        config,
    )
    best_oof = best_oof.sort_values(
        ["dataset_id", "canonical_row_index"], kind="stable"
    ).reset_index(drop=True)
    calibration_weight = compute_training_sample_weights(best_oof)
    positive_participants = int(
        best_oof.loc[best_oof["binary_target"].eq(1), "global_participant_id"].nunique()
    )
    calibrator = PhysiologyProbabilityCalibrator.fit(
        best_oof["raw_probability"],
        best_oof["binary_target"],
        calibration_weight,
        positive_participant_count=positive_participants,
    )
    bundle = PhysiologyExpertBundle(
        model_id=MODEL_ID,
        model_version=MODEL_VERSION,
        bundle_version=MODEL_BUNDLE_VERSION,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        feature_schema_sha256=EXPECTED_FEATURE_SCHEMA_SHA256,
        split_id=SPLIT_ID,
        split_manifest_sha256=EXPECTED_SPLIT_SHA256,
        dataset_ids=PHYSIOLOGY_DATASET_IDS,
        input_feature_names=PHYSIOLOGY_FEATURE_COLUMNS,
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
        "classifier_state_sha256": _classifier_state_sha256(classifier),
        "classifier_convergence": _classifier_convergence(classifier),
    }
    return bundle, output, audit


def _fit_final_bundle(
    inputs: PhysiologyTrainingInputs,
    *,
    raw_oof: pd.DataFrame,
    hyperparameters: Mapping[str, int | float],
    config: PhysiologyTrainingConfig,
) -> tuple[PhysiologyExpertBundle, dict[str, Any]]:
    assignments = _assignment_lookup(inputs.assignments)
    all_keys = set(assignments)
    if (
        len(raw_oof) != sum(len(frame) for frame in inputs.frames.values())
        or raw_oof["prediction_id"].duplicated().any()
        or set(raw_oof["global_participant_id"].astype(str)) != all_keys
    ):
        raise PhysiologyExpertError("production calibrator OOF coverage is invalid")
    partition = _transform_partition(inputs, all_keys)
    table = _available_physiology_rows(partition)
    preprocessor = PhysiologyNumericFoldPreprocessor.fit(table)
    weight = compute_training_sample_weights(table)
    classifier = _fit_classifier(
        preprocessor.transform(table),
        table["binary_target"].to_numpy(dtype="int64"),
        weight,
        hyperparameters,
        config,
    )
    calibration_table = raw_oof[raw_oof["expert_mask"].eq(1)].reset_index(drop=True)
    calibration_weight = compute_training_sample_weights(calibration_table)
    positive_participants = int(
        calibration_table.loc[
            calibration_table["binary_target"].eq(1), "global_participant_id"
        ].nunique()
    )
    calibrator = PhysiologyProbabilityCalibrator.fit(
        calibration_table["raw_probability"],
        calibration_table["binary_target"],
        calibration_weight,
        positive_participant_count=positive_participants,
    )
    bundle = PhysiologyExpertBundle(
        model_id=MODEL_ID,
        model_version=MODEL_VERSION,
        bundle_version=MODEL_BUNDLE_VERSION,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        feature_schema_sha256=EXPECTED_FEATURE_SCHEMA_SHA256,
        split_id=SPLIT_ID,
        split_manifest_sha256=EXPECTED_SPLIT_SHA256,
        dataset_ids=PHYSIOLOGY_DATASET_IDS,
        input_feature_names=PHYSIOLOGY_FEATURE_COLUMNS,
        preprocessor=preprocessor,
        classifier=classifier,
        calibrator=calibrator,
        hyperparameters=dict(hyperparameters),
        training_scope="resilient_real_physiology_rows_for_production",
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
        "classifier_state_sha256": _classifier_state_sha256(classifier),
        "classifier_convergence": _classifier_convergence(classifier),
    }


def _transform_partition(
    inputs: PhysiologyTrainingInputs,
    participant_keys: set[str],
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for dataset_order, dataset_id in enumerate(PHYSIOLOGY_DATASET_IDS):
        base = inputs.frames[dataset_id]
        selected = base[
            base["global_participant_id"].astype(str).isin(participant_keys)
        ]
        if selected.empty:
            raise PhysiologyExpertError(
                "training partition omitted a physiology source"
            )
        part = selected[
            [
                "dataset_id",
                "global_participant_id",
                "binary_target",
                *PHYSIOLOGY_FEATURE_COLUMNS,
                *PHYSIOLOGY_MASK_COLUMNS,
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
    result = pd.concat(parts, ignore_index=True)
    result = result.sort_values(
        ["_dataset_order", "canonical_row_index"],
        kind="stable",
    ).drop(columns="_dataset_order")
    result = result.reset_index(drop=True)
    _masked_physiology_values(result)
    return result


def _available_physiology_rows(frame: pd.DataFrame) -> pd.DataFrame:
    _, masks = _masked_physiology_values(frame)
    available = masks.sum(axis=1) > 0
    result = frame.loc[available].reset_index(drop=True)
    if result.empty:
        raise PhysiologyExpertError(
            "PhysiologyExpert fold has no available feature rows"
        )
    return result


def _fit_classifier(
    matrix: pd.DataFrame,
    target: np.ndarray,
    sample_weight: np.ndarray,
    hyperparameters: Mapping[str, int | float],
    config: PhysiologyTrainingConfig,
) -> LogisticRegression:
    if set(hyperparameters) != {"C", "l1_ratio"}:
        raise PhysiologyExpertError("ElasticNet hyperparameters are invalid")
    classifier = LogisticRegression(
        solver=config.logistic_solver,
        max_iter=config.logistic_max_iter,
        random_state=config.random_seed,
        **dict(hyperparameters),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        classifier.fit(matrix, target, sample_weight=sample_weight)
    return classifier


def _select_final_candidate(
    outer_search_results: Sequence[Mapping[str, Any]],
    outer_candidate_oof: Sequence[Mapping[str, pd.DataFrame]],
    config: PhysiologyTrainingConfig,
) -> dict[str, Any]:
    if (
        len(outer_search_results) != OUTER_FOLD_COUNT
        or len(outer_candidate_oof) != OUTER_FOLD_COUNT
    ):
        raise PhysiologyExpertError("production candidate lacks outer search coverage")
    candidate_ids = [
        str(candidate["candidate_id"])
        for candidate in outer_search_results[0]["candidates"]
    ]
    aggregated: list[dict[str, Any]] = []
    for candidate_id in candidate_ids:
        rows = [
            next(
                candidate
                for candidate in outer["candidates"]
                if str(candidate["candidate_id"]) == candidate_id
            )
            for outer in outer_search_results
        ]
        if any(candidate_id not in values for values in outer_candidate_oof):
            raise PhysiologyExpertError(
                "production candidate lacks outer search coverage"
            )
        pooled = pd.concat(
            [values[candidate_id] for values in outer_candidate_oof],
            ignore_index=True,
        )
        metric = binary_metrics(
            pooled["binary_target"],
            pooled["raw_probability"],
            decision_threshold=config.decision_threshold,
            ece_bin_count=config.ece_bin_count,
        )
        if metric["auprc"] is None:
            raise PhysiologyExpertError("nested inner OOF has no positive rows")
        aggregated.append(
            {
                "candidate_id": candidate_id,
                "params": dict(rows[0]["params"]),
                "pooled_nested_inner_auprc": metric["auprc"],
                "pooled_nested_inner_brier_score": metric["brier_score"],
                "nested_inner_occurrence_count": metric["row_count"],
                "nested_inner_positive_occurrence_count": metric["positive_row_count"],
            }
        )
    ranked = sorted(
        aggregated,
        key=lambda item: (
            -_round_for_tie(
                item["pooled_nested_inner_auprc"],
                config.auprc_tie_tolerance,
            ),
            item["pooled_nested_inner_brier_score"],
            item["candidate_id"],
        ),
    )
    return {
        **ranked[0],
        "selection_rule": (
            "highest_pooled_nested_inner_oof_auprc_then_lowest_pooled_nested_"
            "inner_oof_brier_then_candidate_id_without_outer_test_metrics"
        ),
        "all_candidates": aggregated,
    }


def _build_oof_metrics(
    oof: pd.DataFrame,
    config: PhysiologyTrainingConfig,
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
            for dataset_id in PHYSIOLOGY_DATASET_IDS
        },
        "by_outer_fold": {
            str(outer_fold): pair(oof[oof["outer_fold"].eq(outer_fold)])
            for outer_fold in range(OUTER_FOLD_COUNT)
        },
    }


def _validate_oof_predictions(
    oof: pd.DataFrame,
    inputs: PhysiologyTrainingInputs,
    assignments: Mapping[str, Mapping[str, Any]],
) -> None:
    expected_rows = sum(len(frame) for frame in inputs.frames.values())
    expected_positive = sum(
        int(pd.to_numeric(frame["binary_target"]).sum())
        for frame in inputs.frames.values()
    )
    if (
        len(oof) != expected_rows
        or oof["prediction_id"].duplicated().any()
        or int(oof["binary_target"].sum()) != expected_positive
        or set(oof["dataset_id"].astype(str)) != set(PHYSIOLOGY_DATASET_IDS)
    ):
        raise PhysiologyExpertError("outer OOF prediction coverage is invalid")
    available = oof["expert_mask"].eq(1)
    unavailable = ~available
    if (
        not np.isfinite(oof.loc[available, "raw_probability"]).all()
        or not np.isfinite(oof.loc[available, "calibrated_probability"]).all()
        or not oof.loc[unavailable, "raw_probability"].isna().all()
        or not oof.loc[unavailable, "calibrated_probability"].isna().all()
    ):
        raise PhysiologyExpertError("outer OOF probability-mask relation is invalid")
    expected_outer = (
        oof["global_participant_id"]
        .astype(str)
        .map({key: value["outer_fold"] for key, value in assignments.items()})
    )
    if not expected_outer.astype("int64").equals(oof["outer_fold"].astype("int64")):
        raise PhysiologyExpertError("outer OOF prediction fold assignment drifted")


def _validate_base_physiology_frame(
    frame: pd.DataFrame,
    dataset_id: str,
    binding: Mapping[str, Any],
) -> None:
    required = {
        "dataset_id",
        "global_participant_id",
        "binary_target",
        *PHYSIOLOGY_FEATURE_COLUMNS,
        *PHYSIOLOGY_MASK_COLUMNS,
    }
    if not required.issubset(frame.columns):
        raise PhysiologyExpertError("PhysiologyExpert canonical columns changed")
    if (
        set(frame["dataset_id"].astype(str)) != {dataset_id}
        or len(frame) != int(binding["row_count"])
        or int(pd.to_numeric(frame["binary_target"]).sum())
        != int(binding["positive_row_count"])
        or frame["global_participant_id"].astype(str).nunique()
        != int(binding["participant_count"])
    ):
        raise PhysiologyExpertError("PhysiologyExpert canonical summary changed")
    _, masks = _masked_physiology_values(frame)
    if not np.any(masks):
        raise PhysiologyExpertError(
            "PhysiologyExpert source has no supported physiology evidence"
        )


def _masked_physiology_values(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    missing_columns = set(PHYSIOLOGY_FEATURE_COLUMNS + PHYSIOLOGY_MASK_COLUMNS) - set(
        frame.columns
    )
    if missing_columns:
        raise PhysiologyExpertError(
            "PhysiologyExpert unified feature columns are missing"
        )
    values = np.empty((len(frame), len(PHYSIOLOGY_FEATURE_COLUMNS)), dtype="float64")
    masks = np.empty_like(values, dtype="int8")
    for index, (feature, mask_name) in enumerate(
        zip(PHYSIOLOGY_FEATURE_COLUMNS, PHYSIOLOGY_MASK_COLUMNS, strict=True)
    ):
        try:
            value = pd.to_numeric(frame[feature], errors="raise").to_numpy(
                dtype="float64", na_value=np.nan
            )
            mask_float = pd.to_numeric(frame[mask_name], errors="raise").to_numpy(
                dtype="float64", na_value=np.nan
            )
        except (TypeError, ValueError) as exc:
            raise PhysiologyExpertError(
                "PhysiologyExpert feature values are invalid"
            ) from exc
        if (
            not np.isfinite(mask_float).all()
            or not np.isin(mask_float, [0.0, 1.0]).all()
        ):
            raise PhysiologyExpertError("PhysiologyExpert feature masks are invalid")
        mask = mask_float.astype("int8")
        finite = np.isfinite(value)
        if np.any((mask == 1) & ~finite) or np.any((mask == 0) & finite):
            raise PhysiologyExpertError(
                "PhysiologyExpert value-mask relation is invalid"
            )
        values[:, index] = value
        masks[:, index] = mask
    return values, masks


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
        raise PhysiologyExpertError(
            "participant hash input must be unique and non-empty"
        )
    for participant_id in sorted(values, key=lambda value: value.encode("utf-8")):
        if (
            not isinstance(participant_id, str)
            or not participant_id
            or "\0" in participant_id
        ):
            raise PhysiologyExpertError("participant hash input is invalid")
        digest.update(participant_id.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _round_for_tie(value: float, tolerance: float) -> float:
    if tolerance <= 0:
        return value
    return round(value / tolerance) * tolerance


def _feature_importance(bundle: PhysiologyExpertBundle) -> dict[str, Any]:
    coefficients = bundle.classifier.coef_[0]
    rows = [
        {
            "feature": feature,
            "standardized_coefficient": float(coefficient),
            "absolute_standardized_coefficient": float(abs(coefficient)),
        }
        for feature, coefficient in zip(
            bundle.preprocessor.selected_features,
            coefficients,
            strict=True,
        )
    ]
    rows.sort(
        key=lambda item: (-item["absolute_standardized_coefficient"], item["feature"])
    )
    return {
        "method": "elasticnet_standardized_coefficient",
        "production_attribution_method": "signed_standardized_coefficient",
        "features": rows,
    }


def _classifier_state_sha256(classifier: LogisticRegression) -> str:
    return _sha256_bytes(
        _canonical_json_bytes(
            {
                "classes": [int(value) for value in classifier.classes_],
                "coefficient": [float(value) for value in classifier.coef_[0]],
                "intercept": [float(value) for value in classifier.intercept_],
                "n_iter": [int(value) for value in classifier.n_iter_],
            }
        )
    )


def _classifier_convergence(classifier: LogisticRegression) -> dict[str, Any]:
    iteration_count = int(np.max(classifier.n_iter_))
    return {
        "iteration_count": iteration_count,
        "max_iter": int(classifier.max_iter),
        "converged_before_max_iter": iteration_count < int(classifier.max_iter),
    }


def _assert_bundle_mask_diagnostic(bundle: PhysiologyExpertBundle) -> None:
    frame = pd.DataFrame(
        {
            **{name: [np.nan] for name in PHYSIOLOGY_FEATURE_COLUMNS},
            **{name: [0] for name in PHYSIOLOGY_MASK_COLUMNS},
        }
    )
    predicted = bundle.predict(frame)
    if (
        int(predicted.loc[0, "expert_mask"]) != 0
        or not pd.isna(predicted.loc[0, "raw_probability"])
        or not pd.isna(predicted.loc[0, "calibrated_probability"])
    ):
        raise PhysiologyExpertError(
            "PhysiologyExpert all-missing mask diagnostic failed"
        )


def _training_warnings(
    bundle: PhysiologyExpertBundle,
    inputs: PhysiologyTrainingInputs,
    metrics: Mapping[str, Any],
    outer_search_results: Sequence[Mapping[str, Any]],
    fold_audits: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    unselected = sorted(
        set(PHYSIOLOGY_FEATURE_COLUMNS) - set(bundle.preprocessor.selected_features)
    )
    empty_positive_folds = [
        (int(outer["outer_fold"]), int(fold["inner_fold"]))
        for outer in outer_search_results
        for fold in outer["candidates"][0]["inner_folds"]
        if int(fold["positive_row_count"]) == 0
    ]
    resilient = inputs.frames["resilient"]
    nonconverged_inner_count = sum(
        not bool(fold["classifier_convergence"]["converged_before_max_iter"])
        for outer in outer_search_results
        for candidate in outer["candidates"]
        for fold in candidate["inner_folds"]
    )
    nonconverged_outer_count = sum(
        not bool(audit["classifier_convergence"]["converged_before_max_iter"])
        for audit in fold_audits
    )
    production_converged = _classifier_convergence(bundle.classifier)[
        "converged_before_max_iter"
    ]
    warnings = [
        {
            "code": "PROXY_TRANSFER",
            "message": (
                "RESILIENT wearable-derived physiology fields are "
                "deployment-compatible semantic proxies, not direct validation "
                "of the project physiology device."
            ),
        },
        {
            "code": "SMALL_RESILIENT_SOURCE",
            "message": (
                f"RESILIENT contributes {len(resilient)} participants and "
                f"{int(pd.to_numeric(resilient['binary_target']).sum())} positive "
                f"rows; {metrics['availability']['available_row_count']} rows and "
                f"{metrics['overall']['raw']['positive_row_count']} positive rows "
                "have real physiology evidence."
            ),
        },
        {
            "code": "STRUCTURAL_FEATURE_ABSENCE",
            "message": (
                "Training-fold finite/nonconstant selection excluded: "
                + ", ".join(unselected)
            ),
        },
        {
            "code": "NO_PERFORMANCE_GATE",
            "message": (
                "Metrics are reported as observed; MODEL-004 has no minimum "
                "performance promotion threshold."
            ),
        },
    ]
    if empty_positive_folds:
        warnings.insert(
            2,
            {
                "code": "INNER_VALIDATION_NO_POSITIVE",
                "message": (
                    "Per-fold AUPRC is null for inner validation folds without a "
                    "positive row: "
                    + ", ".join(
                        f"outer {outer}/inner {inner}"
                        for outer, inner in empty_positive_folds
                    )
                    + ". Candidate ranking uses complete pooled strict inner OOF."
                ),
            },
        )
    if nonconverged_inner_count or nonconverged_outer_count or not production_converged:
        warnings.insert(
            3,
            {
                "code": "ELASTICNET_MAX_ITER_REACHED",
                "message": (
                    f"{nonconverged_inner_count} inner candidate fits, "
                    f"{nonconverged_outer_count} selected outer fits, and "
                    f"{0 if production_converged else 1} production fit reached "
                    "the frozen 5000-iteration limit; observed estimates and "
                    "convergence audits are retained without a performance gate."
                ),
            },
        )
    return warnings


def _run_manifest(
    inputs: PhysiologyTrainingInputs,
    config: PhysiologyTrainingConfig,
    metrics: Mapping[str, Any],
    bundle: PhysiologyExpertBundle,
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
    git = _git_state(inputs.repository_root)
    return {
        "run_manifest_version": RUN_MANIFEST_VERSION,
        "run_id": config.run_id,
        "task_id": "MODEL-004",
        "plan_id": "PLAN-PHY-001",
        "status": "completed",
        "started_at_utc": started.isoformat(),
        "ended_at_utc": ended.isoformat(),
        "duration_seconds": (ended - started).total_seconds(),
        "command": list(command),
        "frozen_document_version": "V3.3.3",
        "model_id": MODEL_ID,
        "model_version": MODEL_VERSION,
        "algorithm": "ElasticNet Logistic",
        "code_version": git,
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
                "training_included": dataset_id in PHYSIOLOGY_DATASET_IDS,
            }
            for dataset_id, binding in inputs.input_bindings.items()
        ],
        "excluded_sources": list(EXCLUDED_DATASET_IDS),
        "excluded_source_policy": (
            "hard_excluded_no_all_missing_imputation_rows_enter_training"
        ),
        "upstream_expert_guards": {
            name: dict(binding)
            for name, binding in inputs.upstream_expert_bindings.items()
        },
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
                "source_transform": audit["source_transform"],
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
        "all_mask_fusion_hard_gate": "not_applicable_to_MODEL_004",
        "failures": [],
        "warnings": list(warnings),
        "artifacts": {
            "model_sha256": model_sha256,
            "predictions_sha256": predictions_sha256,
            "metrics_sha256": metrics_sha256,
        },
        "production_package_status": (
            "standalone_physiology_expert_active_full_v3_3_3_package_pending_ART_001"
        ),
        "conclusion": (
            "PhysiologyExpert completed with strict nested participant OOF "
            "evaluation and a production aggregate bundle."
        ),
    }


def _dependency_versions() -> dict[str, str]:
    return {
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
        "src/elderly_monitoring/modules/mental_health/mood_social/experts/physiology.py",
        "scripts/train_mood_social_physiology_expert_v3_3_3.py",
        "scripts/validate_mood_social_physiology_expert_v3_3_3.py",
        "configs/training/mood_social_physiology_expert_v3_3_3.yaml",
    )
    return {
        relative: _sha256_file(root / relative)
        for relative in relative_paths
        if (root / relative).is_file()
    }


def _write_preprocessing_audit(
    directory: Path,
    bundle: PhysiologyExpertBundle,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    _write_json(directory / "preprocessor.json", bundle.preprocessor.to_dict())


def _write_predictions(path: Path, frame: pd.DataFrame) -> None:
    output = frame.copy()
    output.insert(0, "prediction_schema_version", PREDICTIONS_VERSION)
    temporary = path.with_name(f".{path.name}.tmp")
    output.to_parquet(
        temporary,
        index=False,
        engine="pyarrow",
        compression="zstd",
    )
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
        "artifact_manifest_version": "mood-social-physiology-report-artifacts-v1",
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
        raise PhysiologyExpertError(
            "required PhysiologyExpert file is unreadable"
        ) from exc
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def production_model_relative_name(path: Path) -> str:
    return path.name


__all__ = [
    "PHYSIOLOGY_DATASET_IDS",
    "PHYSIOLOGY_FEATURE_COLUMNS",
    "PHYSIOLOGY_MASK_COLUMNS",
    "PhysiologyExpertBundle",
    "PhysiologyExpertError",
    "PhysiologyTrainingConfig",
    "PhysiologyTrainingInputs",
    "EXPECTED_FEATURE_SCHEMA_SHA256",
    "EXPECTED_SPLIT_SHA256",
    "MODEL_BUNDLE_VERSION",
    "MODEL_ID",
    "MODEL_VERSION",
    "PhysiologyNumericFoldPreprocessor",
    "PhysiologyProbabilityCalibrator",
    "binary_metrics",
    "compute_training_sample_weights",
    "deterministic_search_candidates",
    "load_physiology_training_config",
    "load_physiology_training_inputs",
    "participant_set_sha256",
    "sample_weight_audit",
    "train_physiology_expert",
]
