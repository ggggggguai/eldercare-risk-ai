"""Leakage-safe V3.3.3 ActivitySleepJointExpert training artifacts."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
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
from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
import pyarrow
import sklearn
import yaml

from elderly_monitoring.datasets.adapters import nhanes, psyche_d, resilient
from elderly_monitoring.modules.mental_health.mood_social.experts.activity import (
    ACTIVITY_FEATURE_COLUMNS,
    ACTIVITY_MASK_COLUMNS,
    ProbabilityCalibrator,
    binary_metrics,
    compute_training_sample_weights,
    deterministic_search_candidates,
    participant_set_sha256,
    sample_weight_audit,
)
from elderly_monitoring.modules.mental_health.mood_social.experts.sleep import (
    SLEEP_FEATURE_COLUMNS,
    SLEEP_MASK_COLUMNS,
)
from elderly_monitoring.modules.mental_health.mood_social.feature_schema import (
    FEATURE_SCHEMA_VERSION,
)
from elderly_monitoring.modules.mental_health.mood_social.splits import (
    DEFAULT_SPLIT_RELATIVE,
    SPLIT_ID,
    build_split_manifest,
)


MODEL_ID = "M-JNT-001"
MODEL_VERSION = "activity-sleep-joint-expert-v3.3.3"
TRAINING_CONFIG_VERSION = "mood-social-activity-sleep-joint-training-config-v1"
MODEL_BUNDLE_VERSION = "mood-social-activity-sleep-joint-expert-bundle-v1"
PREPROCESSOR_VERSION = "mood-social-joint-numeric-fold-preprocessor-v1"
METRICS_VERSION = "mood-social-binary-metrics-v1"
PREDICTIONS_VERSION = "mood-social-activity-sleep-joint-oof-predictions-v1"
RUN_MANIFEST_VERSION = "mood-social-activity-sleep-joint-training-run-v1"
MODEL_MANIFEST_VERSION = "mood-social-activity-sleep-joint-model-manifest-v1"
EXPECTED_SPLIT_SHA256 = (
    "e9915dbc590a6ea454c34d26558ac5866c57ee8a8e78f44cdcc79c859df77ee3"
)
EXPECTED_FEATURE_SCHEMA_SHA256 = (
    "d2ed18db72601ccb3b0cbb89907bc60ab336bd00f84f0b2c49dcea568bf339e9"
)
JOINT_DATASET_IDS = ("psyche_d", "resilient", "nhanes")
EXCLUDED_DATASET_IDS = ("shenzhen_elderly", "nhanes_ssq_2005_2008")
JOINT_FEATURE_COLUMNS = ACTIVITY_FEATURE_COLUMNS + SLEEP_FEATURE_COLUMNS
JOINT_MASK_COLUMNS = ACTIVITY_MASK_COLUMNS + SLEEP_MASK_COLUMNS
JOINT_ROW_ELIGIBILITY = "activity_and_sleep_each_have_at_least_one_real_model_input"
JOINT_SOURCE_TRANSFORM = (
    "activity_right_continuous_training_fold_ecdf_plus_"
    "sleep_canonical_same_semantics_identity"
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
}
_UPSTREAM_IDENTITIES = {
    "activity": ("M-ACT-001", "activity-expert-v3.3.3"),
    "sleep": ("M-SLP-001", "sleep-expert-v3.3.3"),
}
_ECDF_CLASS_BY_DATASET = {
    "psyche_d": psyche_d.TrainingFoldECDF,
    "resilient": resilient.TrainingFoldECDF,
    "nhanes": nhanes.TrainingFoldECDF,
}


class ActivitySleepJointExpertError(RuntimeError):
    """Raised when MODEL-003 input, training, or artifact checks fail."""


JointProbabilityCalibrator = ProbabilityCalibrator


@dataclass(frozen=True)
class JointTrainingConfig:
    run_id: str
    random_seed: int
    split_id: str
    split_sha256: str
    feature_schema_version: str
    feature_schema_sha256: str
    dataset_ids: tuple[str, ...]
    upstream_expert_guards: Mapping[str, Mapping[str, str]]
    search_space: Mapping[str, tuple[int | float, ...]]
    search_candidate_count: int
    search_traversal: str
    auprc_tie_tolerance: float
    calibration_positive_participant_threshold: int
    decision_threshold: float
    ece_bin_count: int
    lightgbm_n_jobs: int
    config_path: Path
    config_sha256: str


@dataclass(frozen=True)
class JointTrainingInputs:
    repository_root: Path
    split_manifest: Mapping[str, Any]
    split_manifest_sha256: str
    assignments: pd.DataFrame
    frames: Mapping[str, pd.DataFrame]
    input_bindings: Mapping[str, Mapping[str, Any]]
    upstream_expert_bindings: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class JointNumericFoldPreprocessor:
    """Select and impute joint numeric inputs using one training fold only."""

    input_features: tuple[str, ...]
    selected_features: tuple[str, ...]
    medians: tuple[float, ...]
    observed_counts: tuple[int, ...]
    unique_counts: tuple[int, ...]
    version: str = PREPROCESSOR_VERSION
    imputation: str = "training_fold_median"
    encoding: str = "numeric_identity_no_categorical_inputs"
    standardization: str = "none_for_lightgbm"
    feature_selection: str = "training_fold_finite_nonconstant"
    add_missing_indicators: bool = False

    def __post_init__(self) -> None:
        if self.version != PREPROCESSOR_VERSION:
            raise ActivitySleepJointExpertError("unsupported joint preprocessor")
        if self.input_features != JOINT_FEATURE_COLUMNS:
            raise ActivitySleepJointExpertError("joint input feature order changed")
        if not self.selected_features:
            raise ActivitySleepJointExpertError(
                "joint expert selected no usable features"
            )
        if not set(self.selected_features).issubset(self.input_features):
            raise ActivitySleepJointExpertError(
                "joint expert selected an unknown feature"
            )
        if len(self.selected_features) != len(self.medians):
            raise ActivitySleepJointExpertError("joint median count does not match")
        if len(self.input_features) != len(self.observed_counts) or len(
            self.input_features
        ) != len(self.unique_counts):
            raise ActivitySleepJointExpertError("joint selection audit does not match")
        if any(not math.isfinite(value) for value in self.medians):
            raise ActivitySleepJointExpertError("joint medians must be finite")
        if self.add_missing_indicators:
            raise ActivitySleepJointExpertError(
                "feature masks cannot become learned joint risk inputs"
            )

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> JointNumericFoldPreprocessor:
        values, _, _, _ = _masked_joint_values(frame)
        selected: list[str] = []
        medians: list[float] = []
        observed_counts: list[int] = []
        unique_counts: list[int] = []
        for index, feature in enumerate(JOINT_FEATURE_COLUMNS):
            finite = values[:, index][np.isfinite(values[:, index])]
            observed_count = int(len(finite))
            unique_count = int(len(np.unique(finite))) if observed_count else 0
            observed_counts.append(observed_count)
            unique_counts.append(unique_count)
            if observed_count >= 2 and unique_count >= 2:
                selected.append(feature)
                medians.append(float(np.median(finite)))
        return cls(
            input_features=JOINT_FEATURE_COLUMNS,
            selected_features=tuple(selected),
            medians=tuple(medians),
            observed_counts=tuple(observed_counts),
            unique_counts=tuple(unique_counts),
        )

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        values, _, _, _ = _masked_joint_values(frame)
        indices = [self.input_features.index(name) for name in self.selected_features]
        selected = values[:, indices].copy()
        for column, median in enumerate(self.medians):
            selected[~np.isfinite(selected[:, column]), column] = median
        if not np.isfinite(selected).all():
            raise ActivitySleepJointExpertError(
                "joint preprocessing left non-finite data"
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
class ActivitySleepJointExpertBundle:
    """Serialized joint expert used by later fusion and API tasks."""

    model_id: str
    model_version: str
    bundle_version: str
    feature_schema_version: str
    feature_schema_sha256: str
    split_id: str
    split_manifest_sha256: str
    dataset_ids: tuple[str, ...]
    input_feature_names: tuple[str, ...]
    preprocessor: JointNumericFoldPreprocessor
    classifier: LGBMClassifier
    calibrator: JointProbabilityCalibrator
    ecdf_by_dataset: Mapping[str, Any]
    upstream_expert_guards: Mapping[str, Mapping[str, str]]
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
            raise ActivitySleepJointExpertError("joint bundle identity changed")
        if self.feature_schema_version != FEATURE_SCHEMA_VERSION:
            raise ActivitySleepJointExpertError("joint feature schema changed")
        if self.feature_schema_sha256 != EXPECTED_FEATURE_SCHEMA_SHA256:
            raise ActivitySleepJointExpertError("joint feature schema hash changed")
        if self.split_id != SPLIT_ID:
            raise ActivitySleepJointExpertError("joint split ID changed")
        if self.split_manifest_sha256 != EXPECTED_SPLIT_SHA256:
            raise ActivitySleepJointExpertError("joint split hash changed")
        if self.dataset_ids != JOINT_DATASET_IDS:
            raise ActivitySleepJointExpertError("joint dataset membership changed")
        if self.input_feature_names != JOINT_FEATURE_COLUMNS:
            raise ActivitySleepJointExpertError("joint feature order changed")
        if set(self.ecdf_by_dataset) != set(JOINT_DATASET_IDS):
            raise ActivitySleepJointExpertError("joint ECDF membership changed")
        if _plain_mapping(self.upstream_expert_guards) != _plain_mapping(
            UPSTREAM_EXPERT_GUARDS
        ):
            raise ActivitySleepJointExpertError("upstream expert guard changed")
        if self.training_participant_count <= 0 or self.training_row_count <= 0:
            raise ActivitySleepJointExpertError("joint training scope is empty")
        if not _is_sha256(self.training_participant_sha256):
            raise ActivitySleepJointExpertError("joint participant hash is invalid")

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        _, _, activity_observed, sleep_observed = _masked_joint_values(frame)
        matrix = self.preprocessor.transform(frame)
        raw = np.asarray(
            self.classifier.predict_proba(matrix)[:, 1],
            dtype="float64",
        )
        calibrated = self.calibrator.predict(raw)
        expert_mask = ((activity_observed > 0) & (sleep_observed > 0)).astype("int8")
        raw[expert_mask == 0] = np.nan
        calibrated[expert_mask == 0] = np.nan
        total_observed = activity_observed + sleep_observed
        return pd.DataFrame(
            {
                "raw_probability": raw,
                "calibrated_probability": calibrated,
                "expert_mask": expert_mask,
                "observed_feature_count": total_observed,
                "observed_feature_fraction": total_observed
                / len(self.input_feature_names),
                "activity_observed_feature_count": activity_observed,
                "activity_observed_feature_fraction": activity_observed
                / len(ACTIVITY_FEATURE_COLUMNS),
                "sleep_observed_feature_count": sleep_observed,
                "sleep_observed_feature_fraction": sleep_observed
                / len(SLEEP_FEATURE_COLUMNS),
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
            "ecdf": {
                dataset_id: {
                    "sha256": ecdf.sha256,
                    "training_participant_count": ecdf.training_participant_count,
                    "training_participant_sha256": ecdf.training_participant_sha256,
                    "training_value_count": len(ecdf.sorted_training_values),
                }
                for dataset_id, ecdf in self.ecdf_by_dataset.items()
            },
            "source_transform": JOINT_SOURCE_TRANSFORM,
            "joint_row_eligibility": JOINT_ROW_ELIGIBILITY,
            "upstream_expert_guards": _plain_mapping(self.upstream_expert_guards),
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
    preprocessor: JointNumericFoldPreprocessor
    ecdf_by_dataset: Mapping[str, Any]


def load_joint_training_config(path: str | Path) -> JointTrainingConfig:
    config_path = Path(path).resolve()
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ActivitySleepJointExpertError(
            "joint training config is unreadable"
        ) from exc
    if not isinstance(payload, dict):
        raise ActivitySleepJointExpertError("joint training config must be a mapping")
    try:
        search_space = {
            str(name): tuple(values)
            for name, values in payload["search"]["space"].items()
        }
        upstream = {
            str(name): {str(key): str(value) for key, value in guard.items()}
            for name, guard in payload["upstream_expert_guards"].items()
        }
        config = JointTrainingConfig(
            run_id=str(payload["run"]["run_id"]),
            random_seed=int(payload["runtime"]["random_seed"]),
            split_id=str(payload["data"]["split_id"]),
            split_sha256=str(payload["data"]["split_sha256"]),
            feature_schema_version=str(payload["data"]["feature_schema_version"]),
            feature_schema_sha256=str(payload["data"]["feature_schema_sha256"]),
            dataset_ids=tuple(str(value) for value in payload["data"]["datasets"]),
            upstream_expert_guards=upstream,
            search_space=search_space,
            search_candidate_count=int(payload["search"]["candidate_count"]),
            search_traversal=str(payload["search"]["traversal"]),
            auprc_tie_tolerance=float(payload["search"]["auprc_tie_tolerance"]),
            calibration_positive_participant_threshold=int(
                payload["calibration"]["isotonic_min_positive_participants"]
            ),
            decision_threshold=float(payload["evaluation"]["decision_threshold"]),
            ece_bin_count=int(payload["evaluation"]["ece_bin_count"]),
            lightgbm_n_jobs=int(payload["runtime"]["lightgbm_n_jobs"]),
            config_path=config_path,
            config_sha256=_sha256_file(config_path),
        )
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ActivitySleepJointExpertError(
            "joint training config is malformed"
        ) from exc
    _validate_training_config(payload, config)
    return config


def _validate_training_config(
    payload: Mapping[str, Any],
    config: JointTrainingConfig,
) -> None:
    if payload.get("config_version") != TRAINING_CONFIG_VERSION:
        raise ActivitySleepJointExpertError("joint config version changed")
    if (
        payload.get("task_id") != "MODEL-003"
        or payload.get("model_id") != MODEL_ID
        or payload.get("model_version") != MODEL_VERSION
        or config.run_id != "MH-20260731-003"
    ):
        raise ActivitySleepJointExpertError("joint config task identity changed")
    if (
        config.random_seed != 20260728
        or config.split_id != SPLIT_ID
        or config.split_sha256 != EXPECTED_SPLIT_SHA256
        or config.feature_schema_version != FEATURE_SCHEMA_VERSION
        or config.feature_schema_sha256 != EXPECTED_FEATURE_SCHEMA_SHA256
        or config.dataset_ids != JOINT_DATASET_IDS
        or _plain_mapping(config.upstream_expert_guards)
        != _plain_mapping(UPSTREAM_EXPERT_GUARDS)
    ):
        raise ActivitySleepJointExpertError("joint frozen input binding changed")
    data = payload.get("data", {})
    if (
        tuple(data.get("excluded_datasets", ())) != EXCLUDED_DATASET_IDS
        or data.get("target") != "binary_target"
        or data.get("joint_row_eligibility") != JOINT_ROW_ELIGIBILITY
        or data.get("sample_weighting")
        != (
            "dataset_equal_then_class_equal_then_participant_class_equal_"
            "then_repeat_window_equal"
        )
    ):
        raise ActivitySleepJointExpertError("joint data policy changed")
    expected_space = {
        "n_estimators": (200, 400),
        "learning_rate": (0.03, 0.05),
        "num_leaves": (7, 15, 31),
        "max_depth": (3, 5, -1),
        "min_child_samples": (20, 50),
        "subsample": (0.8, 1.0),
        "colsample_bytree": (0.8, 1.0),
        "reg_alpha": (0, 0.1),
        "reg_lambda": (0.1, 1.0),
    }
    if dict(config.search_space) != expected_space:
        raise ActivitySleepJointExpertError("joint LightGBM search space changed")
    full_count = math.prod(len(values) for values in expected_space.values())
    if (
        config.search_traversal != "deterministic_sha256_subset"
        or not 1 <= config.search_candidate_count <= full_count
        or config.auprc_tie_tolerance < 0
        or config.calibration_positive_participant_threshold != 200
        or not 0 < config.decision_threshold < 1
        or config.ece_bin_count < 2
        or config.lightgbm_n_jobs != 1
    ):
        raise ActivitySleepJointExpertError("joint implementation settings invalid")
    if payload.get("preprocessing") != {
        "activity_source_transform": "right_continuous_training_fold_ecdf",
        "sleep_source_transform": "canonical_same_semantics_identity",
        "imputation": "training_fold_median",
        "categorical_encoding": "numeric_identity_no_categorical_inputs",
        "standardization": "none_for_lightgbm",
        "feature_selection": "training_fold_finite_nonconstant",
        "add_missing_indicators": False,
        "use_feature_masks_as_risk_inputs": False,
        "use_feature_coverage_as_risk_input": False,
        "use_standalone_expert_probabilities_as_inputs": False,
    }:
        raise ActivitySleepJointExpertError("joint preprocessing policy changed")


def load_joint_training_inputs(
    repository_root: str | Path,
    *,
    expected_split_sha256: str = EXPECTED_SPLIT_SHA256,
    upstream_expert_guards: Mapping[str, Mapping[str, str]] = (UPSTREAM_EXPERT_GUARDS),
) -> JointTrainingInputs:
    root = Path(repository_root).resolve()
    split_path = root / DEFAULT_SPLIT_RELATIVE
    if _sha256_file(split_path) != expected_split_sha256:
        raise ActivitySleepJointExpertError("frozen DATA-007 split file hash changed")
    try:
        manifest = json.loads(split_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ActivitySleepJointExpertError("frozen DATA-007 split unreadable") from exc
    rebuilt = build_split_manifest(repository_root=root)
    if manifest != rebuilt:
        raise ActivitySleepJointExpertError(
            "split, five inputs, or MH-003 schema drifted"
        )
    if (
        manifest.get("split_id") != SPLIT_ID
        or manifest.get("feature_schema_binding", {}).get("snapshot_sha256")
        != EXPECTED_FEATURE_SCHEMA_SHA256
    ):
        raise ActivitySleepJointExpertError("frozen DATA-007 binding changed")
    upstream_bindings = _validate_upstream_expert_guards(root, upstream_expert_guards)
    inputs = manifest.get("inputs")
    if not isinstance(inputs, list):
        raise ActivitySleepJointExpertError("frozen DATA-007 inputs malformed")
    input_bindings = {
        str(item["dataset_id"]): item for item in inputs if isinstance(item, dict)
    }
    if set(input_bindings) != set(JOINT_DATASET_IDS + EXCLUDED_DATASET_IDS):
        raise ActivitySleepJointExpertError(
            "frozen DATA-007 dataset membership changed"
        )
    frames: dict[str, pd.DataFrame] = {}
    for dataset_id in JOINT_DATASET_IDS:
        binding = input_bindings[dataset_id]
        path = root / str(binding["canonical_relative_path"])
        if _sha256_file(path) != binding["canonical_sha256"]:
            raise ActivitySleepJointExpertError("joint canonical hash changed")
        try:
            frame = pd.read_parquet(path)
        except (OSError, TypeError, ValueError) as exc:
            raise ActivitySleepJointExpertError("joint canonical unreadable") from exc
        if not isinstance(frame.index, pd.RangeIndex):
            frame = frame.reset_index(drop=True)
        _validate_base_joint_frame(frame, dataset_id, binding)
        frames[dataset_id] = frame
    assignments = pd.DataFrame(manifest["participant_assignments"])
    assignments = assignments[
        assignments["dataset_id"].astype(str).isin(JOINT_DATASET_IDS)
    ].copy()
    assignments = assignments.sort_values(
        "global_participant_id", kind="stable"
    ).reset_index(drop=True)
    expected_keys = set().union(
        *(set(frame["global_participant_id"].astype(str)) for frame in frames.values())
    )
    if set(assignments["global_participant_id"].astype(str)) != expected_keys:
        raise ActivitySleepJointExpertError(
            "joint split does not cover its participants"
        )
    return JointTrainingInputs(
        repository_root=root,
        split_manifest=manifest,
        split_manifest_sha256=expected_split_sha256,
        assignments=assignments,
        frames=frames,
        input_bindings={
            dataset_id: input_bindings[dataset_id] for dataset_id in JOINT_DATASET_IDS
        },
        upstream_expert_bindings=upstream_bindings,
    )


def _validate_upstream_expert_guards(
    root: Path,
    guards: Mapping[str, Mapping[str, str]],
) -> dict[str, dict[str, Any]]:
    if _plain_mapping(guards) != _plain_mapping(UPSTREAM_EXPERT_GUARDS):
        raise ActivitySleepJointExpertError("upstream expert guard policy changed")
    result: dict[str, dict[str, Any]] = {}
    for name in ("activity", "sleep"):
        guard = guards[name]
        model_path = root / guard["model_path"]
        manifest_path = root / guard["manifest_path"]
        if (
            _sha256_file(model_path) != guard["model_sha256"]
            or _sha256_file(manifest_path) != guard["manifest_sha256"]
        ):
            raise ActivitySleepJointExpertError(
                f"upstream {name} expert model or manifest drifted"
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ActivitySleepJointExpertError(
                f"upstream {name} expert manifest unreadable"
            ) from exc
        expected_id, expected_version = _UPSTREAM_IDENTITIES[name]
        if (
            manifest.get("model_id") != expected_id
            or manifest.get("model_version") != expected_version
            or manifest.get("model_sha256") != guard["model_sha256"]
            or manifest.get("split_manifest_sha256") != EXPECTED_SPLIT_SHA256
            or manifest.get("feature_schema_sha256") != EXPECTED_FEATURE_SCHEMA_SHA256
        ):
            raise ActivitySleepJointExpertError(
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


def train_activity_sleep_joint_expert(
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
    config = load_joint_training_config(config_path)
    root = Path(repository_root).resolve()
    inputs = load_joint_training_inputs(
        root,
        expected_split_sha256=config.split_sha256,
        upstream_expert_guards=config.upstream_expert_guards,
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
        / "activity_sleep_joint_expert.joblib"
    )
    production_manifest = (
        Path(model_manifest_path).resolve()
        if model_manifest_path is not None
        else production_model.with_name("activity_sleep_joint_expert_manifest.json")
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
    inputs: JointTrainingInputs,
    config: JointTrainingConfig,
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
    all_participants = set(assignments)
    outer_predictions: list[pd.DataFrame] = []
    outer_search_results: list[dict[str, Any]] = []
    fold_audits: list[dict[str, Any]] = []
    raw_production_oof: list[pd.DataFrame] = []

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
        ["dataset_id", "canonical_row_index"], kind="stable"
    ).reset_index(drop=True)
    _validate_oof_predictions(oof, inputs, assignments)
    metrics = _build_oof_metrics(oof, config)
    final_candidate = _select_final_candidate(outer_search_results, config)
    raw_oof = pd.concat(raw_production_oof, ignore_index=True)
    raw_oof = raw_oof.sort_values(
        ["dataset_id", "canonical_row_index"], kind="stable"
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
        stage / "calibration" / "production.json",
        final_bundle.calibrator.to_dict(),
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
    warnings = _training_warnings(final_bundle, metrics)
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
        "model_file": staged_model.name,
        "model_sha256": model_sha256,
        "training_config_sha256": config.config_sha256,
        "split_id": SPLIT_ID,
        "split_manifest_sha256": EXPECTED_SPLIT_SHA256,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_schema_sha256": EXPECTED_FEATURE_SCHEMA_SHA256,
        "dataset_ids": list(JOINT_DATASET_IDS),
        "excluded_dataset_ids": list(EXCLUDED_DATASET_IDS),
        "input_feature_names": list(JOINT_FEATURE_COLUMNS),
        "selected_feature_names": list(final_bundle.preprocessor.selected_features),
        "source_transform": JOINT_SOURCE_TRANSFORM,
        "joint_row_eligibility": JOINT_ROW_ELIGIBILITY,
        "uses_standalone_expert_probabilities_as_inputs": False,
        "upstream_expert_guards": _plain_mapping(config.upstream_expert_guards),
        "hyperparameters": dict(final_bundle.hyperparameters),
        "calibrator": final_bundle.calibrator.to_dict(),
        "ecdf": final_bundle.audit_dict()["ecdf"],
        "training_scope": final_bundle.training_scope,
        "training_participant_count": final_bundle.training_participant_count,
        "training_participant_sha256": final_bundle.training_participant_sha256,
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
                "models/mental_health/mood_social/v3.3.3/"
                "activity_sleep_joint_expert.joblib"
            ),
            "model_manifest_file": (
                "models/mental_health/mood_social/v3.3.3/"
                "activity_sleep_joint_expert_manifest.json"
            ),
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
        "available_row_count": int(oof["expert_mask"].sum()),
        "positive_row_count": int(oof["binary_target"].sum()),
        "participant_count": int(oof["global_participant_id"].nunique()),
        "calibration_method": final_bundle.calibrator.method,
        "selected_features": list(final_bundle.preprocessor.selected_features),
        "final_hyperparameters": dict(final_bundle.hyperparameters),
        "overall_metrics": metrics["overall"]["calibrated"],
    }


def _prepare_inner_fold(
    inputs: JointTrainingInputs,
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
        raise ActivitySleepJointExpertError(
            "inner fold participant relation is invalid"
        )
    ecdf = _fit_ecdfs(inputs, train_keys)
    train_table = _available_joint_rows(_transform_partition(inputs, ecdf, train_keys))
    validation_table = _available_joint_rows(
        _transform_partition(inputs, ecdf, validation_keys)
    )
    preprocessor = JointNumericFoldPreprocessor.fit(train_table)
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
        ecdf_by_dataset=ecdf,
    )


def _search_outer_fold(
    candidates: Sequence[Mapping[str, Any]],
    prepared_folds: Sequence[_PreparedInnerFold],
    *,
    config: JointTrainingConfig,
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
    inputs: JointTrainingInputs,
    *,
    train_keys: set[str],
    test_keys: set[str],
    best_oof: pd.DataFrame,
    hyperparameters: Mapping[str, int | float],
    outer_fold: int,
    config: JointTrainingConfig,
) -> tuple[ActivitySleepJointExpertBundle, pd.DataFrame, dict[str, Any]]:
    ecdf = _fit_ecdfs(inputs, train_keys)
    train_partition = _transform_partition(inputs, ecdf, train_keys)
    test_table = _transform_partition(inputs, ecdf, test_keys)
    train_table = _available_joint_rows(train_partition)
    preprocessor = JointNumericFoldPreprocessor.fit(train_table)
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
    calibrator = JointProbabilityCalibrator.fit(
        best_oof["raw_probability"],
        best_oof["binary_target"],
        calibration_weight,
        positive_participant_count=positive_participants,
        isotonic_min_positive_participants=(
            config.calibration_positive_participant_threshold
        ),
    )
    bundle = ActivitySleepJointExpertBundle(
        model_id=MODEL_ID,
        model_version=MODEL_VERSION,
        bundle_version=MODEL_BUNDLE_VERSION,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        feature_schema_sha256=EXPECTED_FEATURE_SCHEMA_SHA256,
        split_id=SPLIT_ID,
        split_manifest_sha256=EXPECTED_SPLIT_SHA256,
        dataset_ids=JOINT_DATASET_IDS,
        input_feature_names=JOINT_FEATURE_COLUMNS,
        preprocessor=preprocessor,
        classifier=classifier,
        calibrator=calibrator,
        ecdf_by_dataset=ecdf,
        upstream_expert_guards=UPSTREAM_EXPERT_GUARDS,
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
    available = output["expert_mask"].eq(1)
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
            output.loc[available, "binary_target"],
            output.loc[available, "raw_probability"],
            decision_threshold=config.decision_threshold,
            ece_bin_count=config.ece_bin_count,
        ),
        "calibrated_test_metrics": binary_metrics(
            output.loc[available, "binary_target"],
            output.loc[available, "calibrated_probability"],
            decision_threshold=config.decision_threshold,
            ece_bin_count=config.ece_bin_count,
        ),
        "booster_text_sha256": _sha256_bytes(
            classifier.booster_.model_to_string().encode("utf-8")
        ),
    }
    return bundle, output, audit


def _fit_final_bundle(
    inputs: JointTrainingInputs,
    *,
    raw_oof: pd.DataFrame,
    hyperparameters: Mapping[str, int | float],
    config: JointTrainingConfig,
) -> tuple[ActivitySleepJointExpertBundle, dict[str, Any]]:
    assignments = _assignment_lookup(inputs.assignments)
    all_keys = set(assignments)
    if (
        len(raw_oof) != sum(len(frame) for frame in inputs.frames.values())
        or raw_oof["prediction_id"].duplicated().any()
        or set(raw_oof["global_participant_id"].astype(str)) != all_keys
    ):
        raise ActivitySleepJointExpertError(
            "production calibrator OOF coverage is invalid"
        )
    ecdf = _fit_ecdfs(inputs, all_keys)
    partition = _transform_partition(inputs, ecdf, all_keys)
    table = _available_joint_rows(partition)
    preprocessor = JointNumericFoldPreprocessor.fit(table)
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
    calibrator = JointProbabilityCalibrator.fit(
        calibration_table["raw_probability"],
        calibration_table["binary_target"],
        calibration_weight,
        positive_participant_count=positive_participants,
        isotonic_min_positive_participants=(
            config.calibration_positive_participant_threshold
        ),
    )
    bundle = ActivitySleepJointExpertBundle(
        model_id=MODEL_ID,
        model_version=MODEL_VERSION,
        bundle_version=MODEL_BUNDLE_VERSION,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        feature_schema_sha256=EXPECTED_FEATURE_SCHEMA_SHA256,
        split_id=SPLIT_ID,
        split_manifest_sha256=EXPECTED_SPLIT_SHA256,
        dataset_ids=JOINT_DATASET_IDS,
        input_feature_names=JOINT_FEATURE_COLUMNS,
        preprocessor=preprocessor,
        classifier=classifier,
        calibrator=calibrator,
        ecdf_by_dataset=ecdf,
        upstream_expert_guards=UPSTREAM_EXPERT_GUARDS,
        hyperparameters=dict(hyperparameters),
        training_scope="all_activity_sleep_joint_sources_for_production",
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
        "booster_text_sha256": _sha256_bytes(
            classifier.booster_.model_to_string().encode("utf-8")
        ),
    }


def _fit_ecdfs(
    inputs: JointTrainingInputs,
    training_keys: set[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for dataset_id in JOINT_DATASET_IDS:
        frame = inputs.frames[dataset_id]
        participants = sorted(
            set(frame["global_participant_id"].astype(str)) & training_keys,
            key=lambda value: value.encode("utf-8"),
        )
        binding = inputs.input_bindings[dataset_id]
        result[dataset_id] = _ECDF_CLASS_BY_DATASET[dataset_id].fit(
            frame,
            training_participant_ids=participants,
            split_id=SPLIT_ID,
            canonical_artifact_sha256=str(binding["canonical_sha256"]),
            expected_canonical_frame_sha256=str(binding["canonical_frame_sha256"]),
        )
    return result


def _transform_partition(
    inputs: JointTrainingInputs,
    ecdf_by_dataset: Mapping[str, Any],
    participant_keys: set[str],
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for dataset_order, dataset_id in enumerate(JOINT_DATASET_IDS):
        base = inputs.frames[dataset_id]
        selected = base[
            base["global_participant_id"].astype(str).isin(participant_keys)
        ]
        if selected.empty:
            raise ActivitySleepJointExpertError(
                "training partition omitted a joint source"
            )
        transformed = ecdf_by_dataset[dataset_id].transform(selected)
        part = transformed[
            [
                "dataset_id",
                "global_participant_id",
                "binary_target",
                *JOINT_FEATURE_COLUMNS,
                *JOINT_MASK_COLUMNS,
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
        ["_dataset_order", "canonical_row_index"], kind="stable"
    ).drop(columns="_dataset_order")
    result = result.reset_index(drop=True)
    _masked_joint_values(result)
    return result


def _available_joint_rows(frame: pd.DataFrame) -> pd.DataFrame:
    _, _, activity_observed, sleep_observed = _masked_joint_values(frame)
    available = (activity_observed > 0) & (sleep_observed > 0)
    result = frame.loc[available].reset_index(drop=True)
    if result.empty:
        raise ActivitySleepJointExpertError(
            "ActivitySleepJointExpert fold has no jointly available rows"
        )
    return result


def _fit_classifier(
    matrix: pd.DataFrame,
    target: np.ndarray,
    sample_weight: np.ndarray,
    hyperparameters: Mapping[str, int | float],
    config: JointTrainingConfig,
) -> LGBMClassifier:
    classifier = LGBMClassifier(
        objective="binary",
        random_state=config.random_seed,
        n_jobs=config.lightgbm_n_jobs,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
        subsample_freq=1,
        bagging_seed=config.random_seed,
        feature_fraction_seed=config.random_seed,
        data_random_seed=config.random_seed,
        drop_seed=config.random_seed,
        importance_type="gain",
        **dict(hyperparameters),
    )
    classifier.fit(matrix, target, sample_weight=sample_weight)
    return classifier


def _select_final_candidate(
    outer_search_results: Sequence[Mapping[str, Any]],
    config: JointTrainingConfig,
) -> dict[str, Any]:
    by_candidate: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for outer in outer_search_results:
        for candidate in outer["candidates"]:
            by_candidate[str(candidate["candidate_id"])].append(candidate)
    aggregated: list[dict[str, Any]] = []
    for candidate_id, rows in by_candidate.items():
        if len(rows) != OUTER_FOLD_COUNT:
            raise ActivitySleepJointExpertError(
                "production candidate lacks outer search coverage"
            )
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
                item["mean_nested_inner_auprc"],
                config.auprc_tie_tolerance,
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
    config: JointTrainingConfig,
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

    activity_available = oof["activity_observed_feature_count"].gt(0)
    sleep_available = oof["sleep_observed_feature_count"].gt(0)
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
            "activity_only_row_count": int(
                (activity_available & ~sleep_available).sum()
            ),
            "sleep_only_row_count": int((~activity_available & sleep_available).sum()),
            "neither_side_row_count": int(
                (~activity_available & ~sleep_available).sum()
            ),
        },
        "overall": pair(oof),
        "by_dataset": {
            dataset_id: pair(oof[oof["dataset_id"].astype(str).eq(dataset_id)])
            for dataset_id in JOINT_DATASET_IDS
        },
        "by_outer_fold": {
            str(outer_fold): pair(oof[oof["outer_fold"].eq(outer_fold)])
            for outer_fold in range(OUTER_FOLD_COUNT)
        },
    }


def _validate_oof_predictions(
    oof: pd.DataFrame,
    inputs: JointTrainingInputs,
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
        or set(oof["dataset_id"].astype(str)) != set(JOINT_DATASET_IDS)
    ):
        raise ActivitySleepJointExpertError("outer OOF prediction coverage is invalid")
    activity_available = oof["activity_observed_feature_count"].gt(0)
    sleep_available = oof["sleep_observed_feature_count"].gt(0)
    expected_mask = activity_available & sleep_available
    available = oof["expert_mask"].eq(1)
    unavailable = ~available
    if (
        not expected_mask.equals(available)
        or not np.isfinite(oof.loc[available, "raw_probability"]).all()
        or not np.isfinite(oof.loc[available, "calibrated_probability"]).all()
        or not oof.loc[unavailable, "raw_probability"].isna().all()
        or not oof.loc[unavailable, "calibrated_probability"].isna().all()
        or not np.allclose(
            oof["observed_feature_fraction"],
            oof["observed_feature_count"] / len(JOINT_FEATURE_COLUMNS),
        )
    ):
        raise ActivitySleepJointExpertError(
            "outer OOF probability-mask relation is invalid"
        )
    expected_outer = (
        oof["global_participant_id"]
        .astype(str)
        .map({key: value["outer_fold"] for key, value in assignments.items()})
    )
    if not expected_outer.astype("int64").equals(oof["outer_fold"].astype("int64")):
        raise ActivitySleepJointExpertError(
            "outer OOF prediction fold assignment drifted"
        )


def _validate_base_joint_frame(
    frame: pd.DataFrame,
    dataset_id: str,
    binding: Mapping[str, Any],
) -> None:
    required = {
        "dataset_id",
        "global_participant_id",
        "binary_target",
        "x_source_value",
        "x_source_mask",
        *JOINT_FEATURE_COLUMNS,
        *JOINT_MASK_COLUMNS,
    }
    if not required.issubset(frame.columns):
        raise ActivitySleepJointExpertError(
            "ActivitySleepJointExpert canonical columns changed"
        )
    if (
        set(frame["dataset_id"].astype(str)) != {dataset_id}
        or len(frame) != int(binding["row_count"])
        or int(pd.to_numeric(frame["binary_target"]).sum())
        != int(binding["positive_row_count"])
        or frame["global_participant_id"].astype(str).nunique()
        != int(binding["participant_count"])
    ):
        raise ActivitySleepJointExpertError(
            "ActivitySleepJointExpert canonical summary changed"
        )
    activity = pd.to_numeric(
        frame["activity.activity_volume_norm"], errors="raise"
    ).to_numpy(dtype="float64", na_value=np.nan)
    activity_mask = pd.to_numeric(
        frame["feature_mask.activity.activity_volume_norm"], errors="raise"
    ).to_numpy(dtype="float64", na_value=np.nan)
    source = pd.to_numeric(frame["x_source_value"], errors="raise").to_numpy(
        dtype="float64", na_value=np.nan
    )
    source_mask = pd.to_numeric(frame["x_source_mask"], errors="raise").to_numpy(
        dtype="float64", na_value=np.nan
    )
    _, _, _, sleep_observed = _masked_joint_values(frame)
    if (
        not np.isnan(activity).all()
        or np.any(activity_mask != 0)
        or not np.isfinite(source_mask).all()
        or not np.isin(source_mask, [0.0, 1.0]).all()
        or not np.array_equal(source_mask, np.isfinite(source).astype("float64"))
        or not np.any(source_mask == 1)
        or not np.any(sleep_observed > 0)
    ):
        raise ActivitySleepJointExpertError(
            "ActivitySleepJointExpert base source contract changed"
        )


def _masked_joint_values(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    missing_columns = set(JOINT_FEATURE_COLUMNS + JOINT_MASK_COLUMNS) - set(
        frame.columns
    )
    if missing_columns:
        raise ActivitySleepJointExpertError(
            "ActivitySleepJointExpert unified feature columns are missing"
        )
    values = np.empty((len(frame), len(JOINT_FEATURE_COLUMNS)), dtype="float64")
    masks = np.empty_like(values, dtype="int8")
    for index, (feature, mask_name) in enumerate(
        zip(JOINT_FEATURE_COLUMNS, JOINT_MASK_COLUMNS, strict=True)
    ):
        try:
            value = pd.to_numeric(frame[feature], errors="raise").to_numpy(
                dtype="float64", na_value=np.nan
            )
            mask_float = pd.to_numeric(frame[mask_name], errors="raise").to_numpy(
                dtype="float64", na_value=np.nan
            )
        except (TypeError, ValueError) as exc:
            raise ActivitySleepJointExpertError(
                "ActivitySleepJointExpert feature values are invalid"
            ) from exc
        if (
            not np.isfinite(mask_float).all()
            or not np.isin(mask_float, [0.0, 1.0]).all()
        ):
            raise ActivitySleepJointExpertError(
                "ActivitySleepJointExpert feature masks are invalid"
            )
        mask = mask_float.astype("int8")
        finite = np.isfinite(value)
        if np.any((mask == 1) & ~finite) or np.any((mask == 0) & finite):
            raise ActivitySleepJointExpertError(
                "ActivitySleepJointExpert value-mask relation is invalid"
            )
        values[:, index] = value
        masks[:, index] = mask
    activity_count = masks[:, : len(ACTIVITY_FEATURE_COLUMNS)].sum(axis=1)
    sleep_count = masks[:, len(ACTIVITY_FEATURE_COLUMNS) :].sum(axis=1)
    return values, masks, activity_count, sleep_count


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


def _round_for_tie(value: float, tolerance: float) -> float:
    if tolerance <= 0:
        return value
    return round(value / tolerance) * tolerance


def _feature_importance(
    bundle: ActivitySleepJointExpertBundle,
) -> dict[str, Any]:
    gains = bundle.classifier.booster_.feature_importance(importance_type="gain")
    splits = bundle.classifier.booster_.feature_importance(importance_type="split")
    rows = [
        {
            "feature": feature,
            "gain": float(gain),
            "split_count": int(split),
        }
        for feature, gain, split in zip(
            bundle.preprocessor.selected_features,
            gains,
            splits,
            strict=True,
        )
    ]
    rows.sort(key=lambda item: (-item["gain"], item["feature"]))
    return {
        "method": "lightgbm_gain_for_training_audit",
        "production_attribution_method": "shap_deferred_to_inference_explanation_task",
        "features": rows,
    }


def _assert_bundle_mask_diagnostic(
    bundle: ActivitySleepJointExpertBundle,
) -> None:
    frame = pd.DataFrame(
        {
            **{name: [np.nan] * 4 for name in JOINT_FEATURE_COLUMNS},
            **{name: [0] * 4 for name in JOINT_MASK_COLUMNS},
        }
    )
    activity_feature = ACTIVITY_FEATURE_COLUMNS[0]
    sleep_feature = SLEEP_FEATURE_COLUMNS[0]
    frame.loc[[1, 3], activity_feature] = 0.5
    frame.loc[[1, 3], f"feature_mask.{activity_feature}"] = 1
    frame.loc[[2, 3], sleep_feature] = 7.0
    frame.loc[[2, 3], f"feature_mask.{sleep_feature}"] = 1
    predicted = bundle.predict(frame)
    unavailable = predicted.index < 3
    if (
        not predicted.loc[unavailable, "expert_mask"].eq(0).all()
        or not predicted.loc[unavailable, "raw_probability"].isna().all()
        or not predicted.loc[unavailable, "calibrated_probability"].isna().all()
        or int(predicted.loc[3, "expert_mask"]) != 1
        or not math.isfinite(float(predicted.loc[3, "raw_probability"]))
        or not math.isfinite(float(predicted.loc[3, "calibrated_probability"]))
    ):
        raise ActivitySleepJointExpertError(
            "ActivitySleepJointExpert two-sided mask diagnostic failed"
        )


def _training_warnings(
    bundle: ActivitySleepJointExpertBundle,
    metrics: Mapping[str, Any],
) -> list[dict[str, str]]:
    unselected = sorted(
        set(JOINT_FEATURE_COLUMNS) - set(bundle.preprocessor.selected_features)
    )
    availability = metrics["availability"]
    resilient = metrics["by_dataset"]["resilient"]["calibrated"]
    return [
        {
            "code": "PROXY_TRANSFER",
            "message": (
                "Public wearable activity and sleep fields are deployment-compatible "
                "semantic proxies, not direct validation of camera and sleep-device "
                "measurements."
            ),
        },
        {
            "code": "SMALL_RESILIENT_SOURCE",
            "message": (
                f"RESILIENT contributes {resilient['row_count']} jointly available "
                f"rows and {resilient['positive_row_count']} positive rows; "
                "source-specific metrics should be interpreted with that sample size."
            ),
        },
        {
            "code": "JOINT_ROW_ELIGIBILITY",
            "message": (
                f"{availability['unavailable_row_count']} rows lack real evidence on "
                "at least one side and retain expert_mask=0 with null probability."
            ),
        },
        {
            "code": "STRUCTURAL_FEATURE_ABSENCE",
            "message": (
                "Training-fold finite/nonconstant selection excluded: "
                + (", ".join(unselected) if unselected else "none")
            ),
        },
        {
            "code": "NO_PERFORMANCE_GATE",
            "message": (
                "Metrics are reported as observed; MODEL-003 has no minimum "
                "performance promotion threshold."
            ),
        },
    ]


def _run_manifest(
    inputs: JointTrainingInputs,
    config: JointTrainingConfig,
    metrics: Mapping[str, Any],
    bundle: ActivitySleepJointExpertBundle,
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
        "task_id": "MODEL-003",
        "plan_id": "PLAN-JNT-001",
        "status": "completed",
        "started_at_utc": started.isoformat(),
        "ended_at_utc": ended.isoformat(),
        "duration_seconds": (ended - started).total_seconds(),
        "command": list(command),
        "frozen_document_version": "V3.3.3",
        "model_id": MODEL_ID,
        "model_version": MODEL_VERSION,
        "algorithm": "LightGBM",
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
            }
            for dataset_id, binding in inputs.input_bindings.items()
        ],
        "excluded_sources": list(EXCLUDED_DATASET_IDS),
        "upstream_expert_guards": _plain_mapping(config.upstream_expert_guards),
        "upstream_expert_bindings": _plain_mapping(inputs.upstream_expert_bindings),
        "source_transform": JOINT_SOURCE_TRANSFORM,
        "joint_row_eligibility": JOINT_ROW_ELIGIBILITY,
        "uses_standalone_expert_probabilities_as_inputs": False,
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
                "ecdf": audit["ecdf"],
                "source_transform": audit["source_transform"],
            }
            for audit in fold_audits
        ],
        "production_training": bundle.audit_dict(),
        "production_candidate": dict(final_candidate),
        "metrics": {
            "overall": metrics["overall"],
            "by_dataset": metrics["by_dataset"],
            "availability": metrics["availability"],
            "metrics_sha256": metrics_sha256,
        },
        "mask_only_diagnostic": {
            "status": "pass",
            "result": (
                "both_activity_and_sleep_require_real_input_else_mask_0_and_"
                "null_probability"
            ),
        },
        "all_mask_fusion_hard_gate": "not_applicable_to_MODEL_003",
        "failures": [],
        "warnings": list(warnings),
        "artifacts": {
            "model_sha256": model_sha256,
            "predictions_sha256": predictions_sha256,
            "metrics_sha256": metrics_sha256,
        },
        "production_package_status": (
            "standalone_activity_sleep_joint_expert_active_full_v3_3_3_"
            "package_pending_ART_001"
        ),
        "conclusion": (
            "ActivitySleepJointExpert completed with strict nested participant "
            "OOF evaluation and a production aggregate bundle."
        ),
    }


def _dependency_versions() -> dict[str, str]:
    import lightgbm

    return {
        "joblib": joblib.__version__,
        "lightgbm": lightgbm.__version__,
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
        "src/elderly_monitoring/modules/mental_health/mood_social/experts/joint.py",
        "src/elderly_monitoring/modules/mental_health/mood_social/experts/activity.py",
        "src/elderly_monitoring/modules/mental_health/mood_social/experts/sleep.py",
        "scripts/train_mood_social_activity_sleep_joint_expert_v3_3_3.py",
        "scripts/validate_mood_social_activity_sleep_joint_expert_v3_3_3.py",
        "configs/training/mood_social_activity_sleep_joint_expert_v3_3_3.yaml",
    )
    return {
        relative: _sha256_file(root / relative)
        for relative in relative_paths
        if (root / relative).is_file()
    }


def _write_preprocessing_audit(
    directory: Path,
    bundle: ActivitySleepJointExpertBundle,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    _write_json(directory / "preprocessor.json", bundle.preprocessor.to_dict())
    for dataset_id, ecdf in bundle.ecdf_by_dataset.items():
        _write_bytes(directory / f"{dataset_id}_ecdf.json", ecdf.to_bytes())


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
        "artifact_manifest_version": "mood-social-joint-report-artifacts-v1",
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
        raise ActivitySleepJointExpertError(
            "required ActivitySleepJointExpert file is unreadable"
        ) from exc
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _plain_mapping(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_mapping(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_mapping(item) for item in value]
    return value


def production_model_relative_name(path: Path) -> str:
    return path.name


__all__ = [
    "ActivitySleepJointExpertBundle",
    "ActivitySleepJointExpertError",
    "EXPECTED_FEATURE_SCHEMA_SHA256",
    "EXPECTED_SPLIT_SHA256",
    "JOINT_DATASET_IDS",
    "JOINT_FEATURE_COLUMNS",
    "JOINT_MASK_COLUMNS",
    "JointNumericFoldPreprocessor",
    "JointProbabilityCalibrator",
    "JointTrainingConfig",
    "JointTrainingInputs",
    "MODEL_BUNDLE_VERSION",
    "MODEL_ID",
    "MODEL_VERSION",
    "UPSTREAM_EXPERT_GUARDS",
    "binary_metrics",
    "compute_training_sample_weights",
    "deterministic_search_candidates",
    "load_joint_training_config",
    "load_joint_training_inputs",
    "participant_set_sha256",
    "sample_weight_audit",
    "train_activity_sleep_joint_expert",
]
