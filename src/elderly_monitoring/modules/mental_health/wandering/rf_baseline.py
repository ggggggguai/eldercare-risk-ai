"""Leakage-controlled Random Forest comparison baseline for wandering step 5.

This module is an offline research path only.  It never provides a runtime
fallback and it never reads SmartCare official validation.  Development fits
train rows only; frozen WP test evaluation only loads manifest-bound models.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import platform
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)

from elderly_monitoring.modules.mental_health.wandering.handcrafted_features import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    extract_handcrafted_features,
    feature_schema_document,
)
from elderly_monitoring.modules.mental_health.wandering.preprocessing_bundle import (
    BUNDLE_MODE_DEVELOPMENT,
    BUNDLE_MODE_FROZEN_WP_TEST,
    BundleIntegrityError,
    _canonical_json_bytes,
    _commit_new_output_directory,
    _sha256_file,
    load_preprocessing_bundle,
    load_rf_config,
)


TASK_FOUR_CLASS = "four_class"
TASK_BINARY = "binary"
FOUR_CLASS_NAMES = ("direct", "pacing", "lapping", "random")
BINARY_CLASS_NAMES = ("direct_or_non_wandering", "wandering_like")
RF_SEEDS = (20260731, 20260801, 20260802, 20260803, 20260804)
PRIMARY_SEED = 20260731
RF_BASE_PARAMS = {
    "n_estimators": 500,
    "criterion": "gini",
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "bootstrap": True,
    "class_weight": None,
    "n_jobs": 1,
    "oob_score": False,
    "max_samples": None,
}
FEATURE_ROW_SCHEMA_VERSION = "wandering-rf-feature-table-row-v1"
DATA_INDEX_SCHEMA_VERSION = "wandering-rf-data-index-v1"
DEVELOPMENT_MANIFEST_SCHEMA_VERSION = "wandering-rf-development-manifest-v1"
PUBLIC_BENCHMARK_MANIFEST_SCHEMA_VERSION = "wandering-rf-public-shape-benchmark-manifest-v1"


class RFBaselineError(ValueError):
    """The fixed training/evaluation protocol was violated."""


class ModelTrustError(RFBaselineError):
    """A manifest, dependency, joblib, class, or semantic fingerprint is untrusted."""


@dataclass(frozen=True)
class DevelopmentBuildResult:
    output_dir: Path
    manifest_sha256: str
    manifest: Mapping[str, Any]
    runtime_benchmark: Mapping[str, Any]
    validation_metrics: Mapping[str, Any]
    failure_cases: Mapping[str, Any]


@dataclass(frozen=True)
class FrozenBenchmarkBuildResult:
    output_dir: Path
    manifest_sha256: str
    manifest: Mapping[str, Any]
    test_metrics: Mapping[str, Any]
    failure_cases: Mapping[str, Any]


def create_rf_classifier(seed: int) -> RandomForestClassifier:
    """Instantiate exactly the pre-registered RF; no tuning surface is exposed."""

    if isinstance(seed, bool) or seed not in RF_SEEDS:
        raise RFBaselineError(f"seed must be one of the five fixed seeds: {RF_SEEDS}")
    return RandomForestClassifier(**RF_BASE_PARAMS, random_state=seed)


def build_feature_rows(records: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    """Extract sorted, finite feature rows from already phase-filtered ready records."""

    sample_ids = [record.get("sample_id") for record in records]
    if len(set(sample_ids)) != len(sample_ids) or sample_ids != sorted(sample_ids):
        raise RFBaselineError("feature inputs must have unique sorted sample IDs")
    rows: list[dict[str, Any]] = []
    for record in records:
        if record.get("preprocess_status") != "ready":
            raise RFBaselineError("unavailable records must never enter feature extraction")
        vector = extract_handcrafted_features(record)
        if vector.shape != (26,) or not np.isfinite(vector).all():
            raise RFBaselineError("handcrafted feature vector must be finite [26]")
        rows.append(
            {
                "schema_version": FEATURE_ROW_SCHEMA_VERSION,
                "sample_id": record["sample_id"],
                "source_dataset": record["source_dataset"],
                "split": record["split"],
                "binary_label": record["binary_label"],
                "pattern_label": record["pattern_label"],
                "binary_supervision_eligible": record["binary_supervision_eligible"],
                "pattern_supervision_eligible": record["pattern_supervision_eligible"],
                "features": vector.tolist(),
            }
        )
    return tuple(rows)


def build_binary_sample_weights(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Equalize total train weight across the four source x class groups."""

    if not rows or any(row.get("split") != "train" for row in rows):
        raise RFBaselineError("binary sample weights accept train rows only")
    groups = Counter((row.get("source_dataset"), row.get("label")) for row in rows)
    expected_groups = {
        ("wandering_patterns", 0),
        ("wandering_patterns", 1),
        ("smartcare", 0),
        ("smartcare", 1),
    }
    if set(groups) != expected_groups or any(count <= 0 for count in groups.values()):
        raise RFBaselineError("binary train must contain all four source x class groups")
    total = len(rows)
    weights = np.asarray(
        [total / (4.0 * groups[(row["source_dataset"], row["label"])]) for row in rows],
        dtype=np.float64,
    )
    if not np.isfinite(weights).all() or np.any(weights <= 0.0):
        raise RFBaselineError("binary weights must be finite and positive")
    totals: dict[tuple[str, int], float] = defaultdict(float)
    for row, weight in zip(rows, weights, strict=True):
        totals[(row["source_dataset"], int(row["label"]))] += float(weight)
    if max(totals.values()) - min(totals.values()) > 1e-9:
        raise RFBaselineError("binary source x class total weights are unequal")
    return weights


def fit_rf_task(
    feature_rows: Sequence[Mapping[str, Any]],
    *,
    task: str,
    seed: int,
) -> tuple[RandomForestClassifier, np.ndarray, np.ndarray]:
    """Fit one fixed seed using train rows only and return train medians."""

    task_rows = _task_rows(feature_rows, task=task, split="train")
    if len(task_rows) != (1120 if task == TASK_FOUR_CLASS else 1257):
        raise RFBaselineError(f"fixed {task} train count drifted")
    x, y = _xy(task_rows, task)
    model = create_rf_classifier(seed)
    if task == TASK_BINARY:
        weighted_rows = [
            {**row, "label": int(label)} for row, label in zip(task_rows, y, strict=True)
        ]
        weights = build_binary_sample_weights(weighted_rows)
    else:
        counts = Counter(int(value) for value in y)
        if counts != Counter({0: 280, 1: 280, 2: 280, 3: 280}):
            raise RFBaselineError("four-class train labels must remain balanced 280 each")
        weights = np.ones(len(y), dtype=np.float64)
    model.fit(x, y, sample_weight=weights)
    expected_classes = np.arange(4 if task == TASK_FOUR_CLASS else 2)
    if not np.array_equal(model.classes_, expected_classes):
        raise RFBaselineError("trained RF classes do not match the fixed order")
    medians = np.quantile(x, 0.50, axis=0, method="linear")
    if medians.shape != (26,) or not np.isfinite(medians).all():
        raise RFBaselineError("train medians must be finite [26]")
    return model, medians.astype(np.float64), weights


def model_semantic_fingerprint(model: RandomForestClassifier) -> dict[str, Any]:
    """Hash parameters, classes, and every required fitted tree array."""

    if not hasattr(model, "estimators_") or not hasattr(model, "classes_"):
        raise RFBaselineError("semantic fingerprint requires a fitted RF")
    params = _json_normalize(model.get_params(deep=False))
    classes = np.asarray(model.classes_)
    digest = hashlib.sha256()
    digest.update(_canonical_without_lf({"algorithm": "RandomForestClassifier", "params": params}))
    _update_array_digest(digest, "classes", classes)
    for tree_index, estimator in enumerate(model.estimators_):
        tree = estimator.tree_
        digest.update(_canonical_without_lf({"tree_index": tree_index, "node_count": int(tree.node_count)}))
        for name in ("children_left", "children_right", "feature", "threshold", "value"):
            _update_array_digest(digest, f"tree_{tree_index}.{name}", getattr(tree, name))
    return {
        "schema_version": "wandering-rf-semantic-fingerprint-v1",
        "algorithm": "RandomForestClassifier",
        "sha256": digest.hexdigest(),
        "tree_count": len(model.estimators_),
        "classes": classes.astype(int).tolist(),
        "params": params,
        "covered_tree_arrays": ["children_left", "children_right", "feature", "threshold", "value"],
    }


def local_sensitivity(
    model: RandomForestClassifier,
    feature_vector: Sequence[float] | np.ndarray,
    train_medians: Sequence[float] | np.ndarray,
    *,
    feature_names: Sequence[str] = FEATURE_NAMES,
) -> dict[str, Any]:
    """Replace each feature with its train median and record probability deltas."""

    vector = _finite_vector(feature_vector, "feature_vector")
    medians = _finite_vector(train_medians, "train_medians")
    if tuple(feature_names) != FEATURE_NAMES:
        raise RFBaselineError("local sensitivity feature order has drifted")
    base = np.asarray(model.predict_proba(vector.reshape(1, -1))[0], dtype=np.float64)
    predicted_class = int(model.classes_[int(np.argmax(base))])
    perturbed = np.repeat(vector.reshape(1, -1), len(vector), axis=0)
    perturbed[np.arange(len(vector)), np.arange(len(vector))] = medians
    probabilities = np.asarray(model.predict_proba(perturbed), dtype=np.float64)
    perturbations = []
    for index, name in enumerate(feature_names):
        changed = probabilities[index]
        perturbations.append(
            {
                "feature_index": index,
                "feature_name": name,
                "original_value": float(vector[index]),
                "replacement_train_median": float(medians[index]),
                "probabilities_after_replacement": changed.tolist(),
                "probability_delta": (changed - base).tolist(),
                "predicted_class_probability_delta": float(
                    changed[np.where(model.classes_ == predicted_class)[0][0]]
                    - base[np.where(model.classes_ == predicted_class)[0][0]]
                ),
            }
        )
    return {
        "explanation_type": "local_sensitivity_not_shap_not_causal",
        "predicted_class": predicted_class,
        "base_probabilities": base.tolist(),
        "perturbations": perturbations,
    }


def predict_preprocessed_record(
    model: RandomForestClassifier,
    record: Mapping[str, Any],
    *,
    task: str,
    train_medians: Sequence[float] | np.ndarray,
) -> dict[str, Any]:
    """Predict one ready record or pass through an unavailable status unchanged."""

    if record.get("preprocess_status") == "unavailable":
        return {
            "sample_id": record.get("sample_id"),
            "prediction_status": "unavailable",
            "reason_codes": list(record.get("reason_codes", [])),
        }
    vector = extract_handcrafted_features(record)
    probabilities = np.asarray(model.predict_proba(vector.reshape(1, -1))[0], dtype=np.float64)
    predicted = _predicted_labels(probabilities.reshape(1, -1), task)[0]
    return {
        "sample_id": record.get("sample_id"),
        "prediction_status": "ready",
        "task": task,
        "predicted_label": int(predicted),
        "probabilities": probabilities.tolist(),
        "local_sensitivity": local_sensitivity(model, vector, train_medians),
    }


def current_environment_versions() -> dict[str, str]:
    """Dependency versions that must match before joblib deserialization."""

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }


def load_trusted_development_manifest(
    development_dir: str | Path,
    *,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    """Validate external trust root, dependencies, and every artifact byte first."""

    root = Path(development_dir).resolve(strict=True)
    manifest_path = root / "manifest.json"
    if not _valid_sha256(expected_manifest_sha256):
        raise ModelTrustError("expected development manifest SHA-256 is invalid")
    if not manifest_path.is_file() or _sha256_file(manifest_path) != expected_manifest_sha256:
        raise ModelTrustError("development manifest does not match the external trust root")
    try:
        raw = manifest_path.read_bytes()
        manifest = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ModelTrustError("cannot parse trusted development manifest") from exc
    if not isinstance(manifest, dict) or raw != _canonical_json_bytes(manifest):
        raise ModelTrustError("development manifest is not canonical JSON")
    if manifest.get("schema_version") != DEVELOPMENT_MANIFEST_SCHEMA_VERSION:
        raise ModelTrustError("development manifest schema is not trusted")
    if manifest.get("environment") != current_environment_versions():
        raise ModelTrustError("development dependency versions do not match before deserialization")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ModelTrustError("development manifest has no artifacts")
    for relative_name, descriptor in artifacts.items():
        path = _trusted_relative_path(root, relative_name)
        if not path.is_file() or not isinstance(descriptor, dict):
            raise ModelTrustError(f"manifest-bound artifact is missing: {relative_name}")
        if descriptor != {"byte_count": path.stat().st_size, "sha256": _sha256_file(path)}:
            raise ModelTrustError(f"manifest-bound artifact hash/size mismatch: {relative_name}")
    return manifest


def safe_load_development_model(
    development_dir: str | Path,
    *,
    task: str,
    seed: int,
    expected_manifest_sha256: str,
    joblib_loader: Callable[[Any], Any] = joblib.load,
) -> RandomForestClassifier:
    """Verify trust root/joblib/dependencies before load, then classes/fingerprint."""

    if task not in {TASK_FOUR_CLASS, TASK_BINARY} or seed not in RF_SEEDS:
        raise ModelTrustError("unknown task or non-registered seed")
    root = Path(development_dir).resolve(strict=True)
    manifest = load_trusted_development_manifest(
        root,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    key = f"{task}/{seed}"
    metadata = manifest.get("models", {}).get(key)
    if not isinstance(metadata, dict):
        raise ModelTrustError(f"development manifest does not bind model {key}")
    artifact = metadata.get("artifact")
    if not isinstance(artifact, str) or artifact not in manifest["artifacts"]:
        raise ModelTrustError("model artifact is not manifest-bound")
    path = _trusted_relative_path(root, artifact)
    # All byte/dependency checks above happen before this intentionally unsafe boundary.
    try:
        model = joblib_loader(path)
    except Exception as exc:  # noqa: BLE001 - wrap the pickle trust boundary
        raise ModelTrustError("trusted joblib failed to deserialize") from exc
    if not isinstance(model, RandomForestClassifier):
        raise ModelTrustError("deserialized object is not RandomForestClassifier")
    expected_classes = [0, 1, 2, 3] if task == TASK_FOUR_CLASS else [0, 1]
    if metadata.get("classes") != expected_classes or not np.array_equal(model.classes_, expected_classes):
        raise ModelTrustError("loaded model class order drifted")
    actual_fingerprint = model_semantic_fingerprint(model)
    if metadata.get("semantic_fingerprint") != actual_fingerprint:
        raise ModelTrustError("loaded model semantic fingerprint drifted")
    actual_params = model.get_params(deep=False)
    for name, expected in RF_BASE_PARAMS.items():
        if actual_params.get(name) != expected:
            raise ModelTrustError(f"loaded model fixed RF parameter drifted: {name}")
    if actual_params.get("random_state") != seed:
        raise ModelTrustError("loaded model random_state does not equal the registered seed")
    return model


def build_development_artifacts(
    *,
    rf_config_path: str | Path,
    project_root: str | Path,
    output_dir: str | Path,
) -> DevelopmentBuildResult:
    """Fit both tasks/five seeds on train only and atomically freeze development."""

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"development output already exists: {output}")
    config = load_rf_config(rf_config_path)
    bundle = load_preprocessing_bundle(
        rf_config_path=rf_config_path,
        project_root=project_root,
        mode=BUNDLE_MODE_DEVELOPMENT,
    )
    records = bundle.records_for_features()
    feature_rows = build_feature_rows(records)
    if len(feature_rows) != 1535 or Counter(row["split"] for row in feature_rows) != Counter(train=1257, validation=278):
        raise RFBaselineError("development feature table must contain 1,535 train+validation rows")
    feature_schema = feature_schema_document()
    feature_schema_bytes = _canonical_json_bytes(feature_schema)
    feature_table_bytes = _canonical_jsonl_bytes(feature_rows)

    task_models: dict[tuple[str, int], RandomForestClassifier] = {}
    task_medians: dict[str, np.ndarray] = {}
    model_bytes: dict[tuple[str, int], bytes] = {}
    fingerprints: dict[str, Any] = {}
    predictions_by_file: dict[str, bytes] = {}
    metrics: dict[str, Any] = {
        "schema_version": "wandering-rf-validation-metrics-v1",
        "metric_weighting": "unweighted_real_samples",
        "binary_primary_metric": "source_macro_macro_f1",
        "four_class_primary_metric": "wp_validation_macro_f1",
        "tasks": {},
    }
    importance: dict[str, Any] = {
        "schema_version": "wandering-rf-feature-importance-v1",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "permutation_importance": {"n_repeats": 20, "scoring": "macro_f1"},
        "tasks": {},
    }
    local_rows: list[dict[str, Any]] = []
    failure_cases: dict[str, Any] = {
        "schema_version": "wandering-rf-failure-cases-v1",
        "selection": "primary_seed_high_confidence_errors_top_10_per_cohort",
        "cohorts": {},
    }
    model_fingerprint_document: dict[str, Any] = {
        "schema_version": "wandering-rf-model-fingerprints-v1",
        "models": {},
    }

    for task in (TASK_FOUR_CLASS, TASK_BINARY):
        task_metrics: dict[str, Any] = {"seeds": {}}
        task_importance: dict[str, Any] = {"seeds": {}}
        validation_rows = _task_rows(feature_rows, task=task, split="validation")
        if len(validation_rows) != (240 if task == TASK_FOUR_CLASS else 278):
            raise RFBaselineError(f"fixed {task} validation count drifted")
        x_validation, y_validation = _xy(validation_rows, task)
        train_rows = _task_rows(feature_rows, task=task, split="train")
        x_train, _ = _xy(train_rows, task)
        zero_variance = [
            name for name, variance in zip(FEATURE_NAMES, np.var(x_train, axis=0), strict=True)
            if float(variance) <= 0.0
        ]
        for seed in RF_SEEDS:
            model, medians, sample_weights = fit_rf_task(feature_rows, task=task, seed=seed)
            task_models[(task, seed)] = model
            task_medians.setdefault(task, medians)
            if not np.array_equal(task_medians[task], medians):
                raise RFBaselineError("train medians changed across fixed seeds")
            probabilities = np.asarray(model.predict_proba(x_validation), dtype=np.float64)
            predicted = _predicted_labels(probabilities, task)
            prediction_rows = _prediction_rows(
                validation_rows,
                task=task,
                seed=seed,
                y_true=y_validation,
                predicted=predicted,
                probabilities=probabilities,
            )
            predictions_by_file[f"validation_predictions/{task}_seed_{seed}.jsonl"] = _canonical_jsonl_bytes(prediction_rows)
            seed_metrics = _task_metrics(
                validation_rows,
                task=task,
                y_true=y_validation,
                predicted=predicted,
                probabilities=probabilities,
            )
            task_metrics["seeds"][str(seed)] = seed_metrics
            perm = permutation_importance(
                model,
                x_validation,
                y_validation,
                scoring="f1_macro",
                n_repeats=20,
                random_state=seed,
                n_jobs=1,
            )
            task_importance["seeds"][str(seed)] = {
                "impurity": _named_values(model.feature_importances_),
                "permutation_mean": _named_values(perm.importances_mean),
                "permutation_std": _named_values(perm.importances_std),
                "zero_variance_train_features": zero_variance,
            }
            buffer = io.BytesIO()
            joblib.dump(model, buffer, compress=0, protocol=4)
            model_bytes[(task, seed)] = buffer.getvalue()
            fingerprint = model_semantic_fingerprint(model)
            fingerprints[f"{task}/{seed}"] = fingerprint
            model_fingerprint_document["models"][f"{task}/{seed}"] = fingerprint
            if seed == PRIMARY_SEED:
                for row, vector in zip(validation_rows, x_validation, strict=True):
                    local_rows.append(
                        {
                            "schema_version": "wandering-rf-local-sensitivity-row-v1",
                            "task": task,
                            "seed": seed,
                            "sample_id": row["sample_id"],
                            **local_sensitivity(model, vector, medians),
                        }
                    )
                for cohort_name, cohort_indices in _cohort_indices(validation_rows, task).items():
                    failure_cases["cohorts"][f"{task}/{cohort_name}"] = _high_confidence_errors(
                        [validation_rows[index] for index in cohort_indices],
                        task=task,
                        y_true=y_validation[cohort_indices],
                        predicted=predicted[cohort_indices],
                        probabilities=probabilities[cohort_indices],
                        features=x_validation[cohort_indices],
                    )
        task_metrics["summary"] = _summarize_seed_metrics(task_metrics["seeds"], task)
        task_importance["summary"] = _summarize_importance(task_importance["seeds"])
        metrics["tasks"][task] = task_metrics
        importance["tasks"][task] = task_importance

    data_index = _development_data_index(
        config=config,
        bundle=bundle,
        feature_rows=feature_rows,
        task_medians=task_medians,
    )
    files: dict[str, bytes] = {
        "feature_schema.json": feature_schema_bytes,
        "feature_table.jsonl": feature_table_bytes,
        "data_index.json": _canonical_json_bytes(data_index),
        "validation_metrics.json": _canonical_json_bytes(metrics),
        "feature_importance.json": _canonical_json_bytes(importance),
        "model_fingerprints.json": _canonical_json_bytes(model_fingerprint_document),
        "local_sensitivity.jsonl": _canonical_jsonl_bytes(sorted(local_rows, key=lambda row: (row["task"], row["sample_id"]))),
        "failure_cases.json": _canonical_json_bytes(failure_cases),
        **predictions_by_file,
    }
    for (task, seed), payload in model_bytes.items():
        files[f"models/{task}/seed_{seed}.joblib"] = payload
    config_sha = _sha256_file(Path(rf_config_path))
    manifest = _development_manifest(
        files=files,
        config_sha256=config_sha,
        bundle=bundle,
        fingerprints=fingerprints,
    )
    manifest_bytes = _canonical_json_bytes(manifest)
    files["manifest.json"] = manifest_bytes
    _commit_new_output_directory(output, files)
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    runtime = benchmark_runtime(
        records=records,
        models={task: task_models[(task, PRIMARY_SEED)] for task in (TASK_FOUR_CLASS, TASK_BINARY)},
        model_byte_sizes={task: len(model_bytes[(task, PRIMARY_SEED)]) for task in (TASK_FOUR_CLASS, TASK_BINARY)},
        warmup_iterations=int(config["runtime_benchmark"]["warmup_iterations"]),
        measurement_iterations=int(config["runtime_benchmark"]["measurement_iterations"]),
    )
    return DevelopmentBuildResult(
        output_dir=output,
        manifest_sha256=manifest_sha,
        manifest=manifest,
        runtime_benchmark=runtime,
        validation_metrics=metrics,
        failure_cases=failure_cases,
    )


def build_frozen_wp_test_artifacts(
    *,
    rf_config_path: str | Path,
    project_root: str | Path,
    development_dir: str | Path,
    expected_development_manifest_sha256: str,
    output_dir: str | Path,
) -> FrozenBenchmarkBuildResult:
    """Load trusted train-only models and atomically evaluate the frozen WP test."""

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"public benchmark output already exists: {output}")
    trusted_manifest = load_trusted_development_manifest(
        development_dir,
        expected_manifest_sha256=expected_development_manifest_sha256,
    )
    current_config_sha256 = _sha256_file(Path(rf_config_path))
    if trusted_manifest.get("rf_config_sha256") != current_config_sha256:
        raise ModelTrustError("current RF config does not match the trusted development manifest")
    current_feature_schema_sha256 = hashlib.sha256(
        _canonical_json_bytes(feature_schema_document())
    ).hexdigest()
    if (
        trusted_manifest.get("feature_schema_version") != FEATURE_SCHEMA_VERSION
        or trusted_manifest.get("artifacts", {}).get("feature_schema.json", {}).get("sha256")
        != current_feature_schema_sha256
    ):
        raise ModelTrustError("current feature schema does not match trusted development")
    models = {
        (task, seed): safe_load_development_model(
            development_dir,
            task=task,
            seed=seed,
            expected_manifest_sha256=expected_development_manifest_sha256,
        )
        for task in (TASK_FOUR_CLASS, TASK_BINARY)
        for seed in RF_SEEDS
    }
    bundle = load_preprocessing_bundle(
        rf_config_path=rf_config_path,
        project_root=project_root,
        mode=BUNDLE_MODE_FROZEN_WP_TEST,
    )
    feature_rows = build_feature_rows(bundle.records_for_features())
    if len(feature_rows) != 240 or any(row["source_dataset"] != "wandering_patterns" or row["split"] != "test" for row in feature_rows):
        raise RFBaselineError("frozen benchmark must contain exactly 240 WP test features")
    files: dict[str, bytes] = {"test_features.jsonl": _canonical_jsonl_bytes(feature_rows)}
    metrics: dict[str, Any] = {
        "schema_version": "wandering-rf-public-shape-benchmark-metrics-v1",
        "scope": "public_shape_benchmark",
        "limitations": [
            "4054_cross_partition_shape_neighbor_pairs_below_0.05",
            "not_person_level_generalization",
            "not_camera_validation",
            "not_real_elder_validation",
            "not_clinical_validation",
        ],
        "tasks": {},
    }
    failure_cases: dict[str, Any] = {
        "schema_version": "wandering-rf-failure-cases-v1",
        "selection": "primary_seed_high_confidence_errors_top_10_per_cohort",
        "cohorts": {},
    }
    for task in (TASK_FOUR_CLASS, TASK_BINARY):
        rows = _task_rows(feature_rows, task=task, split="test")
        x, y = _xy(rows, task)
        task_metrics: dict[str, Any] = {"seeds": {}}
        for seed in RF_SEEDS:
            model = models[(task, seed)]
            probabilities = np.asarray(model.predict_proba(x), dtype=np.float64)
            predicted = _predicted_labels(probabilities, task)
            predictions = _prediction_rows(
                rows,
                task=task,
                seed=seed,
                y_true=y,
                predicted=predicted,
                probabilities=probabilities,
            )
            files[f"test_predictions/{task}_seed_{seed}.jsonl"] = _canonical_jsonl_bytes(predictions)
            task_metrics["seeds"][str(seed)] = {
                "primary_metric": "wp_test_macro_f1",
                "primary_metric_value": float(f1_score(y, predicted, average="macro", zero_division=0)),
                "cohorts": {"wp_test": _cohort_metrics(y, predicted, probabilities, task)},
            }
            if seed == PRIMARY_SEED:
                failure_cases["cohorts"][f"{task}/wp_test"] = _high_confidence_errors(
                    rows,
                    task=task,
                    y_true=y,
                    predicted=predicted,
                    probabilities=probabilities,
                    features=x,
                )
        task_metrics["summary"] = _summarize_seed_metrics(task_metrics["seeds"], task, test=True)
        metrics["tasks"][task] = task_metrics
    files["test_metrics.json"] = _canonical_json_bytes(metrics)
    files["failure_cases.json"] = _canonical_json_bytes(failure_cases)
    manifest = {
        "schema_version": PUBLIC_BENCHMARK_MANIFEST_SCHEMA_VERSION,
        "scope": "public_shape_benchmark",
        "comparison_only": True,
        "development_manifest_sha256": expected_development_manifest_sha256,
        "development_manifest_schema": trusted_manifest["schema_version"],
        "rf_config_sha256": _sha256_file(Path(rf_config_path)),
        "feature_schema_sha256": trusted_manifest["artifacts"]["feature_schema.json"]["sha256"],
        "temporal_features_enabled": False,
        "split_sha256": bundle.split_sha256,
        "test_feature_count": 240,
        "official_source_opened": False,
        "models_retrained": False,
        "artifacts": _artifact_descriptors(files),
    }
    manifest_bytes = _canonical_json_bytes(manifest)
    files["manifest.json"] = manifest_bytes
    _commit_new_output_directory(output, files)
    return FrozenBenchmarkBuildResult(
        output_dir=output,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        manifest=manifest,
        test_metrics=metrics,
        failure_cases=failure_cases,
    )


def benchmark_runtime(
    *,
    records: Sequence[Mapping[str, Any]],
    models: Mapping[str, RandomForestClassifier],
    model_byte_sizes: Mapping[str, int],
    warmup_iterations: int = 20,
    measurement_iterations: int = 200,
) -> dict[str, Any]:
    """Measure feature, prediction, and combined single-sample CPU latency."""

    if warmup_iterations != 20 or measurement_iterations != 200:
        raise RFBaselineError("runtime benchmark must remain 20 warmups and 200 measurements")
    result: dict[str, Any] = {
        "schema_version": "wandering-rf-runtime-benchmark-v1",
        "n_jobs": 1,
        "warmup_iterations": warmup_iterations,
        "measurement_iterations": measurement_iterations,
        "clock": "perf_counter_ns",
        "deterministic_hash_excluded": True,
        "tasks": {},
    }
    for task in (TASK_FOUR_CLASS, TASK_BINARY):
        record = next(
            row for row in records
            if row["split"] == "validation"
            and (task == TASK_BINARY or row["pattern_supervision_eligible"] is True)
        )
        model = models[task]
        for _ in range(warmup_iterations):
            vector = extract_handcrafted_features(record)
            model.predict_proba(vector.reshape(1, -1))
        feature_ns: list[int] = []
        prediction_ns: list[int] = []
        total_ns: list[int] = []
        cached = extract_handcrafted_features(record)
        for _ in range(measurement_iterations):
            start = time.perf_counter_ns()
            extract_handcrafted_features(record)
            feature_ns.append(time.perf_counter_ns() - start)
            start = time.perf_counter_ns()
            model.predict_proba(cached.reshape(1, -1))
            prediction_ns.append(time.perf_counter_ns() - start)
            start = time.perf_counter_ns()
            vector = extract_handcrafted_features(record)
            model.predict_proba(vector.reshape(1, -1))
            total_ns.append(time.perf_counter_ns() - start)
        result["tasks"][task] = {
            "sample_id": record["sample_id"],
            "feature_extraction_ms": _latency_summary(feature_ns),
            "rf_prediction_ms": _latency_summary(prediction_ns),
            "combined_ms": _latency_summary(total_ns),
            "primary_model_bytes": int(model_byte_sizes[task]),
        }
    return result


def _task_rows(
    feature_rows: Sequence[Mapping[str, Any]],
    *,
    task: str,
    split: str,
) -> tuple[Mapping[str, Any], ...]:
    if task == TASK_FOUR_CLASS:
        return tuple(
            row for row in feature_rows
            if row.get("split") == split
            and row.get("source_dataset") == "wandering_patterns"
            and row.get("pattern_supervision_eligible") is True
        )
    if task == TASK_BINARY:
        return tuple(
            row for row in feature_rows
            if row.get("split") == split and row.get("binary_supervision_eligible") is True
        )
    raise RFBaselineError(f"unknown task: {task!r}")


def _xy(rows: Sequence[Mapping[str, Any]], task: str) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([row["features"] for row in rows], dtype=np.float64)
    if x.shape != (len(rows), 26) or not np.isfinite(x).all():
        raise RFBaselineError("task feature matrix must be finite [N,26]")
    if task == TASK_FOUR_CLASS:
        mapping = {name: index for index, name in enumerate(FOUR_CLASS_NAMES)}
        try:
            y = np.asarray([mapping[str(row["pattern_label"])] for row in rows], dtype=np.int64)
        except KeyError as exc:
            raise RFBaselineError("unknown four-class label") from exc
    else:
        y = np.asarray([row["binary_label"] for row in rows], dtype=np.int64)
        if not np.all(np.isin(y, [0, 1])):
            raise RFBaselineError("binary labels must be 0/1")
    return x, y


def _predicted_labels(probabilities: np.ndarray, task: str) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    expected_width = 4 if task == TASK_FOUR_CLASS else 2
    if values.ndim != 2 or values.shape[1] != expected_width or not np.isfinite(values).all():
        raise RFBaselineError("probability matrix shape drifted")
    if task == TASK_FOUR_CLASS:
        return np.argmax(values, axis=1).astype(np.int64)
    if task == TASK_BINARY:
        return (values[:, 1] >= 0.5).astype(np.int64)
    raise RFBaselineError(f"unknown task: {task!r}")


def _prediction_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    task: str,
    seed: int,
    y_true: np.ndarray,
    predicted: np.ndarray,
    probabilities: np.ndarray,
) -> list[dict[str, Any]]:
    class_names = FOUR_CLASS_NAMES if task == TASK_FOUR_CLASS else BINARY_CLASS_NAMES
    output: list[dict[str, Any]] = []
    for row, truth, prediction, probs in zip(rows, y_true, predicted, probabilities, strict=True):
        output.append(
            {
                "schema_version": "wandering-rf-prediction-row-v1",
                "task": task,
                "seed": seed,
                "sample_id": row["sample_id"],
                "source_dataset": row["source_dataset"],
                "split": row["split"],
                "true_label": int(truth),
                "true_class_name": class_names[int(truth)],
                "predicted_label": int(prediction),
                "predicted_class_name": class_names[int(prediction)],
                "probabilities": [float(value) for value in probs],
                "correct": bool(truth == prediction),
                "probabilities_calibrated": False,
            }
        )
    return output


def _task_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    task: str,
    y_true: np.ndarray,
    predicted: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    cohorts = {}
    for name, indices in _cohort_indices(rows, task).items():
        cohorts[name] = _cohort_metrics(
            y_true[indices], predicted[indices], probabilities[indices], task
        )
    if task == TASK_FOUR_CLASS:
        primary = cohorts["wp_validation"]["macro_f1"]
        primary_name = "wp_validation_macro_f1"
    else:
        primary = float(np.mean([
            cohorts["wp_validation"]["macro_f1"],
            cohorts["smartcare_validation"]["macro_f1"],
        ]))
        primary_name = "source_macro_macro_f1"
    return {
        "primary_metric": primary_name,
        "primary_metric_value": primary,
        "cohorts": cohorts,
    }


def _cohort_indices(rows: Sequence[Mapping[str, Any]], task: str) -> dict[str, np.ndarray]:
    if task == TASK_FOUR_CLASS:
        return {"wp_validation": np.arange(len(rows), dtype=np.int64)}
    sources = np.asarray([row["source_dataset"] for row in rows], dtype=object)
    return {
        "wp_validation": np.flatnonzero(sources == "wandering_patterns"),
        "smartcare_validation": np.flatnonzero(sources == "smartcare"),
    }


def _cohort_metrics(
    y_true: np.ndarray,
    predicted: np.ndarray,
    probabilities: np.ndarray,
    task: str,
) -> dict[str, Any]:
    labels = list(range(4 if task == TASK_FOUR_CLASS else 2))
    class_names = FOUR_CLASS_NAMES if task == TASK_FOUR_CLASS else BINARY_CLASS_NAMES
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, predicted, labels=labels, zero_division=0
    )
    result: dict[str, Any] = {
        "sample_count": len(y_true),
        "class_counts": {str(label): int(np.count_nonzero(y_true == label)) for label in labels},
        "confusion_matrix": confusion_matrix(y_true, predicted, labels=labels).astype(int).tolist(),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "macro_f1": float(f1_score(y_true, predicted, labels=labels, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "per_class": {
            str(label): {
                "class_name": class_names[label],
                "precision": float(precision[label]),
                "recall": float(recall[label]),
                "f1": float(f1[label]),
                "support": int(support[label]),
            }
            for label in labels
        },
    }
    if task == TASK_BINARY:
        positive = np.clip(probabilities[:, 1], 1e-15, 1.0 - 1e-15)
        matrix = confusion_matrix(y_true, predicted, labels=[0, 1])
        tn, fp, fn, tp = (int(value) for value in matrix.ravel())
        result.update(
            {
                "auroc": float(roc_auc_score(y_true, probabilities[:, 1])),
                "auprc": float(average_precision_score(y_true, probabilities[:, 1])),
                "sensitivity": tp / max(tp + fn, 1),
                "specificity": tn / max(tn + fp, 1),
                "brier": float(brier_score_loss(y_true, probabilities[:, 1])),
                "nll": float(log_loss(y_true, np.column_stack((1.0 - positive, positive)), labels=[0, 1])),
                "positive_probability_clip": [1e-15, 1.0 - 1e-15],
                "decision_threshold": 0.5,
                "probabilities_calibrated": False,
            }
        )
    else:
        result["decision_rule"] = "argmax"
    return result


def _summarize_seed_metrics(
    seeds: Mapping[str, Mapping[str, Any]],
    task: str,
    *,
    test: bool = False,
) -> dict[str, Any]:
    primary = np.asarray([value["primary_metric_value"] for value in seeds.values()], dtype=np.float64)
    cohort_names = sorted(next(iter(seeds.values()))["cohorts"])
    cohorts: dict[str, Any] = {}
    for cohort in cohort_names:
        metrics = ("accuracy", "macro_f1", "balanced_accuracy")
        cohorts[cohort] = {
            name: _mean_std([seed["cohorts"][cohort][name] for seed in seeds.values()])
            for name in metrics
        }
        if task == TASK_BINARY:
            for name in ("auroc", "auprc", "sensitivity", "specificity", "brier", "nll"):
                cohorts[cohort][name] = _mean_std(
                    [seed["cohorts"][cohort][name] for seed in seeds.values()]
                )
    return {
        "seed_count": len(seeds),
        "primary_metric": ("wp_test_macro_f1" if test else ("wp_validation_macro_f1" if task == TASK_FOUR_CLASS else "source_macro_macro_f1")),
        "primary_metric_mean_std": _mean_std(primary),
        "population_std_ddof": 0,
        "cohorts": cohorts,
        "primary_artifact_seed": PRIMARY_SEED,
        "seed_selection_used": False,
    }


def _named_values(values: Sequence[float] | np.ndarray) -> list[dict[str, Any]]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (26,) or not np.isfinite(array).all():
        raise RFBaselineError("importance vector must be finite [26]")
    return [
        {"feature_index": index, "feature_name": name, "value": float(value)}
        for index, (name, value) in enumerate(zip(FEATURE_NAMES, array, strict=True))
    ]


def _summarize_importance(seeds: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for kind in ("impurity", "permutation_mean"):
        matrix = np.asarray(
            [[row["value"] for row in seed[kind]] for seed in seeds.values()],
            dtype=np.float64,
        )
        output[kind] = [
            {
                "feature_index": index,
                "feature_name": name,
                "mean": float(matrix[:, index].mean()),
                "std": float(matrix[:, index].std(ddof=0)),
            }
            for index, name in enumerate(FEATURE_NAMES)
        ]
    output["population_std_ddof"] = 0
    output["zero_variance_train_features"] = next(iter(seeds.values()))["zero_variance_train_features"]
    return output


def _high_confidence_errors(
    rows: Sequence[Mapping[str, Any]],
    *,
    task: str,
    y_true: np.ndarray,
    predicted: np.ndarray,
    probabilities: np.ndarray,
    features: np.ndarray,
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for row, truth, prediction, probs, vector in zip(rows, y_true, predicted, probabilities, features, strict=True):
        if int(truth) == int(prediction):
            continue
        errors.append(
            {
                "sample_id": row["sample_id"],
                "source_dataset": row["source_dataset"],
                "true_label": int(truth),
                "predicted_label": int(prediction),
                "predicted_confidence": float(probs[int(prediction)]),
                "probabilities": probs.tolist(),
                "features": {name: float(value) for name, value in zip(FEATURE_NAMES, vector, strict=True)},
            }
        )
    errors.sort(key=lambda row: (-row["predicted_confidence"], row["sample_id"]))
    return errors[:10]


def _development_data_index(
    *,
    config: Mapping[str, Any],
    bundle: Any,
    feature_rows: Sequence[Mapping[str, Any]],
    task_medians: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    return {
        "schema_version": DATA_INDEX_SCHEMA_VERSION,
        "purpose": "comparison_only",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "temporal_features_enabled": False,
        "feature_table_count": len(feature_rows),
        "feature_table_by_split": dict(sorted(Counter(row["split"] for row in feature_rows).items())),
        "bundle_denominator": {
            "total_records": bundle.total_record_count,
            "ready": bundle.ready_count,
            "unavailable": bundle.unavailable_count,
            "unavailable_sample_ids": [row["sample_id"] for row in bundle.unavailable_records()],
            "unavailable_entered_features_or_fit": False,
        },
        "task_counts": {
            task: {
                split: len(_task_rows(feature_rows, task=task, split=split))
                for split in ("train", "validation")
            }
            for task in (TASK_FOUR_CLASS, TASK_BINARY)
        },
        "train_feature_medians": {
            task: medians.tolist() for task, medians in sorted(task_medians.items())
        },
        "input_hashes": dict(sorted(bundle.input_hashes.items())),
        "split_sha256": bundle.split_sha256,
        "environment": current_environment_versions(),
        "cpu": platform.processor() or platform.machine(),
        "rf_parameters": {**RF_BASE_PARAMS, "random_state": "per_seed"},
        "seeds": list(RF_SEEDS),
        "primary_seed": PRIMARY_SEED,
        "development_constraints": {
            "fit_partitions": ["train"],
            "evaluation_partitions": ["validation"],
            "test_features_generated": False,
            "test_predictions_generated": False,
            "test_metrics_generated": False,
            "official_source_opened": False,
            "grid_search_used": False,
            "seed_selection_used": False,
            "probability_calibration_used": False,
            "rejection_used": False,
        },
    }


def _development_manifest(
    *,
    files: Mapping[str, bytes],
    config_sha256: str,
    bundle: Any,
    fingerprints: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    models = {}
    for task in (TASK_FOUR_CLASS, TASK_BINARY):
        classes = [0, 1, 2, 3] if task == TASK_FOUR_CLASS else [0, 1]
        for seed in RF_SEEDS:
            models[f"{task}/{seed}"] = {
                "artifact": f"models/{task}/seed_{seed}.joblib",
                "classes": classes,
                "semantic_fingerprint": fingerprints[f"{task}/{seed}"],
            }
    return {
        "schema_version": DEVELOPMENT_MANIFEST_SCHEMA_VERSION,
        "purpose": "comparison_only",
        "rf_config_sha256": config_sha256,
        "preprocessing_manifest_sha256": bundle.input_hashes["preprocessing_manifest"],
        "split_sha256": bundle.split_sha256,
        "environment": current_environment_versions(),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "temporal_features_enabled": False,
        "feature_table_partitions": ["train", "validation"],
        "fit_partition": "train",
        "validation_used_for_selection": False,
        "test_artifacts_present": False,
        "official_source_opened": False,
        "models": models,
        "artifacts": _artifact_descriptors(files),
    }


def _artifact_descriptors(files: Mapping[str, bytes]) -> dict[str, dict[str, Any]]:
    return {
        name: {"byte_count": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        for name, payload in sorted(files.items())
    }


def _canonical_jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_json_bytes(dict(row)) for row in rows)


def _canonical_without_lf(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _update_array_digest(digest: Any, name: str, value: Any) -> None:
    array = np.asarray(value)
    if array.dtype.kind in "iu":
        normalized = np.ascontiguousarray(array, dtype="<i8")
    elif array.dtype.kind in "f":
        normalized = np.ascontiguousarray(array, dtype="<f8")
    else:
        raise RFBaselineError(f"unsupported semantic array dtype: {name}")
    digest.update(_canonical_without_lf({"name": name, "shape": list(normalized.shape), "dtype": str(normalized.dtype)}))
    digest.update(normalized.tobytes(order="C"))


def _json_normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_normalize(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_json_normalize(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            raise RFBaselineError("non-finite model parameter")
        return value
    return repr(value)


def _finite_vector(value: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise RFBaselineError(f"{name} must be numeric") from exc
    if array.shape != (26,) or not np.isfinite(array).all():
        raise RFBaselineError(f"{name} must be finite [26]")
    return array


def _trusted_relative_path(root: Path, relative_name: Any) -> Path:
    if not isinstance(relative_name, str) or not relative_name:
        raise ModelTrustError("manifest artifact name is invalid")
    path = (root / Path(relative_name)).resolve(strict=False)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ModelTrustError("manifest artifact escapes development directory") from exc
    return path


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _mean_std(values: Sequence[float] | np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "std": float(array.std(ddof=0))}


def _latency_summary(values_ns: Sequence[int]) -> dict[str, float]:
    values_ms = np.asarray(values_ns, dtype=np.float64) / 1_000_000.0
    return {
        "median": float(np.quantile(values_ms, 0.50, method="linear")),
        "p95": float(np.quantile(values_ms, 0.95, method="linear")),
    }


__all__ = [
    "BINARY_CLASS_NAMES",
    "FOUR_CLASS_NAMES",
    "PRIMARY_SEED",
    "RF_BASE_PARAMS",
    "RF_SEEDS",
    "TASK_BINARY",
    "TASK_FOUR_CLASS",
    "DevelopmentBuildResult",
    "FrozenBenchmarkBuildResult",
    "ModelTrustError",
    "RFBaselineError",
    "benchmark_runtime",
    "build_binary_sample_weights",
    "build_development_artifacts",
    "build_feature_rows",
    "build_frozen_wp_test_artifacts",
    "create_rf_classifier",
    "current_environment_versions",
    "fit_rf_task",
    "load_trusted_development_manifest",
    "local_sensitivity",
    "model_semantic_fingerprint",
    "predict_preprocessed_record",
    "safe_load_development_model",
]
