from __future__ import annotations

import hashlib
import json
import os
import platform
try:
    import resource
except ImportError:  # pragma: no cover - Windows does not provide resource.
    resource = None
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from elderly_monitoring.modules.fall_risk.sit_stand_training import (
    SIT_STAND_CHANNELS,
    SIT_STAND_JOINTS,
    SIT_STAND_TABULAR_FEATURES,
)


@dataclass(frozen=True)
class SitStandLogisticConfig:
    seed: int = 42
    regularization_c: float = 1.0
    max_iter: int = 1000
    threshold: float = 0.5
    source_balance: bool = True

    def __post_init__(self) -> None:
        if self.regularization_c <= 0:
            raise ValueError("regularization_c must be positive")
        if self.max_iter < 1:
            raise ValueError("max_iter must be positive")
        if not 0 < self.threshold < 1:
            raise ValueError("threshold must be within (0, 1)")


def train_sit_stand_logistic(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    metadata_path: str | Path | None = None,
    config: SitStandLogisticConfig | None = None,
    allow_provisional: bool = False,
    resume_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    training = config or SitStandLogisticConfig()
    if not allow_provisional:
        raise ValueError(
            "sit-stand logistic training is provisional; pass allow_provisional explicitly"
        )
    source_path = Path(dataset_path)
    source_metadata_path = (
        Path(metadata_path)
        if metadata_path is not None
        else source_path.with_name("metadata.json")
    )
    if not source_path.is_file():
        raise FileNotFoundError(f"sit-stand dataset not found: {source_path}")
    if not source_metadata_path.is_file():
        raise FileNotFoundError(
            f"sit-stand dataset metadata not found: {source_metadata_path}"
        )
    metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    _validate_metadata(metadata, source_path)
    arrays = _load_dataset(source_path)
    _validate_arrays(arrays, metadata)

    destination = Path(output_dir)
    output_paths = {
        "checkpoint": destination / "checkpoint.joblib",
        "checkpoint_metadata": destination / "checkpoint_metadata.json",
        "config": destination / "config.json",
        "environment": destination / "environment.json",
        "log": destination / "training_log.jsonl",
        "metrics": destination / "metrics.json",
        "predictions": destination / "validation_predictions.jsonl",
        "failures": destination / "failure_cases.jsonl",
    }
    existing = [path for path in output_paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"sit-stand training output already exists: {existing[0]}")
    destination.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    log_rows: list[dict[str, Any]] = [
        {
            "event": "run_started",
            "elapsed_sec": 0.0,
            "seed": training.seed,
            "status": "provisional",
            "test_access": False,
        }
    ]
    partitions = arrays["partitions"].astype(str)
    labels = arrays["labels"].astype(np.int64, copy=False)
    features = arrays["tabular_features"].astype(np.float64, copy=False)
    indices = {
        partition: np.flatnonzero(partitions == partition)
        for partition in ("train", "validation")
    }
    for partition, partition_indices in indices.items():
        if len(partition_indices) == 0:
            raise ValueError(f"sit-stand {partition} partition is empty")
        if set(labels[partition_indices].tolist()) != {0, 1}:
            raise ValueError(f"sit-stand {partition} partition lacks a binary class")

    weights = arrays["sample_weights"].astype(np.float64, copy=True)
    train_weights = _balanced_training_weights(
        labels,
        weights,
        arrays["source_group_ids"].astype(str),
        indices["train"],
        source_balance=training.source_balance,
    )
    resumed = resume_checkpoint is not None
    if resume_checkpoint is None:
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "logistic",
                    LogisticRegression(
                        C=training.regularization_c,
                        max_iter=training.max_iter,
                        random_state=training.seed,
                        solver="lbfgs",
                    ),
                ),
            ]
        )
        model.fit(
            features[indices["train"]],
            labels[indices["train"]],
            logistic__sample_weight=train_weights,
        )
        checkpoint_source = None
    else:
        checkpoint_source = Path(resume_checkpoint)
        if not checkpoint_source.is_file():
            raise FileNotFoundError(
                f"sit-stand resume checkpoint not found: {checkpoint_source}"
            )
        payload = joblib.load(checkpoint_source)
        _validate_checkpoint_payload(payload, metadata, training)
        model = payload["model"]
    checkpoint_payload = {
        "schema_version": "sit-stand-logistic-checkpoint-v1",
        "task": metadata["task"],
        "status": "provisional",
        "model_version": "sit-stand-event-presence-logreg-v0.1-provisional",
        "model": model,
        "feature_names": list(SIT_STAND_TABULAR_FEATURES),
        "dataset_sha256": metadata["dataset_sha256"],
        "input_contract": {
            "joint_order": list(SIT_STAND_JOINTS),
            "channel_order": list(SIT_STAND_CHANNELS),
            "preparation_config": metadata["preparation_config"],
            "normalization_contract": metadata["normalization_contract"],
        },
        "training_config": asdict(training),
        "threshold": training.threshold,
    }
    _write_joblib_atomic(output_paths["checkpoint"], checkpoint_payload)
    checkpoint_metadata = {
        key: value for key, value in checkpoint_payload.items() if key != "model"
    }
    checkpoint_metadata["checkpoint_sha256"] = _sha256_file(
        output_paths["checkpoint"]
    )
    checkpoint_metadata["resumed_from"] = (
        checkpoint_source.as_posix() if checkpoint_source else None
    )
    checkpoint_metadata["resumed_from_sha256"] = (
        _sha256_file(checkpoint_source) if checkpoint_source else None
    )
    _write_json_atomic(output_paths["checkpoint_metadata"], checkpoint_metadata)

    validation_probabilities = model.predict_proba(
        features[indices["validation"]]
    )[:, 1]
    train_probabilities = model.predict_proba(features[indices["train"]])[:, 1]
    train_rows = _aggregate_event_predictions(
        _prediction_rows(
            arrays,
            indices["train"],
            train_probabilities,
            threshold=training.threshold,
        ),
        training.threshold,
    )
    train_metrics = _binary_metrics(train_rows, training.threshold)
    validation_rows = _prediction_rows(
        arrays,
        indices["validation"],
        validation_probabilities,
        threshold=training.threshold,
    )
    event_rows = _aggregate_event_predictions(validation_rows, training.threshold)
    validation_metrics = _binary_metrics(event_rows, training.threshold)
    rule_event_rows = _aggregate_rule_predictions(validation_rows)
    rule_metrics = _binary_metrics(rule_event_rows, 0.5)
    dataset_metrics = {
        dataset: _binary_metrics(rows, training.threshold)
        for dataset, rows in _group_rows(event_rows, "dataset").items()
        if {int(row["label"]) for row in rows} == {0, 1}
    }
    failures = [
        row
        for row in event_rows
        if int(row["predicted_label"]) != int(row["label"])
    ]
    failures.sort(
        key=lambda row: abs(float(row["probability"]) - int(row["label"])),
        reverse=True,
    )
    _write_jsonl_atomic(output_paths["predictions"], event_rows)
    _write_jsonl_atomic(output_paths["failures"], failures)

    elapsed = time.perf_counter() - started
    log_rows.append(
        {
            "event": "run_completed",
            "elapsed_sec": round(elapsed, 6),
            "resumed": resumed,
            "validation_event_count": len(event_rows),
            "validation_balanced_accuracy": validation_metrics[
                "balanced_accuracy"
            ],
            "test_access": False,
        }
    )
    _write_jsonl_atomic(output_paths["log"], log_rows)
    config_payload = {
        "schema_version": "sit-stand-logistic-run-config-v1",
        "task": metadata["task"],
        "status": "provisional",
        "run_id": destination.name,
        "training_config": asdict(training),
        "dataset_path": source_path.as_posix(),
        "dataset_sha256": metadata["dataset_sha256"],
        "metadata_path": source_metadata_path.as_posix(),
        "metadata_sha256": _sha256_file(source_metadata_path),
        "data_provenance": _data_provenance(metadata, source_metadata_path),
        "code": _code_fingerprint(),
        "test_access": False,
        "resume_checkpoint": checkpoint_source.as_posix() if checkpoint_source else None,
    }
    _write_json_atomic(output_paths["config"], config_payload)
    run_config_sha256 = _sha256_file(output_paths["config"])
    environment = {
        "schema_version": "sit-stand-training-environment-v1",
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "device": "cpu",
        "peak_rss_mb": (
            round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / (1024.0 * 1024.0),
                3,
            )
            if resource is not None
            else None
        ),
    }
    _write_json_atomic(output_paths["environment"], environment)
    metrics = {
        "schema_version": "sit-stand-logistic-training-report-v1",
        "task": metadata["task"],
        "target_task": metadata["target_task"],
        "status": "provisional",
        "run_id": destination.name,
        "model_version": checkpoint_payload["model_version"],
        "seed": training.seed,
        "threshold": training.threshold,
        "resumed": resumed,
        "elapsed_sec": round(elapsed, 6),
        "train_window_count": int(len(indices["train"])),
        "validation_window_count": int(len(indices["validation"])),
        "validation_event_count": len(event_rows),
        "train": train_metrics,
        "validation": validation_metrics,
        "validation_dataset_metrics": dataset_metrics,
        "e0_rule_validation": rule_metrics,
        "failure_case_count": len(failures),
        "test_evaluated": False,
        "test": None,
        "dataset_sha256": metadata["dataset_sha256"],
        "checkpoint_sha256": checkpoint_metadata["checkpoint_sha256"],
        "run_config_sha256": run_config_sha256,
        "code_sha256": config_payload["code"]["sha256"],
        "input_contract": checkpoint_payload["input_contract"],
        "limitations": [
            "validation metrics are provisional clip-level proxy results",
            "no test features or test metrics were read",
            "the run does not estimate event onset/offset or FP/hour",
            "the probability is not a clinical functional or future fall-risk score",
        ],
    }
    _write_json_atomic(output_paths["metrics"], metrics)
    output_size = sum(
        path.stat().st_size for path in output_paths.values() if path.is_file()
    )
    return {
        "output_dir": destination.as_posix(),
        "run_id": destination.name,
        "checkpoint_path": output_paths["checkpoint"].as_posix(),
        "metrics_path": output_paths["metrics"].as_posix(),
        "log_path": output_paths["log"].as_posix(),
        "failure_cases_path": output_paths["failures"].as_posix(),
        "elapsed_sec": round(elapsed, 6),
        "output_size_bytes": output_size,
        "resumed": resumed,
        "validation": validation_metrics,
        "test_evaluated": False,
    }


def evaluate_sit_stand_logistic(
    dataset_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    metadata_path: str | Path | None = None,
    partition: str = "validation",
) -> dict[str, Any]:
    if partition != "validation":
        raise ValueError("provisional sit-stand evaluator only permits validation")
    source_path = Path(dataset_path)
    source_metadata_path = (
        Path(metadata_path)
        if metadata_path is not None
        else source_path.with_name("metadata.json")
    )
    metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    _validate_metadata(metadata, source_path)
    arrays = _load_dataset(source_path)
    _validate_arrays(arrays, metadata)
    payload = joblib.load(Path(checkpoint_path))
    if payload.get("dataset_sha256") != metadata["dataset_sha256"]:
        raise ValueError("sit-stand checkpoint dataset SHA-256 mismatch")
    indices = np.flatnonzero(arrays["partitions"].astype(str) == partition)
    probabilities = payload["model"].predict_proba(
        arrays["tabular_features"][indices]
    )[:, 1]
    rows = _aggregate_event_predictions(
        _prediction_rows(
            arrays, indices, probabilities, threshold=float(payload["threshold"])
        ),
        float(payload["threshold"]),
    )
    report = {
        "schema_version": "sit-stand-logistic-evaluation-v1",
        "status": "provisional",
        "partition": partition,
        "test_evaluated": False,
        "metrics": _binary_metrics(rows, float(payload["threshold"])),
        "dataset_sha256": metadata["dataset_sha256"],
        "checkpoint_sha256": _sha256_file(Path(checkpoint_path)),
    }
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(f"sit-stand evaluation output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(destination, report)
    return report


def evaluate_sit_stand_rule(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    metadata_path: str | Path | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    source_path = Path(dataset_path)
    source_metadata_path = (
        Path(metadata_path)
        if metadata_path is not None
        else source_path.with_name("metadata.json")
    )
    metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    _validate_metadata(metadata, source_path)
    arrays = _load_dataset(source_path)
    _validate_arrays(arrays, metadata)
    started = time.perf_counter()
    indices = np.flatnonzero(arrays["partitions"].astype(str) == "validation")
    window_rows = _prediction_rows(
        arrays,
        indices,
        arrays["rule_scores"][indices],
        threshold=0.5,
    )
    event_rows = _aggregate_rule_predictions(window_rows)
    metrics = _binary_metrics(event_rows, 0.5)
    failures = [
        row
        for row in event_rows
        if int(row["predicted_label"]) != int(row["label"])
    ]
    destination = Path(output_dir)
    paths = {
        "config": destination / "config.json",
        "environment": destination / "environment.json",
        "log": destination / "evaluation_log.jsonl",
        "metrics": destination / "metrics.json",
        "predictions": destination / "validation_predictions.jsonl",
        "failures": destination / "failure_cases.jsonl",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"sit-stand E0 output already exists: {existing[0]}")
    destination.mkdir(parents=True, exist_ok=True)
    elapsed = time.perf_counter() - started
    code = _code_fingerprint()
    input_contract = _input_contract(metadata)
    config_payload = {
        "schema_version": "sit-stand-rule-run-config-v1",
        "task": metadata["task"],
        "target_task": metadata["target_task"],
        "status": "provisional",
        "run_id": destination.name,
        "model_version": "sit-stand-risk-rule-v0.1",
        "seed": seed,
        "threshold": 0.5,
        "dataset_path": source_path.as_posix(),
        "metadata_path": source_metadata_path.as_posix(),
        "data_provenance": _data_provenance(metadata, source_metadata_path),
        "input_contract": input_contract,
        "code": code,
        "test_access": False,
    }
    _write_json_atomic(paths["config"], config_payload)
    run_config_sha256 = _sha256_file(paths["config"])
    environment = {
        "schema_version": "sit-stand-evaluation-environment-v1",
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "device": "cpu",
        "peak_rss_mb": (
            round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / (1024.0 * 1024.0),
                3,
            )
            if resource is not None
            else None
        ),
    }
    _write_json_atomic(paths["environment"], environment)
    _write_jsonl_atomic(
        paths["log"],
        [
            {
                "event": "run_started",
                "elapsed_sec": 0.0,
                "seed": seed,
                "status": "provisional",
                "test_access": False,
            },
            {
                "event": "run_completed",
                "elapsed_sec": round(elapsed, 6),
                "validation_event_count": len(event_rows),
                "validation_balanced_accuracy": metrics["balanced_accuracy"],
                "test_access": False,
            },
        ],
    )
    report = {
        "schema_version": "sit-stand-rule-evaluation-v1",
        "task": metadata["task"],
        "target_task": metadata["target_task"],
        "status": "provisional",
        "run_id": destination.name,
        "model_version": "sit-stand-risk-rule-v0.1",
        "seed": seed,
        "elapsed_sec": round(elapsed, 6),
        "partition": "validation",
        "threshold": 0.5,
        "metrics": metrics,
        "test_evaluated": False,
        "test": None,
        "dataset_sha256": metadata["dataset_sha256"],
        "run_config_sha256": run_config_sha256,
        "code_sha256": code["sha256"],
        "data_provenance": config_payload["data_provenance"],
        "input_contract": input_contract,
        "limitations": [
            "clip-level event-presence evaluation is not onset/offset localization",
            "FP/hour is unavailable because continuous reviewed background is absent",
            "test pose and test labels are not evaluated",
        ],
    }
    _write_json_atomic(paths["metrics"], report)
    _write_jsonl_atomic(paths["predictions"], event_rows)
    _write_jsonl_atomic(paths["failures"], failures)
    return {
        "output_dir": destination.as_posix(),
        "metrics_path": paths["metrics"].as_posix(),
        "failure_cases_path": paths["failures"].as_posix(),
        "log_path": paths["log"].as_posix(),
        "elapsed_sec": round(elapsed, 6),
        "validation": metrics,
        "test_evaluated": False,
    }


def _validate_metadata(metadata: Mapping[str, Any], dataset_path: Path) -> None:
    if metadata.get("schema_version") != "sit-stand-event-presence-dataset-v1":
        raise ValueError("unsupported sit-stand prepared dataset schema")
    if metadata.get("task") != "sit_stand_event_presence_proxy_v1":
        raise ValueError("sit-stand logistic task contract mismatch")
    if metadata.get("status") != "provisional":
        raise ValueError("sit-stand logistic expects a provisional dataset")
    if metadata.get("test_pose_read") is not False:
        raise ValueError("sit-stand provisional dataset must not read test pose")
    if metadata.get("joint_order") != list(SIT_STAND_JOINTS):
        raise ValueError("sit-stand joint order mismatch")
    if metadata.get("channel_order") != list(SIT_STAND_CHANNELS):
        raise ValueError("sit-stand channel order mismatch")
    if metadata.get("tabular_feature_names") != list(SIT_STAND_TABULAR_FEATURES):
        raise ValueError("sit-stand tabular channel contract mismatch")
    if metadata.get("dataset_sha256") != _sha256_file(dataset_path):
        raise ValueError("sit-stand dataset SHA-256 does not match metadata")


def _load_dataset(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _validate_arrays(arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any]) -> None:
    required = {
        "features",
        "tabular_features",
        "labels",
        "partitions",
        "sample_ids",
        "event_ids",
        "video_ids",
        "split_group_ids",
        "source_group_ids",
        "datasets",
        "action_ids",
        "sample_weights",
        "rule_scores",
    }
    missing = sorted(required - set(arrays))
    if missing:
        raise ValueError(f"sit-stand dataset is missing arrays: {missing}")
    count = len(arrays["labels"])
    if any(len(arrays[name]) != count for name in required):
        raise ValueError("sit-stand prepared arrays have inconsistent lengths")
    if arrays["features"].ndim != 4 or arrays["features"].shape[2:] != (
        len(SIT_STAND_JOINTS),
        len(SIT_STAND_CHANNELS),
    ):
        raise ValueError("sit-stand prepared tensor channel mismatch")
    if arrays["tabular_features"].shape != (count, len(SIT_STAND_TABULAR_FEATURES)):
        raise ValueError("sit-stand tabular feature shape mismatch")
    if not np.isfinite(arrays["features"]).all() or not np.isfinite(
        arrays["tabular_features"]
    ).all():
        raise ValueError("sit-stand prepared features contain non-finite values")
    if set(arrays["partitions"].astype(str).tolist()) != {"train", "validation"}:
        raise ValueError("sit-stand provisional arrays must contain only train/validation")
    if set(arrays["labels"].astype(int).tolist()) != {0, 1}:
        raise ValueError("sit-stand prepared labels must be binary")
    if list(arrays["features"].shape) != metadata.get("feature_shape"):
        raise ValueError("sit-stand feature shape does not match metadata")


def _balanced_training_weights(
    labels: np.ndarray,
    base_weights: np.ndarray,
    source_groups: np.ndarray,
    train_indices: np.ndarray,
    *,
    source_balance: bool,
) -> np.ndarray:
    selected_labels = labels[train_indices]
    weights = base_weights[train_indices].astype(np.float64, copy=True)
    if np.any(weights <= 0) or not np.isfinite(weights).all():
        raise ValueError("sit-stand training weights must be finite and positive")
    for target in (0, 1):
        mask = selected_labels == target
        mass = float(weights[mask].sum())
        if mass <= 0:
            raise ValueError("sit-stand class weight mass is empty")
        weights[mask] *= 0.5 / mass
    if source_balance:
        selected_sources = source_groups[train_indices]
        for target in (0, 1):
            target_mask = selected_labels == target
            unique_sources = sorted(set(selected_sources[target_mask].tolist()))
            target_mass = float(weights[target_mask].sum()) / max(1, len(unique_sources))
            for source in unique_sources:
                mask = target_mask & (selected_sources == source)
                source_mass = float(weights[mask].sum())
                if source_mass > 0:
                    weights[mask] *= target_mass / source_mass
    weights *= len(weights) / float(weights.sum())
    return weights


def _prediction_rows(
    arrays: Mapping[str, np.ndarray],
    indices: np.ndarray,
    probabilities: Sequence[float],
    *,
    threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, probability in zip(indices.tolist(), probabilities, strict=True):
        rows.append(
            {
                "sample_id": str(arrays["sample_ids"][index]),
                "event_id": str(arrays["event_ids"][index]),
                "video_id": str(arrays["video_ids"][index]),
                "split_group_id": str(arrays["split_group_ids"][index]),
                "source_group_id": str(arrays["source_group_ids"][index]),
                "dataset": str(arrays["datasets"][index]),
                "action_id": str(arrays["action_ids"][index]),
                "label": int(arrays["labels"][index]),
                "probability": float(probability),
                "predicted_label": int(float(probability) >= threshold),
                "rule_score": float(arrays["rule_scores"][index]),
            }
        )
    return rows


def _aggregate_event_predictions(
    rows: Sequence[Mapping[str, Any]], threshold: float
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["event_id"])].append(row)
    outputs: list[dict[str, Any]] = []
    for event_id in sorted(grouped):
        event_rows = grouped[event_id]
        labels = {int(row["label"]) for row in event_rows}
        if len(labels) != 1:
            raise ValueError(f"sit-stand event has inconsistent labels: {event_id}")
        probability = float(np.mean([float(row["probability"]) for row in event_rows]))
        first = event_rows[0]
        outputs.append(
            {
                "event_id": event_id,
                "video_id": str(first["video_id"]),
                "split_group_id": str(first["split_group_id"]),
                "source_group_id": str(first["source_group_id"]),
                "dataset": str(first["dataset"]),
                "action_id": str(first["action_id"]),
                "label": next(iter(labels)),
                "probability": probability,
                "predicted_label": int(probability >= threshold),
                "rule_score": float(
                    np.max([float(row["rule_score"]) for row in event_rows])
                ),
                "window_count": len(event_rows),
            }
        )
    return outputs


def _aggregate_rule_predictions(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            **dict(row),
            "probability": float(row["rule_score"]),
            "predicted_label": int(float(row["rule_score"]) >= 0.5),
        }
        for row in _aggregate_event_predictions(rows, 0.5)
    ]


def _binary_metrics(
    rows: Sequence[Mapping[str, Any]], threshold: float
) -> dict[str, Any]:
    if not rows:
        raise ValueError("sit-stand metrics require non-empty predictions")
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    scores = np.asarray([float(row["probability"]) for row in rows], dtype=np.float64)
    predictions = (scores >= threshold).astype(np.int64)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    both_classes = set(labels.tolist()) == {0, 1}
    return {
        "count": len(rows),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)) if both_classes else None,
        "pr_auc": float(average_precision_score(labels, scores)) if both_classes else None,
        "brier": float(brier_score_loss(labels, scores)),
        "log_loss": float(log_loss(labels, scores, labels=[0, 1])),
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def _group_rows(
    rows: Sequence[Mapping[str, Any]], field: str
) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(field, "unknown"))].append(row)
    return dict(grouped)


def _validate_checkpoint_payload(
    payload: Mapping[str, Any],
    metadata: Mapping[str, Any],
    config: SitStandLogisticConfig,
) -> None:
    if payload.get("schema_version") != "sit-stand-logistic-checkpoint-v1":
        raise ValueError("unsupported sit-stand resume checkpoint")
    if payload.get("task") != metadata["task"]:
        raise ValueError("sit-stand resume checkpoint task mismatch")
    if payload.get("dataset_sha256") != metadata["dataset_sha256"]:
        raise ValueError("sit-stand resume checkpoint dataset SHA-256 mismatch")
    if payload.get("feature_names") != list(SIT_STAND_TABULAR_FEATURES):
        raise ValueError("sit-stand resume checkpoint feature contract mismatch")
    if payload.get("training_config") != asdict(config):
        raise ValueError("sit-stand resume checkpoint training config mismatch")
    if "model" not in payload:
        raise ValueError("sit-stand resume checkpoint has no model")


def _code_fingerprint() -> dict[str, Any]:
    paths = [
        Path(__file__),
        Path(__file__).with_name("sit_stand.py"),
        Path(__file__).with_name("sit_stand_training.py"),
    ]
    digest = hashlib.sha256()
    entries: dict[str, str] = {}
    for path in sorted(paths):
        file_digest = _sha256_file(path)
        entries[path.as_posix()] = file_digest
        digest.update(path.name.encode("utf-8"))
        digest.update(file_digest.encode("ascii"))
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        git_dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        git_commit = None
        git_dirty = None
    return {
        "sha256": digest.hexdigest(),
        "files": entries,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
    }


def _input_contract(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "joint_order": list(SIT_STAND_JOINTS),
        "channel_order": list(SIT_STAND_CHANNELS),
        "preparation_config": metadata["preparation_config"],
        "normalization_contract": metadata["normalization_contract"],
    }


def _data_provenance(
    metadata: Mapping[str, Any], metadata_path: Path
) -> dict[str, Any]:
    return {
        "dataset_sha256": metadata["dataset_sha256"],
        "metadata_sha256": _sha256_file(metadata_path),
        "input_sha256": dict(metadata["input_sha256"]),
        "audit_sha256": metadata["audit_sha256"],
        "source_split_id": metadata["source_split_id"],
        "derived_split_sha256": metadata["derived_split_sha256"],
        "derived_assignments_sha256": metadata["derived_assignments_sha256"],
        "samples_sha256": metadata["samples_sha256"],
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_joblib_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    partial = path.with_suffix(f"{path.suffix}.part")
    joblib.dump(payload, partial)
    os.replace(partial, path)


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    partial = path.with_suffix(f"{path.suffix}.part")
    with partial.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(partial, path)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    partial = path.with_suffix(f"{path.suffix}.part")
    partial.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)
