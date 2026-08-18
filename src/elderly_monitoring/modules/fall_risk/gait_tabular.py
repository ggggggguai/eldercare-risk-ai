from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from elderly_monitoring.modules.fall_risk.gait_training import (
    validate_frozen_partition_selection,
)
from elderly_monitoring.modules.fall_risk.gait_tcn import (
    _apply_source_balance,
    _expected_calibration_error,
    _metrics_by_group_field,
    _normal_false_positives_per_hour,
    aggregate_group_predictions,
    compute_balanced_class_weights,
    select_balanced_accuracy_threshold,
)


SUPPORTED_GAIT_BASELINES = ("rule", "logistic", "lightgbm", "ebm")
GAIT_TABULAR_FEATURE_PROFILES = (
    "all",
    "exclude_quality",
    "exclude_quality_and_scale",
)
_GAIT_QUALITY_FEATURES = {
    "usable_frame_ratio",
    "gait_keypoint_coverage",
    "mean_core_keypoint_quality",
    "interpolated_point_ratio",
    "jump_outlier_per_frame",
}


@dataclass(frozen=True)
class GaitTabularTrainingConfig:
    models: tuple[str, ...] = SUPPORTED_GAIT_BASELINES
    seed: int = 42
    lightgbm_estimators: int = 200
    lightgbm_learning_rate: float = 0.03
    ebm_max_rounds: int = 500
    ebm_outer_bags: int = 8
    logistic_c: float = 1.0
    logistic_max_iter: int = 2000
    evaluate_test: bool | None = None
    partition_scheme: str = "frozen"
    feature_profile: str = "all"

    def __post_init__(self) -> None:
        if self.evaluate_test is True:
            raise ValueError(
                "gait training cannot evaluate test; use the release-gated evaluator"
            )
        if not self.models:
            raise ValueError("at least one gait baseline model is required")
        unknown = sorted(set(self.models) - set(SUPPORTED_GAIT_BASELINES))
        if unknown:
            raise ValueError(f"unsupported gait baseline models: {unknown}")
        if self.lightgbm_estimators < 1 or self.ebm_max_rounds < 1:
            raise ValueError("boosting rounds must be positive")
        if self.lightgbm_learning_rate <= 0:
            raise ValueError("lightgbm_learning_rate must be positive")
        if self.ebm_outer_bags < 1:
            raise ValueError("ebm_outer_bags must be positive")
        if self.logistic_c <= 0 or self.logistic_max_iter < 1:
            raise ValueError("logistic_c and logistic_max_iter must be positive")
        if self.partition_scheme not in {"frozen", "fold_a", "fold_b"}:
            raise ValueError("partition_scheme must be frozen, fold_a or fold_b")
        if self.feature_profile not in GAIT_TABULAR_FEATURE_PROFILES:
            raise ValueError(
                "feature_profile must be all, exclude_quality or "
                "exclude_quality_and_scale"
            )


def train_gait_tabular_baselines(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    metadata_path: str | Path | None = None,
    config: GaitTabularTrainingConfig | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    training = config or GaitTabularTrainingConfig()
    source_path = Path(dataset_path)
    source_metadata_path = (
        Path(metadata_path)
        if metadata_path is not None
        else source_path.with_name("metadata.json")
    )
    metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    if metadata.get("dataset_sha256") != _sha256_file(source_path):
        raise ValueError("prepared gait dataset SHA-256 does not match metadata")
    if metadata.get("task") != "gait_instability_vs_normal_activity":
        raise ValueError("tabular gait baselines require gait_instability window labels")

    with np.load(source_path, allow_pickle=False) as archive:
        required = {
            "labels",
            "partitions",
            "sample_ids",
            "tabular_features",
            "rule_scores",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"prepared gait dataset is missing arrays: {missing}")
        arrays = {name: archive[name] for name in archive.files}
    _validate_arrays(arrays, metadata)
    dependencies = _load_dependencies(training.models)
    _validate_dataset_protocol(metadata)
    evaluate_test = (
        training.evaluate_test
        if training.evaluate_test is not None
        else metadata.get("source_split_id") is None
    )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    metrics_path = destination / "metrics.json"
    output_paths = [metrics_path]
    for model_name in training.models:
        output_paths.append(
            destination / f"{model_name}_validation_action_segment_predictions.jsonl"
        )
        if evaluate_test:
            output_paths.append(destination / f"{model_name}_test_window_predictions.jsonl")
        if model_name != "rule":
            output_paths.append(destination / f"{model_name}_model.joblib")
    for path in output_paths:
        if path.exists() and not overwrite:
            raise FileExistsError(f"gait baseline output already exists: {path}")

    labels = arrays["labels"].astype(np.int64, copy=False)
    features = arrays["tabular_features"].astype(np.float32, copy=False)
    feature_names = [str(name) for name in metadata["tabular_feature_names"]]
    if training.feature_profile != "all":
        excluded_features = set(_GAIT_QUALITY_FEATURES)
        if training.feature_profile == "exclude_quality_and_scale":
            excluded_features.add("hip_width_mean")
        selected_feature_indices = [
            index
            for index, name in enumerate(feature_names)
            if name not in excluded_features
        ]
        if not selected_feature_indices:
            raise ValueError("exclude_quality removed every tabular gait feature")
        features = features[:, selected_feature_indices]
        feature_names = [feature_names[index] for index in selected_feature_indices]
    partition_schemes = metadata.get("partition_schemes", {"frozen": "partitions"})
    if training.partition_scheme not in partition_schemes:
        raise ValueError(
            f"prepared dataset does not provide partition scheme {training.partition_scheme}"
        )
    partition_array_name = str(partition_schemes[training.partition_scheme])
    if partition_array_name not in arrays:
        raise ValueError(f"prepared dataset is missing {partition_array_name}")
    partitions = arrays[partition_array_name].astype(str)
    if len(partitions) != len(labels):
        raise ValueError(f"prepared gait {partition_array_name} has inconsistent length")
    if metadata.get("split_protocol") == "frozen_training_labels_v3":
        validate_frozen_partition_selection(arrays["partitions"], partitions)
    sample_ids = arrays["sample_ids"].astype(str)
    sample_weights = arrays.get(
        "sample_weights", np.ones(len(labels), dtype=np.float32)
    ).astype(np.float32, copy=False)
    primary_evaluation_mask = arrays.get(
        "primary_evaluation_mask", np.ones(len(labels), dtype=np.uint8)
    ).astype(bool, copy=False)
    sensitivity_evaluation_mask = arrays.get(
        "sensitivity_evaluation_mask", np.zeros(len(labels), dtype=np.uint8)
    ).astype(bool, copy=False)
    all_partition_indices = {
        partition: np.flatnonzero(partitions == partition)
        for partition in ("train", "validation", "test")
    }
    # Align the validation evaluation with the gait TCN protocol: the main
    # validation口径 uses only primary-evaluation windows (>=10 labeled
    # observations in the annotation span). Weak-context windows are reported
    # as a separate sensitivity口径 and never drive threshold selection. When
    # the prepared dataset has no such masks, all validation windows are used,
    # preserving the legacy behavior.
    indices = {
        # Match the gait TCN: zero-weight windows (audit-only and
        # representation-only evidence tiers) never enter training.
        "train": all_partition_indices["train"][
            sample_weights[all_partition_indices["train"]] > 0
        ],
        "validation": all_partition_indices["validation"][
            primary_evaluation_mask[all_partition_indices["validation"]]
        ],
        "validation_sensitivity": all_partition_indices["validation"][
            sensitivity_evaluation_mask[all_partition_indices["validation"]]
        ],
        "test": all_partition_indices["test"],
    }
    required_partitions = ["train", "validation"] + (["test"] if evaluate_test else [])
    for partition in required_partitions:
        partition_indices = indices[partition]
        if len(partition_indices) == 0:
            raise ValueError(f"prepared gait {partition} partition is empty")
        if set(labels[partition_indices].tolist()) != {0, 1}:
            raise ValueError(f"prepared gait {partition} partition lacks a binary class")
    source_balance_factors: dict[str, float] = {}
    if "datasets" in arrays:
        sample_weights, source_balance_factors = _apply_source_balance(
            sample_weights,
            arrays["datasets"].astype(str),
            indices["train"],
        )
    class_weights = compute_balanced_class_weights(
        labels,
        sample_weights,
        indices["train"],
    )
    model_reports: dict[str, Any] = {}
    for model_name in training.models:
        if model_name == "rule":
            validation_scores = arrays["rule_scores"][indices["validation"]]
            validation_sensitivity_scores = arrays["rule_scores"][
                indices["validation_sensitivity"]
            ]
            test_scores = (
                arrays["rule_scores"][indices["test"]] if evaluate_test else None
            )
            model = None
            feature_importance = None
        else:
            model = _fit_model(
                model_name,
                features[indices["train"]],
                labels[indices["train"]],
                sample_weights[indices["train"]],
                class_weights,
                training,
                dependencies,
            )
            validation_scores = model.predict_proba(
                features[indices["validation"]]
            )[:, 1]
            validation_sensitivity_scores = (
                model.predict_proba(features[indices["validation_sensitivity"]])[:, 1]
                if len(indices["validation_sensitivity"])
                else np.empty(0, dtype=np.float32)
            )
            test_scores = (
                model.predict_proba(features[indices["test"]])[:, 1]
                if evaluate_test
                else None
            )
            model_path = destination / f"{model_name}_model.joblib"
            _write_joblib_atomic(model_path, model, dependencies["joblib"])
            feature_importance = _feature_importance(
                model_name,
                model,
                feature_names,
            )

        validation_rows = _aggregate_rows(
            arrays,
            indices["validation"],
            validation_scores,
            threshold=0.5,
        )
        threshold_source = validation_rows or [
            {"label": int(label), "probability": float(score)}
            for label, score in zip(
                labels[indices["validation"]], validation_scores, strict=True
            )
        ]
        threshold = select_balanced_accuracy_threshold(
            [int(row["label"]) for row in threshold_source],
            [float(row["probability"]) for row in threshold_source],
        )
        validation_rows = _aggregate_rows(
            arrays,
            indices["validation"],
            validation_scores,
            threshold=threshold,
        )
        validation_sensitivity_rows = _aggregate_rows(
            arrays,
            indices["validation_sensitivity"],
            validation_sensitivity_scores,
            threshold=threshold,
        )
        _write_jsonl_atomic(
            destination
            / f"{model_name}_validation_action_segment_predictions.jsonl",
            validation_rows,
        )
        test_rows = (
            _aggregate_rows(arrays, indices["test"], test_scores, threshold=threshold)
            if test_scores is not None
            else []
        )
        if test_scores is not None:
            prediction_rows = _prediction_rows(
                model_name,
                sample_ids[indices["test"]],
                labels[indices["test"]],
                test_scores,
                threshold=threshold,
            )
            _write_jsonl_atomic(
                destination / f"{model_name}_test_window_predictions.jsonl",
                prediction_rows,
            )
        model_reports[model_name] = {
            "selected_threshold": threshold,
            "validation": _binary_metrics(
                labels[indices["validation"]],
                validation_scores,
                dependencies["sklearn_metrics"],
                threshold=threshold,
            ),
            "validation_action_segment": (
                _binary_metrics(
                    [int(row["label"]) for row in validation_rows],
                    [float(row["probability"]) for row in validation_rows],
                    dependencies["sklearn_metrics"],
                    threshold=threshold,
                )
                if validation_rows
                else None
            ),
            "validation_dataset_metrics": _metrics_by_group_field(
                validation_rows, "dataset", threshold
            ),
            "validation_action_metrics": _action_error_metrics(validation_rows),
            "validation_normal_false_positives_per_hour": (
                _normal_false_positives_per_hour(validation_rows)
            ),
            "validation_sensitivity_action_segment": (
                _binary_metrics(
                    [int(row["label"]) for row in validation_sensitivity_rows],
                    [float(row["probability"]) for row in validation_sensitivity_rows],
                    dependencies["sklearn_metrics"],
                    threshold=threshold,
                )
                if validation_sensitivity_rows
                else None
            ),
            "validation_sensitivity_dataset_metrics": (
                _metrics_by_group_field(
                    validation_sensitivity_rows, "dataset", threshold
                )
                if validation_sensitivity_rows
                else None
            ),
            "test": (
                _binary_metrics(
                    labels[indices["test"]],
                    test_scores,
                    dependencies["sklearn_metrics"],
                    threshold=threshold,
                )
                if test_scores is not None
                else None
            ),
            "test_action_segment": (
                _binary_metrics(
                    [int(row["label"]) for row in test_rows],
                    [float(row["probability"]) for row in test_rows],
                    dependencies["sklearn_metrics"],
                    threshold=threshold,
                )
                if test_rows
                else None
            ),
            "test_dataset_metrics": (
                _metrics_by_group_field(test_rows, "dataset", threshold)
                if test_rows
                else None
            ),
            "test_normal_false_positives_per_hour": (
                _normal_false_positives_per_hour(test_rows) if test_rows else None
            ),
            "feature_importance": feature_importance,
        }

    report = {
        "schema_version": "gait-tabular-baselines-report-v1",
        "task": metadata["task"],
        "dataset_path": source_path.as_posix(),
        "dataset_sha256": metadata["dataset_sha256"],
        "training_config": asdict(training),
        "feature_names": feature_names,
        "feature_profile": training.feature_profile,
        "evaluation_protocol": {
            "validation_scope": "primary_evaluation_mask",
            "validation_sensitivity_scope": "sensitivity_evaluation_mask",
            "validation_window_count": int(len(indices["validation"])),
            "validation_sensitivity_window_count": int(
                len(indices["validation_sensitivity"])
            ),
        },
        "models": model_reports,
        "test_evaluated": evaluate_test,
        "test_pose_read": bool(metadata.get("test_pose_read", False)),
        "protocol_status": metadata.get("protocol_status", "development_provisional"),
        "source_split_id": metadata.get("source_split_id"),
        "input_sha256": dict(metadata.get("input_sha256", {})),
        "partition_scheme": training.partition_scheme,
        "source_balance_factors": source_balance_factors,
        "class_weights": class_weights.tolist(),
        "target_contract": metadata.get("target_contract"),
        "split_group": metadata.get("split_group"),
        "split_is_provisional": metadata.get("split_is_provisional"),
        "limitations": list(metadata.get("split_limitations", []))
        + ["rule scores and model probabilities have not been calibrated"],
    }
    _write_json_atomic(metrics_path, report)
    return {
        "output_dir": destination.as_posix(),
        "metrics_path": metrics_path.as_posix(),
        "models": list(training.models),
        "test_metrics": (
            {name: model_reports[name]["test"] for name in training.models}
            if evaluate_test
            else None
        ),
    }


def _validate_dataset_protocol(metadata: Mapping[str, Any]) -> None:
    if metadata.get("test_pose_read") is True or metadata.get("test_tensor_generated") is True:
        raise ValueError("prepared gait training dataset contains test-derived inputs")
    source_split_id = metadata.get("source_split_id")
    source_split_path = metadata.get("source_split_report_path")
    if source_split_id is None:
        return
    if not isinstance(source_split_path, str) or not source_split_path:
        raise ValueError("prepared gait metadata is missing source split report path")
    candidate = Path(source_split_path)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    if not candidate.is_file():
        raise FileNotFoundError(f"prepared gait source split report not found: {candidate}")
    split_report = json.loads(candidate.read_text(encoding="utf-8"))
    if split_report.get("split_id") != source_split_id:
        raise ValueError("prepared gait dataset source_split_id is stale")
    expected_hash = metadata.get("source_split_report_sha256")
    if expected_hash != _sha256_file(candidate):
        raise ValueError("prepared gait source split report SHA-256 is stale")


def _validate_arrays(arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any]) -> None:
    length = len(arrays["labels"])
    if metadata.get("source_split_id") is not None and "test" in {
        str(value) for value in np.unique(arrays["partitions"])
    }:
        raise ValueError("current provisional gait dataset must not contain test tensors")
    for name in ("partitions", "sample_ids", "tabular_features", "rule_scores"):
        if len(arrays[name]) != length:
            raise ValueError("prepared gait tabular arrays have inconsistent lengths")
    for name in (
        "sample_weights",
        "action_segment_ids",
        "datasets",
        "source_group_ids",
        "segment_durations_sec",
        "action_ids",
    ):
        if name in arrays and len(arrays[name]) != length:
            raise ValueError(f"prepared gait {name} has inconsistent length")
    if arrays["tabular_features"].ndim != 2:
        raise ValueError("tabular_features must have shape [N, F]")
    if arrays["tabular_features"].shape[1] != len(metadata["tabular_feature_names"]):
        raise ValueError("tabular feature names do not match the feature matrix")
    if not np.isfinite(arrays["tabular_features"]).all():
        raise ValueError("tabular gait features contain non-finite values")
    if not np.isfinite(arrays["rule_scores"]).all():
        raise ValueError("rule gait scores contain non-finite values")


def _load_dependencies(models: Sequence[str]) -> dict[str, Any]:
    try:
        import joblib
        from sklearn import metrics as sklearn_metrics
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RuntimeError("tabular gait baselines require the project ml dependencies") from exc

    dependencies: dict[str, Any] = {
        "joblib": joblib,
        "sklearn_metrics": sklearn_metrics,
        "LogisticRegression": LogisticRegression,
        "Pipeline": Pipeline,
        "StandardScaler": StandardScaler,
    }
    if "lightgbm" in models:
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise RuntimeError("LightGBM baseline requires lightgbm") from exc
        dependencies["LGBMClassifier"] = LGBMClassifier
    if "ebm" in models:
        try:
            from interpret.glassbox import ExplainableBoostingClassifier
        except ImportError as exc:
            raise RuntimeError(
                "EBM baseline requires the optional interpret dependency"
            ) from exc
        dependencies["ExplainableBoostingClassifier"] = ExplainableBoostingClassifier
    return dependencies


def _fit_model(
    model_name: str,
    features: np.ndarray,
    labels: np.ndarray,
    prepared_weights: np.ndarray,
    class_weights: np.ndarray,
    config: GaitTabularTrainingConfig,
    dependencies: Mapping[str, Any],
) -> Any:
    sample_weight = prepared_weights * class_weights[labels]
    if model_name == "logistic":
        model = dependencies["Pipeline"](
            [
                ("scaler", dependencies["StandardScaler"]()),
                (
                    "classifier",
                    dependencies["LogisticRegression"](
                        C=config.logistic_c,
                        max_iter=config.logistic_max_iter,
                        random_state=config.seed,
                        solver="lbfgs",
                    ),
                ),
            ]
        )
        model.fit(
            features,
            labels,
            scaler__sample_weight=sample_weight,
            classifier__sample_weight=sample_weight,
        )
        return model
    if model_name == "lightgbm":
        model = dependencies["LGBMClassifier"](
            objective="binary",
            n_estimators=config.lightgbm_estimators,
            learning_rate=config.lightgbm_learning_rate,
            max_depth=3,
            num_leaves=7,
            min_child_samples=5,
            subsample=1.0,
            colsample_bytree=1.0,
            reg_lambda=1.0,
            random_state=config.seed,
            n_jobs=1,
            verbosity=-1,
        )
    elif model_name == "ebm":
        model = dependencies["ExplainableBoostingClassifier"](
            interactions=0,
            max_bins=64,
            max_rounds=config.ebm_max_rounds,
            outer_bags=config.ebm_outer_bags,
            validation_size=0.15,
            early_stopping_rounds=50,
            random_state=config.seed,
            n_jobs=1,
        )
    else:
        raise ValueError(f"cannot fit unsupported gait model: {model_name}")
    model.fit(features, labels, sample_weight=sample_weight)
    return model


def _feature_importance(
    model_name: str,
    model: Any,
    feature_names: Sequence[str],
) -> list[dict[str, Any]] | None:
    if model_name == "logistic":
        classifier = model.named_steps["classifier"]
        values = np.abs(np.asarray(classifier.coef_[0], dtype=np.float64))
    elif model_name == "lightgbm":
        values = np.asarray(model.feature_importances_, dtype=np.float64)
    elif model_name == "ebm":
        values = np.asarray(model.term_importances(), dtype=np.float64)
        values = values[: len(feature_names)]
    else:
        return None
    rows = [
        {"feature": str(name), "importance": float(value)}
        for name, value in zip(feature_names, values, strict=False)
    ]
    return sorted(rows, key=lambda row: (-row["importance"], row["feature"]))


def _prediction_rows(
    model_name: str,
    sample_ids: Sequence[str],
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    threshold: float,
) -> list[dict[str, Any]]:
    return [
        {
            "sample_id": str(sample_id),
            "label": int(label),
            "gait_risk_score": float(score),
            "predicted_label": int(float(score) >= threshold),
            "model_version": f"gait-{model_name}-baseline-v1",
        }
        for sample_id, label, score in zip(sample_ids, labels, scores, strict=True)
    ]


def _binary_metrics(
    labels: Sequence[int],
    scores: Sequence[float],
    metrics: Any,
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    y_true = np.asarray(labels, dtype=np.int64)
    y_score = np.clip(np.asarray(scores, dtype=np.float64), 0.0, 1.0)
    y_pred = (y_score >= threshold).astype(np.int64)
    confusion = metrics.confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in confusion.ravel())
    return {
        "sample_count": len(y_true),
        "accuracy": float(metrics.accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(metrics.balanced_accuracy_score(y_true, y_pred)),
        "precision": float(metrics.precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(metrics.recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "f1": float(metrics.f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(metrics.roc_auc_score(y_true, y_score)),
        "average_precision": float(metrics.average_precision_score(y_true, y_score)),
        "brier_score": float(metrics.brier_score_loss(y_true, y_score)),
        "log_loss": float(metrics.log_loss(y_true, y_score, labels=[0, 1])),
        "ece": _expected_calibration_error(y_true, y_score),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "threshold": threshold,
    }


def _aggregate_rows(
    arrays: Mapping[str, np.ndarray],
    indices: np.ndarray,
    scores: Sequence[float] | None,
    *,
    threshold: float,
) -> list[dict[str, Any]]:
    if scores is None or "action_segment_ids" not in arrays:
        return []
    rows = aggregate_group_predictions(
        group_ids=arrays["action_segment_ids"][indices].astype(str),
        labels=arrays["labels"][indices].astype(np.int64),
        probabilities=scores,
        threshold=threshold,
        group_field="action_segment_id",
    )
    datasets = arrays.get("datasets", np.full(len(arrays["labels"]), "unknown"))
    source_groups = arrays.get(
        "source_group_ids", np.full(len(arrays["labels"]), "unknown")
    )
    durations = arrays.get(
        "segment_durations_sec", np.zeros(len(arrays["labels"]), dtype=np.float32)
    )
    action_ids = arrays.get(
        "action_ids", np.full(len(arrays["labels"]), "unknown")
    )
    video_ids = arrays.get(
        "video_ids", np.full(len(arrays["labels"]), "unknown")
    )
    sample_groups = arrays.get(
        "sample_group_ids", np.full(len(arrays["labels"]), "unknown")
    )
    context: dict[str, dict[str, Any]] = {}
    for (
        segment_id,
        dataset,
        source_group,
        duration,
        action_id,
        video_id,
        sample_group,
    ) in zip(
        arrays["action_segment_ids"][indices],
        datasets[indices],
        source_groups[indices],
        durations[indices],
        action_ids[indices],
        video_ids[indices],
        sample_groups[indices],
        strict=True,
    ):
        key = str(segment_id)
        value = {
            "dataset": str(dataset),
            "source_group_id": str(source_group),
            "duration_sec": float(duration),
            "action_id": str(action_id),
            "video_id": str(video_id),
            "sample_group_id": str(sample_group),
        }
        if key in context and context[key] != value:
            raise ValueError(f"inconsistent dataset for gait action segment {key}")
        context[key] = value
    for row in rows:
        row.update(context[str(row["action_segment_id"])])
    return rows


def _action_error_metrics(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for action_id in sorted({str(row.get("action_id", "unknown")) for row in rows}):
        selected = [row for row in rows if str(row.get("action_id", "unknown")) == action_id]
        labels = [int(row["label"]) for row in selected]
        predicted = [int(row["predicted_label"]) for row in selected]
        scores = [float(row["probability"]) for row in selected]
        output[action_id] = {
            "sample_count": len(selected),
            "label": labels[0] if len(set(labels)) == 1 else None,
            "mean_probability": float(np.mean(scores)),
            "predicted_positive_rate": float(np.mean(predicted)),
            "error_count": sum(label != prediction for label, prediction in zip(labels, predicted, strict=True)),
            "false_positive_count": sum(label == 0 and prediction == 1 for label, prediction in zip(labels, predicted, strict=True)),
            "false_negative_count": sum(label == 1 and prediction == 0 for label, prediction in zip(labels, predicted, strict=True)),
        }
    return output


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_joblib_atomic(path: Path, model: Any, joblib: Any) -> None:
    partial = path.with_suffix(f"{path.suffix}.part")
    joblib.dump(model, partial)
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
