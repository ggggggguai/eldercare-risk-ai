from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SUPPORTED_GAIT_BASELINES = ("rule", "lightgbm", "ebm")


@dataclass(frozen=True)
class GaitTabularTrainingConfig:
    models: tuple[str, ...] = SUPPORTED_GAIT_BASELINES
    seed: int = 42
    lightgbm_estimators: int = 200
    lightgbm_learning_rate: float = 0.03
    ebm_max_rounds: int = 500
    ebm_outer_bags: int = 8

    def __post_init__(self) -> None:
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

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    metrics_path = destination / "metrics.json"
    output_paths = [metrics_path]
    for model_name in training.models:
        output_paths.append(destination / f"{model_name}_test_window_predictions.jsonl")
        if model_name != "rule":
            output_paths.append(destination / f"{model_name}_model.joblib")
    for path in output_paths:
        if path.exists() and not overwrite:
            raise FileExistsError(f"gait baseline output already exists: {path}")

    labels = arrays["labels"].astype(np.int64, copy=False)
    features = arrays["tabular_features"].astype(np.float32, copy=False)
    partitions = arrays["partitions"].astype(str)
    sample_ids = arrays["sample_ids"].astype(str)
    indices = {
        partition: np.flatnonzero(partitions == partition)
        for partition in ("train", "validation", "test")
    }
    model_reports: dict[str, Any] = {}
    for model_name in training.models:
        if model_name == "rule":
            validation_scores = arrays["rule_scores"][indices["validation"]]
            test_scores = arrays["rule_scores"][indices["test"]]
            model = None
            feature_importance = None
        else:
            model = _fit_model(
                model_name,
                features[indices["train"]],
                labels[indices["train"]],
                training,
                dependencies,
            )
            validation_scores = model.predict_proba(
                features[indices["validation"]]
            )[:, 1]
            test_scores = model.predict_proba(features[indices["test"]])[:, 1]
            model_path = destination / f"{model_name}_model.joblib"
            _write_joblib_atomic(model_path, model, dependencies["joblib"])
            feature_importance = _feature_importance(
                model_name,
                model,
                metadata["tabular_feature_names"],
            )

        prediction_rows = _prediction_rows(
            model_name,
            sample_ids[indices["test"]],
            labels[indices["test"]],
            test_scores,
        )
        _write_jsonl_atomic(
            destination / f"{model_name}_test_window_predictions.jsonl",
            prediction_rows,
        )
        model_reports[model_name] = {
            "validation": _binary_metrics(
                labels[indices["validation"]],
                validation_scores,
                dependencies["sklearn_metrics"],
            ),
            "test": _binary_metrics(
                labels[indices["test"]],
                test_scores,
                dependencies["sklearn_metrics"],
            ),
            "feature_importance": feature_importance,
        }

    report = {
        "schema_version": "gait-tabular-baselines-report-v1",
        "task": metadata["task"],
        "dataset_path": source_path.as_posix(),
        "dataset_sha256": metadata["dataset_sha256"],
        "training_config": asdict(training),
        "feature_names": metadata["tabular_feature_names"],
        "models": model_reports,
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
        "test_metrics": {
            name: model_reports[name]["test"] for name in training.models
        },
    }


def _validate_arrays(arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any]) -> None:
    length = len(arrays["labels"])
    for name in ("partitions", "sample_ids", "tabular_features", "rule_scores"):
        if len(arrays[name]) != length:
            raise ValueError("prepared gait tabular arrays have inconsistent lengths")
    if arrays["tabular_features"].ndim != 2:
        raise ValueError("tabular_features must have shape [N, F]")
    if arrays["tabular_features"].shape[1] != len(metadata["tabular_feature_names"]):
        raise ValueError("tabular feature names do not match the feature matrix")
    if not np.isfinite(arrays["tabular_features"]).all():
        raise ValueError("tabular gait features contain non-finite values")
    if not np.isfinite(arrays["rule_scores"]).all():
        raise ValueError("rule gait scores contain non-finite values")
    labels = arrays["labels"].astype(np.int64)
    partitions = arrays["partitions"].astype(str)
    for partition in ("train", "validation", "test"):
        partition_labels = set(labels[partitions == partition].tolist())
        if partition_labels != {0, 1}:
            raise ValueError(f"prepared gait {partition} partition lacks a binary class")


def _load_dependencies(models: Sequence[str]) -> dict[str, Any]:
    try:
        import joblib
        from sklearn import metrics as sklearn_metrics
        from sklearn.utils.class_weight import compute_sample_weight
    except ImportError as exc:
        raise RuntimeError("tabular gait baselines require the project ml dependencies") from exc

    dependencies: dict[str, Any] = {
        "joblib": joblib,
        "sklearn_metrics": sklearn_metrics,
        "compute_sample_weight": compute_sample_weight,
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
    config: GaitTabularTrainingConfig,
    dependencies: Mapping[str, Any],
) -> Any:
    sample_weight = dependencies["compute_sample_weight"](
        class_weight="balanced", y=labels
    )
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
    if model_name == "lightgbm":
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
) -> list[dict[str, Any]]:
    return [
        {
            "sample_id": str(sample_id),
            "label": int(label),
            "gait_risk_score": float(score),
            "predicted_label": int(float(score) >= 0.5),
            "model_version": f"gait-{model_name}-baseline-v1",
        }
        for sample_id, label, score in zip(sample_ids, labels, scores, strict=True)
    ]


def _binary_metrics(
    labels: Sequence[int],
    scores: Sequence[float],
    metrics: Any,
) -> dict[str, Any]:
    y_true = np.asarray(labels, dtype=np.int64)
    y_score = np.clip(np.asarray(scores, dtype=np.float64), 0.0, 1.0)
    y_pred = (y_score >= 0.5).astype(np.int64)
    confusion = metrics.confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in confusion.ravel())
    return {
        "sample_count": len(y_true),
        "accuracy": float(metrics.accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(metrics.balanced_accuracy_score(y_true, y_pred)),
        "precision": float(metrics.precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(metrics.recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(metrics.f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(metrics.roc_auc_score(y_true, y_score)),
        "average_precision": float(metrics.average_precision_score(y_true, y_score)),
        "brier_score": float(metrics.brier_score_loss(y_true, y_score)),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "threshold": 0.5,
    }


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
