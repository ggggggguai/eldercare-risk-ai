from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

from elderly_monitoring.modules.fall_risk.near_fall_tabular import (
    MODEL_VERSION,
    SCHEMA_VERSION,
    extract_near_fall_tabular_features,
    sha256_file,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the small-data near-fall ExtraTrees candidate.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-validation-recall", type=float, default=0.90)
    return parser


def train(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.data, args.metadata, args.samples):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    if metadata.get("dataset_sha256") != sha256_file(args.data):
        raise ValueError("near-fall tabular dataset SHA-256 mismatch")
    if metadata.get("training_ready") is not True:
        raise ValueError("near-fall tabular dataset is not training ready")
    if metadata.get("test_pose_read") is not False or metadata.get("test_evaluated") is not False:
        raise ValueError("near-fall tabular dataset violates test lock")
    with np.load(args.data, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    samples = _read_jsonl(args.samples)
    if len(samples) != len(arrays["features"]):
        raise ValueError("near-fall tabular samples/array length mismatch")

    features = np.stack(
        [extract_near_fall_tabular_features(tensor) for tensor in arrays["features"]]
    )
    labels = np.asarray(arrays["labels"], dtype=np.int64)
    partitions = np.asarray(arrays["partitions"]).astype(str)
    train_indices = np.flatnonzero(partitions == "train")
    validation_indices = np.flatnonzero(partitions == "validation")
    model = ExtraTreesClassifier(
        n_estimators=500,
        max_depth=5,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=args.seed,
        n_jobs=1,
    )
    model.fit(
        features[train_indices],
        labels[train_indices],
        sample_weight=np.asarray(arrays["loss_weights"], dtype=np.float32)[train_indices],
    )
    validation_scores = model.predict_proba(features[validation_indices])[:, 1]
    event_labels, event_scores, event_rows = _aggregate_events(
        validation_indices,
        validation_scores,
        labels,
        samples,
    )
    threshold, threshold_rows = _select_threshold(
        event_labels,
        event_scores,
        min_recall=args.min_validation_recall,
    )
    validation_prediction = event_scores >= threshold
    precision, recall, f1, _ = precision_recall_fscore_support(
        event_labels,
        validation_prediction,
        average="binary",
        zero_division=0,
    )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = args.output_dir / "best_model.joblib"
    checkpoint = {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "model": model,
        "validation_threshold": threshold,
        "input_contract": {
            "window_frames": int(arrays["features"].shape[1]),
            "feature_count": int(features.shape[1]),
            "normalization": {
                "mean": np.asarray(arrays["normalization_mean"]).tolist(),
                "std": np.asarray(arrays["normalization_std"]).tolist(),
            },
        },
        "training_contract": {
            "seed": args.seed,
            "dataset_sha256": sha256_file(args.data),
            "metadata_sha256": sha256_file(args.metadata),
            "samples_sha256": sha256_file(args.samples),
            "train_only_fit": True,
            "validation_only_threshold_selection": True,
            "minimum_validation_recall": args.min_validation_recall,
        },
    }
    temporary = checkpoint_path.with_name(f".{checkpoint_path.name}.{os.getpid()}.tmp")
    try:
        joblib.dump(checkpoint, temporary)
        os.replace(temporary, checkpoint_path)
    finally:
        temporary.unlink(missing_ok=True)
    _write_jsonl(args.output_dir / "validation_predictions.jsonl", event_rows)
    metrics = {
        "schema_version": "near-fall-tabular-training-metrics-v1",
        "status": "development_provisional",
        "seed": args.seed,
        "validation": {
            "event_count": len(event_labels),
            "positive_count": int(np.sum(event_labels)),
            "negative_count": int(len(event_labels) - np.sum(event_labels)),
            "threshold": threshold,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "pr_auc": float(average_precision_score(event_labels, event_scores)),
            "roc_auc": float(roc_auc_score(event_labels, event_scores)),
        },
        "threshold_curve": threshold_rows,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "test_evaluated": False,
    }
    _write_json(args.output_dir / "metrics.json", metrics)
    return {
        "checkpoint_path": checkpoint_path.as_posix(),
        "metrics_path": (args.output_dir / "metrics.json").as_posix(),
        "validation_threshold": threshold,
        "validation_f1": float(f1),
    }


def _aggregate_events(
    indices: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    samples: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    event_labels: dict[str, int] = {}
    for index, score in zip(indices, scores, strict=True):
        event_id = str(samples[int(index)]["event_id"])
        grouped[event_id].append(float(score))
        event_labels[event_id] = int(labels[int(index)])
    event_ids = sorted(grouped)
    output_labels = np.asarray([event_labels[event_id] for event_id in event_ids])
    output_scores = np.asarray([max(grouped[event_id]) for event_id in event_ids])
    rows = [
        {
            "event_id": event_id,
            "label": int(event_labels[event_id]),
            "score": float(max(grouped[event_id])),
        }
        for event_id in event_ids
    ]
    return output_labels, output_scores, rows


def _select_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    min_recall: float = 0.90,
) -> tuple[float, list[dict[str, float]]]:
    if not 0.0 < min_recall <= 1.0:
        raise ValueError("minimum validation recall must be within (0, 1]")
    candidates = np.unique(np.concatenate(([0.001, 0.5, 0.999], scores)))
    rows: list[dict[str, float]] = []
    for threshold in candidates:
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels,
            scores >= threshold,
            average="binary",
            zero_division=0,
        )
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        )
    eligible = [row for row in rows if row["recall"] >= min_recall - 1e-12]
    if not eligible:
        raise ValueError("no validation threshold satisfies the recall constraint")
    selected = max(
        eligible,
        key=lambda row: (row["precision"], row["f1"], row["threshold"]),
    )
    return float(selected["threshold"]), rows


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = train(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}")
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
