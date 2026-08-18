from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve

from elderly_monitoring.modules.fall_risk.fall_event_continuous_tcn import ContinuousFallTCN


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit development-only continuous fall TCN checkpoints on validation."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = audit_checkpoints(
            args.data,
            args.samples,
            args.checkpoint,
            args.output_dir,
            batch_size=args.batch_size,
            device_name=args.device,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def audit_checkpoints(
    dataset_path: str | Path,
    samples_path: str | Path,
    checkpoint_paths: Sequence[str | Path],
    output_dir: str | Path,
    *,
    batch_size: int = 128,
    device_name: str = "cpu",
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    data_file = Path(dataset_path)
    sample_file = Path(samples_path)
    destination = Path(output_dir)
    with np.load(data_file, allow_pickle=False) as archive:
        features = archive["features"].astype(np.float32)
        targets = archive["presence_targets"].astype(np.int64)
        onset_targets = archive["onset_targets"].astype(np.int64)
        onset_masks = archive["onset_masks"].astype(bool)
        partitions = archive["partitions"].astype(str)
    samples = [json.loads(line) for line in sample_file.read_text(encoding="utf-8").splitlines() if line]
    if len(samples) != len(features):
        raise ValueError("samples.jsonl and dataset.npz row counts differ")
    validation_indices = np.flatnonzero(partitions == "validation")
    if not len(validation_indices):
        raise ValueError("dataset contains no validation samples")
    for index in validation_indices:
        if samples[int(index)].get("partition") != "validation":
            raise ValueError("samples.jsonl partition order differs from dataset.npz")

    dataset_sha256 = _sha256_file(data_file)
    device = _resolve_device(device_name)
    seed_rows: list[dict[str, Any]] = []
    probabilities_by_seed: list[np.ndarray] = []
    onset_predictions_by_seed: list[np.ndarray] = []
    checkpoint_names: list[str] = []
    for checkpoint_path in checkpoint_paths:
        checkpoint_file = Path(checkpoint_path)
        checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
        _validate_checkpoint(checkpoint, dataset_sha256)
        probability, onset_prediction = _predict(
            checkpoint,
            features[validation_indices],
            batch_size=batch_size,
            device=device,
        )
        metrics = _classification_metrics(targets[validation_indices], probability, threshold=0.5)
        supervised = onset_masks[validation_indices]
        metrics["onset_accuracy"] = (
            round(float(np.mean(onset_prediction[supervised] == onset_targets[validation_indices][supervised])), 6)
            if np.any(supervised)
            else None
        )
        metrics["onset_supervised_count"] = int(np.sum(supervised))
        name = checkpoint_file.parent.name
        checkpoint_names.append(name)
        seed_rows.append(
            {
                "checkpoint": checkpoint_file.as_posix(),
                "name": name,
                "best_epoch": int(checkpoint["best_epoch"]),
                **metrics,
            }
        )
        probabilities_by_seed.append(probability)
        onset_predictions_by_seed.append(onset_prediction)

    stacked = np.stack(probabilities_by_seed)
    ensemble_probability = stacked.mean(axis=0)
    validation_targets = targets[validation_indices]
    selected_threshold = _best_f1_threshold(validation_targets, ensemble_probability)
    fixed_metrics = _classification_metrics(validation_targets, ensemble_probability, threshold=0.5)
    selected_metrics = _classification_metrics(
        validation_targets, ensemble_probability, threshold=selected_threshold
    )
    stacked_onset = np.stack(onset_predictions_by_seed)
    ensemble_onset = np.asarray(
        [np.bincount(stacked_onset[:, index]).argmax() for index in range(stacked_onset.shape[1])]
    )
    supervised = onset_masks[validation_indices]
    selected_metrics["onset_accuracy"] = (
        round(
            float(
                np.mean(
                    ensemble_onset[supervised]
                    == onset_targets[validation_indices][supervised]
                )
            ),
            6,
        )
        if np.any(supervised)
        else None
    )
    selected_metrics["onset_supervised_count"] = int(np.sum(supervised))
    aggregate = _aggregate_seed_metrics(seed_rows)
    validation_samples = [samples[int(index)] for index in validation_indices]
    fixed_strata = {
        "dataset": _stratified_metrics(
            validation_samples, validation_targets, ensemble_probability, "dataset", 0.5
        ),
        "supervision_family": _stratified_metrics(
            validation_samples, validation_targets, ensemble_probability, "supervision_family", 0.5
        ),
        "partial_context": _stratified_metrics(
            validation_samples, validation_targets, ensemble_probability, "partial_context", 0.5
        ),
        "target_presence": _stratified_metrics(
            validation_samples, validation_targets, ensemble_probability, "target_presence", 0.5
        ),
    }
    selected_strata = {
        "dataset": _stratified_metrics(
            validation_samples, validation_targets, ensemble_probability, "dataset", selected_threshold
        ),
        "supervision_family": _stratified_metrics(
            validation_samples,
            validation_targets,
            ensemble_probability,
            "supervision_family",
            selected_threshold,
        ),
        "partial_context": _stratified_metrics(
            validation_samples,
            validation_targets,
            ensemble_probability,
            "partial_context",
            selected_threshold,
        ),
        "target_presence": _stratified_metrics(
            validation_samples,
            validation_targets,
            ensemble_probability,
            "target_presence",
            selected_threshold,
        ),
    }
    result = {
        "schema_version": "fall-event-continuous-tcn-validation-audit-v1",
        "status": "development_provisional",
        "dataset_sha256": dataset_sha256,
        "validation_sample_count": int(len(validation_indices)),
        "checkpoint_count": len(checkpoint_paths),
        "seed_metrics_at_fixed_threshold_0_5": seed_rows,
        "seed_metric_aggregate": aggregate,
        "ensemble": {
            "method": "mean_probability",
            "fixed_threshold_0_5": fixed_metrics,
            "selected_threshold": selected_threshold,
            "threshold_selection": "validation maximum F1; lowest threshold breaks ties",
            "selected_threshold_metrics": selected_metrics,
        },
        "stratified_fixed_threshold_0_5_metrics": fixed_strata,
        "stratified_selected_threshold_metrics": selected_strata,
        "failure_case_summary": {
            "fixed_threshold_0_5": _failure_summary(
                validation_samples, validation_targets, ensemble_probability, 0.5
            ),
            "selected_threshold": _failure_summary(
                validation_samples,
                validation_targets,
                ensemble_probability,
                selected_threshold,
            ),
        },
        "limitations": [
            "validation is used for checkpoint selection and threshold selection",
            "test labels and test pose are not read",
            "onset validation support is too small for a replacement decision",
            "continuous background and elderly-domain gates remain open",
        ],
        "test_evaluated": False,
        "main_path_replacement": False,
    }
    destination.mkdir(parents=True, exist_ok=True)
    aggregate_path = destination / "aggregate_metrics.json"
    predictions_path = destination / "validation_predictions.jsonl"
    aggregate_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    prediction_lines = []
    for row_index, (sample, target, ensemble) in enumerate(
        zip(validation_samples, validation_targets, ensemble_probability, strict=True)
    ):
        prediction_lines.append(
            json.dumps(
                {
                    "sample_id": sample["sample_id"],
                    "video_id": sample["video_id"],
                    "dataset": sample["dataset"],
                    "action_id": sample.get("action_id"),
                    "hard_negative_type": sample.get("hard_negative_type"),
                    "scene_region": sample.get("scene_region"),
                    "supervision_family": sample["supervision_family"],
                    "partial_context": bool(sample["partial_context"]),
                    "target_presence": int(target),
                    "seed_probabilities": {
                        name: round(float(stacked[seed_index, row_index]), 8)
                        for seed_index, name in enumerate(checkpoint_names)
                    },
                    "ensemble_probability": round(float(ensemble), 8),
                    "predicted_at_selected_threshold": int(ensemble >= selected_threshold),
                },
                sort_keys=True,
            )
            + "\n"
        )
    predictions_path.write_text("".join(prediction_lines), encoding="utf-8")
    return {
        "aggregate_metrics_path": aggregate_path.as_posix(),
        "validation_predictions_path": predictions_path.as_posix(),
        "seed_metric_aggregate": aggregate,
        "ensemble": result["ensemble"],
        "test_evaluated": False,
        "main_path_replacement": False,
    }


def _predict(
    checkpoint: dict[str, Any],
    features: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model_config = checkpoint["model_config"]
    model = ContinuousFallTCN(
        input_channels=int(model_config["input_channels"]),
        hidden_channels=int(model_config["hidden_channels"]),
        kernel_size=int(model_config["kernel_size"]),
        dilations=tuple(int(value) for value in model_config["dilations"]),
        dropout=float(model_config["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    mean = np.asarray(checkpoint["normalization"]["mean"], dtype=np.float32)
    std = np.asarray(checkpoint["normalization"]["std"], dtype=np.float32)
    normalized = ((features - mean[None, None, :, :]) / std[None, None, :, :]).astype(np.float32)
    tensor = torch.from_numpy(normalized).permute(0, 2, 3, 1).reshape(len(features), -1, features.shape[1])
    probabilities: list[np.ndarray] = []
    onset_predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(tensor), batch_size):
            presence_logits, onset_logits = model(tensor[start : start + batch_size].to(device))
            probabilities.append(torch.sigmoid(presence_logits).cpu().numpy())
            onset_predictions.append(onset_logits.argmax(dim=-1).cpu().numpy())
    return np.concatenate(probabilities), np.concatenate(onset_predictions)


def _classification_metrics(
    targets: np.ndarray, probabilities: np.ndarray, *, threshold: float
) -> dict[str, Any]:
    predicted = probabilities >= threshold
    positive = targets == 1
    negative = ~positive
    tp = int(np.sum(predicted & positive))
    fp = int(np.sum(predicted & negative))
    tn = int(np.sum(~predicted & negative))
    fn = int(np.sum(~predicted & positive))
    precision = tp / max(1, tp + fp)
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    f1 = (
        2 * precision * recall / max(1e-9, precision + recall)
        if recall is not None
        else None
    )
    return {
        "threshold": round(float(threshold), 8),
        "count": int(len(targets)),
        "positive_count": int(np.sum(positive)),
        "negative_count": int(np.sum(negative)),
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6) if recall is not None else None,
        "specificity": round(specificity, 6) if specificity is not None else None,
        "false_positive_rate": round(1.0 - specificity, 6) if specificity is not None else None,
        "f1": round(f1, 6) if f1 is not None else None,
        "pr_auc": round(float(average_precision_score(targets, probabilities)), 6)
        if len(np.unique(targets)) > 1
        else None,
    }


def _best_f1_threshold(targets: np.ndarray, probabilities: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(targets, probabilities)
    if not len(thresholds):
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-9)
    best = float(np.max(f1))
    candidates = thresholds[np.flatnonzero(np.isclose(f1, best, rtol=0.0, atol=1e-12))]
    return float(np.min(candidates))


def _aggregate_seed_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    keys = ("f1", "precision", "recall", "pr_auc", "onset_accuracy")
    result: dict[str, Any] = {}
    for key in keys:
        values = np.asarray([row[key] for row in rows if row.get(key) is not None], dtype=np.float64)
        result[key] = {
            "mean": round(float(np.mean(values)), 6) if len(values) else None,
            "std_population": round(float(np.std(values)), 6) if len(values) else None,
            "min": round(float(np.min(values)), 6) if len(values) else None,
            "max": round(float(np.max(values)), 6) if len(values) else None,
        }
    return result


def _stratified_metrics(
    samples: Sequence[dict[str, Any]],
    targets: np.ndarray,
    probabilities: np.ndarray,
    field: str,
    threshold: float,
) -> dict[str, Any]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        grouped[str(sample.get(field)).lower()].append(index)
    return {
        value: _classification_metrics(
            targets[np.asarray(indices)], probabilities[np.asarray(indices)], threshold=threshold
        )
        for value, indices in sorted(grouped.items())
    }


def _failure_summary(
    samples: Sequence[dict[str, Any]],
    targets: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    predicted = probabilities >= threshold
    masks = {
        "false_positive": predicted & (targets == 0),
        "false_negative": ~predicted & (targets == 1),
    }
    result: dict[str, Any] = {"threshold": round(float(threshold), 8)}
    for failure_type, mask in masks.items():
        indices = np.flatnonzero(mask)
        result[f"{failure_type}_count"] = int(len(indices))
        for field in ("dataset", "supervision_family", "action_id", "hard_negative_type"):
            counts = Counter(str(samples[int(index)].get(field)) for index in indices)
            result[f"{failure_type}_by_{field}"] = dict(
                sorted(counts.items(), key=lambda item: (-item[1], item[0]))
            )
    return result


def _validate_checkpoint(checkpoint: dict[str, Any], dataset_sha256: str) -> None:
    if checkpoint.get("schema_version") != "fall-event-continuous-tcn-v1":
        raise ValueError("unexpected checkpoint schema")
    if checkpoint.get("dataset_sha256") != dataset_sha256:
        raise ValueError("checkpoint and dataset SHA-256 differ")
    if checkpoint.get("test_evaluated") is not False:
        raise ValueError("checkpoint test_evaluated must remain false")
    if checkpoint.get("main_path_replacement") is not False:
        raise ValueError("checkpoint main_path_replacement must remain false")


def _resolve_device(value: str) -> torch.device:
    if value == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if value == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
