"""Validation prediction and subject-level metrics for MODEL-COG-001."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from torch import Tensor, nn

try:
    from .dataset import MODALITY_ORDER, MocaStatistics
except ImportError:
    from dataset import MODALITY_ORDER, MocaStatistics


COMBINATION_MODALITIES = {
    "audio_text_face": frozenset(("audio", "text", "face")),
    "audio_text": frozenset(("audio", "text")),
    "audio_face": frozenset(("audio", "face")),
    "text_face": frozenset(("text", "face")),
    "audio": frozenset(("audio",)),
    "text": frozenset(("text",)),
    "face": frozenset(("face",)),
}


def forced_missing_mask(original: Tensor, combination: str | None) -> Tensor:
    if combination is None:
        return original.bool()
    if combination not in COMBINATION_MODALITIES:
        raise ValueError(f"unsupported modality combination: {combination}")
    selected = COMBINATION_MODALITIES[combination]
    artificial = torch.tensor(
        [name not in selected for name in MODALITY_ORDER],
        dtype=torch.bool,
        device=original.device,
    )
    return original.bool() | artificial.unsqueeze(0)


def filter_valid_rows(batch: Mapping[str, Any], missing_mask: Tensor) -> tuple[dict[str, Any], Tensor]:
    valid = ~missing_mask.all(dim=1)
    indices = torch.nonzero(valid, as_tuple=False).squeeze(1)
    filtered: dict[str, Any] = {}
    for key, value in batch.items():
        if key == "features":
            filtered[key] = {name: tensor[valid] for name, tensor in value.items()}
        elif isinstance(value, Tensor):
            filtered[key] = value[valid]
        elif isinstance(value, list):
            filtered[key] = [value[index] for index in indices.cpu().tolist()]
        else:
            filtered[key] = value
    filtered["missing_mask"] = missing_mask[valid]
    return filtered, valid


def move_batch_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if key == "features":
            moved[key] = {
                name: tensor.to(device=device, non_blocking=False)
                for name, tensor in value.items()
            }
        elif isinstance(value, Tensor):
            moved[key] = value.to(device=device, non_blocking=False)
        else:
            moved[key] = value
    return moved


@torch.no_grad()
def predict_validation(
    model: nn.Module,
    loader: Iterable[Mapping[str, Any]],
    *,
    device: torch.device,
    moca_statistics: MocaStatistics,
    forced_combination: str | None = None,
    checkpoint_sha256: str | None = None,
) -> list[dict[str, Any]]:
    model.eval()
    records: list[dict[str, Any]] = []
    for raw_batch in loader:
        batch = move_batch_to_device(raw_batch, device)
        missing = forced_missing_mask(batch["missing_mask"], forced_combination)
        filtered, valid = filter_valid_rows(batch, missing)
        if not bool(valid.any()):
            continue
        output = model(filtered["features"], filtered["quality"], filtered["missing_mask"])
        main_logits = output.hc_vs_non_hc_logit.detach().cpu().numpy()
        mci_logits = output.mci_vs_hc_logit.detach().cpu().numpy()
        class_logits = output.ad_mci_hc_logits.detach().cpu().numpy()
        moca_standardized = output.moca_standardized.detach().cpu().numpy()
        weights = output.gate_weights.detach().cpu().numpy()
        qualities = filtered["quality"].detach().cpu().numpy()
        masks = filtered["missing_mask"].detach().cpu().numpy()
        main_labels = filtered["hc_vs_non_hc"].detach().cpu().numpy()
        mci_labels = filtered["mci_vs_hc"].detach().cpu().numpy()
        mci_masks = filtered["mci_vs_hc_mask"].detach().cpu().numpy()
        class_labels = filtered["ad_mci_hc"].detach().cpu().numpy()
        moca_labels = filtered["moca"].detach().cpu().numpy()
        moca_masks = filtered["moca_mask"].detach().cpu().numpy()
        for index, sample_id in enumerate(filtered["sample_id"]):
            standardized = float(moca_standardized[index])
            moca_value = float(np.clip(
                standardized * moca_statistics.std + moca_statistics.mean, 0.0, 30.0
            ))
            logits = [float(value) for value in class_logits[index]]
            records.append(
                {
                    "schema_version": "cognitive_validation_prediction_v1",
                    "checkpoint_sha256": checkpoint_sha256,
                    "sample_id": str(sample_id),
                    "subject_id": str(filtered["subject_id"][index]),
                    "diagnosis_label": str(filtered["diagnosis_label"][index]),
                    "modality_combination": forced_combination or "original",
                    "quality": [float(value) for value in qualities[index]],
                    "missing_mask": [int(value) for value in masks[index]],
                    "gate_weights": [float(value) for value in weights[index]],
                    "hc_vs_non_hc": {
                        "label": int(main_labels[index]),
                        "raw_logit": float(main_logits[index]),
                        "uncalibrated_probability": _sigmoid(float(main_logits[index])),
                    },
                    "mci_vs_hc": {
                        "label": int(mci_labels[index]) if bool(mci_masks[index]) else None,
                        "label_mask": bool(mci_masks[index]),
                        "raw_logit": float(mci_logits[index]),
                        "uncalibrated_probability": _sigmoid(float(mci_logits[index])),
                    },
                    "ad_mci_hc": {
                        "label": int(class_labels[index]),
                        "raw_logits": logits,
                        "uncalibrated_probabilities": _softmax(logits),
                    },
                    "moca": {
                        "label": float(moca_labels[index]) if bool(moca_masks[index]) else None,
                        "label_mask": bool(moca_masks[index]),
                        "standardized_output": standardized,
                        "output": moca_value,
                        "normalization_mean": float(moca_statistics.mean),
                        "normalization_std": float(moca_statistics.std),
                    },
                }
            )
    records.sort(key=lambda row: str(row["sample_id"]))
    return records


def subject_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("Validation prediction records are empty")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["subject_id"])].append(record)

    main_labels: list[int] = []
    main_scores: list[float] = []
    mci_labels: list[int] = []
    mci_scores: list[float] = []
    class_labels: list[int] = []
    class_predictions: list[int] = []
    moca_labels: list[float] = []
    moca_predictions: list[float] = []
    subject_rows: list[dict[str, Any]] = []
    for subject_id, rows in sorted(grouped.items()):
        diagnoses = {str(row["diagnosis_label"]) for row in rows}
        if len(diagnoses) != 1:
            raise ValueError(f"Validation diagnosis conflict for {subject_id}")
        main_label = _unique_label(rows, "hc_vs_non_hc")
        main_logit = float(np.mean([row["hc_vs_non_hc"]["raw_logit"] for row in rows]))
        main_probability = _sigmoid(main_logit)
        main_labels.append(main_label)
        main_scores.append(main_probability)

        valid_mci = [row for row in rows if row["mci_vs_hc"]["label_mask"]]
        mci_payload: dict[str, Any] | None = None
        if valid_mci:
            mci_label = int(valid_mci[0]["mci_vs_hc"]["label"])
            if any(int(row["mci_vs_hc"]["label"]) != mci_label for row in valid_mci):
                raise ValueError(f"Validation MCI label conflict for {subject_id}")
            mci_logit = float(np.mean([row["mci_vs_hc"]["raw_logit"] for row in valid_mci]))
            mci_probability = _sigmoid(mci_logit)
            mci_labels.append(mci_label)
            mci_scores.append(mci_probability)
            mci_payload = {
                "label": mci_label,
                "mean_raw_logit": mci_logit,
                "uncalibrated_probability": mci_probability,
            }

        class_label = int(rows[0]["ad_mci_hc"]["label"])
        if any(int(row["ad_mci_hc"]["label"]) != class_label for row in rows):
            raise ValueError(f"Validation class label conflict for {subject_id}")
        mean_class_logits = np.mean(
            np.asarray([row["ad_mci_hc"]["raw_logits"] for row in rows], dtype=np.float64),
            axis=0,
        )
        class_probability = _softmax(mean_class_logits.tolist())
        class_prediction = int(np.argmax(mean_class_logits))
        class_labels.append(class_label)
        class_predictions.append(class_prediction)

        valid_moca = [row for row in rows if row["moca"]["label_mask"]]
        moca_payload: dict[str, Any] | None = None
        if valid_moca:
            label = float(np.mean([row["moca"]["label"] for row in valid_moca]))
            normalization = {
                (
                    float(row["moca"]["normalization_mean"]),
                    float(row["moca"]["normalization_std"]),
                )
                for row in valid_moca
            }
            if len(normalization) != 1:
                raise ValueError(f"Validation MoCA normalization conflict for {subject_id}")
            mean, std = next(iter(normalization))
            standardized_prediction = float(
                np.mean([row["moca"]["standardized_output"] for row in valid_moca])
            )
            prediction = float(np.clip(standardized_prediction * std + mean, 0.0, 30.0))
            moca_labels.append(label)
            moca_predictions.append(prediction)
            moca_payload = {
                "label": label,
                "mean_standardized_output": standardized_prediction,
                "prediction": prediction,
            }

        subject_rows.append(
            {
                "subject_id": subject_id,
                "diagnosis_label": next(iter(diagnoses)),
                "task_count": len(rows),
                "hc_vs_non_hc": {
                    "label": main_label,
                    "mean_raw_logit": main_logit,
                    "uncalibrated_probability": main_probability,
                },
                "mci_vs_hc": mci_payload,
                "ad_mci_hc": {
                    "label": class_label,
                    "mean_raw_logits": [float(value) for value in mean_class_logits],
                    "uncalibrated_probabilities": class_probability,
                    "prediction": class_prediction,
                },
                "moca": moca_payload,
            }
        )

    main = _binary_metrics(main_labels, main_scores)
    mci = _binary_metrics(mci_labels, mci_scores)
    three_class = {
        "subject_count": len(class_labels),
        "balanced_accuracy": float(balanced_accuracy_score(class_labels, class_predictions)),
        "macro_f1": float(f1_score(class_labels, class_predictions, average="macro")),
    }
    moca = _regression_metrics(moca_labels, moca_predictions)
    return {
        "task_count": len(records),
        "subject_count": len(subject_rows),
        "hc_vs_non_hc": main,
        "mci_vs_hc": mci,
        "ad_mci_hc": three_class,
        "moca": moca,
        "subjects": subject_rows,
    }


def checkpoint_is_better(
    candidate_auc: float,
    candidate_balanced_accuracy: float,
    best_auc: float,
    best_balanced_accuracy: float,
    *,
    auc_tolerance: float = 1e-6,
) -> bool:
    if not math.isfinite(candidate_auc):
        raise ValueError("candidate Validation AUC is not finite")
    if candidate_auc > best_auc + auc_tolerance:
        return True
    if abs(candidate_auc - best_auc) < auc_tolerance:
        return candidate_balanced_accuracy > best_balanced_accuracy
    return False


def _binary_metrics(labels: Sequence[int], probabilities: Sequence[float]) -> dict[str, Any]:
    unique = sorted(set(int(value) for value in labels))
    if len(unique) != 2:
        return {
            "subject_count": len(labels),
            "roc_auc": None,
            "balanced_accuracy_at_0_5": None,
            "reason": "requires_two_classes",
        }
    predicted = [int(value >= 0.5) for value in probabilities]
    return {
        "subject_count": len(labels),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "balanced_accuracy_at_0_5": float(balanced_accuracy_score(labels, predicted)),
        "reason": None,
    }


def _regression_metrics(labels: Sequence[float], predictions: Sequence[float]) -> dict[str, Any]:
    if not labels:
        return {"subject_count": 0, "mae": None, "rmse": None, "reason": "no_labels"}
    return {
        "subject_count": len(labels),
        "mae": float(mean_absolute_error(labels, predictions)),
        "rmse": float(math.sqrt(mean_squared_error(labels, predictions))),
        "reason": None,
    }


def _unique_label(rows: Sequence[Mapping[str, Any]], head: str) -> int:
    labels = {int(row[head]["label"]) for row in rows}
    if len(labels) != 1:
        raise ValueError(f"Validation {head} label conflict")
    return next(iter(labels))


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _softmax(values: Sequence[float]) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    array = array - np.max(array)
    exp_values = np.exp(array)
    return [float(value) for value in exp_values / exp_values.sum()]


__all__ = [
    "COMBINATION_MODALITIES",
    "checkpoint_is_better",
    "filter_valid_rows",
    "forced_missing_mask",
    "move_batch_to_device",
    "predict_validation",
    "subject_metrics",
]
