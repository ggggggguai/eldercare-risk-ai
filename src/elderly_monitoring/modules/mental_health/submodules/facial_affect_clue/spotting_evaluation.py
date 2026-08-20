from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from hashlib import sha256
import itertools
import json
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch import nn
from torch.utils.data import DataLoader

from .microexpression_spotter import (
    MicroExpressionWindowSpotter,
    SpotterFeatureDataset,
    SpotterModelConfig,
)


SPOTTING_EVALUATION_SCHEMA_VERSION = "smic_window_spotter_strict_loso_v1"
LABEL_NAMES = {0: "non_micro", 1: "micro"}
LABELS = (0, 1)


@dataclass(frozen=True)
class SpotterTrainingConfig:
    optimizer: str = "adam"
    learning_rate: float = 1e-3
    max_epochs: int = 80
    min_epochs: int = 10
    patience: int = 12
    batch_size: int = 64
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0

    def __post_init__(self) -> None:
        if self.optimizer != "adam":
            raise ValueError("SPOT-ME-002 freezes optimizer=adam")
        if self.learning_rate <= 0 or self.max_epochs < 1:
            raise ValueError("Invalid spotting optimizer configuration")
        if not 1 <= self.min_epochs <= self.max_epochs:
            raise ValueError("min_epochs must be within [1, max_epochs]")
        if self.patience < 1 or self.batch_size < 1:
            raise ValueError("patience and batch_size must be positive")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sequence_sha256(values: Sequence[str]) -> str:
    encoded = json.dumps(list(values), separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _label_counts(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(int(record["label"]) for record in records)
    return {str(label): int(counts[label]) for label in LABELS}


def _validation_subjects(
    subject_records: Mapping[str, Sequence[Mapping[str, Any]]],
    candidates: Sequence[str],
    count: int,
) -> tuple[str, ...]:
    all_records = [record for subject in candidates for record in subject_records[subject]]
    global_counts = Counter(int(record["label"]) for record in all_records)
    target_size = len(all_records) * 0.20

    def score(selected: tuple[str, ...]) -> tuple[float, float, float, tuple[str, ...]]:
        validation = [record for subject in selected for record in subject_records[subject]]
        counts = Counter(int(record["label"]) for record in validation)
        train_counts = global_counts - counts
        missing = sum(counts[label] == 0 for label in LABELS) + sum(
            train_counts[label] == 0 for label in LABELS
        )
        size_penalty = abs(len(validation) - target_size) / max(target_size, 1.0)
        distribution_penalty = sum(
            abs(
                counts[label] / max(len(validation), 1)
                - global_counts[label] / max(len(all_records), 1)
            )
            for label in LABELS
        )
        return float(missing), distribution_penalty, size_penalty, selected

    options = list(itertools.combinations(sorted(candidates), count))
    if not options:
        raise ValueError("No validation subject combination available")
    return min(options, key=score)


def build_spotting_splits(
    records: Sequence[Mapping[str, Any]], *, validation_subject_count: int = 3
) -> dict[str, Any]:
    if not records:
        raise ValueError("Spotting split records cannot be empty")
    subject_records: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("source_dataset") != "smic_hs":
            raise ValueError("SPOT-ME-002 accepts SMIC records only")
        if int(record["label"]) not in LABELS:
            raise ValueError("SPOT-ME-002 labels must be binary")
        subject_records[str(record["subject_id"])].append(record)
    subjects = sorted(subject_records)
    if len(subjects) < validation_subject_count + 2:
        raise ValueError("Not enough subjects for strict train/validation/test split")

    folds: list[dict[str, Any]] = []
    for test_subject in subjects:
        candidates = [subject for subject in subjects if subject != test_subject]
        validation_subjects = _validation_subjects(
            subject_records, candidates, validation_subject_count
        )
        training_subjects = tuple(
            subject for subject in candidates if subject not in validation_subjects
        )

        def ids(selected: Sequence[str]) -> list[str]:
            return sorted(
                str(record["sample_id"])
                for subject in selected
                for record in subject_records[subject]
            )

        train_ids = ids(training_subjects)
        validation_ids = ids(validation_subjects)
        test_ids = ids((test_subject,))
        folds.append(
            {
                "fold_id": f"loso_{test_subject}",
                "test_subject": test_subject,
                "train_subjects": list(training_subjects),
                "validation_subjects": list(validation_subjects),
                "test_subjects": [test_subject],
                "train_sample_ids": train_ids,
                "validation_sample_ids": validation_ids,
                "test_sample_ids": test_ids,
                "counts": {
                    "train": len(train_ids),
                    "validation": len(validation_ids),
                    "test": len(test_ids),
                },
                "label_counts": {
                    "train": _label_counts(
                        [record for subject in training_subjects for record in subject_records[subject]]
                    ),
                    "validation": _label_counts(
                        [record for subject in validation_subjects for record in subject_records[subject]]
                    ),
                    "test": _label_counts(subject_records[test_subject]),
                },
                "selection_authority": {
                    "checkpoint": "validation_macro_f1_then_balanced_accuracy_then_loss",
                    "threshold": "validation_macro_f1_then_balanced_accuracy_then_loss",
                    "test_used_for_selection": False,
                },
            }
        )
    result = {
        "schema_version": "smic_window_spotter_subject_loso_v1",
        "task_id": "SPOT-ME-002",
        "source_dataset": "smic_hs",
        "sample_count": len(records),
        "subject_count": len(subjects),
        "subjects": subjects,
        "validation_policy": {
            "type": "subject_level_distribution_match",
            "subject_count": validation_subject_count,
            "target_fraction": 0.20,
        },
        "folds": folds,
    }
    validate_spotting_splits(result)
    return result


def validate_spotting_splits(split_manifest: Mapping[str, Any]) -> None:
    subjects = set(str(value) for value in split_manifest["subjects"])
    test_subjects: list[str] = []
    for fold in split_manifest["folds"]:
        fold_sample_ids: set[str] = set()
        train_subjects = set(fold["train_subjects"])
        validation_subjects = set(fold["validation_subjects"])
        test = set(fold["test_subjects"])
        if train_subjects & validation_subjects or train_subjects & test:
            raise ValueError(f"Subject leakage in {fold['fold_id']}")
        if validation_subjects & test or train_subjects | validation_subjects | test != subjects:
            raise ValueError(f"Subject coverage mismatch in {fold['fold_id']}")
        partitions = {
            "train": fold["train_sample_ids"],
            "validation": fold["validation_sample_ids"],
            "test": fold["test_sample_ids"],
        }
        for name, sample_ids in partitions.items():
            if len(sample_ids) != len(set(sample_ids)):
                raise ValueError(f"Duplicate {name} sample in {fold['fold_id']}")
            for sample_id in sample_ids:
                if sample_id in fold_sample_ids:
                    raise ValueError(f"Repeated sample in fold: {sample_id}")
                fold_sample_ids.add(sample_id)
        if fold["selection_authority"]["test_used_for_selection"]:
            raise ValueError("Test selection leakage")
        test_subjects.extend(fold["test_subjects"])
    if sorted(test_subjects) != sorted(subjects):
        raise ValueError("Each subject must be tested exactly once")


def binary_metrics(labels: Sequence[int], predictions: Sequence[int]) -> dict[str, Any]:
    true = np.asarray(labels, dtype=np.int64)
    predicted = np.asarray(predictions, dtype=np.int64)
    if true.size == 0 or true.shape != predicted.shape:
        raise ValueError("Binary metrics require equal non-empty arrays")
    precision, recall, f1, support = precision_recall_fscore_support(
        true, predicted, labels=LABELS, zero_division=0
    )
    macro_f1 = float(f1_score(true, predicted, labels=LABELS, average="macro", zero_division=0))
    balanced = float(balanced_accuracy_score(true, predicted))
    return {
        "sample_count": int(true.size),
        "accuracy": float(accuracy_score(true, predicted)),
        "macro_f1": macro_f1,
        "uf1": macro_f1,
        "balanced_accuracy": balanced,
        "uar": balanced,
        "per_class": {
            LABEL_NAMES[label]: {
                "label": label,
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(LABELS)
        },
        "confusion_matrix": confusion_matrix(true, predicted, labels=LABELS).tolist(),
    }


def select_threshold(labels: Sequence[int], probabilities: Sequence[float]) -> dict[str, Any]:
    true = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(probabilities, dtype=np.float64)
    if true.size == 0 or true.shape != scores.shape or not np.all(np.isfinite(scores)):
        raise ValueError("Invalid validation scores")
    candidates = np.round(np.arange(0.05, 0.951, 0.01), 2)
    scored: list[tuple[tuple[float, float, float, float], float, dict[str, Any]]] = []
    for threshold in candidates:
        predictions = (scores >= threshold).astype(np.int64)
        metrics = binary_metrics(true, predictions)
        key = (
            metrics["macro_f1"],
            metrics["balanced_accuracy"],
            -abs(float(threshold) - 0.5),
            -float(threshold),
        )
        scored.append((key, float(threshold), metrics))
    _, threshold, metrics = max(scored, key=lambda item: item[0])
    return {"threshold": threshold, "metrics": metrics, "candidate_count": len(candidates)}


def _predict(
    model: nn.Module,
    records: Sequence[Mapping[str, Any]],
    device: torch.device,
    dataset: SpotterFeatureDataset | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    dataset = dataset or SpotterFeatureDataset(records)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    probabilities: list[float] = []
    labels: list[int] = []
    model.eval()
    with torch.no_grad():
        for features, batch_labels in loader:
            logits = model(features.to(device))
            probabilities.extend(torch.sigmoid(logits).cpu().tolist())
            labels.extend(batch_labels.to(torch.int64).tolist())
    return np.asarray(labels, dtype=np.int64), np.asarray(probabilities, dtype=np.float64)


def _train_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    device: torch.device,
    gradient_clip_norm: float,
) -> float:
    model.train()
    total, count = 0.0, 0
    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(model(features), labels)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite spotter training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        total += float(loss.detach().cpu()) * len(labels)
        count += len(labels)
    return total / max(count, 1)


def _validation_loss(
    model: nn.Module,
    records: Sequence[Mapping[str, Any]],
    loss_function: nn.Module,
    device: torch.device,
    dataset: SpotterFeatureDataset | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    labels, probabilities = _predict(model, records, device, dataset)
    logits = torch.logit(torch.tensor(probabilities, dtype=torch.float32).clamp(1e-6, 1 - 1e-6))
    target = torch.tensor(labels, dtype=torch.float32, device=device)
    logits = logits.to(device)
    loss = float(loss_function(logits, target).detach().cpu())
    return loss, labels, probabilities


def _write_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return sha256_file(path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(path)
    return sha256_file(path)


def train_spotting_fold(
    *,
    records: Sequence[Mapping[str, Any]],
    train_sample_ids: Sequence[str],
    validation_sample_ids: Sequence[str],
    test_sample_ids: Sequence[str],
    model_config: SpotterModelConfig,
    training_config: SpotterTrainingConfig,
    seed: int,
    device: torch.device,
    output_dir: Path,
    fold_id: str,
    input_hashes: Mapping[str, str],
) -> dict[str, Any]:
    by_id = {str(record["sample_id"]): record for record in records}
    partitions = {
        "train": [by_id[str(value)] for value in train_sample_ids],
        "validation": [by_id[str(value)] for value in validation_sample_ids],
        "test": [by_id[str(value)] for value in test_sample_ids],
    }
    if set(train_sample_ids) & set(validation_sample_ids) or set(train_sample_ids) & set(test_sample_ids) or set(validation_sample_ids) & set(test_sample_ids):
        raise ValueError("Spotting sample partition overlap")
    train_dataset = SpotterFeatureDataset(partitions["train"])
    validation_dataset = SpotterFeatureDataset(partitions["validation"])
    test_dataset = SpotterFeatureDataset(partitions["test"])
    input_dim = int(train_dataset.features.shape[1])
    set_seed(seed)
    model = MicroExpressionWindowSpotter(input_dim, model_config).to(device)
    labels = np.asarray([int(record["label"]) for record in partitions["train"]])
    positive = max(int(labels.sum()), 1)
    negative = max(int(len(labels) - labels.sum()), 1)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negative / positive, device=device))
    optimizer = torch.optim.Adam(
        model.parameters(), lr=training_config.learning_rate, weight_decay=training_config.weight_decay
    )
    loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_selection: tuple[float, float, float, float] | None = None
    best_validation: dict[str, Any] | None = None
    epoch_rows: list[dict[str, Any]] = []
    stale = 0
    for epoch in range(1, training_config.max_epochs + 1):
        train_loss = _train_epoch(model, loader, optimizer, loss_function, device, training_config.gradient_clip_norm)
        validation_loss, validation_labels, validation_probabilities = _validation_loss(
            model, partitions["validation"], loss_function, device, validation_dataset
        )
        selected = select_threshold(validation_labels, validation_probabilities)
        selection = (
            float(selected["metrics"]["macro_f1"]),
            float(selected["metrics"]["balanced_accuracy"]),
            -validation_loss,
            -abs(float(selected["threshold"]) - 0.5),
        )
        epoch_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "validation_threshold": selected["threshold"],
                "validation_metrics": selected["metrics"],
                "test_used_for_selection": False,
            }
        )
        if best_selection is None or selection > best_selection:
            best_selection = selection
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_validation = {"loss": validation_loss, **selected}
            stale = 0
        else:
            stale += 1
        if epoch >= training_config.min_epochs and stale >= training_config.patience:
            break
    if best_state is None or best_validation is None:
        raise RuntimeError("Spotter did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    test_labels, test_probabilities = _predict(model, partitions["test"], device, test_dataset)
    threshold = float(best_validation["threshold"])
    test_predictions = (test_probabilities >= threshold).astype(np.int64)
    test_metrics = binary_metrics(test_labels, test_predictions)
    prediction_rows = [
        {
            "sample_id": str(record["sample_id"]),
            "subject_id": str(record["subject_id"]),
            "label": int(label),
            "prediction": int(prediction),
            "probability_micro": float(probability),
            "threshold": threshold,
            "fold_id": fold_id,
            "seed": seed,
            "test_used_for_selection": False,
        }
        for record, label, prediction, probability in zip(
            partitions["test"], test_labels, test_predictions, test_probabilities, strict=True
        )
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best.pt"
    torch.save(
        {
            "schema_version": SPOTTING_EVALUATION_SCHEMA_VERSION,
            "task_id": "SPOT-ME-002",
            "fold_id": fold_id,
            "seed": seed,
            "input_dim": input_dim,
            "model_config": model_config.as_dict(),
            "training_config": training_config.as_dict(),
            "threshold": threshold,
            "selection_source": "training_side_validation_subjects_only",
            "test_used_for_selection": False,
            "input_hashes": dict(input_hashes),
            "model_state_dict": best_state,
        },
        checkpoint_path,
    )
    reload_payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    reloaded = MicroExpressionWindowSpotter(input_dim, model_config).to(device)
    reloaded.load_state_dict(reload_payload["model_state_dict"])
    reloaded.eval()
    probe_features = train_dataset.features[:1].to(device)
    with torch.no_grad():
        before_reload = model(probe_features)
        after_reload = reloaded(probe_features)
    reload_difference = float(torch.max(torch.abs(before_reload - after_reload)).cpu())
    if reload_difference != 0.0:
        raise RuntimeError(f"Spotter checkpoint reload mismatch: {reload_difference}")
    prediction_path = output_dir / "test_predictions.jsonl"
    prediction_hash = _write_jsonl(prediction_path, prediction_rows)
    epoch_path = output_dir / "epoch_log.jsonl"
    epoch_hash = _write_jsonl(epoch_path, epoch_rows)
    result = {
        "schema_version": SPOTTING_EVALUATION_SCHEMA_VERSION,
        "task_id": "SPOT-ME-002",
        "status": "completed",
        "fold_id": fold_id,
        "seed": seed,
        "model_config": model_config.as_dict(),
        "training_config": training_config.as_dict(),
        "input_dim": input_dim,
        "selection_source": "training_side_validation_subjects_only",
        "test_used_for_selection": False,
        "test_evaluation_count": 1,
        "input_hashes": dict(input_hashes),
        "partitions": {
            name: {
                "sample_count": len(values),
                "sample_ids_sha256": sequence_sha256([str(item["sample_id"]) for item in values]),
                "subject_ids": sorted({str(item["subject_id"]) for item in values}),
            }
            for name, values in partitions.items()
        },
        "validation": best_validation,
        "test": {
            "threshold": threshold,
            "metrics": test_metrics,
            "predictions": prediction_rows,
            "prediction_path": str(prediction_path.resolve()),
            "prediction_sha256": prediction_hash,
        },
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": sha256_file(checkpoint_path),
            "strict_reload_verified": True,
            "strict_reload_max_logit_difference": reload_difference,
        },
        "epoch_log": {"path": str(epoch_path.resolve()), "sha256": epoch_hash, "epoch_count": len(epoch_rows)},
    }
    result["result_sha256"] = _write_json(output_dir / "result.json", result)
    return result
