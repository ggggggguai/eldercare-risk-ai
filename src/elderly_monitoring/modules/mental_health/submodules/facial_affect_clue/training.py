from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import MicroexpressionArtifactDataset, class_counts
from .metrics import classification_metrics
from .model import MHSSATGCN, MHSSATGCNConfig, MODEL_SCHEMA_VERSION, load_model_checkpoint


TRAINING_SCHEMA_VERSION = "smic_loso_training_v1"


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 20260806
    max_epochs: int = 40
    min_epochs: int = 8
    patience: int = 8
    batch_size: int = 32
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    gradient_clip_norm: float = 1.0
    num_workers: int = 0
    use_class_weights: bool = True

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "TrainingConfig":
        known = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{key: value for key, value in values.items() if key in known})

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _loader(
    dataset: MicroexpressionArtifactDataset,
    *,
    config: TrainingConfig,
    shuffle: bool,
    seed: int,
) -> DataLoader[dict[str, Any]]:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "patches": batch["patches"].to(device, non_blocking=True),
        "region_ids": batch["region_ids"].to(device, non_blocking=True),
        "masks": batch["masks"].to(device, non_blocking=True),
        "keypoints": batch["keypoints"].to(device, non_blocking=True),
        "label": batch["label"].to(device, non_blocking=True),
        "sample_id": list(batch["sample_id"]),
        "subject_id": list(batch["subject_id"]),
    }


def _forward(model: MHSSATGCN, batch: Mapping[str, Any]) -> torch.Tensor:
    return model(
        batch["patches"],
        batch["region_ids"],
        batch["masks"],
        batch["keypoints"],
    )


def class_weight_tensor(
    records: Sequence[Mapping[str, Any]], device: torch.device
) -> torch.Tensor:
    counts = class_counts(records)
    total = sum(counts.values())
    weights = [total / (3.0 * max(counts[label], 1)) for label in (0, 1, 2)]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_one_epoch(
    model: MHSSATGCN,
    loader: DataLoader[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    device: torch.device,
    gradient_clip_norm: float,
) -> float:
    model.train()
    total_loss = 0.0
    sample_count = 0
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = _forward(model, batch)
        loss = loss_function(logits, batch["label"])
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        count = int(batch["label"].shape[0])
        total_loss += float(loss.detach().cpu()) * count
        sample_count += count
    return total_loss / max(sample_count, 1)


@torch.no_grad()
def evaluate_model(
    model: MHSSATGCN,
    loader: DataLoader[dict[str, Any]],
    loss_function: nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    labels: list[int] = []
    predictions: list[int] = []
    probabilities: list[list[float]] = []
    sample_ids: list[str] = []
    subject_ids: list[str] = []
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        logits = _forward(model, batch)
        loss = loss_function(logits, batch["label"])
        probability = torch.softmax(logits, dim=-1)
        prediction = probability.argmax(dim=-1)
        count = int(batch["label"].shape[0])
        total_loss += float(loss.detach().cpu()) * count
        labels.extend(batch["label"].detach().cpu().tolist())
        predictions.extend(prediction.detach().cpu().tolist())
        probabilities.extend(probability.detach().cpu().tolist())
        sample_ids.extend(batch["sample_id"])
        subject_ids.extend(batch["subject_id"])
    metrics = classification_metrics(labels, predictions)
    return {
        "loss": total_loss / max(len(labels), 1),
        "metrics": metrics,
        "labels": labels,
        "predictions": predictions,
        "probabilities": probabilities,
        "sample_ids": sample_ids,
        "subject_ids": subject_ids,
    }


def _state_dict_on_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def train_loso_fold(
    *,
    records: Sequence[Mapping[str, Any]],
    fold: Mapping[str, Any],
    model_config: MHSSATGCNConfig,
    training_config: TrainingConfig,
    device: torch.device,
    output_dir: Path,
    artifact_manifest_path: Path,
    artifact_manifest_sha256: str,
    split_manifest_path: Path,
    split_manifest_sha256: str,
    config_path: Path,
    config_sha256: str,
    fold_index: int,
) -> dict[str, Any]:
    fold_id = str(fold["fold_id"])
    fold_dir = output_dir / "folds" / fold_id
    fold_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = fold_dir / "best.pt"
    log_path = fold_dir / "epochs.jsonl"
    seed = training_config.seed + fold_index
    set_reproducible_seed(seed)

    train_dataset = MicroexpressionArtifactDataset(
        records, sample_ids=fold["train_sample_ids"], preload=True
    )
    validation_dataset = MicroexpressionArtifactDataset(
        records, sample_ids=fold["validation_sample_ids"], preload=True
    )
    test_dataset = MicroexpressionArtifactDataset(
        records, sample_ids=fold["test_sample_ids"], preload=True
    )
    train_loader = _loader(
        train_dataset, config=training_config, shuffle=True, seed=seed
    )
    validation_loader = _loader(
        validation_dataset, config=training_config, shuffle=False, seed=seed
    )
    test_loader = _loader(
        test_dataset, config=training_config, shuffle=False, seed=seed
    )

    model = MHSSATGCN(model_config).to(device)
    weights = (
        class_weight_tensor(train_dataset.records, device)
        if training_config.use_class_weights
        else None
    )
    loss_function = nn.CrossEntropyLoss(
        weight=weights,
        label_smoothing=training_config.label_smoothing,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    best_macro_f1 = float("-inf")
    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with log_path.open("w", encoding="utf-8", newline="\n") as log_handle:
        for epoch in range(1, training_config.max_epochs + 1):
            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                loss_function,
                device,
                training_config.gradient_clip_norm,
            )
            validation = evaluate_model(
                model, validation_loader, loss_function, device
            )
            validation_macro_f1 = float(validation["metrics"]["macro_f1"])
            validation_loss = float(validation["loss"])
            improved = validation_macro_f1 > best_macro_f1 + 1e-12 or (
                abs(validation_macro_f1 - best_macro_f1) <= 1e-12
                and validation_loss < best_validation_loss - 1e-12
            )
            log_row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "validation_metrics": validation["metrics"],
                "checkpoint_improved": improved,
                "selection_source": "validation_only",
            }
            log_handle.write(json.dumps(log_row, ensure_ascii=False) + "\n")
            log_handle.flush()
            if improved:
                best_macro_f1 = validation_macro_f1
                best_validation_loss = validation_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                checkpoint = {
                    "model_schema_version": MODEL_SCHEMA_VERSION,
                    "training_schema_version": TRAINING_SCHEMA_VERSION,
                    "model_config": model_config.as_dict(),
                    "training_config": training_config.as_dict(),
                    "model_state_dict": _state_dict_on_cpu(model),
                    "fold_id": fold_id,
                    "fold_index": fold_index,
                    "test_subject": fold["test_subject"],
                    "train_subjects": fold["train_subjects"],
                    "validation_subjects": fold["validation_subjects"],
                    "test_subjects": fold["test_subjects"],
                    "best_epoch": best_epoch,
                    "selection_metric": "validation_macro_f1_then_validation_loss",
                    "best_validation_macro_f1": best_macro_f1,
                    "best_validation_loss": best_validation_loss,
                    "test_used_for_selection": False,
                    "seed": seed,
                    "artifact_manifest": {
                        "path": artifact_manifest_path.resolve().as_posix(),
                        "sha256": artifact_manifest_sha256,
                    },
                    "split_manifest": {
                        "path": split_manifest_path.resolve().as_posix(),
                        "sha256": split_manifest_sha256,
                    },
                    "config": {
                        "path": config_path.resolve().as_posix(),
                        "sha256": config_sha256,
                    },
                    "frame_annotation_source": "estimated",
                    "evaluation_scope": "engineering_only",
                    "paper_reproduction_claim": False,
                }
                torch.save(checkpoint, checkpoint_path)
            else:
                epochs_without_improvement += 1
            if (
                epoch >= training_config.min_epochs
                and epochs_without_improvement >= training_config.patience
            ):
                break

    if not checkpoint_path.is_file():
        raise RuntimeError(f"No checkpoint saved for {fold_id}")
    best_model, checkpoint = load_model_checkpoint(
        str(checkpoint_path), map_location=device
    )
    best_model = best_model.to(device).eval()
    validation = evaluate_model(best_model, validation_loader, loss_function, device)
    test = evaluate_model(best_model, test_loader, loss_function, device)
    duration_seconds = time.perf_counter() - started
    peak_gpu_memory_bytes = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    checkpoint_sha256 = _sha256_file(checkpoint_path)
    return {
        "fold_id": fold_id,
        "status": "completed",
        "test_subject": fold["test_subject"],
        "train_subjects": fold["train_subjects"],
        "validation_subjects": fold["validation_subjects"],
        "best_epoch": int(checkpoint["best_epoch"]),
        "selection_metric": checkpoint["selection_metric"],
        "test_used_for_selection": bool(checkpoint["test_used_for_selection"]),
        "validation": {
            "loss": validation["loss"],
            "metrics": validation["metrics"],
        },
        "test": test,
        "checkpoint": {
            "path": checkpoint_path.resolve().as_posix(),
            "sha256": checkpoint_sha256,
            "strict_reload_verified": True,
        },
        "log": {
            "path": log_path.resolve().as_posix(),
            "sha256": _sha256_file(log_path),
        },
        "duration_seconds": duration_seconds,
        "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
    }
