from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dstm import DSTM, load_dstm_checkpoint, save_dstm_checkpoint
from .dstm_dataset import DSTMArtifactCollator, DSTMArtifactDataset
from .metrics import classification_metrics


DSTM_TRAINING_SCHEMA_VERSION = "dstm_training_v1"


@dataclass(frozen=True)
class DSTMTrainingConfig:
    optimizer: str = "adam"
    learning_rate: float = 1e-4
    batch_size: int = 32
    max_epochs: int = 200
    min_epochs: int = 1
    patience: int = 15
    weight_decay: float = 0.0
    gradient_clip_norm: float = 1.0
    num_workers: int = 0
    loss: str = "cross_entropy_plus_infonce"

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "DSTMTrainingConfig":
        known = {item.name for item in cls.__dataclass_fields__.values()}
        return cls(**{key: value for key, value in values.items() if key in known})

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def set_dstm_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def _loader(
    dataset: DSTMArtifactDataset,
    *,
    config: DSTMTrainingConfig,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=config.num_workers,
        collate_fn=DSTMArtifactCollator(),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def _move(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    moved = dict(batch)
    for key in (
        "patches",
        "masks",
        "keypoints",
        "labels",
        "temporal_features",
        "temporal_mask",
    ):
        moved[key] = moved[key].to(device, non_blocking=True)
    return moved


def _forward_loss(
    model: DSTM,
    batch: Mapping[str, Any],
    *,
    device: torch.device,
    train: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    moved = _move(batch, device)
    if train:
        logits, aux = model(
            moved["patches"],
            moved["masks"],
            moved["keypoints"],
            temporal_features=moved["temporal_features"],
            temporal_mask=moved["temporal_mask"],
            return_aux=True,
        )
    else:
        with torch.no_grad():
            logits, aux = model(
                moved["patches"],
                moved["masks"],
                moved["keypoints"],
                temporal_features=moved["temporal_features"],
                temporal_mask=moved["temporal_mask"],
                return_aux=True,
            )
    classification_loss = torch.nn.functional.cross_entropy(logits, moved["labels"])
    loss = classification_loss + model.config.alignment_lambda * aux["alignment_loss"]
    return loss, logits, aux


def train_epoch(
    model: DSTM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    gradient_clip_norm: float,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_align = 0.0
    total_count = 0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        loss, _, aux = _forward_loss(model, batch, device=device, train=True)
        loss.backward()
        if gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        count = int(batch["labels"].shape[0])
        total_count += count
        total_loss += float(loss.detach()) * count
        total_align += float(aux["alignment_loss"].detach()) * count
        total_ce += float(
            (loss.detach() - model.config.alignment_lambda * aux["alignment_loss"].detach())
        ) * count
    denominator = max(total_count, 1)
    return {
        "loss": total_loss / denominator,
        "classification_loss": total_ce / denominator,
        "alignment_loss": total_align / denominator,
    }


def evaluate_loader(
    model: DSTM,
    loader: DataLoader,
    *,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    sample_ids: list[str] = []
    subject_ids: list[str] = []
    quality_status: list[str] = []
    apex_boundary_status: list[str] = []
    labels: list[int] = []
    predictions: list[int] = []
    probabilities: list[list[float]] = []
    alignment_values: list[float] = []
    for batch in loader:
        loss, logits, aux = _forward_loss(model, batch, device=device, train=False)
        count = int(batch["labels"].shape[0])
        total_count += count
        total_loss += float(loss) * count
        probability = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        prediction = probability.argmax(axis=1)
        labels.extend(int(value) for value in batch["labels"].tolist())
        predictions.extend(int(value) for value in prediction.tolist())
        probabilities.extend(probability.tolist())
        sample_ids.extend(str(value) for value in batch["sample_ids"])
        subject_ids.extend(str(value) for value in batch["subject_ids"])
        quality_status.extend(str(value) for value in batch["quality_status"])
        apex_boundary_status.extend(str(value) for value in batch["apex_boundary_status"])
        alignment_values.append(float(aux["alignment_loss"]))
    metrics = classification_metrics(labels, predictions)
    score = 2.0 * metrics["uf1"] * metrics["uar"] / max(metrics["uf1"] + metrics["uar"], 1e-12)
    rows = [
        {
            "sample_id": sample_id,
            "subject_id": subject_id,
            "label": label,
            "prediction": prediction,
            "probabilities": probability,
            "quality_status": quality,
            "apex_boundary_status": boundary,
        }
        for sample_id, subject_id, label, prediction, probability, quality, boundary in zip(
            sample_ids,
            subject_ids,
            labels,
            predictions,
            probabilities,
            quality_status,
            apex_boundary_status,
            strict=True,
        )
    ]
    return {
        "loss": total_loss / max(total_count, 1),
        "metrics": metrics,
        "selection_score": float(score),
        "predictions": rows,
        "mean_alignment_loss": float(np.mean(alignment_values)) if alignment_values else 0.0,
    }


def train_dstm_fold(
    *,
    model: DSTM,
    au_adjacency: torch.Tensor,
    spatial_records: Sequence[Mapping[str, Any]],
    temporal_records: Sequence[Mapping[str, Any]],
    train_sample_ids: Sequence[str],
    validation_sample_ids: Sequence[str],
    temporal_features: Mapping[str, np.ndarray],
    training_config: DSTMTrainingConfig,
    device: torch.device,
    seed: int,
    output_dir: Path,
    fold_id: str,
    reducer_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    if training_config.optimizer != "adam" or training_config.loss != "cross_entropy_plus_infonce":
        raise ValueError("MODEL-ME-006 uses Adam with CE plus InfoNCE")
    set_dstm_seed(seed)
    train_dataset = DSTMArtifactDataset(
        spatial_records,
        temporal_records,
        sample_ids=train_sample_ids,
        temporal_features=temporal_features,
        preload=True,
    )
    validation_dataset = DSTMArtifactDataset(
        spatial_records,
        temporal_records,
        sample_ids=validation_sample_ids,
        temporal_features=temporal_features,
        preload=True,
    )
    train_loader = _loader(train_dataset, config=training_config, shuffle=True, seed=seed)
    validation_loader = _loader(validation_dataset, config=training_config, shuffle=False, seed=seed)
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    checkpoint_path = output_dir / "best.pt"
    epoch_log_path = output_dir / "epochs.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    best_score = float("-inf")
    best_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    epoch_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.to(device)
    for epoch in range(1, training_config.max_epochs + 1):
        training = train_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            gradient_clip_norm=training_config.gradient_clip_norm,
        )
        validation = evaluate_loader(model, validation_loader, device=device)
        score = float(validation["selection_score"])
        validation_loss = float(validation["loss"])
        improved = score > best_score + 1e-12 or (
            abs(score - best_score) <= 1e-12 and validation_loss < best_loss - 1e-12
        )
        epoch_rows.append(
            {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "training": training,
                "validation_loss": validation_loss,
                "validation_metrics": validation["metrics"],
                "validation_selection_score": score,
                "validation_mean_alignment_loss": validation["mean_alignment_loss"],
                "checkpoint_improved": improved,
                "selection_source": "validation_subjects_only",
                "test_used_for_selection": False,
            }
        )
        epoch_log_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in epoch_rows),
            encoding="utf-8",
        )
        if improved:
            best_score = score
            best_loss = validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            save_dstm_checkpoint(
                checkpoint_path,
                model=model,
                training_config=training_config.as_dict(),
                metadata={
                    "task_id": "MODEL-ME-006",
                    "training_schema_version": DSTM_TRAINING_SCHEMA_VERSION,
                    "fold_id": fold_id,
                    "seed": seed,
                    "best_epoch": best_epoch,
                    "best_validation_selection_score": best_score,
                    "best_validation_loss": best_loss,
                    "selection_source": "validation_subjects_only",
                    "test_used_for_selection": False,
                    "reducer_metadata": dict(reducer_metadata),
                },
            )
        else:
            epochs_without_improvement += 1
        if epoch >= training_config.min_epochs and epochs_without_improvement >= training_config.patience:
            break
    if not checkpoint_path.is_file():
        raise RuntimeError(f"DSTM produced no checkpoint for {fold_id}/{seed}")
    best_model, checkpoint = load_dstm_checkpoint(
        checkpoint_path, au_adjacency=au_adjacency, map_location=device
    )
    best_model.to(device).eval()
    validation = evaluate_loader(best_model, validation_loader, device=device)
    duration = time.perf_counter() - started
    return {
        "schema_version": DSTM_TRAINING_SCHEMA_VERSION,
        "task_id": "MODEL-ME-006",
        "status": "completed",
        "fold_id": fold_id,
        "seed": seed,
        "train_sample_count": len(train_sample_ids),
        "validation_sample_count": len(validation_sample_ids),
        "best_epoch": int(checkpoint["best_epoch"]),
        "best_validation_selection_score": float(checkpoint["best_validation_selection_score"]),
        "best_validation_loss": float(checkpoint["best_validation_loss"]),
        "validation": {
            "metrics": validation["metrics"],
            "loss": validation["loss"],
            "selection_score": validation["selection_score"],
        },
        "checkpoint_path": checkpoint_path.resolve().as_posix(),
        "epoch_log_path": epoch_log_path.resolve().as_posix(),
        "epoch_count": len(epoch_rows),
        "duration_seconds": duration,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "reducer_metadata": dict(reducer_metadata),
        "selection_source": "validation_subjects_only",
        "test_used_for_selection": False,
    }, best_model
