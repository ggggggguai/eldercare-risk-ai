from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from .metrics import classification_metrics
from .paper_dataset import (
    PaperV2ArtifactDataset,
    PaperV2Collator,
)
from .paper_mhssa_tgcn import (
    PaperMHSSATGCN,
    PaperMHSSATGCNConfig,
    load_paper_model_checkpoint,
)
from .paper_training import (
    PaperTrainingConfig,
    save_paper_model_checkpoint,
    sha256_file,
)


PAPER_EVALUATION_SCHEMA_VERSION = "microexpression_strict_nested_loso_v2"


@dataclass(frozen=True)
class EvaluationCandidate:
    variant_id: str = "p3_farneback_optical_strain"
    order_mode: str = "paper"
    attention_variant: str = "paper_exact"
    graph_mode: str = "fused"
    fusion_formula: str = "paper_equation_3_17"
    loss_mode: str = "cross_entropy"
    order_seed: int = 20260806

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def candidate_id(self) -> str:
        encoded = json.dumps(
            self.as_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return sha256(encoded).hexdigest()[:16]


def selection_score(uf1: float, uar: float) -> float:
    return 0.0 if uf1 + uar <= 0 else 2.0 * uf1 * uar / (uf1 + uar)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        cross_entropy = nn.functional.cross_entropy(
            logits, targets, reduction="none"
        )
        probability = torch.exp(-cross_entropy)
        return ((1.0 - probability) ** self.gamma * cross_entropy).mean()


def set_evaluation_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def class_weights(records: Sequence[Mapping[str, Any]]) -> torch.Tensor:
    counts = np.bincount(
        np.asarray([int(record["label"]) for record in records]), minlength=3
    )
    total = int(counts.sum())
    return torch.tensor(
        [total / (3.0 * max(int(count), 1)) for count in counts],
        dtype=torch.float32,
    )


def build_loss(
    mode: str,
    records: Sequence[Mapping[str, Any]],
    device: torch.device,
) -> nn.Module:
    if mode in {"cross_entropy", "balanced_sampler"}:
        return nn.CrossEntropyLoss()
    if mode == "weighted_cross_entropy":
        return nn.CrossEntropyLoss(weight=class_weights(records).to(device))
    if mode == "focal":
        return FocalLoss(gamma=2.0)
    raise ValueError(f"Unsupported evaluation loss: {mode}")


def _loader(
    dataset: PaperV2ArtifactDataset,
    *,
    candidate: EvaluationCandidate,
    training_config: PaperTrainingConfig,
    shuffle: bool,
    seed: int,
) -> DataLoader[dict[str, Any]]:
    generator = torch.Generator().manual_seed(seed)
    sampler = None
    if candidate.loss_mode == "balanced_sampler" and shuffle:
        weights = class_weights(dataset.records)
        sample_weights = torch.tensor(
            [float(weights[int(record["label"])]) for record in dataset.records]
        )
        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(dataset),
            replacement=True,
            generator=generator,
        )
    return DataLoader(
        dataset,
        batch_size=training_config.batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=training_config.num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
        collate_fn=PaperV2Collator(
            order_mode=candidate.order_mode,
            order_seed=candidate.order_seed,
        ),
    )


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "patches": batch["patches"].to(device, non_blocking=True),
        "masks": batch["masks"].to(device, non_blocking=True),
        "keypoints": batch["keypoints"].to(device, non_blocking=True),
        "labels": batch["labels"].to(device, non_blocking=True),
        "sample_ids": list(batch["sample_ids"]),
        "subject_ids": list(batch["subject_ids"]),
        "quality_status": list(batch["quality_status"]),
        "apex_boundary_status": list(batch["apex_boundary_status"]),
    }


def train_epoch(
    model: PaperMHSSATGCN,
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
        logits = model(batch["patches"], batch["masks"], batch["keypoints"])
        loss = loss_function(logits, batch["labels"])
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite paper evaluation training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        if not all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ):
            raise RuntimeError("Non-finite paper evaluation gradients")
        optimizer.step()
        count = int(batch["labels"].shape[0])
        total_loss += float(loss.detach().cpu()) * count
        sample_count += count
    return total_loss / max(sample_count, 1)


@torch.no_grad()
def evaluate_loader(
    model: PaperMHSSATGCN,
    loader: DataLoader[dict[str, Any]],
    loss_function: nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    rows: list[dict[str, Any]] = []
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        logits = model(batch["patches"], batch["masks"], batch["keypoints"])
        loss = loss_function(logits, batch["labels"])
        probabilities = torch.softmax(logits, dim=-1)
        predictions = probabilities.argmax(dim=-1)
        count = int(batch["labels"].shape[0])
        total_loss += float(loss.detach().cpu()) * count
        for index in range(count):
            rows.append(
                {
                    "sample_id": batch["sample_ids"][index],
                    "subject_id": batch["subject_ids"][index],
                    "label": int(batch["labels"][index].detach().cpu()),
                    "prediction": int(predictions[index].detach().cpu()),
                    "probabilities": probabilities[index].detach().cpu().tolist(),
                    "quality_status": batch["quality_status"][index],
                    "apex_boundary_status": batch["apex_boundary_status"][index],
                }
            )
    metrics = classification_metrics(
        [row["label"] for row in rows], [row["prediction"] for row in rows]
    )
    return {
        "loss": total_loss / max(len(rows), 1),
        "metrics": metrics,
        "selection_score": selection_score(metrics["uf1"], metrics["uar"]),
        "predictions": rows,
    }


def model_config_for_candidate(
    base: PaperMHSSATGCNConfig,
    candidate: EvaluationCandidate,
) -> PaperMHSSATGCNConfig:
    graph = replace(
        base.graph,
        mode=candidate.graph_mode,
        fusion_formula=candidate.fusion_formula,
    )
    return replace(
        base,
        attention_variant=candidate.attention_variant,
        graph=graph,
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
    return sha256_file(path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)
    return sha256_file(path)


def _sequence_sha256(values: Sequence[str]) -> str:
    encoded = json.dumps(list(values), separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


def validate_subject_partitions(
    records: Sequence[Mapping[str, Any]],
    *,
    train_sample_ids: Sequence[str],
    validation_sample_ids: Sequence[str],
    test_sample_ids: Sequence[str] | None,
) -> dict[str, Any]:
    by_id = {str(record["sample_id"]): record for record in records}
    if len(by_id) != len(records):
        raise ValueError("Evaluation records contain duplicate sample_id values")
    partitions = {
        "train": [str(value) for value in train_sample_ids],
        "validation": [str(value) for value in validation_sample_ids],
        "test": [str(value) for value in (test_sample_ids or ())],
    }
    for name, sample_ids in partitions.items():
        if name != "test" and not sample_ids:
            raise ValueError(f"{name} partition is empty")
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError(f"{name} partition contains duplicate sample ids")
        missing = sorted(set(sample_ids) - set(by_id))
        if missing:
            raise KeyError(f"{name} partition contains unknown samples: {missing[:5]}")

    names = tuple(partitions)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            overlap = set(partitions[left_name]) & set(partitions[right_name])
            if overlap:
                raise ValueError(
                    f"{left_name}/{right_name} sample overlap: {sorted(overlap)[:5]}"
                )

    subjects = {
        name: sorted(
            {
                str(by_id[sample_id]["subject_id"])
                for sample_id in sample_ids
            }
        )
        for name, sample_ids in partitions.items()
    }
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            overlap = set(subjects[left_name]) & set(subjects[right_name])
            if overlap:
                raise ValueError(
                    f"{left_name}/{right_name} subject leakage: {sorted(overlap)}"
                )
    return {
        name: {
            "sample_count": len(sample_ids),
            "sample_ids_sha256": _sequence_sha256(sample_ids),
            "subject_ids": subjects[name],
        }
        for name, sample_ids in partitions.items()
    }


def run_candidate_training(
    *,
    records: Sequence[Mapping[str, Any]],
    train_sample_ids: Sequence[str],
    validation_sample_ids: Sequence[str],
    test_sample_ids: Sequence[str] | None,
    candidate: EvaluationCandidate,
    base_model_config: PaperMHSSATGCNConfig,
    training_config: PaperTrainingConfig,
    au_adjacency: torch.Tensor,
    seed: int,
    device: torch.device,
    output_dir: Path,
    fold_id: str,
    stage: str,
    source_manifest: Mapping[str, str],
) -> dict[str, Any]:
    run_training_config = replace(training_config, loss=candidate.loss_mode)
    model_config = model_config_for_candidate(base_model_config, candidate)
    partition_contract = validate_subject_partitions(
        records,
        train_sample_ids=train_sample_ids,
        validation_sample_ids=validation_sample_ids,
        test_sample_ids=test_sample_ids,
    )
    run_contract = {
        "fold_id": fold_id,
        "stage": stage,
        "seed": seed,
        "candidate": candidate.as_dict(),
        "candidate_id": candidate.candidate_id,
        "model_config": model_config.as_dict(),
        "training_config": run_training_config.as_dict(),
        "source_manifest": dict(source_manifest),
        "partitions": partition_contract,
    }
    result_path = output_dir / "result.json"
    if result_path.is_file():
        cached = json.loads(result_path.read_text(encoding="utf-8"))
        if cached.get("run_contract") != run_contract:
            raise RuntimeError(
                f"Cached result contract mismatch for {fold_id}/{stage}; "
                "use a distinct output directory"
            )
        return {
            **cached,
            "result_path": result_path.resolve().as_posix(),
            "result_sha256": sha256_file(result_path),
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    set_evaluation_seed(seed)
    train_dataset = PaperV2ArtifactDataset(
        records, sample_ids=train_sample_ids, preload=True
    )
    validation_dataset = PaperV2ArtifactDataset(
        records, sample_ids=validation_sample_ids, preload=True
    )
    train_loader = _loader(
        train_dataset,
        candidate=candidate,
        training_config=training_config,
        shuffle=True,
        seed=seed,
    )
    validation_loader = _loader(
        validation_dataset,
        candidate=candidate,
        training_config=training_config,
        shuffle=False,
        seed=seed,
    )
    model = PaperMHSSATGCN(au_adjacency, model_config).to(device)
    loss_function = build_loss(candidate.loss_mode, train_dataset.records, device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=training_config.lr_step_epochs,
        gamma=training_config.lr_gamma,
    )
    checkpoint_path = output_dir / "best.pt"
    epoch_log_path = output_dir / "epochs.jsonl"
    best_score = float("-inf")
    best_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    epoch_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, training_config.max_epochs + 1):
        training_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            loss_function,
            device,
            training_config.gradient_clip_norm,
        )
        validation = evaluate_loader(
            model, validation_loader, loss_function, device
        )
        score = float(validation["selection_score"])
        validation_loss = float(validation["loss"])
        improved = score > best_score + 1e-12 or (
            abs(score - best_score) <= 1e-12
            and validation_loss < best_loss - 1e-12
        )
        epoch_row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "training_loss": training_loss,
            "validation_loss": validation_loss,
            "validation_metrics": validation["metrics"],
            "validation_selection_score": score,
            "checkpoint_improved": improved,
            "selection_source": "validation_subjects_only",
            "test_used_for_selection": False,
        }
        epoch_rows.append(epoch_row)
        _write_jsonl(epoch_log_path, epoch_rows)
        if improved:
            best_score = score
            best_loss = validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            save_paper_model_checkpoint(
                checkpoint_path,
                model=model,
                training_config=run_training_config,
                metadata={
                    "task_id": "EVAL-ME-002",
                    "evaluation_schema_version": PAPER_EVALUATION_SCHEMA_VERSION,
                    "fold_id": fold_id,
                    "stage": stage,
                    "seed": seed,
                    "candidate": candidate.as_dict(),
                    "candidate_id": candidate.candidate_id,
                    "best_epoch": best_epoch,
                    "best_validation_selection_score": best_score,
                    "best_validation_loss": best_loss,
                    "selection_source": "validation_subjects_only",
                    "test_used_for_selection": False,
                    "source_manifest": dict(source_manifest),
                },
            )
        else:
            epochs_without_improvement += 1
        scheduler.step()
        if (
            epoch >= training_config.min_epochs
            and epochs_without_improvement >= training_config.patience
        ):
            break

    if not checkpoint_path.is_file():
        raise RuntimeError(f"No checkpoint produced for {fold_id}/{stage}")
    best_model, checkpoint = load_paper_model_checkpoint(
        str(checkpoint_path), map_location=device
    )
    best_model = best_model.to(device).eval()
    validation = evaluate_loader(
        best_model, validation_loader, loss_function, device
    )
    validation_prediction_path = output_dir / "validation_predictions.jsonl"
    validation_prediction_sha256 = _write_jsonl(
        validation_prediction_path, validation["predictions"]
    )
    test_result = None
    if test_sample_ids is not None:
        test_dataset = PaperV2ArtifactDataset(
            records, sample_ids=test_sample_ids, preload=True
        )
        test_loader = _loader(
            test_dataset,
            candidate=candidate,
            training_config=training_config,
            shuffle=False,
            seed=seed,
        )
        test_result = evaluate_loader(
            best_model, test_loader, loss_function, device
        )
        test_prediction_path = output_dir / "test_predictions.jsonl"
        test_prediction_sha256 = _write_jsonl(
            test_prediction_path, test_result["predictions"]
        )
        test_result = {
            **test_result,
            "prediction_path": test_prediction_path.resolve().as_posix(),
            "prediction_sha256": test_prediction_sha256,
            "evaluation_count": 1,
            "test_used_for_selection": False,
        }

    duration_seconds = time.perf_counter() - started
    result = {
        "schema_version": PAPER_EVALUATION_SCHEMA_VERSION,
        "task_id": "EVAL-ME-002",
        "status": "completed",
        "fold_id": fold_id,
        "stage": stage,
        "seed": seed,
        "candidate": candidate.as_dict(),
        "candidate_id": candidate.candidate_id,
        "run_contract": run_contract,
        "source_manifest": dict(source_manifest),
        "train_sample_count": len(train_sample_ids),
        "validation_sample_count": len(validation_sample_ids),
        "test_sample_count": len(test_sample_ids or ()),
        "best_epoch": int(checkpoint["best_epoch"]),
        "selection_source": "validation_subjects_only",
        "test_used_for_selection": False,
        "validation": {
            "loss": validation["loss"],
            "metrics": validation["metrics"],
            "selection_score": validation["selection_score"],
            "prediction_path": validation_prediction_path.resolve().as_posix(),
            "prediction_sha256": validation_prediction_sha256,
        },
        "test": test_result,
        "checkpoint": {
            "path": checkpoint_path.resolve().as_posix(),
            "sha256": sha256_file(checkpoint_path),
            "strict_reload_verified": True,
        },
        "epoch_log": {
            "path": epoch_log_path.resolve().as_posix(),
            "sha256": sha256_file(epoch_log_path),
            "epoch_count": len(epoch_rows),
        },
        "duration_seconds": duration_seconds,
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
    }
    result_sha256 = _write_json(result_path, result)
    result["result_path"] = result_path.resolve().as_posix()
    result["result_sha256"] = result_sha256
    return result


def choose_candidate(results: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    completed = [result for result in results if result.get("status") == "completed"]
    if not completed:
        raise RuntimeError("No completed validation candidates are available")
    return max(
        completed,
        key=lambda result: (
            float(result["validation"]["selection_score"]),
            float(result["validation"]["metrics"]["uf1"]),
            float(result["validation"]["metrics"]["uar"]),
            -float(result["validation"]["loss"]),
            str(result["candidate_id"]),
        ),
    )


def subject_bootstrap_ci(
    prediction_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    iterations: int = 2000,
) -> dict[str, list[float]]:
    if iterations <= 0:
        raise ValueError("Bootstrap iterations must be positive")
    by_subject: dict[str, list[Mapping[str, Any]]] = {}
    for row in prediction_rows:
        by_subject.setdefault(str(row["subject_id"]), []).append(row)
    subjects = sorted(by_subject)
    generator = np.random.default_rng(seed)
    values = {"uf1": [], "uar": []}
    for _ in range(iterations):
        sampled = generator.choice(subjects, size=len(subjects), replace=True)
        rows = [row for subject in sampled for row in by_subject[str(subject)]]
        metrics = classification_metrics(
            [int(row["label"]) for row in rows],
            [int(row["prediction"]) for row in rows],
        )
        values["uf1"].append(metrics["uf1"])
        values["uar"].append(metrics["uar"])
    return {
        metric: [
            float(np.quantile(metric_values, 0.025)),
            float(np.quantile(metric_values, 0.975)),
        ]
        for metric, metric_values in values.items()
    }


def aggregate_seed_predictions(
    prediction_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    bootstrap_iterations: int = 2000,
) -> dict[str, Any]:
    pooled = classification_metrics(
        [int(row["label"]) for row in prediction_rows],
        [int(row["prediction"]) for row in prediction_rows],
    )
    subject_metrics: dict[str, Any] = {}
    for subject in sorted({str(row["subject_id"]) for row in prediction_rows}):
        rows = [row for row in prediction_rows if row["subject_id"] == subject]
        subject_metrics[subject] = classification_metrics(
            [int(row["label"]) for row in rows],
            [int(row["prediction"]) for row in rows],
        )
    mean_fold = {
        metric: float(
            np.mean([values[metric] for values in subject_metrics.values()])
        )
        for metric in ("uf1", "uar", "accuracy", "balanced_accuracy")
    }
    zero_score_folds = [
        subject
        for subject, values in subject_metrics.items()
        if values["uf1"] <= 0.0 or values["uar"] <= 0.0
    ]
    strata: dict[str, Any] = {}
    for field in ("quality_status", "apex_boundary_status"):
        for value in sorted({str(row[field]) for row in prediction_rows}):
            rows = [row for row in prediction_rows if row[field] == value]
            strata[f"{field}={value}"] = classification_metrics(
                [int(row["label"]) for row in rows],
                [int(row["prediction"]) for row in rows],
            )
    return {
        "seed": seed,
        "pooled": pooled,
        "mean_fold": mean_fold,
        "subject_metrics": subject_metrics,
        "zero_score_folds": zero_score_folds,
        "bootstrap_95_ci": subject_bootstrap_ci(
            prediction_rows, seed=seed, iterations=bootstrap_iterations
        ),
        "quality_strata": strata,
    }
