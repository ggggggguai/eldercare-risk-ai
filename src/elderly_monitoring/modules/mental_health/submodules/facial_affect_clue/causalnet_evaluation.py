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

from .causalnet import (
    CausalNet,
    CausalNetConfig,
    code_hash,
    load_causalnet_checkpoint,
    save_causalnet_checkpoint,
)
from .causalnet_dataset import CausalNetArtifactDataset, collate_causalnet
from .metrics import classification_metrics


CAUSALNET_EVALUATION_SCHEMA_VERSION = "causalnet_strict_nested_loso_v1"


@dataclass(frozen=True)
class CausalNetTrainingConfig:
    optimizer: str = "adam"
    learning_rate: float = 5e-5
    max_epochs: int = 200
    min_epochs: int = 20
    patience: int = 20
    batch_size: int = 256
    weight_decay: float = 0.0
    gradient_clip_norm: float = 1.0
    num_workers: int = 0
    loss: str = "weighted_cross_entropy"
    lr_step_epochs: int = 50
    lr_gamma: float = 0.5

    def __post_init__(self) -> None:
        if self.optimizer != "adam":
            raise ValueError("Formal CausalNet evaluation currently freezes optimizer=adam")
        if self.loss not in {"cross_entropy", "weighted_cross_entropy", "focal"}:
            raise ValueError(f"Unsupported CausalNet loss: {self.loss}")
        if self.learning_rate <= 0 or self.max_epochs < 1 or self.batch_size < 1:
            raise ValueError("Training hyperparameters must be positive")
        if not 1 <= self.min_epochs <= self.max_epochs:
            raise ValueError("min_epochs must be within [1, max_epochs]")
        if self.patience < 1 or self.lr_step_epochs < 1:
            raise ValueError("patience and lr_step_epochs must be positive")
        if not 0 < self.lr_gamma <= 1 or self.gradient_clip_norm <= 0:
            raise ValueError("lr_gamma and gradient_clip_norm are out of range")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> CausalNetTrainingConfig:
        return cls(**dict(values))

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        losses = nn.functional.cross_entropy(logits, targets, reduction="none")
        probabilities = torch.exp(-losses)
        return (((1.0 - probabilities) ** self.gamma) * losses).mean()


def selection_score(uf1: float, uar: float) -> float:
    return 0.0 if uf1 + uar <= 0 else 2.0 * uf1 * uar / (uf1 + uar)


def set_evaluation_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
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
    test_sample_ids: Sequence[str],
) -> dict[str, Any]:
    by_id = {str(record["sample_id"]): record for record in records}
    if len(by_id) != len(records):
        raise ValueError("Evaluation records contain duplicate sample_id values")
    partitions = {
        "train": [str(value) for value in train_sample_ids],
        "validation": [str(value) for value in validation_sample_ids],
        "test": [str(value) for value in test_sample_ids],
    }
    for name, sample_ids in partitions.items():
        if not sample_ids:
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
        name: sorted({str(by_id[sample_id]["subject_id"]) for sample_id in sample_ids})
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
    if mode == "cross_entropy":
        return nn.CrossEntropyLoss()
    if mode == "weighted_cross_entropy":
        return nn.CrossEntropyLoss(weight=class_weights(records).to(device))
    if mode == "focal":
        return FocalLoss()
    raise ValueError(f"Unsupported CausalNet loss: {mode}")


def _loader(
    dataset: CausalNetArtifactDataset,
    *,
    training_config: CausalNetTrainingConfig,
    shuffle: bool,
    seed: int,
) -> DataLoader[dict[str, Any]]:
    return DataLoader(
        dataset,
        batch_size=training_config.batch_size,
        shuffle=shuffle,
        num_workers=training_config.num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(seed),
        collate_fn=collate_causalnet,
    )


def train_epoch(
    model: CausalNet,
    loader: DataLoader[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    device: torch.device,
    gradient_clip_norm: float,
) -> float:
    model.train()
    total_loss = 0.0
    sample_count = 0
    for batch in loader:
        inputs = batch["inputs"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(model(inputs), labels)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite CausalNet training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        if not all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ):
            raise RuntimeError("Non-finite CausalNet training gradients")
        optimizer.step()
        count = int(labels.shape[0])
        total_loss += float(loss.detach().cpu()) * count
        sample_count += count
    return total_loss / max(sample_count, 1)


@torch.no_grad()
def evaluate_loader(
    model: CausalNet,
    loader: DataLoader[dict[str, Any]],
    loss_function: nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    rows: list[dict[str, Any]] = []
    for batch in loader:
        inputs = batch["inputs"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        logits = model(inputs)
        loss = loss_function(logits, labels)
        probabilities = torch.softmax(logits, dim=-1)
        predictions = probabilities.argmax(dim=-1)
        count = int(labels.shape[0])
        total_loss += float(loss.detach().cpu()) * count
        for index in range(count):
            metadata = batch["metadata"][index]
            rows.append(
                {
                    "sample_id": batch["sample_ids"][index],
                    "subject_id": batch["subject_ids"][index],
                    "label": int(labels[index].detach().cpu()),
                    "prediction": int(predictions[index].detach().cpu()),
                    "probabilities": probabilities[index].detach().cpu().tolist(),
                    "quality_status": str(metadata.get("quality_status", "unknown")),
                    "spatial_variant": str(metadata.get("spatial_variant", "unknown")),
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


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }


def run_nested_loso_fold(
    *,
    records: Sequence[Mapping[str, Any]],
    train_sample_ids: Sequence[str],
    validation_sample_ids: Sequence[str],
    test_sample_ids: Sequence[str],
    model_config: CausalNetConfig,
    training_config: CausalNetTrainingConfig,
    seed: int,
    device: torch.device,
    output_dir: Path,
    fold_id: str,
    input_hashes: Mapping[str, str],
    evaluation_bundle_hash: str,
) -> dict[str, Any]:
    partition_contract = validate_subject_partitions(
        records,
        train_sample_ids=train_sample_ids,
        validation_sample_ids=validation_sample_ids,
        test_sample_ids=test_sample_ids,
    )
    run_contract = {
        "schema_version": CAUSALNET_EVALUATION_SCHEMA_VERSION,
        "task_id": "EVAL-ME-004",
        "fold_id": fold_id,
        "seed": seed,
        "model_config": model_config.as_dict(),
        "training_config": training_config.as_dict(),
        "input_hashes": dict(input_hashes),
        "evaluation_bundle_hash": evaluation_bundle_hash,
        "partitions": partition_contract,
        "configuration_selection": "frozen_before_outer_test",
        "checkpoint_selection": "validation_uf1_uar_harmonic_then_loss",
        "test_used_for_selection": False,
    }
    result_path = output_dir / "result.json"
    if result_path.is_file():
        cached = json.loads(result_path.read_text(encoding="utf-8"))
        if cached.get("run_contract") != run_contract:
            raise RuntimeError(
                f"Cached result contract mismatch for {fold_id}/seed={seed}"
            )
        return {
            **cached,
            "result_path": result_path.resolve().as_posix(),
            "result_sha256": sha256_file(result_path),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    set_evaluation_seed(seed)
    train_dataset = CausalNetArtifactDataset(
        records, sample_ids=train_sample_ids, preload=True
    )
    validation_dataset = CausalNetArtifactDataset(
        records, sample_ids=validation_sample_ids, preload=True
    )
    train_loader = _loader(
        train_dataset,
        training_config=training_config,
        shuffle=True,
        seed=seed,
    )
    validation_loader = _loader(
        validation_dataset,
        training_config=training_config,
        shuffle=False,
        seed=seed,
    )
    model = CausalNet(model_config).to(device)
    loss_function = build_loss(training_config.loss, train_dataset.records, device)
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
    best_score = float("-inf")
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    epoch_rows: list[dict[str, Any]] = []
    epoch_log_path = output_dir / "epochs.jsonl"
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
        epoch_rows.append(
            {
                "epoch": epoch,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "training_loss": training_loss,
                "validation_loss": validation_loss,
                "validation_metrics": validation["metrics"],
                "validation_selection_score": score,
                "checkpoint_improved": improved,
                "selection_source": "training_side_validation_subjects_only",
                "test_used_for_selection": False,
            }
        )
        _write_jsonl(epoch_log_path, epoch_rows)
        if improved:
            best_score = score
            best_loss = validation_loss
            best_epoch = epoch
            best_state = _cpu_state_dict(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        scheduler.step()
        if (
            epoch >= training_config.min_epochs
            and epochs_without_improvement >= training_config.patience
        ):
            break

    if best_state is None:
        raise RuntimeError(f"No validation-selected state produced for {fold_id}")
    model.load_state_dict(best_state, strict=True)
    del best_state, optimizer, scheduler
    probe_inputs = validation_dataset[0]["inputs"].unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        pre_save_logits = model(probe_inputs).detach().cpu()
    checkpoint_path = output_dir / "best.pt"
    checkpoint_training_config = {
        **training_config.as_dict(),
        "task_id": "EVAL-ME-004",
        "evaluation_schema_version": CAUSALNET_EVALUATION_SCHEMA_VERSION,
        "fold_id": fold_id,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_selection_score": best_score,
        "best_validation_loss": best_loss,
        "selection_source": "training_side_validation_subjects_only",
        "configuration_selection": "frozen_before_outer_test",
        "test_used_for_selection": False,
        "evaluation_bundle_hash": evaluation_bundle_hash,
        "input_hashes": dict(input_hashes),
    }
    model = model.cpu()
    save_causalnet_checkpoint(
        checkpoint_path,
        model,
        training_config=checkpoint_training_config,
        source_manifest_hash=input_hashes["artifact_manifest_sha256"],
        preprocessing_config_hash=input_hashes["preprocessing_config_sha256"],
        source_code_hash=code_hash(),
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    best_model, checkpoint = load_causalnet_checkpoint(
        checkpoint_path, map_location=device
    )
    best_model = best_model.to(device).eval()
    with torch.no_grad():
        post_load_logits = best_model(probe_inputs).detach().cpu()
    strict_reload_difference = float(
        (pre_save_logits - post_load_logits).abs().max().item()
    )
    if strict_reload_difference != 0.0:
        raise RuntimeError("CausalNet strict checkpoint reload changed logits")

    validation = evaluate_loader(
        best_model, validation_loader, loss_function, device
    )
    validation_prediction_path = output_dir / "validation_predictions.jsonl"
    validation_prediction_sha256 = _write_jsonl(
        validation_prediction_path, validation["predictions"]
    )

    # The outer test dataset is first constructed after the validation checkpoint is fixed.
    test_dataset = CausalNetArtifactDataset(
        records, sample_ids=test_sample_ids, preload=True
    )
    test_loader = _loader(
        test_dataset,
        training_config=training_config,
        shuffle=False,
        seed=seed,
    )
    test = evaluate_loader(best_model, test_loader, loss_function, device)
    for row in test["predictions"]:
        row["fold_id"] = fold_id
        row["seed"] = seed
        row["test_used_for_selection"] = False
    test_prediction_path = output_dir / "test_predictions.jsonl"
    test_prediction_sha256 = _write_jsonl(
        test_prediction_path, test["predictions"]
    )
    peak_gpu_memory_bytes = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    duration_seconds = time.perf_counter() - started
    result = {
        "schema_version": CAUSALNET_EVALUATION_SCHEMA_VERSION,
        "task_id": "EVAL-ME-004",
        "status": "completed",
        "fold_id": fold_id,
        "seed": seed,
        "run_contract": run_contract,
        "best_epoch": int(checkpoint["training_config"]["best_epoch"]),
        "selection_source": "training_side_validation_subjects_only",
        "configuration_selection": "frozen_before_outer_test",
        "test_used_for_selection": False,
        "validation": {
            "loss": validation["loss"],
            "metrics": validation["metrics"],
            "selection_score": validation["selection_score"],
            "prediction_path": validation_prediction_path.resolve().as_posix(),
            "prediction_sha256": validation_prediction_sha256,
        },
        "test": {
            "loss": test["loss"],
            "metrics": test["metrics"],
            "selection_score": test["selection_score"],
            "predictions": test["predictions"],
            "prediction_path": test_prediction_path.resolve().as_posix(),
            "prediction_sha256": test_prediction_sha256,
            "evaluation_count": 1,
            "test_used_for_selection": False,
        },
        "checkpoint": {
            "path": checkpoint_path.resolve().as_posix(),
            "sha256": sha256_file(checkpoint_path),
            "strict_reload_verified": True,
            "strict_reload_max_logit_difference": strict_reload_difference,
        },
        "epoch_log": {
            "path": epoch_log_path.resolve().as_posix(),
            "sha256": sha256_file(epoch_log_path),
            "epoch_count": len(epoch_rows),
        },
        "duration_seconds": duration_seconds,
        "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
        "environment": {
            "device": str(device),
            "cuda_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "torch_version": torch.__version__,
        },
    }
    result_sha256 = _write_json(result_path, result)
    result["result_path"] = result_path.resolve().as_posix()
    result["result_sha256"] = result_sha256
    del best_model, probe_inputs, pre_save_logits, post_load_logits
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def subject_bootstrap_ci(
    prediction_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    iterations: int,
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
    bootstrap_iterations: int,
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
    return {
        "seed": seed,
        "pooled": pooled,
        "mean_fold": mean_fold,
        "subject_metrics": subject_metrics,
        "zero_score_folds": zero_score_folds,
        "bootstrap_95_ci": subject_bootstrap_ci(
            prediction_rows,
            seed=seed,
            iterations=bootstrap_iterations,
        ),
    }


def release_decision(seed_summaries: Sequence[Mapping[str, Any]]) -> str:
    uf1_values = [float(summary["pooled"]["uf1"]) for summary in seed_summaries]
    uar_values = [float(summary["pooled"]["uar"]) for summary in seed_summaries]
    no_zero_folds = all(not summary["zero_score_folds"] for summary in seed_summaries)
    stable = np.std(uf1_values) <= 0.05 and np.std(uar_values) <= 0.05
    if (
        all(value >= 0.70 for value in uf1_values)
        and all(value >= 0.70 for value in uar_values)
        and stable
        and no_zero_folds
    ):
        return "candidate"
    if np.mean(uf1_values) >= 0.60 and np.mean(uar_values) >= 0.60 and no_zero_folds:
        return "research_baseline"
    return "rejected"


def aggregate_final_results(
    *,
    final_results: Sequence[Mapping[str, Any]],
    expected_sample_ids: set[str],
    seeds: Sequence[int],
    bootstrap_iterations: int,
    output_root: Path,
) -> dict[str, Any]:
    seed_summaries: list[dict[str, Any]] = []
    prediction_files: list[dict[str, Any]] = []
    for seed in seeds:
        seed_results = [result for result in final_results if result["seed"] == seed]
        rows = [row for result in seed_results for row in result["test"]["predictions"]]
        sample_ids = [str(row["sample_id"]) for row in rows]
        if len(sample_ids) != len(set(sample_ids)):
            raise RuntimeError(f"Seed {seed} contains duplicate outer-test predictions")
        if set(sample_ids) != expected_sample_ids:
            raise RuntimeError(f"Seed {seed} outer-test sample coverage mismatch")
        prediction_path = output_root / "aggregate" / f"predictions_seed_{seed}.jsonl"
        prediction_sha256 = _write_jsonl(prediction_path, rows)
        summary = aggregate_seed_predictions(
            rows,
            seed=seed,
            bootstrap_iterations=bootstrap_iterations,
        )
        summary["fold_count"] = len(seed_results)
        summary["prediction_path"] = prediction_path.resolve().as_posix()
        summary["prediction_sha256"] = prediction_sha256
        seed_summaries.append(summary)
        prediction_files.append(
            {
                "seed": seed,
                "path": prediction_path.resolve().as_posix(),
                "sha256": prediction_sha256,
            }
        )
    metric_names = ("uf1", "uar", "accuracy", "balanced_accuracy")
    seed_statistics = {
        metric: {
            "mean": float(
                np.mean([summary["pooled"][metric] for summary in seed_summaries])
            ),
            "std": float(
                np.std([summary["pooled"][metric] for summary in seed_summaries])
            ),
            "values": [summary["pooled"][metric] for summary in seed_summaries],
        }
        for metric in metric_names
    }
    decision = release_decision(seed_summaries)
    return {
        "schema_version": CAUSALNET_EVALUATION_SCHEMA_VERSION,
        "task_id": "EVAL-ME-004",
        "status": "completed",
        "seed_count": len(seed_summaries),
        "final_run_count": len(final_results),
        "bootstrap_iterations": bootstrap_iterations,
        "seed_summaries": seed_summaries,
        "seed_statistics": seed_statistics,
        "prediction_files": prediction_files,
        "release_decision": decision,
        "deployment_eligible": decision == "candidate",
        "test_used_for_selection": False,
        "paper_reproduction_claim": False,
    }


def forward_backward_smoke(
    model: CausalNet,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    *,
    device: torch.device,
) -> dict[str, Any]:
    model = model.to(device)
    inputs = inputs.to(device=device, dtype=torch.float32)
    labels = labels.to(device=device, dtype=torch.long)
    model.train()
    logits = model(inputs)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    if not torch.isfinite(loss):
        raise RuntimeError("CausalNet smoke loss is non-finite")
    loss.backward()
    gradient_max = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            if not torch.isfinite(parameter.grad).all():
                raise RuntimeError("CausalNet smoke gradient is non-finite")
            gradient_max = max(
                gradient_max,
                float(parameter.grad.detach().abs().max().cpu()),
            )
    return {
        "batch_shape": list(inputs.shape),
        "logits_shape": list(logits.shape),
        "loss": float(loss.detach().cpu()),
        "gradient_max": gradient_max,
        "device": str(device),
    }


def checkpoint_smoke(
    model: CausalNet,
    path: Path,
    *,
    source_manifest_hash: str,
    preprocessing_config_hash: str,
    source_code_hash: str,
) -> dict[str, Any]:
    metadata = save_causalnet_checkpoint(
        path,
        model,
        training_config={"smoke_only": True, "training_started": False},
        source_manifest_hash=source_manifest_hash,
        preprocessing_config_hash=preprocessing_config_hash,
        source_code_hash=source_code_hash,
    )
    return {"path": path.resolve().as_posix(), "metadata": metadata}
