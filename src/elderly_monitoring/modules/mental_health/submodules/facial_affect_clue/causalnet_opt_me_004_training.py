from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from .causalnet import CausalNet
from .causalnet_dataset import CausalNetArtifactDataset, collate_causalnet
from .causalnet_opt_me_004 import (
    CANDIDATE_REGISTRY,
    CONSISTENCY_WEIGHT,
    EMA_DECAY,
    ExponentialMovingAverage,
    REDUCED_CAUSALNET_CONFIG,
    augment_route_consistency_batch,
    build_candidate_loss,
    build_me4_causalnet,
    symmetric_kl_divergence,
    train_subject_counts,
)
from .metrics import classification_metrics


CHECKPOINT_SCHEMA = "causalnet_opt_me_004_checkpoint_v1"


@dataclass(frozen=True)
class ME4TrainingConfig:
    optimizer: str = "Adam"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 64
    gradient_clip_norm: float = 1.0
    max_epochs: int = 12
    min_epochs: int = 1
    patience: int = 12


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _records(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "sample_id": str(row["sample_id"]),
            "subject_id": str(row["subject_id"]),
            "label": int(row["label"]),
            "artifact_path": str(row["causalnet_resolved_path"]),
            "external_label_authority": "casme2_three_class_crosswalk_v1",
            "artifact_stored_label": -1,
        }
        for row in rows
    ]


def _loader(
    rows: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    generator: torch.Generator,
    training: bool,
) -> DataLoader[dict[str, Any]]:
    return DataLoader(
        CausalNetArtifactDataset(_records(rows), preload=True),
        batch_size=batch_size,
        shuffle=training,
        num_workers=0,
        collate_fn=collate_causalnet,
        generator=generator,
    )


def _evaluate(
    model: CausalNet, loader: DataLoader[dict[str, Any]], device: torch.device
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["inputs"].to(device))
            probabilities = torch.softmax(logits, dim=1)
            predictions = probabilities.argmax(dim=1)
            for index, sample_id in enumerate(batch["sample_ids"]):
                rows.append(
                    {
                        "sample_id": str(sample_id),
                        "subject_id": str(batch["subject_ids"][index]),
                        "label": int(batch["labels"][index]),
                        "prediction": int(predictions[index]),
                        "logits": logits[index].detach().cpu().tolist(),
                        "probabilities": probabilities[index].detach().cpu().tolist(),
                    }
                )
    metrics = classification_metrics(
        [int(row["label"]) for row in rows],
        [int(row["prediction"]) for row in rows],
    )
    return metrics, rows


def _cpu_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in state.items()}


def _checkpoint_payload(
    *,
    model: CausalNet,
    evaluation_state: Mapping[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    ema: ExponentialMovingAverage | None,
    candidate_id: str,
    training_config: ME4TrainingConfig,
    evidence_hashes: Mapping[str, str],
    epoch: int,
    best_epoch: int,
    best_score: float,
    epochs_without_improvement: int,
    loader_generator: torch.Generator,
    ssl_generator: torch.Generator,
) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_SCHEMA,
        "candidate_id": candidate_id,
        "candidate_contract": asdict(CANDIDATE_REGISTRY[candidate_id]),
        "model_config": REDUCED_CAUSALNET_CONFIG.as_dict(),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "training_config": asdict(training_config),
        "evidence_hashes": dict(evidence_hashes),
        "epoch": int(epoch),
        "best_epoch": int(best_epoch),
        "best_score": float(best_score),
        "epochs_without_improvement": int(epochs_without_improvement),
        "model_state_dict": _cpu_state(model.state_dict()),
        "evaluation_state_dict": _cpu_state(evaluation_state),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "ema_state_dict": None if ema is None else ema.state_dict(),
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "loader_generator": loader_generator.get_state(),
            "ssl_generator": ssl_generator.get_state(),
        },
    }


def _save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(payload), path)


def load_me4_checkpoint(
    path: Path, *, map_location: str | torch.device = "cpu"
) -> tuple[CausalNet, dict[str, Any]]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    required = {
        "schema_version",
        "candidate_id",
        "candidate_contract",
        "model_config",
        "parameter_count",
        "training_config",
        "evidence_hashes",
        "model_state_dict",
        "evaluation_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "ema_state_dict",
        "rng_state",
    }
    if required - set(payload):
        raise ValueError(f"ME4 checkpoint metadata missing: {sorted(required - set(payload))}")
    if payload["schema_version"] != CHECKPOINT_SCHEMA:
        raise ValueError("unsupported ME4 checkpoint schema")
    candidate_id = str(payload["candidate_id"])
    if candidate_id not in CANDIDATE_REGISTRY:
        raise ValueError("checkpoint candidate is outside frozen M0-M5 registry")
    if payload["candidate_contract"] != asdict(CANDIDATE_REGISTRY[candidate_id]):
        raise ValueError("checkpoint candidate contract drift")
    if payload["model_config"] != REDUCED_CAUSALNET_CONFIG.as_dict():
        raise ValueError("checkpoint model configuration drift")
    if int(payload["parameter_count"]) != 623963:
        raise ValueError("checkpoint parameter count drift")
    model = build_me4_causalnet()
    model.load_state_dict(payload["evaluation_state_dict"], strict=True)
    return model, payload


def _evaluation_state(
    model: CausalNet, ema: ExponentialMovingAverage | None
) -> dict[str, torch.Tensor]:
    if ema is None:
        return dict(model.state_dict())
    return dict(ema.shadow)


def train_candidate_fold(
    *,
    train_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    candidate_id: str,
    seed: int,
    device: torch.device,
    output_dir: Path,
    training_config: ME4TrainingConfig,
    evidence_hashes: Mapping[str, str],
    resume: bool = True,
) -> dict[str, Any]:
    if candidate_id not in CANDIDATE_REGISTRY:
        raise ValueError(f"unknown candidate: {candidate_id}")
    if {str(row["subject_id"]) for row in train_rows} & {
        str(row["subject_id"]) for row in validation_rows
    }:
        raise ValueError("subject leakage between train and validation")
    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    loader_generator = torch.Generator().manual_seed(seed)
    ssl_generator = torch.Generator().manual_seed(seed ^ 0x5EED004)
    train_loader = _loader(
        train_rows,
        batch_size=training_config.batch_size,
        generator=loader_generator,
        training=True,
    )
    validation_loader = _loader(
        validation_rows,
        batch_size=training_config.batch_size,
        generator=torch.Generator().manual_seed(seed),
        training=False,
    )
    counts = tuple(sum(int(row["label"]) == label for row in train_rows) for label in (0, 1, 2))
    subject_counts = train_subject_counts(train_rows)
    contract = CANDIDATE_REGISTRY[candidate_id]
    loss_function = build_candidate_loss(candidate_id, class_counts=counts).to(device)
    model = build_me4_causalnet().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    ema = ExponentialMovingAverage(model, EMA_DECAY) if contract.use_ema else None
    best_score = -float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    start_epoch = 1
    epoch_rows: list[dict[str, Any]] = []
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    recovered = False

    if resume and last_path.is_file() and not (output_dir / "result.json").is_file():
        payload = torch.load(last_path, map_location=device, weights_only=False)
        if payload.get("evidence_hashes") != dict(evidence_hashes) or payload.get("training_config") != asdict(training_config):
            raise RuntimeError("technical recovery run-lock/config drift")
        if payload.get("candidate_contract") != asdict(contract):
            raise RuntimeError("technical recovery candidate drift")
        model.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])
        if ema is not None:
            if payload["ema_state_dict"] is None:
                raise RuntimeError("technical recovery EMA state missing")
            ema.load_state_dict(payload["ema_state_dict"], device)
        random.setstate(payload["rng_state"]["python"])
        np.random.set_state(payload["rng_state"]["numpy"])
        torch.set_rng_state(payload["rng_state"]["torch"].cpu())
        if torch.cuda.is_available() and payload["rng_state"]["cuda"]:
            torch.cuda.set_rng_state_all(payload["rng_state"]["cuda"])
        loader_generator.set_state(payload["rng_state"]["loader_generator"].cpu())
        ssl_generator.set_state(payload["rng_state"]["ssl_generator"].cpu())
        best_score = float(payload["best_score"])
        best_epoch = int(payload["best_epoch"])
        epochs_without_improvement = int(payload["epochs_without_improvement"])
        start_epoch = int(payload["epoch"]) + 1
        epoch_log = output_dir / "epochs.jsonl"
        if epoch_log.is_file():
            epoch_rows = [json.loads(line) for line in epoch_log.read_text(encoding="utf-8").splitlines() if line]
        recovered = True

    for epoch in range(start_epoch, training_config.max_epochs + 1):
        model.train()
        total_losses: list[float] = []
        classification_losses: list[float] = []
        consistency_losses: list[float] = []
        max_gradient = 0.0
        for batch in train_loader:
            original_cpu = batch["inputs"]
            inputs = original_cpu.to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            classification_loss = loss_function(
                logits,
                labels,
                epoch=epoch,
                subject_ids=batch["subject_ids"],
                subject_counts=subject_counts,
            )
            consistency_loss = torch.zeros((), device=device)
            if contract.route_consistency_ssl:
                augmented = augment_route_consistency_batch(original_cpu, ssl_generator).to(device)
                augmented_logits = model(augmented)
                consistency_loss = symmetric_kl_divergence(logits, augmented_logits)
            loss = classification_loss + CONSISTENCY_WEIGHT * consistency_loss
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training loss")
            loss.backward()
            for parameter in model.parameters():
                if parameter.grad is not None:
                    if not torch.isfinite(parameter.grad).all():
                        raise RuntimeError("non-finite gradient")
                    max_gradient = max(max_gradient, float(parameter.grad.detach().abs().max()))
            torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.gradient_clip_norm)
            optimizer.step()
            if ema is not None:
                ema.update(model)
            total_losses.append(float(loss.detach().cpu()))
            classification_losses.append(float(classification_loss.detach().cpu()))
            consistency_losses.append(float(consistency_loss.detach().cpu()))
        scheduler.step()

        eval_model = build_me4_causalnet().to(device)
        eval_model.load_state_dict(_evaluation_state(model, ema), strict=True)
        validation_metrics, _ = _evaluate(eval_model, validation_loader, device)
        score = min(float(validation_metrics["uf1"]), float(validation_metrics["uar"]))
        improved = score > best_score + 1e-12
        if improved:
            best_score = score
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        epoch_rows.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(total_losses)),
                "classification_loss": float(np.mean(classification_losses)),
                "consistency_loss": float(np.mean(consistency_losses)),
                "max_abs_gradient_before_clip": max_gradient,
                "validation_metrics": validation_metrics,
                "selection_score": score,
                "checkpoint_updated": improved,
                "learning_rate": float(scheduler.get_last_lr()[0]),
            }
        )
        epoch_log_path = output_dir / "epochs.jsonl"
        epoch_log_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in epoch_rows),
            encoding="utf-8",
        )
        payload = _checkpoint_payload(
            model=model,
            evaluation_state=_evaluation_state(model, ema),
            optimizer=optimizer,
            scheduler=scheduler,
            ema=ema,
            candidate_id=candidate_id,
            training_config=training_config,
            evidence_hashes=evidence_hashes,
            epoch=epoch,
            best_epoch=best_epoch,
            best_score=best_score,
            epochs_without_improvement=epochs_without_improvement,
            loader_generator=loader_generator,
            ssl_generator=ssl_generator,
        )
        _save(last_path, payload)
        if improved:
            _save(best_path, payload)
        if epoch >= training_config.min_epochs and epochs_without_improvement >= training_config.patience:
            break

    loaded, metadata = load_me4_checkpoint(best_path, map_location=device)
    loaded = loaded.to(device)
    best_metrics, predictions = _evaluate(loaded, validation_loader, device)
    direct = build_me4_causalnet().to(device)
    direct.load_state_dict(metadata["evaluation_state_dict"], strict=True)
    _, direct_predictions = _evaluate(direct, validation_loader, device)
    by_id = {row["sample_id"]: np.asarray(row["probabilities"]) for row in predictions}
    direct_by_id = {row["sample_id"]: np.asarray(row["probabilities"]) for row in direct_predictions}
    reload_difference = max(float(np.max(np.abs(by_id[key] - direct_by_id[key]))) for key in by_id)
    if reload_difference > 1e-7:
        raise RuntimeError(f"checkpoint reload changed probabilities: {reload_difference}")
    prediction_path = output_dir / "validation_predictions.jsonl"
    prediction_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions),
        encoding="utf-8",
    )
    result = {
        "schema_version": "causalnet_opt_me_004_fold_result_v1",
        "candidate_id": candidate_id,
        "candidate_contract": asdict(contract),
        "seed": seed,
        "status": "completed",
        "best_epoch": best_epoch,
        "best_selection_score": best_score,
        "validation_metrics": best_metrics,
        "train_sample_count": len(train_rows),
        "validation_sample_count": len(validation_rows),
        "train_subject_ids": sorted({str(row["subject_id"]) for row in train_rows}),
        "validation_subject_ids": sorted({str(row["subject_id"]) for row in validation_rows}),
        "checkpoint_path": best_path.as_posix(),
        "last_checkpoint_path": last_path.as_posix(),
        "epoch_log_path": (output_dir / "epochs.jsonl").as_posix(),
        "prediction_path": prediction_path.as_posix(),
        "optimizer_state_saved": "optimizer_state_dict" in metadata,
        "scheduler_state_saved": "scheduler_state_dict" in metadata,
        "ema_state_saved": metadata["ema_state_dict"] is not None,
        "strict_reload_max_probability_difference": reload_difference,
        "technical_recovery_used": recovered,
        "outer_test_used": False,
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


__all__ = [
    "CHECKPOINT_SCHEMA",
    "ME4TrainingConfig",
    "load_me4_checkpoint",
    "set_seed",
    "train_candidate_fold",
]
