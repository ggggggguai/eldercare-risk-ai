from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .causalnet import CausalNet
from .causalnet_opt_me_004_training import _cpu_state, _evaluate, _loader, set_seed
from .causalnet_opt_me_005 import (
    CANDIDATE_REGISTRY,
    POSITIVE_MARGIN_WEIGHT,
    REDUCED_CAUSALNET_CONFIG,
    augment_route_consistency_batch,
    build_candidate_loss,
    build_me5_causalnet,
    class_weighted_consistency_loss,
    consistency_ramp,
    positive_margin_loss,
)
from .causalnet_opt_me_004 import train_subject_counts


CHECKPOINT_SCHEMA = "causalnet_opt_me_005_checkpoint_v1"


@dataclass(frozen=True)
class ME5TrainingConfig:
    optimizer: str = "Adam"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 64
    gradient_clip_norm: float = 1.0
    max_epochs: int = 12
    min_epochs: int = 1
    patience: int = 12


def _checkpoint_payload(
    *,
    model: CausalNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    candidate_id: str,
    training_config: ME5TrainingConfig,
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
        "evaluation_state_dict": _cpu_state(model.state_dict()),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "ema_state_dict": None,
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


def load_me5_checkpoint(
    path: Path, *, map_location: str | torch.device = "cpu"
) -> tuple[CausalNet, dict[str, Any]]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    required = {
        "schema_version", "candidate_id", "candidate_contract", "model_config",
        "parameter_count", "training_config", "evidence_hashes", "model_state_dict",
        "evaluation_state_dict", "optimizer_state_dict", "scheduler_state_dict",
        "ema_state_dict", "rng_state",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"ME5 checkpoint metadata missing: {sorted(missing)}")
    if payload["schema_version"] != CHECKPOINT_SCHEMA:
        raise ValueError("unsupported ME5 checkpoint schema")
    candidate_id = str(payload["candidate_id"])
    if candidate_id not in CANDIDATE_REGISTRY:
        raise ValueError("checkpoint candidate is outside frozen B0-C5 registry")
    if payload["candidate_contract"] != asdict(CANDIDATE_REGISTRY[candidate_id]):
        raise ValueError("checkpoint candidate contract drift")
    if payload["model_config"] != REDUCED_CAUSALNET_CONFIG.as_dict():
        raise ValueError("checkpoint model configuration drift")
    if int(payload["parameter_count"]) != 623963:
        raise ValueError("checkpoint parameter count drift")
    if payload["ema_state_dict"] is not None:
        raise ValueError("OPT-ME-005 forbids EMA")
    model = build_me5_causalnet()
    model.load_state_dict(payload["evaluation_state_dict"], strict=True)
    return model, payload


def train_candidate_fold(
    *,
    train_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    candidate_id: str,
    seed: int,
    device: torch.device,
    output_dir: Path,
    training_config: ME5TrainingConfig,
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
    ssl_generator = torch.Generator().manual_seed(seed ^ 0x5EED005)
    train_loader = _loader(
        train_rows, batch_size=training_config.batch_size,
        generator=loader_generator, training=True,
    )
    validation_loader = _loader(
        validation_rows, batch_size=training_config.batch_size,
        generator=torch.Generator().manual_seed(seed), training=False,
    )
    counts = tuple(sum(int(row["label"]) == label for row in train_rows) for label in (0, 1, 2))
    subject_counts = train_subject_counts(train_rows)
    contract = CANDIDATE_REGISTRY[candidate_id]
    loss_function = build_candidate_loss(class_counts=counts).to(device)
    model = build_me5_causalnet().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    best_score = -float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    start_epoch = 1
    epoch_rows: list[dict[str, Any]] = []
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    result_path = output_dir / "result.json"
    recovered = False

    if resume and last_path.is_file() and not result_path.is_file():
        payload = torch.load(last_path, map_location=device, weights_only=False)
        if payload.get("evidence_hashes") != dict(evidence_hashes):
            raise RuntimeError("technical recovery run-lock drift")
        if payload.get("training_config") != asdict(training_config):
            raise RuntimeError("technical recovery training config drift")
        if payload.get("candidate_contract") != asdict(contract):
            raise RuntimeError("technical recovery candidate drift")
        model.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])
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
        totals: list[float] = []
        classifications: list[float] = []
        consistencies: list[float] = []
        margins: list[float] = []
        max_gradient = 0.0
        ramp = consistency_ramp(epoch, training_config.max_epochs) if contract.consistency_ramp else 1.0
        for batch in train_loader:
            original_cpu = batch["inputs"]
            labels = batch["labels"].to(device)
            inputs = original_cpu.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            classification = loss_function(
                logits, labels, epoch=epoch, subject_ids=batch["subject_ids"],
                subject_counts=subject_counts,
            )
            consistency = torch.zeros((), device=device)
            if max(contract.consistency_weights) > 0.0:
                augmented = augment_route_consistency_batch(
                    original_cpu, batch["labels"], ssl_generator,
                    dropout_probabilities=contract.route_dropout_probabilities,
                ).to(device)
                consistency = class_weighted_consistency_loss(
                    logits, model(augmented), labels, contract.consistency_weights
                )
            margin = positive_margin_loss(logits, labels) if contract.positive_margin else torch.zeros((), device=device)
            loss = classification + float(ramp) * consistency + POSITIVE_MARGIN_WEIGHT * margin
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
            totals.append(float(loss.detach().cpu()))
            classifications.append(float(classification.detach().cpu()))
            consistencies.append(float(consistency.detach().cpu()))
            margins.append(float(margin.detach().cpu()))
        scheduler.step()
        validation_metrics, _ = _evaluate(model, validation_loader, device)
        score = min(float(validation_metrics["uf1"]), float(validation_metrics["uar"]))
        improved = score > best_score + 1e-12
        if improved:
            best_score, best_epoch, epochs_without_improvement = score, epoch, 0
        else:
            epochs_without_improvement += 1
        epoch_rows.append({
            "epoch": epoch,
            "train_loss": float(np.mean(totals)),
            "classification_loss": float(np.mean(classifications)),
            "weighted_consistency_loss_before_ramp": float(np.mean(consistencies)),
            "consistency_ramp": float(ramp),
            "positive_margin_loss_before_weight": float(np.mean(margins)),
            "positive_margin_weight": POSITIVE_MARGIN_WEIGHT if contract.positive_margin else 0.0,
            "max_abs_gradient_before_clip": max_gradient,
            "validation_metrics": validation_metrics,
            "selection_score": score,
            "checkpoint_updated": improved,
            "learning_rate": float(scheduler.get_last_lr()[0]),
        })
        (output_dir / "epochs.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in epoch_rows), encoding="utf-8"
        )
        payload = _checkpoint_payload(
            model=model, optimizer=optimizer, scheduler=scheduler,
            candidate_id=candidate_id, training_config=training_config,
            evidence_hashes=evidence_hashes, epoch=epoch, best_epoch=best_epoch,
            best_score=best_score, epochs_without_improvement=epochs_without_improvement,
            loader_generator=loader_generator, ssl_generator=ssl_generator,
        )
        _save(last_path, payload)
        if improved:
            _save(best_path, payload)
        if epoch >= training_config.min_epochs and epochs_without_improvement >= training_config.patience:
            break

    loaded, metadata = load_me5_checkpoint(best_path, map_location=device)
    loaded = loaded.to(device)
    best_metrics, predictions = _evaluate(loaded, validation_loader, device)
    direct = build_me5_causalnet().to(device)
    direct.load_state_dict(metadata["evaluation_state_dict"], strict=True)
    _, direct_predictions = _evaluate(direct, validation_loader, device)
    by_id = {row["sample_id"]: np.asarray(row["probabilities"]) for row in predictions}
    direct_by_id = {row["sample_id"]: np.asarray(row["probabilities"]) for row in direct_predictions}
    reload_difference = max(float(np.max(np.abs(by_id[key] - direct_by_id[key]))) for key in by_id)
    if reload_difference > 1e-7:
        raise RuntimeError(f"checkpoint reload changed probabilities: {reload_difference}")
    prediction_path = output_dir / "validation_predictions.jsonl"
    prediction_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions), encoding="utf-8"
    )
    result = {
        "schema_version": "causalnet_opt_me_005_fold_result_v1",
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
        "ema_state_saved": False,
        "strict_reload_max_probability_difference": reload_difference,
        "technical_recovery_used": recovered,
        "outer_test_used": False,
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


__all__ = ["CHECKPOINT_SCHEMA", "ME5TrainingConfig", "load_me5_checkpoint", "set_seed", "train_candidate_fold"]
