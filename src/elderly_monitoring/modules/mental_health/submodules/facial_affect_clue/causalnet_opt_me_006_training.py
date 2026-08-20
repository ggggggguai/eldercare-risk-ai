from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .causalnet import CausalNet
from .causalnet_opt_me_004 import train_subject_counts
from .causalnet_opt_me_004_training import _cpu_state, _evaluate, _loader, set_seed
from .causalnet_opt_me_006 import (
    CANDIDATE_REGISTRY,
    GUARD_WEIGHT,
    POSITIVE_MARGIN_MAXIMUM_WEIGHT,
    REDUCED_CAUSALNET_CONFIG,
    TAIL_MAXIMUM_WEIGHT,
    build_candidate_loss,
    build_me6_causalnet,
    loss_ramp,
    non_positive_boundary_guard_loss,
    positive_margin_loss,
    subject_tail_cvar_loss,
)


CHECKPOINT_SCHEMA = "causalnet_opt_me_006_checkpoint_v1"


@dataclass(frozen=True)
class ME6TrainingConfig:
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
    training_config: ME6TrainingConfig,
    evidence_hashes: Mapping[str, str],
    epoch: int,
    best_epoch: int,
    best_score: float,
    epochs_without_improvement: int,
    loader_generator: torch.Generator,
    auxiliary_generator: torch.Generator,
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
            "auxiliary_generator": auxiliary_generator.get_state(),
        },
    }


def _save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(payload), path)


def load_me6_checkpoint(
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
        raise ValueError(f"ME6 checkpoint metadata missing: {sorted(missing)}")
    if payload["schema_version"] != CHECKPOINT_SCHEMA:
        raise ValueError("unsupported ME6 checkpoint schema")
    candidate_id = str(payload["candidate_id"])
    if candidate_id not in CANDIDATE_REGISTRY:
        raise ValueError("checkpoint candidate is outside frozen B0-P4 registry")
    if payload["candidate_contract"] != asdict(CANDIDATE_REGISTRY[candidate_id]):
        raise ValueError("checkpoint candidate contract drift")
    if payload["model_config"] != REDUCED_CAUSALNET_CONFIG.as_dict():
        raise ValueError("checkpoint model configuration drift")
    if int(payload["parameter_count"]) != 623963:
        raise ValueError("checkpoint parameter count drift")
    if payload["ema_state_dict"] is not None:
        raise ValueError("OPT-ME-006 forbids EMA")
    model = build_me6_causalnet()
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
    training_config: ME6TrainingConfig,
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
    auxiliary_generator = torch.Generator().manual_seed(seed ^ 0x5EED006)
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
    model = build_me6_causalnet().to(device)
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
        auxiliary_generator.set_state(payload["rng_state"]["auxiliary_generator"].cpu())
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
        margins: list[float] = []
        guards: list[float] = []
        tails: list[float] = []
        max_gradient = 0.0
        ramp = loss_ramp(epoch, training_config.max_epochs) if contract.margin_ramp else 1.0
        margin_weight = (
            POSITIVE_MARGIN_MAXIMUM_WEIGHT * ramp if contract.positive_margin else 0.0
        )
        guard_weight = GUARD_WEIGHT if contract.non_positive_guard else 0.0
        tail_weight = TAIL_MAXIMUM_WEIGHT * ramp if contract.subject_tail_cvar else 0.0
        for batch in train_loader:
            labels = batch["labels"].to(device)
            inputs = batch["inputs"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            classification = loss_function(
                logits, labels, epoch=epoch, subject_ids=batch["subject_ids"],
                subject_counts=subject_counts,
            )
            margin = (
                positive_margin_loss(logits, labels)
                if contract.positive_margin else torch.zeros((), device=device)
            )
            guard = (
                non_positive_boundary_guard_loss(logits, labels)
                if contract.non_positive_guard else torch.zeros((), device=device)
            )
            tail = torch.zeros((), device=device)
            if contract.subject_tail_cvar:
                per_sample = loss_function.per_sample(logits, labels, epoch=epoch)
                tail = subject_tail_cvar_loss(per_sample, batch["subject_ids"])
            loss = classification + margin_weight * margin + guard_weight * guard + tail_weight * tail
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
            margins.append(float(margin.detach().cpu()))
            guards.append(float(guard.detach().cpu()))
            tails.append(float(tail.detach().cpu()))
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
            "positive_margin_loss_before_weight": float(np.mean(margins)),
            "positive_margin_weight": float(margin_weight),
            "non_positive_guard_loss_before_weight": float(np.mean(guards)),
            "non_positive_guard_weight": float(guard_weight),
            "subject_tail_cvar_loss_before_weight": float(np.mean(tails)),
            "subject_tail_cvar_weight": float(tail_weight),
            "loss_ramp": float(ramp),
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
            loader_generator=loader_generator, auxiliary_generator=auxiliary_generator,
        )
        _save(last_path, payload)
        if improved:
            _save(best_path, payload)
        if epoch >= training_config.min_epochs and epochs_without_improvement >= training_config.patience:
            break

    loaded, metadata = load_me6_checkpoint(best_path, map_location=device)
    loaded = loaded.to(device)
    best_metrics, predictions = _evaluate(loaded, validation_loader, device)
    direct = build_me6_causalnet().to(device)
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
        "schema_version": "causalnet_opt_me_006_fold_result_v1",
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


__all__ = [
    "CHECKPOINT_SCHEMA", "ME6TrainingConfig", "load_me6_checkpoint", "set_seed",
    "train_candidate_fold",
]
