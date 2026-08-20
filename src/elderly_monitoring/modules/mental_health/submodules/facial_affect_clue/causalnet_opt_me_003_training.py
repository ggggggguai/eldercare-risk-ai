from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .causalnet import CausalNet
from .causalnet_dataset import CausalNetArtifactDataset, collate_causalnet
from .causalnet_opt_me_003 import (
    CANDIDATE_REGISTRY,
    REDUCED_CAUSALNET_CONFIG,
    build_candidate_loss,
    build_me3_causalnet,
    two_level_sample_weights,
)
from .metrics import classification_metrics


CHECKPOINT_SCHEMA = "causalnet_opt_me_003_checkpoint_v1"


@dataclass(frozen=True)
class ME3TrainingConfig:
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


def save_me3_checkpoint(
    path: Path,
    model: CausalNet,
    *,
    candidate_id: str,
    training_config: Mapping[str, Any],
    evidence_hashes: Mapping[str, str],
) -> None:
    if candidate_id not in CANDIDATE_REGISTRY:
        raise ValueError(f"unknown candidate_id: {candidate_id}")
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save(
        {
            "schema_version": CHECKPOINT_SCHEMA,
            "candidate_id": candidate_id,
            "candidate_contract": asdict(CANDIDATE_REGISTRY[candidate_id]),
            "model_config": REDUCED_CAUSALNET_CONFIG.as_dict(),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "training_config": dict(training_config),
            "evidence_hashes": dict(evidence_hashes),
            "model_state_dict": state,
        },
        path,
    )


def load_me3_checkpoint(
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
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"ME3 checkpoint metadata missing: {sorted(missing)}")
    if payload["schema_version"] != CHECKPOINT_SCHEMA:
        raise ValueError("unsupported ME3 checkpoint schema")
    candidate_id = str(payload["candidate_id"])
    if candidate_id not in CANDIDATE_REGISTRY:
        raise ValueError("checkpoint candidate is outside frozen registry")
    if payload["candidate_contract"] != asdict(CANDIDATE_REGISTRY[candidate_id]):
        raise ValueError("checkpoint candidate contract drift")
    if payload["model_config"] != REDUCED_CAUSALNET_CONFIG.as_dict():
        raise ValueError("checkpoint model configuration drift")
    if int(payload["parameter_count"]) != 623963:
        raise ValueError("checkpoint parameter count drift")
    model = build_me3_causalnet()
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model, payload


def _model_records(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
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
    candidate_id: str,
    batch_size: int,
    seed: int,
    training: bool,
) -> DataLoader[dict[str, Any]]:
    dataset = CausalNetArtifactDataset(_model_records(rows), preload=True)
    generator = torch.Generator().manual_seed(seed)
    sampler = None
    if training and CANDIDATE_REGISTRY[candidate_id].sampler_kind == "subject_class_two_level":
        sampler = WeightedRandomSampler(
            two_level_sample_weights(rows),
            num_samples=len(rows),
            replacement=True,
            generator=generator,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training and sampler is None,
        sampler=sampler,
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
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            predictions = probabilities.argmax(axis=1)
            labels = batch["labels"].numpy()
            for index, sample_id in enumerate(batch["sample_ids"]):
                rows.append(
                    {
                        "sample_id": sample_id,
                        "subject_id": batch["subject_ids"][index],
                        "label": int(labels[index]),
                        "prediction": int(predictions[index]),
                        "probabilities": probabilities[index].tolist(),
                    }
                )
    metrics = classification_metrics(
        [int(row["label"]) for row in rows],
        [int(row["prediction"]) for row in rows],
    )
    return metrics, rows


def train_candidate_fold(
    *,
    train_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    candidate_id: str,
    seed: int,
    device: torch.device,
    output_dir: Path,
    training_config: ME3TrainingConfig,
    evidence_hashes: Mapping[str, str],
) -> dict[str, Any]:
    if candidate_id not in CANDIDATE_REGISTRY:
        raise ValueError(f"unknown candidate: {candidate_id}")
    if {str(row["subject_id"]) for row in train_rows} & {
        str(row["subject_id"]) for row in validation_rows
    }:
        raise ValueError("subject leakage between train and validation")
    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_loader = _loader(
        train_rows,
        candidate_id=candidate_id,
        batch_size=training_config.batch_size,
        seed=seed,
        training=True,
    )
    validation_loader = _loader(
        validation_rows,
        candidate_id=candidate_id,
        batch_size=training_config.batch_size,
        seed=seed,
        training=False,
    )
    counts = tuple(
        sum(int(row["label"]) == label for row in train_rows) for label in (0, 1, 2)
    )
    loss_function = build_candidate_loss(candidate_id, class_counts=counts).to(device)
    model = build_me3_causalnet().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    best_score = -float("inf")
    best_epoch = 0
    best_prediction_rows: list[dict[str, Any]] | None = None
    epochs_without_improvement = 0
    checkpoint_path = output_dir / "best.pt"
    epoch_rows: list[dict[str, Any]] = []
    for epoch in range(1, training_config.max_epochs + 1):
        model.train()
        losses: list[float] = []
        max_gradient = 0.0
        for batch in train_loader:
            inputs = batch["inputs"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = loss_function(logits, labels, epoch=epoch)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training loss")
            loss.backward()
            for parameter in model.parameters():
                if parameter.grad is not None:
                    if not torch.isfinite(parameter.grad).all():
                        raise RuntimeError("non-finite gradient")
                    max_gradient = max(max_gradient, float(parameter.grad.detach().abs().max()))
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), training_config.gradient_clip_norm
            )
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation_metrics, validation_prediction_rows = _evaluate(
            model, validation_loader, device
        )
        score = min(float(validation_metrics["uf1"]), float(validation_metrics["uar"]))
        improved = score > best_score + 1e-12
        if improved:
            best_score = score
            best_epoch = epoch
            best_prediction_rows = validation_prediction_rows
            epochs_without_improvement = 0
            save_me3_checkpoint(
                checkpoint_path,
                model,
                candidate_id=candidate_id,
                training_config=asdict(training_config),
                evidence_hashes=evidence_hashes,
            )
        else:
            epochs_without_improvement += 1
        epoch_rows.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "max_abs_gradient_before_clip": max_gradient,
                "validation_metrics": validation_metrics,
                "selection_score": score,
                "checkpoint_updated": improved,
            }
        )
        if (
            epoch >= training_config.min_epochs
            and epochs_without_improvement >= training_config.patience
        ):
            break
    epoch_log_path = output_dir / "epochs.jsonl"
    epoch_log_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in epoch_rows),
        encoding="utf-8",
    )
    loaded, metadata = load_me3_checkpoint(checkpoint_path, map_location=device)
    loaded = loaded.to(device)
    best_metrics, predictions = _evaluate(loaded, validation_loader, device)
    if best_prediction_rows is None:
        raise RuntimeError("best checkpoint was not recorded")
    saved_probabilities = {
        str(row["sample_id"]): np.asarray(row["probabilities"], dtype=np.float64)
        for row in best_prediction_rows
    }
    reloaded_probabilities = {
        str(row["sample_id"]): np.asarray(row["probabilities"], dtype=np.float64)
        for row in predictions
    }
    if saved_probabilities.keys() != reloaded_probabilities.keys():
        raise RuntimeError("checkpoint reload changed validation sample coverage")
    reload_max_probability_difference = max(
        float(np.max(np.abs(saved_probabilities[sample_id] - reloaded_probabilities[sample_id])))
        for sample_id in saved_probabilities
    )
    if reload_max_probability_difference > 1e-7:
        raise RuntimeError(
            "checkpoint reload changed validation probabilities: "
            f"{reload_max_probability_difference}"
        )
    prediction_path = output_dir / "validation_predictions.jsonl"
    prediction_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions),
        encoding="utf-8",
    )
    result = {
        "schema_version": "causalnet_opt_me_003_fold_result_v1",
        "candidate_id": candidate_id,
        "seed": seed,
        "status": "completed",
        "best_epoch": best_epoch,
        "best_selection_score": best_score,
        "validation_metrics": best_metrics,
        "train_sample_count": len(train_rows),
        "validation_sample_count": len(validation_rows),
        "train_subject_ids": sorted({str(row["subject_id"]) for row in train_rows}),
        "validation_subject_ids": sorted(
            {str(row["subject_id"]) for row in validation_rows}
        ),
        "checkpoint_path": checkpoint_path.as_posix(),
        "epoch_log_path": epoch_log_path.as_posix(),
        "prediction_path": prediction_path.as_posix(),
        "checkpoint_candidate_contract": metadata["candidate_contract"],
        "strict_reload_max_probability_difference": reload_max_probability_difference,
        "outer_test_used": False,
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result
