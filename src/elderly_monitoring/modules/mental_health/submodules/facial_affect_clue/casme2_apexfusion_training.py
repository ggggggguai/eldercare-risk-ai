from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, replace
import math
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .casme2_apexfusion import ApexFusionConfig, ApexFusionNet, multitask_loss
from .casme2_apexfusion_dataset import ROUTE_ORDER, load_artifact
from .metrics import classification_metrics


@dataclass(frozen=True)
class TrainingCandidate:
    candidate_id: str
    temporal: str = "tcn"
    sampler: str = "random"
    loss_kind: str = "ce"
    enabled_routes: tuple[str, ...] = ROUTE_ORDER
    aux_weight: float = 0.0
    supcon_weight: float = 0.0
    augmentation: str = "none"
    ssl: str = "none"


CANDIDATES = (
    TrainingCandidate("C0_random_ce_tcn"),
    TrainingCandidate("C1_class_weighted_tcn", sampler="class", loss_kind="weighted_ce", aux_weight=0.2),
    TrainingCandidate("C2_subject_balanced_tcn", sampler="subject", loss_kind="balanced_softmax", aux_weight=0.2),
    TrainingCandidate("C3_twolevel_focal_tcn", sampler="two_level", loss_kind="cb_focal", aux_weight=0.2, supcon_weight=0.05),
    TrainingCandidate("C4_twolevel_ldam_tcn", sampler="two_level", loss_kind="ldam", aux_weight=0.2, supcon_weight=0.05),
    TrainingCandidate("C5_twolevel_gru", temporal="gru", sampler="two_level", loss_kind="balanced_softmax", aux_weight=0.2),
    TrainingCandidate("C6_twolevel_transformer", temporal="transformer", sampler="two_level", loss_kind="balanced_softmax", aux_weight=0.2),
    TrainingCandidate("C7_motion_landmark", sampler="two_level", loss_kind="balanced_softmax", enabled_routes=("motion", "landmark"), aux_weight=0.2),
    TrainingCandidate("C8_noise_augmented_tcn", sampler="two_level", loss_kind="balanced_softmax", aux_weight=0.2, augmentation="mild_noise"),
    TrainingCandidate("C9_ssl_temporal_order", sampler="two_level", loss_kind="balanced_softmax", aux_weight=0.2, ssl="temporal_order"),
    TrainingCandidate("C10_ssl_masked_consistency", sampler="two_level", loss_kind="balanced_softmax", aux_weight=0.2, ssl="masked_consistency"),
)


class MemoryArtifactStore:
    def __init__(self, artifact_rows: Sequence[Mapping[str, Any]]) -> None:
        self.rows = {str(row["sample_id"]): dict(row) for row in artifact_rows}
        self.arrays = {sample_id: load_artifact(Path(row["artifact_path"]), expected_sha256=row["artifact_sha256"]) for sample_id, row in self.rows.items()}

    def dataset(self, sample_ids: Sequence[str], candidate: TrainingCandidate) -> "MemoryDataset":
        return MemoryDataset(self, sample_ids, candidate)


class MemoryDataset(Dataset[dict[str, Any]]):
    def __init__(self, store: MemoryArtifactStore, sample_ids: Sequence[str], candidate: TrainingCandidate) -> None:
        self.store = store
        self.sample_ids = list(sample_ids)
        self.candidate = candidate
        self.route_mask = torch.tensor([route in candidate.enabled_routes for route in ROUTE_ORDER], dtype=torch.bool)

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_id = self.sample_ids[index]
        row, arrays = self.store.rows[sample_id], self.store.arrays[sample_id]
        return {
            "appearance": torch.from_numpy(arrays["appearance"]),
            "motion": torch.from_numpy(np.concatenate((arrays["motion_short"], arrays["motion_long"]), axis=0)),
            "local_roi": torch.from_numpy(arrays["local_roi"]),
            "landmark": torch.from_numpy(arrays["landmark_dynamics"]),
            "route_mask": self.route_mask.clone(),
            "label": torch.tensor(int(row["three_class_id"]), dtype=torch.long),
            "aux_label": torch.tensor(int(row["aux_emotion_id"]), dtype=torch.long),
            "sample_id": sample_id,
            "subject_id": row["subject_id"],
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sample_weights(dataset: MemoryDataset, sampler: str) -> torch.Tensor | None:
    if sampler == "random":
        return None
    rows = [dataset.store.rows[sample_id] for sample_id in dataset.sample_ids]
    class_counts = Counter(int(row["three_class_id"]) for row in rows)
    subject_counts = Counter(str(row["subject_id"]) for row in rows)
    if sampler == "class":
        values = [1.0 / class_counts[int(row["three_class_id"])] for row in rows]
    elif sampler == "subject":
        values = [1.0 / subject_counts[str(row["subject_id"])] for row in rows]
    elif sampler == "two_level":
        values = [1.0 / math.sqrt(class_counts[int(row["three_class_id"])] * subject_counts[str(row["subject_id"])]) for row in rows]
    else:
        raise ValueError(f"Unknown sampler: {sampler}")
    return torch.tensor(values, dtype=torch.double)


def _loader(dataset: MemoryDataset, *, batch_size: int, seed: int, training: bool) -> DataLoader:
    weights = _sample_weights(dataset, dataset.candidate.sampler) if training else None
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True, generator=generator) if weights is not None else None
    return DataLoader(dataset, batch_size=batch_size, shuffle=training and sampler is None, sampler=sampler, num_workers=0, generator=generator)


def _to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def _augment(batch: dict[str, Any], candidate: TrainingCandidate, generator: torch.Generator) -> None:
    if candidate.augmentation != "mild_noise":
        return
    batch["appearance"] = torch.clamp(batch["appearance"] * (0.95 + 0.10 * torch.rand((len(batch["label"]), 1, 1, 1, 1), device=batch["appearance"].device, generator=generator)), 0.0, 1.0)
    batch["local_roi"] = torch.clamp(batch["local_roi"] + 0.01 * torch.randn(batch["local_roi"].shape, device=batch["local_roi"].device, generator=generator), 0.0, 1.0)
    batch["motion"] = torch.clamp(batch["motion"] + 0.01 * torch.randn(batch["motion"].shape, device=batch["motion"].device, generator=generator), -1.0, 1.0)
    batch["landmark"] = batch["landmark"] + 0.002 * torch.randn(batch["landmark"].shape, device=batch["landmark"].device, generator=generator)


def _classification_loss(logits: torch.Tensor, labels: torch.Tensor, counts: torch.Tensor, kind: str) -> torch.Tensor:
    counts = counts.to(logits.device, logits.dtype).clamp_min(1)
    if kind == "ce":
        return nn.functional.cross_entropy(logits, labels)
    if kind == "weighted_ce":
        weights = counts.sum() / (len(counts) * counts)
        return nn.functional.cross_entropy(logits, labels, weight=weights)
    if kind == "balanced_softmax":
        return nn.functional.cross_entropy(logits + counts.log().unsqueeze(0), labels)
    if kind == "cb_focal":
        beta = 0.999
        weights = (1.0 - beta) / (1.0 - torch.pow(torch.tensor(beta, device=logits.device, dtype=logits.dtype), counts))
        weights = weights / weights.sum() * len(counts)
        ce = nn.functional.cross_entropy(logits, labels, weight=weights, reduction="none")
        pt = torch.exp(-ce)
        return (((1 - pt) ** 2) * ce).mean()
    if kind == "ldam":
        margins = 0.5 / torch.pow(counts, 0.25)
        adjusted = logits.clone()
        adjusted[torch.arange(len(labels), device=labels.device), labels] -= margins[labels]
        return nn.functional.cross_entropy(adjusted * 10.0, labels)
    raise ValueError(f"Unknown loss: {kind}")


def _ssl_pretrain(model: ApexFusionNet, loader: DataLoader, candidate: TrainingCandidate, device: torch.device, seed: int, epochs: int) -> None:
    if candidate.ssl == "none" or epochs <= 0:
        return
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    order_head = nn.Linear(model.config.hidden_dim, 2).to(device) if candidate.ssl == "temporal_order" else None
    if order_head is not None:
        optimizer = torch.optim.AdamW(list(model.parameters()) + list(order_head.parameters()), lr=3e-4, weight_decay=1e-4)
    generator = torch.Generator(device=device).manual_seed(seed + 991)
    model.train()
    for _ in range(epochs):
        for raw in loader:
            batch = _to_device(raw, device)
            optimizer.zero_grad(set_to_none=True)
            if candidate.ssl == "temporal_order":
                reverse = torch.rand(len(batch["label"]), device=device, generator=generator) < 0.5
                for key in ("appearance", "motion", "local_roi", "landmark"):
                    batch[key][reverse] = torch.flip(batch[key][reverse], dims=(1,))
                output = model(batch)
                loss = nn.functional.cross_entropy(order_head(output["fused"]), reverse.long())
            else:
                with torch.no_grad():
                    target = model(batch)["fused"].detach()
                masked = {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
                route_index = int(torch.randint(0, 4, (1,), generator=generator, device=device))
                masked["route_mask"][:, route_index] = False
                loss = nn.functional.mse_loss(model(masked)["fused"], target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()


def train_inner_fold(
    store: MemoryArtifactStore,
    *,
    train_ids: Sequence[str],
    validation_ids: Sequence[str],
    candidate: TrainingCandidate,
    seed: int,
    epochs: int,
    ssl_epochs: int,
    batch_size: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    set_seed(seed)
    train_dataset = store.dataset(train_ids, candidate)
    validation_dataset = store.dataset(validation_ids, candidate)
    train_loader = _loader(train_dataset, batch_size=batch_size, seed=seed, training=True)
    validation_loader = _loader(validation_dataset, batch_size=batch_size, seed=seed, training=False)
    model = ApexFusionNet(ApexFusionConfig(temporal=candidate.temporal, temporal_layers=1)).to(device)
    _ssl_pretrain(model, train_loader, candidate, device, seed, ssl_epochs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    train_labels = [int(store.rows[sample_id]["three_class_id"]) for sample_id in train_ids]
    counts = torch.tensor([train_labels.count(label) for label in range(3)], dtype=torch.float32)
    best_score = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    generator = torch.Generator(device=device).manual_seed(seed + 17)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for raw in train_loader:
            batch = _to_device(raw, device)
            _augment(batch, candidate, generator)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            main = _classification_loss(output["main_logits"], batch["label"], counts, candidate.loss_kind)
            auxiliary = nn.functional.cross_entropy(output["aux_logits"], batch["aux_label"])
            supcon = multitask_loss(output, batch["label"], batch["aux_label"], aux_weight=0.0, supcon_weight=1.0)["supcon"]
            loss = main + candidate.aux_weight * auxiliary + candidate.supcon_weight * supcon
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += float(loss.detach().cpu()) * len(batch["label"])
        predictions = predict(model, validation_loader, device)
        metrics = classification_metrics([row["true_label"] for row in predictions], [row["predicted_label"] for row in predictions])
        score = 2 * metrics["uf1"] * metrics["uar"] / max(metrics["uf1"] + metrics["uar"], 1e-12)
        history.append({"epoch": epoch, "train_loss": epoch_loss / len(train_dataset), "validation_uf1": metrics["uf1"], "validation_uar": metrics["uar"], "selection_score": score})
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state, strict=True)
    predictions = predict(model, validation_loader, device)
    return predictions, {"best_epoch": best_epoch, "best_score": best_score, "history": history}


def predict(model: ApexFusionNet, loader: DataLoader, device: torch.device) -> list[dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for raw in loader:
            batch = _to_device(raw, device)
            probabilities = torch.softmax(model(batch)["main_logits"], dim=1).cpu().numpy()
            labels = raw["label"].numpy()
            for index, probability in enumerate(probabilities):
                rows.append({"sample_id": raw["sample_id"][index], "subject_id": raw["subject_id"][index], "true_label": int(labels[index]), "predicted_label": int(np.argmax(probability)), "probabilities": probability.tolist()})
    return rows


def pooled_metrics(predictions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return classification_metrics([int(row["true_label"]) for row in predictions], [int(np.argmax(row["probabilities"])) for row in predictions])


def candidate_rank_key(result: Mapping[str, Any]) -> tuple[float, float, float, float]:
    metrics = result["metrics"]
    score = 2 * metrics["uf1"] * metrics["uar"] / max(metrics["uf1"] + metrics["uar"], 1e-12)
    positive = metrics["per_class"]["positive"]
    zero_subjects = sum(1 for value in result.get("per_subject_scores", {}).values() if value == 0.0)
    return (score, positive["f1"], positive["recall"], -float(zero_subjects))
