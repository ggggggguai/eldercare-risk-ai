from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from .paper_mhssa_tgcn import (
    PAPER_MODEL_SCHEMA_VERSION,
    PaperMHSSATGCN,
)


PAPER_TRAINING_SCHEMA_VERSION = "microexpression_paper_training_v2"


@dataclass(frozen=True)
class PaperTrainingConfig:
    optimizer: str = "adam"
    learning_rate: float = 5e-5
    lr_step_epochs: int = 10
    lr_gamma: float = 0.95
    max_epochs: int = 200
    min_epochs: int = 1
    patience: int = 15
    batch_size: int = 32
    weight_decay: float = 0.0
    gradient_clip_norm: float = 1.0
    num_workers: int = 0
    loss: str = "cross_entropy"

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "PaperTrainingConfig":
        known = {item.name for item in cls.__dataclass_fields__.values()}
        return cls(**{key: value for key, value in values.items() if key in known})

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_paper_training_components(
    model: nn.Module,
    config: PaperTrainingConfig,
) -> tuple[
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.StepLR,
    nn.Module,
]:
    if config.optimizer != "adam":
        raise ValueError("The paper-exact training configuration requires Adam")
    if config.loss != "cross_entropy":
        raise ValueError("The paper-exact training configuration requires CE")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.lr_step_epochs,
        gamma=config.lr_gamma,
    )
    return optimizer, scheduler, nn.CrossEntropyLoss()


def state_dict_on_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_paper_model_checkpoint(
    path: Path,
    *,
    model: PaperMHSSATGCN,
    training_config: PaperTrainingConfig,
    metadata: Mapping[str, Any],
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_schema_version": PAPER_MODEL_SCHEMA_VERSION,
        "training_schema_version": PAPER_TRAINING_SCHEMA_VERSION,
        "model_config": model.config.as_dict(),
        "training_config": training_config.as_dict(),
        "au_adjacency": model.graph_builder.au_adjacency.detach().cpu(),
        "model_state_dict": state_dict_on_cpu(model),
        **dict(metadata),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)
    return sha256_file(path)
