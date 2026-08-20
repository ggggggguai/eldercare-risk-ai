"""Leakage-safe compact sequence models for FORECAST-OPT-004G.

The module is intentionally offline-only.  It builds fixed-length histories
ending at the feature month, fits normalization on the current training fold,
and exposes small PyTorch candidates with explicit value masks and month gaps.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


MODEL_FAMILIES = ("masked_gru", "small_tcn", "controlled_transformer")


def prepare_source(source: pd.DataFrame) -> pd.DataFrame:
    """Derive the frozen participant/month key from the PSYCHE-D index."""

    participants: list[str] = []
    months: list[int] = []
    for value in source.index.tolist():
        if not isinstance(value, str):
            raise ValueError("PSYCHE-D index must contain string keys")
        participant, separator, month = value.rpartition("_")
        if not separator or not participant or not month.isdigit():
            raise ValueError(f"malformed PSYCHE-D participant-month key: {value!r}")
        participants.append(participant)
        months.append(int(month))
    result = source.copy()
    result["__participant_id"] = participants
    result["__nominal_month"] = months
    if result[["__participant_id", "__nominal_month"]].duplicated().any():
        raise ValueError("duplicate PSYCHE-D participant-month key")
    return result


def build_sequence_arrays(
    samples: pd.DataFrame,
    source: pd.DataFrame,
    feature_names: Sequence[str],
    sequence_length: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build histories that end at (and never exceed) feature_month_slot.

    Returns values, per-value masks, relative month gaps, and the number of
    contiguous source months ending at the feature month.
    """

    if sequence_length < 3:
        raise ValueError("sequence_length must be at least 3")
    missing = [name for name in feature_names if name not in source.columns]
    if missing:
        raise ValueError(f"source is missing sequence features: {missing}")
    lookup = source.set_index(["__participant_id", "__nominal_month"])
    n_rows = len(samples)
    n_features = len(feature_names)
    values = np.full((n_rows, sequence_length, n_features), np.nan, dtype="float32")
    masks = np.zeros_like(values, dtype="float32")
    gaps = np.zeros((n_rows, sequence_length, 1), dtype="float32")
    contiguous = np.zeros(n_rows, dtype="int16")
    for row_index, sample in enumerate(samples.itertuples(index=False)):
        participant = str(sample.global_participant_id).split("::", 1)[-1]
        feature_month = int(sample.feature_month_slot)
        months = list(range(feature_month - sequence_length + 1, feature_month + 1))
        gaps[row_index, :, 0] = np.asarray(months, dtype="float32") - float(
            feature_month
        )
        observed_months: set[int] = set()
        for time_index, month in enumerate(months):
            key = (participant, month)
            if key not in lookup.index:
                continue
            raw = lookup.loc[key, list(feature_names)]
            if isinstance(raw, pd.DataFrame):
                raise ValueError("duplicate source row reached sequence builder")
            vector = pd.to_numeric(raw, errors="coerce").to_numpy(dtype="float32")
            finite = np.isfinite(vector)
            values[row_index, time_index, finite] = vector[finite]
            masks[row_index, time_index, finite] = 1.0
            observed_months.add(month)
        month = feature_month
        while month in observed_months:
            contiguous[row_index] += 1
            month -= 1
    return values, masks, gaps, contiguous


@dataclass
class SequencePreprocessor:
    """Fold-local robust filling and standardization for sequence values."""

    medians: np.ndarray | None = None
    means: np.ndarray | None = None
    scales: np.ndarray | None = None

    def fit(self, values: np.ndarray, masks: np.ndarray) -> "SequencePreprocessor":
        if values.shape != masks.shape or values.ndim != 3:
            raise ValueError("values and masks must be same-shape 3D tensors")
        n_features = values.shape[2]
        medians = np.zeros(n_features, dtype="float32")
        means = np.zeros(n_features, dtype="float32")
        scales = np.ones(n_features, dtype="float32")
        for column in range(n_features):
            observed = values[:, :, column][masks[:, :, column] > 0.5]
            observed = observed[np.isfinite(observed)]
            if observed.size:
                medians[column] = float(np.median(observed))
                means[column] = float(np.mean(observed))
                scale = float(np.std(observed))
                scales[column] = scale if np.isfinite(scale) and scale > 1e-6 else 1.0
        self.medians = medians
        self.means = means
        self.scales = scales
        return self

    def transform(
        self, values: np.ndarray, masks: np.ndarray, gaps: np.ndarray
    ) -> np.ndarray:
        if self.medians is None or self.means is None or self.scales is None:
            raise RuntimeError("sequence preprocessor is not fitted")
        if values.shape != masks.shape or gaps.shape[:2] != values.shape[:2]:
            raise ValueError("sequence tensor shapes are inconsistent")
        filled = np.where(np.isfinite(values), values, self.medians[None, None, :])
        standardized = (filled - self.means[None, None, :]) / self.scales[None, None, :]
        standardized = np.clip(standardized, -8.0, 8.0).astype("float32")
        normalized_gap = gaps.astype("float32") / max(1.0, float(values.shape[1] - 1))
        return np.concatenate(
            [standardized, masks.astype("float32"), normalized_gap], axis=2
        )


class MaskedGRU(nn.Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=24, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(24), nn.Linear(24, 1))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(values)
        return self.head(output[:, -1]).squeeze(1)


class SmallTCN(nn.Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(input_size, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Linear(16, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        encoded = self.network(values.transpose(1, 2)).mean(dim=2)
        return self.head(encoded).squeeze(1)


class ControlledTransformer(nn.Module):
    def __init__(self, input_size: int, sequence_length: int) -> None:
        super().__init__()
        self.projection = nn.Linear(input_size, 32)
        self.position = nn.Parameter(torch.zeros(1, sequence_length, 32))
        layer = nn.TransformerEncoderLayer(
            d_model=32,
            nhead=4,
            dim_feedforward=64,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.head = nn.Sequential(nn.LayerNorm(32), nn.Linear(32, 1))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(self.projection(values) + self.position)
        return self.head(encoded.mean(dim=1)).squeeze(1)


def build_model(family: str, input_size: int, sequence_length: int) -> nn.Module:
    if family == "masked_gru":
        return MaskedGRU(input_size)
    if family == "small_tcn":
        return SmallTCN(input_size)
    if family == "controlled_transformer":
        return ControlledTransformer(input_size, sequence_length)
    raise ValueError(f"unsupported sequence family: {family}")


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def participant_equal_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = (
        frame.groupby("global_participant_id", sort=False)["global_participant_id"]
        .transform("size")
        .to_numpy(dtype="float64")
    )
    return 1.0 / (float(frame["global_participant_id"].nunique()) * counts)


def training_weights(frame: pd.DataFrame) -> np.ndarray:
    """Participant-equal weights with equal positive/negative total mass."""

    base = participant_equal_weights(frame)
    labels = frame["future_binary_target"].to_numpy(dtype="int8")
    result = base.copy()
    for label in (0, 1):
        total = float(base[labels == label].sum())
        if total <= 0:
            raise ValueError("training fold lacks a binary target class")
        result[labels == label] *= 0.5 / total
    return result / float(result.mean())


def _stable_validation_mask(participants: Sequence[str], seed: int) -> np.ndarray:
    result = []
    for participant in participants:
        digest = hashlib.sha256(f"{seed}|{participant}".encode()).digest()
        result.append(int.from_bytes(digest[:4], "big") % 10 == 0)
    mask = np.asarray(result, dtype=bool)
    if mask.sum() == 0 or (~mask).sum() == 0:
        mask[: max(1, len(mask) // 10)] = True
    return mask


def fit_predict(
    family: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_weight: np.ndarray,
    train_participants: Sequence[str],
    validation_x: np.ndarray,
    seed: int,
    maximum_epochs: int = 24,
    patience: int = 4,
    batch_size: int = 2048,
    learning_rate: float = 0.003,
    maximum_parameters: int = 250_000,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Train one small candidate using an internal participant holdout."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    model = build_model(family, train_x.shape[2], train_x.shape[1])
    n_parameters = parameter_count(model)
    if n_parameters > maximum_parameters:
        raise ValueError(
            f"{family} has {n_parameters} parameters, above {maximum_parameters}"
        )
    internal_validation = _stable_validation_mask(train_participants, seed)
    internal_train = ~internal_validation
    tensors = TensorDataset(
        torch.from_numpy(train_x[internal_train]),
        torch.from_numpy(train_y[internal_train].astype("float32")),
        torch.from_numpy(train_weight[internal_train].astype("float32")),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        tensors,
        batch_size=min(batch_size, len(tensors)),
        shuffle=True,
        generator=generator,
    )
    x_stop = torch.from_numpy(train_x[internal_validation])
    y_stop = torch.from_numpy(train_y[internal_validation].astype("float32"))
    w_stop = torch.from_numpy(train_weight[internal_validation].astype("float32"))
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    epochs_run = 0
    for epoch in range(maximum_epochs):
        model.train()
        for batch_x, batch_y, batch_weight in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = (loss_fn(model(batch_x), batch_y) * batch_weight).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            stop_loss = float(
                (loss_fn(model(x_stop), y_stop) * w_stop).sum()
                / torch.clamp(w_stop.sum(), min=1e-8)
            )
        epochs_run = epoch + 1
        if stop_loss < best_loss - 1e-5:
            best_loss = stop_loss
            best_state = {
                name: value.detach().clone()
                for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("sequence training failed to create a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(validation_x), batch_size):
            logits = model(torch.from_numpy(validation_x[start : start + batch_size]))
            predictions.append(torch.sigmoid(logits).numpy())
    return np.concatenate(predictions).astype("float64"), {
        "family": family,
        "seed": int(seed),
        "parameter_count": n_parameters,
        "epochs_run": epochs_run,
        "best_internal_validation_loss": best_loss,
        "internal_validation_rows": int(internal_validation.sum()),
    }


def binary_metrics(frame: pd.DataFrame, score_column: str) -> dict[str, float]:
    labels = frame["future_binary_target"].to_numpy(dtype="int8")
    scores = frame[score_column].to_numpy(dtype="float64")
    weights = participant_equal_weights(frame)
    return {
        "auprc": float(average_precision_score(labels, scores, sample_weight=weights)),
        "auroc": float(roc_auc_score(labels, scores, sample_weight=weights)),
    }
