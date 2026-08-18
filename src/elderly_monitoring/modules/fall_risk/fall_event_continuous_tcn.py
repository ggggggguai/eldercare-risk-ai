from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from elderly_monitoring.modules.fall_risk.fall_event_continuous import (
    build_fall_event_causal_window,
)


@dataclass(frozen=True)
class ContinuousFallTCNConfig:
    hidden_channels: int = 48
    kernel_size: int = 3
    dilations: tuple[int, ...] = (1, 2, 4, 8)
    dropout: float = 0.10
    batch_size: int = 64
    epochs: int = 25
    patience: int = 6
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.hidden_channels <= 0 or self.kernel_size < 2:
            raise ValueError("hidden_channels and kernel_size must be valid")
        if not self.dilations or any(value <= 0 for value in self.dilations):
            raise ValueError("dilations must be positive and non-empty")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be within [0, 1)")
        if self.batch_size <= 0 or self.epochs <= 0 or self.patience <= 0:
            raise ValueError("batch_size, epochs and patience must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay nonnegative")
        if self.device not in {"cpu", "cuda", "mps"}:
            raise ValueError("device must be cpu, cuda or mps")


class ContinuousFallTCN(nn.Module):
    def __init__(
        self,
        *,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilations: Sequence[int],
        dropout: float,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        channels = input_channels
        for dilation in dilations:
            blocks.append(
                _CausalResidualBlock(
                    channels,
                    hidden_channels,
                    kernel_size=kernel_size,
                    dilation=int(dilation),
                    dropout=dropout,
                )
            )
            channels = hidden_channels
        self.encoder = nn.Sequential(*blocks)
        self.presence_head = nn.Linear(hidden_channels, 1)
        self.onset_head = nn.Conv1d(hidden_channels, 1, kernel_size=1)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        encoded = self.encoder(features)
        frame_mask = features[:, _frame_mask_channel(features.shape[1]), :]
        pooled = (encoded * frame_mask.unsqueeze(1)).sum(dim=-1)
        denominator = frame_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        pooled = pooled / denominator
        presence_logits = self.presence_head(pooled).squeeze(-1)
        onset_logits = self.onset_head(encoded).squeeze(1)
        return presence_logits, onset_logits


class ContinuousFallTCNShadowPredictor:
    """Runtime adapter for development shadow scoring; it never emits an event."""

    def __init__(
        self,
        checkpoint_paths: Sequence[str | Path],
        *,
        device: str = "cpu",
        threshold: float = 0.5,
        window_sec: float = 4.0,
        target_fps: float = 8.0,
        max_gap_sec: float = 0.25,
        min_observed_frames: int = 8,
        min_valid_joint_ratio: float = 0.5,
    ) -> None:
        paths = tuple(Path(path) for path in checkpoint_paths)
        if not paths:
            raise ValueError("continuous fall shadow predictor requires checkpoints")
        if not 0.0 < threshold < 1.0:
            raise ValueError("continuous fall shadow threshold must be within (0, 1)")
        if window_sec <= 0 or target_fps <= 0 or max_gap_sec <= 0:
            raise ValueError("continuous fall shadow window parameters must be positive")
        if min_observed_frames < 1 or not 0.0 <= min_valid_joint_ratio <= 1.0:
            raise ValueError("continuous fall shadow quality thresholds are invalid")
        self.device = _resolve_device(device)
        self.threshold = float(threshold)
        self.window_sec = float(window_sec)
        self.target_fps = float(target_fps)
        self.max_gap_sec = float(max_gap_sec)
        self.min_observed_frames = int(min_observed_frames)
        self.min_valid_joint_ratio = float(min_valid_joint_ratio)
        self._members: list[tuple[ContinuousFallTCN, np.ndarray, np.ndarray, str]] = []
        for path in paths:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            _validate_shadow_checkpoint(checkpoint)
            model_config = checkpoint["model_config"]
            model = ContinuousFallTCN(
                input_channels=int(model_config["input_channels"]),
                hidden_channels=int(model_config["hidden_channels"]),
                kernel_size=int(model_config["kernel_size"]),
                dilations=tuple(int(value) for value in model_config["dilations"]),
                dropout=float(model_config["dropout"]),
            ).to(self.device)
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            mean = np.asarray(checkpoint["normalization"]["mean"], dtype=np.float32)
            std = np.asarray(checkpoint["normalization"]["std"], dtype=np.float32)
            self._members.append((model, mean, std, path.as_posix()))
        self.model_version = "fall-event-continuous-tcn-v2-shadow"
        self.checkpoint_paths = tuple(path for _, _, _, path in self._members)

    def predict_records(self, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not records:
            return self._unavailable("insufficient_pose_records", observed_frame_count=0)
        cutoff = max(float(record.get("timestamp_sec", 0.0)) for record in records)
        started = time.perf_counter()
        try:
            window = build_fall_event_causal_window(
                records,
                cutoff_time_sec=cutoff,
                window_sec=self.window_sec,
                target_fps=self.target_fps,
                max_gap_sec=self.max_gap_sec,
            )
        except ValueError as exc:
            return self._unavailable(
                "causal_window_unavailable",
                error_type=type(exc).__name__,
                duration_ms=_elapsed_ms(started),
            )
        observed = int(window.metadata["observed_frame_count"])
        valid_ratio = float(window.metadata["valid_joint_ratio"])
        if observed < self.min_observed_frames:
            return self._unavailable(
                "insufficient_observed_frames",
                observed_frame_count=observed,
                valid_joint_ratio=valid_ratio,
                duration_ms=_elapsed_ms(started),
            )
        if valid_ratio < self.min_valid_joint_ratio:
            return self._unavailable(
                "insufficient_valid_joint_ratio",
                observed_frame_count=observed,
                valid_joint_ratio=valid_ratio,
                duration_ms=_elapsed_ms(started),
            )
        scores: list[float] = []
        onset_predictions: list[int] = []
        with torch.no_grad():
            for model, mean, std, _ in self._members:
                normalized = (window.tensor - mean[None, :, :]) / std[None, :, :]
                model_input = (
                    torch.from_numpy(normalized)
                    .float()
                    .permute(1, 2, 0)
                    .reshape(1, -1, normalized.shape[0])
                )
                presence_logits, onset_logits = model(model_input.to(self.device))
                scores.append(float(torch.sigmoid(presence_logits)[0].cpu()))
                onset_predictions.append(int(onset_logits[0].argmax().cpu()))
        score = float(np.mean(scores))
        return {
            "fall_event_tcn_shadow_score": round(score, 6),
            "fall_event_tcn_shadow_detected": score >= self.threshold,
            "fall_event_tcn_shadow_threshold": self.threshold,
            "fall_event_tcn_shadow_onset_frame": int(round(float(np.mean(onset_predictions)))),
            "fall_event_tcn_shadow_model_version": self.model_version,
            "fall_event_tcn_shadow_status": "provisional_shadow",
            "fall_event_tcn_shadow_checkpoint_count": len(self._members),
            "fall_event_tcn_shadow_observed_frame_count": observed,
            "fall_event_tcn_shadow_valid_joint_ratio": round(valid_ratio, 6),
            "fall_event_tcn_shadow_duration_ms": _elapsed_ms(started),
        }

    def _unavailable(self, reason: str, **details: Any) -> dict[str, Any]:
        return {
            "fall_event_tcn_shadow_score": None,
            "fall_event_tcn_shadow_detected": False,
            "fall_event_tcn_shadow_model_version": self.model_version,
            "fall_event_tcn_shadow_status": "unavailable",
            "fall_event_tcn_shadow_reason": reason,
            **details,
        }


class ContinuousFallTCNRuntimePredictor:
    """Runtime adapter for the opt-in provisional primary fall-event branch.

    The checkpoint and causal preprocessing are shared with the shadow adapter.
    This wrapper only translates the diagnostic output into the active branch
    contract; callers still retain rule fallback when the window is unavailable.
    """

    def __init__(
        self,
        checkpoint_paths: Sequence[str | Path],
        *,
        device: str = "cpu",
        threshold: float = 0.5,
        window_sec: float = 4.0,
        target_fps: float = 8.0,
        max_gap_sec: float = 0.25,
        min_observed_frames: int = 8,
        min_valid_joint_ratio: float = 0.5,
    ) -> None:
        self._shadow = ContinuousFallTCNShadowPredictor(
            checkpoint_paths,
            device=device,
            threshold=threshold,
            window_sec=window_sec,
            target_fps=target_fps,
            max_gap_sec=max_gap_sec,
            min_observed_frames=min_observed_frames,
            min_valid_joint_ratio=min_valid_joint_ratio,
        )
        self.threshold = self._shadow.threshold
        self.checkpoint_paths = self._shadow.checkpoint_paths
        self.model_version = "fall-event-continuous-tcn-v2-runtime-provisional"

    def predict_records(self, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        result = self._shadow.predict_records(records)
        score = result.get("fall_event_tcn_shadow_score")
        if score is None:
            return {
                "fall_event_tcn_score": None,
                "fall_event_tcn_detected": False,
                "fall_event_tcn_threshold": self.threshold,
                "fall_event_tcn_status": "unavailable",
                "fall_event_tcn_reason": result.get(
                    "fall_event_tcn_shadow_reason", "runtime_window_unavailable"
                ),
                "fall_event_tcn_model_version": self.model_version,
                "fall_event_tcn_observed_frame_count": result.get(
                    "fall_event_tcn_shadow_observed_frame_count"
                ),
                "fall_event_tcn_valid_joint_ratio": result.get(
                    "fall_event_tcn_shadow_valid_joint_ratio"
                ),
                "fall_event_tcn_duration_ms": result.get(
                    "fall_event_tcn_shadow_duration_ms"
                ),
            }
        return {
            "fall_event_tcn_score": float(score),
            "fall_event_tcn_detected": bool(
                result.get("fall_event_tcn_shadow_detected", False)
            ),
            "fall_event_tcn_threshold": self.threshold,
            "fall_event_tcn_onset_frame": result.get(
                "fall_event_tcn_shadow_onset_frame"
            ),
            "fall_event_tcn_status": "valid",
            "fall_event_tcn_score_source": "continuous_tcn",
            "fall_event_tcn_model_version": self.model_version,
            "fall_event_tcn_checkpoint_count": result.get(
                "fall_event_tcn_shadow_checkpoint_count"
            ),
            "fall_event_tcn_observed_frame_count": result.get(
                "fall_event_tcn_shadow_observed_frame_count"
            ),
            "fall_event_tcn_valid_joint_ratio": result.get(
                "fall_event_tcn_shadow_valid_joint_ratio"
            ),
            "fall_event_tcn_duration_ms": result.get(
                "fall_event_tcn_shadow_duration_ms"
            ),
        }


class _CausalResidualBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        *,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.padding = padding
        self.conv1 = nn.Conv1d(
            input_channels,
            output_channels,
            kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.conv2 = nn.Conv1d(
            output_channels,
            output_channels,
            kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.norm1 = nn.LayerNorm(output_channels)
        self.norm2 = nn.LayerNorm(output_channels)
        self.dropout = nn.Dropout(dropout)
        self.skip = (
            nn.Identity()
            if input_channels == output_channels
            else nn.Conv1d(input_channels, output_channels, kernel_size=1)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        residual = self.skip(inputs)
        hidden = self._trim_right(self.conv1(inputs))
        hidden = torch.relu(self.norm1(hidden.transpose(1, 2)).transpose(1, 2))
        hidden = self.dropout(hidden)
        hidden = self._trim_right(self.conv2(hidden))
        hidden = torch.relu(self.norm2(hidden.transpose(1, 2)).transpose(1, 2))
        hidden = self.dropout(hidden)
        return torch.relu(hidden + residual)

    def _trim_right(self, value: Tensor) -> Tensor:
        # Conv1d padding is symmetric; discard the future-side positions.
        return value[..., :-self.padding] if self.padding else value


class _ContinuousFallDataset(Dataset[dict[str, Tensor]]):
    def __init__(self, arrays: dict[str, np.ndarray], indices: np.ndarray) -> None:
        self.features = torch.from_numpy(arrays["features"][indices]).float()
        self.presence = torch.from_numpy(arrays["presence_targets"][indices]).float()
        self.onset = torch.from_numpy(arrays["onset_targets"][indices]).long()
        self.onset_mask = torch.from_numpy(arrays["onset_masks"][indices]).float()
        self.sample_weight = torch.from_numpy(arrays["sample_weights"][indices]).float()
        self.presence_weight = torch.from_numpy(
            arrays["presence_loss_weights"][indices]
        ).float()
        self.onset_weight = torch.from_numpy(arrays["onset_loss_weights"][indices]).float()

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        feature = self.features[index].permute(1, 2, 0).reshape(-1, self.features.shape[1])
        return {
            "features": feature,
            "presence": self.presence[index],
            "onset": self.onset[index],
            "onset_mask": self.onset_mask[index],
            "sample_weight": self.sample_weight[index],
            "presence_weight": self.presence_weight[index],
            "onset_weight": self.onset_weight[index],
        }


def train_continuous_fall_tcn(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    metadata_path: str | Path | None = None,
    config: ContinuousFallTCNConfig | None = None,
) -> dict[str, Any]:
    training = config or ContinuousFallTCNConfig()
    _set_seed(training.seed)
    source_path = Path(dataset_path)
    metadata_file = Path(metadata_path) if metadata_path else source_path.with_name("metadata.json")
    arrays = _load_arrays(source_path)
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    partitions = arrays["partitions"].astype(str)
    train_indices = np.flatnonzero(partitions == "train")
    validation_indices = np.flatnonzero(partitions == "validation")
    if not len(train_indices) or not len(validation_indices):
        raise ValueError("continuous fall dataset needs train and validation samples")
    mean, std = _fit_normalization(arrays["features"][train_indices])
    normalized = dict(arrays)
    normalized["features"] = (
        (arrays["features"] - mean[None, None, :, :])
        / std[None, None, :, :]
    ).astype(np.float32)
    train_set = _ContinuousFallDataset(normalized, train_indices)
    validation_set = _ContinuousFallDataset(normalized, validation_indices)
    device = _resolve_device(training.device)
    model = ContinuousFallTCN(
        input_channels=normalized["features"].shape[2] * normalized["features"].shape[3],
        hidden_channels=training.hidden_channels,
        kernel_size=training.kernel_size,
        dilations=training.dilations,
        dropout=training.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training.learning_rate, weight_decay=training.weight_decay
    )
    train_loader = DataLoader(train_set, batch_size=training.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=training.batch_size, shuffle=False)
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"continuous fall TCN output already exists: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    history: list[dict[str, Any]] = []
    best_score = -1.0
    best_epoch = -1
    stale = 0
    best_state: dict[str, Tensor] | None = None
    for epoch in range(1, training.epochs + 1):
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            batch = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            presence_logits, onset_logits = model(batch["features"])
            losses = _losses(presence_logits, onset_logits, batch)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(float(losses["loss"].detach().cpu()))
        validation = _evaluate(model, validation_loader, device)
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            **validation,
        }
        history.append(record)
        score = float(validation["presence_f1"])
        if score > best_score + 1e-8:
            best_score = score
            best_epoch = epoch
            stale = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= training.patience:
                break
    if best_state is None:
        raise RuntimeError("continuous fall TCN did not produce a checkpoint")
    model.load_state_dict(best_state)
    # Keep checkpoint and CLI artifacts bounded: validation metrics are aggregate-only.
    final_validation = _evaluate(model, validation_loader, device)
    checkpoint = {
        "schema_version": "fall-event-continuous-tcn-v1",
        "status": "development_provisional",
        "task": "fall_event_continuous_presence_onset",
        "target_task": "fall_event_v1",
        "model_config": {
            "input_channels": int(normalized["features"].shape[2] * normalized["features"].shape[3]),
            "hidden_channels": training.hidden_channels,
            "kernel_size": training.kernel_size,
            "dilations": list(training.dilations),
            "dropout": training.dropout,
        },
        "dataset_sha256": _sha256_file(source_path),
        "metadata_sha256": _sha256_file(metadata_file),
        "normalization": {"mean": mean.tolist(), "std": std.tolist()},
        "best_epoch": best_epoch,
        "validation_metrics": final_validation,
        "test_evaluated": False,
        "main_path_replacement": False,
        "state_dict": best_state,
    }
    torch.save(checkpoint, destination / "best_model.pt")
    (destination / "config.json").write_text(
        json.dumps({"schema_version": "fall-event-continuous-training-config-v1", "training": asdict(training)}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (destination / "history.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in history), encoding="utf-8"
    )
    (destination / "metrics.json").write_text(
        json.dumps(
            {
                "schema_version": "fall-event-continuous-tcn-metrics-v1",
                "status": "development_provisional",
                "best_epoch": best_epoch,
                "history": history,
                "validation": final_validation,
                "test_evaluated": False,
                "main_path_replacement": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "output_dir": destination.as_posix(),
        "checkpoint_path": (destination / "best_model.pt").as_posix(),
        "metrics_path": (destination / "metrics.json").as_posix(),
        "best_epoch": best_epoch,
        "validation": final_validation,
    }


def _losses(presence_logits: Tensor, onset_logits: Tensor, batch: dict[str, Tensor]) -> dict[str, Tensor]:
    presence_loss = nn.functional.binary_cross_entropy_with_logits(
        presence_logits,
        batch["presence"],
        reduction="none",
    )
    presence_weight = batch["sample_weight"] * batch["presence_weight"]
    presence_loss = (presence_loss * presence_weight).sum() / presence_weight.sum().clamp_min(1e-6)
    onset_mask = batch["onset_mask"] > 0
    if torch.any(onset_mask):
        onset_loss = nn.functional.cross_entropy(
            onset_logits[onset_mask], batch["onset"][onset_mask], reduction="none"
        )
        onset_weight = (
            batch["sample_weight"][onset_mask] * batch["onset_weight"][onset_mask]
        )
        onset_loss = (onset_loss * onset_weight).sum() / onset_weight.sum().clamp_min(1e-6)
    else:
        onset_loss = presence_loss.new_zeros(())
    return {"loss": presence_loss + 0.5 * onset_loss, "presence_loss": presence_loss, "onset_loss": onset_loss}


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    onset_hits = 0
    onset_count = 0
    losses: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            presence_logits, onset_logits = model(batch["features"])
            losses.append(float(_losses(presence_logits, onset_logits, batch)["loss"].cpu()))
            probabilities.append(torch.sigmoid(presence_logits).cpu().numpy())
            targets.append(batch["presence"].cpu().numpy())
            mask = batch["onset_mask"] > 0
            if torch.any(mask):
                onset_hits += int((onset_logits[mask].argmax(dim=-1) == batch["onset"][mask]).sum())
                onset_count += int(mask.sum())
    probability = np.concatenate(probabilities)
    target = np.concatenate(targets).astype(np.int64)
    predicted = probability >= 0.5
    tp = int(np.sum(predicted & (target == 1)))
    fp = int(np.sum(predicted & (target == 0)))
    fn = int(np.sum(~predicted & (target == 1)))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    metrics: dict[str, Any] = {
        "loss": float(np.mean(losses)),
        "presence_precision": round(precision, 6),
        "presence_recall": round(recall, 6),
        "presence_f1": round(f1, 6),
        "presence_pr_auc": round(float(average_precision_score(target, probability)), 6)
        if len(np.unique(target)) > 1
        else None,
        "presence_positive_count": int(np.sum(target == 1)),
        "presence_negative_count": int(np.sum(target == 0)),
        "onset_accuracy": round(onset_hits / onset_count, 6) if onset_count else None,
        "onset_supervised_count": onset_count,
    }
    return metrics


def _fit_normalization(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = features.mean(axis=(0, 1))
    std = features.std(axis=(0, 1))
    # Keep frame-mask channels in their native 0/1 representation for pooling.
    mean[:, 14] = 0.0
    std[:, 14] = 1.0
    std = np.where(std < 1e-5, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "features",
            "presence_targets",
            "onset_targets",
            "onset_masks",
            "sample_weights",
            "presence_loss_weights",
            "onset_loss_weights",
            "partitions",
        }
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"continuous fall dataset missing arrays: {sorted(missing)}")
        return {name: archive[name] for name in archive.files}


def _to_device(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _resolve_device(value: str) -> torch.device:
    if value == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if value == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _frame_mask_channel(input_channels: int) -> int:
    channel_count = 20
    joint_count = 17
    expected = joint_count * channel_count
    if input_channels != expected:
        raise ValueError(f"continuous fall TCN requires {expected} flattened channels")
    return 14


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_shadow_checkpoint(checkpoint: Mapping[str, Any]) -> None:
    if checkpoint.get("schema_version") != "fall-event-continuous-tcn-v1":
        raise ValueError("unsupported continuous fall shadow checkpoint")
    if checkpoint.get("status") != "development_provisional":
        raise ValueError("continuous fall shadow checkpoint status is invalid")
    if checkpoint.get("task") != "fall_event_continuous_presence_onset":
        raise ValueError("continuous fall shadow checkpoint task mismatch")
    if checkpoint.get("test_evaluated") is not False:
        raise ValueError("continuous fall shadow checkpoint must remain test locked")
    if checkpoint.get("main_path_replacement") is not False:
        raise ValueError("continuous fall shadow checkpoint cannot be a main-path model")
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, Mapping) or int(model_config.get("input_channels", 0)) != 340:
        raise ValueError("continuous fall shadow checkpoint input contract mismatch")
    normalization = checkpoint.get("normalization")
    if not isinstance(normalization, Mapping):
        raise ValueError("continuous fall shadow checkpoint normalization is missing")
    mean = np.asarray(normalization.get("mean"), dtype=np.float32)
    std = np.asarray(normalization.get("std"), dtype=np.float32)
    if mean.shape != (17, 20) or std.shape != (17, 20):
        raise ValueError("continuous fall shadow checkpoint normalization shape mismatch")


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 4)
