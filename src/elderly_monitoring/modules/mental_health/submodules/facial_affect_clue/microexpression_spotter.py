from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


SPOTTER_SCHEMA_VERSION = "smic_window_spotter_features_v1"
SPOTTER_TASK_ID = "SPOT-ME-002"
WINDOW_LABELS = {0: "non_micro", 1: "micro"}
_IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class SpotterFeatureConfig:
    image_size: int = 64
    max_frames: int = 16
    temporal_bins: int = 8
    spatial_grid: int = 4
    flow_winsize: int = 15
    flow_levels: int = 3
    flow_iterations: int = 3
    flow_poly_n: int = 5
    flow_poly_sigma: float = 1.2

    def __post_init__(self) -> None:
        if self.image_size < 16 or self.max_frames < 3 or self.temporal_bins < 2:
            raise ValueError("image_size and temporal_bins are too small")
        if self.spatial_grid < 2 or self.image_size % self.spatial_grid:
            raise ValueError("spatial_grid must divide image_size")
        if self.flow_winsize < 3 or self.flow_levels < 1:
            raise ValueError("invalid optical-flow configuration")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode("utf-8")).hexdigest()


def _natural_key(path: Path) -> list[Any]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def list_frame_paths(frame_dir: str | Path) -> list[Path]:
    directory = Path(frame_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Frame directory does not exist: {directory}")
    paths = sorted(
        (path for path in directory.iterdir() if path.suffix.lower() in _IMAGE_SUFFIXES),
        key=_natural_key,
    )
    if len(paths) < 3:
        raise ValueError(f"A spotting window requires at least 3 frames: {directory}")
    return paths


def _read_gray(path: Path, image_size: int) -> np.ndarray:
    encoded = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot decode frame: {path}")
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    image = cv2.equalizeHist(image).astype(np.float32) / 255.0
    return image


def _temporal_profile(values: np.ndarray, bins: int) -> np.ndarray:
    if values.ndim != 3 or values.shape[0] == 0:
        raise ValueError("Temporal profile expects [T,H,W] with T > 0")
    chunks: list[np.ndarray] = []
    for index in range(bins):
        start = int(np.floor(index * values.shape[0] / bins))
        stop = int(np.floor((index + 1) * values.shape[0] / bins))
        stop = max(stop, start + 1)
        chunk = values[start:min(stop, values.shape[0])]
        chunks.extend(
            [
                np.mean(chunk, axis=(1, 2)).mean(),
                np.std(chunk, axis=(1, 2)).mean(),
                np.quantile(chunk, 0.90, axis=(1, 2)).mean(),
            ]
        )
    return np.asarray(chunks, dtype=np.float32)


def _spatial_profile(values: np.ndarray, grid: int) -> np.ndarray:
    if values.ndim != 3 or values.shape[0] == 0:
        raise ValueError("Spatial profile expects [T,H,W] with T > 0")
    height, width = values.shape[1:]
    cell_h, cell_w = height // grid, width // grid
    result: list[float] = []
    for row in range(grid):
        for column in range(grid):
            cell = values[
                :, row * cell_h : (row + 1) * cell_h,
                column * cell_w : (column + 1) * cell_w,
            ]
            result.append(float(cell.mean()))
    return np.asarray(result, dtype=np.float32)


def extract_window_features(
    frame_paths: Sequence[str | Path],
    config: SpotterFeatureConfig | None = None,
) -> np.ndarray:
    config = config or SpotterFeatureConfig()
    paths = [Path(path) for path in frame_paths]
    if len(paths) < 3:
        raise ValueError("A spotting window requires at least 3 frames")
    if len(paths) > config.max_frames:
        indices = np.linspace(0, len(paths) - 1, config.max_frames, dtype=np.int64)
        paths = [paths[int(index)] for index in indices]
    frames = np.stack([_read_gray(path, config.image_size) for path in paths])
    deltas = np.abs(np.diff(frames, axis=0))
    displacement = np.abs(frames[1:] - frames[0])

    flow_magnitudes: list[np.ndarray] = []
    for first, second in zip(frames[:-1], frames[1:], strict=True):
        flow = cv2.calcOpticalFlowFarneback(
            first,
            second,
            None,
            0.5,
            config.flow_levels,
            config.flow_winsize,
            config.flow_iterations,
            config.flow_poly_n,
            config.flow_poly_sigma,
            0,
        )
        magnitude = cv2.magnitude(flow[..., 0], flow[..., 1])
        scale = float(np.quantile(magnitude, 0.995))
        flow_magnitudes.append(np.clip(magnitude / max(scale, 1e-6), 0.0, 1.0))
    flow_values = np.stack(flow_magnitudes)

    features = np.concatenate(
        [
            _temporal_profile(deltas, config.temporal_bins),
            _temporal_profile(displacement, config.temporal_bins),
            _temporal_profile(flow_values, config.temporal_bins),
            _spatial_profile(deltas, config.spatial_grid),
            _spatial_profile(displacement, config.spatial_grid),
            _spatial_profile(flow_values, config.spatial_grid),
        ]
    ).astype(np.float32)
    if not np.all(np.isfinite(features)):
        raise ValueError("Spotting feature extraction produced NaN or Inf")
    return features


def feature_schema(config: SpotterFeatureConfig) -> dict[str, Any]:
    dimension = config.temporal_bins * 9 + 3 * config.spatial_grid**2
    return {
        "schema_version": SPOTTER_SCHEMA_VERSION,
        "task_id": SPOTTER_TASK_ID,
        "feature_dim": dimension,
        "feature_groups": [
            "temporal_absolute_frame_difference",
            "temporal_onset_displacement",
            "temporal_optical_flow_magnitude",
            "spatial_absolute_frame_difference",
            "spatial_onset_displacement",
            "spatial_optical_flow_magnitude",
        ],
        "config": config.as_dict(),
        "config_sha256": config.fingerprint(),
        "uses_frame_count_as_feature": False,
    }


class SpotterFeatureDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, records: Sequence[Mapping[str, Any]]) -> None:
        if not records:
            raise ValueError("Spotter dataset cannot be empty")
        self.records = list(records)
        arrays = []
        labels = []
        for record in self.records:
            path = Path(str(record["feature_path"]))
            with np.load(path, allow_pickle=False) as data:
                features = np.asarray(data["features"], dtype=np.float32)
            if features.ndim != 1 or not np.all(np.isfinite(features)):
                raise ValueError(f"Invalid feature artifact: {path}")
            arrays.append(features)
            labels.append(int(record["label"]))
        dimensions = {array.shape[0] for array in arrays}
        if len(dimensions) != 1:
            raise ValueError("Spotter feature dimensions are inconsistent")
        self.features = torch.from_numpy(np.stack(arrays))
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


@dataclass(frozen=True)
class SpotterModelConfig:
    hidden_dim: int = 64
    dropout: float = 0.20

    def __post_init__(self) -> None:
        if self.hidden_dim < 4 or not 0 <= self.dropout < 1:
            raise ValueError("Invalid spotter model configuration")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class MicroExpressionWindowSpotter(nn.Module):
    def __init__(self, input_dim: int, config: SpotterModelConfig | None = None) -> None:
        super().__init__()
        config = config or SpotterModelConfig()
        self.config = config
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError(f"Expected [B,F] features, got {tuple(features.shape)}")
        logits = self.network(features).squeeze(-1)
        if not torch.isfinite(logits).all():
            raise ValueError("Spotter logits contain NaN or Inf")
        return logits
