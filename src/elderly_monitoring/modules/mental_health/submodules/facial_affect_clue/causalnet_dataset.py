from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


EXPECTED_INPUT_SHAPE = (4, 3, 28, 28)
EXPECTED_ROUTE_ORDER = (
    "onset_to_apex_flow",
    "apex_to_offset_flow",
    "onset_to_apex_direction",
    "apex_to_offset_direction",
)


@dataclass(frozen=True)
class CausalNetAugmentationConfig:
    enabled: bool = False
    horizontal_flip_probability: float = 0.0
    flow_magnitude_jitter: float = 0.0
    spatial_shift_pixels: int = 0
    route_dropout_probability: float = 0.0
    flow_noise_std: float = 0.0
    brightness_jitter: float = 0.0

    def __post_init__(self) -> None:
        probabilities = (
            self.horizontal_flip_probability,
            self.route_dropout_probability,
        )
        if any(value < 0.0 or value > 1.0 for value in probabilities):
            raise ValueError("Augmentation probabilities must be in [0, 1]")
        if self.flow_magnitude_jitter < 0.0 or self.flow_noise_std < 0.0:
            raise ValueError("Flow augmentation magnitudes must be non-negative")
        if self.spatial_shift_pixels < 0 or self.spatial_shift_pixels > 2:
            raise ValueError("Spatial shift is deliberately limited to at most two pixels")
        if self.brightness_jitter < 0.0 or self.brightness_jitter > 0.1:
            raise ValueError("Brightness jitter is deliberately limited to [0, 0.1]")


def _augment_inputs(
    inputs: torch.Tensor,
    config: CausalNetAugmentationConfig,
    generator: torch.Generator,
) -> torch.Tensor:
    if not config.enabled:
        return inputs
    result = inputs.clone()
    if torch.rand((), generator=generator).item() < config.horizontal_flip_probability:
        result = torch.flip(result, dims=(-1,))
        # Horizontal reflection reverses the u component in both flow routes.
        result[0:2, 0] *= -1.0
        # Direction routes store angle in HSV hue. Mirror theta to pi-theta.
        for route in (2, 3):
            rgb = result[route].permute(1, 2, 0).numpy()
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            hsv[..., 0] = np.mod(180.0 - hsv[..., 0], 360.0)
            mirrored = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            result[route] = torch.from_numpy(mirrored).permute(2, 0, 1)
    if config.flow_magnitude_jitter > 0.0:
        lower = 1.0 - config.flow_magnitude_jitter
        upper = 1.0 + config.flow_magnitude_jitter
        factor = torch.empty((), dtype=result.dtype).uniform_(lower, upper, generator=generator)
        result[0:2] *= factor
    if config.spatial_shift_pixels:
        shift_y = int(torch.randint(-config.spatial_shift_pixels, config.spatial_shift_pixels + 1, (), generator=generator).item())
        shift_x = int(torch.randint(-config.spatial_shift_pixels, config.spatial_shift_pixels + 1, (), generator=generator).item())
        padding = config.spatial_shift_pixels
        padded = F.pad(result, (padding, padding, padding, padding))
        start_y = padding - shift_y
        start_x = padding - shift_x
        result = padded[..., start_y : start_y + 28, start_x : start_x + 28]
    if torch.rand((), generator=generator).item() < config.route_dropout_probability:
        route = int(torch.randint(0, result.shape[0], (), generator=generator).item())
        result[route] = 0.0
    if config.flow_noise_std > 0.0:
        noise = torch.randn(result[0:2].shape, generator=generator, dtype=result.dtype)
        result[0:2] += noise * config.flow_noise_std
    if config.brightness_jitter > 0.0:
        factor = torch.empty((), dtype=result.dtype).uniform_(
            1.0 - config.brightness_jitter,
            1.0 + config.brightness_jitter,
            generator=generator,
        )
        result[2:4] *= factor
    result[0:2, 0:2].clamp_(-1.0, 1.0)
    result[0:2, 2].clamp_(0.0, 1.0)
    result[2:4].clamp_(0.0, 1.0)
    return result


def _records_by_id(records: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    by_id = {str(record["sample_id"]): record for record in records}
    if len(by_id) != len(records):
        raise ValueError("CausalNet manifest contains duplicate sample_id values")
    return by_id


class CausalNetArtifactDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        sample_ids: Sequence[str] | None = None,
        preload: bool = False,
        augmentation: CausalNetAugmentationConfig | None = None,
        augmentation_seed: int = 0,
    ) -> None:
        by_id = _records_by_id(records)
        if sample_ids is None:
            selected = list(records)
        else:
            missing = sorted(set(sample_ids) - set(by_id))
            if missing:
                raise KeyError(f"Unknown CausalNet sample ids: {missing[:5]}")
            selected = [by_id[sample_id] for sample_id in sample_ids]
        if not selected:
            raise ValueError("CausalNet dataset selection is empty")
        self.records = [dict(record) for record in selected]
        self._cache = [self._load(record) for record in self.records] if preload else None
        self.augmentation = augmentation or CausalNetAugmentationConfig()
        self._augmentation_rng = torch.Generator().manual_seed(augmentation_seed)

    @staticmethod
    def _load(record: Mapping[str, Any]) -> tuple[np.ndarray, int, dict[str, Any]]:
        path = Path(str(record["artifact_path"]))
        with np.load(path, allow_pickle=False) as artifact:
            missing = {"inputs", "label", "metadata_json"} - set(artifact.files)
            if missing:
                raise ValueError(f"{record['sample_id']} missing arrays: {sorted(missing)}")
            inputs = np.asarray(artifact["inputs"], dtype=np.float32)
            label = int(np.asarray(artifact["label"]).item())
            metadata = json.loads(str(np.asarray(artifact["metadata_json"]).item()))
        if inputs.shape != EXPECTED_INPUT_SHAPE:
            raise ValueError(f"{record['sample_id']} has shape {inputs.shape}, expected {EXPECTED_INPUT_SHAPE}")
        if not np.isfinite(inputs).all():
            raise ValueError(f"{record['sample_id']} contains NaN or Inf")
        evaluation_label = int(record["label"])
        if label in (0, 1, 2):
            if label != evaluation_label:
                raise ValueError(f"{record['sample_id']} has mismatched label")
        elif label == -1:
            if (
                record.get("external_label_authority")
                != "casme2_three_class_crosswalk_v1"
                or metadata.get("source_dataset") != "casme2_official"
                or metadata.get("project_three_class_label") is not None
                or record.get("artifact_stored_label") != -1
                or evaluation_label not in (0, 1, 2)
            ):
                raise ValueError(
                    f"{record['sample_id']} has unaudited external label authority"
                )
            label = evaluation_label
        else:
            raise ValueError(f"{record['sample_id']} has invalid stored label")
        if tuple(metadata.get("route_order", ())) != EXPECTED_ROUTE_ORDER:
            raise ValueError(f"{record['sample_id']} has invalid route order")
        if metadata.get("input_shape") != list(EXPECTED_INPUT_SHAPE):
            raise ValueError(f"{record['sample_id']} has invalid metadata input shape")
        if not int(metadata["onset_index"]) < int(metadata["apex_index"]) < int(metadata["offset_index"]):
            raise ValueError(f"{record['sample_id']} has invalid key-frame order")
        return inputs, label, metadata

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        inputs, label, metadata = (
            self._cache[index] if self._cache is not None else self._load(record)
        )
        tensor = torch.from_numpy(inputs)
        if self.augmentation.enabled:
            tensor = _augment_inputs(tensor, self.augmentation, self._augmentation_rng)
        return {
            "inputs": tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "sample_id": str(record["sample_id"]),
            "subject_id": str(record["subject_id"]),
            "evaluation_dataset": str(
                record.get("evaluation_dataset", metadata.get("source_dataset", "unknown"))
            ),
            "metadata": metadata,
        }


def collate_causalnet(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate an empty CausalNet batch")
    return {
        "inputs": torch.stack([sample["inputs"] for sample in samples]),
        "labels": torch.stack([sample["label"] for sample in samples]),
        "sample_ids": [str(sample["sample_id"]) for sample in samples],
        "subject_ids": [str(sample["subject_id"]) for sample in samples],
        "evaluation_datasets": [
            str(sample["evaluation_dataset"]) for sample in samples
        ],
        "metadata": [sample["metadata"] for sample in samples],
    }
