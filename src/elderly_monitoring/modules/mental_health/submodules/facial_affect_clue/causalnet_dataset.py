from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


EXPECTED_INPUT_SHAPE = (4, 3, 28, 28)
EXPECTED_ROUTE_ORDER = (
    "onset_to_apex_flow",
    "apex_to_offset_flow",
    "onset_to_apex_direction",
    "apex_to_offset_direction",
)


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
        if label not in (0, 1, 2) or label != int(record["label"]):
            raise ValueError(f"{record['sample_id']} has mismatched label")
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
        return {
            "inputs": torch.from_numpy(inputs),
            "label": torch.tensor(label, dtype=torch.long),
            "sample_id": str(record["sample_id"]),
            "subject_id": str(record["subject_id"]),
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
        "metadata": [sample["metadata"] for sample in samples],
    }
