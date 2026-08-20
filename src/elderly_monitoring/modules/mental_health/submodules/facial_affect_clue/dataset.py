from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


ARTIFACT_ARRAY_SHAPES = {
    "patches": (72, 147),
    "region_ids": (72,),
    "masks": (72,),
    "keypoints": (72, 2),
}


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_artifact_manifest(path: Path) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    sample_ids = [str(record["sample_id"]) for record in records]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Artifact manifest contains duplicate sample_id values")
    for record in records:
        artifact_path = Path(str(record["artifact_path"]))
        if not artifact_path.is_file():
            raise FileNotFoundError(artifact_path)
        if record.get("label") not in (0, 1, 2):
            raise ValueError(f"Invalid classification label: {record.get('label')}")
    return records


def records_by_sample_id(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {str(record["sample_id"]): dict(record) for record in records}


class MicroexpressionArtifactDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        sample_ids: Sequence[str] | None = None,
        preload: bool = True,
    ) -> None:
        by_id = records_by_sample_id(records)
        if sample_ids is None:
            selected = [dict(record) for record in records]
        else:
            missing = sorted(set(sample_ids) - set(by_id))
            if missing:
                raise KeyError(f"Unknown sample ids: {missing[:5]}")
            selected = [by_id[sample_id] for sample_id in sample_ids]
        if not selected:
            raise ValueError("Dataset selection is empty")
        self.records = selected
        self._cache: list[dict[str, np.ndarray] | None] = [None] * len(selected)
        if preload:
            self._cache = [self._load_arrays(record) for record in selected]

    def _load_arrays(self, record: Mapping[str, Any]) -> dict[str, np.ndarray]:
        artifact_path = Path(str(record["artifact_path"]))
        with np.load(artifact_path, allow_pickle=False) as artifact:
            arrays = {name: np.asarray(artifact[name]).copy() for name in ARTIFACT_ARRAY_SHAPES}
            label = int(artifact["label"])
        for name, expected_shape in ARTIFACT_ARRAY_SHAPES.items():
            if arrays[name].shape != expected_shape:
                raise ValueError(
                    f"{record['sample_id']} {name} has shape {arrays[name].shape}, "
                    f"expected {expected_shape}"
                )
            if not np.all(np.isfinite(arrays[name])):
                raise ValueError(f"{record['sample_id']} {name} contains non-finite values")
        if label != int(record["label"]):
            raise ValueError(f"{record['sample_id']} label mismatch")
        arrays["label"] = np.asarray(label, dtype=np.int64)
        return arrays

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        arrays = self._cache[index]
        if arrays is None:
            arrays = self._load_arrays(record)
        return {
            "patches": torch.from_numpy(arrays["patches"]).float(),
            "region_ids": torch.from_numpy(arrays["region_ids"]).long(),
            "masks": torch.from_numpy(arrays["masks"]).float(),
            "keypoints": torch.from_numpy(arrays["keypoints"]).float(),
            "label": torch.as_tensor(int(arrays["label"]), dtype=torch.long),
            "sample_id": str(record["sample_id"]),
            "subject_id": str(record["subject_id"]),
        }


def class_counts(records: Sequence[Mapping[str, Any]]) -> dict[int, int]:
    counts = {0: 0, 1: 0, 2: 0}
    for record in records:
        counts[int(record["label"])] += 1
    return counts
