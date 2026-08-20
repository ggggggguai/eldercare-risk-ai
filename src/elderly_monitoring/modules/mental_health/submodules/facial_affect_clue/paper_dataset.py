from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


PAPER_REGION_COUNTS = (5, 5, 6, 6, 9, 12)
PAPER_REGION_OFFSETS = (0, 5, 10, 16, 22, 31, 43)
PAPER_PATCH_DIM = 75
PAPER_PATCH_COUNT = 43
PAPER_MAX_PATCHES_PER_REGION = 12
PAPER_ORDERED_LANDMARK_INDICES = (
    21,
    20,
    19,
    18,
    17,
    22,
    23,
    24,
    25,
    26,
    39,
    38,
    40,
    37,
    41,
    36,
    42,
    43,
    47,
    44,
    46,
    45,
    27,
    28,
    29,
    30,
    31,
    35,
    32,
    34,
    33,
    48,
    54,
    49,
    53,
    59,
    55,
    50,
    52,
    58,
    56,
    51,
    57,
)


def _records_by_id(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    by_id = {str(record["sample_id"]): record for record in records}
    if len(by_id) != len(records):
        raise ValueError("Paper-v2 records contain duplicate sample_id values")
    return by_id


class PaperV2ArtifactDataset(Dataset[dict[str, Any]]):
    """Strict reader for the 43 x 75 FLOW-ME-002 classification artifacts."""

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        sample_ids: Sequence[str] | None = None,
        preload: bool = True,
    ) -> None:
        by_id = _records_by_id(records)
        if sample_ids is None:
            selected = list(records)
        else:
            missing = sorted(set(sample_ids) - set(by_id))
            if missing:
                raise KeyError(f"Unknown paper-v2 sample ids: {missing[:5]}")
            selected = [by_id[sample_id] for sample_id in sample_ids]
        if not selected:
            raise ValueError("Paper-v2 dataset selection is empty")
        self.records = [dict(record) for record in selected]
        self._cache: list[dict[str, np.ndarray] | None] = [None] * len(selected)
        if preload:
            self._cache = [self._load_arrays(record) for record in self.records]

    @staticmethod
    def _load_arrays(record: Mapping[str, Any]) -> dict[str, np.ndarray]:
        artifact_path = Path(str(record["artifact_path"]))
        if not artifact_path.is_file():
            raise FileNotFoundError(artifact_path)
        with np.load(artifact_path, allow_pickle=False) as artifact:
            required = {
                "patches",
                "region_ids",
                "region_offsets",
                "keypoints",
                "ordered_landmark_indices",
                "label",
            }
            missing = required - set(artifact.files)
            if missing:
                raise ValueError(
                    f"{record['sample_id']} is missing arrays: {sorted(missing)}"
                )
            arrays = {name: np.asarray(artifact[name]).copy() for name in required}

        expected_shapes = {
            "patches": (PAPER_PATCH_COUNT, PAPER_PATCH_DIM),
            "region_ids": (PAPER_PATCH_COUNT,),
            "region_offsets": (len(PAPER_REGION_OFFSETS),),
            "keypoints": (PAPER_PATCH_COUNT, 2),
            "ordered_landmark_indices": (PAPER_PATCH_COUNT,),
        }
        for name, expected in expected_shapes.items():
            if arrays[name].shape != expected:
                raise ValueError(
                    f"{record['sample_id']} {name} has shape {arrays[name].shape}, "
                    f"expected {expected}"
                )
            if not np.all(np.isfinite(arrays[name])):
                raise ValueError(
                    f"{record['sample_id']} {name} contains non-finite values"
                )

        expected_region_ids = np.repeat(np.arange(6), PAPER_REGION_COUNTS)
        if not np.array_equal(arrays["region_ids"], expected_region_ids):
            raise ValueError(f"{record['sample_id']} has invalid region ordering")
        if tuple(arrays["region_offsets"].tolist()) != PAPER_REGION_OFFSETS:
            raise ValueError(f"{record['sample_id']} has invalid region offsets")
        if tuple(arrays["ordered_landmark_indices"].tolist()) != (
            PAPER_ORDERED_LANDMARK_INDICES
        ):
            raise ValueError(f"{record['sample_id']} has invalid paper landmark order")
        label = int(arrays["label"])
        if label not in (0, 1, 2) or label != int(record["label"]):
            raise ValueError(f"{record['sample_id']} has an invalid or mismatched label")
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
            "region_offsets": torch.from_numpy(arrays["region_offsets"]).long(),
            "keypoints": torch.from_numpy(arrays["keypoints"]).float(),
            "ordered_landmark_indices": torch.from_numpy(
                arrays["ordered_landmark_indices"]
            ).long(),
            "label": torch.as_tensor(int(arrays["label"]), dtype=torch.long),
            "sample_id": str(record["sample_id"]),
            "subject_id": str(record["subject_id"]),
            "quality_status": str(record.get("quality_status", "unknown")),
            "apex_boundary_status": str(
                record.get("apex_boundary_status", "unknown")
            ),
        }


class PaperV2Collator:
    def __init__(self, *, order_mode: str = "paper", order_seed: int = 20260806):
        if order_mode not in {"paper", "standard_index", "random_fixed"}:
            raise ValueError(f"Unsupported patch order mode: {order_mode}")
        self.order_mode = order_mode
        self.order_seed = order_seed
        generator = np.random.default_rng(order_seed)
        self.random_permutations = tuple(
            torch.from_numpy(generator.permutation(count)).long()
            for count in PAPER_REGION_COUNTS
        )

    def _permutation(
        self,
        sample: Mapping[str, Any],
        region: int,
        start: int,
        stop: int,
    ) -> torch.Tensor:
        count = stop - start
        if self.order_mode == "paper":
            return torch.arange(count)
        if self.order_mode == "random_fixed":
            return self.random_permutations[region]
        indices = sample["ordered_landmark_indices"][start:stop]
        return torch.argsort(indices, stable=True)

    def __call__(self, samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Pad each of the six real regions independently to 12 model slots."""
        if not samples:
            raise ValueError("Cannot collate an empty paper-v2 batch")
        batch_size = len(samples)
        patches = torch.zeros(
            batch_size,
            6,
            PAPER_MAX_PATCHES_PER_REGION,
            PAPER_PATCH_DIM,
            dtype=torch.float32,
        )
        keypoints = torch.zeros(
            batch_size, 6, PAPER_MAX_PATCHES_PER_REGION, 2, dtype=torch.float32
        )
        masks = torch.zeros(
            batch_size, 6, PAPER_MAX_PATCHES_PER_REGION, dtype=torch.bool
        )
        landmark_indices = torch.full(
            (batch_size, 6, PAPER_MAX_PATCHES_PER_REGION), -1, dtype=torch.long
        )

        for batch_index, sample in enumerate(samples):
            offsets = tuple(int(value) for value in sample["region_offsets"].tolist())
            if offsets != PAPER_REGION_OFFSETS:
                raise ValueError(f"Unexpected region offsets in batch item {batch_index}")
            for region, (start, stop) in enumerate(
                zip(offsets[:-1], offsets[1:], strict=True)
            ):
                count = stop - start
                order = self._permutation(sample, region, start, stop)
                patches[batch_index, region, :count] = sample["patches"][start:stop][
                    order
                ]
                keypoints[batch_index, region, :count] = sample["keypoints"][
                    start:stop
                ][order]
                masks[batch_index, region, :count] = True
                landmark_indices[batch_index, region, :count] = sample[
                    "ordered_landmark_indices"
                ][start:stop][order]

        return {
            "patches": patches,
            "keypoints": keypoints,
            "masks": masks,
            "region_ids": torch.arange(6).view(1, 6, 1).expand(
                batch_size, 6, PAPER_MAX_PATCHES_PER_REGION
            ).contiguous(),
            "ordered_landmark_indices": landmark_indices,
            "labels": torch.stack([sample["label"] for sample in samples]),
            "sample_ids": [str(sample["sample_id"]) for sample in samples],
            "subject_ids": [str(sample["subject_id"]) for sample in samples],
            "quality_status": [str(sample["quality_status"]) for sample in samples],
            "apex_boundary_status": [
                str(sample["apex_boundary_status"]) for sample in samples
            ],
            "order_mode": self.order_mode,
        }


def collate_paper_v2(
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return PaperV2Collator()(samples)
