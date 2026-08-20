from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .paper_dataset import PaperV2ArtifactDataset, PaperV2Collator


TEMPORAL_FEATURE_SHAPE_ERROR = "DSTM temporal input must be [T, 3, 32, 32]"


def _record_map(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {str(record["sample_id"]): dict(record) for record in records}
    if len(result) != len(records):
        raise ValueError("DSTM records contain duplicate sample ids")
    return result


class DSTMArtifactDataset(Dataset[dict[str, Any]]):
    """Join the paper-v2 spatial artifact with a complete temporal artifact."""

    def __init__(
        self,
        spatial_records: Sequence[Mapping[str, Any]],
        temporal_records: Sequence[Mapping[str, Any]],
        *,
        sample_ids: Sequence[str] | None = None,
        temporal_features: Mapping[str, np.ndarray] | None = None,
        preload: bool = True,
    ) -> None:
        spatial = _record_map(spatial_records)
        temporal = _record_map(temporal_records)
        ids = list(sample_ids) if sample_ids is not None else sorted(spatial)
        if not ids:
            raise ValueError("DSTM dataset selection is empty")
        missing_temporal = sorted(set(ids) - set(temporal))
        if missing_temporal:
            raise KeyError(f"Missing DSTM temporal artifacts: {missing_temporal[:5]}")
        self.records = [spatial[sample_id] for sample_id in ids]
        self.temporal_records = [temporal[sample_id] for sample_id in ids]
        self.temporal_features = temporal_features
        self._spatial = PaperV2ArtifactDataset(self.records, preload=preload)
        self._temporal_cache: list[np.ndarray | None] = [None] * len(ids)
        if preload:
            self._temporal_cache = [self._load_temporal(record) for record in self.temporal_records]

    @staticmethod
    def _load_temporal(record: Mapping[str, Any]) -> np.ndarray:
        path = Path(str(record["artifact_path"]))
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as artifact:
            required = {"flow_sequence", "frame_pairs", "label"}
            missing = required - set(artifact.files)
            if missing:
                raise ValueError(f"{record['sample_id']} missing {sorted(missing)}")
            sequence = np.asarray(artifact["flow_sequence"]).copy()
            pairs = np.asarray(artifact["frame_pairs"])
            label = int(artifact["label"])
        if sequence.ndim != 4 or sequence.shape[1:] != (3, 32, 32):
            raise ValueError(f"{record['sample_id']} {TEMPORAL_FEATURE_SHAPE_ERROR}: {sequence.shape}")
        if pairs.shape != (sequence.shape[0], 2):
            raise ValueError(f"{record['sample_id']} has invalid frame_pairs shape")
        onset_index = int(record.get("onset_index", 0))
        apex_index = int(record.get("apex_index", onset_index + sequence.shape[0]))
        if sequence.shape[0] != apex_index - onset_index:
            raise ValueError(
                f"{record['sample_id']} does not cover the onset-to-apex interval"
            )
        expected_pairs = np.column_stack(
            (
                np.arange(onset_index, apex_index),
                np.arange(onset_index + 1, apex_index + 1),
            )
        )
        if not np.array_equal(pairs, expected_pairs):
            raise ValueError(f"{record['sample_id']} does not contain adjacent frame pairs")
        if not np.isfinite(sequence).all():
            raise ValueError(f"{record['sample_id']} temporal sequence is non-finite")
        if not np.any(np.abs(sequence) > 1e-7):
            raise ValueError(f"{record['sample_id']} temporal sequence is all zero")
        if label != int(record["label"]):
            raise ValueError(f"{record['sample_id']} temporal label mismatch")
        return sequence.astype(np.float32)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        spatial = self._spatial[index]
        temporal = self._temporal_cache[index]
        if temporal is None:
            temporal = self._load_temporal(self.temporal_records[index])
        item = dict(spatial)
        item["flow_sequence"] = torch.from_numpy(temporal).float()
        if self.temporal_features is not None:
            sample_id = str(item["sample_id"])
            if sample_id not in self.temporal_features:
                raise KeyError(f"Missing fold temporal features for {sample_id}")
            transformed = np.asarray(self.temporal_features[sample_id], dtype=np.float32)
            if transformed.ndim != 2 or transformed.shape[0] != temporal.shape[0]:
                raise ValueError(
                    f"Fold temporal features for {sample_id} do not match sequence length"
                )
            if not np.isfinite(transformed).all():
                raise ValueError(f"Fold temporal features for {sample_id} are non-finite")
            item["temporal_features"] = torch.from_numpy(transformed).float()
        return item


class DSTMArtifactCollator:
    def __init__(self, *, order_mode: str = "paper") -> None:
        self.spatial_collator = PaperV2Collator(order_mode=order_mode)

    def __call__(self, samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not samples:
            raise ValueError("Cannot collate an empty DSTM batch")
        result = self.spatial_collator(samples)
        lengths = [int(sample["flow_sequence"].shape[0]) for sample in samples]
        max_length = max(lengths)
        batch_size = len(samples)
        flow_sequences = torch.zeros(
            batch_size, max_length, 3, 32, 32, dtype=torch.float32
        )
        temporal_mask = torch.zeros(batch_size, max_length, dtype=torch.bool)
        for index, sample in enumerate(samples):
            sequence = sample["flow_sequence"]
            length = sequence.shape[0]
            flow_sequences[index, :length] = sequence
            temporal_mask[index, :length] = True
        result["flow_sequences"] = flow_sequences
        result["temporal_mask"] = temporal_mask
        result["temporal_lengths"] = torch.as_tensor(lengths, dtype=torch.long)
        if all("temporal_features" in sample for sample in samples):
            feature_dim = int(samples[0]["temporal_features"].shape[-1])
            features = torch.zeros(batch_size, max_length, feature_dim, dtype=torch.float32)
            for index, sample in enumerate(samples):
                value = sample["temporal_features"]
                if value.shape[-1] != feature_dim:
                    raise ValueError("Temporal feature dimensions differ within a batch")
                features[index, : value.shape[0]] = value
            result["temporal_features"] = features
        elif any("temporal_features" in sample for sample in samples):
            raise ValueError("Either all or none of the DSTM batch items need temporal features")
        return result
