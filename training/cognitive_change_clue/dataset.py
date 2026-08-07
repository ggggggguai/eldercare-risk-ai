"""Strict Train/Validation dataset for cognitive-change clue V3.3."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

try:
    from .common import (
        DEFAULT_MANIFEST_PATH,
        DEFAULT_PROCESSED_ROOT,
        DEFAULT_SPLIT_PATH,
        FEATURE_VERSION,
        resolve_workspace_path,
        sha256_file,
    )
except ImportError:
    from common import (
        DEFAULT_MANIFEST_PATH,
        DEFAULT_PROCESSED_ROOT,
        DEFAULT_SPLIT_PATH,
        FEATURE_VERSION,
        resolve_workspace_path,
        sha256_file,
    )


MODALITY_ORDER = ("audio", "text", "face")
OUTPUT_DIMS = {"audio": 768, "text": 768, "face": 512}
TEXT_QUALITY_VERSION = "cognitive_text_quality_v3.3.1"
EXPECTED_INPUT_SHA256 = {
    "manifest": "ce0c4d0c59111b14f3304db2772b314cf1cc2be032872fd72a08a028c92f743d",
    "split": "05ee63247c721c10e355408c98218df86fae74a19be7eb735e8510d7d7b48e86",
    "audio_index": "f6e0ddf7867e22c021d39d8889e71075eb9d265cf2c3990ffb51196e980c6269",
    "text_index": "8d98bb590c7f6d8e0c98bad6082a1130eea0473371db2e08f288343c1a63eb33",
    "face_index": "6e3bdd841645097e9c92bcba602d4a859ea3fb4162b34bd8699be4fc8785e741",
    "feature_cache_manifest": "986de6445f73dbcbf1c54f9091de3b4ca78ce792f91b9432b7006959ec836262",
}


class CognitiveDatasetError(RuntimeError):
    pass


@dataclass(frozen=True)
class MocaStatistics:
    mean: float
    std: float
    subject_count: int

    def to_dict(self) -> dict[str, float | int]:
        return {"mean": self.mean, "std": self.std, "subject_count": self.subject_count}


@dataclass(frozen=True)
class TrainingWeights:
    hc_vs_non_hc_pos_weight: float
    mci_vs_hc_pos_weight: float
    ad_mci_hc_class_weights: tuple[float, float, float]
    task_counts: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "hc_vs_non_hc_pos_weight": self.hc_vs_non_hc_pos_weight,
            "mci_vs_hc_pos_weight": self.mci_vs_hc_pos_weight,
            "ad_mci_hc_class_weights": list(self.ad_mci_hc_class_weights),
            "task_counts": dict(self.task_counts),
        }


def default_input_paths(
    processed_root: str | Path = DEFAULT_PROCESSED_ROOT,
) -> dict[str, Path]:
    root = Path(processed_root)
    feature_root = root / "features"
    return {
        "manifest": root / "cogpic_manifest.parquet",
        "split": root / "cogpic_subject_split_v33.json",
        "audio_index": feature_root / "audio_index.jsonl",
        "text_index": feature_root / "text_index.jsonl",
        "face_index": feature_root / "face_index.jsonl",
        "feature_cache_manifest": feature_root / "feature_cache_manifest.json",
    }


def verify_frozen_input_hashes(
    paths: Mapping[str, str | Path],
    expected: Mapping[str, str] = EXPECTED_INPUT_SHA256,
) -> dict[str, str]:
    actual: dict[str, str] = {}
    for key, expected_hash in expected.items():
        if key not in paths:
            raise CognitiveDatasetError(f"missing input path: {key}")
        path = Path(paths[key])
        if not path.is_file():
            raise CognitiveDatasetError(f"input file is missing: {path}")
        actual[key] = sha256_file(path)
        if actual[key] != expected_hash:
            raise CognitiveDatasetError(
                f"frozen input hash mismatch for {key}: {actual[key]} != {expected_hash}"
            )
    return actual


class CognitiveFeatureDataset(Dataset[dict[str, Any]]):
    """Eagerly load only the requested Train or Validation feature rows."""

    def __init__(
        self,
        *,
        split_name: str,
        manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
        split_path: str | Path = DEFAULT_SPLIT_PATH,
        feature_root: str | Path = DEFAULT_PROCESSED_ROOT / "features",
        complete_only: bool = False,
        evaluation_only: bool = False,
        verify_hashes: bool = False,
        expected_hashes: Mapping[str, str] = EXPECTED_INPUT_SHA256,
    ) -> None:
        allowed_splits = {"train", "validation", "test"} if evaluation_only else {"train", "validation"}
        if split_name not in allowed_splits:
            raise CognitiveDatasetError(
                "Test is only permitted through an explicit evaluation-only dataset"
            )
        self.split_name = split_name
        self.manifest_path = Path(manifest_path)
        self.split_path = Path(split_path)
        self.feature_root = Path(feature_root)
        paths = {
            "manifest": self.manifest_path,
            "split": self.split_path,
            **{
                f"{name}_index": self.feature_root / f"{name}_index.jsonl"
                for name in MODALITY_ORDER
            },
            "feature_cache_manifest": self.feature_root / "feature_cache_manifest.json",
        }
        self.input_hashes = (
            verify_frozen_input_hashes(paths, expected_hashes)
            if verify_hashes
            else {key: sha256_file(value) for key, value in paths.items()}
        )

        frame = pd.read_parquet(self.manifest_path, engine="pyarrow")
        _validate_manifest(frame)
        split_payload = json.loads(self.split_path.read_text(encoding="utf-8"))
        allowed_subjects = set(split_payload.get("splits", {}).get(split_name, ()))
        if not allowed_subjects:
            raise CognitiveDatasetError(f"split contains no subjects: {split_name}")
        selected = frame.loc[frame["subject_id"].astype(str).isin(allowed_subjects)].copy()
        if set(selected["derived_split"].astype(str)) != {split_name}:
            raise CognitiveDatasetError("manifest derived_split disagrees with frozen split")
        expected_official_split = "test" if split_name == "test" else "train"
        if selected["official_split"].astype(str).ne(expected_official_split).any():
            if split_name != "test":
                raise CognitiveDatasetError("official Test row reached the training dataset")
            raise CognitiveDatasetError(
                f"official split disagrees with requested {split_name} dataset"
            )
        globally_sorted = frame.sort_values("sample_id", kind="stable").reset_index(drop=True)
        expected_manifest_indices = {
            str(sample_id): index
            for index, sample_id in enumerate(globally_sorted["sample_id"].astype(str))
        }
        selected = selected.sort_values("sample_id", kind="stable").reset_index(drop=True)

        indexes = {
            name: _read_index(self.feature_root / f"{name}_index.jsonl", name)
            for name in MODALITY_ORDER
        }
        self.records: list[dict[str, Any]] = []
        for row in selected.to_dict(orient="records"):
            sample_id = str(row["sample_id"])
            modality_records = {name: indexes[name].get(sample_id) for name in MODALITY_ORDER}
            if any(record is None for record in modality_records.values()):
                raise CognitiveDatasetError(f"feature index row is missing for {sample_id}")
            _validate_index_alignment(row, modality_records)
            manifest_index = int(modality_records["audio"]["manifest_index"])
            if manifest_index != expected_manifest_indices[sample_id]:
                raise CognitiveDatasetError(
                    f"global manifest_index mismatch for {sample_id}: "
                    f"{manifest_index} != {expected_manifest_indices[sample_id]}"
                )
            missing = np.asarray(
                [int(modality_records[name]["missing"]) for name in MODALITY_ORDER],
                dtype=np.bool_,
            )
            if complete_only and bool(missing.any()):
                continue
            features = {
                name: _load_embedding_or_zero(modality_records[name], OUTPUT_DIMS[name])
                for name in MODALITY_ORDER
            }
            quality = np.asarray(
                [float(modality_records[name]["quality"]) for name in MODALITY_ORDER],
                dtype=np.float32,
            )
            if not np.isfinite(quality).all() or np.any((quality < 0.0) | (quality > 1.0)):
                raise CognitiveDatasetError(f"invalid quality vector for {sample_id}")
            diagnosis = str(row["diagnosis_label"])
            moca = _finite_moca(row.get("moca_label"))
            self.records.append(
                {
                    "sample_id": sample_id,
                    "subject_id": str(row["subject_id"]),
                    "manifest_index": manifest_index,
                    "diagnosis_label": diagnosis,
                    "asr_status": str(row.get("asr_status") or "unknown"),
                    "features": features,
                    "quality": quality,
                    "missing_mask": missing,
                    "hc_vs_non_hc": float(diagnosis != "HC"),
                    "mci_vs_hc": float(diagnosis == "MCI") if diagnosis != "AD" else 0.0,
                    "mci_vs_hc_mask": diagnosis != "AD",
                    "ad_mci_hc": {"HC": 0, "MCI": 1, "AD": 2}[diagnosis],
                    "moca": moca if moca is not None else 0.0,
                    "moca_mask": moca is not None,
                }
            )
        if not self.records:
            raise CognitiveDatasetError(f"no usable records for split {split_name}")
        manifest_indices = [int(record["manifest_index"]) for record in self.records]
        if len(manifest_indices) != len(set(manifest_indices)):
            raise CognitiveDatasetError("manifest_index is not unique")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        return {
            "sample_id": record["sample_id"],
            "subject_id": record["subject_id"],
            "manifest_index": record["manifest_index"],
            "diagnosis_label": record["diagnosis_label"],
            "features": {
                name: torch.from_numpy(record["features"][name].copy())
                for name in MODALITY_ORDER
            },
            "quality": torch.from_numpy(record["quality"].copy()),
            "missing_mask": torch.from_numpy(record["missing_mask"].copy()),
            "hc_vs_non_hc": torch.tensor(record["hc_vs_non_hc"], dtype=torch.float32),
            "mci_vs_hc": torch.tensor(record["mci_vs_hc"], dtype=torch.float32),
            "mci_vs_hc_mask": torch.tensor(record["mci_vs_hc_mask"], dtype=torch.bool),
            "ad_mci_hc": torch.tensor(record["ad_mci_hc"], dtype=torch.long),
            "moca": torch.tensor(record["moca"], dtype=torch.float32),
            "moca_mask": torch.tensor(record["moca_mask"], dtype=torch.bool),
        }


def cognitive_collate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot collate an empty batch")
    return {
        "sample_id": [str(row["sample_id"]) for row in rows],
        "subject_id": [str(row["subject_id"]) for row in rows],
        "diagnosis_label": [str(row["diagnosis_label"]) for row in rows],
        "manifest_index": torch.tensor(
            [int(row["manifest_index"]) for row in rows], dtype=torch.long
        ),
        "features": {
            name: torch.stack([row["features"][name] for row in rows])
            for name in MODALITY_ORDER
        },
        "quality": torch.stack([row["quality"] for row in rows]),
        "missing_mask": torch.stack([row["missing_mask"] for row in rows]),
        "hc_vs_non_hc": torch.stack([row["hc_vs_non_hc"] for row in rows]),
        "mci_vs_hc": torch.stack([row["mci_vs_hc"] for row in rows]),
        "mci_vs_hc_mask": torch.stack([row["mci_vs_hc_mask"] for row in rows]),
        "ad_mci_hc": torch.stack([row["ad_mci_hc"] for row in rows]),
        "moca": torch.stack([row["moca"] for row in rows]),
        "moca_mask": torch.stack([row["moca_mask"] for row in rows]),
    }


def compute_moca_statistics(records: Iterable[Mapping[str, Any]]) -> MocaStatistics:
    per_subject: dict[str, list[float]] = {}
    for record in records:
        if not bool(record["moca_mask"]):
            continue
        per_subject.setdefault(str(record["subject_id"]), []).append(float(record["moca"]))
    if not per_subject:
        raise CognitiveDatasetError("Train has no valid MoCA labels")
    subject_means = np.asarray(
        [sum(values) / len(values) for _, values in sorted(per_subject.items())],
        dtype=np.float64,
    )
    mean = float(subject_means.mean())
    std = float(subject_means.std(ddof=0))
    if std < 1e-6:
        std = 1.0
    return MocaStatistics(mean=mean, std=std, subject_count=len(subject_means))


def compute_training_weights(records: Iterable[Mapping[str, Any]]) -> TrainingWeights:
    diagnoses = [str(record["diagnosis_label"]) for record in records]
    if not diagnoses:
        raise CognitiveDatasetError("Train has no labels")
    counts = {label: diagnoses.count(label) for label in ("HC", "MCI", "AD")}
    if any(value <= 0 for value in counts.values()):
        raise CognitiveDatasetError(f"Train is missing a diagnosis class: {counts}")
    main_negative = counts["HC"]
    main_positive = counts["MCI"] + counts["AD"]
    mci_negative = counts["HC"]
    mci_positive = counts["MCI"]
    total = sum(counts.values())
    return TrainingWeights(
        hc_vs_non_hc_pos_weight=main_negative / main_positive,
        mci_vs_hc_pos_weight=mci_negative / mci_positive,
        ad_mci_hc_class_weights=tuple(total / (3.0 * counts[label]) for label in ("HC", "MCI", "AD")),
        task_counts={
            "HC": counts["HC"],
            "MCI": counts["MCI"],
            "AD": counts["AD"],
            "total": total,
        },
    )


def _read_index(path: Path, modality: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        sample_id = str(record.get("sample_id"))
        if sample_id in records:
            raise CognitiveDatasetError(f"duplicate {modality} sample_id: {sample_id}")
        records[sample_id] = record
    return records


def _validate_manifest(frame: pd.DataFrame) -> None:
    required = {
        "sample_id",
        "subject_id",
        "diagnosis_label",
        "moca_label",
        "official_split",
        "derived_split",
    }
    missing = required - set(frame.columns)
    if missing:
        raise CognitiveDatasetError(f"manifest is missing columns: {sorted(missing)}")
    if frame["sample_id"].duplicated().any():
        raise CognitiveDatasetError("manifest sample_id is not unique")
    if not set(frame["diagnosis_label"].astype(str)).issubset({"HC", "MCI", "AD"}):
        raise CognitiveDatasetError("manifest contains an unsupported diagnosis")


def _validate_index_alignment(
    row: Mapping[str, Any], modality_records: Mapping[str, Mapping[str, Any] | None]
) -> None:
    sample_id = str(row["sample_id"])
    expected_subject = str(row["subject_id"])
    expected_split = str(row["derived_split"])
    indices: set[int] = set()
    for name, optional_record in modality_records.items():
        if optional_record is None:
            raise CognitiveDatasetError(f"missing {name} record for {sample_id}")
        record = optional_record
        if str(record.get("subject_id")) != expected_subject:
            raise CognitiveDatasetError(f"{name} subject mismatch for {sample_id}")
        if str(record.get("derived_split")) != expected_split:
            raise CognitiveDatasetError(f"{name} split mismatch for {sample_id}")
        if str(record.get("feature_version")) != FEATURE_VERSION:
            raise CognitiveDatasetError(f"{name} feature version mismatch for {sample_id}")
        if int(record.get("missing", 1)) == 0:
            try:
                output_dim = int(record.get("output_dim", -1))
            except (TypeError, ValueError) as exc:
                raise CognitiveDatasetError(
                    f"{name} output dimension is invalid for {sample_id}"
                ) from exc
            if output_dim != OUTPUT_DIMS[name]:
                raise CognitiveDatasetError(
                    f"{name} output dimension mismatch for {sample_id}"
                )
        if name == "text" and record.get("quality_version") != TEXT_QUALITY_VERSION:
            raise CognitiveDatasetError(f"text quality version mismatch for {sample_id}")
        indices.add(int(record.get("manifest_index", -1)))
    if len(indices) != 1 or next(iter(indices)) < 0:
        raise CognitiveDatasetError(f"manifest_index mismatch for {sample_id}")


def _load_embedding_or_zero(record: Mapping[str, Any], output_dim: int) -> np.ndarray:
    if int(record.get("missing", 1)) == 1:
        return np.zeros(output_dim, dtype=np.float32)
    value = record.get("feature_path")
    if not value:
        raise CognitiveDatasetError(f"available embedding has no path: {record.get('sample_id')}")
    path = resolve_workspace_path(str(value))
    if not path.is_file():
        raise CognitiveDatasetError(f"embedding file is missing: {path}")
    with np.load(path, allow_pickle=False) as payload:
        embedding = np.asarray(payload["embedding"], dtype=np.float32)
    if embedding.shape != (output_dim,):
        raise CognitiveDatasetError(f"invalid embedding shape for {path}: {embedding.shape}")
    if not np.isfinite(embedding).all() or not np.any(embedding != 0.0):
        raise CognitiveDatasetError(f"invalid embedding values for {path}")
    return embedding


def _finite_moca(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or numeric < 0.0 or numeric > 30.0:
        return None
    return numeric


__all__ = [
    "CognitiveDatasetError",
    "CognitiveFeatureDataset",
    "EXPECTED_INPUT_SHA256",
    "MODALITY_ORDER",
    "MocaStatistics",
    "OUTPUT_DIMS",
    "TEXT_QUALITY_VERSION",
    "TrainingWeights",
    "cognitive_collate",
    "compute_moca_statistics",
    "compute_training_weights",
    "default_input_paths",
    "verify_frozen_input_hashes",
]
