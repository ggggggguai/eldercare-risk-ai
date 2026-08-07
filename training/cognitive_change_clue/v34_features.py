"""Leakage-safe feature variants for OPT-COG-003."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from .cache_v34_lightweight import DEFAULT_OUTPUT_ROOT as LIGHTWEIGHT_ROOT
    from .cache_v34_pooling import DEFAULT_OUTPUT_ROOT as POOLING_ROOT
    from .common import DEFAULT_PROCESSED_ROOT, resolve_workspace_path, sha256_file
    from .dataset import MODALITY_ORDER, OUTPUT_DIMS
except ImportError:
    from cache_v34_lightweight import DEFAULT_OUTPUT_ROOT as LIGHTWEIGHT_ROOT  # type: ignore[no-redef]
    from cache_v34_pooling import DEFAULT_OUTPUT_ROOT as POOLING_ROOT  # type: ignore[no-redef]
    from common import DEFAULT_PROCESSED_ROOT, resolve_workspace_path, sha256_file  # type: ignore[no-redef]
    from dataset import MODALITY_ORDER, OUTPUT_DIMS  # type: ignore[no-redef]


VARIANTS = {
    "baseline": {"operation": "identity"},
    "audio_basic": {
        "operation": "append",
        "modality": "audio",
        "group": "audio_basic",
        "dimension": 9,
    },
    "egemaps": {
        "operation": "append",
        "modality": "audio",
        "group": "egemaps",
        "dimension": 88,
    },
    "text_stats": {
        "operation": "append",
        "modality": "text",
        "group": "text_stats",
        "dimension": 12,
    },
    "text_continuous_quality": {
        "operation": "restore_existing_text",
        "modality": "text",
    },
    "wavlm_mean_std": {
        "operation": "replace",
        "modality": "audio",
        "group": "audio_stats",
        "dimension": 1536,
    },
    "roberta_mean": {
        "operation": "replace_preserve_mask",
        "modality": "text",
        "group": "text_mean",
        "dimension": 768,
    },
}


class V34FeatureVariantError(RuntimeError):
    pass


@dataclass(frozen=True)
class FoldFeatureView:
    records: list[dict[str, Any]]
    input_dims: dict[str, int]
    audit: dict[str, Any]


def _read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sample_id = str(row["sample_id"])
        if sample_id in rows:
            raise V34FeatureVariantError(f"duplicate feature sample: {sample_id}")
        rows[sample_id] = row
    return rows


def _load_npz_vector(path: Path, key: str, dimension: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        value = np.asarray(payload[key], dtype=np.float32)
    if value.shape != (dimension,) or not np.isfinite(value).all():
        raise V34FeatureVariantError(f"invalid {key} vector: {path}")
    return value


def fit_standardizer(
    values: Sequence[np.ndarray], *, dimension: int
) -> tuple[np.ndarray, np.ndarray]:
    if not values:
        raise V34FeatureVariantError("cannot fit a feature standardizer without rows")
    matrix = np.stack([np.asarray(value, dtype=np.float64) for value in values])
    if matrix.ndim != 2 or matrix.shape[1] != dimension or not np.isfinite(matrix).all():
        raise V34FeatureVariantError("standardizer input is invalid")
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0, ddof=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def _lightweight_values(
    group: str,
    dimension: int,
) -> tuple[dict[str, np.ndarray | None], dict[str, Any]]:
    manifest_path = LIGHTWEIGHT_ROOT / "cache_manifest.json"
    index_path = LIGHTWEIGHT_ROOT / "lightweight_index.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "complete"
        or manifest.get("official_test_media_read") is not False
        or int(manifest.get("indexed_rows", -1)) != 1377
    ):
        raise V34FeatureVariantError("lightweight feature cache is not complete and isolated")
    rows = _read_jsonl(index_path)
    values: dict[str, np.ndarray | None] = {}
    for sample_id, row in rows.items():
        if int(row["dimensions"][group]) != dimension:
            raise V34FeatureVariantError(f"{group} dimension mismatch")
        if bool(row["missing"][group]):
            values[sample_id] = None
            continue
        path = resolve_workspace_path(str(row["feature_path"]))
        if sha256_file(path) != str(row["feature_sha256"]):
            raise V34FeatureVariantError(f"lightweight feature hash mismatch: {sample_id}")
        values[sample_id] = _load_npz_vector(path, group, dimension)
    return values, {
        "cache_manifest": str(manifest_path),
        "cache_manifest_sha256": sha256_file(manifest_path),
        "cache_index": str(index_path),
        "cache_index_sha256": sha256_file(index_path),
    }


def _pooling_values(
    group: str,
    dimension: int,
) -> tuple[dict[str, np.ndarray | None], dict[str, Any]]:
    manifest_path = POOLING_ROOT / "cache_manifest.json"
    index_path = POOLING_ROOT / f"{group}_index.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "complete"
        or manifest.get("official_test_media_read") is not False
    ):
        raise V34FeatureVariantError("pooling feature cache is not complete and isolated")
    rows = _read_jsonl(index_path)
    values: dict[str, np.ndarray | None] = {}
    for sample_id, row in rows.items():
        if int(row.get("missing", 1)) == 1:
            values[sample_id] = None
            continue
        if int(row["output_dim"]) != dimension:
            raise V34FeatureVariantError(f"{group} dimension mismatch")
        path = resolve_workspace_path(str(row["feature_path"]))
        if sha256_file(path) != str(row["feature_sha256"]):
            raise V34FeatureVariantError(f"pooling feature hash mismatch: {sample_id}")
        values[sample_id] = _load_npz_vector(path, "embedding", dimension)
    return values, {
        "cache_manifest": str(manifest_path),
        "cache_manifest_sha256": sha256_file(manifest_path),
        "cache_index": str(index_path),
        "cache_index_sha256": sha256_file(index_path),
    }


def _restore_text_values() -> tuple[dict[str, np.ndarray | None], dict[str, Any]]:
    index_path = DEFAULT_PROCESSED_ROOT / "features" / "text_index.jsonl"
    rows = _read_jsonl(index_path)
    values: dict[str, np.ndarray | None] = {}
    for sample_id, row in rows.items():
        value = row.get("feature_path")
        if not value:
            values[sample_id] = None
            continue
        path = resolve_workspace_path(str(value))
        if sha256_file(path) != str(row["feature_sha256"]):
            raise V34FeatureVariantError(f"V3.3 text feature hash mismatch: {sample_id}")
        values[sample_id] = _load_npz_vector(path, "embedding", 768)
    return values, {
        "cache_index": str(index_path),
        "cache_index_sha256": sha256_file(index_path),
    }


def build_fold_feature_view(
    records: Sequence[Mapping[str, Any]],
    *,
    inner_train_subjects: Sequence[str],
    variant: str,
) -> FoldFeatureView:
    if variant not in VARIANTS:
        raise V34FeatureVariantError(f"unsupported feature variant: {variant}")
    spec = VARIANTS[variant]
    input_dims = dict(OUTPUT_DIMS)
    if spec["operation"] == "identity":
        return FoldFeatureView(
            records=[copy.deepcopy(dict(record)) for record in records],
            input_dims=input_dims,
            audit={"variant": variant, "operation": "identity", "normalization": None},
        )
    modality = str(spec["modality"])
    modality_index = MODALITY_ORDER.index(modality)
    operation = str(spec["operation"])
    dimension = int(spec.get("dimension", OUTPUT_DIMS[modality]))
    if operation == "append":
        values, cache_audit = _lightweight_values(str(spec["group"]), dimension)
    elif operation in {"replace", "replace_preserve_mask"}:
        values, cache_audit = _pooling_values(str(spec["group"]), dimension)
    elif operation == "restore_existing_text":
        values, cache_audit = _restore_text_values()
    else:
        raise V34FeatureVariantError(f"unsupported feature operation: {operation}")

    train_subjects = set(str(value) for value in inner_train_subjects)
    normalizer_mean: np.ndarray | None = None
    normalizer_std: np.ndarray | None = None
    if operation == "append":
        fitting = [
            values[str(record["sample_id"])]
            for record in records
            if str(record["subject_id"]) in train_subjects
            and not bool(record["missing_mask"][modality_index])
            and values.get(str(record["sample_id"])) is not None
        ]
        normalizer_mean, normalizer_std = fit_standardizer(
            [value for value in fitting if value is not None], dimension=dimension
        )
        input_dims[modality] += dimension
    elif operation == "replace":
        input_dims[modality] = dimension
    elif operation == "replace_preserve_mask":
        input_dims[modality] = dimension

    transformed: list[dict[str, Any]] = []
    recovered = 0
    unavailable = 0
    for source in records:
        row = copy.deepcopy(dict(source))
        row["features"] = {
            name: np.asarray(source["features"][name], dtype=np.float32).copy()
            for name in MODALITY_ORDER
        }
        row["missing_mask"] = np.asarray(source["missing_mask"], dtype=np.bool_).copy()
        value = values.get(str(row["sample_id"]))
        if operation == "append":
            if value is None:
                standardized = np.zeros(dimension, dtype=np.float32)
                unavailable += 1
            else:
                standardized = (value - normalizer_mean) / normalizer_std
            row["features"][modality] = np.concatenate(
                (row["features"][modality], standardized.astype(np.float32, copy=False))
            )
        elif operation == "replace":
            if value is None:
                row["features"][modality] = np.zeros(dimension, dtype=np.float32)
                row["missing_mask"][modality_index] = True
                unavailable += 1
            else:
                row["features"][modality] = value.copy()
        elif operation == "replace_preserve_mask":
            row["features"][modality] = (
                np.zeros(dimension, dtype=np.float32) if value is None else value.copy()
            )
            if value is None:
                row["missing_mask"][modality_index] = True
                unavailable += 1
        elif operation == "restore_existing_text":
            if value is not None and bool(row["missing_mask"][modality_index]):
                row["features"][modality] = value.copy()
                row["missing_mask"][modality_index] = False
                recovered += 1
        expected = input_dims[modality]
        if row["features"][modality].shape != (expected,):
            raise V34FeatureVariantError(
                f"{variant} produced invalid {modality} shape for {row['sample_id']}"
            )
        transformed.append(row)
    audit = {
        "schema_version": "cognitive_v34_fold_feature_view_v1",
        "variant": variant,
        "operation": operation,
        "modality": modality,
        "input_dims": input_dims,
        "inner_train_subject_count": len(train_subjects),
        "normalization_scope": "inner_train_only" if operation == "append" else None,
        "normalization_mean": normalizer_mean.tolist() if normalizer_mean is not None else None,
        "normalization_std": normalizer_std.tolist() if normalizer_std is not None else None,
        "recovered_text_task_count": recovered,
        "unavailable_feature_task_count": unavailable,
        **cache_audit,
    }
    return FoldFeatureView(records=transformed, input_dims=input_dims, audit=audit)


__all__ = [
    "FoldFeatureView",
    "VARIANTS",
    "V34FeatureVariantError",
    "build_fold_feature_view",
    "fit_standardizer",
]
