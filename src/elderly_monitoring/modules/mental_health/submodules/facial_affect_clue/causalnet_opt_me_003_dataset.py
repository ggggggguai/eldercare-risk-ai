from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


TASK_ID = "OPT-ME-003"
DEVELOPMENT_SPLIT_NAMESPACE = "OPT-ME-003:development-subject-split:v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repository_suffix(path: str) -> str:
    normalized = path.replace("\\", "/")
    marker = "algorithm/eldercare-risk-ai-main/"
    if marker in normalized:
        return normalized.split(marker, 1)[1]
    for prefix in ("data/", "reports/", "configs/", "src/", "scripts/"):
        token = f"/{prefix}"
        if token in normalized:
            return prefix + normalized.split(token, 1)[1]
    raise ValueError(f"cannot derive repository suffix from {path}")


def resolve_artifact_record(
    record: Mapping[str, Any], repo_root: Path, *, source_name: str
) -> dict[str, Any]:
    historical_path = str(record["artifact_path"])
    expected_hash = str(record["artifact_sha256"])
    raw_path = Path(historical_path)
    if raw_path.is_file():
        resolved = raw_path.resolve()
        resolution = "path_exists"
    else:
        resolved = (repo_root.resolve() / _repository_suffix(historical_path)).resolve()
        resolution = "repository_suffix_mapping"
    if not resolved.is_file():
        raise ValueError(f"artifact missing after runtime resolution: {resolved}")
    try:
        resolved.relative_to(repo_root.resolve())
    except ValueError as error:
        raise ValueError(f"resolved artifact escapes repository root: {resolved}") from error
    observed_hash = sha256_file(resolved)
    if observed_hash != expected_hash:
        raise ValueError(
            f"artifact hash mismatch for {record.get('sample_id')}: "
            f"expected {expected_hash}, observed {observed_hash}"
        )
    return {
        "sample_id": str(record["sample_id"]),
        "source": source_name,
        "historical_path": historical_path,
        "resolved_path": resolved.as_posix(),
        "repo_relative_path": resolved.relative_to(repo_root.resolve()).as_posix(),
        "size_bytes": resolved.stat().st_size,
        "sha256": observed_hash,
        "resolution": resolution,
        "historical_record_mutated": False,
    }


def normalize_casme2_subject_id(value: str) -> str:
    normalized = str(value).lower().strip()
    if normalized.startswith("casme2__"):
        normalized = normalized.split("casme2__", 1)[1]
    if not normalized.startswith("sub"):
        raise ValueError(f"unsupported CASME II subject id: {value}")
    return normalized


def _unique_by_sample(records: Sequence[Mapping[str, Any]], source: str) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for record in records:
        sample_id = str(record["sample_id"])
        if sample_id in result:
            raise ValueError(f"duplicate {source} sample_id: {sample_id}")
        result[sample_id] = record
    return result


def align_artifact_records(
    causalnet_records: Sequence[Mapping[str, Any]],
    traditional_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    causalnet = _unique_by_sample(causalnet_records, "causalnet")
    traditional = _unique_by_sample(traditional_records, "traditional")
    if set(causalnet) != set(traditional):
        missing_causalnet = sorted(set(traditional) - set(causalnet))
        missing_traditional = sorted(set(causalnet) - set(traditional))
        raise ValueError(
            "sample coverage mismatch: "
            f"missing_causalnet={missing_causalnet[:5]}, "
            f"missing_traditional={missing_traditional[:5]}"
        )
    aligned: list[dict[str, Any]] = []
    for sample_id in sorted(causalnet):
        causal_row = causalnet[sample_id]
        traditional_row = traditional[sample_id]
        causal_subject = normalize_casme2_subject_id(str(causal_row["subject_id"]))
        traditional_subject = normalize_casme2_subject_id(str(traditional_row["subject_id"]))
        if causal_subject != traditional_subject:
            raise ValueError(f"subject mismatch for {sample_id}")
        causal_label = int(causal_row["label"])
        traditional_label = int(traditional_row["three_class_id"])
        if causal_label != traditional_label:
            raise ValueError(f"label mismatch for {sample_id}")
        if causal_label not in (0, 1, 2):
            raise ValueError(f"invalid label for {sample_id}: {causal_label}")
        aligned.append(
            {
                "sample_id": sample_id,
                "subject_id": causal_subject,
                "label": causal_label,
                "label_name": str(
                    traditional_row.get("three_class_name", causal_row.get("label_name", ""))
                ),
                "causalnet_artifact_path": str(causal_row["artifact_path"]),
                "causalnet_artifact_sha256": str(causal_row["artifact_sha256"]),
                "traditional_artifact_path": str(traditional_row["artifact_path"]),
                "traditional_artifact_sha256": str(traditional_row["artifact_sha256"]),
            }
        )
    return aligned


def development_repeat_seeds(repeat_count: int) -> list[int]:
    if repeat_count <= 0:
        raise ValueError("repeat_count must be positive")
    seeds: list[int] = []
    for repeat_index in range(repeat_count):
        payload = f"{DEVELOPMENT_SPLIT_NAMESPACE}:repeat:{repeat_index}".encode("utf-8")
        seed = int(hashlib.sha256(payload).hexdigest()[:8], 16) & 0x7FFFFFFF
        seeds.append(seed or repeat_index + 1)
    if len(set(seeds)) != len(seeds):
        raise RuntimeError("derived development split seeds are not unique")
    return seeds


def _sample_ids_hash(sample_ids: Sequence[str]) -> str:
    payload = "\n".join(sorted(str(value) for value in sample_ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _label_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(int(row["label"]) for row in rows)
    return {str(label): int(counts.get(label, 0)) for label in (0, 1, 2)}


def build_development_splits(
    rows: Sequence[Mapping[str, Any]], *, repeat_count: int = 3, fold_count: int = 5
) -> dict[str, Any]:
    by_id = _unique_by_sample(rows, "development")
    ordered = [by_id[sample_id] for sample_id in sorted(by_id)]
    if len(ordered) < fold_count:
        raise ValueError("not enough development samples for requested fold count")
    labels = np.asarray([int(row["label"]) for row in ordered], dtype=np.int64)
    groups = np.asarray([str(row["subject_id"]) for row in ordered], dtype=object)
    if set(labels.tolist()) != {0, 1, 2}:
        raise ValueError("development labels must contain exactly classes 0, 1, 2")
    seeds = development_repeat_seeds(repeat_count)
    repeats: list[dict[str, Any]] = []
    for repeat_index, seed in enumerate(seeds):
        splitter = StratifiedGroupKFold(
            n_splits=fold_count,
            shuffle=True,
            random_state=seed,
        )
        folds: list[dict[str, Any]] = []
        for fold_index, (train_index, validation_index) in enumerate(
            splitter.split(np.zeros(len(ordered)), labels, groups)
        ):
            train_rows = [ordered[int(index)] for index in train_index]
            validation_rows = [ordered[int(index)] for index in validation_index]
            train_ids = [str(row["sample_id"]) for row in train_rows]
            validation_ids = [str(row["sample_id"]) for row in validation_rows]
            folds.append(
                {
                    "fold_index": fold_index,
                    "fold_id": f"development_repeat_{repeat_index:02d}_fold_{fold_index:02d}",
                    "train_sample_ids": sorted(train_ids),
                    "validation_sample_ids": sorted(validation_ids),
                    "train_subject_ids": sorted({str(row["subject_id"]) for row in train_rows}),
                    "validation_subject_ids": sorted(
                        {str(row["subject_id"]) for row in validation_rows}
                    ),
                    "train_label_counts": _label_counts(train_rows),
                    "validation_label_counts": _label_counts(validation_rows),
                    "train_sample_ids_sha256": _sample_ids_hash(train_ids),
                    "validation_sample_ids_sha256": _sample_ids_hash(validation_ids),
                    "fit_statistics_scope": "train_sample_ids_only",
                }
            )
        repeats.append(
            {
                "repeat_index": repeat_index,
                "seed": seed,
                "roles": ["rescreen"] + (["screening"] if repeat_index < 2 else []),
                "folds": folds,
            }
        )
    return {
        "schema_version": "opt_me_003_development_subject_splits_v1",
        "task_id": TASK_ID,
        "dataset_scope": "CASME II exposed development benchmark",
        "splitter": "StratifiedGroupKFold",
        "group_field": "subject_id",
        "label_field": "label",
        "fold_count": fold_count,
        "repeat_count": repeat_count,
        "screening_repeat_indices": [0, 1],
        "rescreen_repeat_indices": list(range(repeat_count)),
        "seed_derivation": DEVELOPMENT_SPLIT_NAMESPACE,
        "test_used_for_selection": False,
        "repeats": repeats,
    }


def audit_development_splits(
    rows: Sequence[Mapping[str, Any]],
    payload: Mapping[str, Any],
    *,
    expected_repeats: int = 3,
    expected_folds: int = 5,
) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    by_id = _unique_by_sample(rows, "development")
    expected_ids = set(by_id)
    repeats = list(payload.get("repeats", []))
    if len(repeats) != expected_repeats:
        errors.append(
            {"code": "repeat_count", "expected": expected_repeats, "observed": len(repeats)}
        )
    coverage_per_repeat: list[int] = []
    for repeat in repeats:
        folds = list(repeat.get("folds", []))
        if len(folds) != expected_folds:
            errors.append(
                {
                    "code": "fold_count",
                    "repeat_index": repeat.get("repeat_index"),
                    "expected": expected_folds,
                    "observed": len(folds),
                }
            )
        validation_seen: list[str] = []
        for fold in folds:
            train_ids = [str(value) for value in fold.get("train_sample_ids", [])]
            validation_ids = [str(value) for value in fold.get("validation_sample_ids", [])]
            unknown = (set(train_ids) | set(validation_ids)) - expected_ids
            if unknown:
                errors.append({"code": "unknown_sample", "fold_id": fold.get("fold_id")})
            if set(train_ids) & set(validation_ids):
                errors.append({"code": "sample_leakage", "fold_id": fold.get("fold_id")})
            train_subjects = {str(by_id[value]["subject_id"]) for value in train_ids}
            validation_subjects = {str(by_id[value]["subject_id"]) for value in validation_ids}
            if train_subjects & validation_subjects:
                errors.append({"code": "subject_leakage", "fold_id": fold.get("fold_id")})
            if set(train_ids) | set(validation_ids) != expected_ids:
                errors.append({"code": "partition_coverage", "fold_id": fold.get("fold_id")})
            labels = {int(by_id[value]["label"]) for value in validation_ids}
            if labels != {0, 1, 2}:
                errors.append(
                    {
                        "code": "validation_class_coverage",
                        "fold_id": fold.get("fold_id"),
                        "labels": sorted(labels),
                    }
                )
            validation_seen.extend(validation_ids)
        if len(validation_seen) != len(set(validation_seen)):
            errors.append(
                {"code": "duplicate_validation_sample", "repeat_index": repeat.get("repeat_index")}
            )
        if set(validation_seen) != expected_ids:
            errors.append(
                {"code": "repeat_validation_coverage", "repeat_index": repeat.get("repeat_index")}
            )
        coverage_per_repeat.append(len(validation_seen))
    return {
        "schema_version": "opt_me_003_development_split_audit_v1",
        "status": "passed" if not errors else "failed",
        "error_count": len(errors),
        "errors": errors,
        "sample_count": len(expected_ids),
        "subject_count": len({str(row["subject_id"]) for row in rows}),
        "validation_coverage_per_repeat": coverage_per_repeat,
        "test_used_for_selection": False,
    }

