"""Independently validate the frozen DATA-007 nested participant splits."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import pandas as pd

from elderly_monitoring.modules.mental_health.mood_social.feature_schema import (
    FEATURE_SCHEMA_VERSION,
    feature_schema_manifest,
)


MANIFEST_VERSION = "mood-social-nested-participant-split-manifest-v1"
ALGORITHM_VERSION = "mood-social-participant-nested-stratified-hash-v1"
SPLIT_ID = "mood-social-v3.3.3-participant-nested-5x5-seed-20260728-v1"
RANDOM_SEED = 20260728
OUTER_FOLD_COUNT = 5
INNER_FOLD_COUNT = 5
FEATURE_SCHEMA_SHA256 = (
    "d2ed18db72601ccb3b0cbb89907bc60ab336bd00f84f0b2c49dcea568bf339e9"
)
PROCESSED_RELATIVE = Path("data/processed/mental_health/mood_social/v3.3.3")
SCHEMA_RELATIVE = PROCESSED_RELATIVE / "manifests" / "feature_schema_manifest.json"
DEFAULT_MANIFEST_RELATIVE = PROCESSED_RELATIVE / "splits" / "split_manifest.json"
EXPECTED_DATASETS: tuple[dict[str, Any], ...] = (
    {
        "dataset_id": "psyche_d",
        "directory_name": "psyche_d",
        "canonical_name": "canonical_psyche_d.parquet",
        "mapping_name": "psyche_d_v3_3_3_mapping.json",
        "artifact_manifest_sha256": (
            "bf4c7e871b48eb85bd5be0cc9a6df4af6d4c638eddea520babcfe454927b9e24"
        ),
        "canonical_sha256": (
            "c913768ea03dd509f1036e85bd66aa7560066023847306b5abddad950cc30b33"
        ),
        "mapping_sha256": (
            "99c3a684c9dd902d0ef2a4f2a198abec3708acbf087029f2d68e26dff7082a74"
        ),
        "frame_sha256": (
            "0ec9d9e42f2a5e068053ae69f4a66d1064819e4e7c914d04f74981cfd822cfa8"
        ),
        "row_count": 10_866,
        "participant_count": 4_036,
        "positive_row_count": 3_444,
    },
    {
        "dataset_id": "resilient",
        "directory_name": "resilient",
        "canonical_name": "canonical_resilient.parquet",
        "mapping_name": "resilient_v3_3_3_mapping.json",
        "artifact_manifest_sha256": (
            "08f6bbdd6e0e09446b8a0e69e20bc90b864600acbcb4aa8cbf25fb37cce27156"
        ),
        "canonical_sha256": (
            "fa2ddd3b79ccd3a7d3cb9073512f9a3813e26b48abd5621cd70cae53d1039229"
        ),
        "mapping_sha256": (
            "8c878c33ee719f600cb08628f023d2a897e030e0eebe4633286999df19519b91"
        ),
        "frame_sha256": (
            "5df4077cdab59f1e3a51c468cca32c1def88fb4c490bfa295eb1d5f5564bca7d"
        ),
        "row_count": 73,
        "participant_count": 73,
        "positive_row_count": 10,
    },
    {
        "dataset_id": "nhanes",
        "directory_name": "nhanes",
        "canonical_name": "canonical_nhanes.parquet",
        "mapping_name": "nhanes_v3_3_3_mapping.json",
        "artifact_manifest_sha256": (
            "477a47965068f27766dbb25d470abbf5db0cb5acd63a339be9a949797907963e"
        ),
        "canonical_sha256": (
            "e6438d99f99b12f3607e1ee6482a48c6cc9235b04028f0184b5b96bd4320654f"
        ),
        "mapping_sha256": (
            "78bafb92230949efa5ea09378f6133965b75cfac73da32f99dd9382dbfcd8802"
        ),
        "frame_sha256": (
            "68cc1d90d3c028dd5c09f749d9d0cea432706c2cd71bd3173c72345e670bee63"
        ),
        "row_count": 2_775,
        "participant_count": 2_775,
        "positive_row_count": 254,
    },
    {
        "dataset_id": "shenzhen_elderly",
        "directory_name": "shenzhen",
        "canonical_name": "canonical_shenzhen.parquet",
        "mapping_name": "shenzhen_v3_3_3_mapping.json",
        "artifact_manifest_sha256": (
            "da809d71393a6993b0c723a2ee27d8b76fb0fe605750e31aa8e356b397122e35"
        ),
        "canonical_sha256": (
            "85ea64ba940e671427a3a9e9ea7a515a90fa57c88520251b3b9964759ebfa054"
        ),
        "mapping_sha256": (
            "9e7c087b2906465abdb0d179838efef42abcb353b617e7261570601a14d7df90"
        ),
        "frame_sha256": (
            "d5cdc2f585feb1e3573c09e092a56246439fa930d245d07191a02d3f42e7e1ac"
        ),
        "row_count": 5_327,
        "participant_count": 5_327,
        "positive_row_count": 186,
    },
    {
        "dataset_id": "nhanes_ssq_2005_2008",
        "directory_name": "nhanes_ssq_2005_2008",
        "canonical_name": "canonical_nhanes_ssq_2005_2008.parquet",
        "mapping_name": "nhanes_ssq_2005_2008_v3_3_3_mapping.json",
        "artifact_manifest_sha256": (
            "e610fea41e25b70b656aadfcbef21a9b8a9445ba419b267e4d7cb10c431e3f9e"
        ),
        "canonical_sha256": (
            "b0da21d9677d3cf59bdec347eedb3a9c34324374003e27c014b1b46f91db52ce"
        ),
        "mapping_sha256": (
            "945bbd504cb4d2578c40b87ec1e039770ce45f1d02154bbc7bc6903cde486aea"
        ),
        "frame_sha256": (
            "0fdb9c743f43d6fa2b5e8397ad1aa54d3f2c6e6e7ab09c4640055ca5923e2610"
        ),
        "row_count": 3_150,
        "participant_count": 3_150,
        "positive_row_count": 185,
    },
)


class ValidationError(RuntimeError):
    """Raised without including participant values in the message."""


@dataclass(frozen=True, slots=True)
class _Unit:
    dataset_id: str
    global_participant_id: str
    row_count: int
    positive_row_count: int

    @property
    def stratum(self) -> tuple[str, int, int]:
        return (
            self.dataset_id,
            self.row_count,
            self.positive_row_count,
        )


def _canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(chunk_size), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValidationError("required file is not readable") from exc
    return digest.hexdigest()


def _sha256_json(payload: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValidationError("required JSON is not readable") from exc


def _check(
    condition: bool,
    name: str,
    checks: list[dict[str, str]],
) -> None:
    if not condition:
        raise ValidationError(f"validation failed: {name}")
    checks.append({"check": name, "status": "pass"})


def _digest_parts(*parts: object) -> bytes:
    values = [str(part) for part in parts]
    if any("\0" in value for value in values):
        raise ValidationError("split hash input contains a NUL character")
    return hashlib.sha256(
        b"\0".join(value.encode("utf-8") for value in values)
    ).digest()


def _assign(
    units: Sequence[_Unit],
    *,
    scope: str,
    fold_count: int,
) -> dict[str, int]:
    strata: dict[tuple[str, int, int], list[_Unit]] = defaultdict(list)
    for unit in units:
        strata[unit.stratum].append(unit)
    assignments: dict[str, int] = {}
    for stratum in sorted(
        strata,
        key=lambda value: (
            value[0].encode("utf-8"),
            value[1],
            value[2],
        ),
    ):
        members = strata[stratum]
        if len(members) < fold_count:
            raise ValidationError("stratification cell is smaller than fold count")
        ordered = sorted(
            members,
            key=lambda unit: (
                _digest_parts(
                    ALGORITHM_VERSION,
                    scope,
                    RANDOM_SEED,
                    "participant",
                    unit.global_participant_id,
                ),
                unit.global_participant_id.encode("utf-8"),
            ),
        )
        offset = (
            int.from_bytes(
                _digest_parts(
                    ALGORITHM_VERSION,
                    scope,
                    RANDOM_SEED,
                    "stratum",
                    *stratum,
                )[:8],
                "big",
            )
            % fold_count
        )
        for rank, unit in enumerate(ordered):
            assignments[unit.global_participant_id] = (offset + rank) % fold_count
    return assignments


def _nested_assignments(units: Sequence[_Unit]) -> list[dict[str, Any]]:
    ordered = sorted(
        units,
        key=lambda value: value.global_participant_id.encode("utf-8"),
    )
    outer = _assign(ordered, scope="outer", fold_count=OUTER_FOLD_COUNT)
    inner: dict[int, dict[str, int]] = {}
    for outer_fold in range(OUTER_FOLD_COUNT):
        outer_train = [
            unit for unit in ordered if outer[unit.global_participant_id] != outer_fold
        ]
        inner[outer_fold] = _assign(
            outer_train,
            scope=f"inner:outer={outer_fold}",
            fold_count=INNER_FOLD_COUNT,
        )
    return [
        {
            "dataset_id": unit.dataset_id,
            "global_participant_id": unit.global_participant_id,
            "inner_validation_fold_by_outer_fold": {
                str(outer_fold): (
                    None
                    if outer_fold == outer[unit.global_participant_id]
                    else inner[outer_fold][unit.global_participant_id]
                )
                for outer_fold in range(OUTER_FOLD_COUNT)
            },
            "outer_fold": outer[unit.global_participant_id],
        }
        for unit in ordered
    ]


def _safe_path(root: Path, relative: str) -> Path:
    candidate_relative = Path(relative)
    if candidate_relative.is_absolute() or ".." in candidate_relative.parts:
        raise ValidationError("artifact path is unsafe")
    candidate = (root / candidate_relative).resolve()
    resolved_root = root.resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise ValidationError("artifact path escapes processed root")
    return candidate


def _load_units_and_binding(
    repository_root: Path,
    processed_root: Path,
    spec: Mapping[str, Any],
    checks: list[dict[str, str]],
) -> tuple[list[_Unit], dict[str, Any]]:
    directory = str(spec["directory_name"])
    canonical_name = str(spec["canonical_name"])
    mapping_name = str(spec["mapping_name"])
    artifact_relative = PROCESSED_RELATIVE / directory / "artifact_manifest.json"
    canonical_relative = PROCESSED_RELATIVE / directory / canonical_name
    metadata_relative = PROCESSED_RELATIVE / directory / "adapter_metadata.json"
    mapping_relative = PROCESSED_RELATIVE / "mappings" / mapping_name
    artifact_path = repository_root / artifact_relative
    canonical_path = repository_root / canonical_relative
    metadata_path = repository_root / metadata_relative
    mapping_path = repository_root / mapping_relative

    dataset_id = str(spec["dataset_id"])
    _check(
        _sha256_file(artifact_path) == spec["artifact_manifest_sha256"],
        f"{dataset_id}_artifact_manifest_hash",
        checks,
    )
    artifact = _load_json(artifact_path)
    _check(
        artifact.get("dataset_id") == dataset_id
        and artifact.get("complete") is True
        and artifact.get("artifact_count") == 4,
        f"{dataset_id}_artifact_manifest_contract",
        checks,
    )
    artifact_files = artifact.get("artifacts")
    _check(
        isinstance(artifact_files, dict) and len(artifact_files) == 4,
        f"{dataset_id}_artifact_file_count",
        checks,
    )
    for relative, expected in artifact_files.items():
        path = _safe_path(processed_root, str(relative))
        _check(
            path.stat().st_size == expected.get("bytes")
            and _sha256_file(path) == expected.get("sha256"),
            f"{dataset_id}_bound_artifact_files",
            checks,
        )
    _check(
        _sha256_file(canonical_path) == spec["canonical_sha256"],
        f"{dataset_id}_canonical_hash",
        checks,
    )
    _check(
        _sha256_file(mapping_path) == spec["mapping_sha256"],
        f"{dataset_id}_mapping_hash",
        checks,
    )
    canonical_key = f"{directory}/{canonical_name}"
    mapping_key = f"mappings/{mapping_name}"
    metadata_key = f"{directory}/adapter_metadata.json"
    _check(
        artifact_files.get(canonical_key, {}).get("sha256") == spec["canonical_sha256"]
        and artifact_files.get(mapping_key, {}).get("sha256") == spec["mapping_sha256"]
        and metadata_key in artifact_files,
        f"{dataset_id}_required_artifact_bindings",
        checks,
    )

    metadata = _load_json(metadata_path)
    _check(
        metadata.get("dataset_id") == dataset_id
        and metadata.get("feature_schema_version") == FEATURE_SCHEMA_VERSION
        and metadata.get("canonical_frame_sha256") == spec["frame_sha256"]
        and metadata.get("canonical_row_count") == spec["row_count"]
        and metadata.get("canonical_participant_count") == spec["participant_count"],
        f"{dataset_id}_metadata_binding",
        checks,
    )
    source_binding = artifact.get("input_binding")
    _check(
        isinstance(source_binding, dict)
        and source_binding.get("feature_schema_sha256") == FEATURE_SCHEMA_SHA256
        and source_binding.get("all_bindings_validated") is True,
        f"{dataset_id}_source_binding",
        checks,
    )

    try:
        frame = pd.read_parquet(
            canonical_path,
            columns=[
                "dataset_id",
                "global_participant_id",
                "participant_id",
                "binary_target",
            ],
        )
    except (OSError, TypeError, ValueError) as exc:
        raise ValidationError("canonical split columns are not readable") from exc
    _check(
        len(frame) == spec["row_count"] and not frame.isna().any().any(),
        f"{dataset_id}_canonical_split_columns",
        checks,
    )
    participant_ids = frame["participant_id"].astype(str)
    global_ids = frame["global_participant_id"].astype(str)
    _check(
        set(frame["dataset_id"].astype(str)) == {dataset_id}
        and global_ids.equals(dataset_id + "::" + participant_ids),
        f"{dataset_id}_participant_namespace",
        checks,
    )
    try:
        targets = pd.to_numeric(frame["binary_target"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValidationError("canonical target is not numeric") from exc
    _check(
        targets.isin([0, 1]).all() and int(targets.sum()) == spec["positive_row_count"],
        f"{dataset_id}_binary_target",
        checks,
    )
    split_frame = pd.DataFrame(
        {
            "dataset_id": dataset_id,
            "global_participant_id": global_ids,
            "binary_target": targets.astype("int64"),
        }
    )
    grouped = (
        split_frame.groupby(
            ["dataset_id", "global_participant_id"],
            sort=True,
            observed=True,
        )["binary_target"]
        .agg(row_count="size", positive_row_count="sum")
        .reset_index()
    )
    _check(
        len(grouped) == spec["participant_count"],
        f"{dataset_id}_participant_count",
        checks,
    )
    units = [
        _Unit(
            dataset_id=str(row.dataset_id),
            global_participant_id=str(row.global_participant_id),
            row_count=int(row.row_count),
            positive_row_count=int(row.positive_row_count),
        )
        for row in grouped.itertuples(index=False)
    ]
    binding = {
        "adapter_metadata_relative_path": metadata_relative.as_posix(),
        "adapter_metadata_sha256": artifact_files[metadata_key]["sha256"],
        "artifact_manifest_relative_path": artifact_relative.as_posix(),
        "artifact_manifest_sha256": spec["artifact_manifest_sha256"],
        "canonical_column_order_version": metadata.get(
            "canonical_column_order_version"
        ),
        "canonical_frame_sha256": spec["frame_sha256"],
        "canonical_relative_path": canonical_relative.as_posix(),
        "canonical_sha256": spec["canonical_sha256"],
        "dataset_id": dataset_id,
        "mapping_relative_path": mapping_relative.as_posix(),
        "mapping_sha256": spec["mapping_sha256"],
        "negative_row_count": (spec["row_count"] - spec["positive_row_count"]),
        "participant_count": spec["participant_count"],
        "positive_row_count": spec["positive_row_count"],
        "row_count": spec["row_count"],
        "source_input_binding": source_binding,
    }
    return units, binding


def _partition_counts(units: Sequence[_Unit]) -> dict[str, Any]:
    by_dataset: dict[str, dict[str, int]] = {}
    for spec in EXPECTED_DATASETS:
        selected = [unit for unit in units if unit.dataset_id == spec["dataset_id"]]
        if not selected:
            continue
        rows = sum(unit.row_count for unit in selected)
        positive = sum(unit.positive_row_count for unit in selected)
        by_dataset[str(spec["dataset_id"])] = {
            "negative_row_count": rows - positive,
            "participant_count": len(selected),
            "positive_row_count": positive,
            "row_count": rows,
        }
    rows = sum(unit.row_count for unit in units)
    positive = sum(unit.positive_row_count for unit in units)
    return {
        "by_dataset": by_dataset,
        "negative_row_count": rows - positive,
        "participant_count": len(units),
        "positive_row_count": positive,
        "row_count": rows,
    }


def _fold_summaries(
    units: Sequence[_Unit],
    assignments: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_key = {str(row["global_participant_id"]): row for row in assignments}
    outer: list[dict[str, Any]] = []
    inner: list[dict[str, Any]] = []
    for outer_fold in range(OUTER_FOLD_COUNT):
        outer_test = [
            unit
            for unit in units
            if by_key[unit.global_participant_id]["outer_fold"] == outer_fold
        ]
        outer_train = [
            unit
            for unit in units
            if by_key[unit.global_participant_id]["outer_fold"] != outer_fold
        ]
        outer.append(
            {
                "outer_fold": outer_fold,
                "test": _partition_counts(outer_test),
                "train": _partition_counts(outer_train),
            }
        )
        inner_folds: list[dict[str, Any]] = []
        for inner_fold in range(INNER_FOLD_COUNT):
            validation = [
                unit
                for unit in outer_train
                if by_key[unit.global_participant_id][
                    "inner_validation_fold_by_outer_fold"
                ][str(outer_fold)]
                == inner_fold
            ]
            validation_keys = {unit.global_participant_id for unit in validation}
            training = [
                unit
                for unit in outer_train
                if unit.global_participant_id not in validation_keys
            ]
            inner_folds.append(
                {
                    "inner_fold": inner_fold,
                    "train": _partition_counts(training),
                    "validation": _partition_counts(validation),
                }
            )
        inner.append(
            {
                "folds": inner_folds,
                "outer_fold": outer_fold,
                "outer_train": _partition_counts(outer_train),
            }
        )
    return outer, inner


def _expected_protocol() -> dict[str, Any]:
    return {
        "assignment_algorithm_version": ALGORITHM_VERSION,
        "balance_guarantee": (
            "participant counts differ by at most one within each "
            "stratification cell and split scope"
        ),
        "fold_indexing": "zero_based",
        "global_participant_key_format": "dataset_id::participant_id",
        "inner_fold_count": INNER_FOLD_COUNT,
        "inner_role": (
            "cross_fitting_validation_assignment_within_each_outer_training_set"
        ),
        "inner_scope_template": "inner:outer={outer_fold}",
        "label_usage": (
            "binary_target is used only for participant-level stratification "
            "and aggregate fold statistics"
        ),
        "outer_fold_count": OUTER_FOLD_COUNT,
        "outer_test_role": "evaluation_only",
        "participant_key_field": "global_participant_id",
        "participant_ordering": (
            "seeded SHA-256 digest with UTF-8 global key tie-break"
        ),
        "preprocessing_fit_scope": {
            "categorical_encoding": "corresponding_training_fold_only",
            "ecdf": "corresponding_training_fold_only",
            "feature_selection": "corresponding_training_fold_only",
            "fusion": "corresponding_training_fold_only",
            "imputation": "corresponding_training_fold_only",
            "probability_calibration": "corresponding_training_fold_only",
            "standardization": "corresponding_training_fold_only",
        },
        "random_seed": RANDOM_SEED,
        "stratification_fields": [
            "dataset_id",
            "participant_row_count",
            "participant_positive_row_count",
        ],
        "stratum_offset": (
            "first eight SHA-256 digest bytes as unsigned big-endian "
            "integer modulo fold count"
        ),
    }


def _validate_partition_relations(
    assignments: Sequence[Mapping[str, Any]],
    checks: list[dict[str, str]],
) -> None:
    all_keys = {str(row["global_participant_id"]) for row in assignments}
    outer_tests: list[set[str]] = []
    for outer_fold in range(OUTER_FOLD_COUNT):
        test = {
            str(row["global_participant_id"])
            for row in assignments
            if row["outer_fold"] == outer_fold
        }
        train = all_keys - test
        outer_tests.append(test)
        inner_validation: list[set[str]] = []
        for inner_fold in range(INNER_FOLD_COUNT):
            validation = {
                str(row["global_participant_id"])
                for row in assignments
                if row["inner_validation_fold_by_outer_fold"][str(outer_fold)]
                == inner_fold
            }
            inner_validation.append(validation)
            _check(
                validation.isdisjoint(test),
                f"outer_{outer_fold}_inner_{inner_fold}_excludes_test",
                checks,
            )
        _check(
            set().union(*inner_validation) == train
            and sum(map(len, inner_validation)) == len(train),
            f"outer_{outer_fold}_inner_coverage_and_disjointness",
            checks,
        )
        _check(
            all(
                row["inner_validation_fold_by_outer_fold"][str(outer_fold)] is None
                for row in assignments
                if row["outer_fold"] == outer_fold
            ),
            f"outer_{outer_fold}_test_has_no_inner_assignment",
            checks,
        )
    _check(
        set().union(*outer_tests) == all_keys
        and sum(map(len, outer_tests)) == len(all_keys),
        "outer_coverage_and_disjointness",
        checks,
    )


def validate(
    repository_root: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    checks: list[dict[str, str]] = []
    repository_root = repository_root.resolve()
    processed_root = repository_root / PROCESSED_RELATIVE
    _check(
        (repository_root / "pyproject.toml").is_file() and processed_root.is_dir(),
        "repository_root",
        checks,
    )
    manifest = _load_json(manifest_path)
    _check(
        manifest.get("manifest_version") == MANIFEST_VERSION
        and manifest.get("status") == "frozen"
        and manifest.get("task_id") == "DATA-007"
        and manifest.get("split_id") == SPLIT_ID
        and manifest.get("frozen_on") == "2026-07-31",
        "manifest_identity",
        checks,
    )

    schema_path = repository_root / SCHEMA_RELATIVE
    live_schema = feature_schema_manifest()
    _check(
        _sha256_file(schema_path) == FEATURE_SCHEMA_SHA256
        and _load_json(schema_path) == live_schema
        and live_schema.get("schema_version") == FEATURE_SCHEMA_VERSION
        and _sha256_json(live_schema) == FEATURE_SCHEMA_SHA256,
        "feature_schema_binding",
        checks,
    )
    expected_schema_binding = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "snapshot_relative_path": SCHEMA_RELATIVE.as_posix(),
        "snapshot_sha256": FEATURE_SCHEMA_SHA256,
    }
    _check(
        manifest.get("feature_schema_binding") == expected_schema_binding,
        "manifest_feature_schema_binding",
        checks,
    )
    protocol = _expected_protocol()
    _check(
        manifest.get("protocol") == protocol,
        "split_protocol",
        checks,
    )

    units: list[_Unit] = []
    input_bindings: list[dict[str, Any]] = []
    for spec in EXPECTED_DATASETS:
        dataset_units, binding = _load_units_and_binding(
            repository_root,
            processed_root,
            spec,
            checks,
        )
        units.extend(dataset_units)
        input_bindings.append(binding)
    unit_keys = [unit.global_participant_id for unit in units]
    _check(
        len(unit_keys) == len(set(unit_keys)) == 15_361,
        "global_participant_uniqueness",
        checks,
    )
    _check(
        manifest.get("inputs") == input_bindings,
        "manifest_input_bindings",
        checks,
    )
    row_count = sum(unit.row_count for unit in units)
    positive = sum(unit.positive_row_count for unit in units)
    totals = {
        "dataset_count": 5,
        "negative_row_count": row_count - positive,
        "participant_count": len(units),
        "positive_row_count": positive,
        "row_count": row_count,
    }
    _check(
        totals
        == {
            "dataset_count": 5,
            "negative_row_count": 18_112,
            "participant_count": 15_361,
            "positive_row_count": 4_079,
            "row_count": 22_191,
        }
        and manifest.get("totals") == totals,
        "manifest_totals",
        checks,
    )

    assignments = manifest.get("participant_assignments")
    _check(
        isinstance(assignments, list) and len(assignments) == len(units),
        "assignment_count",
        checks,
    )
    expected_assignment_keys = {
        "dataset_id",
        "global_participant_id",
        "inner_validation_fold_by_outer_fold",
        "outer_fold",
    }
    _check(
        all(
            isinstance(row, dict) and set(row) == expected_assignment_keys
            for row in assignments
        ),
        "assignment_field_contract",
        checks,
    )
    _check(
        [row["global_participant_id"] for row in assignments]
        == sorted(
            unit_keys,
            key=lambda value: value.encode("utf-8"),
        ),
        "assignment_sort_and_coverage",
        checks,
    )
    expected_assignments = _nested_assignments(units)
    _check(
        assignments == expected_assignments,
        "deterministic_assignment_rebuild",
        checks,
    )
    _validate_partition_relations(assignments, checks)

    outer, inner = _fold_summaries(units, assignments)
    _check(
        manifest.get("outer_folds") == outer,
        "outer_fold_statistics",
        checks,
    )
    _check(
        manifest.get("inner_cross_fitting_folds") == inner,
        "inner_fold_statistics",
        checks,
    )

    integrity = manifest.get("integrity")
    _check(
        isinstance(integrity, dict)
        and set(integrity)
        == {
            "input_bindings_sha256",
            "manifest_core_sha256",
            "participant_assignments_sha256",
            "protocol_sha256",
        },
        "integrity_field_contract",
        checks,
    )
    core = {key: value for key, value in manifest.items() if key != "integrity"}
    _check(
        integrity["input_bindings_sha256"] == _sha256_json(input_bindings)
        and integrity["participant_assignments_sha256"] == _sha256_json(assignments)
        and integrity["protocol_sha256"] == _sha256_json(protocol)
        and integrity["manifest_core_sha256"] == _sha256_json(core),
        "integrity_hashes",
        checks,
    )

    return {
        "check_count": len(checks),
        "inner_fold_count": INNER_FOLD_COUNT,
        "outer_fold_count": OUTER_FOLD_COUNT,
        "participant_count": len(units),
        "row_count": row_count,
        "split_id": SPLIT_ID,
        "split_manifest_sha256": _sha256_file(manifest_path),
        "status": "pass",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("--manifest", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository_root = (
        args.repository_root.resolve()
        if args.repository_root is not None
        else Path(__file__).resolve().parents[1]
    )
    manifest_path = (
        args.manifest
        if args.manifest is not None
        else repository_root / DEFAULT_MANIFEST_RELATIVE
    )
    try:
        result = validate(repository_root, manifest_path)
    except ValidationError as exc:
        print(f"DATA-007 validation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
