"""Build the DATA-001 read-only audit for the five V3.3.3 datasets.

The generated artefacts contain file metadata, field names, aggregate counts,
and hashes only. Participant-level values are inspected in memory solely for
key-quality checks and are never written to disk or stdout.

Run from the algorithm repository:

    conda run -n eldercare-ai python \
        scripts/audit/audit_mood_social_v3_datasets.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import numbers
import os
import tempfile
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd
import yaml

from elderly_monitoring.modules.mental_health.mood_social import (
    feature_schema_manifest,
)


AUDIT_VERSION = "mood-social-dataset-audit-v3.3.3-data001"
AUDIT_RUN_ID = "DATA-001-2026-07-30"
HASH_SPEC_VERSION = "mood-social-collection-sha256-v1"
MANIFEST_VERSION = "mood-social-file-manifest-v1"
VALIDATION_VERSION = "mood-social-dataset-audit-validation-v1"
DEFAULT_OUTPUT_RELATIVE = Path(
    "data/processed/mental_health/mood_social/v3.3.3/manifests"
)
FILE_MANIFEST_NAME = "dataset_file_manifest.jsonl"
SUMMARY_NAME = "dataset_audit_summary.json"
VALIDATION_NAME = "validation_checks.json"
FEATURE_SCHEMA_NAME = "feature_schema_manifest.json"

RESILIENT_SENSOR_FILES = frozenset(
    {
        "ScanWatch_HR.csv",
        "ScanWatch_Steps.csv",
        "Sleep_physio.csv",
        "Sleep_state.csv",
    }
)
NHANES_PHQ_FIELDS = tuple(f"DPQ0{index}0" for index in range(1, 10))
HEFEI_PHQ_FIELD_CANDIDATES = tuple(f"h{index}" for index in range(1, 10))


class AuditError(RuntimeError):
    """A source or contract error safe to report without source values."""


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    display_name: str
    source_relative: str
    participant_key_candidate: str
    label_fields: tuple[str, ...]
    time_granularity: str
    adapter_entrypoint: str
    adapter_task: str
    canonical_output: str
    adapter_boundary: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "display_name": self.display_name,
            "source_relative": self.source_relative,
            "participant_key_candidate": self.participant_key_candidate,
            "label_fields": list(self.label_fields),
            "time_granularity": self.time_granularity,
            "adapter_entrypoint": self.adapter_entrypoint,
            "adapter_task": self.adapter_task,
            "canonical_output": self.canonical_output,
            "adapter_boundary": self.adapter_boundary,
            "adapter_implemented_by_data001": False,
        }


DATASET_SPECS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        dataset_id="psyche_d",
        display_name="PSYCHE-D",
        source_relative="数据集/心理/PSYCHE-D",
        participant_key_candidate=(
            "Parquet sample_id index prefix before the final underscore"
        ),
        label_fields=(
            "phq9_score_start",
            "phq9_score_end",
            "phq9_cat_start",
            "phq9_cat_end",
        ),
        time_granularity=(
            "participant nominal-month rows; publisher-defined wearable "
            "aggregates near PHQ-9, not reconstructable natural days"
        ),
        adapter_entrypoint=(
            "elderly_monitoring.modules.mental_health.validation."
            "public_datasets.psyche_d"
        ),
        adapter_task="DATA-002",
        canonical_output="canonical_psyche_d.parquet",
        adapter_boundary=(
            "Read the Parquet matrix and field dictionary; derive participant_id "
            "from sample_id; do not treat the legacy P0 mapping as the V3.3.3 "
            "schema; fit steps_awake_mean ECDF inside each outer training fold."
        ),
    ),
    DatasetSpec(
        dataset_id="resilient",
        display_name="RESILIENT",
        source_relative="数据集/心理/RESILIENT",
        participant_key_candidate="Demographics.user_id and sensor directory key",
        label_fields=("phq_total", "gds_total", "gad_total"),
        time_granularity=(
            "participant assessment rows plus timestamped sensor records; "
            "future adapter uses the first 14 calendar days from first valid sensor day"
        ),
        adapter_entrypoint=(
            "elderly_monitoring.modules.mental_health.validation."
            "public_datasets.resilient"
        ),
        adapter_task="DATA-003",
        canonical_output="canonical_resilient.parquet",
        adapter_boundary=(
            "Read Demographics and four real sensor CSV types; exclude AppleDouble "
            "and .DS_Store from tabular parsing while retaining them in the raw "
            "collection manifest and hash."
        ),
    ),
    DatasetSpec(
        dataset_id="nhanes",
        display_name="NHANES",
        source_relative="数据集/心理/NHANES",
        participant_key_candidate="cycle plus SEQN",
        label_fields=NHANES_PHQ_FIELDS,
        time_granularity=(
            "survey-cycle participant tables plus day-level PAM records; "
            "future adapter uses at most seven valid PAM days after DPQ"
        ),
        adapter_entrypoint=(
            "elderly_monitoring.modules.mental_health.validation.public_datasets.nhanes"
        ),
        adapter_task="DATA-004",
        canonical_output="canonical_nhanes.parquet",
        adapter_boundary=(
            "Namespace SEQN by G/H cycle, join DPQ/DEMO/PAM without claiming "
            "pre-assessment prediction, and compute PHQ-9 only after DPQ missing "
            "codes are handled by DATA-004."
        ),
    ),
    DatasetSpec(
        dataset_id="shenzhen_elderly",
        display_name="深圳社区老人",
        source_relative="数据集/心理/DRYAD深证社区老年心理健康",
        participant_key_candidate="code",
        label_fields=("PHQ9score", "Depressivesymptoms"),
        time_granularity="cross-sectional, one row per participant",
        adapter_entrypoint=(
            "elderly_monitoring.modules.mental_health.validation."
            "public_datasets.shenzhen_elderly"
        ),
        adapter_task="DATA-005",
        canonical_output="canonical_shenzhen.parquet",
        adapter_boundary=(
            "Use PHQ9score for the V3.3.3 >=10 target; keep published scale "
            "outcomes and neighbouring mental-health scales out of production inputs."
        ),
    ),
    DatasetSpec(
        dataset_id="hefei_elderly",
        display_name="合肥社区老人",
        source_relative="数据集/心理/合肥社区老年衰弱数据",
        participant_key_candidate=(
            "publisher field a1 pending duplicate resolution; deterministic row key "
            "is allowed only if DATA-006 confirms no stable publisher key"
        ),
        label_fields=HEFEI_PHQ_FIELD_CANDIDATES,
        time_granularity="cross-sectional, one worksheet row per record",
        adapter_entrypoint=(
            "elderly_monitoring.modules.mental_health.validation."
            "public_datasets.hefei_elderly"
        ),
        adapter_task="DATA-006",
        canonical_output="canonical_hefei.parquet",
        adapter_boundary=(
            "Read the legacy XLS without overwriting it; DATA-006 must resolve the "
            "a1 duplicate and confirm coded PHQ-9/social-frailty fields from a "
            "publisher codebook before canonicalization."
        ),
    ),
)


@dataclass(frozen=True)
class AuditPaths:
    workspace_root: Path
    algorithm_root: Path
    mental_root: Path
    output_root: Path


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


def _encode_jsonl(rows: Sequence[dict[str, Any]]) -> bytes:
    return b"".join(
        (
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        for row in rows
    )


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _collection_sha256(rows: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    ordered = sorted(
        rows,
        key=lambda row: str(
            row.get("_collection_relative_path", row["relative_path"])
        ).encode("utf-8"),
    )
    for row in ordered:
        collection_path = str(
            row.get("_collection_relative_path", row["relative_path"])
        )
        digest.update(
            (f"{collection_path}\0{row['bytes']}\0{row['sha256']}\n").encode("utf-8")
        )
    return digest.hexdigest()


def _find_workspace_root(start: Path) -> Path:
    resolved = start.resolve()
    for candidate in (resolved, *resolved.parents):
        if (candidate / "数据集" / "心理").is_dir() and (
            candidate / "algorithm" / "eldercare-risk-ai-main"
        ).is_dir():
            return candidate
    raise AuditError("workspace root containing 数据集/心理 was not found")


def resolve_paths(
    workspace_root: Path | None = None,
    output_root: Path | None = None,
) -> AuditPaths:
    workspace = _find_workspace_root(workspace_root or Path.cwd())
    algorithm = (workspace / "algorithm" / "eldercare-risk-ai-main").resolve()
    # Keep the workspace-visible path: 数据集 may be an intentional directory
    # junction to a separate data volume on Windows.
    mental = workspace / "数据集" / "心理"
    output = (
        output_root.resolve()
        if output_root is not None
        else (algorithm / DEFAULT_OUTPUT_RELATIVE).resolve()
    )
    if not output.is_relative_to(algorithm):
        raise AuditError("audit output must stay inside the algorithm repository")
    if any(
        output.is_relative_to((workspace / spec.source_relative).resolve())
        for spec in DATASET_SPECS
    ):
        raise AuditError("audit output must not be inside a source dataset")
    return AuditPaths(workspace, algorithm, mental, output)


def _path_is_reparse_point(path: Path) -> bool:
    attributes = getattr(path.stat(follow_symlinks=False), "st_file_attributes", 0)
    return bool(attributes & 0x400)


def _classify_file(path: Path, dataset_root: Path) -> tuple[str, str]:
    relative_parts = path.relative_to(dataset_root).parts
    if path.name == ".DS_Store":
        return "os_metadata", "ds_store"
    if "__MACOSX" in relative_parts:
        return "os_metadata", "appledouble"
    suffix = path.suffix.lower()
    if suffix == ".csv":
        role = (
            "source_metadata"
            if "Resilient_metadata" in relative_parts
            else "source_data"
        )
        return role, "csv"
    if suffix == ".tsv":
        return "source_metadata", "tsv"
    if suffix == ".xpt":
        return "source_data", "sas_xport"
    if suffix == ".xls":
        return "source_data", "xls"
    if suffix in {".yaml", ".yml"}:
        return "source_metadata", "yaml"
    if suffix == ".md":
        return "source_documentation", "markdown"
    with path.open("rb") as handle:
        prefix = handle.read(4)
        if path.stat().st_size >= 4:
            handle.seek(-4, os.SEEK_END)
            suffix_magic = handle.read(4)
        else:
            suffix_magic = b""
    if prefix == b"PAR1" and suffix_magic == b"PAR1":
        return "source_data", "parquet"
    raise AuditError(f"unsupported or unrecognised file format: {path.name}")


def _resilient_participant_aliases(source_root: Path) -> dict[str, str]:
    names = sorted(
        {
            path.parent.name
            for path in source_root.rglob("*.csv")
            if path.name in RESILIENT_SENSOR_FILES
            and "__MACOSX" not in path.relative_to(source_root).parts
        },
        key=lambda value: value.encode("utf-8"),
    )
    if len(names) != len(set(names)) or not names:
        raise AuditError("RESILIENT participant directory aliases cannot be built")
    return {name: f"participant_{index:04d}" for index, name in enumerate(names, 1)}


def _public_relative_path(
    spec: DatasetSpec,
    file_path: Path,
    source_root: Path,
    participant_aliases: dict[str, str],
) -> str:
    dataset_parts = list(file_path.relative_to(source_root).parts)
    if spec.dataset_id == "resilient":
        for index, part in enumerate(dataset_parts):
            prefix = "._" if part.startswith("._") else ""
            candidate = part[2:] if prefix else part
            alias = participant_aliases.get(candidate)
            if alias is not None:
                dataset_parts[index] = f"{prefix}{alias}"
    return (Path(spec.source_relative) / Path(*dataset_parts)).as_posix()


def build_file_manifest(paths: AuditPaths) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    normalized_paths: dict[str, str] = {}
    for spec in DATASET_SPECS:
        source_root = paths.workspace_root / spec.source_relative
        if not source_root.is_dir():
            raise AuditError(f"missing dataset directory for {spec.dataset_id}")
        participant_aliases = (
            _resilient_participant_aliases(source_root)
            if spec.dataset_id == "resilient"
            else {}
        )
        files = sorted(
            (path for path in source_root.rglob("*") if path.is_file()),
            key=lambda path: path.relative_to(paths.workspace_root)
            .as_posix()
            .encode("utf-8"),
        )
        if not files:
            raise AuditError(f"dataset directory is empty: {spec.dataset_id}")
        for file_path in files:
            if file_path.is_symlink() or _path_is_reparse_point(file_path):
                raise AuditError(
                    f"links and reparse points are not allowed: {spec.dataset_id}"
                )
            collection_relative = file_path.relative_to(paths.workspace_root).as_posix()
            public_relative = _public_relative_path(
                spec, file_path, source_root, participant_aliases
            )
            normalized = unicodedata.normalize("NFKC", collection_relative).casefold()
            prior = normalized_paths.get(normalized)
            if prior is not None and prior != collection_relative:
                raise AuditError("normalised relative-path collision detected")
            normalized_paths[normalized] = collection_relative
            before = file_path.stat()
            file_sha256 = _sha256_file(file_path)
            after = file_path.stat()
            if (
                before.st_size != after.st_size
                or before.st_mtime_ns != after.st_mtime_ns
            ):
                raise AuditError(f"source file changed during audit: {spec.dataset_id}")
            role, file_format = _classify_file(file_path, source_root)
            rows.append(
                {
                    "_collection_relative_path": collection_relative,
                    "_source_path": file_path,
                    "bytes": int(after.st_size),
                    "dataset_id": spec.dataset_id,
                    "file_role": role,
                    "format": file_format,
                    "relative_path": public_relative,
                    "sha256": file_sha256,
                }
            )
    return sorted(rows, key=lambda row: row["relative_path"].encode("utf-8"))


def _normalise_key(value: Any) -> tuple[str | None, str | None]:
    if value is None:
        return None, None
    try:
        if bool(pd.isna(value)):
            return None, None
    except (TypeError, ValueError):
        pass
    if isinstance(value, numbers.Integral):
        raw = str(int(value))
    elif isinstance(value, numbers.Real) and math.isfinite(float(value)):
        numeric = float(value)
        raw = str(int(numeric)) if numeric.is_integer() else str(numeric)
    else:
        raw = str(value).strip()
    if not raw:
        return None, None
    return unicodedata.normalize("NFKC", raw).casefold(), raw


def _key_profile(values: Iterable[Any]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    raw_forms: dict[str, set[str]] = defaultdict(set)
    missing = 0
    for value in values:
        normalised, raw = _normalise_key(value)
        if normalised is None or raw is None:
            missing += 1
            continue
        counts[normalised] += 1
        raw_forms[normalised].add(raw)
    return {
        "record_count": int(sum(counts.values()) + missing),
        "missing_key_count": missing,
        "unique_key_count": len(counts),
        "duplicate_record_count": int(sum(count - 1 for count in counts.values())),
        "duplicate_key_group_count": int(sum(count > 1 for count in counts.values())),
        "normalisation_collision_group_count": int(
            sum(len(forms) > 1 for forms in raw_forms.values())
        ),
    }


def _normalised_key_set(values: Iterable[Any]) -> set[str]:
    result: set[str] = set()
    for value in values:
        normalised, _ = _normalise_key(value)
        if normalised is not None:
            result.add(normalised)
    return result


def _read_delimited_structure(path: Path, delimiter: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle, delimiter=delimiter, strict=True)
            fields = next(reader)
            if not fields:
                raise AuditError(f"empty delimited header: {path.name}")
            row_count = 0
            inconsistent_rows = 0
            for row in reader:
                row_count += 1
                inconsistent_rows += len(row) != len(fields)
    except AuditError:
        raise
    except Exception as exc:
        raise AuditError(
            f"delimited file is not readable: {path.name} ({type(exc).__name__})"
        ) from None
    if inconsistent_rows:
        raise AuditError(f"inconsistent delimited row width: {path.name}")
    return {
        "column_count": len(fields),
        "duplicate_field_count": len(fields) - len(set(fields)),
        "fields": [str(field) for field in fields],
        "row_count": row_count,
    }


def _dtype_counts(frame: pd.DataFrame) -> dict[str, int]:
    return dict(sorted(Counter(str(dtype) for dtype in frame.dtypes).items()))


def _validate_support_files(
    paths: AuditPaths, manifest: Sequence[dict[str, Any]]
) -> None:
    for row in manifest:
        file_path = row["_source_path"]
        file_format = row["format"]
        try:
            if file_format == "markdown":
                file_path.read_text(encoding="utf-8")
            elif file_format == "yaml":
                with file_path.open("r", encoding="utf-8") as handle:
                    yaml.safe_load(handle)
            elif file_format == "appledouble":
                with file_path.open("rb") as handle:
                    if handle.read(4) != b"\x00\x05\x16\x07":
                        raise AuditError("invalid AppleDouble metadata header")
            elif file_format == "ds_store":
                with file_path.open("rb") as handle:
                    if not handle.read(4):
                        raise AuditError("empty .DS_Store metadata file")
        except AuditError:
            raise
        except Exception as exc:
            raise AuditError(
                f"support file is not readable: {file_path.name} ({type(exc).__name__})"
            ) from None


def _relative_to_dataset(path: Path, dataset_root: Path) -> str:
    return path.relative_to(dataset_root).as_posix()


def _audit_psyche(
    root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    matrix_path = root / "anon_processed_df_parquet"
    dictionary_path = root / "Feature explanations - anonymized data - Sheet1.tsv"
    mapping_path = root / "p0_feature_mapping_v1.yaml"
    try:
        matrix = pd.read_parquet(matrix_path)
    except Exception as exc:
        raise AuditError(
            f"PSYCHE-D Parquet is not readable ({type(exc).__name__})"
        ) from None
    fields = [str(field) for field in matrix.columns]
    missing_labels = sorted(set(DATASET_SPECS[0].label_fields) - set(fields))
    if missing_labels:
        raise AuditError("PSYCHE-D required label fields are missing")
    sample_ids = [str(value) for value in matrix.index]
    participant_ids: list[str] = []
    malformed_index_count = 0
    for sample_id in sample_ids:
        participant, separator, month = sample_id.rpartition("_")
        if not separator or not participant or not month.isdigit():
            malformed_index_count += 1
        else:
            participant_ids.append(participant)
    if malformed_index_count:
        raise AuditError("PSYCHE-D sample_id index contains malformed keys")
    sample_profile = _key_profile(sample_ids)
    participant_profile = _key_profile(participant_ids)
    if sample_profile["duplicate_record_count"]:
        raise AuditError("PSYCHE-D sample_id index is not unique")
    dictionary = _read_delimited_structure(dictionary_path, "\t")
    try:
        with mapping_path.open("r", encoding="utf-8") as handle:
            mapping = yaml.safe_load(handle)
    except Exception as exc:
        raise AuditError(
            f"PSYCHE-D YAML is not readable ({type(exc).__name__})"
        ) from None
    mapping_keys = (
        sorted(str(key) for key in mapping) if isinstance(mapping, dict) else []
    )
    tables = [
        {
            "column_count": int(matrix.shape[1]),
            "dtype_counts": _dtype_counts(matrix),
            "fields": fields,
            "index_name": None if matrix.index.name is None else str(matrix.index.name),
            "relative_path": matrix_path.name,
            "row_count": int(matrix.shape[0]),
            "table_format": "parquet",
        },
        {
            **dictionary,
            "relative_path": dictionary_path.name,
            "table_format": "tsv",
        },
        {
            "relative_path": mapping_path.name,
            "table_format": "yaml",
            "top_level_keys": mapping_keys,
            "v3_3_3_authoritative_schema": False,
        },
    ]
    audit = {
        "key_audit": {
            "candidate": "sample_id index -> rsplit('_', 1)[0]",
            "participant_profile": participant_profile,
            "sample_id_profile": sample_profile,
            "status": "pass",
        },
        "label_audit": {
            "fields_present": list(DATASET_SPECS[0].label_fields),
            "rows_with_all_label_fields": int(
                matrix[list(DATASET_SPECS[0].label_fields)].notna().all(axis=1).sum()
            ),
        },
        "tables": tables,
    }
    return audit, []


def _resilient_path_template(path: Path, root: Path) -> str:
    relative = list(path.relative_to(root).parts)
    if path.name in RESILIENT_SENSOR_FILES and len(relative) >= 2:
        relative[-2] = "{participant}"
    return "/".join(relative)


def _aggregate_delimited_schemas(
    files: Sequence[Path], root: Path, *, resilient_templates: bool = False
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
    for file_path in files:
        structure = _read_delimited_structure(
            file_path, "\t" if file_path.suffix.lower() == ".tsv" else ","
        )
        template = (
            _resilient_path_template(file_path, root)
            if resilient_templates
            else _relative_to_dataset(file_path, root)
        )
        key = (template, tuple(structure["fields"]))
        group = groups.setdefault(
            key,
            {
                "column_count": structure["column_count"],
                "duplicate_field_count": structure["duplicate_field_count"],
                "fields": structure["fields"],
                "file_count": 0,
                "path_template": template,
                "row_count_max": 0,
                "row_count_min": None,
                "row_count_total": 0,
                "table_format": "csv",
            },
        )
        group["file_count"] += 1
        group["row_count_total"] += structure["row_count"]
        group["row_count_max"] = max(group["row_count_max"], structure["row_count"])
        current_min = group["row_count_min"]
        group["row_count_min"] = (
            structure["row_count"]
            if current_min is None
            else min(current_min, structure["row_count"])
        )
    return sorted(groups.values(), key=lambda row: row["path_template"].encode("utf-8"))


def _audit_resilient(
    root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    real_csv_files = sorted(
        (
            path
            for path in root.rglob("*.csv")
            if "__MACOSX" not in path.relative_to(root).parts
        ),
        key=lambda path: path.relative_to(root).as_posix().encode("utf-8"),
    )
    table_schemas = _aggregate_delimited_schemas(
        real_csv_files, root, resilient_templates=True
    )
    demographics_path = root / "Demographics.csv"
    summary_path = root / "summary_stats_per_participant.csv"
    try:
        demographics = pd.read_csv(demographics_path, dtype="string")
        sensor_summary = pd.read_csv(summary_path, dtype="string")
    except Exception as exc:
        raise AuditError(
            f"RESILIENT key tables are not readable ({type(exc).__name__})"
        ) from None
    required_demographic = {"user_id", "phq_total", "gds_total", "gad_total"}
    if not required_demographic.issubset(demographics.columns):
        raise AuditError("RESILIENT required key or label fields are missing")
    if not {"participant", "file"}.issubset(sensor_summary.columns):
        raise AuditError("RESILIENT sensor summary key fields are missing")
    sensor_files = [
        path for path in real_csv_files if path.name in RESILIENT_SENSOR_FILES
    ]
    files_by_directory: dict[str, set[str]] = defaultdict(set)
    for path in sensor_files:
        normalised, _ = _normalise_key(path.parent.name)
        if normalised is not None:
            files_by_directory[normalised].add(path.name)
    demographics_keys = _normalised_key_set(demographics["user_id"])
    summary_keys = _normalised_key_set(sensor_summary["participant"])
    directory_key_set = set(files_by_directory)
    incomplete_directories = sum(
        names != RESILIENT_SENSOR_FILES for names in files_by_directory.values()
    )
    mismatch_counts = {
        "demographics_not_in_sensor_directories": len(
            demographics_keys - directory_key_set
        ),
        "sensor_directories_not_in_demographics": len(
            directory_key_set - demographics_keys
        ),
        "summary_not_in_sensor_directories": len(summary_keys - directory_key_set),
        "sensor_directories_not_in_summary": len(directory_key_set - summary_keys),
    }
    user_profile = _key_profile(demographics["user_id"])
    summary_profile = _key_profile(sensor_summary["participant"])
    directory_profile = _key_profile(files_by_directory.keys())
    key_passed = (
        user_profile["missing_key_count"] == 0
        and user_profile["duplicate_record_count"] == 0
        and incomplete_directories == 0
        and not any(mismatch_counts.values())
    )
    issues: list[dict[str, Any]] = []
    if not key_passed:
        issues.append(
            {
                "category": "participant_key_conflict",
                "dataset_id": "resilient",
                "issue_id": "DATA001-RESILIENT-KEY-001",
                "next_action": "Resolve key cardinality before DATA-003 adaptation.",
                "severity": "blocks_adapter",
                "summary": "Aggregate participant-key cardinality checks did not pass.",
            }
        )
    return (
        {
            "key_audit": {
                "candidate": "user_id plus sensor directory key",
                "demographics": user_profile,
                "incomplete_sensor_directory_count": incomplete_directories,
                "mismatch_counts": mismatch_counts,
                "sensor_directories": directory_profile,
                "sensor_summary": summary_profile,
                "status": "pass" if key_passed else "issue",
            },
            "label_audit": {"fields_present": ["phq_total", "gds_total", "gad_total"]},
            "os_metadata": {
                "included_in_collection_hash": True,
                "parsed_as_tabular_data": False,
            },
            "tables": table_schemas,
        },
        issues,
    )


def _audit_nhanes(
    root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    xpt_files = sorted(root.glob("*.xpt"), key=lambda path: path.name.encode("utf-8"))
    tables: list[dict[str, Any]] = []
    duplicate_key_tables: list[str] = []
    dpq_key_count = 0
    for file_path in xpt_files:
        try:
            frame = pd.read_sas(file_path, format="xport")
        except Exception as exc:
            raise AuditError(
                f"NHANES XPT is not readable: {file_path.name} ({type(exc).__name__})"
            ) from None
        fields = [str(field) for field in frame.columns]
        table: dict[str, Any] = {
            "column_count": int(frame.shape[1]),
            "dtype_counts": _dtype_counts(frame),
            "fields": fields,
            "relative_path": file_path.name,
            "row_count": int(frame.shape[0]),
            "table_format": "sas_xport",
        }
        if "SEQN" in frame.columns:
            profile = _key_profile(frame["SEQN"])
            table["participant_key_profile"] = profile
            if profile["duplicate_record_count"]:
                duplicate_key_tables.append(file_path.name)
        if file_path.stem.startswith("DPQ_"):
            missing = sorted(set(NHANES_PHQ_FIELDS) - set(fields))
            if missing:
                raise AuditError(f"NHANES PHQ fields are missing in {file_path.name}")
            dpq_key_count += _key_profile(frame["SEQN"])["unique_key_count"]
        tables.append(table)
    day_path = root / "NHANES Preliminary Day Level Output.csv"
    day_structure = _read_delimited_structure(day_path, ",")
    required_day_fields = {"SEQN", "calendar_date", "M10VALUE"}
    if not required_day_fields.issubset(day_structure["fields"]):
        raise AuditError("NHANES day-level required fields are missing")
    try:
        day_keys = pd.read_csv(
            day_path,
            usecols=["SEQN", "calendar_date", "window_number"],
            dtype="string",
        )
    except Exception as exc:
        raise AuditError(
            f"NHANES day-level keys are not readable ({type(exc).__name__})"
        ) from None
    participant_day_keys = (
        day_keys["SEQN"].fillna("") + "\x1f" + day_keys["calendar_date"].fillna("")
    )
    participant_window_keys = (
        day_keys["SEQN"].fillna("") + "\x1f" + day_keys["window_number"].fillna("")
    )
    participant_day_profile = _key_profile(participant_day_keys)
    day_key_profile = _key_profile(participant_window_keys)
    participant_profile = _key_profile(day_keys["SEQN"])
    day_key_component_missing = {
        field: int(day_keys[field].isna().sum())
        for field in ("SEQN", "calendar_date", "window_number")
    }
    tables.append(
        {
            **day_structure,
            "participant_day_key_profile": participant_day_profile,
            "participant_key_profile": participant_profile,
            "participant_window_key_profile": day_key_profile,
            "relative_path": day_path.name,
            "table_format": "csv",
        }
    )
    key_passed = (
        not duplicate_key_tables
        and day_key_profile["duplicate_record_count"] == 0
        and not any(day_key_component_missing.values())
    )
    issues: list[dict[str, Any]] = []
    if not key_passed:
        issues.append(
            {
                "category": "participant_key_conflict",
                "dataset_id": "nhanes",
                "issue_id": "DATA001-NHANES-KEY-001",
                "next_action": "Resolve duplicate source keys before DATA-004 adaptation.",
                "severity": "blocks_adapter",
                "summary": "Aggregate cycle/SEQN key checks found duplicate source keys.",
            }
        )
    return (
        {
            "key_audit": {
                "candidate": "cycle plus SEQN; day table row key uses SEQN plus window_number",
                "day_participant_calendar_date_profile": participant_day_profile,
                "day_key_component_missing_counts": day_key_component_missing,
                "day_participant_key_profile": participant_profile,
                "day_participant_window_key_profile": day_key_profile,
                "dpq_cycle_participant_count": dpq_key_count,
                "tables_with_duplicate_seqn_count": len(duplicate_key_tables),
                "status": "pass" if key_passed else "issue",
            },
            "label_audit": {"fields_present": list(NHANES_PHQ_FIELDS)},
            "tables": sorted(
                tables, key=lambda row: row["relative_path"].encode("utf-8")
            ),
        },
        issues,
    )


def _audit_shenzhen(
    root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    csv_path = root / "Mental_Health_Survey_of_the_Elderly.csv"
    structure = _read_delimited_structure(csv_path, ",")
    try:
        frame = pd.read_csv(csv_path, dtype="string")
    except Exception as exc:
        raise AuditError(
            f"Shenzhen CSV is not readable ({type(exc).__name__})"
        ) from None
    required = {"code", "PHQ9score", "Depressivesymptoms"}
    if not required.issubset(frame.columns):
        raise AuditError("Shenzhen required key or label fields are missing")
    profile = _key_profile(frame["code"])
    key_passed = (
        profile["missing_key_count"] == 0 and profile["duplicate_record_count"] == 0
    )
    issues: list[dict[str, Any]] = []
    if not key_passed:
        issues.append(
            {
                "category": "participant_key_conflict",
                "dataset_id": "shenzhen_elderly",
                "issue_id": "DATA001-SHENZHEN-KEY-001",
                "next_action": "Resolve code conflicts before DATA-005 adaptation.",
                "severity": "blocks_adapter",
                "summary": "Aggregate code uniqueness checks did not pass.",
            }
        )
    return (
        {
            "key_audit": {
                "candidate": "code",
                "profile": profile,
                "status": "pass" if key_passed else "issue",
            },
            "label_audit": {
                "fields_present": ["PHQ9score", "Depressivesymptoms"],
                "v3_3_3_primary_source_field": "PHQ9score",
            },
            "tables": [
                {
                    **structure,
                    "relative_path": csv_path.name,
                    "table_format": "csv",
                }
            ],
        },
        issues,
    )


def _hefei_conflicting_key_groups(frame: pd.DataFrame, key_field: str) -> int:
    normalised = frame[key_field].map(lambda value: _normalise_key(value)[0])
    conflict_count = 0
    for _, group in frame.assign(_audit_key=normalised).groupby(
        "_audit_key", dropna=False, sort=False
    ):
        if len(group) <= 1:
            continue
        comparison = group.drop(columns=[key_field, "_audit_key"])
        if len(comparison.astype("string").drop_duplicates()) > 1:
            conflict_count += 1
    return conflict_count


def _audit_hefei(
    root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    workbook_path = root / "社区老年人社会衰弱与抑郁症状研究原始数据.xls"
    try:
        workbook = pd.ExcelFile(workbook_path, engine="xlrd")
        sheets = {
            str(sheet): pd.read_excel(workbook, sheet_name=sheet)
            for sheet in workbook.sheet_names
        }
    except Exception as exc:
        raise AuditError(f"Hefei XLS is not readable ({type(exc).__name__})") from None
    if not sheets:
        raise AuditError("Hefei XLS contains no worksheets")
    tables: list[dict[str, Any]] = []
    for sheet_name, frame in sheets.items():
        tables.append(
            {
                "column_count": int(frame.shape[1]),
                "dtype_counts": _dtype_counts(frame),
                "fields": [str(field) for field in frame.columns],
                "row_count": int(frame.shape[0]),
                "sheet_name": sheet_name,
                "table_format": "xls_worksheet",
            }
        )
    primary = max(sheets.values(), key=len)
    fields = {str(field) for field in primary.columns}
    if "a1" not in fields:
        raise AuditError("Hefei participant-key candidate a1 is missing")
    if not set(HEFEI_PHQ_FIELD_CANDIDATES).issubset(fields):
        raise AuditError("Hefei PHQ-9 field candidates are missing")
    profile = _key_profile(primary["a1"])
    conflicting_groups = _hefei_conflicting_key_groups(primary, "a1")
    issues: list[dict[str, Any]] = []
    key_passed = (
        profile["missing_key_count"] == 0 and profile["duplicate_record_count"] == 0
    )
    if not key_passed:
        issues.append(
            {
                "category": "participant_key_conflict",
                "dataset_id": "hefei_elderly",
                "issue_id": "DATA001-HEFEI-KEY-001",
                "next_action": (
                    "DATA-006 must resolve a1 against the publisher codebook or "
                    "freeze the permitted deterministic row-key fallback."
                ),
                "severity": "blocks_adapter",
                "summary": (
                    "Candidate a1 is not unique; counts are retained without "
                    "recording the duplicated value."
                ),
            }
        )
    issues.append(
        {
            "category": "missing_codebook",
            "dataset_id": "hefei_elderly",
            "issue_id": "DATA001-HEFEI-SCHEMA-002",
            "next_action": (
                "DATA-006 must obtain or verify the publisher codebook before "
                "assigning coded PHQ-9 and social-frailty semantics."
            ),
            "severity": "blocks_adapter",
            "summary": (
                "The local XLS exposes coded field names only; h1-h9 remain PHQ-9 "
                "candidates rather than a confirmed mapping."
            ),
        }
    )
    return (
        {
            "key_audit": {
                "candidate": "a1",
                "conflicting_duplicate_key_group_count": conflicting_groups,
                "profile": profile,
                "status": "pass" if key_passed else "issue",
            },
            "label_audit": {
                "candidate_fields_present": list(HEFEI_PHQ_FIELD_CANDIDATES),
                "mapping_status": "unverified_without_publisher_codebook",
            },
            "tables": tables,
        },
        issues,
    )


def _build_collections(manifest: Sequence[dict[str, Any]]) -> dict[str, Any]:
    collections: dict[str, Any] = {}
    for spec in DATASET_SPECS:
        rows = [row for row in manifest if row["dataset_id"] == spec.dataset_id]
        role_counts = Counter(str(row["file_role"]) for row in rows)
        format_counts = Counter(str(row["format"]) for row in rows)
        collections[spec.dataset_id] = {
            "collection_sha256": _collection_sha256(rows),
            "file_count": len(rows),
            "file_role_counts": dict(sorted(role_counts.items())),
            "format_counts": dict(sorted(format_counts.items())),
            "total_bytes": int(sum(int(row["bytes"]) for row in rows)),
        }
    return collections


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def build_audit(paths: AuditPaths) -> dict[str, Any]:
    private_manifest = build_file_manifest(paths)
    _validate_support_files(paths, private_manifest)
    collections = _build_collections(private_manifest)
    manifest = [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in private_manifest
    ]
    auditors = {
        "psyche_d": _audit_psyche,
        "resilient": _audit_resilient,
        "nhanes": _audit_nhanes,
        "shenzhen_elderly": _audit_shenzhen,
        "hefei_elderly": _audit_hefei,
    }
    dataset_summaries: dict[str, Any] = {}
    issues: list[dict[str, Any]] = []
    for spec in DATASET_SPECS:
        root = paths.workspace_root / spec.source_relative
        dataset_audit, dataset_issues = auditors[spec.dataset_id](root)
        dataset_summaries[spec.dataset_id] = {
            "collection": collections[spec.dataset_id],
            "contract": spec.as_dict(),
            "structure_audit": dataset_audit,
        }
        issues.extend(dataset_issues)

    schema_payload = feature_schema_manifest()
    schema_bytes = _canonical_json_bytes(schema_payload)
    manifest_bytes = _encode_jsonl(manifest)
    manifest_sha256 = _sha256_bytes(manifest_bytes)
    schema_sha256 = _sha256_bytes(schema_bytes)
    total_files = sum(item["file_count"] for item in collections.values())
    total_bytes = sum(item["total_bytes"] for item in collections.values())
    summary = {
        "audit_run_id": AUDIT_RUN_ID,
        "audit_version": AUDIT_VERSION,
        "datasets": dataset_summaries,
        "feature_schema_target": {
            "loader": (
                "elderly_monitoring.modules.mental_health.mood_social."
                "feature_schema_manifest"
            ),
            "schema_version": schema_payload["schema_version"],
            "sha256": schema_sha256,
            "snapshot_file": FEATURE_SCHEMA_NAME,
        },
        "file_manifest": {
            "file": FILE_MANIFEST_NAME,
            "manifest_sha256": manifest_sha256,
            "manifest_version": MANIFEST_VERSION,
            "record_count": len(manifest),
        },
        "hash_contract": {
            "algorithm": "SHA-256",
            "collection_byte_stream": (
                "relative_path + NUL + decimal_bytes + NUL + lowercase_file_sha256 + LF"
            ),
            "includes_all_regular_files": True,
            "includes_mtime": False,
            "collection_path_source": (
                "workspace-visible source logical path; participant directory "
                "segments are used in-memory for hashing but are not persisted"
            ),
            "manifest_path_privacy": (
                "RESILIENT participant directory segments are deterministic "
                "participant_0001 aliases and contain no source identifier"
            ),
            "path_encoding": "UTF-8",
            "path_form": "workspace-relative POSIX",
            "sort_order": "ascending UTF-8 bytes of relative_path",
            "version": HASH_SPEC_VERSION,
        },
        "issues": sorted(issues, key=lambda issue: issue["issue_id"]),
        "privacy_contract": {
            "contains_contact_information": False,
            "contains_participant_rows": False,
            "contains_source_participant_identifiers_or_values": False,
            "contains_only": [
                "file metadata",
                "field and worksheet names",
                "aggregate counts",
                "hashes",
                "adapter boundary metadata",
            ],
            "source_data_read_only": True,
        },
        "scope": {
            "canonical_parquet_created": False,
            "dataset_adapters_created": False,
            "dataset_count": len(DATASET_SPECS),
            "ecdf_or_split_created": False,
            "model_training_performed": False,
            "total_bytes": total_bytes,
            "total_files": total_files,
        },
        "status": "complete_with_recorded_data_quality_issues"
        if issues
        else "complete",
    }
    summary_bytes = _canonical_json_bytes(summary)
    checks = [
        {
            "check": "five_dataset_contracts_present",
            "passed": set(dataset_summaries)
            == {spec.dataset_id for spec in DATASET_SPECS},
        },
        {
            "check": "all_regular_files_hashed",
            "observed": total_files,
            "passed": total_files == len(manifest) and total_files > 0,
        },
        {
            "check": "source_structures_readable",
            "passed": True,
        },
        {
            "check": "mh003_feature_schema_loaded",
            "observed": schema_payload["schema_version"],
            "passed": schema_payload["schema_version"]
            == "mood_social_feature_schema_v3_3_3",
        },
        {
            "check": "open_data_quality_issues_recorded",
            "observed": len(issues),
            "passed": all(issue.get("issue_id") for issue in issues),
        },
    ]
    validation = {
        "artifact_sha256": {
            FEATURE_SCHEMA_NAME: schema_sha256,
            FILE_MANIFEST_NAME: manifest_sha256,
            SUMMARY_NAME: _sha256_bytes(summary_bytes),
        },
        "audit_complete": all(bool(check["passed"]) for check in checks),
        "checks": checks,
        "has_open_data_quality_issues": bool(issues),
        "validation_version": VALIDATION_VERSION,
    }
    validation_bytes = _canonical_json_bytes(validation)
    _atomic_write(paths.output_root / FEATURE_SCHEMA_NAME, schema_bytes)
    _atomic_write(paths.output_root / FILE_MANIFEST_NAME, manifest_bytes)
    _atomic_write(paths.output_root / SUMMARY_NAME, summary_bytes)
    _atomic_write(paths.output_root / VALIDATION_NAME, validation_bytes)
    return {
        "manifest": manifest,
        "summary": summary,
        "validation": validation,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the read-only DATA-001 V3.3.3 dataset audit"
    )
    parser.add_argument("--workspace-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        paths = resolve_paths(args.workspace_root, args.output_root)
        result = build_audit(paths)
    except AuditError as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 2
    summary = result["summary"]
    print(
        json.dumps(
            {
                "dataset_count": summary["scope"]["dataset_count"],
                "file_count": summary["scope"]["total_files"],
                "open_issue_count": len(summary["issues"]),
                "output_root": paths.output_root.relative_to(
                    paths.algorithm_root
                ).as_posix(),
                "status": summary["status"],
                "total_bytes": summary["scope"]["total_bytes"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result["validation"]["audit_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
