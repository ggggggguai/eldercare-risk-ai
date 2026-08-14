"""Source-specific, deterministic split construction for public wandering data.

This module is intentionally isolated from the mental-health runtime.  It only
consumes the hash-frozen, inert outputs from wandering conversion step 2 and
writes the five immutable artifacts required by step 3.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml

from elderly_monitoring.modules.mental_health.wandering.datasets import (
    load_trajectory_jsonl,
)
from elderly_monitoring.modules.mental_health.wandering.manifests import (
    SPLIT_MANIFEST_SCHEMA_VERSION,
    ConversionManifest,
    ManifestValidationError,
    SplitManifest,
    build_split_manifest,
)
from elderly_monitoring.modules.mental_health.wandering.schemas import (
    PatternLabel,
    TrajectorySample,
)


SPLIT_CONFIG_SCHEMA_VERSION = "wandering-split-config-v1"
ASSIGNMENTS_SCHEMA_VERSION = "wandering-split-assignments-v1"
NEAR_NEIGHBOR_AUDIT_SCHEMA_VERSION = "wandering-near-neighbor-audit-v1"
SPLIT_REPORT_SCHEMA_VERSION = "wandering-split-report-v1"

_FIXED_SEED = 20260731
_FIXED_GROUP_POLICY = "source_specific_v1"
_FIXED_SMARTCARE_VALIDATION_DATES = ("2020-09-20", "2020-10-07")
_PARTITIONS = ("train", "validation", "test", "sealed_external_test")
_WP_LABELS = ("direct", "pacing", "lapping", "random")
_INPUT_ROLES = (
    "wandering_patterns_manifest",
    "wandering_patterns_samples",
    "smartcare_manifest",
    "smartcare_train_pool",
    "smartcare_official_validation",
    "step2_human_review",
)
_CONFIG_FIELDS = frozenset(
    {
        "schema_version",
        "split_version",
        "seed",
        "group_policy",
        "inputs",
        "wandering_patterns",
        "smartcare",
        "near_neighbor_audit",
        "development_loader",
        "output_schemas",
    }
)
_WP_CONFIG_FIELDS = frozenset(
    {
        "source_name",
        "input_role",
        "labels",
        "per_class_counts",
        "evaluation_scope",
    }
)
_SMARTCARE_CONFIG_FIELDS = frozenset(
    {
        "source_name",
        "development_input_role",
        "sealed_input_role",
        "validation_dates",
        "expected_counts",
        "expected_development_binary_label_counts",
        "declared_sealed_binary_label_counts",
    }
)
_NEAR_NEIGHBOR_CONFIG_FIELDS = frozenset(
    {
        "resample_points",
        "thresholds",
        "allow_time_reversal",
        "allow_rotation",
        "allow_reflection",
        "expected_lt_0_05_total_pairs",
        "expected_lt_0_05_same_label_pairs",
    }
)
_LOADER_CONFIG_FIELDS = frozenset(
    {
        "require_split_manifest",
        "allow_runtime_random_split",
        "default_partition",
        "explicit_evaluation_partitions",
        "forbidden_partitions",
    }
)
_OUTPUT_SCHEMA_FIELDS = frozenset(
    {"assignments", "near_neighbor_audit", "split_report"}
)
_INPUT_DESCRIPTOR_FIELDS = frozenset({"path", "sha256"})
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
_SEALED_SAMPLE_ID_PATTERN = re.compile(rb'"sample_id":"([A-Za-z0-9._:-]+)"')


class SplitDataError(ValueError):
    """Invalid or unsafe data encountered while constructing the fixed split."""


class SplitConfigError(SplitDataError):
    """The frozen step-3 configuration is invalid or has drifted."""


class InputHashMismatchError(SplitDataError):
    """A required step-2 input no longer has its reviewed SHA-256."""


class SplitAccessError(SplitDataError):
    """A development loader attempted to read a protected partition."""


@dataclass(frozen=True)
class WanderingSplitBuildResult:
    """In-memory summary of one successfully committed split directory."""

    output_dir: Path
    split: SplitManifest
    report: Mapping[str, Any]


def load_split_config(path: str | Path) -> dict[str, Any]:
    """Load and strictly validate the one allowed step-3 configuration."""

    config_path = Path(path)
    if not config_path.is_file():
        raise SplitConfigError(f"split config not found: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise SplitConfigError(f"cannot read split config: {config_path}") from exc
    if not isinstance(raw, dict):
        raise SplitConfigError("split config must be a mapping")
    _require_exact_fields(raw, _CONFIG_FIELDS, "split config")

    if raw["schema_version"] != SPLIT_CONFIG_SCHEMA_VERSION:
        raise SplitConfigError(
            f"schema_version must be {SPLIT_CONFIG_SCHEMA_VERSION!r}"
        )
    if raw["split_version"] != SPLIT_MANIFEST_SCHEMA_VERSION:
        raise SplitConfigError(
            f"split_version must be {SPLIT_MANIFEST_SCHEMA_VERSION!r}"
        )
    if raw["seed"] != _FIXED_SEED or isinstance(raw["seed"], bool):
        raise SplitConfigError(f"seed must remain frozen at {_FIXED_SEED}")
    if raw["group_policy"] != _FIXED_GROUP_POLICY:
        raise SplitConfigError(
            f"group_policy must remain {_FIXED_GROUP_POLICY!r}"
        )
    _validate_inputs_config(raw["inputs"])
    _validate_wp_config(raw["wandering_patterns"])
    _validate_smartcare_config(raw["smartcare"])
    _validate_near_neighbor_config(raw["near_neighbor_audit"])
    _validate_loader_config(raw["development_loader"])
    _validate_output_schemas(raw["output_schemas"])
    return raw


def derive_smartcare_day(source_group_id: str) -> str:
    """Parse the exact SmartCare timestamp and return its calendar-day proxy."""

    if not isinstance(source_group_id, str):
        raise SplitDataError("SmartCare source_group_id must be a string")
    try:
        value = datetime.strptime(source_group_id, "%d/%m/%Y %H:%M")
    except ValueError as exc:
        raise SplitDataError(
            "SmartCare source_group_id must exactly match %d/%m/%Y %H:%M: "
            f"{source_group_id!r}"
        ) from exc
    if value.strftime("%d/%m/%Y %H:%M") != source_group_id:
        raise SplitDataError(
            "SmartCare source_group_id is not in canonical %d/%m/%Y %H:%M form: "
            f"{source_group_id!r}"
        )
    return value.date().isoformat()


def assign_wandering_patterns(
    samples: Sequence[TrajectorySample],
    *,
    seed: int,
    per_class_counts: Mapping[str, int],
) -> dict[str, str]:
    """Assign each WP class by its frozen SHA-256 order and exact counts."""

    if seed != _FIXED_SEED or isinstance(seed, bool):
        raise SplitConfigError(f"WanderingPatterns seed must be {_FIXED_SEED}")
    counts = _validated_wp_counts(per_class_counts)
    by_label: dict[str, list[TrajectorySample]] = defaultdict(list)
    seen: set[str] = set()
    for sample in samples:
        if not isinstance(sample, TrajectorySample):
            raise SplitDataError("WanderingPatterns assignments require TrajectorySample")
        if sample.source_dataset != "wandering_patterns":
            raise SplitDataError(
                f"unexpected WanderingPatterns source_dataset: {sample.source_dataset!r}"
            )
        label = sample.pattern_label.value
        if label not in _WP_LABELS:
            raise SplitDataError(
                f"unsupported WanderingPatterns pattern_label: {label!r}"
            )
        if sample.sample_id in seen:
            raise SplitDataError(f"duplicate sample_id {sample.sample_id!r}")
        seen.add(sample.sample_id)
        by_label[label].append(sample)

    expected_per_class = sum(counts.values())
    assignments: dict[str, str] = {}
    for label in _WP_LABELS:
        class_samples = by_label.get(label, [])
        if len(class_samples) != expected_per_class:
            raise SplitDataError(
                f"WanderingPatterns {label} count must be {expected_per_class}, "
                f"got {len(class_samples)}"
            )
        ordered = sorted(
            class_samples,
            key=lambda sample: (
                hashlib.sha256(
                    f"{seed}:wandering_patterns:{sample.sample_id}".encode("utf-8")
                ).hexdigest(),
                sample.sample_id,
            ),
        )
        train_end = counts["train"]
        validation_end = train_end + counts["validation"]
        for index, sample in enumerate(ordered):
            if index < train_end:
                partition = "train"
            elif index < validation_end:
                partition = "validation"
            else:
                partition = "test"
            assignments[sample.sample_id] = partition
    return assignments


def assign_smartcare_development(
    samples: Sequence[TrajectorySample],
    *,
    validation_dates: Sequence[str],
) -> dict[str, str]:
    """Keep complete SmartCare natural-day proxy groups in train or validation."""

    normalized_dates = _validated_validation_dates(validation_dates)
    assignments: dict[str, str] = {}
    day_partitions: dict[str, str] = {}
    for sample in samples:
        if not isinstance(sample, TrajectorySample):
            raise SplitDataError("SmartCare assignments require TrajectorySample")
        if sample.source_dataset != "smartcare":
            raise SplitDataError(
                f"unexpected SmartCare source_dataset: {sample.source_dataset!r}"
            )
        if sample.pattern_label is not PatternLabel.UNKNOWN:
            raise SplitDataError(
                f"SmartCare sample {sample.sample_id!r} must have pattern_label=unknown"
            )
        if sample.source_group_id is None:
            raise SplitDataError(
                f"SmartCare sample {sample.sample_id!r} has no source_group_id"
            )
        if sample.sample_id in assignments:
            raise SplitDataError(f"duplicate sample_id {sample.sample_id!r}")
        day = derive_smartcare_day(sample.source_group_id)
        partition = "validation" if day in normalized_dates else "train"
        previous = day_partitions.setdefault(day, partition)
        if previous != partition:
            raise SplitDataError(f"SmartCare calendar day {day} crosses partitions")
        assignments[sample.sample_id] = partition
    return assignments


def extract_sealed_sample_ids(path: str | Path) -> tuple[str, ...]:
    """Stream only sample IDs from the already hash-verified official file.

    This reader deliberately does not decode JSON objects, coordinates or labels.
    The caller must verify the complete file SHA-256 before using the result in a
    build.
    """

    input_path = Path(path)
    if not input_path.is_file() or input_path.suffix.lower() != ".jsonl":
        raise SplitDataError(f"sealed SmartCare JSONL not found: {input_path}")
    sample_ids: list[str] = []
    seen: set[str] = set()
    with input_path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip(b"\r\n")
            if not line:
                raise SplitDataError(
                    f"{input_path}:{line_number}: blank sealed JSONL line"
                )
            matches = _SEALED_SAMPLE_ID_PATTERN.findall(line)
            if len(matches) != 1:
                raise SplitDataError(
                    f"{input_path}:{line_number}: expected exactly one canonical sample_id"
                )
            sample_id = matches[0].decode("ascii")
            if not _IDENTIFIER_PATTERN.fullmatch(sample_id):
                raise SplitDataError(
                    f"{input_path}:{line_number}: invalid sample_id {sample_id!r}"
                )
            if sample_id in seen:
                raise SplitDataError(
                    f"{input_path}:{line_number}: duplicate sample_id {sample_id!r}"
                )
            seen.add(sample_id)
            sample_ids.append(sample_id)
    if not sample_ids:
        raise SplitDataError("sealed SmartCare file must not be empty")
    return tuple(sample_ids)


def audit_exact_duplicates(
    samples: Sequence[TrajectorySample],
    assignments: Mapping[str, str],
    *,
    source_name: str,
) -> dict[str, Any]:
    """Audit exact and time-reversed point sequences and enforce binding."""

    forward_groups: dict[str, list[TrajectorySample]] = defaultdict(list)
    canonical_groups: dict[str, list[tuple[TrajectorySample, str]]] = defaultdict(list)
    for sample in samples:
        forward_hash = _points_sha256(sample.points)
        reverse_hash = _points_sha256(tuple(reversed(sample.points)))
        forward_groups[forward_hash].append(sample)
        canonical_groups[min(forward_hash, reverse_hash)].append(
            (sample, forward_hash)
        )

    exact_groups: list[dict[str, Any]] = []
    for digest, group in sorted(forward_groups.items()):
        if len(group) > 1:
            exact_groups.append(
                _validate_duplicate_group(
                    group,
                    assignments,
                    source_name=source_name,
                    duplicate_type="exact",
                    canonical_sha256=digest,
                )
            )

    reverse_groups: list[dict[str, Any]] = []
    for digest, items in sorted(canonical_groups.items()):
        forward_hashes = {forward_hash for _, forward_hash in items}
        if len(items) > 1 and len(forward_hashes) > 1:
            reverse_groups.append(
                _validate_duplicate_group(
                    [sample for sample, _ in items],
                    assignments,
                    source_name=source_name,
                    duplicate_type="reverse",
                    canonical_sha256=digest,
                )
            )

    return {
        "canonicalization": "canonical_json_points_sha256_forward_and_reverse",
        "exact_duplicate_group_count": len(exact_groups),
        "exact_duplicate_pair_count": sum(
            len(group["sample_ids"]) * (len(group["sample_ids"]) - 1) // 2
            for group in exact_groups
        ),
        "exact_duplicate_groups": exact_groups,
        "reverse_duplicate_group_count": len(reverse_groups),
        "reverse_duplicate_pair_count": sum(
            len(group["sample_ids"]) * (len(group["sample_ids"]) - 1) // 2
            for group in reverse_groups
        ),
        "reverse_duplicate_groups": reverse_groups,
    }


def procrustes_shape_distance(
    first: Sequence[Sequence[float]],
    second: Sequence[Sequence[float]],
    *,
    resample_points: int = 32,
) -> float:
    """Return the minimum full-Procrustes RMS distance for two trajectories."""

    first_shape = _normalized_resampled_shape(first, resample_points)
    second_shape = _normalized_resampled_shape(second, resample_points)
    forward = _orthogonal_similarity(first_shape, second_shape)
    reverse = _orthogonal_similarity(first_shape, second_shape[::-1])
    squared = max(0.0, 2.0 - 2.0 * max(forward, reverse))
    if squared < 1e-14:
        return 0.0
    return math.sqrt(squared)


def inherit_parent_split(
    parent_sample_id: str,
    split: SplitManifest | Mapping[str, Sequence[str]],
    *,
    requested_split: str | None = None,
) -> str:
    """Resolve a derived window/view split solely from its original parent."""

    if isinstance(split, SplitManifest):
        partitions: Mapping[str, Sequence[str]] = {
            "train": split.train,
            "validation": split.validation,
            "test": split.test,
            "sealed_external_test": split.sealed_external_test,
        }
    elif isinstance(split, Mapping):
        partitions = split
    else:
        raise SplitDataError("split must be a SplitManifest or partition mapping")
    found = [name for name in _PARTITIONS if parent_sample_id in partitions.get(name, ())]
    if not found:
        raise SplitDataError(f"unknown parent_sample_id {parent_sample_id!r}")
    if len(found) != 1:
        raise SplitDataError(f"parent_sample_id {parent_sample_id!r} overlaps partitions")
    inherited = found[0]
    if requested_split is not None and requested_split != inherited:
        raise SplitDataError(
            f"derived samples must inherit parent split {inherited!r}, "
            f"not {requested_split!r}"
        )
    return inherited


def load_development_partition_ids(
    split_path: str | Path,
    *,
    partition: str = "train",
    allow_evaluation_partition: bool = False,
) -> tuple[str, ...]:
    """Load IDs for development without ever exposing the sealed partition."""

    if partition == "sealed_external_test":
        raise SplitAccessError(
            "sealed_external_test is forbidden in training and development loaders"
        )
    if partition not in ("train", "validation", "test"):
        raise SplitAccessError(f"unsupported development partition: {partition!r}")
    if partition != "train" and not allow_evaluation_partition:
        raise SplitAccessError(
            f"partition {partition!r} requires explicit evaluation access"
        )
    path = Path(split_path)
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        split = SplitManifest.from_dict(record)
    except (OSError, UnicodeError, json.JSONDecodeError, ManifestValidationError) as exc:
        raise SplitAccessError(f"cannot load validated split manifest: {path}") from exc
    return tuple(getattr(split, partition))


def build_wandering_split_from_files(
    *,
    config_path: str | Path,
    project_root: str | Path,
    output_dir: str | Path,
) -> WanderingSplitBuildResult:
    """Verify all inputs, construct all artifacts, and commit a new directory."""

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"wandering split output already exists: {output}")
    config_file = Path(config_path)
    config = load_split_config(config_file)
    root = Path(project_root).resolve(strict=True)
    if not root.is_dir():
        raise SplitDataError(f"project_root is not a directory: {root}")

    input_paths, input_hashes = _verify_all_input_hashes(config["inputs"], root)
    config_hash = _sha256_file(config_file)

    wp_manifest = _load_conversion_manifest(input_paths["wandering_patterns_manifest"])
    smartcare_manifest = _load_conversion_manifest(input_paths["smartcare_manifest"])
    wp_samples = load_trajectory_jsonl(input_paths["wandering_patterns_samples"])
    smartcare_development = load_trajectory_jsonl(
        input_paths["smartcare_train_pool"]
    )
    official_ids = extract_sealed_sample_ids(
        input_paths["smartcare_official_validation"]
    )
    _validate_conversion_inputs(
        wp_manifest=wp_manifest,
        smartcare_manifest=smartcare_manifest,
        wp_samples=wp_samples,
        smartcare_development=smartcare_development,
        official_ids=official_ids,
    )

    wp_assignments = assign_wandering_patterns(
        wp_samples,
        seed=config["seed"],
        per_class_counts=config["wandering_patterns"]["per_class_counts"],
    )
    smartcare_assignments = assign_smartcare_development(
        smartcare_development,
        validation_dates=config["smartcare"]["validation_dates"],
    )
    _validate_fixed_counts(
        config=config,
        wp_samples=wp_samples,
        wp_assignments=wp_assignments,
        smartcare_development=smartcare_development,
        smartcare_assignments=smartcare_assignments,
        official_ids=official_ids,
    )

    all_assignments = _build_assignment_records(
        wp_samples=wp_samples,
        wp_assignments=wp_assignments,
        smartcare_development=smartcare_development,
        smartcare_assignments=smartcare_assignments,
        official_ids=official_ids,
    )
    assignment_index = {
        row["sample_id"]: row["split"] for row in all_assignments
    }
    duplicate_audits = {
        "wandering_patterns": audit_exact_duplicates(
            wp_samples,
            assignment_index,
            source_name="wandering_patterns",
        ),
        "smartcare_development": audit_exact_duplicates(
            smartcare_development,
            assignment_index,
            source_name="smartcare",
        ),
    }
    near_neighbor_audit = _build_near_neighbor_audit(
        config=config,
        wp_samples=wp_samples,
        assignments=assignment_index,
        duplicate_audits=duplicate_audits,
        smartcare_development=smartcare_development,
    )

    partitions = {
        partition: tuple(
            sorted(
                row["sample_id"]
                for row in all_assignments
                if row["split"] == partition
            )
        )
        for partition in _PARTITIONS
    }
    source_hashes = {**input_hashes, "split_config": config_hash}
    known_ids = tuple(sorted(row["sample_id"] for row in all_assignments))
    try:
        split = build_split_manifest(
            split_version=config["split_version"],
            seed=config["seed"],
            group_policy=config["group_policy"],
            train=partitions["train"],
            validation=partitions["validation"],
            test=partitions["test"],
            sealed_external_test=partitions["sealed_external_test"],
            source_hashes=source_hashes,
            known_sample_ids=known_ids,
        )
    except ManifestValidationError as exc:
        raise SplitDataError(f"invalid wandering split manifest: {exc}") from exc

    split_bytes = split.to_bytes()
    split_sha_bytes = f"{split.split_sha256}\n".encode("ascii")
    assignment_bytes = _canonical_jsonl_bytes(all_assignments)
    audit_bytes = _canonical_json_bytes(near_neighbor_audit)
    artifact_hashes = {
        "split.json": _sha256_bytes(split_bytes),
        "split.sha256": _sha256_bytes(split_sha_bytes),
        "assignments.jsonl": _sha256_bytes(assignment_bytes),
        "near_neighbor_audit.json": _sha256_bytes(audit_bytes),
    }
    report_without_hash = _build_split_report(
        config=config,
        source_hashes=source_hashes,
        assignments=all_assignments,
        near_neighbor_audit=near_neighbor_audit,
        split=split,
        artifact_hashes=artifact_hashes,
    )
    report = {
        **report_without_hash,
        "report_payload_sha256": _sha256_bytes(
            _canonical_json_bytes(report_without_hash)
        ),
    }
    report_bytes = _canonical_json_bytes(report)
    files = {
        "split.json": split_bytes,
        "split.sha256": split_sha_bytes,
        "assignments.jsonl": assignment_bytes,
        "near_neighbor_audit.json": audit_bytes,
        "split_report.json": report_bytes,
    }
    _commit_new_output_directory(output, files)
    return WanderingSplitBuildResult(output_dir=output, split=split, report=report)


def _validate_inputs_config(value: Any) -> None:
    if not isinstance(value, dict):
        raise SplitConfigError("inputs must be a mapping")
    if set(value) != set(_INPUT_ROLES):
        raise SplitConfigError(
            f"inputs must contain exactly: {', '.join(_INPUT_ROLES)}"
        )
    for role in _INPUT_ROLES:
        descriptor = value[role]
        _require_exact_fields(
            descriptor, _INPUT_DESCRIPTOR_FIELDS, f"inputs.{role}"
        )
        _validate_relative_path(descriptor["path"], f"inputs.{role}.path")
        _validate_sha256(descriptor["sha256"], f"inputs.{role}.sha256")


def _validate_wp_config(value: Any) -> None:
    _require_exact_fields(value, _WP_CONFIG_FIELDS, "wandering_patterns")
    if value["source_name"] != "wandering_patterns":
        raise SplitConfigError("WanderingPatterns source_name must be wandering_patterns")
    if value["input_role"] != "samples":
        raise SplitConfigError("WanderingPatterns input_role must be samples")
    if value["labels"] != list(_WP_LABELS):
        raise SplitConfigError(f"WanderingPatterns labels must be {list(_WP_LABELS)!r}")
    counts = _validated_wp_counts(value["per_class_counts"])
    if counts != {"train": 280, "validation": 60, "test": 60}:
        raise SplitConfigError("WanderingPatterns per-class counts must be 280/60/60")
    if value["evaluation_scope"] != "public_shape_benchmark":
        raise SplitConfigError(
            "WanderingPatterns evaluation_scope must be public_shape_benchmark"
        )


def _validate_smartcare_config(value: Any) -> None:
    _require_exact_fields(value, _SMARTCARE_CONFIG_FIELDS, "smartcare")
    if value["source_name"] != "smartcare":
        raise SplitConfigError("SmartCare source_name must be smartcare")
    if value["development_input_role"] != "train_pool":
        raise SplitConfigError("SmartCare development_input_role must be train_pool")
    if value["sealed_input_role"] != "official_validation":
        raise SplitConfigError("SmartCare sealed_input_role must be official_validation")
    _validated_validation_dates(value["validation_dates"])
    expected_counts = value["expected_counts"]
    if expected_counts != {
        "train": 152,
        "validation": 38,
        "sealed_external_test": 20,
    }:
        raise SplitConfigError("SmartCare expected counts must remain 152/38/20")
    expected_labels = value["expected_development_binary_label_counts"]
    if expected_labels != {
        "train": {0: 73, 1: 79},
        "validation": {0: 17, 1: 21},
    }:
        raise SplitConfigError("SmartCare development label counts have drifted")
    if value["declared_sealed_binary_label_counts"] != {0: 10, 1: 10}:
        raise SplitConfigError("SmartCare declared sealed label counts must remain 10/10")


def _validate_near_neighbor_config(value: Any) -> None:
    _require_exact_fields(value, _NEAR_NEIGHBOR_CONFIG_FIELDS, "near_neighbor_audit")
    if value["resample_points"] != 32 or isinstance(value["resample_points"], bool):
        raise SplitConfigError("near-neighbor resample_points must be 32")
    if value["thresholds"] != [0.02, 0.05]:
        raise SplitConfigError("near-neighbor thresholds must be [0.02, 0.05]")
    for field in ("allow_time_reversal", "allow_rotation", "allow_reflection"):
        if value[field] is not True:
            raise SplitConfigError(f"near_neighbor_audit.{field} must be true")
    if value["expected_lt_0_05_total_pairs"] != 9061:
        raise SplitConfigError("expected <0.05 near-neighbor pair count must be 9061")
    if value["expected_lt_0_05_same_label_pairs"] != 9055:
        raise SplitConfigError("expected <0.05 same-label pair count must be 9055")


def _validate_loader_config(value: Any) -> None:
    _require_exact_fields(value, _LOADER_CONFIG_FIELDS, "development_loader")
    expected = {
        "require_split_manifest": True,
        "allow_runtime_random_split": False,
        "default_partition": "train",
        "explicit_evaluation_partitions": ["validation", "test"],
        "forbidden_partitions": ["sealed_external_test"],
    }
    if value != expected:
        raise SplitConfigError(
            "development loader must require split.json, default to train, "
            "and forbid sealed_external_test"
        )


def _validate_output_schemas(value: Any) -> None:
    _require_exact_fields(value, _OUTPUT_SCHEMA_FIELDS, "output_schemas")
    expected = {
        "assignments": ASSIGNMENTS_SCHEMA_VERSION,
        "near_neighbor_audit": NEAR_NEIGHBOR_AUDIT_SCHEMA_VERSION,
        "split_report": SPLIT_REPORT_SCHEMA_VERSION,
    }
    if value != expected:
        raise SplitConfigError("output schema versions have drifted")


def _validated_wp_counts(value: Mapping[str, int]) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != {
        "train",
        "validation",
        "test",
    }:
        raise SplitConfigError(
            "WanderingPatterns counts need train, validation and test"
        )
    counts = dict(value)
    if any(
        isinstance(count, bool) or not isinstance(count, int) or count < 0
        for count in counts.values()
    ):
        raise SplitConfigError("WanderingPatterns counts must be non-negative integers")
    return counts


def _validated_validation_dates(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise SplitConfigError("SmartCare validation_dates must be a sequence")
    normalized = tuple(values)
    if normalized != _FIXED_SMARTCARE_VALIDATION_DATES:
        raise SplitConfigError(
            "SmartCare validation dates must remain 2020-09-20 and 2020-10-07"
        )
    return normalized


def _verify_all_input_hashes(
    descriptors: Mapping[str, Mapping[str, str]],
    root: Path,
) -> tuple[dict[str, Path], dict[str, str]]:
    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for role in _INPUT_ROLES:
        descriptor = descriptors[role]
        path = (root / PurePosixPath(descriptor["path"])).resolve(strict=False)
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise SplitConfigError(f"input {role} escapes project_root") from exc
        if not path.is_file():
            raise InputHashMismatchError(f"input {role} is missing: {descriptor['path']}")
        actual = _sha256_file(path)
        expected = descriptor["sha256"]
        if actual != expected:
            raise InputHashMismatchError(
                f"input {role} SHA-256 drift: expected {expected}, got {actual}"
            )
        paths[role] = path
        hashes[role] = actual
    return paths, hashes


def _load_conversion_manifest(path: Path) -> ConversionManifest:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        return ConversionManifest.from_dict(record)
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        ManifestValidationError,
    ) as exc:
        raise SplitDataError(f"invalid conversion manifest: {path}") from exc


def _validate_conversion_inputs(
    *,
    wp_manifest: ConversionManifest,
    smartcare_manifest: ConversionManifest,
    wp_samples: Sequence[TrajectorySample],
    smartcare_development: Sequence[TrajectorySample],
    official_ids: Sequence[str],
) -> None:
    if wp_manifest.source_name != "wandering_patterns":
        raise SplitDataError("WanderingPatterns manifest source_name mismatch")
    if smartcare_manifest.source_name != "smartcare":
        raise SplitDataError("SmartCare manifest source_name mismatch")
    wp_ids = tuple(sample.sample_id for sample in wp_samples)
    smartcare_ids = tuple(sample.sample_id for sample in smartcare_development)
    if len(wp_ids) != 1600:
        raise SplitDataError(f"WanderingPatterns must contain 1600 samples, got {len(wp_ids)}")
    if len(smartcare_ids) != 190:
        raise SplitDataError(
            f"SmartCare development pool must contain 190 samples, got {len(smartcare_ids)}"
        )
    if len(official_ids) != 20:
        raise SplitDataError(
            f"SmartCare official validation must contain 20 IDs, got {len(official_ids)}"
        )
    if set(wp_manifest.sample_ids) != set(wp_ids):
        raise SplitDataError("WanderingPatterns manifest sample IDs do not match samples")
    if set(smartcare_manifest.sample_ids) != set(smartcare_ids) | set(official_ids):
        raise SplitDataError("SmartCare manifest sample IDs do not match pool plus sealed IDs")
    all_ids = wp_ids + smartcare_ids + tuple(official_ids)
    if len(all_ids) != len(set(all_ids)):
        raise SplitDataError("sample IDs overlap across wandering input roles")


def _validate_fixed_counts(
    *,
    config: Mapping[str, Any],
    wp_samples: Sequence[TrajectorySample],
    wp_assignments: Mapping[str, str],
    smartcare_development: Sequence[TrajectorySample],
    smartcare_assignments: Mapping[str, str],
    official_ids: Sequence[str],
) -> None:
    for label in _WP_LABELS:
        counts = Counter(
            wp_assignments[sample.sample_id]
            for sample in wp_samples
            if sample.pattern_label.value == label
        )
        if counts != Counter({"train": 280, "validation": 60, "test": 60}):
            raise SplitDataError(f"WanderingPatterns {label} split count mismatch: {counts}")
    smartcare_counts = Counter(smartcare_assignments.values())
    if smartcare_counts != Counter({"train": 152, "validation": 38}):
        raise SplitDataError(f"SmartCare split count mismatch: {smartcare_counts}")
    label_counts = Counter(
        (smartcare_assignments[sample.sample_id], sample.binary_label)
        for sample in smartcare_development
    )
    expected = Counter(
        {
            (partition, label): count
            for partition, values in config["smartcare"][
                "expected_development_binary_label_counts"
            ].items()
            for label, count in values.items()
        }
    )
    if label_counts != expected:
        raise SplitDataError(
            f"SmartCare development label counts mismatch: {dict(label_counts)}"
        )
    if len(official_ids) != config["smartcare"]["expected_counts"][
        "sealed_external_test"
    ]:
        raise SplitDataError("SmartCare sealed count mismatch")


def _build_assignment_records(
    *,
    wp_samples: Sequence[TrajectorySample],
    wp_assignments: Mapping[str, str],
    smartcare_development: Sequence[TrajectorySample],
    smartcare_assignments: Mapping[str, str],
    official_ids: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in wp_samples:
        rows.append(
            {
                "schema_version": ASSIGNMENTS_SCHEMA_VERSION,
                "sample_id": sample.sample_id,
                "source_name": "wandering_patterns",
                "input_role": "samples",
                "split": wp_assignments[sample.sample_id],
                "leakage_group_id": f"wandering_patterns_sample:{sample.sample_id}",
                "binary_label": sample.binary_label,
                "pattern_label": sample.pattern_label.value,
                "binary_supervision_eligible": True,
                "pattern_supervision_eligible": True,
            }
        )
    for sample in smartcare_development:
        if sample.source_group_id is None:
            raise SplitDataError(f"SmartCare sample {sample.sample_id!r} has no timestamp")
        day = derive_smartcare_day(sample.source_group_id)
        rows.append(
            {
                "schema_version": ASSIGNMENTS_SCHEMA_VERSION,
                "sample_id": sample.sample_id,
                "source_name": "smartcare",
                "input_role": "train_pool",
                "split": smartcare_assignments[sample.sample_id],
                "leakage_group_id": f"smartcare_day:{day}",
                "calendar_day": day,
                "binary_label": sample.binary_label,
                "pattern_label": sample.pattern_label.value,
                "binary_supervision_eligible": True,
                "pattern_supervision_eligible": False,
            }
        )
    for sample_id in official_ids:
        rows.append(
            {
                "schema_version": ASSIGNMENTS_SCHEMA_VERSION,
                "sample_id": sample_id,
                "source_name": "smartcare",
                "input_role": "official_validation",
                "split": "sealed_external_test",
                "leakage_group_id": f"smartcare_official:{sample_id}",
                "binary_supervision_eligible": False,
                "pattern_supervision_eligible": False,
            }
        )
    rows.sort(key=lambda row: row["sample_id"])
    if len(rows) != 1810 or len({row["sample_id"] for row in rows}) != 1810:
        raise SplitDataError("exactly 1810 unique assignment rows are required")
    return rows


def _build_near_neighbor_audit(
    *,
    config: Mapping[str, Any],
    wp_samples: Sequence[TrajectorySample],
    assignments: Mapping[str, str],
    duplicate_audits: Mapping[str, Mapping[str, Any]],
    smartcare_development: Sequence[TrajectorySample],
) -> dict[str, Any]:
    near_config = config["near_neighbor_audit"]
    statistics = _wp_near_neighbor_statistics(
        wp_samples,
        assignments=assignments,
        resample_points=near_config["resample_points"],
        thresholds=near_config["thresholds"],
    )
    lt_005 = statistics["lt_0_05"]
    if lt_005["total_pairs"] != near_config["expected_lt_0_05_total_pairs"]:
        raise SplitDataError(
            "WanderingPatterns <0.05 near-neighbor count drift: expected "
            f"{near_config['expected_lt_0_05_total_pairs']}, "
            f"got {lt_005['total_pairs']}"
        )
    if (
        lt_005["same_label_pairs"]
        != near_config["expected_lt_0_05_same_label_pairs"]
    ):
        raise SplitDataError(
            "WanderingPatterns <0.05 same-label near-neighbor count drift: expected "
            f"{near_config['expected_lt_0_05_same_label_pairs']}, "
            f"got {lt_005['same_label_pairs']}"
        )

    day_counts = Counter(
        derive_smartcare_day(sample.source_group_id)
        for sample in smartcare_development
        if sample.source_group_id is not None
    )
    label_counts = Counter(
        str(sample.binary_label) for sample in smartcare_development
    )
    return {
        "schema_version": NEAR_NEIGHBOR_AUDIT_SCHEMA_VERSION,
        "parameters": {
            "resample_points": near_config["resample_points"],
            "centering": "coordinate_mean",
            "scale": "isotropic_frobenius_norm",
            "distance": "full_procrustes_rms",
            "allow_time_reversal": True,
            "allow_rotation": True,
            "allow_reflection": True,
            "threshold_comparison": "strict_less_than",
            "thresholds": near_config["thresholds"],
        },
        "wandering_patterns": {
            **duplicate_audits["wandering_patterns"],
            "near_neighbors": statistics,
            "near_neighbors_change_fixed_assignment": False,
            "test_scope": "public_shape_benchmark",
        },
        "smartcare_development": {
            **duplicate_audits["smartcare_development"],
            "calendar_day_counts": dict(sorted(day_counts.items())),
            "binary_label_counts": dict(sorted(label_counts.items())),
            "group_guarantee": "calendar_day_proxy_groups_are_disjoint",
        },
        "official_validation_shape_audit_performed": False,
    }


def _wp_near_neighbor_statistics(
    samples: Sequence[TrajectorySample],
    *,
    assignments: Mapping[str, str],
    resample_points: int,
    thresholds: Sequence[float],
) -> dict[str, dict[str, int]]:
    shapes = np.stack(
        [_normalized_resampled_shape(sample.points, resample_points) for sample in samples]
    )
    x = shapes[:, :, 0]
    y = shapes[:, :, 1]
    reverse_x = x[:, ::-1]
    reverse_y = y[:, ::-1]

    forward_00 = x @ x.T
    forward_01 = x @ y.T
    forward_10 = y @ x.T
    forward_11 = y @ y.T
    reverse_00 = x @ reverse_x.T
    reverse_01 = x @ reverse_y.T
    reverse_10 = y @ reverse_x.T
    reverse_11 = y @ reverse_y.T
    forward_similarity = _orthogonal_similarity_matrix(
        forward_00, forward_01, forward_10, forward_11
    )
    reverse_similarity = _orthogonal_similarity_matrix(
        reverse_00, reverse_01, reverse_10, reverse_11
    )
    similarity = np.minimum(1.0, np.maximum(forward_similarity, reverse_similarity))
    distances = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * similarity))
    left, right = np.triu_indices(len(samples), k=1)
    pair_distances = distances[left, right]
    labels = np.asarray([sample.pattern_label.value for sample in samples])
    partitions = np.asarray([assignments[sample.sample_id] for sample in samples])
    same_label = labels[left] == labels[right]
    cross_partition = partitions[left] != partitions[right]

    result: dict[str, dict[str, int]] = {}
    for threshold in thresholds:
        selected = pair_distances < threshold
        total = int(np.count_nonzero(selected))
        same = int(np.count_nonzero(selected & same_label))
        cross = int(np.count_nonzero(selected & cross_partition))
        key = f"lt_{threshold:.2f}".replace(".", "_")
        result[key] = {
            "total_pairs": total,
            "same_label_pairs": same,
            "cross_label_pairs": total - same,
            "cross_partition_pairs": cross,
            "within_partition_pairs": total - cross,
        }
    return result


def _normalized_resampled_shape(
    points: Sequence[Sequence[float]],
    resample_points: int,
) -> np.ndarray:
    if isinstance(resample_points, bool) or resample_points < 2:
        raise SplitDataError("resample_points must be an integer >= 2")
    try:
        array = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise SplitDataError("trajectory points must be finite numeric pairs") from exc
    if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] != 2:
        raise SplitDataError("trajectory must contain at least two XY points")
    if not np.isfinite(array).all():
        raise SplitDataError("trajectory points must be finite")
    segment_lengths = np.linalg.norm(np.diff(array, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total_length = float(cumulative[-1])
    if total_length <= 0.0:
        raise SplitDataError("trajectory arc length must be positive")
    keep = np.concatenate(([True], np.diff(cumulative) > 0.0))
    unique_distances = cumulative[keep]
    unique_points = array[keep]
    targets = np.linspace(0.0, total_length, resample_points, dtype=np.float64)
    resampled = np.column_stack(
        (
            np.interp(targets, unique_distances, unique_points[:, 0]),
            np.interp(targets, unique_distances, unique_points[:, 1]),
        )
    )
    centered = resampled - resampled.mean(axis=0, keepdims=True)
    scale = float(np.linalg.norm(centered))
    if scale <= 0.0:
        raise SplitDataError("trajectory isotropic scale must be positive")
    return centered / scale


def _orthogonal_similarity(first: np.ndarray, second: np.ndarray) -> float:
    cross = first.T @ second
    return float(
        math.sqrt(
            max(
                0.0,
                float(np.square(cross).sum()) + 2.0 * abs(float(np.linalg.det(cross))),
            )
        )
    )


def _orthogonal_similarity_matrix(
    c00: np.ndarray,
    c01: np.ndarray,
    c10: np.ndarray,
    c11: np.ndarray,
) -> np.ndarray:
    frobenius_squared = c00 * c00 + c01 * c01 + c10 * c10 + c11 * c11
    determinant = c00 * c11 - c01 * c10
    return np.sqrt(np.maximum(0.0, frobenius_squared + 2.0 * np.abs(determinant)))


def _validate_duplicate_group(
    samples: Sequence[TrajectorySample],
    assignments: Mapping[str, str],
    *,
    source_name: str,
    duplicate_type: str,
    canonical_sha256: str,
) -> dict[str, Any]:
    label_signatures = {
        (sample.binary_label, sample.pattern_label.value) for sample in samples
    }
    ids = tuple(sorted(sample.sample_id for sample in samples))
    if len(label_signatures) != 1:
        raise SplitDataError(
            f"{source_name} {duplicate_type} label conflict for sample IDs: "
            + ", ".join(ids)
        )
    try:
        partitions = {assignments[sample_id] for sample_id in ids}
    except KeyError as exc:
        raise SplitDataError(f"missing assignment for duplicate sample {exc.args[0]!r}") from exc
    if len(partitions) != 1:
        raise SplitDataError(
            f"{source_name} {duplicate_type} duplicate trajectory crosses partition: "
            + ", ".join(ids)
        )
    binary_label, pattern_label = next(iter(label_signatures))
    return {
        "canonical_sha256": canonical_sha256,
        "sample_ids": list(ids),
        "binary_label": binary_label,
        "pattern_label": pattern_label,
        "split": next(iter(partitions)),
    }


def _build_split_report(
    *,
    config: Mapping[str, Any],
    source_hashes: Mapping[str, str],
    assignments: Sequence[Mapping[str, Any]],
    near_neighbor_audit: Mapping[str, Any],
    split: SplitManifest,
    artifact_hashes: Mapping[str, str],
) -> dict[str, Any]:
    partition_counts = Counter(row["split"] for row in assignments)
    source_partition_counts: dict[str, dict[str, int]] = {}
    for source in ("wandering_patterns", "smartcare"):
        counts = Counter(
            row["split"] for row in assignments if row["source_name"] == source
        )
        source_partition_counts[source] = {
            partition: counts.get(partition, 0) for partition in _PARTITIONS
        }
    wp_pattern_counts: dict[str, dict[str, int]] = {}
    for label in _WP_LABELS:
        counts = Counter(
            row["split"]
            for row in assignments
            if row["source_name"] == "wandering_patterns"
            and row["pattern_label"] == label
        )
        wp_pattern_counts[label] = {
            partition: counts.get(partition, 0)
            for partition in ("train", "validation", "test")
        }
    smartcare_days: dict[str, dict[str, int]] = defaultdict(
        lambda: {"train": 0, "validation": 0}
    )
    smartcare_labels: dict[str, dict[str, int]] = {
        "train": {"0": 0, "1": 0},
        "validation": {"0": 0, "1": 0},
    }
    for row in assignments:
        if row["source_name"] != "smartcare" or row["input_role"] != "train_pool":
            continue
        smartcare_days[row["calendar_day"]][row["split"]] += 1
        smartcare_labels[row["split"]][str(row["binary_label"])] += 1
    return {
        "schema_version": SPLIT_REPORT_SCHEMA_VERSION,
        "split_version": split.split_version,
        "split_sha256": split.split_sha256,
        "seed": config["seed"],
        "group_policy": config["group_policy"],
        "source_hashes": dict(source_hashes),
        "artifact_hashes": dict(artifact_hashes),
        "partition_counts": {
            partition: partition_counts.get(partition, 0)
            for partition in _PARTITIONS
        },
        "source_partition_counts": source_partition_counts,
        "wandering_patterns_pattern_counts": wp_pattern_counts,
        "smartcare_development_binary_label_counts": smartcare_labels,
        "smartcare_calendar_day_counts": {
            day: counts for day, counts in sorted(smartcare_days.items())
        },
        "declared_sealed_binary_label_counts": {
            str(label): count
            for label, count in config["smartcare"][
                "declared_sealed_binary_label_counts"
            ].items()
        },
        "declared_sealed_counts_source": "frozen_config_not_decoded_by_split_builder",
        "evaluation_scopes": {
            "wandering_patterns_test": "public_shape_benchmark",
            "smartcare_official_validation": "sealed_external_binary_evaluation_only",
        },
        "group_guarantees": {
            "wandering_patterns": "original_sample_ids_only_no_person_or_session_groups",
            "smartcare_development": "calendar_day_proxy_groups_are_disjoint",
        },
        "constraints": {
            "all_1810_samples_assigned_exactly_once": len(assignments) == 1810
            and len({row["sample_id"] for row in assignments}) == 1810,
            "partitions_are_pairwise_disjoint": True,
            "official_ids_only_in_sealed_external_test": all(
                row["split"] == "sealed_external_test"
                for row in assignments
                if row["input_role"] == "official_validation"
            ),
            "smartcare_pattern_supervision_disabled": all(
                not row["pattern_supervision_eligible"]
                for row in assignments
                if row["source_name"] == "smartcare"
            ),
            "runtime_random_split_allowed": False,
            "development_loader_default_partition": "train",
            "development_loader_forbids_sealed_external_test": True,
        },
        "near_neighbor_summary": near_neighbor_audit["wandering_patterns"][
            "near_neighbors"
        ],
        "limitations": [
            "wandering_patterns_has_no_person_or_session_groups",
            "smartcare_calendar_day_is_a_proxy_not_a_person_or_session_identifier",
            "cross_partition_shape_neighbors_remain_in_public_shape_benchmark",
            "public_trajectory_results_do_not_validate_camera_or_clinical_use",
        ],
    }


def _points_sha256(points: Sequence[Sequence[float]]) -> str:
    value = [[float(x), float(y)] for x, y in points]
    return _sha256_bytes(_canonical_json_bytes(value))


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SplitDataError("artifact is not canonical JSON data") from exc


def _canonical_jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_json_bytes(dict(row)) for row in rows)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise SplitDataError(f"cannot hash input file: {path}") from exc
    return digest.hexdigest()


def _commit_new_output_directory(
    output: Path,
    files: Mapping[str, bytes],
) -> None:
    if output.exists():
        raise FileExistsError(f"wandering split output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    )
    committed = False
    try:
        for name, payload in files.items():
            path = temp_dir / name
            with path.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        os.replace(temp_dir, output)
        committed = True
    finally:
        if not committed:
            shutil.rmtree(temp_dir, ignore_errors=True)


def _require_exact_fields(
    value: Any,
    expected: frozenset[str],
    field: str,
) -> None:
    if not isinstance(value, dict):
        raise SplitConfigError(f"{field} must be a mapping")
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        parts = []
        if missing:
            parts.append(f"missing fields: {', '.join(missing)}")
        if unknown:
            parts.append(f"unknown fields: {', '.join(unknown)}")
        raise SplitConfigError(f"{field} has " + "; ".join(parts))


def _validate_relative_path(value: Any, field: str) -> None:
    if not isinstance(value, str) or not value:
        raise SplitConfigError(f"{field} must be a relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value or ":" in value:
        raise SplitConfigError(f"{field} must be a portable relative POSIX path")


def _validate_sha256(value: Any, field: str) -> None:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise SplitConfigError(f"{field} must be a lowercase 64-character SHA-256")
