"""Deterministic provenance, conversion-report, and split contracts.

The contracts in this module are source-agnostic. Source adapters may inspect
raw data, but they must hand validated, inert values to these contracts and
must write every generated artifact outside the raw source tree.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from numbers import Real
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


SOURCE_MANIFEST_SCHEMA_VERSION = "wandering-source-manifest-v1"
CONVERSION_REPORT_SCHEMA_VERSION = "wandering-conversion-report-v1"
SPLIT_MANIFEST_SCHEMA_VERSION = "wandering-split-v1"

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SLUG_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_VERSION_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_WARNING_CODE_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_WINDOWS_ABSOLUTE_PATH_PATTERN = re.compile(r"^[A-Za-z]:[/\\]")

_SOURCE_FILE_FIELDS = frozenset({"role", "path", "byte_count", "sha256"})
_OUTPUT_ARTIFACT_FIELDS = frozenset(
    {"role", "path", "byte_count", "sha256", "sample_count"}
)
_WARNING_FIELDS = frozenset({"code", "count", "sample_ids"})
_CONVERSION_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "source_name",
        "converter_version",
        "source_files",
        "outputs",
        "sample_count_read",
        "sample_count_written",
        "sample_count_rejected",
        "sample_ids",
        "warnings",
    }
)
_CONVERSION_REPORT_FIELDS = frozenset(
    {
        "schema_version",
        "source_name",
        "converter_version",
        "source_sha256",
        "sample_count_read",
        "sample_count_written",
        "sample_count_rejected",
        "class_counts",
        "length_summary",
        "coordinate_summary",
        "group_fields_found",
        "warnings",
        "output_sha256",
    }
)
_SPLIT_MANIFEST_FIELDS = frozenset(
    {
        "split_version",
        "seed",
        "group_policy",
        "train",
        "validation",
        "test",
        "sealed_external_test",
        "source_hashes",
        "split_sha256",
    }
)


class ManifestValidationError(ValueError):
    """An invalid provenance, conversion-report, or split contract."""


@dataclass(frozen=True)
class SourceFile:
    """One immutable raw input, addressed relative to a registered source root."""

    role: str
    path: str
    byte_count: int
    sha256: str

    def __post_init__(self) -> None:
        _validate_slug(self.role, "source file role")
        _validate_relative_posix_path(self.path, "source file path")
        _validate_non_negative_int(self.byte_count, "source file byte_count")
        _validate_sha256(self.sha256, "source file sha256")

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "path": self.path,
            "byte_count": self.byte_count,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "SourceFile":
        _validate_exact_fields(record, _SOURCE_FILE_FIELDS, "source file")
        return cls(
            role=record["role"],
            path=record["path"],
            byte_count=record["byte_count"],
            sha256=record["sha256"],
        )


@dataclass(frozen=True)
class OutputArtifact:
    """One deterministic safe-format output produced by a converter."""

    role: str
    path: str
    byte_count: int
    sha256: str
    sample_count: int

    def __post_init__(self) -> None:
        _validate_slug(self.role, "output role")
        _validate_relative_posix_path(self.path, "output path")
        _validate_non_negative_int(self.byte_count, "output byte_count")
        _validate_sha256(self.sha256, "output sha256")
        _validate_non_negative_int(self.sample_count, "output sample_count")

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "path": self.path,
            "byte_count": self.byte_count,
            "sha256": self.sha256,
            "sample_count": self.sample_count,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "OutputArtifact":
        _validate_exact_fields(record, _OUTPUT_ARTIFACT_FIELDS, "output artifact")
        return cls(
            role=record["role"],
            path=record["path"],
            byte_count=record["byte_count"],
            sha256=record["sha256"],
            sample_count=record["sample_count"],
        )


@dataclass(frozen=True)
class ConversionWarning:
    """A deterministic warning summary, suitable for machine comparison."""

    code: str
    count: int
    sample_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.code, str)
            or _WARNING_CODE_PATTERN.fullmatch(self.code) is None
        ):
            raise ManifestValidationError(
                "warning code must be a lowercase snake_case identifier"
            )
        _validate_non_negative_int(self.count, "warning count")
        if self.count == 0:
            raise ManifestValidationError("warning count must be positive")
        sample_ids = _normalize_identifiers(self.sample_ids, "warning sample_id")
        if len(sample_ids) > self.count:
            raise ManifestValidationError(
                "warning count must be at least the number of listed sample_ids"
            )
        object.__setattr__(self, "sample_ids", sample_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "count": self.count,
            "sample_ids": list(self.sample_ids),
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "ConversionWarning":
        _validate_exact_fields(record, _WARNING_FIELDS, "conversion warning")
        return cls(
            code=record["code"],
            count=record["count"],
            sample_ids=_as_tuple(record["sample_ids"], "warning sample_ids"),
        )


@dataclass(frozen=True)
class ConversionManifest:
    """Source files, converter identity, output artifacts, and sample identity."""

    schema_version: str
    source_name: str
    converter_version: str
    source_files: tuple[SourceFile, ...]
    outputs: tuple[OutputArtifact, ...]
    sample_count_read: int
    sample_count_written: int
    sample_count_rejected: int
    sample_ids: tuple[str, ...]
    warnings: tuple[ConversionWarning, ...]

    def __post_init__(self) -> None:
        if self.schema_version != SOURCE_MANIFEST_SCHEMA_VERSION:
            raise ManifestValidationError(
                f"schema_version must be {SOURCE_MANIFEST_SCHEMA_VERSION!r}"
            )
        _validate_slug(self.source_name, "source_name")
        _validate_version(self.converter_version, "converter_version")
        source_files = _normalize_typed_values(
            self.source_files,
            SourceFile,
            "source_files",
            key=lambda item: (item.role, item.path),
        )
        if not source_files:
            raise ManifestValidationError("source_files must not be empty")
        _validate_unique_values(
            (item.role for item in source_files), "duplicate source file role"
        )
        _validate_unique_values(
            (item.path for item in source_files), "duplicate source file path"
        )
        outputs = _normalize_typed_values(
            self.outputs,
            OutputArtifact,
            "outputs",
            key=lambda item: (item.role, item.path),
        )
        _validate_unique_values((item.role for item in outputs), "duplicate output role")
        _validate_unique_values((item.path for item in outputs), "duplicate output path")
        warnings = _normalize_warnings(self.warnings)
        sample_ids = _normalize_identifiers(self.sample_ids, "sample_id")
        _validate_counts(
            self.sample_count_read,
            self.sample_count_written,
            self.sample_count_rejected,
        )
        if len(sample_ids) != self.sample_count_written:
            raise ManifestValidationError(
                "sample_count_written must equal the number of sample_ids"
            )
        if sum(item.sample_count for item in outputs) != self.sample_count_written:
            raise ManifestValidationError(
                "output sample_count values must sum to sample_count_written"
            )
        object.__setattr__(self, "source_files", source_files)
        object.__setattr__(self, "outputs", outputs)
        object.__setattr__(self, "sample_ids", sample_ids)
        object.__setattr__(self, "warnings", warnings)

    @property
    def sha256(self) -> str:
        """SHA-256 of the exact canonical manifest bytes."""

        return hashlib.sha256(self.to_bytes()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_name": self.source_name,
            "converter_version": self.converter_version,
            "source_files": [item.to_dict() for item in self.source_files],
            "outputs": [item.to_dict() for item in self.outputs],
            "sample_count_read": self.sample_count_read,
            "sample_count_written": self.sample_count_written,
            "sample_count_rejected": self.sample_count_rejected,
            "sample_ids": list(self.sample_ids),
            "warnings": [item.to_dict() for item in self.warnings],
        }

    def to_bytes(self) -> bytes:
        return _canonical_json_bytes(self.to_dict())

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "ConversionManifest":
        _validate_exact_fields(
            record,
            _CONVERSION_MANIFEST_FIELDS,
            "conversion manifest",
        )
        return cls(
            schema_version=record["schema_version"],
            source_name=record["source_name"],
            converter_version=record["converter_version"],
            source_files=tuple(
                SourceFile.from_dict(item)
                for item in _as_tuple(record["source_files"], "source_files")
            ),
            outputs=tuple(
                OutputArtifact.from_dict(item)
                for item in _as_tuple(record["outputs"], "outputs")
            ),
            sample_count_read=record["sample_count_read"],
            sample_count_written=record["sample_count_written"],
            sample_count_rejected=record["sample_count_rejected"],
            sample_ids=_as_tuple(record["sample_ids"], "sample_ids"),
            warnings=tuple(
                ConversionWarning.from_dict(item)
                for item in _as_tuple(record["warnings"], "warnings")
            ),
        )


@dataclass(frozen=True)
class ConversionReport:
    """Deterministic conversion statistics and anomaly summaries."""

    schema_version: str
    source_name: str
    converter_version: str
    source_sha256: tuple[str, ...]
    sample_count_read: int
    sample_count_written: int
    sample_count_rejected: int
    class_counts: Mapping[str, int]
    length_summary: Mapping[str, int | float]
    coordinate_summary: Mapping[str, int | float]
    group_fields_found: tuple[str, ...]
    warnings: tuple[ConversionWarning, ...]
    output_sha256: Mapping[str, str]

    def __post_init__(self) -> None:
        if self.schema_version != CONVERSION_REPORT_SCHEMA_VERSION:
            raise ManifestValidationError(
                f"schema_version must be {CONVERSION_REPORT_SCHEMA_VERSION!r}"
            )
        _validate_slug(self.source_name, "source_name")
        _validate_version(self.converter_version, "converter_version")
        source_sha256 = _as_tuple(self.source_sha256, "source_sha256")
        if not source_sha256:
            raise ManifestValidationError("source_sha256 must not be empty")
        for digest in source_sha256:
            _validate_sha256(digest, "source_sha256")
        source_sha256 = tuple(sorted(source_sha256))
        _validate_counts(
            self.sample_count_read,
            self.sample_count_written,
            self.sample_count_rejected,
        )
        class_counts = _normalize_count_mapping(self.class_counts, "class_counts")
        if sum(class_counts.values()) != self.sample_count_written:
            raise ManifestValidationError(
                "class_counts values must sum to sample_count_written"
            )
        length_summary = _normalize_numeric_mapping(
            self.length_summary, "length_summary"
        )
        coordinate_summary = _normalize_numeric_mapping(
            self.coordinate_summary, "coordinate_summary"
        )
        group_fields_found = _normalize_slugs(
            self.group_fields_found, "group_fields_found"
        )
        warnings = _normalize_warnings(self.warnings)
        output_sha256 = _normalize_hash_mapping(
            self.output_sha256, "output_sha256"
        )
        object.__setattr__(self, "source_sha256", source_sha256)
        object.__setattr__(self, "class_counts", class_counts)
        object.__setattr__(self, "length_summary", length_summary)
        object.__setattr__(self, "coordinate_summary", coordinate_summary)
        object.__setattr__(self, "group_fields_found", group_fields_found)
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(self, "output_sha256", output_sha256)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_name": self.source_name,
            "converter_version": self.converter_version,
            "source_sha256": list(self.source_sha256),
            "sample_count_read": self.sample_count_read,
            "sample_count_written": self.sample_count_written,
            "sample_count_rejected": self.sample_count_rejected,
            "class_counts": dict(self.class_counts),
            "length_summary": dict(self.length_summary),
            "coordinate_summary": dict(self.coordinate_summary),
            "group_fields_found": list(self.group_fields_found),
            "warnings": [item.to_dict() for item in self.warnings],
            "output_sha256": dict(self.output_sha256),
        }

    def to_bytes(self) -> bytes:
        return _canonical_json_bytes(self.to_dict())

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "ConversionReport":
        _validate_exact_fields(
            record,
            _CONVERSION_REPORT_FIELDS,
            "conversion report",
        )
        return cls(
            schema_version=record["schema_version"],
            source_name=record["source_name"],
            converter_version=record["converter_version"],
            source_sha256=_as_tuple(record["source_sha256"], "source_sha256"),
            sample_count_read=record["sample_count_read"],
            sample_count_written=record["sample_count_written"],
            sample_count_rejected=record["sample_count_rejected"],
            class_counts=record["class_counts"],
            length_summary=record["length_summary"],
            coordinate_summary=record["coordinate_summary"],
            group_fields_found=_as_tuple(
                record["group_fields_found"], "group_fields_found"
            ),
            warnings=tuple(
                ConversionWarning.from_dict(item)
                for item in _as_tuple(record["warnings"], "warnings")
            ),
            output_sha256=record["output_sha256"],
        )


@dataclass(frozen=True)
class SplitManifest:
    """A hash-bound, mutually exclusive four-partition sample assignment."""

    split_version: str
    seed: int
    group_policy: str
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]
    sealed_external_test: tuple[str, ...]
    source_hashes: Mapping[str, str]
    split_sha256: str

    def __post_init__(self) -> None:
        if self.split_version != SPLIT_MANIFEST_SCHEMA_VERSION:
            raise ManifestValidationError(
                f"split_version must be {SPLIT_MANIFEST_SCHEMA_VERSION!r}"
            )
        _validate_non_negative_int(self.seed, "seed")
        _validate_slug(self.group_policy, "group_policy")
        train = _normalize_identifiers(self.train, "train sample_id")
        validation = _normalize_identifiers(
            self.validation, "validation sample_id"
        )
        test = _normalize_identifiers(self.test, "test sample_id")
        sealed = _normalize_identifiers(
            self.sealed_external_test, "sealed_external_test sample_id"
        )
        _validate_partition_overlap(
            {
                "train": train,
                "validation": validation,
                "test": test,
                "sealed_external_test": sealed,
            }
        )
        source_hashes = _normalize_hash_mapping(self.source_hashes, "source_hashes")
        if not source_hashes:
            raise ManifestValidationError("source_hashes must not be empty")
        _validate_sha256(self.split_sha256, "split_sha256")
        object.__setattr__(self, "train", train)
        object.__setattr__(self, "validation", validation)
        object.__setattr__(self, "test", test)
        object.__setattr__(self, "sealed_external_test", sealed)
        object.__setattr__(self, "source_hashes", source_hashes)
        expected_sha256 = _split_payload_sha256(self._payload_without_hash())
        if self.split_sha256 != expected_sha256:
            raise ManifestValidationError(
                "split_sha256 does not match the canonical split payload"
            )

    @property
    def all_sample_ids(self) -> tuple[str, ...]:
        return self.train + self.validation + self.test + self.sealed_external_test

    def validate_sample_ids(
        self,
        known_sample_ids: Iterable[str],
        *,
        require_complete: bool = True,
    ) -> None:
        known = _normalize_identifiers(tuple(known_sample_ids), "known sample_id")
        assigned = set(self.all_sample_ids)
        known_set = set(known)
        unknown = sorted(assigned - known_set)
        if unknown:
            raise ManifestValidationError(
                f"split contains unknown sample_id(s): {', '.join(unknown)}"
            )
        if require_complete:
            missing = sorted(known_set - assigned)
            if missing:
                raise ManifestValidationError(
                    f"split is missing sample_id(s): {', '.join(missing)}"
                )

    def _payload_without_hash(self) -> dict[str, Any]:
        return {
            "split_version": self.split_version,
            "seed": self.seed,
            "group_policy": self.group_policy,
            "train": list(self.train),
            "validation": list(self.validation),
            "test": list(self.test),
            "sealed_external_test": list(self.sealed_external_test),
            "source_hashes": dict(self.source_hashes),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_hash(), "split_sha256": self.split_sha256}

    def to_bytes(self) -> bytes:
        return _canonical_json_bytes(self.to_dict())

    @classmethod
    def from_dict(
        cls,
        record: Mapping[str, Any],
        *,
        known_sample_ids: Iterable[str] | None = None,
        require_complete: bool = True,
    ) -> "SplitManifest":
        _validate_exact_fields(record, _SPLIT_MANIFEST_FIELDS, "split manifest")
        split = cls(
            split_version=record["split_version"],
            seed=record["seed"],
            group_policy=record["group_policy"],
            train=_as_tuple(record["train"], "train"),
            validation=_as_tuple(record["validation"], "validation"),
            test=_as_tuple(record["test"], "test"),
            sealed_external_test=_as_tuple(
                record["sealed_external_test"], "sealed_external_test"
            ),
            source_hashes=record["source_hashes"],
            split_sha256=record["split_sha256"],
        )
        if known_sample_ids is not None:
            split.validate_sample_ids(
                known_sample_ids,
                require_complete=require_complete,
            )
        return split


def derive_sample_id(
    *,
    source_dataset: str,
    source_role: str,
    source_sha256: str,
    source_record_id: str,
) -> str:
    """Derive a stable ID from a raw-file version and its natural record key."""

    _validate_slug(source_dataset, "source_dataset")
    _validate_slug(source_role, "source_role")
    _validate_sha256(source_sha256, "source_sha256")
    _validate_identifier(source_record_id, "source_record_id")
    payload = {
        "source_dataset": source_dataset,
        "source_role": source_role,
        "source_sha256": source_sha256,
        "source_record_id": source_record_id,
    }
    digest = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()[:24]
    return f"{source_dataset}_{digest}"


def describe_source_file(
    path: str | Path,
    *,
    source_root: str | Path,
    role: str,
) -> SourceFile:
    """Read size and SHA-256 without writing to the raw source tree."""

    root = Path(source_root).resolve(strict=True)
    source_path = Path(path).resolve(strict=True)
    if not root.is_dir():
        raise ManifestValidationError(f"source_root is not a directory: {root}")
    if not source_path.is_file():
        raise ManifestValidationError(f"source file is not a regular file: {source_path}")
    try:
        relative_path = source_path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ManifestValidationError(
            "source file must remain inside source_root"
        ) from exc
    return SourceFile(
        role=role,
        path=relative_path,
        byte_count=source_path.stat().st_size,
        sha256=_sha256_file(source_path),
    )


def validate_conversion_bundle(
    manifest: ConversionManifest,
    report: ConversionReport,
) -> None:
    """Cross-check the summary duplicated across a manifest and its report."""

    if not isinstance(manifest, ConversionManifest) or not isinstance(
        report, ConversionReport
    ):
        raise ManifestValidationError(
            "validate_conversion_bundle expects a manifest and conversion report"
        )
    matching_fields = (
        "source_name",
        "converter_version",
        "sample_count_read",
        "sample_count_written",
        "sample_count_rejected",
        "warnings",
    )
    for field in matching_fields:
        if getattr(manifest, field) != getattr(report, field):
            raise ManifestValidationError(
                f"manifest and report {field} values do not match"
            )
    manifest_source_hashes = tuple(sorted(item.sha256 for item in manifest.source_files))
    if manifest_source_hashes != report.source_sha256:
        raise ManifestValidationError(
            "manifest and report source SHA-256 values do not match"
        )
    manifest_output_hashes = {item.role: item.sha256 for item in manifest.outputs}
    if manifest_output_hashes != dict(report.output_sha256):
        raise ManifestValidationError(
            "manifest and report output SHA-256 values do not match"
        )


def build_split_manifest(
    *,
    split_version: str,
    seed: int,
    group_policy: str,
    train: Iterable[str],
    validation: Iterable[str],
    test: Iterable[str],
    sealed_external_test: Iterable[str],
    source_hashes: Mapping[str, str],
    known_sample_ids: Iterable[str],
    require_complete: bool = True,
) -> SplitManifest:
    """Build and validate a canonical split, including its semantic payload hash."""

    normalized_train = _normalize_identifiers(tuple(train), "train sample_id")
    normalized_validation = _normalize_identifiers(
        tuple(validation), "validation sample_id"
    )
    normalized_test = _normalize_identifiers(tuple(test), "test sample_id")
    normalized_sealed = _normalize_identifiers(
        tuple(sealed_external_test), "sealed_external_test sample_id"
    )
    normalized_hashes = _normalize_hash_mapping(source_hashes, "source_hashes")
    payload = {
        "split_version": split_version,
        "seed": seed,
        "group_policy": group_policy,
        "train": list(normalized_train),
        "validation": list(normalized_validation),
        "test": list(normalized_test),
        "sealed_external_test": list(normalized_sealed),
        "source_hashes": dict(normalized_hashes),
    }
    split = SplitManifest(
        split_version=split_version,
        seed=seed,
        group_policy=group_policy,
        train=normalized_train,
        validation=normalized_validation,
        test=normalized_test,
        sealed_external_test=normalized_sealed,
        source_hashes=normalized_hashes,
        split_sha256=_split_payload_sha256(payload),
    )
    split.validate_sample_ids(
        known_sample_ids,
        require_complete=require_complete,
    )
    return split


def write_conversion_manifest(
    path: str | Path,
    manifest: ConversionManifest,
    *,
    overwrite: bool = False,
) -> None:
    if not isinstance(manifest, ConversionManifest):
        raise ManifestValidationError(
            "write_conversion_manifest expects a ConversionManifest"
        )
    _write_json_artifact(path, manifest.to_bytes(), overwrite=overwrite)


def write_conversion_report(
    path: str | Path,
    report: ConversionReport,
    *,
    overwrite: bool = False,
) -> None:
    if not isinstance(report, ConversionReport):
        raise ManifestValidationError(
            "write_conversion_report expects a ConversionReport"
        )
    _write_json_artifact(path, report.to_bytes(), overwrite=overwrite)


def write_split_manifest(
    path: str | Path,
    split: SplitManifest,
    *,
    checksum_path: str | Path | None = None,
    overwrite: bool = False,
) -> None:
    if not isinstance(split, SplitManifest):
        raise ManifestValidationError("write_split_manifest expects a SplitManifest")
    output_path = _validated_json_path(path)
    digest_path = (
        Path(checksum_path)
        if checksum_path is not None
        else output_path.with_suffix(".sha256")
    )
    if digest_path.suffix.lower() != ".sha256":
        raise ManifestValidationError("split checksum path must end with .sha256")
    _preflight_outputs((output_path, digest_path), overwrite=overwrite)
    _atomic_write_bytes(output_path, split.to_bytes(), overwrite=overwrite)
    try:
        _atomic_write_bytes(
            digest_path,
            f"{split.split_sha256}\n".encode("ascii"),
            overwrite=overwrite,
        )
    except Exception:
        if not overwrite:
            output_path.unlink(missing_ok=True)
        raise


def _normalize_typed_values(
    values: Any,
    expected_type: type,
    field: str,
    *,
    key: Any,
) -> tuple[Any, ...]:
    sequence = _as_tuple(values, field)
    if any(not isinstance(item, expected_type) for item in sequence):
        raise ManifestValidationError(
            f"{field} must contain only {expected_type.__name__} values"
        )
    return tuple(sorted(sequence, key=key))


def _normalize_warnings(values: Any) -> tuple[ConversionWarning, ...]:
    warnings = _normalize_typed_values(
        values,
        ConversionWarning,
        "warnings",
        key=lambda item: item.code,
    )
    _validate_unique_values((item.code for item in warnings), "duplicate warning code")
    return warnings


def _normalize_identifiers(values: Any, field: str) -> tuple[str, ...]:
    sequence = _as_tuple(values, field)
    for value in sequence:
        _validate_identifier(value, field)
    duplicates = _duplicates(sequence)
    if duplicates:
        raise ManifestValidationError(
            f"duplicate sample_id in {field}: {', '.join(duplicates)}"
        )
    return tuple(sorted(sequence))


def _normalize_slugs(values: Any, field: str) -> tuple[str, ...]:
    sequence = _as_tuple(values, field)
    for value in sequence:
        _validate_slug(value, field)
    duplicates = _duplicates(sequence)
    if duplicates:
        raise ManifestValidationError(
            f"{field} must not contain duplicates: {', '.join(duplicates)}"
        )
    return tuple(sorted(sequence))


def _normalize_count_mapping(value: Any, field: str) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise ManifestValidationError(f"{field} must be an object")
    normalized: dict[str, int] = {}
    for key, count in value.items():
        _validate_slug(key, f"{field} key")
        _validate_non_negative_int(count, f"{field}.{key}")
        normalized[key] = count
    return dict(sorted(normalized.items()))


def _normalize_numeric_mapping(
    value: Any,
    field: str,
) -> dict[str, int | float]:
    if not isinstance(value, Mapping):
        raise ManifestValidationError(f"{field} must be an object")
    normalized: dict[str, int | float] = {}
    for key, number in value.items():
        _validate_slug(key, f"{field} key")
        if (
            isinstance(number, bool)
            or not isinstance(number, Real)
            or not math.isfinite(float(number))
        ):
            raise ManifestValidationError(f"{field}.{key} must be a finite number")
        normalized[key] = number
    return dict(sorted(normalized.items()))


def _normalize_hash_mapping(value: Any, field: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ManifestValidationError(f"{field} must be an object")
    normalized: dict[str, str] = {}
    for key, digest in value.items():
        _validate_slug(key, f"{field} key")
        _validate_sha256(digest, f"{field}.{key}")
        normalized[key] = digest
    return dict(sorted(normalized.items()))


def _validate_counts(read: Any, written: Any, rejected: Any) -> None:
    _validate_non_negative_int(read, "sample_count_read")
    _validate_non_negative_int(written, "sample_count_written")
    _validate_non_negative_int(rejected, "sample_count_rejected")
    if read != written + rejected:
        raise ManifestValidationError(
            "sample_count_read must equal sample_count_written + "
            "sample_count_rejected"
        )


def _validate_partition_overlap(partitions: Mapping[str, Sequence[str]]) -> None:
    owner: dict[str, str] = {}
    for partition, sample_ids in partitions.items():
        for sample_id in sample_ids:
            previous = owner.get(sample_id)
            if previous is not None:
                raise ManifestValidationError(
                    "split overlap for sample_id "
                    f"{sample_id!r}: {previous} and {partition}"
                )
            owner[sample_id] = partition


def _validate_exact_fields(
    record: Mapping[str, Any],
    expected: frozenset[str],
    label: str,
) -> None:
    if not isinstance(record, Mapping):
        raise ManifestValidationError(f"{label} must be an object")
    unknown = sorted(set(record) - expected)
    missing = sorted(expected - set(record))
    if unknown:
        raise ManifestValidationError(
            f"{label} has unknown fields: {', '.join(unknown)}"
        )
    if missing:
        raise ManifestValidationError(
            f"{label} is missing fields: {', '.join(missing)}"
        )


def _validate_slug(value: Any, field: str) -> None:
    if not isinstance(value, str) or _SLUG_PATTERN.fullmatch(value) is None:
        raise ManifestValidationError(
            f"{field} must contain only lowercase letters, digits, '_' or '-'"
        )


def _validate_version(value: Any, field: str) -> None:
    if not isinstance(value, str) or _VERSION_PATTERN.fullmatch(value) is None:
        raise ManifestValidationError(f"{field} must be a stable lowercase version")


def _validate_identifier(value: Any, field: str) -> None:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ManifestValidationError(f"{field} must be a non-empty trimmed string")


def _validate_non_negative_int(value: Any, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ManifestValidationError(f"{field} must be a non-negative integer")


def _validate_sha256(value: Any, field: str) -> None:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ManifestValidationError(
            f"{field} must be a lowercase 64-character SHA-256 digest"
        )


def _validate_relative_posix_path(value: Any, field: str) -> None:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ManifestValidationError(f"{field} must be a non-empty trimmed string")
    if (
        "\\" in value
        or _WINDOWS_ABSOLUTE_PATH_PATTERN.match(value) is not None
        or PurePosixPath(value).is_absolute()
        or ".." in PurePosixPath(value).parts
    ):
        raise ManifestValidationError(
            f"{field} must be a portable relative POSIX path without '..'"
        )


def _validate_unique_values(values: Iterable[str], message: str) -> None:
    duplicates = _duplicates(tuple(values))
    if duplicates:
        raise ManifestValidationError(f"{message}: {', '.join(duplicates)}")


def _duplicates(values: Sequence[Any]) -> list[str]:
    seen: set[Any] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(str(value))
        seen.add(value)
    return sorted(duplicates)


def _as_tuple(value: Any, field: str) -> tuple[Any, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ManifestValidationError(f"{field} must be an array")
    return tuple(value)


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ManifestValidationError("artifact is not canonical JSON data") from exc
    return rendered.encode("utf-8") + b"\n"


def _split_payload_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_json_path(path: str | Path) -> Path:
    if not isinstance(path, (str, Path)):
        raise ManifestValidationError("JSON artifact path must be a string or Path")
    output = Path(path)
    if output.suffix.lower() != ".json":
        raise ManifestValidationError("JSON artifact path must end with .json")
    return output


def _write_json_artifact(path: str | Path, payload: bytes, *, overwrite: bool) -> None:
    output = _validated_json_path(path)
    _atomic_write_bytes(output, payload, overwrite=overwrite)


def _preflight_outputs(paths: Sequence[Path], *, overwrite: bool) -> None:
    if len(set(paths)) != len(paths):
        raise ManifestValidationError("artifact output paths must be different")
    if not overwrite:
        existing = [path for path in paths if path.exists()]
        if existing:
            raise FileExistsError(f"artifact output already exists: {existing[0]}")


def _atomic_write_bytes(path: Path, payload: bytes, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"artifact output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)
