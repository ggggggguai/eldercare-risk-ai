from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import tempfile
from typing import Any, Iterable, Iterator, Mapping, Sequence

from elderly_monitoring.common.config import load_yaml
from elderly_monitoring.modules.fall_risk.annotation_validation import (
    load_validation_config,
)


TASK_TYPES: dict[str, str] = {
    "fall_event_v1": "fall_event",
    "near_fall_event_v1": "near_fall_event",
    "functional_proxy_v1": "functional_proxy",
    "longitudinal_baseline_v1": "longitudinal_baseline",
}
PARTITION_ORDER = ("train", "validation", "test")
ELIGIBLE_REVIEW_STATUSES = frozenset({"reviewed", "final"})

_UNKNOWN_IDENTIFIERS = frozenset(
    {"", "unknown", "none", "null", "n/a", "na", "unavailable", "missing"}
)
_LEAKAGE_AUDIT_KEYS = (
    "subject_id",
    "source_group_id",
    "original_event_id",
    "content_sha256",
    "leakage_component_id",
)
_PARENT_REFERENCE_FIELDS = (
    "parent_asset_id",
    "derived_from_asset_id",
    "source_asset_id",
    "duplicate_of_asset_id",
)
_ADJACENT_REFERENCE_FIELDS = ("adjacent_asset_ids", "neighbor_asset_ids")
_DERIVATION_GROUP_FIELDS = (
    "derivation_group_id",
    "derived_group_id",
    "duplicate_group_id",
)
_ADJACENCY_GROUP_FIELDS = (
    "adjacent_window_group_id",
    "adjacency_group_id",
    "window_group_id",
)
_FORMAL_LABEL_FILES = {
    "fall_event_v1": "event_labels.jsonl",
    "near_fall_event_v1": "event_labels.jsonl",
    "functional_proxy_v1": "risk_labels.jsonl",
    "longitudinal_baseline_v1": "risk_labels.jsonl",
}
_WRITE_LOCK_NAME = ".fall-risk-splits.write.lock"
_ATTESTATION_CREATION_TOKEN = object()


class SplitConfigError(ValueError):
    pass


class SplitDataError(ValueError):
    pass


class SplitLeakageError(ValueError):
    pass


class FrozenSplitError(RuntimeError):
    pass


class _VerifiedValidationReport:
    """Opaque proof produced only after a report and all inputs are verified."""

    __slots__ = (
        "_labels_canonical_sha256",
        "_manifest_canonical_sha256",
        "_manifest_file_sha256",
        "_sha256",
    )

    def __init__(
        self,
        sha256: str,
        *,
        manifest_file_sha256: str,
        manifest_canonical_sha256: str,
        labels_canonical_sha256: Mapping[str, str],
        _creation_token: object,
    ) -> None:
        if _creation_token is not _ATTESTATION_CREATION_TOKEN:
            raise TypeError("validation attestations can only be created by file verification")
        self._sha256 = sha256
        self._manifest_file_sha256 = manifest_file_sha256
        self._manifest_canonical_sha256 = manifest_canonical_sha256
        self._labels_canonical_sha256 = dict(labels_canonical_sha256)

    @property
    def sha256(self) -> str:
        return self._sha256

    def assert_matches(
        self,
        *,
        manifest_file_sha256: str,
        manifest_canonical_sha256: str,
        labels_canonical_sha256: Mapping[str, str],
    ) -> None:
        if (
            manifest_file_sha256 != self._manifest_file_sha256
            or manifest_canonical_sha256 != self._manifest_canonical_sha256
            or dict(labels_canonical_sha256) != self._labels_canonical_sha256
        ):
            raise SplitDataError(
                "verified validation report attestation does not match split inputs"
            )


def build_fall_risk_splits(
    manifest_rows: Iterable[Mapping[str, Any]],
    labels_by_task: Mapping[str, Iterable[Mapping[str, Any]]],
    config: Mapping[str, Any],
    *,
    manifest_sha256: str | None = None,
    _validation_attestation: _VerifiedValidationReport | None = None,
) -> dict[str, dict[str, Any]]:
    """Build deterministic, label-free assignments for all four fall-risk tasks."""
    normalized_config = _validate_config(config)
    manifests = [dict(row) for row in manifest_rows]
    manifest_by_id, manifest_by_reference = _index_manifest(manifests)
    manifest_canonical_sha256 = _rows_sha256(manifests)
    materialized_labels = {
        split_name: [dict(row) for row in labels_by_task.get(split_name, ())]
        for split_name in TASK_TYPES
    }
    labels_canonical_sha256 = {
        split_name: _rows_sha256(rows)
        for split_name, rows in materialized_labels.items()
    }
    source_manifest_sha256 = manifest_sha256 or manifest_canonical_sha256
    if _normalize_sha256(source_manifest_sha256) is None:
        raise SplitDataError("manifest_sha256 must be a valid SHA-256")
    if normalized_config["protocol_status"] == "frozen":
        if not isinstance(_validation_attestation, _VerifiedValidationReport):
            raise SplitDataError(
                "frozen split requires a verified formal validation report"
            )
        _validation_attestation.assert_matches(
            manifest_file_sha256=source_manifest_sha256,
            manifest_canonical_sha256=manifest_canonical_sha256,
            labels_canonical_sha256=labels_canonical_sha256,
        )
        validation_report_sha256 = _validation_attestation.sha256
    else:
        if _validation_attestation is not None:
            raise SplitDataError(
                "validation attestation is only accepted for a frozen split protocol"
            )
        validation_report_sha256 = None

    artifacts: dict[str, dict[str, Any]] = {}
    for split_name, expected_task_type in TASK_TYPES.items():
        task_config = normalized_config["tasks"][split_name]
        raw_labels = materialized_labels[split_name]
        task_labels = _filter_task_labels(raw_labels, task_config)
        labels_sha256 = _rows_sha256(task_labels)
        effective_config = {
            "schema_version": normalized_config["schema_version"],
            "protocol_status": normalized_config["protocol_status"],
            "seed": normalized_config["seed"],
            "partitions": normalized_config["partitions"],
            "task": task_config,
        }
        config_sha256 = _value_sha256(effective_config)
        candidates, excluded_counts = _eligible_candidates(
            task_labels,
            manifest_by_id=manifest_by_id,
            manifest_by_reference=manifest_by_reference,
        )
        artifacts[split_name] = _build_task_artifact(
            split_name=split_name,
            task_type=expected_task_type,
            candidates=candidates,
            task_labels=task_labels,
            config=normalized_config,
            task_config=task_config,
            excluded_counts=excluded_counts,
            manifest_sha256=source_manifest_sha256,
            manifest_canonical_sha256=manifest_canonical_sha256,
            validation_report_sha256=validation_report_sha256,
            labels_sha256=labels_sha256,
            config_sha256=config_sha256,
        )
    return artifacts


def audit_split_leakage(
    assignments: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return deterministic cross-partition leakage findings."""
    rows = [dict(row) for row in assignments]
    issues: list[dict[str, Any]] = []
    for key in _LEAKAGE_AUDIT_KEYS:
        partitions_by_value: dict[str, set[str]] = defaultdict(set)
        samples_by_value: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            value = row.get(key)
            if key != "leakage_component_id" and not _is_known_identifier(value):
                continue
            if key == "leakage_component_id" and not _is_known_identifier(value):
                continue
            normalized_value = str(value)
            partitions_by_value[normalized_value].add(str(row.get("partition", "")))
            samples_by_value[normalized_value].add(str(row.get("sample_id", "")))
        for value in sorted(partitions_by_value):
            partitions = sorted(partitions_by_value[value])
            if len(partitions) <= 1:
                continue
            issues.append(
                {
                    "key": key,
                    "value": value,
                    "partitions": partitions,
                    "sample_ids": sorted(samples_by_value[value]),
                }
            )
    return sorted(
        issues,
        key=lambda issue: (issue["key"], issue["value"], issue["partitions"]),
    )


def write_split_artifacts(
    artifacts: Mapping[str, Mapping[str, Any]],
    output_dir: Path | str,
    *,
    overwrite_development: bool = False,
) -> None:
    """Write all task artifacts while holding one cross-process output lock.

    A lock left by an interrupted process is never removed automatically. An operator
    must first verify that no writer is active, then remove the lock file manually.
    """
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    with _exclusive_output_lock(root):
        _write_split_artifacts_locked(
            artifacts,
            root,
            overwrite_development=overwrite_development,
        )


def _write_split_artifacts_locked(
    artifacts: Mapping[str, Mapping[str, Any]],
    root: Path,
    *,
    overwrite_development: bool,
) -> None:
    _preflight_output_paths(
        artifacts,
        root,
        overwrite_development=overwrite_development,
    )
    staging_root = Path(
        tempfile.mkdtemp(dir=root, prefix=".fall-risk-splits.", suffix=".staging")
    )
    backup_root = Path(
        tempfile.mkdtemp(dir=root, prefix=".fall-risk-splits.", suffix=".backup")
    )
    rollback_root = staging_root / ".rollback"
    rollback_root.mkdir()
    attempted: list[tuple[Path, Path, bool]] = []
    keep_backup = False
    try:
        # Materialize the complete generation before changing any visible task.
        for split_name in sorted(artifacts):
            artifact = artifacts[split_name]
            metadata = dict(artifact["metadata"])
            assignments = [dict(row) for row in artifact["assignments"]]
            staging_dir = staging_root / split_name
            staging_dir.mkdir()
            _write_atomic(staging_dir / "split.json", _render_json(metadata))
            _write_atomic(
                staging_dir / "assignments.jsonl", _render_jsonl(assignments)
            )

        # Keep every prior task generation until the whole batch has committed.
        for split_name in sorted(artifacts):
            task_dir = root / split_name
            staging_dir = staging_root / split_name
            backup_dir = backup_root / split_name
            had_original = task_dir.exists()
            attempted.append((task_dir, backup_dir, had_original))
            if task_dir.exists():
                _assert_task_dir_replaceable(
                    task_dir, overwrite_development=overwrite_development
                )
                os.replace(task_dir, backup_dir)
            os.replace(staging_dir, task_dir)
    except BaseException as write_error:
        try:
            _rollback_split_batch(attempted, rollback_root)
        except BaseException as rollback_error:
            keep_backup = True
            raise SplitDataError(
                "split batch write failed and rollback was incomplete; "
                f"recovery backups were retained at {backup_root}: {rollback_error}"
            ) from write_error
        raise
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root)
        if not keep_backup and backup_root.exists():
            shutil.rmtree(backup_root)


def _rollback_split_batch(
    attempted: Sequence[tuple[Path, Path, bool]], rollback_root: Path
) -> None:
    errors: list[str] = []
    for task_dir, backup_dir, had_original in reversed(attempted):
        try:
            rollback_dir = rollback_root / task_dir.name
            if backup_dir.exists():
                if task_dir.exists():
                    os.replace(task_dir, rollback_dir)
                os.replace(backup_dir, task_dir)
            elif not had_original and task_dir.exists():
                os.replace(task_dir, rollback_dir)
            elif had_original and not task_dir.exists():
                raise SplitDataError(
                    "both the original task and its transaction backup are missing"
                )
        except BaseException as exc:
            errors.append(f"{task_dir.name}: {type(exc).__name__}: {exc}")
    if errors:
        raise SplitDataError("; ".join(errors))


def _assert_task_dir_replaceable(
    task_dir: Path, *, overwrite_development: bool
) -> None:
    metadata_path = task_dir / "split.json"
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SplitDataError(
                f"cannot verify split created during write: {metadata_path}: {exc}"
            ) from exc
        if isinstance(metadata, dict) and metadata.get("status") == "frozen":
            raise FrozenSplitError(
                f"frozen split must be versioned instead of rewritten: {task_dir.name}"
            )
    if not overwrite_development:
        raise FileExistsError(
            f"split output appeared during write and will not be overwritten: {task_dir}"
        )


def build_splits_from_files(
    *,
    manifest_path: Path | str,
    annotations_dir: Path | str,
    config_path: Path | str,
    output_dir: Path | str,
    overwrite_development: bool = False,
    validation_report_path: Path | str | None = None,
    validation_config_path: Path | str = Path(
        "configs/data/fall_risk_label_validation_v1.yaml"
    ),
) -> dict[str, dict[str, Any]]:
    config = load_yaml(config_path)
    normalized_config = _validate_config(config)
    manifest_rows, manifest_file_sha256 = _read_jsonl_with_sha(Path(manifest_path))
    annotation_root = Path(annotations_dir)
    label_cache: dict[str, list[dict[str, Any]]] = {}
    label_file_sha256: dict[str, str] = {}
    missing_label_files: set[str] = set()
    labels_by_task: dict[str, list[dict[str, Any]]] = {}
    for split_name in TASK_TYPES:
        label_file = normalized_config["tasks"][split_name]["label_file"]
        if label_file not in label_cache:
            label_path = annotation_root / label_file
            if label_path.is_file():
                label_cache[label_file], label_file_sha256[label_file] = (
                    _read_jsonl_with_sha(label_path)
                )
            else:
                label_cache[label_file] = []
                missing_label_files.add(label_file)
        labels_by_task[split_name] = label_cache[label_file]

    validation_attestation = None
    if normalized_config["protocol_status"] == "frozen":
        try:
            load_validation_config(validation_config_path)
        except (FileNotFoundError, OSError, ValueError) as exc:
            raise SplitDataError(
                f"frozen split validation config is unsafe or unreadable: {exc}"
            ) from exc
        for split_name, expected_label_file in _FORMAL_LABEL_FILES.items():
            actual_label_file = normalized_config["tasks"][split_name]["label_file"]
            if actual_label_file != expected_label_file:
                raise SplitDataError(
                    f"frozen task {split_name} label_file must be "
                    f"{expected_label_file!r} so it is covered by the validation report"
                )
        expected_input_sha256 = {
            "manifest": manifest_file_sha256,
            "event_labels": _required_cached_file_sha256(
                label_file_sha256, "event_labels.jsonl"
            ),
            "risk_labels": _required_cached_file_sha256(
                label_file_sha256, "risk_labels.jsonl"
            ),
        }
        ancillary_input_paths = {
            "action_labels": annotation_root / "action_labels.jsonl",
            "subject_profiles": annotation_root / "subject_profiles.json",
            "review_log": annotation_root / "annotation_review_log.jsonl",
            "validation_config": Path(validation_config_path),
        }
        for field, path in ancillary_input_paths.items():
            _, expected_input_sha256[field] = _read_bytes_with_sha(path)
        validation_attestation = _verify_formal_validation_report(
            Path(validation_report_path) if validation_report_path is not None else None,
            expected_input_sha256=expected_input_sha256,
            manifest_canonical_sha256=_rows_sha256(manifest_rows),
            labels_canonical_sha256={
                split_name: _rows_sha256(rows)
                for split_name, rows in labels_by_task.items()
            },
        )
    artifacts = build_fall_risk_splits(
        manifest_rows,
        labels_by_task,
        normalized_config,
        manifest_sha256=manifest_file_sha256,
        _validation_attestation=validation_attestation,
    )
    for split_name, artifact in artifacts.items():
        label_file = normalized_config["tasks"][split_name]["label_file"]
        if label_file not in missing_label_files:
            continue
        artifact["metadata"]["status"] = "blocked"
        artifact["metadata"]["split_id"] = None
        artifact["metadata"]["split_sha256"] = None
        artifact["metadata"]["blockers"] = [
            {
                "code": "missing_label_file",
                "message": f"Configured label file is missing: {label_file}",
            }
        ]
        artifact["assignments"] = []
    write_split_artifacts(
        artifacts,
        output_dir,
        overwrite_development=overwrite_development,
    )
    return artifacts


def _build_task_artifact(
    *,
    split_name: str,
    task_type: str,
    candidates: dict[str, dict[str, Any]],
    task_labels: list[dict[str, Any]],
    config: dict[str, Any],
    task_config: dict[str, Any],
    excluded_counts: Counter[str],
    manifest_sha256: str,
    manifest_canonical_sha256: str,
    validation_report_sha256: str | None,
    labels_sha256: str,
    config_sha256: str,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "schema_version": config["schema_version"],
        "split_name": split_name,
        "task_type": task_type,
        "protocol_status": config["protocol_status"],
        "status": "blocked",
        "split_id": None,
        "split_sha256": None,
        "assignments_sha256": _bytes_sha256(b""),
        "manifest_sha256": manifest_sha256,
        "manifest_canonical_sha256": manifest_canonical_sha256,
        "validation_report_sha256": validation_report_sha256,
        "labels_sha256": labels_sha256,
        "config_sha256": config_sha256,
        "allocation_method": "sha256(seed+split_name+component_id)",
        "seed": config["seed"],
        "partitions": dict(config["partitions"]),
        "counts": {partition: 0 for partition in PARTITION_ORDER},
        "eligible_sample_count": len(candidates),
        "eligible_label_count": len(task_labels) - sum(excluded_counts.values()),
        "excluded_counts": dict(sorted(excluded_counts.items())),
        "leakage_audit_keys": list(_LEAKAGE_AUDIT_KEYS),
        "blockers": [],
    }
    if not candidates:
        metadata["blockers"] = [
            {
                "code": "no_eligible_samples",
                "message": (
                    "No manifest assets have eligibility=true with reviewed/final "
                    "task labels and conservative leakage identifiers."
                ),
            }
        ]
        return {"metadata": metadata, "assignments": []}

    components = _build_leakage_components(candidates)
    base_assignments: list[dict[str, Any]] = []
    for component_id, asset_ids in components:
        partition = _partition_for_component(
            seed=config["seed"],
            split_name=split_name,
            component_id=component_id,
            partitions=config["partitions"],
        )
        for asset_id in asset_ids:
            candidate = candidates[asset_id]
            base_assignments.append(
                {
                    "sample_id": asset_id,
                    "asset_id": asset_id,
                    "video_id": candidate["video_id"],
                    "task_type": task_type,
                    "dataset": candidate["dataset"],
                    "partition": partition,
                    "subject_id": candidate["subject_id"],
                    "source_group_id": candidate["source_group_id"],
                    "original_event_id": candidate["original_event_id"],
                    "content_sha256": candidate["content_sha256"],
                    "leakage_component_id": component_id,
                }
            )
    base_assignments.sort(key=_assignment_sort_key)
    leakage_issues = audit_split_leakage(base_assignments)
    if leakage_issues:
        raise SplitLeakageError(
            "split builder produced cross-partition leakage: "
            + json.dumps(leakage_issues, ensure_ascii=False, sort_keys=True)
        )

    status = "frozen" if task_config["frozen"] else "ready"
    split_payload = {
        "schema_version": config["schema_version"],
        "split_name": split_name,
        "task_type": task_type,
        "protocol_status": config["protocol_status"],
        "status": status,
        "manifest_sha256": manifest_sha256,
        "manifest_canonical_sha256": manifest_canonical_sha256,
        "validation_report_sha256": validation_report_sha256,
        "labels_sha256": labels_sha256,
        "config_sha256": config_sha256,
        "assignments": base_assignments,
    }
    split_sha256 = _value_sha256(split_payload)
    split_id = f"{split_name}:sha256:{split_sha256}"
    assignments = [dict(row, split_id=split_id) for row in base_assignments]
    assignments.sort(key=_assignment_sort_key)
    assignments_payload = _render_jsonl(assignments).encode("utf-8")

    metadata.update(
        {
            "status": status,
            "split_id": split_id,
            "split_sha256": split_sha256,
            "assignments_sha256": _bytes_sha256(assignments_payload),
        }
    )
    metadata["counts"] = {
        partition: sum(row["partition"] == partition for row in assignments)
        for partition in PARTITION_ORDER
    }
    return {"metadata": metadata, "assignments": assignments}


def _eligible_candidates(
    labels: Iterable[Mapping[str, Any]],
    *,
    manifest_by_id: Mapping[str, Mapping[str, Any]],
    manifest_by_reference: Mapping[str, str],
) -> tuple[dict[str, dict[str, Any]], Counter[str]]:
    candidates: dict[str, dict[str, Any]] = {}
    eligible_labels: dict[str, list[dict[str, Any]]] = defaultdict(list)
    excluded: Counter[str] = Counter()
    for label in sorted((dict(row) for row in labels), key=_canonical_sort_key):
        if str(label.get("review_status", "")).strip().lower() not in ELIGIBLE_REVIEW_STATUSES:
            excluded["review_status"] += 1
            continue
        if label.get("eligibility") is not True:
            excluded["label_eligibility"] += 1
            continue
        review_evidence = label.get("review_evidence_ids")
        if not isinstance(review_evidence, list) or not any(
            isinstance(value, str) and value.strip() for value in review_evidence
        ):
            excluded["missing_review_evidence"] += 1
            continue
        reference = _label_asset_reference(label)
        if reference is None:
            excluded["missing_asset_reference"] += 1
            continue
        asset_id = manifest_by_reference.get(reference)
        if asset_id is None:
            excluded["missing_manifest_asset"] += 1
            continue
        manifest = dict(manifest_by_id[asset_id])
        if manifest.get("eligibility") is not True:
            excluded["manifest_ineligible"] += 1
            continue
        license_id = str(manifest.get("license_id") or "").strip().lower()
        source_uri = str(manifest.get("source_uri") or "").strip()
        exclusion_reasons = manifest.get("exclusion_reasons")
        if (
            not source_uri
            or not license_id
            or "unknown" in license_id
            or not isinstance(exclusion_reasons, list)
            or bool(exclusion_reasons)
        ):
            excluded["manifest_provenance"] += 1
            continue
        raw_content_sha256 = manifest.get("sha256") or manifest.get("content_sha256")
        if not _is_known_identifier(raw_content_sha256):
            excluded["missing_content_hash"] += 1
            continue
        content_sha256 = _normalize_sha256(raw_content_sha256)
        if content_sha256 is None:
            excluded["invalid_content_hash"] += 1
            continue
        subject_id = _normalized_identifier(manifest.get("subject_id"))
        source_group_id = _normalized_identifier(manifest.get("source_group_id"))
        if subject_id == "unknown" and source_group_id == "unknown":
            excluded["missing_conservative_group"] += 1
            continue
        eligible_labels[asset_id].append(label)
        candidates[asset_id] = {
            "asset_id": asset_id,
            "video_id": _optional_identifier(manifest.get("video_id")),
            "dataset": str(manifest.get("dataset") or "unknown"),
            "subject_id": subject_id,
            "source_group_id": source_group_id,
            "original_event_id": _optional_identifier(manifest.get("original_event_id")),
            "content_sha256": content_sha256,
            "manifest": manifest,
            "labels": eligible_labels[asset_id],
        }
    return candidates, excluded


def _build_leakage_components(
    candidates: Mapping[str, Mapping[str, Any]],
) -> list[tuple[str, list[str]]]:
    union_find = _UnionFind(candidates)
    owner_by_token: dict[str, str] = {}
    for asset_id in sorted(candidates):
        for token in sorted(_leakage_tokens(candidates[asset_id])):
            owner = owner_by_token.setdefault(token, asset_id)
            union_find.union(asset_id, owner)

    members_by_root: dict[str, list[str]] = defaultdict(list)
    for asset_id in sorted(candidates):
        members_by_root[union_find.find(asset_id)].append(asset_id)

    components: list[tuple[str, list[str]]] = []
    for members in members_by_root.values():
        sorted_members = sorted(members)
        digest = _bytes_sha256(("\n".join(sorted_members) + "\n").encode("utf-8"))
        components.append((f"component:sha256:{digest}", sorted_members))
    return sorted(components, key=lambda component: component[0])


def _leakage_tokens(candidate: Mapping[str, Any]) -> set[str]:
    manifest = candidate["manifest"]
    labels = candidate["labels"]
    asset_id = str(candidate["asset_id"])
    tokens = {f"asset:{asset_id}", f"content:{candidate['content_sha256']}"}
    if _is_known_identifier(candidate["subject_id"]):
        tokens.add(f"subject:{candidate['subject_id']}")
    if _is_known_identifier(candidate["source_group_id"]):
        tokens.add(f"source-group:{candidate['source_group_id']}")
    if _is_known_identifier(candidate["original_event_id"]):
        tokens.add(f"original-event:{candidate['original_event_id']}")

    for source in (manifest, *labels):
        for field in _PARENT_REFERENCE_FIELDS:
            for value in _field_values(source.get(field)):
                if _is_known_identifier(value):
                    tokens.add(f"asset:{_identifier_text(value)}")
        for field in _ADJACENT_REFERENCE_FIELDS:
            for value in _field_values(source.get(field)):
                if _is_known_identifier(value):
                    tokens.add(f"asset:{_identifier_text(value)}")
        for field in _DERIVATION_GROUP_FIELDS:
            if _is_known_identifier(source.get(field)):
                tokens.add(f"derivation-group:{_identifier_text(source[field])}")
        for field in _ADJACENCY_GROUP_FIELDS:
            if _is_known_identifier(source.get(field)):
                tokens.add(f"adjacency-group:{_identifier_text(source[field])}")
    return tokens


def _partition_for_component(
    *,
    seed: str | int,
    split_name: str,
    component_id: str,
    partitions: Mapping[str, float],
) -> str:
    payload = f"{seed}\0{split_name}\0{component_id}".encode("utf-8")
    rank = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / 2**64
    cumulative = 0.0
    for partition in PARTITION_ORDER:
        cumulative += float(partitions[partition])
        if rank < cumulative or partition == PARTITION_ORDER[-1]:
            return partition
    raise AssertionError("partition allocation did not select a partition")


def _filter_task_labels(
    labels: Iterable[Mapping[str, Any]],
    task_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    event_types = set(task_config.get("event_types", ()))
    label_task_types = set(task_config.get("label_task_types", ()))
    filtered: list[dict[str, Any]] = []
    for source in labels:
        row = dict(source)
        if event_types and str(row.get("event_type", "")) not in event_types:
            continue
        if label_task_types and str(row.get("task_type", "")) not in label_task_types:
            continue
        filtered.append(row)
    return sorted(filtered, key=_canonical_sort_key)


def _index_manifest(
    manifests: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    by_id: dict[str, dict[str, Any]] = {}
    by_reference: dict[str, str] = {}
    for source in manifests:
        row = dict(source)
        asset_id = str(row.get("asset_id", "")).strip()
        if not asset_id:
            raise SplitDataError("manifest row is missing a non-empty asset_id")
        if asset_id in by_id:
            raise SplitDataError(f"duplicate manifest asset_id: {asset_id}")
        by_id[asset_id] = row
        references = {asset_id}
        if _is_known_identifier(row.get("video_id")):
            references.add(_identifier_text(row["video_id"]))
        for reference in references:
            existing = by_reference.get(reference)
            if existing is not None and existing != asset_id:
                raise SplitDataError(
                    f"manifest reference {reference!r} maps to both {existing!r} and {asset_id!r}"
                )
            by_reference[reference] = asset_id
    return by_id, by_reference


def _validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise SplitConfigError("split config must be a mapping")
    schema_version = str(config.get("schema_version", "")).strip()
    if not schema_version:
        raise SplitConfigError("schema_version must be a non-empty string")
    protocol_status = str(config.get("protocol_status", "")).strip()
    if protocol_status not in {"provisional", "frozen"}:
        raise SplitConfigError("protocol_status must be provisional or frozen")
    seed = config.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, (str, int)) or str(seed) == "":
        raise SplitConfigError("seed must be a non-empty string or integer")

    raw_partitions = config.get("partitions")
    if not isinstance(raw_partitions, Mapping) or set(raw_partitions) != set(PARTITION_ORDER):
        raise SplitConfigError(
            "partitions must contain exactly train, validation and test"
        )
    partitions: dict[str, float] = {}
    for partition in PARTITION_ORDER:
        value = raw_partitions[partition]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SplitConfigError(f"partition {partition} must be numeric")
        number = float(value)
        if not math.isfinite(number) or number < 0.0 or number > 1.0:
            raise SplitConfigError(f"partition {partition} must be between 0 and 1")
        partitions[partition] = number
    if not math.isclose(sum(partitions.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise SplitConfigError("partition proportions must sum to 1.0")

    raw_tasks = config.get("tasks")
    if not isinstance(raw_tasks, Mapping) or set(raw_tasks) != set(TASK_TYPES):
        raise SplitConfigError(
            "tasks must contain exactly: " + ", ".join(sorted(TASK_TYPES))
        )
    tasks: dict[str, dict[str, Any]] = {}
    for split_name, expected_task_type in TASK_TYPES.items():
        raw_task = raw_tasks[split_name]
        if not isinstance(raw_task, Mapping):
            raise SplitConfigError(f"task {split_name} must be a mapping")
        task_type = str(raw_task.get("task_type", ""))
        if task_type != expected_task_type:
            raise SplitConfigError(
                f"task {split_name} must use task_type={expected_task_type}"
            )
        label_file = str(raw_task.get("label_file", "")).strip()
        if not label_file or Path(label_file).is_absolute() or ".." in Path(label_file).parts:
            raise SplitConfigError(
                f"task {split_name} label_file must be a relative file path"
            )
        frozen = raw_task.get("frozen", False)
        if not isinstance(frozen, bool):
            raise SplitConfigError(f"task {split_name} frozen must be boolean")
        task = dict(raw_task)
        task["task_type"] = task_type
        task["label_file"] = label_file
        task["frozen"] = frozen
        for field in ("event_types", "label_task_types", "stratify_by"):
            values = task.get(field, [])
            if not isinstance(values, list) or any(
                not isinstance(value, str) or not value for value in values
            ):
                raise SplitConfigError(
                    f"task {split_name} {field} must be a list of non-empty strings"
                )
            task[field] = list(values)
        if task["stratify_by"]:
            raise SplitConfigError(
                f"task {split_name} stratify_by is not implemented; keep it empty"
            )
        if task["frozen"] and protocol_status != "frozen":
            raise SplitConfigError(
                f"task {split_name} cannot be frozen under a provisional protocol"
            )
        tasks[split_name] = task
    return {
        "schema_version": schema_version,
        "protocol_status": protocol_status,
        "seed": seed,
        "partitions": partitions,
        "tasks": tasks,
    }


def _preflight_output_paths(
    artifacts: Mapping[str, Mapping[str, Any]],
    root: Path,
    *,
    overwrite_development: bool,
) -> None:
    if set(artifacts) != set(TASK_TYPES):
        raise SplitDataError(
            "artifacts must contain exactly: " + ", ".join(sorted(TASK_TYPES))
        )
    for split_name in sorted(artifacts):
        task_dir = root / split_name
        metadata_path = task_dir / "split.json"
        assignments_path = task_dir / "assignments.jsonl"
        existing = metadata_path.exists() or assignments_path.exists()
        if not existing:
            continue
        if metadata_path.is_file():
            try:
                existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise SplitDataError(
                    f"cannot verify existing split metadata: {metadata_path}: {exc}"
                ) from exc
            if isinstance(existing_metadata, dict) and existing_metadata.get("status") == "frozen":
                raise FrozenSplitError(
                    f"frozen split must be versioned instead of rewritten: {split_name}"
                )
        if not overwrite_development:
            raise FileExistsError(
                f"split output already exists; use a new version or explicit development override: {task_dir}"
            )


@contextmanager
def _exclusive_output_lock(root: Path) -> Iterator[None]:
    lock_path = root / _WRITE_LOCK_NAME
    owner = secrets.token_hex(16)
    payload = _canonical_bytes({"owner": owner, "pid": os.getpid()}) + b"\n"
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except FileExistsError as exc:
        raise SplitDataError(
            f"split write lock already exists: {lock_path}; if it is stale, "
            "verify that no writer is active before removing it manually"
        ) from exc
    except OSError as exc:
        raise SplitDataError(f"unable to create split write lock {lock_path}: {exc}") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        yield
    finally:
        try:
            current_payload = lock_path.read_bytes()
        except FileNotFoundError:
            current_payload = None
        except OSError as exc:
            raise SplitDataError(
                f"unable to verify owned split write lock {lock_path}: {exc}"
            ) from exc
        if current_payload == payload:
            try:
                lock_path.unlink()
            except OSError as exc:
                raise SplitDataError(
                    f"unable to remove owned split write lock {lock_path}: {exc}"
                ) from exc


def _required_cached_file_sha256(
    hashes_by_file: Mapping[str, str], file_name: str
) -> str:
    digest = hashes_by_file.get(file_name)
    if digest is None:
        raise SplitDataError(
            f"frozen split requires validator input file: {file_name}"
        )
    return digest


def _read_bytes_with_sha(path: Path) -> tuple[bytes, str]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise SplitDataError(f"unable to read {path}: {exc}") from exc
    return payload, _bytes_sha256(payload)


def _read_jsonl_with_sha(path: Path) -> tuple[list[dict[str, Any]], str]:
    payload, digest = _read_bytes_with_sha(path)
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SplitDataError(f"unable to decode {path} as UTF-8") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SplitDataError(
                f"{path}:{line_number}: invalid JSON: {exc.msg}"
            ) from exc
        if not isinstance(value, dict):
            raise SplitDataError(f"{path}:{line_number}: JSONL record must be an object")
        rows.append(value)
    return rows, digest


def _verify_formal_validation_report(
    report_path: Path | None,
    *,
    expected_input_sha256: Mapping[str, str],
    manifest_canonical_sha256: str,
    labels_canonical_sha256: Mapping[str, str],
) -> _VerifiedValidationReport:
    if report_path is None or not report_path.is_file():
        raise SplitDataError("frozen split requires an existing validation report")
    try:
        payload = report_path.read_bytes()
        report = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SplitDataError(f"unable to read formal validation report: {exc}") from exc
    if not isinstance(report, dict):
        raise SplitDataError("formal validation report must be a JSON object")
    if (
        report.get("schema_version") != "fall-risk-label-validation-report-v1"
        or report.get("mode") != "formal"
        or report.get("valid") is not True
    ):
        raise SplitDataError("validation report is not a valid formal report")
    counts = report.get("counts")
    if not isinstance(counts, dict):
        raise SplitDataError("formal validation report counts must be an object")
    for field in ("errors", "blockers"):
        value = counts.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value != 0:
            raise SplitDataError(
                "formal validation report contains an error or blocker count"
            )
    issues = report.get("issues")
    if not isinstance(issues, list) or any(
        not isinstance(issue, dict) for issue in issues
    ):
        raise SplitDataError("formal validation report issues must be a list of objects")
    if any(
        isinstance(issue, dict) and issue.get("severity") in {"error", "blocker"}
        for issue in issues
    ):
        raise SplitDataError("formal validation report contains an error or blocker issue")
    hashes = report.get("input_sha256")
    if not isinstance(hashes, dict):
        raise SplitDataError("formal validation report input_sha256 must be an object")
    required_fields = {
        "manifest",
        "action_labels",
        "event_labels",
        "risk_labels",
        "subject_profiles",
        "review_log",
        "validation_config",
    }
    if set(expected_input_sha256) != required_fields:
        raise SplitDataError("internal formal validation input set is incomplete")
    for field in sorted(required_fields):
        expected = _normalize_sha256(expected_input_sha256[field])
        if expected is None or hashes.get(field) != expected:
            raise SplitDataError(
                f"validation report input SHA-256 mismatch: {field}"
            )
    return _VerifiedValidationReport(
        _bytes_sha256(payload),
        manifest_file_sha256=expected_input_sha256["manifest"],
        manifest_canonical_sha256=manifest_canonical_sha256,
        labels_canonical_sha256=labels_canonical_sha256,
        _creation_token=_ATTESTATION_CREATION_TOKEN,
    )


def _write_atomic(path: Path, payload: str) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _render_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ) + "\n"


def _render_jsonl(rows: Iterable[Mapping[str, Any]]) -> str:
    return "".join(_render_json(row) for row in rows)


def _rows_sha256(rows: Iterable[Mapping[str, Any]]) -> str:
    rendered = sorted(_canonical_bytes(dict(row)) for row in rows)
    return _bytes_sha256(b"".join(line + b"\n" for line in rendered))


def _value_sha256(value: Any) -> str:
    return _bytes_sha256(_canonical_bytes(value))


def _bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SplitDataError(f"split inputs must contain canonical JSON values: {exc}") from exc


def _canonical_sort_key(value: Mapping[str, Any]) -> bytes:
    return _canonical_bytes(dict(value))


def _assignment_sort_key(row: Mapping[str, Any]) -> tuple[int, str]:
    partition = str(row.get("partition", ""))
    try:
        partition_index = PARTITION_ORDER.index(partition)
    except ValueError:
        partition_index = len(PARTITION_ORDER)
    return partition_index, str(row.get("sample_id", ""))


def _label_asset_reference(label: Mapping[str, Any]) -> str | None:
    for field in ("asset_id", "video_id"):
        if _is_known_identifier(label.get(field)):
            return _identifier_text(label[field])
    return None


def _normalized_identifier(value: Any) -> str:
    return _identifier_text(value) if _is_known_identifier(value) else "unknown"


def _optional_identifier(value: Any) -> str | None:
    return _identifier_text(value) if _is_known_identifier(value) else None


def _identifier_text(value: Any) -> str:
    return str(value).strip()


def _normalize_sha256(value: Any) -> str | None:
    text = _identifier_text(value)
    if re.fullmatch(r"[0-9a-fA-F]{64}", text) is None:
        return None
    return text.lower()


def _is_known_identifier(value: Any) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() not in _UNKNOWN_IDENTIFIERS


def _field_values(value: Any) -> Sequence[Any]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(value)
    return (value,)


class _UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}
        self.rank = {value: 0 for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1
