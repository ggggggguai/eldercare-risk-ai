"""Fail-closed access to the frozen step-4 wandering preprocessing bundle.

The bundle stores development and frozen WP test records in one JSONL file.
Every mode therefore parses all 1,790 records for integrity, but the public
views deliberately expose only train/validation or only frozen WP test rows.
No step-2 source path, especially SmartCare official validation, is accepted.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml


BUNDLE_MODE_DEVELOPMENT = "development"
BUNDLE_MODE_FROZEN_WP_TEST = "frozen_wp_test"
RF_CONFIG_SCHEMA_VERSION = "wandering-rf-config-v1"
PREPROCESSED_SAMPLE_SCHEMA_VERSION = "wandering-preprocessed-sample-v1"
PREPROCESSING_MANIFEST_SCHEMA_VERSION = "wandering-preprocessing-manifest-v1"
FEATURE_STATS_SCHEMA_VERSION = "wandering-feature-stats-v1"
PREPROCESSING_REPORT_SCHEMA_VERSION = "wandering-preprocessing-report-v1"
ASSIGNMENT_SCHEMA_VERSION = "wandering-split-assignments-v1"
NEAR_NEIGHBOR_SCHEMA_VERSION = "wandering-near-neighbor-audit-v1"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_INPUT_ROLES = (
    "preprocessing_config",
    "preprocessing_samples",
    "preprocessing_feature_stats",
    "preprocessing_report",
    "preprocessing_manifest",
    "split_json",
    "split_sha256_file",
    "assignments",
    "split_config",
    "near_neighbor_audit",
    "human_review",
)
_EXPECTED_HASHES = {
    "preprocessing_config": "5b69243337cddeeec8beed4f081330b06c47eadeaee06dc3b3e4ba026f36ff45",
    "preprocessing_samples": "323ec1258a03ed2daab1e84e3291edc62644c68b1c292dc93a332c01faa08cda",
    "preprocessing_feature_stats": "249ad8383043dba1d95452533ddc6004dfecc2c9e99be43f788dcfcbfb434c8d",
    "preprocessing_report": "d600fbd16efdc8c2b6e89db8c35c965fbb8322f099c464e67600abff5d8804e8",
    "preprocessing_manifest": "242072bdfe4b969d320a091ecc499aff445c30a937dfb1e6739ad869aa938593",
    "split_json": "7fa934c8041f538ff033260d722c21f5a4dc0e6cd8d23fa42836aa2e24ed808f",
    "split_sha256_file": "426175e2c5a7f822706f91445b2bd80c37704fb4b9943891a5fcfcd1b1efb96f",
    "assignments": "993307c13cb30484dba9fa1359c9c4fce714bb45b7d776e9a36ac6dcbee6eac6",
    "split_config": "579ad16e13b72f2914c5a2d14dfa73ef4a10a9a50f7ecc13bc20a67380a2e8ca",
    "near_neighbor_audit": "72d6f9f10f092fcadcd81f71e424d51f3b26938d6bfccbdcecced5852f851d10",
    "human_review": "f017f207053fb8948e217bd5b72d6bdf13d19a31ccd635f48b1f48915d60aa8c",
}
_EXPECTED_PATHS = {
    "preprocessing_config": "configs/data/wandering_preprocessing_v1.yaml",
    "preprocessing_samples": "data/processed/wandering/preprocessing/v1/samples.jsonl",
    "preprocessing_feature_stats": "data/processed/wandering/preprocessing/v1/feature_stats.json",
    "preprocessing_report": "data/processed/wandering/preprocessing/v1/preprocessing_report.json",
    "preprocessing_manifest": "data/processed/wandering/preprocessing/v1/manifest.json",
    "split_json": "data/splits/mental_health/wandering/v1/split.json",
    "split_sha256_file": "data/splits/mental_health/wandering/v1/split.sha256",
    "assignments": "data/splits/mental_health/wandering/v1/assignments.jsonl",
    "split_config": "configs/data/wandering_split_v1.yaml",
    "near_neighbor_audit": "data/splits/mental_health/wandering/v1/near_neighbor_audit.json",
    "human_review": "reports/mental_health/wandering_step4/HUMAN_REVIEW.md",
}
_EXPECTED_SPLIT_SHA256 = "4ac4a3877a056809066562cb09e4d30aa1d38baafcb4600f1f8a8a672776adbf"
_EXPECTED_CONFIG_FIELDS = frozenset(
    {
        "schema_version",
        "purpose",
        "inputs",
        "split_sha256",
        "feature_schema",
        "feature_count",
        "epsilon",
        "quantile_method",
        "std_ddof",
        "revisit_index_gap",
        "nearest_far_distance_clip",
        "tasks",
        "random_forest",
        "seeds",
        "primary_seed",
        "permutation_importance",
        "runtime_benchmark",
        "expected_counts",
        "output_schemas",
    }
)
_COMMON_RECORD_FIELDS = frozenset(
    {
        "schema_version",
        "sample_id",
        "parent_sample_id",
        "source_dataset",
        "input_role",
        "split",
        "binary_label",
        "pattern_label",
        "binary_supervision_eligible",
        "pattern_supervision_eligible",
        "input_coordinate_system",
        "input_point_count",
        "split_sha256",
        "preprocessing_config_sha256",
        "preprocess_status",
        "reason_codes",
        "valid_input_point_count",
        "trimmed_edge_point_count",
        "interpolated_input_point_count",
        "resample_mode",
        "resampled_source_points",
        "image_normalized_points",
        "shape_normalized_points",
        "point_mask",
        "raw_features",
        "model_features",
        "topology",
    }
)
_TOPOLOGY_FIELDS = frozenset(
    {
        "abs_curvature",
        "absolute_winding",
        "max_abs_curvature",
        "reversal_events",
        "revisit_pair_count",
        "selected_revisit_links",
    }
)
_RF_BASE_PARAMS = {
    "n_estimators": 500,
    "criterion": "gini",
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "bootstrap": True,
    "class_weight": None,
    "n_jobs": 1,
    "oob_score": False,
    "max_samples": None,
}


class BundleIntegrityError(ValueError):
    """A frozen hash, schema, record, split, or human gate has drifted."""


class BundleAccessError(BundleIntegrityError):
    """A caller requested a partition forbidden in the selected phase."""


@dataclass(frozen=True)
class WanderingPreprocessingBundle:
    """Validated records with a phase-limited public view."""

    mode: str
    _records: tuple[dict[str, Any], ...]
    input_hashes: Mapping[str, str]
    split_sha256: str
    preprocessing_config_sha256: str
    integrity_report: Mapping[str, Any]

    @property
    def total_record_count(self) -> int:
        return len(self._records)

    @property
    def ready_count(self) -> int:
        return sum(row["preprocess_status"] == "ready" for row in self._records)

    @property
    def unavailable_count(self) -> int:
        return sum(row["preprocess_status"] == "unavailable" for row in self._records)

    def records_for_features(self) -> tuple[dict[str, Any], ...]:
        allowed = (
            {"train", "validation"}
            if self.mode == BUNDLE_MODE_DEVELOPMENT
            else {"test"}
        )
        return tuple(
            row
            for row in self._records
            if row["preprocess_status"] == "ready" and row["split"] in allowed
        )

    def unavailable_records(self) -> tuple[dict[str, Any], ...]:
        if self.mode != BUNDLE_MODE_DEVELOPMENT:
            return ()
        return tuple(
            row
            for row in self._records
            if row["preprocess_status"] == "unavailable"
            and row["split"] in {"train", "validation"}
        )

    def records_for_split(self, split: str) -> tuple[dict[str, Any], ...]:
        if self.mode == BUNDLE_MODE_DEVELOPMENT and split not in {"train", "validation"}:
            raise BundleAccessError("development mode cannot expose test or sealed records")
        if self.mode == BUNDLE_MODE_FROZEN_WP_TEST and split != "test":
            raise BundleAccessError("frozen_wp_test mode can expose only WP test")
        return tuple(row for row in self.records_for_features() if row["split"] == split)

    def task_records(self, task: str) -> tuple[dict[str, Any], ...]:
        rows = self.records_for_features()
        if task == "four_class":
            return tuple(
                row
                for row in rows
                if row["source_dataset"] == "wandering_patterns"
                and row["pattern_supervision_eligible"] is True
            )
        if task == "binary":
            return tuple(row for row in rows if row["binary_supervision_eligible"] is True)
        raise BundleAccessError(f"unknown RF task: {task!r}")


def load_rf_config(path: str | Path) -> dict[str, Any]:
    """Load the exact step-5 configuration; silent extensions are rejected."""

    config_path = Path(path)
    try:
        value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise BundleIntegrityError(f"cannot read RF config: {config_path}") from exc
    if not isinstance(value, dict) or frozenset(value) != _EXPECTED_CONFIG_FIELDS:
        raise BundleIntegrityError("RF config fields have drifted")
    if value["schema_version"] != RF_CONFIG_SCHEMA_VERSION or value["purpose"] != "comparison_only":
        raise BundleIntegrityError("RF config schema/purpose has drifted")
    inputs = value["inputs"]
    if not isinstance(inputs, dict) or tuple(inputs) != _INPUT_ROLES:
        raise BundleIntegrityError("RF config input roles/order has drifted")
    for role in _INPUT_ROLES:
        descriptor = inputs[role]
        expected_fields = {"path", "sha256", "required_status"} if role == "human_review" else {"path", "sha256"}
        if not isinstance(descriptor, dict) or set(descriptor) != expected_fields:
            raise BundleIntegrityError(f"RF input descriptor has drifted: {role}")
        if descriptor["path"] != _EXPECTED_PATHS[role] or descriptor["sha256"] != _EXPECTED_HASHES[role]:
            raise BundleIntegrityError(f"RF input binding has drifted: {role}")
    if inputs["human_review"]["required_status"] != "human_review_passed":
        raise BundleIntegrityError("human review required status has drifted")
    if value["split_sha256"] != _EXPECTED_SPLIT_SHA256:
        raise BundleIntegrityError("canonical split SHA-256 has drifted")
    expected_scalars = {
        "feature_schema": "wandering-handcrafted-features-v1",
        "feature_count": 26,
        "epsilon": 1e-8,
        "quantile_method": "linear",
        "std_ddof": 0,
        "revisit_index_gap": 8,
        "nearest_far_distance_clip": 0.10,
        "primary_seed": 20260731,
    }
    for name, expected in expected_scalars.items():
        if value[name] != expected:
            raise BundleIntegrityError(f"RF config {name} has drifted")
    if value["tasks"] != {
        "four_class": {"class_names": ["direct", "pacing", "lapping", "random"], "threshold": "argmax"},
        "binary": {"class_names": ["direct_or_non_wandering", "wandering_like"], "threshold": 0.5, "weighting": "equal_source_x_class_total"},
    }:
        raise BundleIntegrityError("RF task contract has drifted")
    if value["random_forest"] != _RF_BASE_PARAMS:
        raise BundleIntegrityError("RF parameters have drifted")
    if value["seeds"] != [20260731, 20260801, 20260802, 20260803, 20260804]:
        raise BundleIntegrityError("RF seed set has drifted")
    if value["permutation_importance"] != {"n_repeats": 20, "scoring": "macro_f1"}:
        raise BundleIntegrityError("permutation importance contract has drifted")
    if value["runtime_benchmark"] != {"warmup_iterations": 20, "measurement_iterations": 200}:
        raise BundleIntegrityError("runtime benchmark contract has drifted")
    if value["expected_counts"] != {
        "total_records": 1790,
        "ready": 1775,
        "unavailable": 15,
        "ready_by_split": {"train": 1257, "validation": 278, "test": 240},
        "four_class": {"train": 1120, "validation": 240, "test": 240},
        "binary": {"train": 1257, "validation": 278, "test": 240},
    }:
        raise BundleIntegrityError("RF expected counts have drifted")
    if value["output_schemas"] != {
        "feature_table": "wandering-rf-feature-table-row-v1",
        "data_index": "wandering-rf-data-index-v1",
        "development_manifest": "wandering-rf-development-manifest-v1",
        "public_shape_benchmark_manifest": "wandering-rf-public-shape-benchmark-manifest-v1",
    }:
        raise BundleIntegrityError("RF output schemas have drifted")
    return value


def load_preprocessing_bundle(
    *,
    rf_config_path: str | Path,
    project_root: str | Path,
    mode: str,
) -> WanderingPreprocessingBundle:
    """Verify every frozen input and return a phase-limited bundle view."""

    if mode not in {BUNDLE_MODE_DEVELOPMENT, BUNDLE_MODE_FROZEN_WP_TEST}:
        raise BundleAccessError(f"unsupported bundle mode: {mode!r}")
    config = load_rf_config(rf_config_path)
    root = Path(project_root).resolve(strict=True)
    if not root.is_dir():
        raise BundleIntegrityError("project_root must be an existing directory")
    paths, hashes = _verify_bound_files(root, config["inputs"])
    _validate_human_review(paths["human_review"], required_status="human_review_passed")
    manifest = _load_canonical_json(paths["preprocessing_manifest"])
    stats = _load_canonical_json(paths["preprocessing_feature_stats"])
    report = _load_canonical_json(paths["preprocessing_report"])
    split = _load_canonical_json(paths["split_json"])
    audit = _load_canonical_json(paths["near_neighbor_audit"])
    _validate_machine_artifact_binding(manifest, hashes, paths)
    _validate_bundle_sidecars(stats, report, split, audit, hashes)
    assignment_rows = _load_canonical_jsonl(paths["assignments"])
    records = _load_canonical_jsonl(paths["preprocessing_samples"])
    _validate_assignments_and_records(
        assignments=assignment_rows,
        records=records,
        split=split,
        expected_preprocessing_config_sha256=hashes["preprocessing_config"],
    )
    return WanderingPreprocessingBundle(
        mode=mode,
        _records=tuple(records),
        input_hashes=dict(hashes),
        split_sha256=_EXPECTED_SPLIT_SHA256,
        preprocessing_config_sha256=hashes["preprocessing_config"],
        integrity_report={
            "all_frozen_hashes_verified": True,
            "manifest_binding_verified": True,
            "human_review_status": "human_review_passed",
            "near_neighbor_cross_partition_lt_0_05": 4054,
            "official_source_path_present": False,
            "total_records": len(records),
        },
    )


def _verify_bound_files(
    root: Path,
    descriptors: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Path], dict[str, str]]:
    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for role in _INPUT_ROLES:
        descriptor = descriptors[role]
        path = (root / Path(descriptor["path"])).resolve(strict=False)
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise BundleIntegrityError(f"RF input escapes project root: {role}") from exc
        if not path.is_file():
            raise BundleIntegrityError(f"frozen RF input is missing: {role}")
        digest = _sha256_file(path)
        if digest != descriptor["sha256"]:
            raise BundleIntegrityError(f"frozen RF input SHA-256 drift: {role}")
        paths[role] = path
        hashes[role] = digest
    return paths, hashes


def _validate_human_review(path: Path, *, required_status: str) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise BundleIntegrityError("cannot read step-4 human review") from exc
    match = re.search(r"^状态：`([^`]+)`\s*$", text, flags=re.MULTILINE)
    if match is None or match.group(1) != required_status:
        raise BundleIntegrityError("step-4 human review gate is not passed")


def _validate_machine_artifact_binding(
    manifest: Mapping[str, Any],
    hashes: Mapping[str, str],
    paths: Mapping[str, Path],
) -> None:
    if set(manifest) != {
        "artifacts", "generation_order", "input_hashes", "manifest_self_hash_embedded",
        "preprocessing_config_sha256", "schema_version", "split_sha256",
    } or manifest.get("schema_version") != PREPROCESSING_MANIFEST_SCHEMA_VERSION:
        raise BundleIntegrityError("step-4 manifest schema/fields have drifted")
    if manifest.get("preprocessing_config_sha256") != hashes["preprocessing_config"]:
        raise BundleIntegrityError("step-4 manifest preprocessing config binding drifted")
    if manifest.get("split_sha256") != _EXPECTED_SPLIT_SHA256:
        raise BundleIntegrityError("step-4 manifest split binding drifted")
    expected_artifacts = {
        "samples.jsonl": "preprocessing_samples",
        "feature_stats.json": "preprocessing_feature_stats",
        "preprocessing_report.json": "preprocessing_report",
    }
    if set(manifest.get("artifacts", {})) != set(expected_artifacts):
        raise BundleIntegrityError("step-4 manifest artifact set drifted")
    for name, role in expected_artifacts.items():
        descriptor = manifest["artifacts"][name]
        if descriptor != {"byte_count": paths[role].stat().st_size, "sha256": hashes[role]}:
            raise BundleIntegrityError(f"step-4 manifest artifact binding drifted: {name}")
    if manifest.get("generation_order") != [
        "samples.jsonl", "feature_stats.json", "preprocessing_report.json", "manifest.json"
    ] or manifest.get("manifest_self_hash_embedded") is not False:
        raise BundleIntegrityError("step-4 manifest generation contract drifted")


def _validate_bundle_sidecars(
    stats: Mapping[str, Any],
    report: Mapping[str, Any],
    split: Mapping[str, Any],
    audit: Mapping[str, Any],
    hashes: Mapping[str, str],
) -> None:
    if stats.get("schema_version") != FEATURE_STATS_SCHEMA_VERSION:
        raise BundleIntegrityError("feature_stats schema drifted")
    if stats.get("ready_train_sample_count") != 1257 or stats.get("valid_position_count") != 100560:
        raise BundleIntegrityError("feature_stats counts drifted")
    if stats.get("temporal_features_enabled") is not False:
        raise BundleIntegrityError("feature_stats temporal contract drifted")
    if report.get("schema_version") != PREPROCESSING_REPORT_SCHEMA_VERSION:
        raise BundleIntegrityError("preprocessing report schema drifted")
    expected_counts = {
        "nonsealed_processed": 1790,
        "original_by_split": {"test": 240, "train": 1272, "validation": 278},
        "ready": 1775,
        "ready_by_split": {"test": 240, "train": 1257, "validation": 278},
        "sealed_excluded": 20,
        "source_status": {"smartcare": {"ready": 175, "unavailable": 15}, "wandering_patterns": {"ready": 1600, "unavailable": 0}},
        "unavailable": 15,
        "unavailable_reason_codes": {"too_few_valid_points": 15},
    }
    if report.get("counts") != expected_counts:
        raise BundleIntegrityError("preprocessing report counts drifted")
    if report.get("preprocessing_config_sha256") != hashes["preprocessing_config"] or report.get("split_sha256") != _EXPECTED_SPLIT_SHA256:
        raise BundleIntegrityError("preprocessing report bindings drifted")
    if split.get("split_version") != "wandering-split-v1" or split.get("split_sha256") != _EXPECTED_SPLIT_SHA256:
        raise BundleIntegrityError("canonical split schema/hash drifted")
    if len(split.get("train", [])) != 1272 or len(split.get("validation", [])) != 278 or len(split.get("test", [])) != 240 or len(split.get("sealed_external_test", [])) != 20:
        raise BundleIntegrityError("canonical split counts drifted")
    if audit.get("schema_version") != NEAR_NEIGHBOR_SCHEMA_VERSION:
        raise BundleIntegrityError("near-neighbor audit schema drifted")
    if audit.get("official_validation_shape_audit_performed") is not False:
        raise BundleIntegrityError("official validation entered shape audit")
    near = audit.get("wandering_patterns", {}).get("near_neighbors", {}).get("lt_0_05", {})
    if near.get("cross_partition_pairs") != 4054 or audit.get("wandering_patterns", {}).get("test_scope") != "public_shape_benchmark":
        raise BundleIntegrityError("near-neighbor audit public benchmark contract drifted")


def _validate_assignments_and_records(
    *,
    assignments: Sequence[Mapping[str, Any]],
    records: Sequence[dict[str, Any]],
    split: Mapping[str, Any],
    expected_preprocessing_config_sha256: str,
) -> None:
    if len(assignments) != 1810:
        raise BundleIntegrityError("assignments must contain exactly 1810 rows")
    assignment_ids = [row.get("sample_id") for row in assignments]
    if len(set(assignment_ids)) != 1810 or assignment_ids != sorted(assignment_ids):
        raise BundleIntegrityError("assignments IDs must be unique and sorted")
    assignment_index: dict[str, Mapping[str, Any]] = {}
    for row in assignments:
        if row.get("schema_version") != ASSIGNMENT_SCHEMA_VERSION:
            raise BundleIntegrityError("assignment schema drifted")
        assignment_index[str(row["sample_id"])] = row
    sealed = set(split["sealed_external_test"])
    development_ids = set(split["train"]) | set(split["validation"]) | set(split["test"])
    record_ids = [row.get("sample_id") for row in records]
    if len(records) != 1790 or len(set(record_ids)) != 1790 or record_ids != sorted(record_ids):
        raise BundleIntegrityError("preprocessing records must be 1790 unique sorted IDs")
    if set(record_ids) != development_ids or set(record_ids) & sealed:
        raise BundleIntegrityError("preprocessing record IDs do not equal non-sealed split IDs")
    for row in records:
        _validate_preprocessed_record(
            row,
            expected_split_sha256=_EXPECTED_SPLIT_SHA256,
            expected_preprocessing_config_sha256=expected_preprocessing_config_sha256,
        )
        assignment = assignment_index[row["sample_id"]]
        for field in (
            "split", "input_role", "binary_label", "pattern_label",
            "binary_supervision_eligible", "pattern_supervision_eligible",
        ):
            if row[field] != assignment.get(field):
                raise BundleIntegrityError(f"record/assignment {field} mismatch: {row['sample_id']}")
        if row["source_dataset"] != assignment.get("source_name"):
            raise BundleIntegrityError(f"record/assignment source mismatch: {row['sample_id']}")
    statuses = Counter(row["preprocess_status"] for row in records)
    ready_splits = Counter(row["split"] for row in records if row["preprocess_status"] == "ready")
    if statuses != Counter(ready=1775, unavailable=15) or ready_splits != Counter(train=1257, validation=278, test=240):
        raise BundleIntegrityError("preprocessing record status/split counts drifted")
    unavailable = [row for row in records if row["preprocess_status"] == "unavailable"]
    if any(row["source_dataset"] != "smartcare" or row["split"] != "train" or row["reason_codes"] != ["too_few_valid_points"] for row in unavailable):
        raise BundleIntegrityError("the 15 unavailable records drifted")
    _validate_task_counts(records)


def _validate_preprocessed_record(
    row: Mapping[str, Any],
    *,
    expected_split_sha256: str,
    expected_preprocessing_config_sha256: str,
) -> None:
    if not isinstance(row, Mapping) or frozenset(row) != _COMMON_RECORD_FIELDS:
        raise BundleIntegrityError("preprocessed record fields have drifted")
    sample_id = row.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id or row.get("parent_sample_id") != sample_id:
        raise BundleIntegrityError("preprocessed sample/parent ID is invalid")
    if row.get("schema_version") != PREPROCESSED_SAMPLE_SCHEMA_VERSION:
        raise BundleIntegrityError("preprocessed sample schema drifted")
    if row.get("split") not in {"train", "validation", "test"}:
        raise BundleIntegrityError("preprocessed sample split is invalid")
    if row.get("split_sha256") != expected_split_sha256 or row.get("preprocessing_config_sha256") != expected_preprocessing_config_sha256:
        raise BundleIntegrityError("preprocessed sample frozen hash binding drifted")
    source = row.get("source_dataset")
    if source == "wandering_patterns":
        if row.get("input_role") != "samples" or row.get("input_coordinate_system") != "source_native":
            raise BundleIntegrityError("WP source/coordinate contract drifted")
        if row.get("pattern_label") not in {"direct", "pacing", "lapping", "random"}:
            raise BundleIntegrityError("WP pattern label drifted")
        expected_binary = 0 if row["pattern_label"] == "direct" else 1
        if row.get("binary_label") != expected_binary or row.get("binary_supervision_eligible") is not True or row.get("pattern_supervision_eligible") is not True:
            raise BundleIntegrityError("WP supervision contract drifted")
    elif source == "smartcare":
        if row.get("input_role") != "train_pool" or row.get("input_coordinate_system") != "image_normalized":
            raise BundleIntegrityError("SmartCare source/coordinate contract drifted")
        if row.get("pattern_label") != "unknown" or row.get("binary_label") not in {0, 1}:
            raise BundleIntegrityError("SmartCare label contract drifted")
        if row.get("binary_supervision_eligible") is not True or row.get("pattern_supervision_eligible") is not False:
            raise BundleIntegrityError("SmartCare supervision contract drifted")
        if row.get("split") == "test":
            raise BundleIntegrityError("SmartCare must not enter frozen WP test")
    else:
        raise BundleIntegrityError("unknown preprocessed source_dataset")
    for field in ("input_point_count", "valid_input_point_count", "trimmed_edge_point_count", "interpolated_input_point_count"):
        value = row.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise BundleIntegrityError(f"invalid nonnegative integer field: {field}")
    status = row.get("preprocess_status")
    if status == "unavailable":
        nullable = (
            "resample_mode", "resampled_source_points", "image_normalized_points",
            "shape_normalized_points", "point_mask", "raw_features", "model_features", "topology",
        )
        if any(row.get(field) is not None for field in nullable) or row.get("reason_codes") != ["too_few_valid_points"]:
            raise BundleIntegrityError("unavailable payload contract drifted")
        return
    if (
        status != "ready"
        or row.get("reason_codes") != []
        or row.get("resample_mode") != "cumulative_arc_length_linear"
    ):
        raise BundleIntegrityError("ready status/reason/resample contract drifted")
    _finite_array(row.get("resampled_source_points"), (80, 2), "resampled_source_points")
    _finite_array(row.get("shape_normalized_points"), (80, 2), "shape_normalized_points")
    mask = _finite_array(row.get("point_mask"), (80,), "point_mask")
    if not np.array_equal(mask, np.ones(80, dtype=np.float64)):
        raise BundleIntegrityError("ready point_mask must be 80 ones")
    raw = _finite_array(row.get("raw_features"), (80, 14), "raw_features")
    model = _finite_array(row.get("model_features"), (80, 14), "model_features")
    if not np.array_equal(raw[:, 12], mask) or not np.array_equal(model[:, 12], mask):
        raise BundleIntegrityError("feature mask channel drifted")
    if not np.array_equal(raw[:, 10:12], np.zeros((80, 2))) or not np.array_equal(model[:, 10:12], np.zeros((80, 2))):
        raise BundleIntegrityError("public temporal channels must remain zero")
    if source == "wandering_patterns":
        if row.get("image_normalized_points") is not None:
            raise BundleIntegrityError("WP must not fabricate image-normalized points")
    else:
        image = _finite_array(row.get("image_normalized_points"), (80, 2), "image_normalized_points")
        if np.any(image < 0.0) or np.any(image > 1.0):
            raise BundleIntegrityError("SmartCare image-normalized points left [0,1]")
    _validate_topology(row.get("topology"))


def _validate_topology(value: Any) -> None:
    if not isinstance(value, Mapping) or frozenset(value) != _TOPOLOGY_FIELDS:
        raise BundleIntegrityError("topology fields drifted")
    curvature = _finite_array(value.get("abs_curvature"), (80,), "topology.abs_curvature")
    for field in ("absolute_winding", "max_abs_curvature"):
        scalar = value.get(field)
        if isinstance(scalar, bool) or not isinstance(scalar, (int, float)) or not math.isfinite(float(scalar)) or float(scalar) < 0.0:
            raise BundleIntegrityError(f"invalid topology scalar: {field}")
    if not math.isclose(float(np.max(curvature)), float(value["max_abs_curvature"]), rel_tol=0.0, abs_tol=1e-12):
        raise BundleIntegrityError("topology max curvature drifted")
    count = value.get("revisit_pair_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise BundleIntegrityError("invalid revisit_pair_count")
    events = value.get("reversal_events")
    links = value.get("selected_revisit_links")
    if not isinstance(events, list) or not isinstance(links, list) or len(links) > 5:
        raise BundleIntegrityError("invalid topology event/link arrays")
    for event in events:
        if set(event) != {"index", "angle_deg"} or not isinstance(event["index"], int) or not _finite_number(event["angle_deg"]):
            raise BundleIntegrityError("invalid reversal event")
    for link in links:
        if set(link) != {"distance", "i", "j"} or not isinstance(link["i"], int) or not isinstance(link["j"], int) or not _finite_number(link["distance"]):
            raise BundleIntegrityError("invalid revisit link")


def _validate_task_counts(records: Sequence[Mapping[str, Any]]) -> None:
    ready = [row for row in records if row["preprocess_status"] == "ready"]
    four = [row for row in ready if row["pattern_supervision_eligible"]]
    binary = [row for row in ready if row["binary_supervision_eligible"]]
    if Counter(row["split"] for row in four) != Counter(train=1120, validation=240, test=240):
        raise BundleIntegrityError("four-class cohort counts drifted")
    if Counter((row["split"], row["pattern_label"]) for row in four) != Counter({
        **{("train", label): 280 for label in ("direct", "pacing", "lapping", "random")},
        **{("validation", label): 60 for label in ("direct", "pacing", "lapping", "random")},
        **{("test", label): 60 for label in ("direct", "pacing", "lapping", "random")},
    }):
        raise BundleIntegrityError("four-class label counts drifted")
    if Counter(row["split"] for row in binary) != Counter(train=1257, validation=278, test=240):
        raise BundleIntegrityError("binary cohort counts drifted")
    expected = Counter({
        ("train", "wandering_patterns", 0): 280,
        ("train", "wandering_patterns", 1): 840,
        ("train", "smartcare", 0): 63,
        ("train", "smartcare", 1): 74,
        ("validation", "wandering_patterns", 0): 60,
        ("validation", "wandering_patterns", 1): 180,
        ("validation", "smartcare", 0): 17,
        ("validation", "smartcare", 1): 21,
        ("test", "wandering_patterns", 0): 60,
        ("test", "wandering_patterns", 1): 180,
    })
    if Counter((row["split"], row["source_dataset"], row["binary_label"]) for row in binary) != expected:
        raise BundleIntegrityError("binary source/label counts drifted")


def _finite_array(value: Any, shape: tuple[int, ...], field: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise BundleIntegrityError(f"{field} must be numeric") from exc
    if array.shape != shape or not np.isfinite(array).all():
        raise BundleIntegrityError(f"{field} must be finite shape {shape}")
    return array


def _finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def _load_canonical_json(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BundleIntegrityError(f"cannot load canonical JSON: {path}") from exc
    if not isinstance(value, dict) or raw != _canonical_json_bytes(value):
        raise BundleIntegrityError(f"JSON is not canonical: {path}")
    return value


def _load_canonical_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("rb") as handle:
            for line_no, raw in enumerate(handle, start=1):
                if not raw or raw == b"\n":
                    raise BundleIntegrityError(f"blank JSONL line: {path}:{line_no}")
                value = json.loads(raw.decode("utf-8"))
                if not isinstance(value, dict) or raw != _canonical_json_bytes(value):
                    raise BundleIntegrityError(f"non-canonical JSONL row: {path}:{line_no}")
                rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BundleIntegrityError(f"cannot load canonical JSONL: {path}") from exc
    return rows


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise BundleIntegrityError("value is not finite canonical JSON") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _commit_new_output_directory(output_dir: str | Path, files: Mapping[str, bytes]) -> None:
    """Write a complete directory to a sibling temp path, then rename once."""

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent))
    try:
        for relative_name, payload in files.items():
            if not isinstance(relative_name, str) or not isinstance(payload, bytes):
                raise TypeError("atomic output files must map relative names to bytes")
            relative = Path(relative_name)
            if relative.is_absolute() or ".." in relative.parts:
                raise BundleIntegrityError("atomic output path escapes target directory")
            path = temp_dir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        os.replace(temp_dir, output)
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


__all__ = [
    "BUNDLE_MODE_DEVELOPMENT",
    "BUNDLE_MODE_FROZEN_WP_TEST",
    "BundleAccessError",
    "BundleIntegrityError",
    "WanderingPreprocessingBundle",
    "load_preprocessing_bundle",
    "load_rf_config",
]
