"""Manifest-bound M0-R candidate loading and phase-aware WP-only evaluation.

This module is deliberately separate from the M0-S training implementation.
It never trains, fits, selects a checkpoint, or searches a threshold.  The
only production test entry is the manifest-bound ``score-frozen-wp``
controller, which performs every preflight check before it opens the trusted
frozen-WP accessor.  Development parity never opens that accessor.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Mapping, Sequence
import zipfile

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from . import model as model_module
from . import preprocessing_bundle as preprocessing_bundle_module
from .model import TopoWanderMPT, safe_load_topowander_model
from .preprocessing_bundle import (
    BUNDLE_MODE_FROZEN_WP_TEST,
    load_preprocessing_bundle,
    load_rf_config,
)


TRAINING_IDENTITY_SCHEMA = "wandering-m0r-training-candidate-identity-v1"
RELEASE_IDENTITY_SCHEMA = "wandering-m0rh-release-implementation-identity-v3"
CANDIDATE_MANIFEST_SCHEMA = "wandering-m0rh-scoring-candidate-manifest-v3"
FIXED_COHORT_WP_PREFIX_PADDING = 38
WP_METRICS_SCHEMA = "wandering-m0r-wp-only-metrics-v1"
WP_PREDICTION_SCHEMA = "wandering-m0r-wp-only-prediction-v1"
PATTERN_LABELS = ("direct", "pacing", "lapping", "random")
SUBTYPE_LABELS = ("pacing", "lapping", "random")
BINARY_LABELS = ("direct_or_non_wandering", "wandering_like")
ALLOWED_PHASES = ("validation", "test")
PRIMARY_SEED = 20260731
PRIMARY_BEST_EPOCH = 5
_SHA256_CHARS = frozenset("0123456789abcdef")
FROZEN_WP_RF_CONFIG = Path("configs/modules/wandering_rf_v1.yaml")
RELEASE_MODULE_SOURCE = Path(
    "src/elderly_monitoring/modules/mental_health/wandering/release.py"
)
RELEASE_CLI_SOURCE = Path("scripts/wandering/release_wandering_candidate.py")
MODEL_SOURCE = Path("src/elderly_monitoring/modules/mental_health/wandering/model.py")
PERFORMANCE_SOURCE = Path(
    "src/elderly_monitoring/modules/mental_health/wandering/performance.py"
)
PREPROCESSING_BUNDLE_SOURCE = Path(
    "src/elderly_monitoring/modules/mental_health/wandering/preprocessing_bundle.py"
)
RELEASE_TEST_SOURCE = Path("tests/test_wandering_release.py")
RELEASE_IMPLEMENTATION_SOURCE_FILES = (
    FROZEN_WP_RF_CONFIG,
    RELEASE_CLI_SOURCE,
    MODEL_SOURCE,
    PERFORMANCE_SOURCE,
    PREPROCESSING_BUNDLE_SOURCE,
    RELEASE_MODULE_SOURCE,
    RELEASE_TEST_SOURCE,
)
ACTIVE_EXECUTION_SOURCE_FILES = (
    FROZEN_WP_RF_CONFIG,
    RELEASE_CLI_SOURCE,
    MODEL_SOURCE,
    PERFORMANCE_SOURCE,
    PREPROCESSING_BUNDLE_SOURCE,
    RELEASE_MODULE_SOURCE,
)
FIXED_CPU_RUNTIME = {
    "device": "cpu",
    "intra_op_threads": 8,
    "inter_op_threads": 1,
    "batch_size": 64,
    "num_workers": 0,
    "pin_memory": False,
}
FROZEN_WP_INPUT_ROLES = (
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


class CandidateManifestError(ValueError):
    """A candidate manifest, trusted digest, or bound artifact is invalid."""


class ReleaseArtifactError(ValueError):
    """A release identity, source archive, or compact bundle is invalid."""


class WPReleaseDataError(ValueError):
    """A phase-aware WP record, prediction, metric, or parity input is invalid."""


@dataclass(frozen=True)
class ValidatedWPRecord:
    sample_id: str
    split: str
    binary_label: int
    pattern_label_index: int
    pattern_label_name: str
    model_features: np.ndarray
    shape_normalized_points: np.ndarray
    point_mask: np.ndarray


@dataclass(frozen=True)
class WPEvaluationResult:
    metrics: Mapping[str, Any]
    predictions: tuple[dict[str, Any], ...]
    confusion: Mapping[str, Any]
    errors: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class FrozenReleaseIdentity:
    identity_path: Path
    archive_path: Path
    identity: Mapping[str, Any]
    identity_sha256: str


@dataclass(frozen=True)
class CandidateBundleResult:
    bundle_dir: Path
    manifest_path: Path
    manifest_sha256: str
    manifest: Mapping[str, Any]


@dataclass(frozen=True)
class CandidateRuntime:
    bundle_root: Path
    manifest_path: Path
    manifest_sha256: str
    manifest: Mapping[str, Any]
    model: TopoWanderMPT

    def predict_logits(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        expected_split: str,
        batch_size: int,
        require_fixed_cohort: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run the bound model on CPU under inference mode and preserve order."""

        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise WPReleaseDataError("batch_size must be a positive integer")
        validated = (
            validate_wp_release_cohort(records, expected_split=expected_split)
            if require_fixed_cohort
            else tuple(
                validate_wp_release_record(record, expected_split=expected_split)
                for record in records
            )
        )
        if not validated:
            raise WPReleaseDataError("candidate inference requires at least one WP record")
        if self.model.training:
            raise CandidateManifestError("candidate model must remain in eval mode")
        if any(parameter.device.type != "cpu" for parameter in self.model.parameters()):
            raise CandidateManifestError("candidate model must remain on CPU")
        prefix_padding = 0
        inference_records = validated
        contract = self.manifest["inference_contract"]
        runtime_contract = contract["runtime"]
        if require_fixed_cohort:
            if batch_size != runtime_contract["batch_size"]:
                raise WPReleaseDataError("fixed WP cohort inference batch size must be 64")
            prefix_padding = int(contract["fixed_cohort_wp_prefix_padding"])
            inference_records = (validated[0],) * prefix_padding + validated
        dataset = _WPReleaseDataset(inference_records)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=int(runtime_contract["num_workers"]),
            pin_memory=bool(runtime_contract["pin_memory"]),
            collate_fn=_collate_wp_release_batch,
        )
        binary_logits: list[torch.Tensor] = []
        subtype_logits: list[torch.Tensor] = []
        observed_ids: list[str] = []
        with torch.inference_mode():
            for batch in loader:
                outputs = self.model(
                    batch["model_features"],
                    batch["shape_normalized_points"],
                    batch["point_mask"],
                )
                binary_logits.append(outputs["binary_logit"].detach().cpu())
                subtype_logits.append(outputs["subtype_logits"].detach().cpu())
                observed_ids.extend(batch["sample_id"])
        expected_ids = [record.sample_id for record in inference_records]
        if observed_ids != expected_ids:
            raise WPReleaseDataError("candidate inference order drifted from validated records")
        binary = torch.cat(binary_logits, dim=0).numpy().astype(np.float64, copy=False)
        subtype = torch.cat(subtype_logits, dim=0).numpy().astype(np.float64, copy=False)
        if binary.ndim == 2 and binary.shape[1] == 1:
            binary = binary[:, 0]
        if prefix_padding:
            binary = binary[prefix_padding:]
            subtype = subtype[prefix_padding:]
        if binary.shape != (len(validated),) or subtype.shape != (len(validated), 3):
            raise WPReleaseDataError("candidate output shapes drifted")
        if not np.isfinite(binary).all() or not np.isfinite(subtype).all():
            raise WPReleaseDataError("candidate outputs must be finite")
        return np.ascontiguousarray(binary), np.ascontiguousarray(subtype)


class _WPReleaseDataset(Dataset[dict[str, Any]]):
    def __init__(self, records: Sequence[ValidatedWPRecord]) -> None:
        self.records = tuple(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        return {
            "sample_id": record.sample_id,
            "model_features": torch.from_numpy(record.model_features),
            "shape_normalized_points": torch.from_numpy(record.shape_normalized_points),
            "point_mask": torch.from_numpy(record.point_mask),
        }


def _collate_wp_release_batch(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise WPReleaseDataError("cannot collate an empty release batch")
    return {
        "sample_id": [str(sample["sample_id"]) for sample in samples],
        "model_features": torch.stack([sample["model_features"] for sample in samples]),
        "shape_normalized_points": torch.stack(
            [sample["shape_normalized_points"] for sample in samples]
        ),
        "point_mask": torch.stack([sample["point_mask"] for sample in samples]),
    }


def validate_wp_release_record(
    record: Mapping[str, Any], *, expected_split: str
) -> ValidatedWPRecord:
    """Validate one WP record for exactly the caller-declared phase."""

    _validate_phase(expected_split)
    if not isinstance(record, Mapping):
        raise WPReleaseDataError("WP release record must be a mapping")
    required = {
        "sample_id",
        "source_dataset",
        "split",
        "preprocess_status",
        "binary_label",
        "binary_supervision_eligible",
        "pattern_label",
        "pattern_supervision_eligible",
        "model_features",
        "shape_normalized_points",
        "point_mask",
    }
    if not required.issubset(record):
        raise WPReleaseDataError("WP release record fields are incomplete")
    sample_id = record["sample_id"]
    if not isinstance(sample_id, str) or not sample_id:
        raise WPReleaseDataError("sample_id must be non-empty")
    if record["source_dataset"] != "wandering_patterns":
        raise WPReleaseDataError("release evaluator accepts WanderingPatterns records only")
    if record["split"] != expected_split:
        raise WPReleaseDataError(
            f"record split does not match expected split {expected_split}; relabeling is forbidden"
        )
    if record["preprocess_status"] != "ready":
        raise WPReleaseDataError("WP release record must be ready")
    if record["binary_supervision_eligible"] is not True:
        raise WPReleaseDataError("WP release record must have binary supervision")
    if record["pattern_supervision_eligible"] is not True:
        raise WPReleaseDataError("WP release record must have pattern supervision")
    pattern_name = record["pattern_label"]
    if pattern_name not in PATTERN_LABELS:
        raise WPReleaseDataError("WP pattern label is invalid")
    binary_label = record["binary_label"]
    expected_binary = 0 if pattern_name == "direct" else 1
    if (
        not isinstance(binary_label, int)
        or isinstance(binary_label, bool)
        or binary_label != expected_binary
    ):
        raise WPReleaseDataError("WP binary and pattern labels are inconsistent")
    model_features = _finite_array(record["model_features"], (80, 14), "model_features")
    shape_points = _finite_array(
        record["shape_normalized_points"], (80, 2), "shape_normalized_points"
    )
    point_mask = _finite_array(record["point_mask"], (80,), "point_mask")
    if not np.all((point_mask == 0.0) | (point_mask == 1.0)) or point_mask.sum() < 8:
        raise WPReleaseDataError("point_mask must be binary with at least eight valid points")
    if not np.array_equal(model_features[:, 12], point_mask):
        raise WPReleaseDataError("model feature mask channel does not match point_mask")
    if np.any(model_features[:, 13] < 0.0) or np.any(model_features[:, 13] > 1.0):
        raise WPReleaseDataError("model feature quality channel must remain in [0,1]")
    return ValidatedWPRecord(
        sample_id=sample_id,
        split=expected_split,
        binary_label=binary_label,
        pattern_label_index=PATTERN_LABELS.index(pattern_name),
        pattern_label_name=pattern_name,
        model_features=model_features,
        shape_normalized_points=shape_points,
        point_mask=point_mask,
    )


def validate_wp_release_cohort(
    records: Sequence[Mapping[str, Any]], *, expected_split: str
) -> tuple[ValidatedWPRecord, ...]:
    """Validate the fixed 240-row, balanced four-shape WP cohort."""

    _validate_phase(expected_split)
    validated = tuple(
        validate_wp_release_record(record, expected_split=expected_split) for record in records
    )
    if len(validated) != 240:
        raise WPReleaseDataError("fixed WP release cohort must contain exactly 240 records")
    ids = [record.sample_id for record in validated]
    if len(ids) != len(set(ids)):
        raise WPReleaseDataError("fixed WP release cohort sample IDs must be unique")
    patterns = Counter(record.pattern_label_name for record in validated)
    if patterns != Counter({name: 60 for name in PATTERN_LABELS}):
        raise WPReleaseDataError("fixed WP release cohort must contain 60 records per pattern")
    binary = Counter(record.binary_label for record in validated)
    if binary != Counter({0: 60, 1: 180}):
        raise WPReleaseDataError("fixed WP release cohort binary counts must be 60/180")
    return validated


def canonical_wp_cohort_identity(
    records: Sequence[Mapping[str, Any]], *, expected_split: str
) -> dict[str, Any]:
    """Hash the exact accessor order and canonical record bytes after validation."""

    validated = validate_wp_release_cohort(records, expected_split=expected_split)
    ordered_ids = [record.sample_id for record in validated]
    pattern_counts = Counter(record.pattern_label_name for record in validated)
    binary_counts = Counter(record.binary_label for record in validated)
    return {
        "schema_version": "wandering-m0rh-frozen-wp-cohort-identity-v1",
        "phase": expected_split,
        "sample_count": len(validated),
        "canonical_records_sha256": hashlib.sha256(
            _canonical_jsonl_bytes(records)
        ).hexdigest(),
        "ordered_sample_ids_sha256": hashlib.sha256(
            _canonical_json_bytes(ordered_ids)
        ).hexdigest(),
        "order_contract": "accessor_records_for_split_order_preserved_without_resort",
        "sample_ids_unique": len(ordered_ids) == len(set(ordered_ids)),
        "pattern_counts": {
            name: int(pattern_counts[name]) for name in PATTERN_LABELS
        },
        "binary_counts": {
            str(label): int(binary_counts[label]) for label in (0, 1)
        },
    }


def evaluate_wp_predictions(
    records: Sequence[Mapping[str, Any]],
    binary_logits: np.ndarray | torch.Tensor | Sequence[float],
    subtype_logits: np.ndarray | torch.Tensor | Sequence[Sequence[float]],
    *,
    expected_split: str,
    require_fixed_cohort: bool = True,
) -> WPEvaluationResult:
    """Evaluate only WP binary and hierarchical four-class tasks."""

    validated = (
        validate_wp_release_cohort(records, expected_split=expected_split)
        if require_fixed_cohort
        else tuple(
            validate_wp_release_record(record, expected_split=expected_split)
            for record in records
        )
    )
    if not validated:
        raise WPReleaseDataError("WP evaluator requires at least one record")
    binary = _to_numpy(binary_logits)
    subtype = _to_numpy(subtype_logits)
    if binary.ndim == 2 and binary.shape[1] == 1:
        binary = binary[:, 0]
    if binary.shape != (len(validated),):
        raise WPReleaseDataError("binary logits shape must be [N] or [N,1]")
    if subtype.shape != (len(validated), 3):
        raise WPReleaseDataError("subtype logits shape must be [N,3]")
    if not np.isfinite(binary).all() or not np.isfinite(subtype).all():
        raise WPReleaseDataError("WP evaluator logits must be finite")

    binary_probability = 1.0 / (1.0 + np.exp(-np.clip(binary, -60.0, 60.0)))
    shifted = subtype - np.max(subtype, axis=1, keepdims=True)
    subtype_probability = np.exp(shifted)
    subtype_probability /= np.sum(subtype_probability, axis=1, keepdims=True)
    four_probability = np.concatenate(
        ((1.0 - binary_probability)[:, None], binary_probability[:, None] * subtype_probability),
        axis=1,
    )
    if not np.isfinite(four_probability).all() or not np.allclose(
        four_probability.sum(axis=1), 1.0, atol=1.0e-7
    ):
        raise WPReleaseDataError("hierarchical four-class probabilities are invalid")
    binary_predicted = (binary_probability >= 0.5).astype(np.int64)
    pattern_predicted = np.argmax(four_probability, axis=1).astype(np.int64)
    binary_true = np.asarray([record.binary_label for record in validated], dtype=np.int64)
    pattern_true = np.asarray(
        [record.pattern_label_index for record in validated], dtype=np.int64
    )
    wp_four = _classification_metrics(pattern_true, pattern_predicted, labels=PATTERN_LABELS)
    wp_binary = _classification_metrics(binary_true, binary_predicted, labels=BINARY_LABELS)
    predictions: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for index, record in enumerate(validated):
        row = {
            "schema_version": WP_PREDICTION_SCHEMA,
            "sample_id": record.sample_id,
            "source_dataset": "wandering_patterns",
            "split": expected_split,
            "true_binary_label": int(record.binary_label),
            "predicted_binary_label": int(binary_predicted[index]),
            "binary_probability": float(binary_probability[index]),
            "true_pattern_label": record.pattern_label_name,
            "predicted_pattern_label": PATTERN_LABELS[int(pattern_predicted[index])],
            "subtype_probabilities": {
                name: float(subtype_probability[index, subtype_index])
                for subtype_index, name in enumerate(SUBTYPE_LABELS)
            },
            "four_class_probabilities": {
                name: float(four_probability[index, pattern_index])
                for pattern_index, name in enumerate(PATTERN_LABELS)
            },
        }
        predictions.append(row)
        binary_correct = row["true_binary_label"] == row["predicted_binary_label"]
        pattern_correct = row["true_pattern_label"] == row["predicted_pattern_label"]
        if not binary_correct or not pattern_correct:
            errors.append(
                {
                    "sample_id": record.sample_id,
                    "binary_correct": binary_correct,
                    "four_class_correct": pattern_correct,
                    "true_binary_label": row["true_binary_label"],
                    "predicted_binary_label": row["predicted_binary_label"],
                    "true_pattern_label": row["true_pattern_label"],
                    "predicted_pattern_label": row["predicted_pattern_label"],
                }
            )
    metrics = {
        "schema_version": WP_METRICS_SCHEMA,
        "phase": expected_split,
        "sample_count": len(validated),
        "wp_four_class": wp_four,
        "wp_binary": wp_binary,
        "decision_contract": {
            "binary": "sigmoid >= 0.5",
            "four_class": "argmax hierarchical probability",
            "four_class_order": list(PATTERN_LABELS),
            "subtype_order": list(SUBTYPE_LABELS),
        },
    }
    confusion = {
        "schema_version": "wandering-m0r-wp-only-confusion-v1",
        "phase": expected_split,
        "wp_four_class": wp_four["confusion_matrix"],
        "wp_binary": wp_binary["confusion_matrix"],
    }
    return WPEvaluationResult(
        metrics=metrics,
        predictions=tuple(predictions),
        confusion=confusion,
        errors=tuple(errors),
    )


def compare_primary_validation_parity(
    candidate: WPEvaluationResult,
    *,
    reference_predictions: Sequence[Mapping[str, Any]],
    reference_wp_four_metrics: Mapping[str, Any],
    reference_wp_binary_metrics: Mapping[str, Any],
    tolerance: float,
) -> dict[str, Any]:
    """Compare identity, labels, probabilities, and WP-only metric leaves."""

    if not isinstance(tolerance, float) or tolerance <= 0.0 or not math.isfinite(tolerance):
        raise WPReleaseDataError("parity tolerance must be a positive finite float")
    if len(candidate.predictions) != len(reference_predictions):
        raise WPReleaseDataError("parity prediction counts differ")
    maximum_probability_difference = 0.0
    for index, (actual, reference) in enumerate(
        zip(candidate.predictions, reference_predictions, strict=True)
    ):
        if actual["sample_id"] != reference.get("sample_id"):
            raise WPReleaseDataError(f"parity sample identity differs at row {index}")
        if (
            actual["true_binary_label"] != reference.get("true_binary_label")
            or actual["true_pattern_label"] != reference.get("true_pattern_label")
        ):
            raise WPReleaseDataError(f"parity true label differs at row {index}")
        maximum_probability_difference = max(
            maximum_probability_difference,
            abs(float(actual["binary_probability"]) - float(reference["binary_probability"])),
        )
        for field, names in (
            ("subtype_probabilities", SUBTYPE_LABELS),
            ("four_class_probabilities", PATTERN_LABELS),
        ):
            actual_values = actual[field]
            reference_values = reference.get(field)
            if not isinstance(reference_values, Mapping) or set(reference_values) != set(names):
                raise WPReleaseDataError(f"parity reference {field} schema differs")
            for name in names:
                maximum_probability_difference = max(
                    maximum_probability_difference,
                    abs(float(actual_values[name]) - float(reference_values[name])),
                )
    four_difference = _metric_max_abs_difference(
        candidate.metrics["wp_four_class"], reference_wp_four_metrics, path="wp_four_class"
    )
    binary_difference = _metric_max_abs_difference(
        candidate.metrics["wp_binary"], reference_wp_binary_metrics, path="wp_binary"
    )
    maximum_metric_difference = max(four_difference, binary_difference)
    return {
        "schema_version": "wandering-m0r-primary-validation-parity-v1",
        "sample_count": len(candidate.predictions),
        "sample_id_and_label_exact": True,
        "tolerance": tolerance,
        "max_abs_probability_difference": maximum_probability_difference,
        "max_abs_wp_only_metric_difference": maximum_metric_difference,
        "wp_four_class_macro_f1": float(candidate.metrics["wp_four_class"]["macro_f1"]),
        "wp_binary_macro_f1": float(candidate.metrics["wp_binary"]["macro_f1"]),
        "passed": bool(
            maximum_probability_difference <= tolerance
            and maximum_metric_difference <= tolerance
        ),
        "batch_size": 64,
        "runtime": "cpu",
        "fit_performed": False,
        "optimizer_loaded": False,
        "threshold_search_performed": False,
        "wp_test_accessed_by_release_path": False,
    }


def freeze_release_implementation_identity(
    *,
    project_root: str | Path,
    source_files: Sequence[Path],
    archive_path: str | Path,
    identity_path: str | Path,
    git_head: str,
) -> FrozenReleaseIdentity:
    """Persist exact release source bytes in a deterministic local-only ZIP."""

    root = Path(project_root).resolve(strict=True)
    archive = Path(archive_path).resolve()
    identity_file = Path(identity_path).resolve()
    _require_output_inside_root(root, archive, "release source archive")
    _require_output_inside_root(root, identity_file, "release identity")
    if archive.exists() or identity_file.exists():
        raise FileExistsError("refusing to overwrite release implementation identity artifacts")
    if not isinstance(git_head, str) or not git_head:
        raise ReleaseArtifactError("git_head must be non-empty")
    relative_files = sorted({_strict_relative_path(path, "release source file") for path in source_files})
    if not relative_files:
        raise ReleaseArtifactError("release implementation source file list cannot be empty")
    descriptors: dict[str, dict[str, Any]] = {}
    payloads: dict[str, bytes] = {}
    for relative in relative_files:
        source = _bound_project_file(root, relative, "release source file")
        payload = source.read_bytes()
        name = relative.as_posix()
        payloads[name] = payload
        descriptors[name] = {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
    archive.parent.mkdir(parents=True, exist_ok=True)
    identity_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_archive = archive.with_name(f".{archive.name}.{os.getpid()}.tmp")
    temporary_identity = identity_file.with_name(f".{identity_file.name}.{os.getpid()}.tmp")
    try:
        with zipfile.ZipFile(temporary_archive, mode="x", compression=zipfile.ZIP_STORED) as bundle:
            for name in sorted(payloads):
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_STORED
                info.create_system = 3
                info.external_attr = 0o600 << 16
                bundle.writestr(info, payloads[name])
        archive_bytes = temporary_archive.read_bytes()
        file_bundle_sha256 = hashlib.sha256(_canonical_json_bytes(descriptors)).hexdigest()
        identity = {
            "schema_version": RELEASE_IDENTITY_SCHEMA,
            "identity_method": "persistent deterministic source archive plus per-file SHA-256",
            "git_head_context": git_head,
            "files": descriptors,
            "file_bundle_sha256": file_bundle_sha256,
            "source_archive": {
                "path": archive.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(archive_bytes).hexdigest(),
                "size_bytes": len(archive_bytes),
                "compression": "stored",
                "availability": "current_machine_local_only",
            },
            "scope": {
                "loader": True,
                "wp_only_validator_and_evaluator": True,
                "cli": True,
                "tests": True,
                "training_source_rewritten": False,
                "model_weights_changed": False,
            },
        }
        temporary_identity.write_bytes(_canonical_json_bytes(identity))
        os.replace(temporary_archive, archive)
        os.replace(temporary_identity, identity_file)
    except Exception:
        temporary_archive.unlink(missing_ok=True)
        temporary_identity.unlink(missing_ok=True)
        raise
    identity_sha256 = _sha256_file(identity_file)
    return FrozenReleaseIdentity(
        identity_path=identity_file,
        archive_path=archive,
        identity=identity,
        identity_sha256=identity_sha256,
    )


def build_candidate_bundle(
    *,
    project_root: str | Path,
    training_identity_path: str | Path,
    expected_training_identity_sha256: str,
    release_identity_path: str | Path,
    expected_release_identity_sha256: str,
    output_dir: str | Path,
) -> CandidateBundleResult:
    """Build one compact, immutable-by-path v3 scoring candidate bundle."""

    root = Path(project_root).resolve(strict=True)
    training = _load_externally_hashed_json(
        Path(training_identity_path),
        expected_training_identity_sha256,
        TRAINING_IDENTITY_SCHEMA,
        ReleaseArtifactError,
        "training candidate identity",
    )
    release = _load_externally_hashed_json(
        Path(release_identity_path),
        expected_release_identity_sha256,
        RELEASE_IDENTITY_SCHEMA,
        ReleaseArtifactError,
        "release implementation identity",
    )
    _validate_primary_training_identity(training)
    _validate_release_identity(root, release)
    model_source = _verify_project_descriptor(root, training["model_state"], "model state")
    forward_source = _verify_project_descriptor(root, training["forward_config"], "forward config")
    performance_source = _verify_project_descriptor(
        root, training["performance_config"], "performance config"
    )
    rf_source = _bound_project_file(root, FROZEN_WP_RF_CONFIG, "frozen WP RF config")
    rf_config = load_rf_config(rf_source)
    upstream_artifacts: dict[str, dict[str, Any]] = {}
    for role, descriptor in rf_config["inputs"].items():
        relative = _strict_relative_path(descriptor["path"], f"RF input {role}")
        source = _bound_project_file(root, relative, f"RF input {role}")
        normalized = {
            "path": relative.as_posix(),
            "sha256": descriptor["sha256"],
            "size_bytes": source.stat().st_size,
        }
        if "required_status" in descriptor:
            normalized["required_status"] = descriptor["required_status"]
        upstream_artifacts[role] = normalized
    training_identity_file = Path(training_identity_path).resolve(strict=True)
    release_identity_file = Path(release_identity_path).resolve(strict=True)
    for identity_file, role in (
        (training_identity_file, "training identity"),
        (release_identity_file, "release identity"),
    ):
        if root not in identity_file.parents or not identity_file.is_file():
            raise ReleaseArtifactError(f"{role} must be a file inside project_root")
    output = Path(output_dir).resolve()
    _require_output_inside_root(root, output, "candidate bundle")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite candidate bundle: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        destinations = {
            "model_state": staging / "model_state.npz",
            "forward_config": staging / "forward_config.yaml",
            "performance_config": staging / "performance_config.yaml",
            "frozen_wp_rf_config": staging / "frozen_wp_rf_config.yaml",
        }
        shutil.copyfile(model_source, destinations["model_state"])
        shutil.copyfile(forward_source, destinations["forward_config"])
        shutil.copyfile(performance_source, destinations["performance_config"])
        shutil.copyfile(rf_source, destinations["frozen_wp_rf_config"])
        artifacts = {
            role: {
                "path": destination.name,
                "sha256": _sha256_file(destination),
                "size_bytes": destination.stat().st_size,
            }
            for role, destination in destinations.items()
        }
        manifest = {
            "schema_version": CANDIDATE_MANIFEST_SCHEMA,
            "candidate_id": "topowander-m0s-seed20260731-epoch0005",
            "training_candidate_identity": {
                "original_bytes_sha256": expected_training_identity_sha256,
                "identity": training,
            },
            "release_implementation_identity": {
                "original_bytes_sha256": expected_release_identity_sha256,
                "identity": release,
            },
            "identity_files": {
                "training_candidate_identity": _project_file_descriptor(
                    root, training_identity_file
                ),
                "release_implementation_identity": _project_file_descriptor(
                    root, release_identity_file
                ),
            },
            "artifacts": artifacts,
            "inference_contract": {
                "runtime": dict(FIXED_CPU_RUNTIME),
                "model_mode": "eval",
                "autograd_mode": "torch.inference_mode",
                "binary_decision": "sigmoid >= 0.5",
                "four_class_decision": "argmax hierarchical probability",
                "four_class_order": list(PATTERN_LABELS),
                "subtype_order": list(SUBTYPE_LABELS),
                "input_shapes": {
                    "model_features": [80, 14],
                    "shape_normalized_points": [80, 2],
                    "point_mask": [80],
                },
                "fixed_cohort_wp_prefix_padding": FIXED_COHORT_WP_PREFIX_PADDING,
                "prefix_padding_record": "first_validated_wp_record",
                "prefix_padding_outputs_discarded": True,
            },
            "formal_score_entry": {
                "command": "score-frozen-wp",
                "controller_api": "run_authorized_frozen_wp_score",
                "caller_records_allowed": False,
                "caller_split_allowed": False,
                "caller_batch_or_runtime_allowed": False,
                "caller_threshold_or_class_mapping_allowed": False,
                "accessor_loader": "load_preprocessing_bundle",
                "accessor_mode": BUNDLE_MODE_FROZEN_WP_TEST,
                "records_method": "records_for_split",
                "records_split": "test",
                "preflight_before_accessor": True,
                "preflight_before_model_inference": True,
                "final_output_must_not_exist": True,
                "same_filesystem_staging_and_atomic_commit": True,
            },
            "frozen_wp_accessor": {
                "rf_config_project": _project_file_descriptor(root, rf_source),
                "rf_config_bundle_artifact": "frozen_wp_rf_config",
                "upstream_artifacts": upstream_artifacts,
                "split_sha256": rf_config["split_sha256"],
                "preprocessing_config_sha256": rf_config["inputs"][
                    "preprocessing_config"
                ]["sha256"],
                "expected_test_count": 240,
                "expected_pattern_counts": {name: 60 for name in PATTERN_LABELS},
                "expected_binary_counts": {"0": 60, "1": 180},
                "canonical_records_sha256": None,
                "ordered_sample_ids_sha256": None,
                "cohort_hashes_generated_only_during_authorized_score": True,
            },
            "bundle_scope": {
                "complete_model_state_including_projection_arrays": True,
                "optimizer_state_included": False,
                "rng_state_included": False,
                "training_history_included": False,
                "all_seed_checkpoints_included": False,
                "availability": "current_machine_local_only",
            },
            "m0rh_data_access": {
                "wp_validation": True,
                "wp_test_accessor_called": False,
                "wp_test_inference": False,
                "wp_test_scoring": False,
                "wp_raw": False,
                "smartcare_official_or_raw": False,
                "sealed_camera": False,
            },
        }
        manifest_path = staging / "candidate_manifest.json"
        manifest_path.write_bytes(_canonical_json_bytes(manifest))
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    final_manifest = output / "candidate_manifest.json"
    return CandidateBundleResult(
        bundle_dir=output,
        manifest_path=final_manifest,
        manifest_sha256=_sha256_file(final_manifest),
        manifest=manifest,
    )


def load_candidate_from_manifest(
    manifest_path: str | Path, *, expected_manifest_sha256: str
) -> CandidateRuntime:
    """Load a compact candidate only after an external manifest digest matches."""

    if not _valid_sha256(expected_manifest_sha256):
        raise CandidateManifestError("external manifest SHA-256 must be lowercase hex")
    manifest_file = Path(manifest_path).resolve(strict=True)
    raw = manifest_file.read_bytes()
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_sha256 != expected_manifest_sha256:
        raise CandidateManifestError("external manifest SHA-256 mismatch")
    manifest = _parse_json_bytes(raw, CandidateManifestError, "candidate manifest")
    _validate_candidate_manifest(manifest)
    bundle_root = manifest_file.parent.resolve(strict=True)
    model_path = _verify_bundle_descriptor(
        bundle_root, manifest["artifacts"]["model_state"], "model state"
    )
    forward_path = _verify_bundle_descriptor(
        bundle_root, manifest["artifacts"]["forward_config"], "forward config"
    )
    _verify_bundle_descriptor(
        bundle_root, manifest["artifacts"]["performance_config"], "performance config"
    )
    rf_path = _verify_bundle_descriptor(
        bundle_root, manifest["artifacts"]["frozen_wp_rf_config"], "frozen WP RF config"
    )
    load_rf_config(rf_path)
    training_identity = manifest["training_candidate_identity"]["identity"]
    if training_identity["model_state"]["sha256"] != manifest["artifacts"]["model_state"]["sha256"]:
        raise CandidateManifestError("manifest model state does not match frozen training identity")
    if training_identity["forward_config"]["sha256"] != manifest["artifacts"]["forward_config"]["sha256"]:
        raise CandidateManifestError("manifest forward config does not match frozen training identity")
    if training_identity["performance_config"]["sha256"] != manifest["artifacts"]["performance_config"]["sha256"]:
        raise CandidateManifestError("manifest performance config does not match frozen training identity")
    if (
        manifest["frozen_wp_accessor"]["rf_config_project"]["sha256"]
        != manifest["artifacts"]["frozen_wp_rf_config"]["sha256"]
    ):
        raise CandidateManifestError("bundled RF config does not match frozen accessor binding")
    model = safe_load_topowander_model(
        model_path,
        config_path=forward_path,
        expected_config_sha256=manifest["artifacts"]["forward_config"]["sha256"],
    )
    model.to(torch.device("cpu"))
    model.eval()
    return CandidateRuntime(
        bundle_root=bundle_root,
        manifest_path=manifest_file,
        manifest_sha256=actual_sha256,
        manifest=manifest,
        model=model,
    )


def write_wp_evaluation_artifacts(
    output_dir: str | Path,
    *,
    result: WPEvaluationResult,
    execution: Mapping[str, Any],
    parity: Mapping[str, Any] | None = None,
) -> Path:
    """Write a complete WP evaluation into staging, then atomically commit."""

    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite release evaluation output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        payloads = {
            "metrics.json": _canonical_json_bytes(result.metrics),
            "predictions.jsonl": _canonical_jsonl_bytes(result.predictions),
            "confusion.json": _canonical_json_bytes(result.confusion),
            "errors.jsonl": _canonical_jsonl_bytes(result.errors),
            "execution.json": _canonical_json_bytes(execution),
        }
        if parity is not None:
            payloads["parity.json"] = _canonical_json_bytes(parity)
        for name, payload in payloads.items():
            (staging / name).write_bytes(payload)
        descriptors = {
            name: {
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
            for name, payload in sorted(payloads.items())
        }
        artifact_manifest = {
            "schema_version": "wandering-m0rh-evaluation-artifact-manifest-v1",
            "artifacts": descriptors,
            "exact_artifact_set_verified_before_commit": True,
            "same_filesystem_atomic_commit": True,
        }
        manifest_payload = _canonical_json_bytes(artifact_manifest)
        (staging / "artifact_manifest.json").write_bytes(manifest_payload)
        expected_names = sorted((*payloads, "artifact_manifest.json"))
        observed_names = sorted(path.name for path in staging.iterdir())
        if observed_names != expected_names:
            raise ReleaseArtifactError("evaluation staging artifact set is incomplete")
        for name, payload in payloads.items():
            if (staging / name).read_bytes() != payload:
                raise ReleaseArtifactError(f"evaluation staging bytes mismatch: {name}")
        if (staging / "artifact_manifest.json").read_bytes() != manifest_payload:
            raise ReleaseArtifactError("evaluation artifact manifest bytes mismatch")
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def run_primary_validation_parity(
    *,
    project_root: str | Path,
    manifest_path: str | Path,
    expected_manifest_sha256: str,
    output_dir: str | Path,
    tolerance: float = 1.0e-7,
) -> Path:
    """Fresh-load the candidate and reproduce the frozen 240-row WP validation view."""

    root = Path(project_root).resolve(strict=True)
    runtime = load_candidate_from_manifest(
        manifest_path, expected_manifest_sha256=expected_manifest_sha256
    )
    contract = runtime.manifest["inference_contract"]
    runtime_contract = contract["runtime"]
    if runtime_contract != FIXED_CPU_RUNTIME:
        raise CandidateManifestError("validation parity runtime must remain CPU 8/1 batch64")
    from .performance import load_development_records, load_performance_config

    bundled_performance = runtime.bundle_root / runtime.manifest["artifacts"]["performance_config"]["path"]
    config = load_performance_config(bundled_performance, project_root=root)
    expected_performance_runtime = {
        key: runtime_contract[key]
        for key in (
            "device",
            "intra_op_threads",
            "inter_op_threads",
            "num_workers",
            "pin_memory",
        )
    }
    if (
        config["runtime"] != expected_performance_runtime
        or int(config["training"]["batch_size"]) != runtime_contract["batch_size"]
    ):
        raise CandidateManifestError("primary validation runtime or batch size drifted")
    observed_runtime = configure_release_cpu_runtime(runtime_contract)
    _train, validation, data_identity = load_development_records(config, project_root=root)
    wp_records = tuple(
        record for record in validation if record["source_dataset"] == "wandering_patterns"
    )
    validate_wp_release_cohort(wp_records, expected_split="validation")
    binary, subtype = runtime.predict_logits(
        wp_records,
        expected_split="validation",
        batch_size=observed_runtime["batch_size"],
        require_fixed_cohort=True,
    )
    result = evaluate_wp_predictions(
        wp_records,
        binary,
        subtype,
        expected_split="validation",
        require_fixed_cohort=True,
    )
    training_identity = runtime.manifest["training_candidate_identity"]["identity"]
    reference = training_identity["primary_validation_reference"]
    metrics_path = _verify_project_descriptor(root, reference["metrics"], "reference metrics")
    predictions_path = _verify_project_descriptor(
        root, reference["predictions"], "reference predictions"
    )
    reference_metrics = _load_json_file(metrics_path, WPReleaseDataError, "reference metrics")
    reference_predictions = tuple(
        row
        for row in _load_jsonl_file(
            predictions_path, WPReleaseDataError, "reference predictions"
        )
        if row.get("source_dataset") == "wandering_patterns"
    )
    reference_four = reference_metrics["joint_metrics"]["wp_four_class"]
    reference_binary = reference_metrics["joint_metrics"]["binary"]["wp"]
    if (
        len(reference_predictions) != 240
        or float(reference_four["macro_f1"]) != 0.9874973951700553
        or float(reference_binary["macro_f1"]) != 0.9944750109348742
    ):
        raise WPReleaseDataError("frozen primary validation reference has drifted")
    parity = compare_primary_validation_parity(
        result,
        reference_predictions=reference_predictions,
        reference_wp_four_metrics=reference_four,
        reference_wp_binary_metrics=reference_binary,
        tolerance=tolerance,
    )
    parity.update(
        {
            "candidate_manifest_sha256": runtime.manifest_sha256,
            "training_candidate_identity_sha256": runtime.manifest[
                "training_candidate_identity"
            ]["original_bytes_sha256"],
            "release_implementation_identity_sha256": runtime.manifest[
                "release_implementation_identity"
            ]["original_bytes_sha256"],
            "development_records_sha256": data_identity["development_records_sha256"],
            "reference_metrics_sha256": reference["metrics"]["sha256"],
            "reference_predictions_sha256": reference["predictions"]["sha256"],
            "expected_wp_four_class_macro_f1": 0.9874973951700553,
            "expected_wp_binary_macro_f1": 0.9944750109348742,
            "source_equal_metric_used_for_parity": False,
        }
    )
    if not parity["passed"]:
        raise WPReleaseDataError(f"primary validation parity failed: {parity}")
    execution = {
        "schema_version": "wandering-m0rh-validation-parity-execution-v3",
        "phase": "validation",
        "candidate_manifest_sha256": runtime.manifest_sha256,
        "runtime": observed_runtime,
        "torch_inference_mode": True,
        "model_eval": True,
        "fit_performed": False,
        "optimizer_loaded": False,
        "threshold_search_performed": False,
        "wp_test_accessor_called": False,
        "wp_test_inference_performed": False,
        "wp_test_scoring_performed": False,
        "wp_raw_read": False,
        "smartcare_official_or_raw_read": False,
        "sealed_camera_read": False,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "fixed_cohort_wp_prefix_padding": contract["fixed_cohort_wp_prefix_padding"],
        "prefix_padding_record": contract["prefix_padding_record"],
        "prefix_padding_outputs_discarded": contract[
            "prefix_padding_outputs_discarded"
        ],
    }
    return write_wp_evaluation_artifacts(
        output_dir, result=result, execution=execution, parity=parity
    )


def run_wp_records_evaluation(
    *,
    manifest_path: str | Path,
    expected_manifest_sha256: str,
    records_path: str | Path,
    expected_split: str,
    output_dir: str | Path,
    batch_size: int = 64,
) -> Path:
    """Evaluate an explicitly supplied WP *validation* cohort for development only."""

    if expected_split != "validation":
        raise WPReleaseDataError("evaluate-records is validation-only; test requires score-frozen-wp")
    runtime = load_candidate_from_manifest(
        manifest_path, expected_manifest_sha256=expected_manifest_sha256
    )
    records_file = Path(records_path).resolve(strict=True)
    records = tuple(_load_jsonl_file(records_file, WPReleaseDataError, "WP records"))
    validate_wp_release_cohort(records, expected_split=expected_split)
    binary, subtype = runtime.predict_logits(
        records,
        expected_split=expected_split,
        batch_size=batch_size,
        require_fixed_cohort=True,
    )
    result = evaluate_wp_predictions(
        records,
        binary,
        subtype,
        expected_split=expected_split,
        require_fixed_cohort=True,
    )
    execution = {
        "schema_version": "wandering-m0r-wp-evaluation-execution-v1",
        "phase": expected_split,
        "records_path": str(records_file),
        "records_sha256": _sha256_file(records_file),
        "candidate_manifest_sha256": runtime.manifest_sha256,
        "device": "cpu",
        "batch_size": batch_size,
        "torch_inference_mode": True,
        "model_eval": True,
        "fit_performed": False,
        "optimizer_loaded": False,
        "threshold_search_performed": False,
        "accessor_called_by_this_path": False,
        "fixed_cohort_wp_prefix_padding": runtime.manifest["inference_contract"][
            "fixed_cohort_wp_prefix_padding"
        ],
        "prefix_padding_record": runtime.manifest["inference_contract"][
            "prefix_padding_record"
        ],
        "prefix_padding_outputs_discarded": runtime.manifest["inference_contract"][
            "prefix_padding_outputs_discarded"
        ],
    }
    return write_wp_evaluation_artifacts(output_dir, result=result, execution=execution)


def _validate_loaded_frozen_bundle_identity(
    bundle: Any, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    accessor = manifest["frozen_wp_accessor"]
    expected_hashes = {
        role: descriptor["sha256"]
        for role, descriptor in accessor["upstream_artifacts"].items()
    }
    if bundle.mode != BUNDLE_MODE_FROZEN_WP_TEST:
        raise WPReleaseDataError("trusted accessor returned the wrong mode")
    if dict(bundle.input_hashes) != expected_hashes:
        raise WPReleaseDataError("trusted accessor input identity differs from manifest")
    if bundle.split_sha256 != accessor["split_sha256"]:
        raise WPReleaseDataError("trusted accessor split identity differs from manifest")
    if bundle.preprocessing_config_sha256 != accessor["preprocessing_config_sha256"]:
        raise WPReleaseDataError(
            "trusted accessor preprocessing identity differs from manifest"
        )
    required_integrity = {
        "all_frozen_hashes_verified": True,
        "manifest_binding_verified": True,
        "human_review_status": "human_review_passed",
        "near_neighbor_cross_partition_lt_0_05": 4054,
        "official_source_path_present": False,
        "total_records": 1790,
    }
    if dict(bundle.integrity_report) != required_integrity:
        raise WPReleaseDataError("trusted accessor integrity attestation differs")
    return {
        "input_hashes": expected_hashes,
        "split_sha256": bundle.split_sha256,
        "preprocessing_config_sha256": bundle.preprocessing_config_sha256,
        "integrity_report": required_integrity,
    }


def run_authorized_frozen_wp_score(
    *,
    project_root: str | Path,
    manifest_path: str | Path,
    expected_manifest_sha256: str,
    output_dir: str | Path,
    active_cli_path: str | Path,
) -> Path:
    """Run the sole formal score entry after caller-provided authorization.

    Invoking this API is the authorization-bearing action.  Every source,
    archive, identity, candidate, upstream, output, and runtime check completes
    before the trusted frozen accessor is opened; records cannot be supplied by
    the caller.
    """

    root = Path(project_root).resolve(strict=True)
    output = Path(output_dir).resolve()
    _require_output_inside_root(root, output, "formal WP score")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite formal WP score output: {output}")

    manifest_file = Path(manifest_path).resolve(strict=True)
    if root not in manifest_file.parents:
        raise CandidateManifestError("formal score manifest must stay inside project_root")

    runtime = load_candidate_from_manifest(
        manifest_file, expected_manifest_sha256=expected_manifest_sha256
    )
    release_identity = runtime.manifest["release_implementation_identity"]["identity"]
    active_source_audit = _verify_active_release_sources(
        root, release_identity, active_cli_path=active_cli_path
    )
    archive_audit = _verify_release_source_archive(root, release_identity)
    identity_audit = _verify_identity_files_and_cross_bindings(root, runtime.manifest)
    upstream_audit = _verify_frozen_wp_upstream_files(root, runtime.manifest, runtime)
    observed_runtime = configure_release_cpu_runtime(
        runtime.manifest["inference_contract"]["runtime"]
    )

    bundled_rf_path = runtime.bundle_root / runtime.manifest["artifacts"][
        "frozen_wp_rf_config"
    ]["path"]
    bundle = load_preprocessing_bundle(
        rf_config_path=bundled_rf_path,
        project_root=root,
        mode=BUNDLE_MODE_FROZEN_WP_TEST,
    )
    accessor_identity = _validate_loaded_frozen_bundle_identity(bundle, runtime.manifest)
    records = tuple(bundle.records_for_split("test"))
    validate_wp_release_cohort(records, expected_split="test")
    cohort_identity = canonical_wp_cohort_identity(records, expected_split="test")
    binary, subtype = runtime.predict_logits(
        records,
        expected_split="test",
        batch_size=observed_runtime["batch_size"],
        require_fixed_cohort=True,
    )
    result = evaluate_wp_predictions(
        records,
        binary,
        subtype,
        expected_split="test",
        require_fixed_cohort=True,
    )
    execution = {
        "schema_version": "wandering-m0rh-formal-frozen-wp-execution-v1",
        "phase": "test",
        "candidate_manifest_path": str(runtime.manifest_path),
        "candidate_manifest_sha256": runtime.manifest_sha256,
        "candidate_id": runtime.manifest["candidate_id"],
        "formal_command": "score-frozen-wp",
        "accessor_mode": BUNDLE_MODE_FROZEN_WP_TEST,
        "accessor_records_method": "records_for_split",
        "caller_records_allowed": False,
        "preflight_completed_before_accessor": True,
        "preflight_completed_before_model_inference": True,
        "runtime": observed_runtime,
        "torch_inference_mode": True,
        "model_eval": True,
        "fit_performed": False,
        "optimizer_loaded": False,
        "threshold_search_performed": False,
        "model_or_threshold_modified": False,
        "cohort_identity": cohort_identity,
        "accessor_identity": accessor_identity,
        "active_source_audit": active_source_audit,
        "source_archive_audit": archive_audit,
        "identity_cross_binding_audit": identity_audit,
        "upstream_preflight_audit": upstream_audit,
        "output_contract": {
            "final_previously_absent": True,
            "same_filesystem_staging": True,
            "atomic_directory_commit": True,
            "overwrite_refused": True,
            "metrics_not_printed_before_commit": True,
        },
        "data_access": {
            "wp_public_holdout": True,
            "wp_raw": False,
            "smartcare_official_or_raw": False,
            "sealed_camera": False,
        },
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
    }
    return write_wp_evaluation_artifacts(output, result=result, execution=execution)


def _validate_candidate_manifest(manifest: Mapping[str, Any]) -> None:
    expected = {
        "schema_version",
        "candidate_id",
        "training_candidate_identity",
        "release_implementation_identity",
        "identity_files",
        "artifacts",
        "inference_contract",
        "formal_score_entry",
        "frozen_wp_accessor",
        "bundle_scope",
        "m0rh_data_access",
    }
    if set(manifest) != expected or manifest.get("schema_version") != CANDIDATE_MANIFEST_SCHEMA:
        raise CandidateManifestError("candidate manifest schema has drifted")
    if manifest["candidate_id"] != "topowander-m0s-seed20260731-epoch0005":
        raise CandidateManifestError("candidate ID has drifted")
    training_bound = manifest["training_candidate_identity"]
    release_bound = manifest["release_implementation_identity"]
    if not isinstance(training_bound, Mapping) or set(training_bound) != {
        "original_bytes_sha256",
        "identity",
    }:
        raise CandidateManifestError("training identity binding is invalid")
    if not isinstance(release_bound, Mapping) or set(release_bound) != {
        "original_bytes_sha256",
        "identity",
    }:
        raise CandidateManifestError("release identity binding is invalid")
    if not _valid_sha256(training_bound["original_bytes_sha256"]) or not _valid_sha256(
        release_bound["original_bytes_sha256"]
    ):
        raise CandidateManifestError("identity binding SHA-256 is invalid")
    _validate_primary_training_identity(training_bound["identity"])
    release_identity = release_bound["identity"]
    if not isinstance(release_identity, Mapping) or release_identity.get("schema_version") != RELEASE_IDENTITY_SCHEMA:
        raise CandidateManifestError("embedded release identity is invalid")
    release_files = release_identity.get("files")
    if not isinstance(release_files, Mapping) or set(release_files) != {
        path.as_posix() for path in RELEASE_IMPLEMENTATION_SOURCE_FILES
    }:
        raise CandidateManifestError("embedded release source set is not the v3 exact set")
    identity_files = manifest["identity_files"]
    if not isinstance(identity_files, Mapping) or set(identity_files) != {
        "training_candidate_identity",
        "release_implementation_identity",
    }:
        raise CandidateManifestError("candidate identity file bindings are invalid")
    for role, descriptor in identity_files.items():
        _validate_project_descriptor_shape(
            descriptor, f"{role} file", error_type=CandidateManifestError
        )
    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "model_state",
        "forward_config",
        "performance_config",
        "frozen_wp_rf_config",
    }:
        raise CandidateManifestError("candidate artifact set is not compact or complete")
    for role, descriptor in artifacts.items():
        if not isinstance(descriptor, Mapping) or set(descriptor) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            raise CandidateManifestError(f"candidate artifact descriptor is invalid: {role}")
    contract = manifest["inference_contract"]
    expected_contract = {
        "runtime": dict(FIXED_CPU_RUNTIME),
        "model_mode": "eval",
        "autograd_mode": "torch.inference_mode",
        "binary_decision": "sigmoid >= 0.5",
        "four_class_decision": "argmax hierarchical probability",
        "four_class_order": list(PATTERN_LABELS),
        "subtype_order": list(SUBTYPE_LABELS),
        "input_shapes": {
            "model_features": [80, 14],
            "shape_normalized_points": [80, 2],
            "point_mask": [80],
        },
        "fixed_cohort_wp_prefix_padding": FIXED_COHORT_WP_PREFIX_PADDING,
        "prefix_padding_record": "first_validated_wp_record",
        "prefix_padding_outputs_discarded": True,
    }
    if contract != expected_contract:
        raise CandidateManifestError("candidate inference contract has drifted")
    if manifest["formal_score_entry"] != {
        "command": "score-frozen-wp",
        "controller_api": "run_authorized_frozen_wp_score",
        "caller_records_allowed": False,
        "caller_split_allowed": False,
        "caller_batch_or_runtime_allowed": False,
        "caller_threshold_or_class_mapping_allowed": False,
        "accessor_loader": "load_preprocessing_bundle",
        "accessor_mode": BUNDLE_MODE_FROZEN_WP_TEST,
        "records_method": "records_for_split",
        "records_split": "test",
        "preflight_before_accessor": True,
        "preflight_before_model_inference": True,
        "final_output_must_not_exist": True,
        "same_filesystem_staging_and_atomic_commit": True,
    }:
        raise CandidateManifestError("formal frozen-WP score entry contract has drifted")
    accessor = manifest["frozen_wp_accessor"]
    expected_accessor_fields = {
        "rf_config_project",
        "rf_config_bundle_artifact",
        "upstream_artifacts",
        "split_sha256",
        "preprocessing_config_sha256",
        "expected_test_count",
        "expected_pattern_counts",
        "expected_binary_counts",
        "canonical_records_sha256",
        "ordered_sample_ids_sha256",
        "cohort_hashes_generated_only_during_authorized_score",
    }
    if not isinstance(accessor, Mapping) or set(accessor) != expected_accessor_fields:
        raise CandidateManifestError("frozen WP accessor contract fields have drifted")
    _validate_project_descriptor_shape(
        accessor["rf_config_project"],
        "frozen WP RF config",
        error_type=CandidateManifestError,
    )
    if accessor["rf_config_project"]["path"] != FROZEN_WP_RF_CONFIG.as_posix():
        raise CandidateManifestError("frozen WP RF config path has drifted")
    upstream = accessor["upstream_artifacts"]
    if not isinstance(upstream, Mapping) or set(upstream) != set(FROZEN_WP_INPUT_ROLES):
        raise CandidateManifestError("frozen WP upstream descriptor roles have drifted")
    for role, descriptor in upstream.items():
        expected_fields = (
            {"path", "sha256", "size_bytes", "required_status"}
            if role == "human_review"
            else {"path", "sha256", "size_bytes"}
        )
        if not isinstance(descriptor, Mapping) or set(descriptor) != expected_fields:
            raise CandidateManifestError(f"frozen WP upstream descriptor is invalid: {role}")
        _validate_project_descriptor_shape(
            {key: descriptor[key] for key in ("path", "sha256", "size_bytes")},
            f"frozen WP upstream {role}",
            error_type=CandidateManifestError,
        )
    if (
        accessor["rf_config_bundle_artifact"] != "frozen_wp_rf_config"
        or not _valid_sha256(accessor["split_sha256"])
        or not _valid_sha256(accessor["preprocessing_config_sha256"])
        or accessor["expected_test_count"] != 240
        or accessor["expected_pattern_counts"] != {name: 60 for name in PATTERN_LABELS}
        or accessor["expected_binary_counts"] != {"0": 60, "1": 180}
        or accessor["canonical_records_sha256"] is not None
        or accessor["ordered_sample_ids_sha256"] is not None
        or accessor["cohort_hashes_generated_only_during_authorized_score"] is not True
    ):
        raise CandidateManifestError("frozen WP accessor values have drifted")
    if manifest["bundle_scope"] != {
        "complete_model_state_including_projection_arrays": True,
        "optimizer_state_included": False,
        "rng_state_included": False,
        "training_history_included": False,
        "all_seed_checkpoints_included": False,
        "availability": "current_machine_local_only",
    }:
        raise CandidateManifestError("candidate bundle scope has drifted")
    if manifest["m0rh_data_access"] != {
        "wp_validation": True,
        "wp_test_accessor_called": False,
        "wp_test_inference": False,
        "wp_test_scoring": False,
        "wp_raw": False,
        "smartcare_official_or_raw": False,
        "sealed_camera": False,
    }:
        raise CandidateManifestError("candidate M0-RH data boundary has drifted")


def _validate_primary_training_identity(identity: Any) -> None:
    if not isinstance(identity, Mapping) or identity.get("schema_version") != TRAINING_IDENTITY_SCHEMA:
        raise ReleaseArtifactError("training candidate identity schema is invalid")
    candidate = identity.get("candidate")
    if not isinstance(candidate, Mapping):
        raise ReleaseArtifactError("training candidate fields are invalid")
    expected = {
        "seed": PRIMARY_SEED,
        "best_epoch": PRIMARY_BEST_EPOCH,
        "historical_m0_allowed": False,
        "ensemble": False,
        "retrained_for_release": False,
    }
    if any(candidate.get(name) != value for name, value in expected.items()):
        raise ReleaseArtifactError("training candidate is not the frozen M0-S primary")
    if candidate.get("family") != "TopoWander-MPT" or candidate.get("stage") != "M0-S-primary-development-candidate":
        raise ReleaseArtifactError("training candidate family or stage is invalid")
    for descriptor_name in ("model_state", "forward_config", "performance_config"):
        descriptor = identity.get(descriptor_name)
        if not isinstance(descriptor, Mapping):
            raise ReleaseArtifactError(f"training {descriptor_name} descriptor is invalid")
        if not _valid_sha256(descriptor.get("sha256")):
            raise ReleaseArtifactError(f"training {descriptor_name} SHA-256 is invalid")
    if identity["model_state"].get("contains_complete_model_state") is not True:
        raise ReleaseArtifactError("training candidate must retain complete model state")
    metadata = identity.get("checkpoint_metadata")
    best = identity.get("best_pointer")
    if not isinstance(metadata, Mapping) or metadata.get("seed") != PRIMARY_SEED or metadata.get("epoch") != PRIMARY_BEST_EPOCH:
        raise ReleaseArtifactError("checkpoint metadata identity is not seed 20260731 epoch 5")
    if not isinstance(best, Mapping) or best.get("epoch") != PRIMARY_BEST_EPOCH or best.get("checkpoint") != "epoch-0005":
        raise ReleaseArtifactError("best pointer identity is not epoch 5")
    source = identity.get("training_source_bundle")
    if not isinstance(source, Mapping) or not _valid_sha256(source.get("bundle_sha256")):
        raise ReleaseArtifactError("training source bundle identity is invalid")
    data = identity.get("data_identity")
    reference = identity.get("primary_validation_reference")
    if not isinstance(data, Mapping) or data.get("wp_validation_count") != 240:
        raise ReleaseArtifactError("training data identity is invalid")
    if (
        not isinstance(reference, Mapping)
        or reference.get("wp_sample_count") != 240
        or reference.get("wp_four_class_macro_f1") != 0.9874973951700553
        or reference.get("wp_binary_macro_f1") != 0.9944750109348742
    ):
        raise ReleaseArtifactError("primary validation reference identity is invalid")


def _validate_release_identity(root: Path, identity: Mapping[str, Any]) -> None:
    if identity.get("schema_version") != RELEASE_IDENTITY_SCHEMA:
        raise ReleaseArtifactError("release implementation identity schema is invalid")
    files = identity.get("files")
    expected_files = {path.as_posix() for path in RELEASE_IMPLEMENTATION_SOURCE_FILES}
    if not isinstance(files, Mapping) or set(files) != expected_files:
        raise ReleaseArtifactError("release implementation file identity must use the v3 exact set")
    for name, descriptor in files.items():
        relative = _strict_relative_path(name, "release implementation file")
        source = _bound_project_file(root, relative, "release implementation file")
        _verify_file_digest_and_size(source, descriptor, "release implementation file")
    archive = identity.get("source_archive")
    if not isinstance(archive, Mapping):
        raise ReleaseArtifactError("release source archive descriptor is invalid")
    archive_path = _bound_project_file(
        root, _strict_relative_path(archive.get("path"), "release source archive"), "release source archive"
    )
    _verify_file_digest_and_size(archive_path, archive, "release source archive")
    if archive.get("availability") != "current_machine_local_only":
        raise ReleaseArtifactError("release source archive availability must be local-only")
    if archive.get("compression") != "stored":
        raise ReleaseArtifactError("release source archive compression must be stored")
    expected_bundle_sha = hashlib.sha256(_canonical_json_bytes(files)).hexdigest()
    if identity.get("file_bundle_sha256") != expected_bundle_sha:
        raise ReleaseArtifactError("release implementation file-bundle identity mismatch")
    _verify_release_source_archive(root, identity)


def _verify_release_source_archive(
    root: Path, identity: Mapping[str, Any]
) -> dict[str, Any]:
    files = identity["files"]
    archive_descriptor = identity["source_archive"]
    archive_path = _verify_project_descriptor(
        root, archive_descriptor, "release source archive"
    )
    expected_names = sorted(files)
    try:
        with zipfile.ZipFile(archive_path, mode="r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if names != expected_names or len(names) != len(set(names)):
                raise ReleaseArtifactError("release source archive entry set/order mismatch")
            for info in infos:
                if info.is_dir() or info.compress_type != zipfile.ZIP_STORED:
                    raise ReleaseArtifactError("release source archive entry metadata mismatch")
                payload = archive.read(info)
                descriptor = files[info.filename]
                if (
                    len(payload) != descriptor["size_bytes"]
                    or hashlib.sha256(payload).hexdigest() != descriptor["sha256"]
                ):
                    raise ReleaseArtifactError(
                        f"release source archive entry bytes mismatch: {info.filename}"
                    )
    except (OSError, zipfile.BadZipFile, KeyError) as exc:
        raise ReleaseArtifactError("cannot verify release source archive") from exc
    return {
        "source_archive_path": str(archive_path),
        "source_archive_sha256": archive_descriptor["sha256"],
        "source_archive_size_bytes": archive_descriptor["size_bytes"],
        "source_archive_entries": expected_names,
        "source_archive_outer_and_entries_verified": True,
    }


def _verify_active_release_sources(
    root: Path,
    identity: Mapping[str, Any],
    *,
    active_cli_path: str | Path,
) -> dict[str, Any]:
    from . import performance as performance_module

    expected_actual_paths = {
        RELEASE_MODULE_SOURCE: Path(__file__).resolve(strict=True),
        RELEASE_CLI_SOURCE: Path(active_cli_path).resolve(strict=True),
        MODEL_SOURCE: Path(model_module.__file__).resolve(strict=True),
        PERFORMANCE_SOURCE: Path(performance_module.__file__).resolve(strict=True),
        PREPROCESSING_BUNDLE_SOURCE: Path(
            preprocessing_bundle_module.__file__
        ).resolve(strict=True),
        FROZEN_WP_RF_CONFIG: (root / FROZEN_WP_RF_CONFIG).resolve(strict=True),
    }
    for relative, actual in expected_actual_paths.items():
        expected = (root / relative).resolve(strict=True)
        if actual != expected:
            raise ReleaseArtifactError(
                f"active release source path mismatch: {relative.as_posix()}"
            )
    files = identity["files"]
    verified: dict[str, dict[str, Any]] = {}
    for name in sorted(files):
        relative = _strict_relative_path(name, "active release source")
        path = _bound_project_file(root, relative, "active release source")
        _verify_file_digest_and_size(
            path, files[name], f"active release source {name}"
        )
        verified[name] = {
            "path": str(path),
            "sha256": files[name]["sha256"],
            "size_bytes": files[name]["size_bytes"],
        }
    return {
        "active_execution_paths_verified": True,
        "active_source_files": verified,
    }


def _verify_embedded_identity_file(
    root: Path,
    descriptor: Mapping[str, Any],
    embedded: Mapping[str, Any],
    original_bytes_sha256: str,
    *,
    role: str,
) -> Path:
    path = _verify_project_descriptor(root, descriptor, role)
    if descriptor["sha256"] != original_bytes_sha256:
        raise ReleaseArtifactError(f"{role} descriptor/original SHA cross-binding mismatch")
    parsed = _load_json_file(path, ReleaseArtifactError, role)
    if parsed != embedded:
        raise ReleaseArtifactError(f"{role} bytes do not equal embedded identity")
    return path


def _verify_training_candidate_files(
    root: Path,
    training: Mapping[str, Any],
    release_identity: Mapping[str, Any],
) -> dict[str, Any]:
    for role in (
        "model_state",
        "checkpoint_metadata",
        "best_pointer",
        "forward_config",
        "performance_config",
        "resolved_config",
    ):
        _verify_project_descriptor(root, training[role], f"training candidate {role}")
    reference = training["primary_validation_reference"]
    for role in ("metrics", "predictions"):
        _verify_project_descriptor(
            root, reference[role], f"training validation reference {role}"
        )
    source_bundle = training["training_source_bundle"]
    files = source_bundle.get("files")
    snapshot_root_value = source_bundle.get("persistent_snapshot_root")
    if not isinstance(files, Mapping) or not isinstance(snapshot_root_value, str):
        raise ReleaseArtifactError("training source bundle files/snapshot are invalid")
    expected_training_files = {
        "configs/modules/wandering_performance_v1.yaml",
        "configs/modules/wandering_topowander_mpt_v1.yaml",
        "scripts/wandering/train_wandering_performance.py",
        MODEL_SOURCE.as_posix(),
        PERFORMANCE_SOURCE.as_posix(),
        PREPROCESSING_BUNDLE_SOURCE.as_posix(),
    }
    if set(files) != expected_training_files:
        raise ReleaseArtifactError("training source bundle exact file set has drifted")
    snapshot_root = _strict_relative_path(
        snapshot_root_value, "training source snapshot root"
    )
    verified: dict[str, dict[str, Any]] = {}
    for name in sorted(files):
        descriptor = files[name]
        relative = _strict_relative_path(name, "training source file")
        active = _bound_project_file(root, relative, "training source file")
        snapshot = _bound_project_file(
            root, snapshot_root / relative, "training source snapshot file"
        )
        _verify_file_digest_and_size(active, descriptor, f"training source active {name}")
        _verify_file_digest_and_size(
            snapshot, descriptor, f"training source snapshot {name}"
        )
        if name in release_identity["files"] and release_identity["files"][name] != descriptor:
            raise ReleaseArtifactError(
                f"training/release source descriptor cross-binding mismatch: {name}"
            )
        verified[name] = {
            "sha256": descriptor["sha256"],
            "size_bytes": descriptor["size_bytes"],
            "active_and_snapshot_verified": True,
        }
    return {
        "training_source_bundle_sha256": source_bundle["bundle_sha256"],
        "training_source_files": verified,
        "training_source_active_and_snapshot_verified": True,
    }


def _verify_identity_files_and_cross_bindings(
    root: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    training_bound = manifest["training_candidate_identity"]
    release_bound = manifest["release_implementation_identity"]
    identity_files = manifest["identity_files"]
    training_path = _verify_embedded_identity_file(
        root,
        identity_files["training_candidate_identity"],
        training_bound["identity"],
        training_bound["original_bytes_sha256"],
        role="training candidate identity",
    )
    release_path = _verify_embedded_identity_file(
        root,
        identity_files["release_implementation_identity"],
        release_bound["identity"],
        release_bound["original_bytes_sha256"],
        role="release implementation identity",
    )
    training_audit = _verify_training_candidate_files(
        root, training_bound["identity"], release_bound["identity"]
    )
    rf_descriptor = release_bound["identity"]["files"][
        FROZEN_WP_RF_CONFIG.as_posix()
    ]
    accessor_descriptor = manifest["frozen_wp_accessor"]["rf_config_project"]
    if (
        rf_descriptor["sha256"] != accessor_descriptor["sha256"]
        or rf_descriptor["size_bytes"] != accessor_descriptor["size_bytes"]
    ):
        raise ReleaseArtifactError("release/RF accessor descriptor cross-binding mismatch")
    return {
        "training_candidate_identity_path": str(training_path),
        "training_candidate_identity_sha256": training_bound["original_bytes_sha256"],
        "release_implementation_identity_path": str(release_path),
        "release_implementation_identity_sha256": release_bound[
            "original_bytes_sha256"
        ],
        "identity_files_equal_embedded_identities": True,
        "training_release_cross_binding_verified": True,
        **training_audit,
    }


def _verify_frozen_wp_upstream_files(
    root: Path, manifest: Mapping[str, Any], runtime: CandidateRuntime
) -> dict[str, Any]:
    accessor = manifest["frozen_wp_accessor"]
    project_rf = _verify_project_descriptor(
        root, accessor["rf_config_project"], "frozen WP project RF config"
    )
    bundled_rf = _verify_bundle_descriptor(
        runtime.bundle_root,
        manifest["artifacts"]["frozen_wp_rf_config"],
        "frozen WP bundled RF config",
    )
    if project_rf.read_bytes() != bundled_rf.read_bytes():
        raise ReleaseArtifactError("project and bundled frozen WP RF configs differ")
    config = load_rf_config(bundled_rf)
    verified: dict[str, dict[str, Any]] = {}
    for role in FROZEN_WP_INPUT_ROLES:
        descriptor = accessor["upstream_artifacts"][role]
        config_descriptor = config["inputs"][role]
        expected_config_descriptor = {
            key: descriptor[key]
            for key in ("path", "sha256", "required_status")
            if key in descriptor
        }
        if config_descriptor != expected_config_descriptor:
            raise ReleaseArtifactError(f"RF/upstream descriptor binding mismatch: {role}")
        path = _verify_project_descriptor(root, descriptor, f"frozen WP upstream {role}")
        verified[role] = {
            "path": str(path),
            "sha256": descriptor["sha256"],
            "size_bytes": descriptor["size_bytes"],
        }
    if (
        config["split_sha256"] != accessor["split_sha256"]
        or config["inputs"]["preprocessing_config"]["sha256"]
        != accessor["preprocessing_config_sha256"]
    ):
        raise ReleaseArtifactError("RF split/preprocessing binding mismatch")
    return {
        "rf_config_path": str(bundled_rf),
        "rf_config_sha256": accessor["rf_config_project"]["sha256"],
        "all_upstream_hashes_verified": True,
        "upstream_artifacts": verified,
    }


def _load_externally_hashed_json(
    path: Path,
    expected_sha256: str,
    expected_schema: str,
    error_type: type[ValueError],
    role: str,
) -> dict[str, Any]:
    if not _valid_sha256(expected_sha256):
        raise error_type(f"{role} external SHA-256 is invalid")
    try:
        raw = path.resolve(strict=True).read_bytes()
    except OSError as exc:
        raise error_type(f"cannot read {role}: {path}") from exc
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise error_type(f"{role} external SHA-256 mismatch")
    value = _parse_json_bytes(raw, error_type, role)
    if value.get("schema_version") != expected_schema:
        raise error_type(f"{role} schema is invalid")
    return value


def _parse_json_bytes(
    raw: bytes, error_type: type[ValueError], role: str
) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise error_type(f"cannot parse {role} JSON") from exc
    if not isinstance(value, dict):
        raise error_type(f"{role} JSON root must be an object")
    return value


def _verify_project_descriptor(root: Path, descriptor: Any, role: str) -> Path:
    if not isinstance(descriptor, Mapping):
        raise ReleaseArtifactError(f"{role} descriptor is invalid")
    relative = _strict_relative_path(descriptor.get("path"), role)
    path = _bound_project_file(root, relative, role)
    _verify_file_digest_and_size(path, descriptor, role)
    return path


def _validate_project_descriptor_shape(
    descriptor: Any,
    role: str,
    *,
    error_type: type[ValueError] = ReleaseArtifactError,
) -> None:
    if not isinstance(descriptor, Mapping) or set(descriptor) != {
        "path",
        "sha256",
        "size_bytes",
    }:
        raise error_type(f"{role} descriptor is invalid")
    _strict_relative_path(descriptor["path"], role, error_type=error_type)
    if not _valid_sha256(descriptor["sha256"]):
        raise error_type(f"{role} SHA-256 is invalid")
    size = descriptor["size_bytes"]
    if not isinstance(size, int) or isinstance(size, bool) or size < 1:
        raise error_type(f"{role} size is invalid")


def _project_file_descriptor(root: Path, path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if root not in resolved.parents or not resolved.is_file():
        raise ReleaseArtifactError("project descriptor target must be a file inside project_root")
    return {
        "path": resolved.relative_to(root).as_posix(),
        "sha256": _sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _verify_bundle_descriptor(root: Path, descriptor: Any, role: str) -> Path:
    if not isinstance(descriptor, Mapping) or set(descriptor) != {"path", "sha256", "size_bytes"}:
        raise CandidateManifestError(f"{role} descriptor is invalid")
    relative = _strict_relative_path(
        descriptor["path"], role, error_type=CandidateManifestError
    )
    try:
        path = (root / relative).resolve(strict=True)
    except OSError as exc:
        raise CandidateManifestError(f"cannot resolve {role}") from exc
    if root not in path.parents or not path.is_file():
        raise CandidateManifestError(f"{role} escapes candidate bundle")
    _verify_file_digest_and_size(path, descriptor, role, error_type=CandidateManifestError)
    return path


def _verify_file_digest_and_size(
    path: Path,
    descriptor: Any,
    role: str,
    *,
    error_type: type[ValueError] = ReleaseArtifactError,
) -> None:
    if not isinstance(descriptor, Mapping):
        raise error_type(f"{role} descriptor is invalid")
    sha256 = descriptor.get("sha256")
    size = descriptor.get("size_bytes")
    if not _valid_sha256(sha256):
        raise error_type(f"{role} SHA-256 is invalid")
    if not isinstance(size, int) or isinstance(size, bool) or size < 1:
        raise error_type(f"{role} size is invalid")
    if path.stat().st_size != size or _sha256_file(path) != sha256:
        raise error_type(f"{role} size or SHA-256 mismatch")


def _bound_project_file(root: Path, relative: Path, role: str) -> Path:
    try:
        path = (root / relative).resolve(strict=True)
    except OSError as exc:
        raise ReleaseArtifactError(f"cannot resolve {role}: {relative}") from exc
    if root not in path.parents or not path.is_file():
        raise ReleaseArtifactError(f"{role} must be a file inside project_root")
    return path


def _strict_relative_path(
    value: Any,
    role: str,
    *,
    error_type: type[ValueError] = ReleaseArtifactError,
) -> Path:
    if isinstance(value, Path):
        text = value.as_posix()
    elif isinstance(value, str):
        text = value
    else:
        raise error_type(f"{role} relative path is invalid")
    if not text or "\\" in text:
        raise error_type(f"{role} must use a non-empty POSIX relative path")
    path = Path(text)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise error_type(f"{role} must use a bounded relative path")
    return path


def _require_output_inside_root(root: Path, output: Path, role: str) -> None:
    if output == root or root not in output.parents:
        raise ReleaseArtifactError(f"{role} output must stay inside project_root")


def _validate_phase(expected_split: str) -> None:
    if expected_split not in ALLOWED_PHASES:
        raise WPReleaseDataError("expected_split must be validation or test")


def _finite_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise WPReleaseDataError(f"{name} must be numeric") from exc
    if array.shape != shape:
        raise WPReleaseDataError(f"{name} shape must be {shape}")
    if not np.isfinite(array).all():
        raise WPReleaseDataError(f"{name} must be finite")
    return np.ascontiguousarray(array)


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(np.float64, copy=False)
    try:
        return np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise WPReleaseDataError("WP evaluator logits must be numeric") from exc


def _classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, *, labels: Sequence[str]
) -> dict[str, Any]:
    if y_true.shape != y_pred.shape or y_true.ndim != 1 or y_true.size == 0:
        raise WPReleaseDataError("classification arrays must be aligned non-empty vectors")
    label_count = len(labels)
    if (
        np.any(y_true < 0)
        or np.any(y_true >= label_count)
        or np.any(y_pred < 0)
        or np.any(y_pred >= label_count)
    ):
        raise WPReleaseDataError("classification label index is out of range")
    matrix = np.zeros((label_count, label_count), dtype=np.int64)
    for true, predicted in zip(y_true.astype(np.int64), y_pred.astype(np.int64), strict=True):
        matrix[true, predicted] += 1
    per_class: dict[str, dict[str, Any]] = {}
    f1_values: list[float] = []
    recall_values: list[float] = []
    for index, name in enumerate(labels):
        true_positive = int(matrix[index, index])
        false_positive = int(matrix[:, index].sum() - true_positive)
        false_negative = int(matrix[index, :].sum() - true_positive)
        support = int(matrix[index, :].sum())
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": support,
        }
        f1_values.append(f1)
        recall_values.append(recall)
    return {
        "sample_count": int(y_true.size),
        "labels": list(labels),
        "accuracy": float(np.mean(y_true == y_pred)),
        "balanced_accuracy": float(np.mean(recall_values)),
        "macro_f1": float(np.mean(f1_values)),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }


def _metric_max_abs_difference(actual: Any, reference: Any, *, path: str) -> float:
    if isinstance(actual, Mapping):
        if not isinstance(reference, Mapping) or set(actual) != set(reference):
            raise WPReleaseDataError(f"parity metric schema differs at {path}")
        return max(
            (
                _metric_max_abs_difference(actual[key], reference[key], path=f"{path}.{key}")
                for key in actual
            ),
            default=0.0,
        )
    if isinstance(actual, list):
        if not isinstance(reference, list) or len(actual) != len(reference):
            raise WPReleaseDataError(f"parity metric list differs at {path}")
        return max(
            (
                _metric_max_abs_difference(item, reference[index], path=f"{path}[{index}]")
                for index, item in enumerate(actual)
            ),
            default=0.0,
        )
    if isinstance(actual, bool) or isinstance(reference, bool):
        if actual != reference:
            raise WPReleaseDataError(f"parity metric boolean differs at {path}")
        return 0.0
    if isinstance(actual, (int, float)) and isinstance(reference, (int, float)):
        return abs(float(actual) - float(reference))
    if actual != reference:
        raise WPReleaseDataError(f"parity metric value differs at {path}")
    return 0.0


def configure_release_cpu_runtime(runtime: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the exact scoring runtime and return an observed-value attestation."""

    if dict(runtime) != FIXED_CPU_RUNTIME:
        raise CandidateManifestError("release runtime contract must be fixed CPU 8/1 batch64")
    torch.set_num_threads(int(runtime["intra_op_threads"]))
    try:
        torch.set_num_interop_threads(int(runtime["inter_op_threads"]))
    except RuntimeError:
        if torch.get_num_interop_threads() != int(runtime["inter_op_threads"]):
            raise CandidateManifestError("torch inter-op thread count is already incompatible")
    observed = {
        "device": "cpu",
        "intra_op_threads": torch.get_num_threads(),
        "inter_op_threads": torch.get_num_interop_threads(),
        "batch_size": int(runtime["batch_size"]),
        "num_workers": int(runtime["num_workers"]),
        "pin_memory": bool(runtime["pin_memory"]),
    }
    if observed != FIXED_CPU_RUNTIME:
        raise CandidateManifestError("release runtime readback does not match fixed CPU contract")
    return observed


def _configure_release_cpu_runtime(runtime: Mapping[str, Any]) -> None:
    configure_release_cpu_runtime(runtime)


def _load_json_file(
    path: Path, error_type: type[ValueError], role: str
) -> dict[str, Any]:
    try:
        return _parse_json_bytes(path.read_bytes(), error_type, role)
    except OSError as exc:
        raise error_type(f"cannot read {role}: {path}") from exc


def _load_jsonl_file(
    path: Path, error_type: type[ValueError], role: str
) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        rows = [json.loads(line) for line in lines if line]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise error_type(f"cannot read {role}: {path}") from exc
    if any(not isinstance(row, dict) for row in rows):
        raise error_type(f"{role} rows must be JSON objects")
    return rows


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _canonical_jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_json_bytes(row) for row in rows)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON values must be finite")
        return value
    raise TypeError(f"value is not JSON serializable: {type(value)!r}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(_SHA256_CHARS)
    )


__all__ = [
    "ALLOWED_PHASES",
    "BINARY_LABELS",
    "CANDIDATE_MANIFEST_SCHEMA",
    "FIXED_CPU_RUNTIME",
    "CandidateBundleResult",
    "CandidateManifestError",
    "CandidateRuntime",
    "FrozenReleaseIdentity",
    "PATTERN_LABELS",
    "RELEASE_IMPLEMENTATION_SOURCE_FILES",
    "RELEASE_IDENTITY_SCHEMA",
    "ReleaseArtifactError",
    "SUBTYPE_LABELS",
    "TRAINING_IDENTITY_SCHEMA",
    "ValidatedWPRecord",
    "WPEvaluationResult",
    "WPReleaseDataError",
    "build_candidate_bundle",
    "canonical_wp_cohort_identity",
    "compare_primary_validation_parity",
    "configure_release_cpu_runtime",
    "evaluate_wp_predictions",
    "freeze_release_implementation_identity",
    "load_candidate_from_manifest",
    "run_authorized_frozen_wp_score",
    "run_primary_validation_parity",
    "run_wp_records_evaluation",
    "validate_wp_release_cohort",
    "validate_wp_release_record",
    "write_wp_evaluation_artifacts",
]
