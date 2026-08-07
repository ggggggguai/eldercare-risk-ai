"""Frozen RF/TCN compatibility reporting for wandering step 8."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from elderly_monitoring.modules.mental_health.wandering.augmentation import (
    commit_new_output_directory,
)
from elderly_monitoring.modules.mental_health.wandering.camera_adapter import (
    canonical_json_bytes,
    canonical_jsonl_bytes,
)
from elderly_monitoring.modules.mental_health.wandering.camera_corruption import (
    AUGMENTATION_CONFIG_SCHEMA_VERSION_V2,
    AUGMENTATION_CONFIG_SCHEMA_VERSION_V3,
    augmentation_config_sha256,
    load_augmentation_config,
    verify_augmentation_trust_roots,
)
from elderly_monitoring.modules.mental_health.wandering.camera_inference import (
    load_camera_config,
    load_trusted_camera_models,
    predict_camera_window,
)
from elderly_monitoring.modules.mental_health.wandering.preprocessing_bundle import (
    BUNDLE_MODE_DEVELOPMENT,
    load_preprocessing_bundle,
)


COMPATIBILITY_SCHEMA_VERSION = "wandering-anchorless-compatibility-v1"
COMPATIBILITY_PREDICTION_SCHEMA_VERSION = "wandering-compatibility-prediction-v1"
COMPATIBILITY_MANIFEST_SCHEMA_VERSION = "wandering-compatibility-manifest-v1"

_FOUR_CLASS_ORDER = ("direct", "pacing", "lapping", "random")
_BINARY_CLASS_ORDER = ("direct_or_non_wandering", "wandering_like")
_PROBABILITY_SUM_ABS_TOL = 1e-6
_AUGMENTATION_ARTIFACTS = (
    "augmentation_attempts.jsonl",
    "augmentation_report.json",
    "qc_faults.jsonl",
    "train_pairs.jsonl",
    "validation_pressure.jsonl",
)
_EXPECTED_STEP8_SOURCE_PATHS = {
    "camera_corruption.py": "src/elderly_monitoring/modules/mental_health/wandering/camera_corruption.py",
    "augmentation.py": "src/elderly_monitoring/modules/mental_health/wandering/augmentation.py",
    "compatibility_report.py": "src/elderly_monitoring/modules/mental_health/wandering/compatibility_report.py",
}


class CompatibilityReportError(ValueError):
    """A bundle, model binding, probability, or metric violates step 8."""


@dataclass(frozen=True)
class LoadedAugmentationBundle:
    bundle_dir: Path
    manifest: Mapping[str, Any]
    manifest_sha256: str
    train_pair_records: tuple[Mapping[str, Any], ...]
    pressure_records: tuple[Mapping[str, Any], ...]
    fault_records: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class CompatibilityBuildResult:
    output_dir: Path
    manifest_sha256: str
    prediction_count: int
    metric_group_count: int
    warning_count: int


@dataclass(frozen=True)
class VisualReviewEvidence:
    visual_review_dir: Path
    visual_manifest_sha256: str
    human_review_sha256: str
    selection_index_sha256: str
    human_review_schema_version: str
    human_review_status: str
    reviewed_pair_count: int


def jensen_shannon_base2(
    left: Sequence[float] | np.ndarray, right: Sequence[float] | np.ndarray
) -> float:
    """Return finite base-2 Jensen-Shannon divergence within [0,1]."""

    p = _probability_vector(left)
    q = _probability_vector(right)
    if p.shape != q.shape:
        raise CompatibilityReportError("probability vectors must have the same width")
    middle = (p + q) / 2.0

    def kl(value: np.ndarray) -> float:
        positive = value > 0.0
        return float(np.sum(value[positive] * np.log2(value[positive] / middle[positive])))

    result = 0.5 * kl(p) + 0.5 * kl(q)
    if not math.isfinite(result) or result < -1e-12 or result > 1.0 + 1e-12:
        raise CompatibilityReportError("Jensen-Shannon divergence left [0,1]")
    return min(1.0, max(0.0, result))


def summarize_prediction_groups(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Report coverage on all views and conditional metrics only on ready views."""

    clean_lookup: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in rows:
        if row.get("severity") == "clean" and row.get("view_status") == "ready":
            clean_lookup[(str(row["parent_sample_id"]), str(row["model"]), str(row["task"]))] = row
    grouped: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["source_dataset"]),
            str(row["task"]),
            str(row["model"]),
            str(row["severity"]),
        )
        grouped[key].append(row)
    summaries: list[dict[str, Any]] = []
    for (source, task, model, severity), group in sorted(grouped.items()):
        ready = [row for row in group if row.get("view_status") == "ready"]
        class_order = _FOUR_CLASS_ORDER if task == "four_class" else _BINARY_CLASS_ORDER
        classification = _classification_metrics(ready, class_order=class_order)
        paired = []
        for row in ready:
            clean = clean_lookup.get((str(row["parent_sample_id"]), model, task))
            if clean is not None:
                paired.append((clean, row))
        if severity == "clean":
            agreement = None
            confidence_delta = None
            js = None
        elif paired:
            agreement = float(
                np.mean(
                    [clean["predicted_label"] == corrupted["predicted_label"] for clean, corrupted in paired]
                )
            )
            confidence_delta = float(
                np.mean(
                    [
                        max(corrupted["probabilities"]) - max(clean["probabilities"])
                        for clean, corrupted in paired
                    ]
                )
            )
            js = float(
                np.mean(
                    [
                        jensen_shannon_base2(clean["probabilities"], corrupted["probabilities"])
                        for clean, corrupted in paired
                    ]
                )
            )
        else:
            agreement = confidence_delta = js = None
        summaries.append(
            {
                "source_dataset": source,
                "task": task,
                "model": model,
                "severity": severity,
                "coverage_denominator": len(group),
                "ready_count": len(ready),
                "unavailable_count": len(group) - len(ready),
                "ready_coverage": len(ready) / len(group) if group else 0.0,
                "conditional_denominator": len(ready),
                **classification,
                "paired_ready_denominator": len(paired) if severity != "clean" else None,
                "clean_corrupted_label_agreement": agreement,
                "mean_confidence_delta": confidence_delta,
                "mean_jensen_shannon_base2": js,
            }
        )
    return summaries


def load_augmentation_bundle(
    *,
    bundle_dir: str | Path,
    expected_manifest_sha256: str,
    config: Mapping[str, Any],
    project_root: str | Path,
) -> LoadedAugmentationBundle:
    """Verify the externally supplied manifest hash before opening artifacts."""

    directory = Path(bundle_dir).resolve(strict=True)
    if not directory.is_dir():
        raise CompatibilityReportError("augmentation bundle must be an existing directory")
    manifest_path = directory / "manifest.json"
    try:
        manifest_payload = manifest_path.read_bytes()
    except OSError as exc:
        raise CompatibilityReportError("augmentation manifest is missing") from exc
    actual_manifest_sha = hashlib.sha256(manifest_payload).hexdigest()
    if expected_manifest_sha256 != actual_manifest_sha:
        raise CompatibilityReportError("external augmentation manifest SHA-256 mismatch")
    manifest = _load_canonical_json_payload(manifest_payload, "augmentation manifest")
    if manifest.get("schema_version") != config["output_schemas"]["manifest"]:
        raise CompatibilityReportError("augmentation manifest schema drifted")
    if manifest.get("augmentation_config_sha256") != augmentation_config_sha256(config):
        raise CompatibilityReportError("augmentation config binding drifted")
    if manifest.get("trust_roots") != config.get("trust_roots"):
        raise CompatibilityReportError("augmentation trust-root bindings drifted")
    if (
        manifest.get("validation_scope") != "synthetic_camera_corruption"
        or manifest.get("official_source_opened") is not False
        or manifest.get("test_split_opened") is not False
        or manifest.get("model_training_performed") is not False
    ):
        raise CompatibilityReportError("augmentation scope or forbidden-access flags drifted")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or tuple(artifacts) != _AUGMENTATION_ARTIFACTS:
        raise CompatibilityReportError("augmentation artifact set/order drifted")
    for relative_name, descriptor in artifacts.items():
        path = _trusted_artifact(directory, relative_name)
        payload = path.read_bytes()
        if descriptor != {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "byte_count": len(payload),
        }:
            raise CompatibilityReportError(f"augmentation artifact binding mismatch: {relative_name}")
    _verify_step8_source_hashes(manifest, Path(project_root))
    train_pairs = tuple(_load_canonical_jsonl(directory / "train_pairs.jsonl"))
    pressure = tuple(_load_canonical_jsonl(directory / "validation_pressure.jsonl"))
    faults = tuple(_load_canonical_jsonl(directory / "qc_faults.jsonl"))
    if len(pressure) != 834:
        raise CompatibilityReportError("augmentation pressure count must be 834")
    roles = Counter(row.get("record_role") for row in faults)
    if roles != Counter(gated_fault=240, limitation_control=48):
        raise CompatibilityReportError("augmentation fault/control counts drifted")
    if any(row.get("split") != "validation" for row in pressure):
        raise CompatibilityReportError("non-validation record entered pressure bundle")
    if any(row.get("split") != "train" for row in train_pairs):
        raise CompatibilityReportError("non-train record entered paired augmentation bundle")
    if any(row.get("schema_version") != config["output_schemas"]["pair"] for row in train_pairs):
        raise CompatibilityReportError("augmentation pair schema drifted")
    if any(row.get("schema_version") != config["output_schemas"]["pressure"] for row in pressure):
        raise CompatibilityReportError("augmentation pressure schema drifted")
    if any(row.get("schema_version") != config["output_schemas"]["fault"] for row in faults):
        raise CompatibilityReportError("augmentation fault schema drifted")
    return LoadedAugmentationBundle(
        directory, manifest, actual_manifest_sha, train_pairs, pressure, faults
    )


def load_visual_review_evidence(
    *,
    visual_review_dir: str | Path,
    expected_visual_manifest_sha256: str,
    expected_human_review_sha256: str,
    expected_augmentation_manifest_sha256: str,
) -> VisualReviewEvidence:
    """Verify the immutable visual set and externally hashed human sign-off."""

    for role, value in (
        ("visual manifest", expected_visual_manifest_sha256),
        ("human review", expected_human_review_sha256),
        ("augmentation manifest", expected_augmentation_manifest_sha256),
    ):
        _validate_external_sha256(value, role)
    requested_root = Path(visual_review_dir)
    if requested_root.is_symlink():
        raise CompatibilityReportError("visual review directory must not be a symlink")
    try:
        root = requested_root.resolve(strict=True)
    except OSError as exc:
        raise CompatibilityReportError("visual review directory is missing") from exc
    if not root.is_dir():
        raise CompatibilityReportError("visual review directory must be a directory")

    visual_path = root / "figures/visual_manifest.json"
    try:
        visual_payload = visual_path.read_bytes()
    except OSError as exc:
        raise CompatibilityReportError("v3 visual manifest is missing") from exc
    actual_visual_sha = hashlib.sha256(visual_payload).hexdigest()
    if actual_visual_sha != expected_visual_manifest_sha256:
        raise CompatibilityReportError("external visual manifest SHA-256 mismatch")
    visual = _load_canonical_json_payload(visual_payload, "v3 visual manifest")
    if frozenset(visual) != {
        "schema_version",
        "augmentation_manifest_sha256",
        "artifacts",
        "human_review_contract",
    } or visual["schema_version"] != "wandering-augmentation-visual-manifest-v3":
        raise CompatibilityReportError("v3 visual manifest schema or fields drifted")
    if visual["augmentation_manifest_sha256"] != expected_augmentation_manifest_sha256:
        raise CompatibilityReportError("visual/augmentation manifest binding mismatch")
    if visual["human_review_contract"] != {
        "path": "HUMAN_REVIEW.md",
        "initial_status": "pending_human_review",
        "required_final_status": "human_review_passed",
        "final_sha256_source": "external_cli",
    }:
        raise CompatibilityReportError("visual human-review contract drifted")
    artifacts = visual["artifacts"]
    if not isinstance(artifacts, dict):
        raise CompatibilityReportError("visual artifacts must be an object")
    for relative, descriptor in artifacts.items():
        artifact = _trusted_visual_artifact(root, relative)
        payload = artifact.read_bytes()
        if descriptor != {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "byte_count": len(payload),
        }:
            raise CompatibilityReportError(f"visual artifact binding mismatch: {relative}")
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path != visual_path
        and path != root / "HUMAN_REVIEW.md"
    }
    if set(artifacts) != actual_files:
        raise CompatibilityReportError("visual artifact exact file set drifted")

    selection_path = _trusted_visual_artifact(root, "figures/selection_index.json")
    selection_payload = selection_path.read_bytes()
    selection_sha = hashlib.sha256(selection_payload).hexdigest()
    selection = _load_canonical_json_payload(selection_payload, "v3 visual selection")
    _validate_visual_selection(
        selection,
        artifacts=artifacts,
        expected_augmentation_manifest_sha256=expected_augmentation_manifest_sha256,
    )

    human_path = root / "HUMAN_REVIEW.md"
    try:
        human_payload = human_path.read_bytes()
    except OSError as exc:
        raise CompatibilityReportError("HUMAN_REVIEW.md is missing") from exc
    actual_human_sha = hashlib.sha256(human_payload).hexdigest()
    if actual_human_sha != expected_human_review_sha256:
        raise CompatibilityReportError("external human-review SHA-256 mismatch")
    header = _parse_human_review_header(human_payload)
    if header["augmentation_manifest_sha256"] != expected_augmentation_manifest_sha256:
        raise CompatibilityReportError("human/augmentation manifest binding mismatch")
    if header["visual_manifest_sha256"] != expected_visual_manifest_sha256:
        raise CompatibilityReportError("human/visual manifest binding mismatch")
    if header["selection_index_sha256"] != selection_sha:
        raise CompatibilityReportError("human/selection index binding mismatch")
    if header["status"] != "human_review_passed" or header["final_decision"] != "human_review_passed":
        raise CompatibilityReportError("human review must be human_review_passed")
    if header["reviewed_pair_count"] != 440:
        raise CompatibilityReportError("human review must cover exactly 440 pairs")
    if header["rejected_child_ids"] != []:
        raise CompatibilityReportError("human review contains unresolved rejected child IDs")
    if header["reviewer"] in {"", "pending"}:
        raise CompatibilityReportError("human reviewer is not signed")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", header["review_date"]):
        raise CompatibilityReportError("human review date must be YYYY-MM-DD")
    return VisualReviewEvidence(
        visual_review_dir=root,
        visual_manifest_sha256=actual_visual_sha,
        human_review_sha256=actual_human_sha,
        selection_index_sha256=selection_sha,
        human_review_schema_version=str(header["schema_version"]),
        human_review_status=str(header["status"]),
        reviewed_pair_count=int(header["reviewed_pair_count"]),
    )


def build_compatibility_report(
    *,
    config_path: str | Path,
    project_root: str | Path,
    augmentation_bundle: str | Path,
    expected_augmentation_manifest_sha256: str,
    visual_review_dir: str | Path,
    expected_visual_manifest_sha256: str,
    expected_human_review_sha256: str,
    rf_development_dir: str | Path,
    expected_rf_development_manifest_sha256: str,
    tcn_development_dir: str | Path,
    expected_tcn_development_manifest_sha256: str,
    output_dir: str | Path,
) -> CompatibilityBuildResult:
    """Run four frozen primary-seed models separately; never train or fuse."""

    output = Path(output_dir)
    if output.exists():
        raise CompatibilityReportError(f"output directory already exists: {output}")
    for role, value in (
        ("augmentation manifest", expected_augmentation_manifest_sha256),
        ("visual manifest", expected_visual_manifest_sha256),
        ("human review", expected_human_review_sha256),
        ("RF development manifest", expected_rf_development_manifest_sha256),
        ("TCN development manifest", expected_tcn_development_manifest_sha256),
    ):
        _validate_external_sha256(value, role)
    try:
        config = load_augmentation_config(config_path)
        loaded = load_augmentation_bundle(
            bundle_dir=augmentation_bundle,
            expected_manifest_sha256=expected_augmentation_manifest_sha256,
            config=config,
            project_root=project_root,
        )
        visual_evidence = load_visual_review_evidence(
            visual_review_dir=visual_review_dir,
            expected_visual_manifest_sha256=expected_visual_manifest_sha256,
            expected_human_review_sha256=expected_human_review_sha256,
            expected_augmentation_manifest_sha256=expected_augmentation_manifest_sha256,
        )
        trusted = verify_augmentation_trust_roots(config, project_root)
        if expected_rf_development_manifest_sha256 != config["trust_roots"]["rf_development_manifest"]["sha256"]:
            raise CompatibilityReportError("external RF development manifest SHA-256 mismatch")
        if expected_tcn_development_manifest_sha256 != config["trust_roots"]["tcn_development_manifest"]["sha256"]:
            raise CompatibilityReportError("external TCN development manifest SHA-256 mismatch")
        camera_config = load_camera_config(trusted["camera_config"])
        models = load_trusted_camera_models(
            config=camera_config,
            project_root=project_root,
            rf_development_dir=rf_development_dir,
            expected_rf_development_manifest_sha256=expected_rf_development_manifest_sha256,
            tcn_development_dir=tcn_development_dir,
            expected_tcn_development_manifest_sha256=expected_tcn_development_manifest_sha256,
        )
        development = load_preprocessing_bundle(
            rf_config_path=trusted["rf_config"],
            project_root=project_root,
            mode=BUNDLE_MODE_DEVELOPMENT,
        )
    except CompatibilityReportError:
        raise
    except Exception as exc:
        raise CompatibilityReportError("compatibility trust chain or safe model load failed") from exc
    parents = {
        str(row["sample_id"]): row
        for row in development.records_for_split("validation")
    }
    if len(parents) != 278:
        raise CompatibilityReportError("compatibility clean parent count drifted")
    prediction_schema, compatibility_manifest_schema = compatibility_output_schemas(config)
    primary_seed = int(config["compatibility"]["primary_seed"])

    predictions: list[dict[str, Any]] = []
    for parent_id in sorted(parents):
        parent = parents[parent_id]
        prepared = {
            "window_id": f"clean-{parent_id}",
            "window_status": "ready",
            "quality_flags": [],
            "shape_normalized_points": parent["shape_normalized_points"],
            "point_mask": parent["point_mask"],
            "raw_features": parent["raw_features"],
            "model_features": parent["model_features"],
        }
        result = predict_camera_window(
            prepared, models, validation_scope="synthetic_camera_corruption"
        )
        predictions.extend(
            _flatten_predictions(
                parent,
                None,
                "clean",
                result,
                prediction_schema=prediction_schema,
                primary_seed=primary_seed,
            )
        )
    for pressure in loaded.pressure_records:
        parent_id = str(pressure["parent_sample_id"])
        parent = parents.get(parent_id)
        if parent is None:
            raise CompatibilityReportError("pressure parent is outside frozen validation")
        if pressure.get("qc_status") == "ready":
            prepared = {
                "window_id": pressure["window_id"],
                "window_status": "ready",
                "quality_flags": list(pressure.get("quality_flags", [])),
                "shape_normalized_points": pressure["shape_normalized_points"],
                "point_mask": pressure["point_mask"],
                "raw_features": pressure["raw_features"],
                "model_features": pressure["model_features"],
            }
            result = predict_camera_window(
                prepared, models, validation_scope="synthetic_camera_corruption"
            )
        else:
            result = {
                "window_status": "unavailable",
                "reason_codes": list(pressure.get("qc_reason_codes", [])),
                "predictions": None,
                "model_invocation_skipped": True,
            }
        predictions.extend(
            _flatten_predictions(
                parent,
                pressure,
                str(pressure["severity"]),
                result,
                prediction_schema=prediction_schema,
                primary_seed=primary_seed,
            )
        )
    predictions.sort(
        key=lambda row: (
            str(row["parent_sample_id"]),
            ("clean", "low", "medium", "high").index(str(row["severity"])),
            str(row["model"]),
        )
    )
    groups = summarize_prediction_groups(predictions)
    warnings = _compatibility_warnings(groups, config["compatibility"])
    fault_metrics = _fault_metrics(loaded.fault_records)
    metrics = {
        "schema_version": config["output_schemas"]["compatibility"],
        "validation_scope": "synthetic_camera_corruption",
        "primary_seed": int(config["compatibility"]["primary_seed"]),
        "model_purpose": "comparison_only",
        "fusion_performed": False,
        "training_performed": False,
        "groups": groups,
        "observation_targets": {
            "low_label_agreement": float(config["compatibility"]["low_label_agreement_target"]),
            "medium_label_agreement": float(config["compatibility"]["medium_label_agreement_target"]),
            "medium_macro_f1_max_drop": float(config["compatibility"]["medium_macro_f1_max_drop_target"]),
        },
        "model_compatibility_warnings": warnings,
        "gated_fault_results": fault_metrics["gated_fault_results"],
        "limitation_control_result": fault_metrics["limitation_control_result"],
        "calibration_status": "not_run_step12",
        "prototype_energy_status": "not_run_step12",
        "uncertain_status": "not_run_step12",
        "capability_statements": [
            "general synthetic corruption is not a measured target-camera error distribution",
            "bbox-only QC can reject observable ID discontinuities but cannot identify geometrically continuous identity switches",
        ],
    }
    failure_cases = _failure_cases(predictions)
    files: dict[str, bytes] = {
        "predictions.jsonl": canonical_jsonl_bytes(predictions),
        "metrics.json": canonical_json_bytes(metrics),
        "failure_cases.jsonl": canonical_jsonl_bytes(failure_cases),
    }
    manifest = {
        "schema_version": compatibility_manifest_schema,
        "validation_scope": "synthetic_camera_corruption",
        "augmentation_manifest_sha256": loaded.manifest_sha256,
        "visual_manifest_sha256": visual_evidence.visual_manifest_sha256,
        "human_review_sha256": visual_evidence.human_review_sha256,
        "rf_development_manifest_sha256": expected_rf_development_manifest_sha256,
        "tcn_development_manifest_sha256": expected_tcn_development_manifest_sha256,
        "human_review_schema_version": visual_evidence.human_review_schema_version,
        "human_review_status": visual_evidence.human_review_status,
        "selection_index_sha256": visual_evidence.selection_index_sha256,
        "reviewed_pair_count": visual_evidence.reviewed_pair_count,
        "primary_seed": int(config["compatibility"]["primary_seed"]),
        "artifacts": {
            name: {"sha256": hashlib.sha256(payload).hexdigest(), "byte_count": len(payload)}
            for name, payload in files.items()
        },
        "model_training_performed": False,
        "fusion_performed": False,
        "test_or_official_used": False,
    }
    files["manifest.json"] = canonical_json_bytes(manifest)
    try:
        commit_new_output_directory(output, files)
    except ValueError as exc:
        raise CompatibilityReportError("compatibility output commit failed") from exc
    return CompatibilityBuildResult(
        output_dir=output,
        manifest_sha256=hashlib.sha256(files["manifest.json"]).hexdigest(),
        prediction_count=len(predictions),
        metric_group_count=len(groups),
        warning_count=len(warnings),
    )


def _flatten_predictions(
    parent: Mapping[str, Any],
    pressure: Mapping[str, Any] | None,
    severity: str,
    result: Mapping[str, Any],
    *,
    prediction_schema: str,
    primary_seed: int,
) -> list[dict[str, Any]]:
    source = str(parent["source_dataset"])
    applicable = (
        ("rf_four_class", "tcn_four_class", "rf_binary", "tcn_binary")
        if source == "wandering_patterns"
        else ("rf_binary", "tcn_binary")
    )
    output: list[dict[str, Any]] = []
    for model_name in applicable:
        task = "four_class" if model_name.endswith("four_class") else "binary"
        true_label = (
            str(parent["pattern_label"])
            if task == "four_class"
            else _BINARY_CLASS_ORDER[int(parent["binary_label"])]
        )
        prediction = (
            result["predictions"][model_name]
            if result.get("window_status") == "ready"
            else None
        )
        child_id = None if pressure is None else pressure["child_id"]
        identity = f"{parent['sample_id']}|{child_id}|{severity}|{model_name}"
        output.append(
            {
                "schema_version": prediction_schema,
                "prediction_id": hashlib.sha256(identity.encode("utf-8")).hexdigest(),
                "parent_sample_id": parent["sample_id"],
                "child_id": child_id,
                "source_dataset": source,
                "task": task,
                "model": model_name,
                "severity": severity,
                "view_status": str(result["window_status"]),
                "reason_codes": list(result.get("reason_codes", [])),
                "semantic_status": "clean"
                if pressure is None
                else str(pressure["semantic_status"]),
                "true_label": true_label,
                "class_order": list(
                    _FOUR_CLASS_ORDER if task == "four_class" else _BINARY_CLASS_ORDER
                ),
                "predicted_label": None if prediction is None else prediction["predicted_label"],
                "probabilities": None if prediction is None else prediction["probabilities"],
                "model_invocation_skipped": prediction is None,
                "primary_seed": primary_seed,
                "probability_calibrated": False,
                "fusion_performed": False,
                "validation_scope": "synthetic_camera_corruption",
            }
        )
    return output


def _classification_metrics(
    rows: Sequence[Mapping[str, Any]], *, class_order: Sequence[str]
) -> dict[str, Any]:
    matrix = np.zeros((len(class_order), len(class_order)), dtype=np.int64)
    index = {label: position for position, label in enumerate(class_order)}
    for row in rows:
        true = str(row["true_label"])
        predicted = str(row["predicted_label"])
        if true not in index or predicted not in index:
            raise CompatibilityReportError("prediction label is outside frozen class order")
        matrix[index[true], index[predicted]] += 1
    recalls: list[float] = []
    f1_values: list[float] = []
    for position in range(len(class_order)):
        tp = int(matrix[position, position])
        fn = int(matrix[position, :].sum()) - tp
        fp = int(matrix[:, position].sum()) - tp
        recalls.append(tp / (tp + fn) if tp + fn else 0.0)
        denominator = 2 * tp + fp + fn
        f1_values.append(2 * tp / denominator if denominator else 0.0)
    return {
        "class_order": list(class_order),
        "conditional_macro_f1": float(np.mean(f1_values)) if rows else None,
        "conditional_balanced_accuracy": float(np.mean(recalls)) if rows else None,
        "conditional_confusion_matrix": matrix.tolist(),
    }


def _compatibility_warnings(
    groups: Sequence[Mapping[str, Any]], compatibility: Mapping[str, Any]
) -> list[dict[str, Any]]:
    by_key = {
        (row["source_dataset"], row["task"], row["model"], row["severity"]): row
        for row in groups
    }
    warnings: list[dict[str, Any]] = []
    base_keys = sorted({key[:3] for key in by_key})
    for key in base_keys:
        clean = by_key.get((*key, "clean"))
        low = by_key.get((*key, "low"))
        medium = by_key.get((*key, "medium"))
        if clean is None or low is None or medium is None:
            raise CompatibilityReportError("clean/low/medium metric group is missing")
        if (
            low["clean_corrupted_label_agreement"] is not None
            and low["clean_corrupted_label_agreement"]
            < float(compatibility["low_label_agreement_target"])
        ):
            warnings.append({"group": list(key), "reason": "low_label_agreement_below_target"})
        if (
            medium["clean_corrupted_label_agreement"] is not None
            and medium["clean_corrupted_label_agreement"]
            < float(compatibility["medium_label_agreement_target"])
        ):
            warnings.append({"group": list(key), "reason": "medium_label_agreement_below_target"})
        clean_f1 = clean["conditional_macro_f1"]
        medium_f1 = medium["conditional_macro_f1"]
        if (
            clean_f1 is not None
            and medium_f1 is not None
            and clean_f1 - medium_f1
            > float(compatibility["medium_macro_f1_max_drop_target"])
        ):
            warnings.append({"group": list(key), "reason": "medium_macro_f1_drop_above_target"})
    return warnings


def _fault_metrics(faults: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    gated: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    controls: list[Mapping[str, Any]] = []
    for row in faults:
        if row["record_role"] == "gated_fault":
            gated[str(row["fault_or_control_type"])].append(row)
        else:
            controls.append(row)
    gated_results = []
    for fault_type, rows in sorted(gated.items()):
        missed = sorted(
            str(row["parent_sample_id"])
            for row in rows
            if row["observed_qc_status"] != "unavailable"
        )
        gated_results.append(
            {
                "fault_type": fault_type,
                "denominator": len(rows),
                "qc_rejected_count": len(rows) - len(missed),
                "qc_rejection_rate": (len(rows) - len(missed)) / len(rows),
                "missed_parent_sample_ids": missed,
            }
        )
    return {
        "gated_fault_results": gated_results,
        "limitation_control_result": {
            "control_type": "geometrically_continuous_splice_control",
            "denominator": len(controls),
            "ready_count": sum(row["observed_qc_status"] == "ready" for row in controls),
            "unavailable_count": sum(
                row["observed_qc_status"] == "unavailable" for row in controls
            ),
            "reason_counts": dict(
                sorted(
                    Counter(
                        reason
                        for row in controls
                        for reason in row["observed_qc_reason_codes"]
                    ).items()
                )
            ),
            "identity_recognition_accuracy": None,
            "capability_boundary": "bbox-only QC can reject observable ID discontinuities but cannot identify geometrically continuous identity switches",
        },
    }


def _failure_cases(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    clean = {
        (str(row["parent_sample_id"]), str(row["model"]), str(row["task"])): row
        for row in rows
        if row["severity"] == "clean"
    }
    failures: list[dict[str, Any]] = []
    for row in rows:
        reasons: list[str] = []
        if row["view_status"] != "ready":
            reasons.append("unavailable")
        elif row["predicted_label"] != row["true_label"]:
            reasons.append("misclassified")
        if row["severity"] != "clean" and row["view_status"] == "ready":
            base = clean[(str(row["parent_sample_id"]), str(row["model"]), str(row["task"]))]
            if row["predicted_label"] != base["predicted_label"]:
                reasons.append("clean_corrupted_disagreement")
        if reasons:
            failures.append(
                {
                    "prediction_id": row["prediction_id"],
                    "parent_sample_id": row["parent_sample_id"],
                    "child_id": row["child_id"],
                    "source_dataset": row["source_dataset"],
                    "task": row["task"],
                    "model": row["model"],
                    "severity": row["severity"],
                    "failure_reasons": reasons,
                    "qc_reason_codes": row["reason_codes"],
                    "true_label": row["true_label"],
                    "predicted_label": row["predicted_label"],
                }
            )
    return failures


def _probability_vector(value: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if (
        array.ndim != 1
        or len(array) < 2
        or not np.isfinite(array).all()
        or np.any(array < 0.0)
        or not math.isclose(
            float(array.sum()),
            1.0,
            rel_tol=0.0,
            abs_tol=_PROBABILITY_SUM_ABS_TOL,
        )
    ):
        raise CompatibilityReportError("probability vector must be finite and sum to one")
    return array


def _trusted_artifact(root: Path, relative_name: str) -> Path:
    relative = PurePosixPath(relative_name)
    if relative.is_absolute() or any(part in ("", ".", "..") for part in relative.parts):
        raise CompatibilityReportError("augmentation artifact path is invalid")
    path = root.joinpath(*relative.parts).resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise CompatibilityReportError("augmentation artifact escapes bundle") from exc
    if not path.is_file():
        raise CompatibilityReportError("augmentation artifact is not a file")
    return path


def _trusted_visual_artifact(root: Path, relative_name: str) -> Path:
    relative = PurePosixPath(relative_name)
    if relative.is_absolute() or any(part in ("", ".", "..") for part in relative.parts):
        raise CompatibilityReportError("visual artifact path is invalid")
    lexical = root.joinpath(*relative.parts)
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise CompatibilityReportError("visual artifact path contains a symlink")
    try:
        path = lexical.resolve(strict=True)
        path.relative_to(root)
    except (OSError, ValueError) as exc:
        raise CompatibilityReportError("visual artifact path escapes review root") from exc
    if not path.is_file():
        raise CompatibilityReportError("visual artifact is not a file")
    return path


def _validate_visual_selection(
    selection: Mapping[str, Any],
    *,
    artifacts: Mapping[str, Any],
    expected_augmentation_manifest_sha256: str,
) -> None:
    if frozenset(selection) != {
        "schema_version",
        "selection",
        "augmentation_manifest_sha256",
        "records",
    }:
        raise CompatibilityReportError("v3 visual selection fields drifted")
    if (
        selection["schema_version"] != "wandering-augmentation-visual-selection-v3"
        or selection["selection"] != "sha256_child_id_ascending"
        or selection["augmentation_manifest_sha256"]
        != expected_augmentation_manifest_sha256
    ):
        raise CompatibilityReportError("v3 visual selection contract drifted")
    records = selection["records"]
    if not isinstance(records, list) or len(records) != 440:
        raise CompatibilityReportError("v3 visual selection must contain 440 pairs")
    expected_groups = [
        ("wandering_patterns", label, severity, 50)
        for label in _FOUR_CLASS_ORDER
        for severity in ("low", "medium")
    ] + [
        ("smartcare", label, severity, 10)
        for label in (0, 1)
        for severity in ("low", "medium")
    ]
    expected_sequence = [
        (source, label, severity)
        for source, label, severity, count in expected_groups
        for _ in range(count)
    ]
    actual_sequence: list[tuple[str, Any, str]] = []
    child_ids: set[str] = set()
    figures: set[str] = set()
    by_group: dict[tuple[str, Any, str], list[str]] = defaultdict(list)
    for row in records:
        if not isinstance(row, dict) or frozenset(row) != {
            "source_dataset",
            "label",
            "severity",
            "child_id",
            "parent_sample_id",
            "figure",
        }:
            raise CompatibilityReportError("v3 visual selection record fields drifted")
        key = (str(row["source_dataset"]), row["label"], str(row["severity"]))
        actual_sequence.append(key)
        child_id = str(row["child_id"])
        figure = str(row["figure"])
        if child_id in child_ids or figure in figures:
            raise CompatibilityReportError("v3 visual selection IDs or figures are duplicated")
        child_ids.add(child_id)
        figures.add(figure)
        by_group[key].append(child_id)
        if figure not in artifacts or not figure.endswith(".png"):
            raise CompatibilityReportError("v3 visual selection figure binding is missing")
    if actual_sequence != expected_sequence:
        raise CompatibilityReportError("v3 visual selection strata/order drifted")
    for child_group in by_group.values():
        if child_group != sorted(
            child_group, key=lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
        ):
            raise CompatibilityReportError("v3 visual selection child hash order drifted")
    if "README.md" not in artifacts or "figures/selection_index.json" not in artifacts:
        raise CompatibilityReportError("v3 visual README or selection binding is missing")
    if any(
        relative not in {"README.md", "figures/selection_index.json"}
        and not relative.endswith(".png")
        for relative in artifacts
    ):
        raise CompatibilityReportError("v3 visual artifact set contains an unexpected type")


def _parse_human_review_header(payload: bytes) -> dict[str, Any]:
    try:
        text = payload.decode("utf-8")
    except UnicodeError as exc:
        raise CompatibilityReportError("HUMAN_REVIEW.md is not UTF-8") from exc
    marker = "<!-- wandering-augmentation-human-review-v3\n"
    if not text.startswith(marker) or text.count(marker) != 1 or text.count("\n-->") != 1:
        raise CompatibilityReportError("human review must contain one v3 machine header")
    header_text, _body = text[len(marker) :].split("\n-->", 1)

    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CompatibilityReportError(f"duplicate human-review header key: {key}")
            result[key] = value
        return result

    try:
        header = json.loads(header_text, object_pairs_hook=reject_duplicates)
    except json.JSONDecodeError as exc:
        raise CompatibilityReportError("human-review machine header is invalid JSON") from exc
    expected_fields = {
        "schema_version",
        "status",
        "augmentation_manifest_sha256",
        "visual_manifest_sha256",
        "selection_index_sha256",
        "reviewed_pair_count",
        "reviewer",
        "review_date",
        "rejected_child_ids",
        "final_decision",
    }
    if not isinstance(header, dict) or set(header) != expected_fields:
        raise CompatibilityReportError("human-review machine header fields drifted")
    if header["schema_version"] != "wandering-augmentation-human-review-v3":
        raise CompatibilityReportError("human-review schema drifted")
    for field in (
        "status",
        "augmentation_manifest_sha256",
        "visual_manifest_sha256",
        "selection_index_sha256",
        "reviewer",
        "review_date",
        "final_decision",
    ):
        if not isinstance(header[field], str):
            raise CompatibilityReportError(f"human-review field must be a string: {field}")
    for field in (
        "augmentation_manifest_sha256",
        "visual_manifest_sha256",
        "selection_index_sha256",
    ):
        _validate_external_sha256(header[field], f"human-review {field}")
    if isinstance(header["reviewed_pair_count"], bool) or not isinstance(
        header["reviewed_pair_count"], int
    ):
        raise CompatibilityReportError("human-review reviewed_pair_count must be an integer")
    rejected = header["rejected_child_ids"]
    if not isinstance(rejected, list) or any(not isinstance(value, str) for value in rejected):
        raise CompatibilityReportError("human-review rejected_child_ids must be a string list")
    if len(rejected) != len(set(rejected)):
        raise CompatibilityReportError("human-review rejected_child_ids are duplicated")
    return header


def _validate_external_sha256(value: Any, role: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CompatibilityReportError(f"external {role} SHA-256 is invalid")


def compatibility_output_schemas(config: Mapping[str, Any]) -> tuple[str, str]:
    if config["schema_version"] == AUGMENTATION_CONFIG_SCHEMA_VERSION_V3:
        return "wandering-compatibility-prediction-v3", "wandering-compatibility-manifest-v3"
    if config["schema_version"] == AUGMENTATION_CONFIG_SCHEMA_VERSION_V2:
        return "wandering-compatibility-prediction-v2", "wandering-compatibility-manifest-v2"
    return COMPATIBILITY_PREDICTION_SCHEMA_VERSION, COMPATIBILITY_MANIFEST_SCHEMA_VERSION


def _verify_step8_source_hashes(manifest: Mapping[str, Any], project_root: Path) -> None:
    bindings = manifest.get("step8_source_hashes")
    if not isinstance(bindings, dict) or set(bindings) != set(_EXPECTED_STEP8_SOURCE_PATHS):
        raise CompatibilityReportError("Step-8 source binding set drifted")
    root = project_root.resolve(strict=True)
    for name, relative_name in _EXPECTED_STEP8_SOURCE_PATHS.items():
        descriptor = bindings.get(name)
        if not isinstance(descriptor, dict) or descriptor.get("path") != relative_name:
            raise CompatibilityReportError(f"Step-8 source binding path drifted: {name}")
        path = root.joinpath(*PurePosixPath(relative_name).parts).resolve(strict=True)
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise CompatibilityReportError("Step-8 source binding escapes project root") from exc
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if descriptor.get("sha256") != actual:
            raise CompatibilityReportError(f"Step-8 source binding mismatch: {name}")


def _load_canonical_json_payload(payload: bytes, role: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                CompatibilityReportError(f"non-finite JSON constant in {role}: {token}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CompatibilityReportError(f"cannot parse {role}") from exc
    if not isinstance(value, dict) or canonical_json_bytes(value) != payload:
        raise CompatibilityReportError(f"{role} is not canonical JSON")
    return value


def _load_canonical_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise CompatibilityReportError(f"cannot read JSONL artifact: {path.name}") from exc
    rows: list[dict[str, Any]] = []
    for line in payload.splitlines(keepends=True):
        row = _load_canonical_json_payload(line, path.name)
        rows.append(row)
    if canonical_jsonl_bytes(rows) != payload:
        raise CompatibilityReportError(f"{path.name} is not canonical JSONL")
    return rows


__all__ = [
    "COMPATIBILITY_MANIFEST_SCHEMA_VERSION",
    "COMPATIBILITY_PREDICTION_SCHEMA_VERSION",
    "COMPATIBILITY_SCHEMA_VERSION",
    "CompatibilityBuildResult",
    "CompatibilityReportError",
    "LoadedAugmentationBundle",
    "VisualReviewEvidence",
    "build_compatibility_report",
    "compatibility_output_schemas",
    "jensen_shannon_base2",
    "load_augmentation_bundle",
    "load_visual_review_evidence",
    "summarize_prediction_groups",
]
