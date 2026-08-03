"""Validate MODEL-002 artifacts, OOF coverage, and fold leakage boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

from elderly_monitoring.modules.mental_health.mood_social.experts.sleep import (
    SLEEP_DATASET_IDS,
    SLEEP_FEATURE_COLUMNS,
    SLEEP_MASK_COLUMNS,
    EXPECTED_FEATURE_SCHEMA_SHA256,
    EXPECTED_SPLIT_SHA256,
    MODEL_BUNDLE_VERSION,
    MODEL_ID,
    MODEL_VERSION,
    SleepExpertBundle,
    binary_metrics,
    load_sleep_training_config,
    load_sleep_training_inputs,
    participant_set_sha256,
)
from elderly_monitoring.modules.mental_health.mood_social.feature_schema import (
    FEATURE_SCHEMA_VERSION,
)
from elderly_monitoring.modules.mental_health.mood_social.splits import SPLIT_ID


DEFAULT_CONFIG = Path("configs/training/mood_social_sleep_expert_v3_3_3.yaml")
DEFAULT_REPORT = Path("reports/mental_health/mood_social/MH-20260731-002")
DEFAULT_MODEL = Path("models/mental_health/mood_social/v3.3.3/sleep_expert.joblib")
DEFAULT_MODEL_MANIFEST = Path(
    "models/mental_health/mood_social/v3.3.3/sleep_expert_manifest.json"
)
PREDICTION_COLUMNS = (
    "prediction_schema_version",
    "prediction_id",
    "dataset_id",
    "global_participant_id",
    "canonical_row_index",
    "binary_target",
    "outer_fold",
    "raw_probability",
    "calibrated_probability",
    "expert_mask",
    "observed_feature_count",
    "observed_feature_fraction",
)


class ValidationError(RuntimeError):
    """Raised without including participant-level values in the message."""


def validate(
    repository_root: Path,
    report_dir: Path,
    model_path: Path,
    model_manifest_path: Path,
    config_path: Path,
) -> dict[str, Any]:
    checks: list[str] = []
    root = repository_root.resolve()
    report = report_dir.resolve()
    model_file = model_path.resolve()
    model_manifest_file = model_manifest_path.resolve()
    config = load_sleep_training_config(config_path)
    inputs = load_sleep_training_inputs(root)
    assignments = {
        str(row["global_participant_id"]): row
        for row in inputs.assignments.to_dict(orient="records")
    }
    _check(report.is_dir(), "report_directory", checks)
    _check(model_file.is_file(), "production_model_file", checks)
    _check(model_manifest_file.is_file(), "model_manifest_file", checks)

    run = _read_json(report / "run.json")
    metrics = _read_json(report / "metrics.json")
    model_manifest = _read_json(model_manifest_file)
    model_artifact = _read_json(report / "model_artifact.json")
    warnings = _read_json(report / "failures" / "warnings.json")
    _check(
        run.get("status") == "completed"
        and run.get("task_id") == "MODEL-002"
        and run.get("model_id") == MODEL_ID
        and run.get("model_version") == MODEL_VERSION
        and run.get("run_id") == config.run_id,
        "run_identity",
        checks,
    )
    _check(
        run.get("split", {}).get("split_id") == SPLIT_ID
        and run.get("split", {}).get("sha256") == EXPECTED_SPLIT_SHA256,
        "run_split_binding",
        checks,
    )
    _check(
        run.get("feature_schema", {}).get("version") == FEATURE_SCHEMA_VERSION
        and run.get("feature_schema", {}).get("sha256")
        == EXPECTED_FEATURE_SCHEMA_SHA256,
        "run_schema_binding",
        checks,
    )
    _check(
        run.get("failures") == []
        and isinstance(warnings.get("warnings"), list)
        and warnings.get("failures") == [],
        "failure_and_warning_contract",
        checks,
    )
    _check(
        model_manifest.get("status") == "active"
        and model_manifest.get("model_id") == MODEL_ID
        and model_manifest.get("model_version") == MODEL_VERSION
        and model_manifest.get("run_id") == config.run_id
        and model_manifest.get("dataset_ids") == list(SLEEP_DATASET_IDS),
        "model_manifest_identity",
        checks,
    )
    _check(
        model_manifest.get("split_id") == SPLIT_ID
        and model_manifest.get("split_manifest_sha256") == EXPECTED_SPLIT_SHA256
        and model_manifest.get("feature_schema_version") == FEATURE_SCHEMA_VERSION
        and model_manifest.get("feature_schema_sha256")
        == EXPECTED_FEATURE_SCHEMA_SHA256
        and model_manifest.get("source_transform")
        == "canonical_same_semantics_identity"
        and "ecdf" not in model_manifest,
        "model_manifest_frozen_bindings",
        checks,
    )
    model_sha256 = _sha256_file(model_file)
    _check(
        model_manifest.get("model_sha256") == model_sha256
        and model_artifact.get("model_sha256") == model_sha256
        and run.get("artifacts", {}).get("model_sha256") == model_sha256,
        "model_hash_binding",
        checks,
    )
    _check(
        model_manifest.get("training_config_sha256") == config.config_sha256
        and run.get("training_config", {}).get("sha256") == config.config_sha256
        and _sha256_file(report / "training_config.yaml") == config.config_sha256,
        "training_config_hash_binding",
        checks,
    )

    bundle = joblib.load(model_file)
    _check(isinstance(bundle, SleepExpertBundle), "model_bundle_type", checks)
    _validate_bundle_identity(bundle, checks)
    all_keys = set(assignments)
    production_available_keys, production_available_rows = _available_scope(
        inputs,
        all_keys,
    )
    _check(
        bundle.training_scope == "all_sleep_sources_for_production"
        and bundle.training_participant_count == len(production_available_keys)
        and bundle.training_participant_sha256
        == participant_set_sha256(production_available_keys),
        "production_training_participant_scope",
        checks,
    )
    expected_rows = sum(len(frame) for frame in inputs.frames.values())
    _check(
        bundle.training_row_count == production_available_rows,
        "production_training_row_scope",
        checks,
    )
    _validate_bundle_mask_behavior(bundle, checks)

    predictions = pd.read_parquet(report / "predictions.parquet")
    _check(
        tuple(predictions.columns) == PREDICTION_COLUMNS,
        "prediction_column_contract",
        checks,
    )
    _check(
        predictions["prediction_schema_version"].nunique() == 1
        and predictions["prediction_schema_version"].iloc[0]
        == "mood-social-sleep-oof-predictions-v1",
        "prediction_schema_version",
        checks,
    )
    _check(
        len(predictions) == expected_rows
        and not predictions["prediction_id"].duplicated().any()
        and set(predictions["dataset_id"].astype(str)) == set(SLEEP_DATASET_IDS),
        "prediction_coverage",
        checks,
    )
    _check(
        predictions["expert_mask"].isin([0, 1]).all(),
        "prediction_mask_contract",
        checks,
    )
    available = predictions["expert_mask"].eq(1)
    unavailable = ~available
    _check(
        np.isfinite(predictions.loc[available, "raw_probability"]).all()
        and np.isfinite(predictions.loc[available, "calibrated_probability"]).all()
        and predictions.loc[available, "raw_probability"].between(0, 1).all()
        and predictions.loc[available, "calibrated_probability"].between(0, 1).all()
        and predictions.loc[unavailable, "raw_probability"].isna().all()
        and predictions.loc[unavailable, "calibrated_probability"].isna().all(),
        "prediction_probability_contract",
        checks,
    )
    _validate_prediction_rows(predictions, inputs, assignments, checks)
    _check(
        _sha256_file(report / "predictions.parquet")
        == run.get("artifacts", {}).get("predictions_sha256")
        == model_manifest.get("predictions_sha256"),
        "prediction_hash_binding",
        checks,
    )
    _check(
        _sha256_file(report / "metrics.json")
        == run.get("artifacts", {}).get("metrics_sha256")
        == model_manifest.get("metrics_sha256"),
        "metrics_hash_binding",
        checks,
    )
    _validate_metrics(predictions, metrics, config, checks)
    _validate_outer_fold_bundles(
        report,
        inputs,
        assignments,
        config.run_id,
        checks,
    )
    _validate_report_artifacts(report, checks)
    return {
        "status": "pass",
        "check_count": len(checks),
        "run_id": config.run_id,
        "model_id": MODEL_ID,
        "model_sha256": model_sha256,
        "split_id": SPLIT_ID,
        "split_manifest_sha256": EXPECTED_SPLIT_SHA256,
        "dataset_count": len(SLEEP_DATASET_IDS),
        "participant_count": int(predictions["global_participant_id"].nunique()),
        "row_count": len(predictions),
        "positive_row_count": int(predictions["binary_target"].sum()),
        "calibration_method": bundle.calibrator.method,
        "selected_features": list(bundle.preprocessor.selected_features),
    }


def _validate_bundle_identity(
    bundle: SleepExpertBundle,
    checks: list[str],
) -> None:
    _check(
        bundle.model_id == MODEL_ID
        and bundle.model_version == MODEL_VERSION
        and bundle.bundle_version == MODEL_BUNDLE_VERSION,
        "bundle_identity",
        checks,
    )
    _check(
        bundle.feature_schema_version == FEATURE_SCHEMA_VERSION
        and bundle.feature_schema_sha256 == EXPECTED_FEATURE_SCHEMA_SHA256
        and bundle.split_id == SPLIT_ID
        and bundle.split_manifest_sha256 == EXPECTED_SPLIT_SHA256,
        "bundle_frozen_bindings",
        checks,
    )
    _check(
        bundle.dataset_ids == SLEEP_DATASET_IDS
        and bundle.input_feature_names == SLEEP_FEATURE_COLUMNS
        and bundle.audit_dict().get("source_transform")
        == "canonical_same_semantics_identity"
        and set(bundle.preprocessor.selected_features).issubset(SLEEP_FEATURE_COLUMNS)
        and not any(
            value in bundle.preprocessor.selected_features
            for value in (
                "dataset_id",
                "binary_target",
                "sleep.feature_coverage",
            )
        ),
        "bundle_feature_boundary",
        checks,
    )
    _check(
        bundle.calibrator.method == "isotonic"
        and bundle.calibrator.positive_participant_count >= 200,
        "bundle_calibration_rule",
        checks,
    )


def _validate_bundle_mask_behavior(
    bundle: SleepExpertBundle,
    checks: list[str],
) -> None:
    missing = pd.DataFrame(
        {
            **{name: [np.nan] for name in SLEEP_FEATURE_COLUMNS},
            **{name: [0] for name in SLEEP_MASK_COLUMNS},
        }
    )
    unavailable = bundle.predict(missing)
    _check(
        int(unavailable.loc[0, "expert_mask"]) == 0
        and pd.isna(unavailable.loc[0, "raw_probability"])
        and pd.isna(unavailable.loc[0, "calibrated_probability"])
        and float(unavailable.loc[0, "observed_feature_fraction"]) == 0.0,
        "bundle_all_missing_mask",
        checks,
    )
    values = {name: [np.nan] for name in SLEEP_FEATURE_COLUMNS}
    masks = {name: [0] for name in SLEEP_MASK_COLUMNS}
    for feature, median in zip(
        bundle.preprocessor.selected_features,
        bundle.preprocessor.medians,
        strict=True,
    ):
        values[feature] = [median]
        masks[f"feature_mask.{feature}"] = [1]
    available = bundle.predict(pd.DataFrame({**values, **masks}))
    _check(
        int(available.loc[0, "expert_mask"]) == 1
        and np.isfinite(available.loc[0, "calibrated_probability"]),
        "bundle_observed_prediction",
        checks,
    )


def _validate_prediction_rows(
    predictions: pd.DataFrame,
    inputs: Any,
    assignments: Mapping[str, Mapping[str, Any]],
    checks: list[str],
) -> None:
    expected_parts: list[pd.DataFrame] = []
    for dataset_id in SLEEP_DATASET_IDS:
        frame = inputs.frames[dataset_id]
        part = pd.DataFrame(
            {
                "prediction_id": (
                    dataset_id
                    + "::row="
                    + pd.Series(frame.index, dtype="int64").astype(str)
                ),
                "dataset_id": dataset_id,
                "global_participant_id": frame["global_participant_id"].astype(str),
                "canonical_row_index": frame.index.astype("int64"),
                "binary_target": pd.to_numeric(frame["binary_target"]).astype("int64"),
            }
        )
        part["outer_fold"] = part["global_participant_id"].map(
            {key: int(value["outer_fold"]) for key, value in assignments.items()}
        )
        expected_parts.append(part)
    expected = (
        pd.concat(expected_parts, ignore_index=True)
        .sort_values(
            ["dataset_id", "canonical_row_index"],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    actual = (
        predictions[list(expected.columns)]
        .sort_values(
            ["dataset_id", "canonical_row_index"],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    for column in ("canonical_row_index", "binary_target", "outer_fold"):
        actual[column] = pd.to_numeric(actual[column], errors="raise").astype("int64")
        expected[column] = pd.to_numeric(expected[column], errors="raise").astype(
            "int64"
        )
    _check(actual.equals(expected), "prediction_row_alignment", checks)
    participant_folds = predictions.groupby("global_participant_id", sort=False)[
        "outer_fold"
    ].nunique()
    _check(
        participant_folds.eq(1).all(),
        "prediction_participant_fold_atomicity",
        checks,
    )


def _validate_metrics(
    predictions: pd.DataFrame,
    metrics: Mapping[str, Any],
    config: Any,
    checks: list[str],
) -> None:
    def pair(frame: pd.DataFrame) -> dict[str, Any]:
        available = frame[frame["expert_mask"].eq(1)]
        return {
            "raw": binary_metrics(
                available["binary_target"],
                available["raw_probability"],
                decision_threshold=config.decision_threshold,
                ece_bin_count=config.ece_bin_count,
            ),
            "calibrated": binary_metrics(
                available["binary_target"],
                available["calibrated_probability"],
                decision_threshold=config.decision_threshold,
                ece_bin_count=config.ece_bin_count,
            ),
        }

    _check(
        metrics.get("overall") == pair(predictions),
        "overall_metric_recalculation",
        checks,
    )
    _check(
        metrics.get("availability")
        == {
            "row_count": len(predictions),
            "available_row_count": int(predictions["expert_mask"].sum()),
            "unavailable_row_count": int(predictions["expert_mask"].eq(0).sum()),
        },
        "metric_availability_counts",
        checks,
    )
    for dataset_id in SLEEP_DATASET_IDS:
        selected = predictions[predictions["dataset_id"].astype(str).eq(dataset_id)]
        _check(
            metrics.get("by_dataset", {}).get(dataset_id) == pair(selected),
            f"{dataset_id}_metric_recalculation",
            checks,
        )
    for outer_fold in range(5):
        selected = predictions[predictions["outer_fold"].eq(outer_fold)]
        _check(
            metrics.get("by_outer_fold", {}).get(str(outer_fold)) == pair(selected),
            f"outer_{outer_fold}_metric_recalculation",
            checks,
        )


def _validate_outer_fold_bundles(
    report: Path,
    inputs: Any,
    assignments: Mapping[str, Mapping[str, Any]],
    run_id: str,
    checks: list[str],
) -> None:
    all_keys = set(assignments)
    for outer_fold in range(5):
        bundle_path = report / "folds" / f"outer_fold_{outer_fold}.joblib"
        audit_path = report / "folds" / f"outer_fold_{outer_fold}.json"
        _check(
            bundle_path.is_file() and audit_path.is_file(),
            f"outer_{outer_fold}_artifact_files",
            checks,
        )
        bundle = joblib.load(bundle_path)
        audit = _read_json(audit_path)
        _check(
            isinstance(bundle, SleepExpertBundle)
            and bundle.run_id == run_id
            and bundle.training_scope == f"outer_fold_{outer_fold}_train",
            f"outer_{outer_fold}_bundle_identity",
            checks,
        )
        test_keys = {
            key
            for key, value in assignments.items()
            if int(value["outer_fold"]) == outer_fold
        }
        train_keys = all_keys - test_keys
        available_train_keys, available_train_rows = _available_scope(
            inputs,
            train_keys,
        )
        _check(
            bundle.training_participant_count == len(available_train_keys)
            and bundle.training_participant_sha256
            == participant_set_sha256(available_train_keys)
            and bundle.training_row_count == available_train_rows
            and audit.get("test_participant_count") == len(test_keys)
            and audit.get("test_participant_sha256")
            == participant_set_sha256(test_keys),
            f"outer_{outer_fold}_participant_scope",
            checks,
        )
        _check(
            bundle.calibrator.method == "isotonic"
            and bundle.calibrator.fit_row_count == bundle.training_row_count,
            f"outer_{outer_fold}_calibration_scope",
            checks,
        )


def _available_scope(
    inputs: Any,
    participant_keys: set[str],
) -> tuple[set[str], int]:
    available_keys: set[str] = set()
    available_rows = 0
    for dataset_id in SLEEP_DATASET_IDS:
        frame = inputs.frames[dataset_id]
        selected = frame[
            frame["global_participant_id"].astype(str).isin(participant_keys)
        ]
        masks = selected[list(SLEEP_MASK_COLUMNS)].apply(
            pd.to_numeric,
            errors="raise",
        )
        available = masks.sum(axis=1).gt(0)
        available_rows += int(available.sum())
        available_keys.update(
            selected.loc[available, "global_participant_id"].astype(str)
        )
    if not available_keys or available_rows <= 0:
        raise ValidationError("SleepExpert available training scope is empty")
    return available_keys, available_rows


def _validate_report_artifacts(report: Path, checks: list[str]) -> None:
    manifest = _read_json(report / "artifacts.json")
    rows = manifest.get("artifacts")
    _check(
        manifest.get("artifact_manifest_version")
        == "mood-social-sleep-report-artifacts-v1"
        and isinstance(rows, list)
        and manifest.get("artifact_count") == len(rows),
        "report_artifact_manifest_contract",
        checks,
    )
    expected_paths = set()
    for row in rows:
        relative = str(row["path"])
        path = _safe_report_path(report, relative)
        _check(
            path.is_file()
            and path.stat().st_size == int(row["bytes"])
            and _sha256_file(path) == row["sha256"],
            f"report_artifact_{len(expected_paths):03d}",
            checks,
        )
        expected_paths.add(relative)
    actual_paths = {
        path.relative_to(report).as_posix()
        for path in report.rglob("*")
        if path.is_file() and path.name != "artifacts.json"
    }
    _check(actual_paths == expected_paths, "report_artifact_file_set", checks)


def _safe_report_path(report: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValidationError("report artifact path is unsafe")
    resolved = (report / candidate).resolve()
    if report.resolve() not in resolved.parents:
        raise ValidationError("report artifact path escapes report directory")
    return resolved


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValidationError("required MODEL-002 JSON is unreadable") from exc


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(chunk_size), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValidationError("required MODEL-002 file is unreadable") from exc
    return digest.hexdigest()


def _check(condition: bool, name: str, checks: list[str]) -> None:
    if not condition:
        raise ValidationError(f"validation failed: {name}")
    checks.append(name)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--model-manifest-path", type=Path)
    parser.add_argument("--config", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = (
        args.repository_root.resolve()
        if args.repository_root is not None
        else Path(__file__).resolve().parents[1]
    )
    try:
        result = validate(
            root,
            args.report_dir or root / DEFAULT_REPORT,
            args.model_path or root / DEFAULT_MODEL,
            args.model_manifest_path or root / DEFAULT_MODEL_MANIFEST,
            args.config or root / DEFAULT_CONFIG,
        )
    except (ValidationError, OSError, TypeError, ValueError, RuntimeError) as exc:
        print(f"MODEL-002 validation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
