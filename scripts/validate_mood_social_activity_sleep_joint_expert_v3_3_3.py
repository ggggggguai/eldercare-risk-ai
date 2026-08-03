"""Validate MODEL-003 artifacts, joint masks, and fold leakage boundaries."""

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

from elderly_monitoring.modules.mental_health.mood_social.experts import joint
from elderly_monitoring.modules.mental_health.mood_social.experts.activity import (
    ACTIVITY_FEATURE_COLUMNS,
    ACTIVITY_MASK_COLUMNS,
    binary_metrics,
    participant_set_sha256,
)
from elderly_monitoring.modules.mental_health.mood_social.experts.joint import (
    EXPECTED_FEATURE_SCHEMA_SHA256,
    EXPECTED_SPLIT_SHA256,
    JOINT_DATASET_IDS,
    JOINT_FEATURE_COLUMNS,
    JOINT_MASK_COLUMNS,
    MODEL_BUNDLE_VERSION,
    MODEL_ID,
    MODEL_VERSION,
    UPSTREAM_EXPERT_GUARDS,
    ActivitySleepJointExpertBundle,
    load_joint_training_config,
    load_joint_training_inputs,
)
from elderly_monitoring.modules.mental_health.mood_social.experts.sleep import (
    SLEEP_FEATURE_COLUMNS,
    SLEEP_MASK_COLUMNS,
)
from elderly_monitoring.modules.mental_health.mood_social.feature_schema import (
    FEATURE_SCHEMA_VERSION,
)
from elderly_monitoring.modules.mental_health.mood_social.splits import SPLIT_ID


DEFAULT_CONFIG = Path(
    "configs/training/mood_social_activity_sleep_joint_expert_v3_3_3.yaml"
)
DEFAULT_REPORT = Path("reports/mental_health/mood_social/MH-20260731-003")
DEFAULT_MODEL = Path(
    "models/mental_health/mood_social/v3.3.3/activity_sleep_joint_expert.joblib"
)
DEFAULT_MODEL_MANIFEST = Path(
    "models/mental_health/mood_social/v3.3.3/activity_sleep_joint_expert_manifest.json"
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
    "activity_observed_feature_count",
    "activity_observed_feature_fraction",
    "sleep_observed_feature_count",
    "sleep_observed_feature_fraction",
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
    config = load_joint_training_config(config_path)
    inputs = load_joint_training_inputs(root)
    assignments = {
        str(row["global_participant_id"]): row
        for row in inputs.assignments.to_dict(orient="records")
    }
    _check(report.is_dir(), "report_directory", checks)
    _check(model_file.is_file(), "production_model_file", checks)
    _check(model_manifest_file.is_file(), "model_manifest_file", checks)

    run = _read_json(report / "run.json")
    metrics = _read_json(report / "metrics.json")
    search = _read_json(report / "search_results.json")
    model_manifest = _read_json(model_manifest_file)
    model_artifact = _read_json(report / "model_artifact.json")
    warnings = _read_json(report / "failures" / "warnings.json")
    _check(
        run.get("status") == "completed"
        and run.get("task_id") == "MODEL-003"
        and run.get("plan_id") == "PLAN-JNT-001"
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
        run.get("upstream_expert_guards") == _plain(UPSTREAM_EXPERT_GUARDS)
        and set(run.get("upstream_expert_bindings", {})) == {"activity", "sleep"}
        and run.get("uses_standalone_expert_probabilities_as_inputs") is False,
        "run_upstream_expert_boundary",
        checks,
    )
    _check(
        run.get("failures") == []
        and isinstance(warnings.get("warnings"), list)
        and warnings.get("failures") == []
        and {item.get("code") for item in warnings.get("warnings", [])}
        >= {
            "PROXY_TRANSFER",
            "SMALL_RESILIENT_SOURCE",
            "JOINT_ROW_ELIGIBILITY",
            "NO_PERFORMANCE_GATE",
        },
        "failure_and_warning_contract",
        checks,
    )
    _check(
        model_manifest.get("status") == "active"
        and model_manifest.get("model_id") == MODEL_ID
        and model_manifest.get("model_version") == MODEL_VERSION
        and model_manifest.get("run_id") == config.run_id
        and model_manifest.get("dataset_ids") == list(JOINT_DATASET_IDS)
        and model_manifest.get("excluded_dataset_ids")
        == ["shenzhen_elderly", "nhanes_ssq_2005_2008"]
        and model_manifest.get("uses_standalone_expert_probabilities_as_inputs")
        is False,
        "model_manifest_identity",
        checks,
    )
    _check(
        model_manifest.get("split_id") == SPLIT_ID
        and model_manifest.get("split_manifest_sha256") == EXPECTED_SPLIT_SHA256
        and model_manifest.get("feature_schema_version") == FEATURE_SCHEMA_VERSION
        and model_manifest.get("feature_schema_sha256")
        == EXPECTED_FEATURE_SCHEMA_SHA256
        and model_manifest.get("upstream_expert_guards")
        == _plain(UPSTREAM_EXPERT_GUARDS),
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
    _validate_code_bindings(root, run, model_manifest, checks)
    _validate_search(search, config, checks)

    bundle = joblib.load(model_file)
    _check(
        isinstance(bundle, ActivitySleepJointExpertBundle),
        "model_bundle_type",
        checks,
    )
    _validate_bundle_identity(bundle, checks)
    all_keys = set(assignments)
    production_available_keys, production_available_rows = _available_scope(
        inputs,
        bundle,
        all_keys,
    )
    _check(
        bundle.training_scope == "all_activity_sleep_joint_sources_for_production"
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
    for dataset_id in JOINT_DATASET_IDS:
        expected_dataset_keys = set(
            inputs.frames[dataset_id]["global_participant_id"].astype(str)
        )
        ecdf = bundle.ecdf_by_dataset[dataset_id]
        _check(
            ecdf.training_participant_count == len(expected_dataset_keys)
            and ecdf.training_participant_sha256
            == participant_set_sha256(expected_dataset_keys)
            and ecdf.split_id == SPLIT_ID,
            f"production_{dataset_id}_ecdf_scope",
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
        == "mood-social-activity-sleep-joint-oof-predictions-v1",
        "prediction_schema_version",
        checks,
    )
    _check(
        len(predictions) == expected_rows
        and not predictions["prediction_id"].duplicated().any()
        and set(predictions["dataset_id"].astype(str)) == set(JOINT_DATASET_IDS),
        "prediction_coverage",
        checks,
    )
    _validate_prediction_masks(predictions, checks)
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
        predictions,
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
        "dataset_count": len(JOINT_DATASET_IDS),
        "participant_count": int(predictions["global_participant_id"].nunique()),
        "row_count": len(predictions),
        "available_row_count": int(predictions["expert_mask"].sum()),
        "positive_row_count": int(predictions["binary_target"].sum()),
        "available_positive_row_count": int(
            predictions.loc[predictions["expert_mask"].eq(1), "binary_target"].sum()
        ),
        "calibration_method": bundle.calibrator.method,
        "selected_features": list(bundle.preprocessor.selected_features),
    }


def _validate_bundle_identity(
    bundle: ActivitySleepJointExpertBundle,
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
    selected = set(bundle.preprocessor.selected_features)
    _check(
        bundle.dataset_ids == JOINT_DATASET_IDS
        and bundle.input_feature_names == JOINT_FEATURE_COLUMNS
        and selected.issubset(JOINT_FEATURE_COLUMNS)
        and bool(selected & set(ACTIVITY_FEATURE_COLUMNS))
        and bool(selected & set(SLEEP_FEATURE_COLUMNS))
        and not selected & set(JOINT_MASK_COLUMNS)
        and not any("feature_coverage" in value for value in selected)
        and not any("probability" in value for value in selected),
        "bundle_feature_boundary",
        checks,
    )
    _check(
        _plain(bundle.upstream_expert_guards) == _plain(UPSTREAM_EXPERT_GUARDS),
        "bundle_upstream_expert_guards",
        checks,
    )
    _check(
        bundle.calibrator.method == "isotonic"
        and bundle.calibrator.positive_participant_count >= 200,
        "bundle_calibration_rule",
        checks,
    )


def _validate_bundle_mask_behavior(
    bundle: ActivitySleepJointExpertBundle,
    checks: list[str],
) -> None:
    frame = pd.DataFrame(
        {
            **{name: [np.nan] * 4 for name in JOINT_FEATURE_COLUMNS},
            **{name: [0] * 4 for name in JOINT_MASK_COLUMNS},
        }
    )
    activity = ACTIVITY_FEATURE_COLUMNS[0]
    sleep = SLEEP_FEATURE_COLUMNS[0]
    frame.loc[[1, 3], activity] = 0.5
    frame.loc[[1, 3], f"feature_mask.{activity}"] = 1
    frame.loc[[2, 3], sleep] = 7.0
    frame.loc[[2, 3], f"feature_mask.{sleep}"] = 1
    predicted = bundle.predict(frame)
    _check(
        predicted["expert_mask"].tolist() == [0, 0, 0, 1]
        and predicted.loc[:2, "raw_probability"].isna().all()
        and predicted.loc[:2, "calibrated_probability"].isna().all()
        and np.isfinite(predicted.loc[3, "raw_probability"])
        and np.isfinite(predicted.loc[3, "calibrated_probability"]),
        "bundle_two_sided_mask",
        checks,
    )


def _validate_prediction_masks(
    predictions: pd.DataFrame,
    checks: list[str],
) -> None:
    activity_available = predictions["activity_observed_feature_count"].gt(0)
    sleep_available = predictions["sleep_observed_feature_count"].gt(0)
    expected_mask = activity_available & sleep_available
    available = predictions["expert_mask"].eq(1)
    unavailable = ~available
    _check(
        predictions["expert_mask"].isin([0, 1]).all()
        and expected_mask.equals(available),
        "prediction_two_sided_mask_contract",
        checks,
    )
    _check(
        predictions["observed_feature_count"]
        .eq(
            predictions["activity_observed_feature_count"]
            + predictions["sleep_observed_feature_count"]
        )
        .all()
        and np.allclose(
            predictions["observed_feature_fraction"],
            predictions["observed_feature_count"] / len(JOINT_FEATURE_COLUMNS),
        )
        and np.allclose(
            predictions["activity_observed_feature_fraction"],
            predictions["activity_observed_feature_count"]
            / len(ACTIVITY_FEATURE_COLUMNS),
        )
        and np.allclose(
            predictions["sleep_observed_feature_fraction"],
            predictions["sleep_observed_feature_count"] / len(SLEEP_FEATURE_COLUMNS),
        ),
        "prediction_observed_feature_counts",
        checks,
    )
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


def _validate_prediction_rows(
    predictions: pd.DataFrame,
    inputs: Any,
    assignments: Mapping[str, Mapping[str, Any]],
    checks: list[str],
) -> None:
    expected_parts: list[pd.DataFrame] = []
    for dataset_id in JOINT_DATASET_IDS:
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
        .sort_values(["dataset_id", "canonical_row_index"], kind="stable")
        .reset_index(drop=True)
    )
    actual = (
        predictions[list(expected.columns)]
        .sort_values(["dataset_id", "canonical_row_index"], kind="stable")
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

    activity_available = predictions["activity_observed_feature_count"].gt(0)
    sleep_available = predictions["sleep_observed_feature_count"].gt(0)
    expected_availability = {
        "row_count": len(predictions),
        "available_row_count": int(predictions["expert_mask"].sum()),
        "unavailable_row_count": int(predictions["expert_mask"].eq(0).sum()),
        "activity_only_row_count": int((activity_available & ~sleep_available).sum()),
        "sleep_only_row_count": int((~activity_available & sleep_available).sum()),
        "neither_side_row_count": int((~activity_available & ~sleep_available).sum()),
    }
    _check(
        metrics.get("overall") == pair(predictions),
        "overall_metric_recalculation",
        checks,
    )
    _check(
        metrics.get("availability") == expected_availability,
        "metric_availability_counts",
        checks,
    )
    for dataset_id in JOINT_DATASET_IDS:
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
    predictions: pd.DataFrame,
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
            isinstance(bundle, ActivitySleepJointExpertBundle)
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
            bundle,
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
        for dataset_id in JOINT_DATASET_IDS:
            dataset_keys = set(
                inputs.frames[dataset_id]["global_participant_id"].astype(str)
            )
            expected_train = dataset_keys & train_keys
            ecdf = bundle.ecdf_by_dataset[dataset_id]
            _check(
                ecdf.training_participant_count == len(expected_train)
                and ecdf.training_participant_sha256
                == participant_set_sha256(expected_train),
                f"outer_{outer_fold}_{dataset_id}_ecdf_scope",
                checks,
            )
        _check(
            bundle.calibrator.method == "isotonic"
            and bundle.calibrator.fit_row_count == bundle.training_row_count,
            f"outer_{outer_fold}_calibration_scope",
            checks,
        )
        test_table = joint._transform_partition(
            inputs,
            bundle.ecdf_by_dataset,
            test_keys,
        )
        test_table = test_table.sort_values(
            ["dataset_id", "canonical_row_index"], kind="stable"
        ).reset_index(drop=True)
        expected_prediction = bundle.predict(test_table).reset_index(drop=True)
        actual_prediction = (
            predictions[predictions["outer_fold"].eq(outer_fold)]
            .sort_values(["dataset_id", "canonical_row_index"], kind="stable")
            .reset_index(drop=True)
        )
        _check(
            actual_prediction["prediction_id"].tolist()
            == test_table["prediction_id"].tolist()
            and actual_prediction["expert_mask"]
            .astype("int8")
            .equals(expected_prediction["expert_mask"].astype("int8"))
            and np.allclose(
                actual_prediction["raw_probability"],
                expected_prediction["raw_probability"],
                equal_nan=True,
            )
            and np.allclose(
                actual_prediction["calibrated_probability"],
                expected_prediction["calibrated_probability"],
                equal_nan=True,
            ),
            f"outer_{outer_fold}_prediction_replay",
            checks,
        )


def _available_scope(
    inputs: Any,
    bundle: ActivitySleepJointExpertBundle,
    participant_keys: set[str],
) -> tuple[set[str], int]:
    available_keys: set[str] = set()
    available_rows = 0
    for dataset_id in JOINT_DATASET_IDS:
        frame = inputs.frames[dataset_id]
        selected = frame[
            frame["global_participant_id"].astype(str).isin(participant_keys)
        ]
        transformed = bundle.ecdf_by_dataset[dataset_id].transform(selected)
        activity_masks = transformed[list(ACTIVITY_MASK_COLUMNS)].apply(
            pd.to_numeric,
            errors="raise",
        )
        sleep_masks = transformed[list(SLEEP_MASK_COLUMNS)].apply(
            pd.to_numeric,
            errors="raise",
        )
        available = activity_masks.sum(axis=1).gt(0) & sleep_masks.sum(axis=1).gt(0)
        available_rows += int(available.sum())
        available_keys.update(
            transformed.loc[available, "global_participant_id"].astype(str)
        )
    if not available_keys or available_rows <= 0:
        raise ValidationError(
            "ActivitySleepJointExpert available training scope is empty"
        )
    return available_keys, available_rows


def _validate_code_bindings(
    root: Path,
    run: Mapping[str, Any],
    model_manifest: Mapping[str, Any],
    checks: list[str],
) -> None:
    bindings = run.get("code_bindings")
    _check(
        isinstance(bindings, dict)
        and bindings == model_manifest.get("code_bindings")
        and (
            "src/elderly_monitoring/modules/mental_health/mood_social/"
            "experts/activity.py"
        )
        in bindings
        and (
            "src/elderly_monitoring/modules/mental_health/mood_social/experts/joint.py"
        )
        in bindings,
        "code_binding_contract",
        checks,
    )
    for index, (relative, expected_sha256) in enumerate(bindings.items()):
        path = _safe_root_path(root, str(relative))
        _check(
            path.is_file() and _sha256_file(path) == expected_sha256,
            f"code_binding_{index:02d}",
            checks,
        )


def _validate_search(
    search: Mapping[str, Any],
    config: Any,
    checks: list[str],
) -> None:
    outer = search.get("outer_folds")
    production = search.get("production_selection")
    _check(
        search.get("traversal") == "deterministic_sha256_subset"
        and search.get("candidate_count") == 32
        and len(search.get("candidate_ids", [])) == 32
        and isinstance(outer, list)
        and len(outer) == 5
        and isinstance(production, dict)
        and len(production.get("all_candidates", [])) == 32,
        "search_contract",
        checks,
    )
    _check(
        "outer_test_metrics" not in production
        and production.get("selection_rule")
        == (
            "highest_mean_nested_inner_auprc_then_lowest_mean_nested_inner_"
            "brier_then_candidate_id_without_outer_test_metrics"
        )
        and config.auprc_tie_tolerance == 1.0e-12,
        "production_selection_boundary",
        checks,
    )
    for outer_fold, result in enumerate(outer):
        _check(
            result.get("outer_fold") == outer_fold
            and len(result.get("candidates", [])) == 32
            and result.get("best_candidate", {}).get("candidate_id")
            in search.get("candidate_ids", []),
            f"outer_{outer_fold}_search_coverage",
            checks,
        )


def _validate_report_artifacts(report: Path, checks: list[str]) -> None:
    manifest = _read_json(report / "artifacts.json")
    rows = manifest.get("artifacts")
    _check(
        manifest.get("artifact_manifest_version")
        == "mood-social-joint-report-artifacts-v1"
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


def _safe_root_path(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValidationError("code binding path is unsafe")
    resolved = (root / candidate).resolve()
    if root.resolve() not in resolved.parents:
        raise ValidationError("code binding path escapes repository root")
    return resolved


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValidationError("required MODEL-003 JSON is unreadable") from exc


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(chunk_size), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValidationError("required MODEL-003 file is unreadable") from exc
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
        print(f"MODEL-003 validation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
