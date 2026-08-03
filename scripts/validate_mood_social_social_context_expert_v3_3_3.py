"""Validate MODEL-005 artifacts, OOF coverage, and leakage boundaries."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from catboost import CatBoostClassifier
import joblib
import numpy as np
import pandas as pd

from elderly_monitoring.modules.mental_health.mood_social.experts import social_context
from elderly_monitoring.modules.mental_health.mood_social.experts.social_context import (
    EXCLUDED_DATASET_IDS,
    EXPECTED_FEATURE_SCHEMA_SHA256,
    EXPECTED_SPLIT_SHA256,
    MODEL_BUNDLE_VERSION,
    MODEL_ID,
    MODEL_VERSION,
    PREDICTIONS_VERSION,
    SOCIAL_CONTEXT_CATEGORICAL_COLUMNS,
    SOCIAL_CONTEXT_DATASET_IDS,
    SOCIAL_CONTEXT_FEATURE_COLUMNS,
    SOCIAL_CONTEXT_MASK_COLUMNS,
    SOCIAL_CONTEXT_NUMERIC_COLUMNS,
    UPSTREAM_EXPERT_GUARDS,
    SocialContextExpertBundle,
    SocialContextFoldPreprocessor,
    binary_metrics,
    compute_training_sample_weights,
    deterministic_search_candidates,
    load_social_context_training_config,
    load_social_context_training_inputs,
    participant_set_sha256,
    sample_weight_audit,
)
from elderly_monitoring.modules.mental_health.mood_social.feature_schema import (
    FEATURE_SCHEMA_VERSION,
)
from elderly_monitoring.modules.mental_health.mood_social.splits import SPLIT_ID


DEFAULT_CONFIG = Path("configs/training/mood_social_social_context_expert_v3_3_3.yaml")
DEFAULT_REPORT = Path("reports/mental_health/mood_social/MH-20260801-005")
DEFAULT_MODEL = Path(
    "models/mental_health/mood_social/v3.3.3/social_context_expert.joblib"
)
DEFAULT_MODEL_MANIFEST = Path(
    "models/mental_health/mood_social/v3.3.3/social_context_expert_manifest.json"
)
EXPECTED_SELECTED_FEATURES = (
    "social_context.age_group",
    "social_context.sex",
    "social_context.marital_status",
    "social_context.self_rated_health",
    "social_context.education_level",
    "social_context.economic_status",
)
EXPECTED_CATEGORICAL_FEATURES = tuple(
    feature
    for feature in EXPECTED_SELECTED_FEATURES
    if feature in SOCIAL_CONTEXT_CATEGORICAL_COLUMNS
)
EXPECTED_NUMERIC_FEATURES = tuple(
    feature
    for feature in EXPECTED_SELECTED_FEATURES
    if feature in SOCIAL_CONTEXT_NUMERIC_COLUMNS
)
EXPECTED_FORBIDDEN_INPUTS = (
    "questionnaire_items",
    "publisher_labels",
    "ssq_specific_fields",
    "activity",
    "sleep",
    "physiology",
    "social_contact",
    "s10_call_fields",
)
EXPECTED_WARNING_CODES = {
    "CROSS_SOURCE_PROFILE_PROXY",
    "SSQ_FIELDS_EXCLUDED",
    "SMALL_RESILIENT_SOURCE",
    "STRUCTURAL_FEATURE_ABSENCE",
    "NO_PERFORMANCE_GATE",
}
EXPECTED_CODE_BINDINGS = {
    "src/elderly_monitoring/modules/mental_health/mood_social/experts/social_context.py",
    "scripts/train_mood_social_social_context_expert_v3_3_3.py",
    "scripts/validate_mood_social_social_context_expert_v3_3_3.py",
    "configs/training/mood_social_social_context_expert_v3_3_3.yaml",
}
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
    config = load_social_context_training_config(config_path)
    inputs = load_social_context_training_inputs(root)
    assignments = social_context._assignment_lookup(inputs.assignments)

    _check(report.is_dir(), "report_directory", checks)
    _check(model_file.is_file(), "production_model_file", checks)
    _check(model_manifest_file.is_file(), "model_manifest_file", checks)

    run = _read_json(report / "run.json")
    metrics = _read_json(report / "metrics.json")
    search = _read_json(report / "search_results.json")
    model_manifest = _read_json(model_manifest_file)
    model_artifact = _read_json(report / "model_artifact.json")
    warnings = _read_json(report / "failures" / "warnings.json")

    _validate_run(run, config, inputs, warnings, checks)
    _validate_model_manifest(model_manifest, config, inputs, checks)
    _validate_code_bindings(root, run, model_manifest, checks)

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
    _check(isinstance(bundle, SocialContextExpertBundle), "model_bundle_type", checks)
    _validate_bundle_identity(bundle, config, checks)
    _validate_production_scope(bundle, inputs, assignments, checks)
    _validate_bundle_mask_behavior(bundle, checks)

    predictions = pd.read_parquet(report / "predictions.parquet")
    _validate_predictions(predictions, inputs, assignments, checks)
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
    _validate_search(search, config, checks)
    _validate_outer_fold_bundles(
        report,
        inputs,
        assignments,
        predictions,
        config,
        checks,
    )
    _validate_final_audits(report, bundle, checks)
    _validate_feature_importance(report, bundle, checks)
    _validate_report_artifacts(report, checks)
    _validate_upstream_hashes(root, inputs.upstream_expert_bindings, checks)

    return {
        "status": "pass",
        "check_count": len(checks),
        "run_id": config.run_id,
        "model_id": MODEL_ID,
        "model_sha256": model_sha256,
        "split_id": SPLIT_ID,
        "split_manifest_sha256": EXPECTED_SPLIT_SHA256,
        "dataset_count": len(SOCIAL_CONTEXT_DATASET_IDS),
        "participant_count": int(predictions["global_participant_id"].nunique()),
        "row_count": len(predictions),
        "available_row_count": int(predictions["expert_mask"].sum()),
        "positive_row_count": int(predictions["binary_target"].sum()),
        "calibration_method": bundle.calibrator.method,
        "selected_features": list(bundle.preprocessor.selected_features),
        "final_hyperparameters": dict(bundle.hyperparameters),
    }


def _validate_run(
    run: Mapping[str, Any],
    config: Any,
    inputs: Any,
    warnings: Mapping[str, Any],
    checks: list[str],
) -> None:
    _check(
        run.get("status") == "completed"
        and run.get("task_id") == "MODEL-005"
        and run.get("model_id") == MODEL_ID
        and run.get("model_version") == MODEL_VERSION
        and run.get("run_id") == config.run_id
        and run.get("algorithm") == "CatBoost",
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
    expected_bindings = [
        {
            "dataset_id": dataset_id,
            "canonical_relative_path": binding["canonical_relative_path"],
            "canonical_sha256": binding["canonical_sha256"],
            "canonical_frame_sha256": binding["canonical_frame_sha256"],
            "artifact_manifest_sha256": binding["artifact_manifest_sha256"],
            "mapping_sha256": binding["mapping_sha256"],
            "row_count": binding["row_count"],
            "participant_count": binding["participant_count"],
            "positive_row_count": binding["positive_row_count"],
        }
        for dataset_id, binding in inputs.input_bindings.items()
    ]
    _check(
        run.get("input_bindings") == expected_bindings
        and run.get("excluded_sources") == list(EXCLUDED_DATASET_IDS),
        "run_source_bindings",
        checks,
    )
    _check(
        run.get("upstream_expert_guards")
        == {
            name: dict(binding)
            for name, binding in inputs.upstream_expert_bindings.items()
        },
        "run_upstream_expert_guards",
        checks,
    )
    warning_codes = {item.get("code") for item in warnings.get("warnings", [])}
    _check(
        run.get("failures") == []
        and warnings.get("failures") == []
        and run.get("warnings") == warnings.get("warnings")
        and warning_codes == EXPECTED_WARNING_CODES,
        "failure_and_warning_contract",
        checks,
    )
    _check(
        run.get("forbidden_input_audit", {}).get("status") == "pass"
        and run.get("mask_only_diagnostic", {}).get("status") == "pass"
        and run.get("all_mask_fusion_hard_gate") == "not_applicable_to_MODEL_005",
        "run_boundary_diagnostics",
        checks,
    )


def _validate_model_manifest(
    manifest: Mapping[str, Any],
    config: Any,
    inputs: Any,
    checks: list[str],
) -> None:
    _check(
        manifest.get("status") == "active"
        and manifest.get("model_id") == MODEL_ID
        and manifest.get("model_version") == MODEL_VERSION
        and manifest.get("run_id") == config.run_id,
        "model_manifest_identity",
        checks,
    )
    _check(
        manifest.get("split_id") == SPLIT_ID
        and manifest.get("split_manifest_sha256") == EXPECTED_SPLIT_SHA256
        and manifest.get("feature_schema_version") == FEATURE_SCHEMA_VERSION
        and manifest.get("feature_schema_sha256") == EXPECTED_FEATURE_SCHEMA_SHA256,
        "model_manifest_frozen_bindings",
        checks,
    )
    _check(
        manifest.get("dataset_ids") == list(SOCIAL_CONTEXT_DATASET_IDS)
        and manifest.get("excluded_dataset_ids") == list(EXCLUDED_DATASET_IDS)
        and manifest.get("source_transform") == "canonical_same_semantics_identity",
        "model_manifest_source_policy",
        checks,
    )
    _check(
        manifest.get("input_feature_names") == list(SOCIAL_CONTEXT_FEATURE_COLUMNS)
        and manifest.get("selected_feature_names") == list(EXPECTED_SELECTED_FEATURES)
        and manifest.get("categorical_feature_names")
        == list(EXPECTED_CATEGORICAL_FEATURES)
        and manifest.get("numeric_feature_names") == list(EXPECTED_NUMERIC_FEATURES),
        "model_manifest_feature_contract",
        checks,
    )
    calibrator = manifest.get("calibrator", {})
    _check(
        calibrator.get("method") == "isotonic"
        and calibrator.get("positive_participant_count") == 635
        and calibrator.get("fit_row_count") == 11_325
        and manifest.get("expert_mask_rule")
        == "at_least_one_selected_social_context_input"
        and manifest.get("forbidden_inputs") == list(EXPECTED_FORBIDDEN_INPUTS),
        "model_manifest_model_contract",
        checks,
    )
    _check(
        manifest.get("upstream_expert_guards")
        == {
            name: dict(binding)
            for name, binding in inputs.upstream_expert_bindings.items()
        },
        "model_manifest_upstream_expert_guards",
        checks,
    )


def _validate_bundle_identity(
    bundle: SocialContextExpertBundle,
    config: Any,
    checks: list[str],
) -> None:
    _check(
        bundle.model_id == MODEL_ID
        and bundle.model_version == MODEL_VERSION
        and bundle.bundle_version == MODEL_BUNDLE_VERSION
        and bundle.feature_schema_version == FEATURE_SCHEMA_VERSION
        and bundle.feature_schema_sha256 == EXPECTED_FEATURE_SCHEMA_SHA256
        and bundle.split_id == SPLIT_ID
        and bundle.split_manifest_sha256 == EXPECTED_SPLIT_SHA256,
        "bundle_frozen_identity",
        checks,
    )
    _check(
        bundle.dataset_ids == SOCIAL_CONTEXT_DATASET_IDS
        and bundle.input_feature_names == SOCIAL_CONTEXT_FEATURE_COLUMNS
        and bundle.preprocessor.selected_features == EXPECTED_SELECTED_FEATURES
        and bundle.preprocessor.categorical_features == EXPECTED_CATEGORICAL_FEATURES
        and bundle.preprocessor.numeric_features == EXPECTED_NUMERIC_FEATURES
        and not bundle.preprocessor.add_missing_indicators,
        "bundle_feature_contract",
        checks,
    )
    params = bundle.classifier.get_params()
    metadata = dict(bundle.classifier.get_metadata())
    expected_metadata = social_context._deterministic_catboost_metadata(
        config,
        bundle.hyperparameters,
        training_identity="production_aggregate",
    )
    _check(
        isinstance(bundle.classifier, CatBoostClassifier)
        and params.get("random_seed") == 20260728
        and params.get("thread_count") == 1
        and params.get("allow_writing_files") is False
        and params.get("bootstrap_type") == "No"
        and float(params.get("random_strength")) == 0.0
        and set(bundle.hyperparameters)
        == {"iterations", "depth", "learning_rate", "l2_leaf_reg"}
        and bundle.calibrator.method == "isotonic"
        and bundle.calibrator.positive_participant_count
        >= config.calibration_positive_participant_threshold,
        "bundle_algorithm_contract",
        checks,
    )
    _check(
        all(metadata.get(key) == value for key, value in expected_metadata.items()),
        "bundle_deterministic_catboost_metadata",
        checks,
    )


def _validate_production_scope(
    bundle: SocialContextExpertBundle,
    inputs: Any,
    assignments: Mapping[str, Mapping[str, Any]],
    checks: list[str],
) -> None:
    all_keys = set(assignments)
    table = social_context._available_social_rows(
        social_context._transform_partition(inputs, all_keys)
    )
    fitted = SocialContextFoldPreprocessor.fit(table)
    _check(
        bundle.training_scope == "all_social_context_sources_for_production"
        and bundle.training_participant_count == len(all_keys) == 11_325
        and bundle.training_participant_sha256 == participant_set_sha256(all_keys)
        and bundle.training_row_count == len(table) == 11_325,
        "production_training_scope",
        checks,
    )
    _check(
        bundle.preprocessor.to_dict() == fitted.to_dict(),
        "production_preprocessing_scope",
        checks,
    )
    _check(
        bundle.calibrator.fit_row_count == len(table)
        and bundle.calibrator.positive_participant_count
        == int(
            table.loc[table["binary_target"].eq(1), "global_participant_id"].nunique()
        ),
        "production_calibration_scope",
        checks,
    )


def _validate_bundle_mask_behavior(
    bundle: SocialContextExpertBundle,
    checks: list[str],
) -> None:
    values = {
        name: (
            pd.Series([pd.NA], dtype="string")
            if name in SOCIAL_CONTEXT_CATEGORICAL_COLUMNS
            else pd.Series([np.nan], dtype="float64")
        )
        for name in SOCIAL_CONTEXT_FEATURE_COLUMNS
    }
    frame = pd.DataFrame(
        {
            **values,
            **{name: [0] for name in SOCIAL_CONTEXT_MASK_COLUMNS},
        }
    )
    missing = bundle.predict(frame)
    _check(
        int(missing.loc[0, "expert_mask"]) == 0
        and pd.isna(missing.loc[0, "raw_probability"])
        and pd.isna(missing.loc[0, "calibrated_probability"])
        and int(missing.loc[0, "observed_feature_count"]) == 0,
        "bundle_all_missing_mask",
        checks,
    )
    feature = bundle.preprocessor.categorical_features[0]
    frame.loc[0, feature] = bundle.preprocessor.category_levels[feature][0]
    frame.loc[0, f"feature_mask.{feature}"] = 1
    observed = bundle.predict(frame)
    _check(
        int(observed.loc[0, "expert_mask"]) == 1
        and int(observed.loc[0, "observed_feature_count"]) == 1
        and np.isfinite(observed.loc[0, "raw_probability"])
        and np.isfinite(observed.loc[0, "calibrated_probability"]),
        "bundle_one_observed_mask",
        checks,
    )


def _validate_predictions(
    predictions: pd.DataFrame,
    inputs: Any,
    assignments: Mapping[str, Mapping[str, Any]],
    checks: list[str],
) -> None:
    _check(
        tuple(predictions.columns) == PREDICTION_COLUMNS,
        "prediction_column_contract",
        checks,
    )
    _check(
        predictions["prediction_schema_version"].nunique() == 1
        and predictions["prediction_schema_version"].iloc[0] == PREDICTIONS_VERSION,
        "prediction_schema_version",
        checks,
    )
    _check(
        len(predictions) == 11_325
        and predictions["global_participant_id"].nunique() == 11_325
        and int(predictions["binary_target"].sum()) == 635
        and not predictions["prediction_id"].duplicated().any()
        and set(predictions["dataset_id"].astype(str))
        == set(SOCIAL_CONTEXT_DATASET_IDS),
        "prediction_coverage",
        checks,
    )
    expected_parts: list[pd.DataFrame] = []
    selected_masks = [f"feature_mask.{name}" for name in EXPECTED_SELECTED_FEATURES]
    for dataset_id in SOCIAL_CONTEXT_DATASET_IDS:
        frame = inputs.frames[dataset_id]
        observed = (
            frame[selected_masks].apply(pd.to_numeric, errors="raise").sum(axis=1)
        )
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
                "expert_mask": observed.gt(0).astype("int8"),
                "observed_feature_count": observed.astype("int64"),
                "observed_feature_fraction": observed
                / len(SOCIAL_CONTEXT_FEATURE_COLUMNS),
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
    for column in (
        "canonical_row_index",
        "binary_target",
        "outer_fold",
        "expert_mask",
        "observed_feature_count",
    ):
        expected[column] = pd.to_numeric(expected[column]).astype("int64")
        actual[column] = pd.to_numeric(actual[column]).astype("int64")
    identity_columns = [
        column for column in expected.columns if column != "observed_feature_fraction"
    ]
    _check(
        actual[identity_columns].equals(expected[identity_columns])
        and np.allclose(
            actual["observed_feature_fraction"],
            expected["observed_feature_fraction"],
        ),
        "prediction_row_and_mask_alignment",
        checks,
    )
    available = predictions["expert_mask"].eq(1)
    unavailable = ~available
    _check(
        int(available.sum()) == 11_325
        and np.isfinite(predictions.loc[available, "raw_probability"]).all()
        and np.isfinite(predictions.loc[available, "calibrated_probability"]).all()
        and predictions.loc[available, "raw_probability"].between(0, 1).all()
        and predictions.loc[available, "calibrated_probability"].between(0, 1).all()
        and predictions.loc[unavailable, "raw_probability"].isna().all()
        and predictions.loc[unavailable, "calibrated_probability"].isna().all(),
        "prediction_probability_mask_contract",
        checks,
    )
    _check(
        predictions.groupby("global_participant_id", sort=False)["outer_fold"]
        .nunique()
        .eq(1)
        .all(),
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
        metrics.get("prediction_scope") == "strict_outer_fold_oof"
        and metrics.get("overall") == pair(predictions),
        "overall_metric_recalculation",
        checks,
    )
    _check(
        metrics.get("availability")
        == {
            "row_count": 11_325,
            "available_row_count": 11_325,
            "unavailable_row_count": 0,
        },
        "metric_availability_counts",
        checks,
    )
    for dataset_id in SOCIAL_CONTEXT_DATASET_IDS:
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


def _validate_search(
    search: Mapping[str, Any],
    config: Any,
    checks: list[str],
) -> None:
    candidates = deterministic_search_candidates(
        config.search_space,
        random_seed=config.random_seed,
        candidate_count=config.search_candidate_count,
    )
    candidate_ids = [str(item["candidate_id"]) for item in candidates]
    candidate_params = {
        str(item["candidate_id"]): dict(item["params"]) for item in candidates
    }
    _check(
        search.get("traversal") == "deterministic_full_grid"
        and search.get("candidate_count") == 16
        and search.get("full_grid_count") == 16
        and search.get("candidate_ids") == candidate_ids,
        "complete_search_grid",
        checks,
    )
    outer_rows = search.get("outer_folds")
    _check(
        isinstance(outer_rows, list) and len(outer_rows) == 5,
        "outer_search_count",
        checks,
    )
    by_candidate: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for outer_fold, outer in enumerate(outer_rows):
        rows = outer.get("candidates")
        _check(
            outer.get("outer_fold") == outer_fold
            and outer.get("selection_rule")
            == "highest_mean_inner_auprc_then_lowest_mean_inner_brier_then_candidate_id"
            and isinstance(rows, list)
            and len(rows) == 16
            and [item.get("candidate_id") for item in rows] == candidate_ids,
            f"outer_{outer_fold}_candidate_coverage",
            checks,
        )
        for row in rows:
            candidate_id = str(row["candidate_id"])
            folds = row.get("inner_folds")
            _check(
                row.get("params") == candidate_params[candidate_id]
                and isinstance(folds, list)
                and [fold.get("inner_fold") for fold in folds] == list(range(5))
                and all(
                    fold.get("selected_features") == list(EXPECTED_SELECTED_FEATURES)
                    for fold in folds
                )
                and np.isclose(
                    float(row["mean_inner_auprc"]),
                    np.mean([float(fold["auprc"]) for fold in folds]),
                )
                and np.isclose(
                    float(row["mean_inner_brier_score"]),
                    np.mean([float(fold["brier_score"]) for fold in folds]),
                ),
                f"outer_{outer_fold}_{candidate_id}_inner_audit",
                checks,
            )
            by_candidate[candidate_id].append(row)
        expected_best = sorted(
            rows,
            key=lambda item: (
                -_round_for_tie(
                    float(item["mean_inner_auprc"]), config.auprc_tie_tolerance
                ),
                float(item["mean_inner_brier_score"]),
                str(item["candidate_id"]),
            ),
        )[0]
        _check(
            outer.get("best_candidate") == expected_best,
            f"outer_{outer_fold}_best_candidate",
            checks,
        )

    production = search.get("production_selection")
    _check(
        isinstance(production, dict)
        and production.get("selection_rule")
        == (
            "highest_mean_nested_inner_auprc_then_lowest_mean_nested_inner_"
            "brier_then_candidate_id_without_outer_test_metrics"
        )
        and not _contains_mapping_key(production, "outer_test_metrics")
        and len(production.get("all_candidates", [])) == 16,
        "production_selection_boundary",
        checks,
    )
    expected_aggregated = []
    for candidate_id in candidate_ids:
        rows = by_candidate[candidate_id]
        expected_aggregated.append(
            {
                "candidate_id": candidate_id,
                "params": candidate_params[candidate_id],
                "mean_nested_inner_auprc": float(
                    np.mean([row["mean_inner_auprc"] for row in rows])
                ),
                "mean_nested_inner_brier_score": float(
                    np.mean([row["mean_inner_brier_score"] for row in rows])
                ),
            }
        )
    actual_aggregated = {
        str(item["candidate_id"]): item for item in production["all_candidates"]
    }
    _check(
        set(actual_aggregated) == set(candidate_ids)
        and all(
            actual_aggregated[item["candidate_id"]] == item
            for item in expected_aggregated
        ),
        "production_candidate_aggregation",
        checks,
    )
    expected_final = sorted(
        expected_aggregated,
        key=lambda item: (
            -_round_for_tie(
                item["mean_nested_inner_auprc"], config.auprc_tie_tolerance
            ),
            item["mean_nested_inner_brier_score"],
            item["candidate_id"],
        ),
    )[0]
    _check(
        all(production.get(key) == value for key, value in expected_final.items()),
        "production_best_candidate",
        checks,
    )


def _validate_outer_fold_bundles(
    report: Path,
    inputs: Any,
    assignments: Mapping[str, Mapping[str, Any]],
    predictions: pd.DataFrame,
    config: Any,
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
            isinstance(bundle, SocialContextExpertBundle)
            and bundle.run_id == config.run_id
            and bundle.training_scope == f"outer_fold_{outer_fold}_train"
            and bundle.preprocessor.selected_features == EXPECTED_SELECTED_FEATURES,
            f"outer_{outer_fold}_bundle_identity",
            checks,
        )
        expected_metadata = social_context._deterministic_catboost_metadata(
            config,
            bundle.hyperparameters,
            training_identity=f"outer_fold_{outer_fold}_aggregate",
        )
        _check(
            all(
                dict(bundle.classifier.get_metadata()).get(key) == value
                for key, value in expected_metadata.items()
            ),
            f"outer_{outer_fold}_deterministic_catboost_metadata",
            checks,
        )
        test_keys = {
            key
            for key, value in assignments.items()
            if int(value["outer_fold"]) == outer_fold
        }
        train_keys = all_keys - test_keys
        train_partition = social_context._transform_partition(inputs, train_keys)
        train_table = social_context._available_social_rows(train_partition)
        independent_preprocessor = SocialContextFoldPreprocessor.fit(train_table)
        expected_weight_audit = sample_weight_audit(
            train_table, compute_training_sample_weights(train_table)
        )
        _check(
            bundle.training_participant_count == len(train_keys)
            and bundle.training_participant_sha256 == participant_set_sha256(train_keys)
            and bundle.training_row_count == len(train_table)
            and bundle.preprocessor.to_dict() == independent_preprocessor.to_dict(),
            f"outer_{outer_fold}_training_and_preprocessing_scope",
            checks,
        )
        expected_positive = int(
            train_table.loc[
                train_table["binary_target"].eq(1), "global_participant_id"
            ].nunique()
        )
        _check(
            bundle.calibrator.method == "isotonic"
            and bundle.calibrator.fit_row_count == len(train_table)
            and bundle.calibrator.positive_participant_count == expected_positive
            and _sample_weight_audit_close(
                audit.get("training_sample_weights"), expected_weight_audit
            )
            and _sample_weight_audit_close(
                audit.get("calibration_sample_weights"), expected_weight_audit
            ),
            f"outer_{outer_fold}_weight_and_calibration_scope",
            checks,
        )
        _check(
            audit.get("test_participant_count") == len(test_keys)
            and audit.get("test_participant_sha256")
            == participant_set_sha256(test_keys)
            and audit.get("test_row_count") == len(test_keys)
            and audit.get("preprocessing_participant_count") == len(train_keys)
            and audit.get("preprocessing_participant_sha256")
            == participant_set_sha256(train_keys)
            and audit.get("catboost_model_sha256")
            == social_context._catboost_model_sha256(bundle.classifier),
            f"outer_{outer_fold}_fold_audit",
            checks,
        )
        test_table = (
            social_context._transform_partition(inputs, test_keys)
            .sort_values(["dataset_id", "canonical_row_index"], kind="stable")
            .reset_index(drop=True)
        )
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
        _check(
            _read_json(
                report
                / "preprocessing"
                / f"outer_fold_{outer_fold}"
                / "preprocessor.json"
            )
            == bundle.preprocessor.to_dict()
            and _read_json(report / "calibration" / f"outer_fold_{outer_fold}.json")
            == bundle.calibrator.to_dict(),
            f"outer_{outer_fold}_preprocessing_and_calibration_artifacts",
            checks,
        )


def _validate_final_audits(
    report: Path,
    bundle: SocialContextExpertBundle,
    checks: list[str],
) -> None:
    preprocessor = _read_json(report / "preprocessing" / "final" / "preprocessor.json")
    bundle_audit = _read_json(report / "preprocessing" / "final" / "bundle.json")
    calibrator = _read_json(report / "calibration" / "production.json")
    _check(
        preprocessor == bundle.preprocessor.to_dict()
        and calibrator == bundle.calibrator.to_dict()
        and bundle_audit.get("preprocessor") == bundle.preprocessor.to_dict()
        and bundle_audit.get("calibrator") == bundle.calibrator.to_dict()
        and bundle_audit.get("catboost_model_sha256")
        == social_context._catboost_model_sha256(bundle.classifier),
        "production_preprocessing_and_calibration_artifacts",
        checks,
    )


def _validate_feature_importance(
    report: Path,
    bundle: SocialContextExpertBundle,
    checks: list[str],
) -> None:
    payload = _read_json(report / "explanations" / "feature_importance.json")
    rows = payload.get("features")
    _check(
        payload.get("method") == "catboost_prediction_values_change_for_training_audit"
        and isinstance(rows, list)
        and {row.get("feature") for row in rows}
        == set(bundle.preprocessor.selected_features)
        and all(np.isfinite(float(row.get("importance"))) for row in rows),
        "feature_importance_contract",
        checks,
    )


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
        and set(bindings) == EXPECTED_CODE_BINDINGS,
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


def _validate_upstream_hashes(
    root: Path,
    bindings: Mapping[str, Mapping[str, Any]],
    checks: list[str],
) -> None:
    _check(
        tuple(bindings) == ("activity", "sleep", "joint", "physiology"),
        "upstream_guard_order",
        checks,
    )
    for name, expected in UPSTREAM_EXPERT_GUARDS.items():
        actual = bindings[name]
        model_path = _safe_root_path(root, str(expected["model_path"]))
        manifest_path = _safe_root_path(root, str(expected["manifest_path"]))
        _check(
            actual.get("model_sha256") == expected["model_sha256"]
            and actual.get("manifest_sha256") == expected["manifest_sha256"]
            and _sha256_file(model_path) == expected["model_sha256"]
            and _sha256_file(manifest_path) == expected["manifest_sha256"],
            f"upstream_{name}_protected_hashes",
            checks,
        )


def _validate_report_artifacts(report: Path, checks: list[str]) -> None:
    manifest = _read_json(report / "artifacts.json")
    rows = manifest.get("artifacts")
    _check(
        manifest.get("artifact_manifest_version")
        == "mood-social-social-context-report-artifacts-v1"
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
        raise ValidationError("repository binding path is unsafe")
    resolved = (root / candidate).resolve()
    if root.resolve() not in resolved.parents:
        raise ValidationError("repository binding path escapes repository root")
    return resolved


def _round_for_tie(value: float, tolerance: float) -> float:
    if tolerance <= 0:
        return value
    return round(value / tolerance) * tolerance


def _contains_mapping_key(value: Any, target: str) -> bool:
    if isinstance(value, Mapping):
        return target in value or any(
            _contains_mapping_key(item, target) for item in value.values()
        )
    if isinstance(value, list):
        return any(_contains_mapping_key(item, target) for item in value)
    return False


def _sample_weight_audit_close(actual: Any, expected: Mapping[str, Any]) -> bool:
    if not isinstance(actual, Mapping):
        return False
    if actual.get("definition") != expected.get("definition") or not np.isclose(
        float(actual.get("normalized_total", np.nan)),
        float(expected["normalized_total"]),
    ):
        return False
    actual_datasets = actual.get("by_dataset")
    expected_datasets = expected["by_dataset"]
    if not isinstance(actual_datasets, Mapping) or set(actual_datasets) != set(
        expected_datasets
    ):
        return False
    for dataset_id, expected_row in expected_datasets.items():
        actual_row = actual_datasets[dataset_id]
        if not isinstance(actual_row, Mapping) or not np.isclose(
            float(actual_row.get("total", np.nan)), float(expected_row["total"])
        ):
            return False
        actual_classes = actual_row.get("by_class")
        expected_classes = expected_row["by_class"]
        if not isinstance(actual_classes, Mapping) or set(actual_classes) != set(
            expected_classes
        ):
            return False
        if any(
            not np.isclose(float(actual_classes[key]), float(expected_value))
            for key, expected_value in expected_classes.items()
        ):
            return False
    return True


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValidationError("required MODEL-005 JSON is unreadable") from exc


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(chunk_size), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValidationError("required MODEL-005 file is unreadable") from exc
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
        print(f"MODEL-005 validation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
