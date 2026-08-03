"""Validate MODEL-004 artifacts, OOF coverage, and leakage boundaries."""

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
from sklearn.linear_model import LogisticRegression

from elderly_monitoring.modules.mental_health.mood_social.experts.physiology import (
    EXCLUDED_DATASET_IDS,
    PHYSIOLOGY_DATASET_IDS,
    PHYSIOLOGY_FEATURE_COLUMNS,
    PHYSIOLOGY_MASK_COLUMNS,
    EXPECTED_FEATURE_SCHEMA_SHA256,
    EXPECTED_SPLIT_SHA256,
    MODEL_BUNDLE_VERSION,
    MODEL_ID,
    MODEL_VERSION,
    PhysiologyExpertBundle,
    binary_metrics,
    deterministic_search_candidates,
    load_physiology_training_config,
    load_physiology_training_inputs,
    participant_set_sha256,
)
from elderly_monitoring.modules.mental_health.mood_social.feature_schema import (
    FEATURE_SCHEMA_VERSION,
)
from elderly_monitoring.modules.mental_health.mood_social.splits import SPLIT_ID


DEFAULT_CONFIG = Path("configs/training/mood_social_physiology_expert_v3_3_3.yaml")
DEFAULT_REPORT = Path("reports/mental_health/mood_social/MH-20260731-004")
DEFAULT_MODEL = Path("models/mental_health/mood_social/v3.3.3/physiology_expert.joblib")
DEFAULT_MODEL_MANIFEST = Path(
    "models/mental_health/mood_social/v3.3.3/physiology_expert_manifest.json"
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
EXPECTED_SELECTED_FEATURES = tuple(
    feature
    for feature in PHYSIOLOGY_FEATURE_COLUMNS
    if feature
    not in {
        "physiology.respiratory_abnormal_ratio",
        "physiology.valid_nights",
    }
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
    config = load_physiology_training_config(config_path)
    inputs = load_physiology_training_inputs(root)
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
    warning_artifact = _read_json(report / "failures" / "warnings.json")
    _validate_run(
        run,
        config,
        inputs,
        warning_artifact,
        checks,
    )
    _validate_model_manifest(model_manifest, config, inputs, checks)

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
    _check(isinstance(bundle, PhysiologyExpertBundle), "model_bundle_type", checks)
    _validate_bundle_identity(bundle, checks)
    available_keys, available_rows = _available_scope(inputs, set(assignments))
    _check(
        bundle.training_scope == "resilient_real_physiology_rows_for_production"
        and bundle.training_participant_count == len(available_keys)
        and bundle.training_participant_sha256 == participant_set_sha256(available_keys)
        and bundle.training_row_count == available_rows == 71,
        "production_training_scope",
        checks,
    )
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
    _validate_outer_fold_bundles(report, inputs, assignments, config.run_id, checks)
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
        "dataset_count": len(PHYSIOLOGY_DATASET_IDS),
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


def _validate_run(
    run: Mapping[str, Any],
    config: Any,
    inputs: Any,
    warning_artifact: Mapping[str, Any],
    checks: list[str],
) -> None:
    _check(
        run.get("status") == "completed"
        and run.get("task_id") == "MODEL-004"
        and run.get("model_id") == MODEL_ID
        and run.get("model_version") == MODEL_VERSION
        and run.get("run_id") == config.run_id
        and run.get("algorithm") == "ElasticNet Logistic",
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
    bindings = run.get("input_bindings")
    _check(
        isinstance(bindings, list)
        and len(bindings) == 5
        and [item["dataset_id"] for item in bindings] == list(inputs.input_bindings)
        and [item["training_included"] for item in bindings]
        == [False, True, False, False, False],
        "run_five_input_bindings",
        checks,
    )
    _check(
        run.get("excluded_sources") == list(EXCLUDED_DATASET_IDS)
        and run.get("excluded_source_policy")
        == "hard_excluded_no_all_missing_imputation_rows_enter_training",
        "run_excluded_source_policy",
        checks,
    )
    _check(
        run.get("upstream_expert_guards")
        == {
            name: dict(value) for name, value in inputs.upstream_expert_bindings.items()
        },
        "run_upstream_guards",
        checks,
    )
    warning_codes = {item.get("code") for item in warning_artifact.get("warnings", [])}
    _check(
        run.get("failures") == []
        and warning_artifact.get("failures") == []
        and warning_codes
        == {
            "PROXY_TRANSFER",
            "SMALL_RESILIENT_SOURCE",
            "INNER_VALIDATION_NO_POSITIVE",
            "ELASTICNET_MAX_ITER_REACHED",
            "STRUCTURAL_FEATURE_ABSENCE",
            "NO_PERFORMANCE_GATE",
        }
        and run.get("warnings") == warning_artifact.get("warnings"),
        "failure_and_warning_contract",
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
        and manifest.get("run_id") == config.run_id
        and manifest.get("algorithm") == "ElasticNet Logistic",
        "model_manifest_identity",
        checks,
    )
    _check(
        manifest.get("dataset_ids") == ["resilient"]
        and manifest.get("excluded_dataset_ids") == list(EXCLUDED_DATASET_IDS)
        and manifest.get("source_transform")
        == "resilient_canonical_same_semantics_identity"
        and manifest.get("row_eligibility")
        == "at_least_one_real_physiology_model_input",
        "model_manifest_source_policy",
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
        manifest.get("input_feature_names") == list(PHYSIOLOGY_FEATURE_COLUMNS)
        and manifest.get("selected_feature_names") == list(EXPECTED_SELECTED_FEATURES)
        and manifest.get("calibrator", {}).get("method") == "platt"
        and set(manifest.get("hyperparameters", {})) == {"C", "l1_ratio"},
        "model_manifest_model_contract",
        checks,
    )
    _check(
        manifest.get("upstream_expert_guards")
        == {
            name: dict(value) for name, value in inputs.upstream_expert_bindings.items()
        },
        "model_manifest_upstream_guards",
        checks,
    )


def _validate_bundle_identity(
    bundle: PhysiologyExpertBundle,
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
        bundle.dataset_ids == ("resilient",)
        and bundle.input_feature_names == PHYSIOLOGY_FEATURE_COLUMNS
        and bundle.preprocessor.selected_features == EXPECTED_SELECTED_FEATURES
        and bundle.preprocessor.standardization
        == "training_fold_mean_std_after_median_imputation"
        and not bundle.preprocessor.add_missing_indicators,
        "bundle_feature_contract",
        checks,
    )
    _check(
        isinstance(bundle.classifier, LogisticRegression)
        and bundle.classifier.solver == "saga"
        and bundle.classifier.max_iter == 5000
        and bundle.classifier.l1_ratio is not None
        and bundle.calibrator.method == "platt",
        "bundle_algorithm_contract",
        checks,
    )


def _validate_bundle_mask_behavior(
    bundle: PhysiologyExpertBundle,
    checks: list[str],
) -> None:
    all_missing = pd.DataFrame(
        {
            **{name: [np.nan] for name in PHYSIOLOGY_FEATURE_COLUMNS},
            **{name: [0] for name in PHYSIOLOGY_MASK_COLUMNS},
        }
    )
    missing_result = bundle.predict(all_missing)
    _check(
        int(missing_result.loc[0, "expert_mask"]) == 0
        and pd.isna(missing_result.loc[0, "raw_probability"])
        and pd.isna(missing_result.loc[0, "calibrated_probability"]),
        "bundle_all_missing_mask_behavior",
        checks,
    )
    one_observed = all_missing.copy()
    feature = "physiology.heart_rate_mean_bpm"
    one_observed.loc[0, feature] = 60.0
    one_observed.loc[0, f"feature_mask.{feature}"] = 1
    observed_result = bundle.predict(one_observed)
    _check(
        int(observed_result.loc[0, "expert_mask"]) == 1
        and np.isfinite(observed_result.loc[0, "raw_probability"])
        and np.isfinite(observed_result.loc[0, "calibrated_probability"]),
        "bundle_one_observed_mask_behavior",
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
        and predictions["prediction_schema_version"].iloc[0]
        == "mood-social-physiology-oof-predictions-v1",
        "prediction_schema_version",
        checks,
    )
    _check(
        len(predictions) == 73
        and not predictions["prediction_id"].duplicated().any()
        and set(predictions["dataset_id"].astype(str)) == {"resilient"}
        and predictions["global_participant_id"].nunique() == 73
        and int(predictions["binary_target"].sum()) == 10,
        "prediction_coverage",
        checks,
    )
    available = predictions["expert_mask"].eq(1)
    unavailable = ~available
    _check(
        int(available.sum()) == 71
        and int(predictions.loc[available, "binary_target"].sum()) == 9
        and np.isfinite(predictions.loc[available, "raw_probability"]).all()
        and np.isfinite(predictions.loc[available, "calibrated_probability"]).all()
        and predictions.loc[available, "raw_probability"].between(0, 1).all()
        and predictions.loc[available, "calibrated_probability"].between(0, 1).all()
        and predictions.loc[unavailable, "raw_probability"].isna().all()
        and predictions.loc[unavailable, "calibrated_probability"].isna().all(),
        "prediction_probability_mask_contract",
        checks,
    )
    source = inputs.frames["resilient"].copy()
    source["canonical_row_index"] = source.index.astype("int64")
    source["prediction_id"] = "resilient::row=" + source["canonical_row_index"].astype(
        str
    )
    masks = source[list(PHYSIOLOGY_MASK_COLUMNS)].apply(pd.to_numeric, errors="raise")
    source["expert_mask"] = masks.sum(axis=1).gt(0).astype("int8")
    source["observed_feature_count"] = masks.sum(axis=1).astype("int64")
    source["observed_feature_fraction"] = source["observed_feature_count"] / len(
        PHYSIOLOGY_FEATURE_COLUMNS
    )
    source["outer_fold"] = (
        source["global_participant_id"]
        .astype(str)
        .map({key: int(value["outer_fold"]) for key, value in assignments.items()})
    )
    expected_columns = [
        "prediction_id",
        "dataset_id",
        "global_participant_id",
        "canonical_row_index",
        "binary_target",
        "outer_fold",
        "expert_mask",
        "observed_feature_count",
        "observed_feature_fraction",
    ]
    expected = source[expected_columns].reset_index(drop=True)
    actual = predictions[expected_columns].reset_index(drop=True)
    for column in (
        "canonical_row_index",
        "binary_target",
        "outer_fold",
        "expert_mask",
        "observed_feature_count",
    ):
        expected[column] = pd.to_numeric(expected[column]).astype("int64")
        actual[column] = pd.to_numeric(actual[column]).astype("int64")
    _check(actual.equals(expected), "prediction_row_alignment", checks)


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

    _check(metrics.get("overall") == pair(predictions), "overall_metrics", checks)
    _check(
        metrics.get("availability")
        == {
            "row_count": 73,
            "available_row_count": 71,
            "unavailable_row_count": 2,
        },
        "metric_availability_counts",
        checks,
    )
    _check(
        metrics.get("by_dataset", {}).get("resilient") == pair(predictions),
        "resilient_metrics",
        checks,
    )
    for outer_fold in range(5):
        selected = predictions[predictions["outer_fold"].eq(outer_fold)]
        _check(
            metrics.get("by_outer_fold", {}).get(str(outer_fold)) == pair(selected),
            f"outer_{outer_fold}_metrics",
            checks,
        )


def _validate_search(
    search: Mapping[str, Any],
    config: Any,
    checks: list[str],
) -> None:
    expected = deterministic_search_candidates(
        config.search_space,
        random_seed=config.random_seed,
        candidate_count=config.search_candidate_count,
    )
    expected_ids = [item["candidate_id"] for item in expected]
    _check(
        search.get("traversal") == "deterministic_full_grid"
        and search.get("candidate_count") == 20
        and search.get("full_grid_count") == 20
        and search.get("candidate_ids") == expected_ids,
        "complete_search_grid",
        checks,
    )
    outer_rows = search.get("outer_folds")
    _check(
        isinstance(outer_rows, list) and len(outer_rows) == 5,
        "outer_search_count",
        checks,
    )
    empty_positive_folds: set[tuple[int, int]] = set()
    for outer_fold, outer in enumerate(outer_rows):
        candidates = outer.get("candidates")
        _check(
            outer.get("outer_fold") == outer_fold
            and isinstance(candidates, list)
            and len(candidates) == 20
            and [item["candidate_id"] for item in candidates] == expected_ids,
            f"outer_{outer_fold}_candidate_coverage",
            checks,
        )
        for fold in candidates[0]["inner_folds"]:
            _check(
                isinstance(fold.get("classifier_convergence"), dict)
                and fold["classifier_convergence"].get("max_iter") == 5000,
                f"outer_{outer_fold}_inner_{fold['inner_fold']}_convergence_audit",
                checks,
            )
            if int(fold["positive_row_count"]) == 0:
                empty_positive_folds.add((outer_fold, int(fold["inner_fold"])))
                _check(
                    fold["auprc"] is None,
                    f"outer_{outer_fold}_inner_{fold['inner_fold']}_null_auprc",
                    checks,
                )
        ranked = sorted(
            candidates,
            key=lambda item: (
                -_round_for_tie(
                    float(item["pooled_inner_auprc"]),
                    config.auprc_tie_tolerance,
                ),
                float(item["pooled_inner_brier_score"]),
                str(item["candidate_id"]),
            ),
        )
        _check(
            outer.get("best_candidate") == ranked[0],
            f"outer_{outer_fold}_candidate_ranking",
            checks,
        )
    _check(
        empty_positive_folds == {(2, 3), (4, 0)},
        "empty_positive_inner_fold_set",
        checks,
    )
    production = search.get("production_selection")
    candidates = production.get("all_candidates")
    _check(
        isinstance(candidates, list)
        and len(candidates) == 20
        and {item["candidate_id"] for item in candidates} == set(expected_ids),
        "production_candidate_coverage",
        checks,
    )
    ranked = sorted(
        candidates,
        key=lambda item: (
            -_round_for_tie(
                float(item["pooled_nested_inner_auprc"]),
                config.auprc_tie_tolerance,
            ),
            float(item["pooled_nested_inner_brier_score"]),
            str(item["candidate_id"]),
        ),
    )
    _check(
        production.get("candidate_id") == ranked[0]["candidate_id"]
        and production.get("params") == ranked[0]["params"],
        "production_candidate_ranking",
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
    expected_test_available = [(15, 2), (15, 2), (15, 2), (12, 1), (14, 2)]
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
            isinstance(bundle, PhysiologyExpertBundle)
            and bundle.run_id == run_id
            and bundle.training_scope == f"outer_fold_{outer_fold}_train"
            and bundle.calibrator.method == "platt",
            f"outer_{outer_fold}_bundle_identity",
            checks,
        )
        test_keys = {
            key
            for key, value in assignments.items()
            if int(value["outer_fold"]) == outer_fold
        }
        train_keys = all_keys - test_keys
        train_available_keys, train_available_rows = _available_scope(
            inputs, train_keys
        )
        test_available_keys, test_available_rows = _available_scope(inputs, test_keys)
        test_positive = int(
            inputs.frames["resilient"]
            .loc[
                inputs.frames["resilient"]["global_participant_id"]
                .astype(str)
                .isin(test_available_keys),
                "binary_target",
            ]
            .sum()
        )
        _check(
            (test_available_rows, test_positive) == expected_test_available[outer_fold],
            f"outer_{outer_fold}_available_test_count",
            checks,
        )
        _check(
            bundle.training_participant_count == len(train_available_keys)
            and bundle.training_participant_sha256
            == participant_set_sha256(train_available_keys)
            and bundle.training_row_count == train_available_rows
            and bundle.calibrator.fit_row_count == train_available_rows
            and audit.get("test_participant_count") == len(test_keys)
            and audit.get("test_participant_sha256")
            == participant_set_sha256(test_keys),
            f"outer_{outer_fold}_participant_scope",
            checks,
        )
        _check(
            audit.get("classifier_state_sha256")
            == _classifier_state_sha256(bundle.classifier)
            and audit.get("classifier_convergence")
            == _classifier_convergence(bundle.classifier),
            f"outer_{outer_fold}_classifier_state",
            checks,
        )


def _available_scope(inputs: Any, participant_keys: set[str]) -> tuple[set[str], int]:
    frame = inputs.frames["resilient"]
    selected = frame[frame["global_participant_id"].astype(str).isin(participant_keys)]
    masks = selected[list(PHYSIOLOGY_MASK_COLUMNS)].apply(pd.to_numeric, errors="raise")
    available = masks.sum(axis=1).gt(0)
    keys = set(selected.loc[available, "global_participant_id"].astype(str))
    if not keys:
        raise ValidationError("PhysiologyExpert available scope is empty")
    return keys, int(available.sum())


def _classifier_state_sha256(classifier: LogisticRegression) -> str:
    return hashlib.sha256(
        _canonical_json_bytes(
            {
                "classes": [int(value) for value in classifier.classes_],
                "coefficient": [float(value) for value in classifier.coef_[0]],
                "intercept": [float(value) for value in classifier.intercept_],
                "n_iter": [int(value) for value in classifier.n_iter_],
            }
        )
    ).hexdigest()


def _classifier_convergence(classifier: LogisticRegression) -> dict[str, Any]:
    iteration_count = int(np.max(classifier.n_iter_))
    return {
        "iteration_count": iteration_count,
        "max_iter": int(classifier.max_iter),
        "converged_before_max_iter": iteration_count < int(classifier.max_iter),
    }


def _validate_upstream_hashes(
    root: Path,
    bindings: Mapping[str, Mapping[str, Any]],
    checks: list[str],
) -> None:
    for name in ("activity", "sleep", "joint"):
        binding = bindings[name]
        _check(
            _sha256_file(root / binding["model_path"]) == binding["model_sha256"]
            and _sha256_file(root / binding["manifest_path"])
            == binding["manifest_sha256"],
            f"upstream_{name}_hash",
            checks,
        )


def _validate_report_artifacts(report: Path, checks: list[str]) -> None:
    manifest = _read_json(report / "artifacts.json")
    rows = manifest.get("artifacts")
    _check(
        manifest.get("artifact_manifest_version")
        == "mood-social-physiology-report-artifacts-v1"
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


def _round_for_tie(value: float, tolerance: float) -> float:
    if tolerance <= 0:
        return value
    return round(value / tolerance) * tolerance


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValidationError("required MODEL-004 JSON is unreadable") from exc


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(chunk_size), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValidationError("required MODEL-004 file is unreadable") from exc
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
        print(f"MODEL-004 validation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
