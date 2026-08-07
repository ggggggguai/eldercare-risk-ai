"""Train the frozen V3.4 PSYCHE-D forecast experiment.

This command performs all model selection inside the outer training
participants, then writes strict outer OOF predictions and fold bundles.  It
never imports or modifies the backend or the V3.3 production package.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from itertools import product
from pathlib import Path
from typing import Any, Mapping

import joblib
import pandas as pd
import yaml

ALGORITHM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ALGORITHM_ROOT / "src"))

from elderly_monitoring.modules.mental_health.mood_social.forecast_experiment import (  # noqa: E402
    sha256_file,
)
from elderly_monitoring.modules.mental_health.mood_social.forecast_modeling import (  # noqa: E402
    MODEL_FAMILIES,
    SEED,
    crossfit_inner,
    fit_predict,
    isotonic_calibrate,
    json_params,
    participant_equal_weights,
    predict_fitted,
    select_threshold,
    weighted_metrics,
)


EXPERIMENT_VERSION = "mood-social-forecast-v3.4"
OUTPUT_ROOT = (
    ALGORITHM_ROOT / "artifacts" / "mental_health" / "mood_social" / "forecast_v3.4"
)
CONFIG_ROOT = ALGORITHM_ROOT / "configs" / "experiments"
WORKSPACE_ROOT = ALGORITHM_ROOT.parents[1]
FEATURE_NAMES = tuple(
    json.loads((OUTPUT_ROOT / "raw_feature_names.json").read_text(encoding="utf-8"))[
        "raw_feature_names"
    ]
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _verify_environment() -> None:
    lock = json.loads(
        (CONFIG_ROOT / "mood_social_forecast_v3_4_environment.lock").read_text(
            encoding="utf-8"
        )
    )
    actual_python = ".".join(str(value) for value in sys.version_info[:3])
    if actual_python != lock["python"]:
        raise RuntimeError(
            f"formal training requires Python {lock['python']}, got {actual_python}"
        )
    distribution_names = {
        "numpy": "numpy",
        "pandas": "pandas",
        "pyarrow": "pyarrow",
        "scikit-learn": "scikit-learn",
        "lightgbm": "lightgbm",
        "PyYAML": "PyYAML",
        "pytest": "pytest",
    }
    mismatches = []
    for key, distribution in distribution_names.items():
        actual = importlib.metadata.version(distribution)
        expected = str(lock["packages"][key])
        if actual != expected:
            mismatches.append(f"{key}={actual} (expected {expected})")
    if mismatches:
        raise RuntimeError(
            "formal training environment mismatch: " + ", ".join(mismatches)
        )


def _load_frozen() -> tuple[
    dict[str, Any], dict[str, Any], dict[str, list[dict[str, Any]]]
]:
    _verify_environment()
    manifest_path = OUTPUT_ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = yaml.safe_load(
        (CONFIG_ROOT / "mood_social_forecast_v3_4.yaml").read_text(encoding="utf-8")
    )
    inner = json.loads(
        (OUTPUT_ROOT / "inner_split_manifest.json").read_text(encoding="utf-8")
    )
    if sha256_file(manifest_path) != config.get("context_manifest_sha256"):
        raise RuntimeError("context manifest hash does not match the frozen config")
    source_root = WORKSPACE_ROOT / "数据集" / "心理" / "PSYCHE-D"
    for name, expected in config["source_artifact_hashes"].items():
        if sha256_file(source_root / name) != expected:
            raise RuntimeError(f"source hash mismatch: {name}")
    acceptance_path = ALGORITHM_ROOT / config["v3_3_acceptance_path"]
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    if acceptance.get("status") != "passed":
        raise RuntimeError("V3.3 acceptance status is not passed")
    if sha256_file(acceptance_path) != config["v3_3_acceptance_sha256"]:
        raise RuntimeError("V3.3 acceptance hash does not match")
    split_path = ALGORITHM_ROOT / config["outer_fold_manifest"]
    if sha256_file(split_path) != config["outer_fold_manifest_sha256"]:
        raise RuntimeError("V3.3 outer split hash does not match")
    frozen_files = {
        CONFIG_ROOT / "mood_social_forecast_v3_4_search_space.json": config[
            "search_space_sha256"
        ],
        CONFIG_ROOT / "mood_social_forecast_v3_4_lightgbm_candidates.json": config[
            "lightgbm_candidates_sha256"
        ],
        CONFIG_ROOT / "mood_social_forecast_v3_4_environment.lock": config[
            "environment_lock_sha256"
        ],
    }
    for path, expected in frozen_files.items():
        if sha256_file(path) != expected:
            raise RuntimeError(f"frozen configuration hash mismatch: {path.name}")
    if manifest.get("task_counts") != {
        "common": 8494,
        "forecast_1m": 9393,
        "forecast_2m": 9280,
    }:
        raise RuntimeError("frozen context task counts do not match V3.4 contract")
    if manifest.get("participant_counts") != {"forecast_1m": 3635, "forecast_2m": 3593}:
        raise RuntimeError(
            "frozen context participant counts do not match V3.4 contract"
        )
    if (
        manifest.get("release_status") != "experimental"
        or manifest.get("execution_mode") != "offline_only"
        or manifest.get("decision_authority") != "shadow_only"
        or manifest.get("product_visible") is not False
    ):
        raise RuntimeError("forecast release boundary is not frozen")
    if (
        sha256_file(OUTPUT_ROOT / "inner_split_manifest.json")
        != "64f9e4a8d548f9c5152275b9d1fc8933814bbc9730ca74af5557284657a4e4ef"
    ):
        raise RuntimeError("inner split hash does not match frozen context")
    candidates_path = CONFIG_ROOT / "mood_social_forecast_v3_4_lightgbm_candidates.json"
    candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
    if (
        candidates.get("candidate_count") != 64
        or len(candidates.get("candidates", [])) != 64
    ):
        raise RuntimeError("LightGBM candidate snapshot is not the frozen 64-item list")
    if len(inner.get("rows", [])) != 15_236:
        raise RuntimeError("inner split row count does not match frozen context")
    by_outer: dict[str, list[dict[str, Any]]] = {str(index): [] for index in range(5)}
    for row in inner["rows"]:
        by_outer[str(int(row["outer_fold_id"]))].append(row)
    return manifest, config, by_outer


def _param_grids() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    space = json.loads(
        (CONFIG_ROOT / "mood_social_forecast_v3_4_search_space.json").read_text(
            encoding="utf-8"
        )
    )
    elastic = [
        {"C": float(c), "l1_ratio": float(ratio)}
        for c, ratio in product(
            space["elasticnet_logistic"]["C"], space["elasticnet_logistic"]["l1_ratio"]
        )
    ]
    candidates = json.loads(
        (CONFIG_ROOT / "mood_social_forecast_v3_4_lightgbm_candidates.json").read_text(
            encoding="utf-8"
        )
    )["candidates"]
    return elastic, [dict(candidate) for candidate in candidates]


def _inner_assignment(rows: list[dict[str, Any]], outer_fold: int) -> dict[str, int]:
    return {
        str(row["global_participant_id"]): int(row["inner_fold_id"])
        for row in rows
        if int(row["outer_fold_id"]) == outer_fold
    }


def _score_inner(predictions: pd.DataFrame) -> dict[str, float]:
    weights = participant_equal_weights(predictions)
    y = predictions["future_binary_target"].to_numpy(dtype="int8")
    p = predictions["p_raw"].to_numpy(dtype="float64")
    metrics = weighted_metrics(y, p, p, 0.5, weights)
    return {"auprc": metrics["auprc"], "brier": metrics["brier"]}


def _search_family(
    family: str,
    frame: pd.DataFrame,
    assignment: Mapping[str, int],
    outer_fold: int,
    grid: list[dict[str, Any]],
) -> tuple[dict[str, Any], pd.DataFrame, list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    best_params: dict[str, Any] | None = None
    best_prediction: pd.DataFrame | None = None
    best_auprc = -1.0
    best_brier = float("inf")
    candidates = grid if family != "dummy" else [{}]
    for index, params in enumerate(candidates):
        prediction = crossfit_inner(
            family, frame, assignment, outer_fold, FEATURE_NAMES, params
        )
        score = _score_inner(prediction)
        records.append({"candidate_index": index, "params": dict(params), **score})
        if score["auprc"] > best_auprc + 1e-12 or (
            abs(score["auprc"] - best_auprc) <= 1e-12
            and score["brier"] < best_brier - 1e-12
        ):
            best_auprc = score["auprc"]
            best_brier = score["brier"]
            best_params = dict(params)
            best_prediction = prediction
    if best_params is None or best_prediction is None:
        raise RuntimeError(f"no inner candidate selected for {family}")
    return best_params, best_prediction, records


def _train_outer_family(
    family: str,
    frame: pd.DataFrame,
    outer_fold: int,
    assignment: Mapping[str, int],
    params: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    train = frame[frame["outer_fold_id"] != outer_fold].copy()
    test = frame[frame["outer_fold_id"] == outer_fold].copy()
    if train.empty or test.empty:
        raise RuntimeError(f"outer fold {outer_fold} has an empty split")
    inner_raw = crossfit_inner(
        family, train, assignment, outer_fold, FEATURE_NAMES, params
    )
    inner_weights = participant_equal_weights(inner_raw)
    if family == "dummy":
        calibrator = None
        inner_raw["p_calibrated"] = inner_raw["p_raw"]
    else:
        calibrator = isotonic_calibrate(
            inner_raw["future_binary_target"], inner_raw["p_raw"], inner_weights
        )
        inner_raw["p_calibrated"] = calibrator.predict(
            inner_raw["p_raw"].to_numpy(dtype="float64")
        )
    threshold, threshold_record = select_threshold(
        inner_raw["future_binary_target"], inner_raw["p_calibrated"], inner_weights
    )
    _, estimator, preprocessor = fit_predict(family, train, test, FEATURE_NAMES, params)
    p_raw = predict_fitted(family, estimator, preprocessor, test, FEATURE_NAMES)
    p_calibrated = p_raw if calibrator is None else calibrator.predict(p_raw)
    output = test[
        [
            "participant_id",
            "global_participant_id",
            "target_window_id",
            "task_id",
            "outer_fold_id",
            "future_binary_target",
        ]
    ].copy()
    output = output.rename(columns={"future_binary_target": "y_true"})
    output["model_family"] = family
    output["p_raw"] = p_raw
    output["p_calibrated"] = p_calibrated
    output["fold_threshold"] = threshold
    output["train_sample_weight_policy"] = (
        "participant_class_equal_half"
        if family != "dummy"
        else "participant_equal_prior_no_class_balance"
    )
    output["evaluation_weight"] = 0.0
    output["selected_params_json"] = json_params(params)
    output["inner_calibration_participant_count"] = int(
        inner_raw["global_participant_id"].nunique()
    )
    bundle = {
        "schema_version": "mood-social-forecast-v3.4-fold-bundle-v1",
        "experiment_version": EXPERIMENT_VERSION,
        "task_id": str(frame["task_id"].iloc[0]),
        "model_family": family,
        "outer_fold_id": outer_fold,
        "selected_params": dict(params),
        "estimator": estimator,
        "preprocessor": preprocessor,
        "calibrator": calibrator,
        "threshold": threshold,
        "threshold_record": threshold_record,
        "train_participant_ids": sorted(
            train["global_participant_id"].unique().tolist()
        ),
        "test_participant_ids": sorted(test["global_participant_id"].unique().tolist()),
        "inner_calibration_participant_ids": sorted(
            inner_raw["global_participant_id"].unique().tolist()
        ),
        "raw_feature_names": list(FEATURE_NAMES),
        "derived_feature_names": list(preprocessor.derived_feature_names)
        if preprocessor is not None
        else [],
        "train_sample_weight_policy": output["train_sample_weight_policy"].iloc[0],
        "evaluation_weighting": "participant_equal",
        "release_status": "experimental",
        "execution_mode": "offline_only",
        "decision_authority": "shadow_only",
        "product_visible": False,
    }
    return output, {
        "bundle": bundle,
        "inner_predictions": inner_raw,
        "threshold_record": threshold_record,
    }


def run_training() -> dict[str, Any]:
    manifest, config, inner_rows_by_outer = _load_frozen()
    elastic_grid, lgb_grid = _param_grids()
    if len(elastic_grid) != 20 or len(lgb_grid) != 64:
        raise RuntimeError("frozen hyperparameter grids have changed")
    all_oof: list[pd.DataFrame] = []
    training_summary: dict[str, Any] = {
        "tasks": {},
        "model_families": list(MODEL_FAMILIES),
    }
    for task_id in ("forecast_1m", "forecast_2m"):
        samples = pd.read_parquet(OUTPUT_ROOT / task_id / "samples.parquet")
        if len(samples) != int(manifest["task_counts"][task_id]):
            raise RuntimeError(f"{task_id} sample count changed")
        task_root = OUTPUT_ROOT / task_id
        task_root.mkdir(parents=True, exist_ok=True)
        task_summary: dict[str, Any] = {
            "sample_count": len(samples),
            "participant_count": int(samples.global_participant_id.nunique()),
            "models": {},
        }
        for family in MODEL_FAMILIES:
            family_root = task_root / family
            (family_root / "folds").mkdir(parents=True, exist_ok=True)
            (family_root / "search").mkdir(parents=True, exist_ok=True)
            family_oof: list[pd.DataFrame] = []
            family_summary: dict[str, Any] = {"outer_folds": []}
            grid = (
                [{}]
                if family == "dummy"
                else (elastic_grid if family == "elasticnet_logistic" else lgb_grid)
            )
            for outer_fold in range(5):
                outer_train = samples[samples["outer_fold_id"] != outer_fold].copy()
                assignment = _inner_assignment(
                    inner_rows_by_outer[str(outer_fold)], outer_fold
                )
                eligible_ids = set(outer_train.global_participant_id)
                if not set(assignment).issuperset(eligible_ids):
                    raise RuntimeError(
                        "inner split does not cover outer training participants"
                    )
                selected, _, search_records = _search_family(
                    family, outer_train, assignment, outer_fold, grid
                )
                _write_json(
                    family_root / "search" / f"outer_fold_{outer_fold}.json",
                    {
                        "schema_version": "mood-social-forecast-v3.4-search-record-v1",
                        "outer_fold_id": outer_fold,
                        "model_family": family,
                        "candidate_count": len(search_records),
                        "selected_params": selected,
                        "records": search_records,
                        "inner_split_sha256": sha256_file(
                            OUTPUT_ROOT / "inner_split_manifest.json"
                        ),
                    },
                )
                fold_oof, details = _train_outer_family(
                    family, samples, outer_fold, assignment, selected
                )
                family_oof.append(fold_oof)
                joblib.dump(
                    details["bundle"],
                    family_root / "folds" / f"outer_fold_{outer_fold}.joblib",
                )
                details["inner_predictions"].to_parquet(
                    family_root
                    / "folds"
                    / f"outer_fold_{outer_fold}_inner_oof.parquet",
                    index=False,
                )
                _write_json(
                    family_root / "folds" / f"outer_fold_{outer_fold}_threshold.json",
                    details["threshold_record"],
                )
                _write_json(
                    family_root / "folds" / f"outer_fold_{outer_fold}_scope.json",
                    {
                        "schema_version": "mood-social-forecast-v3.4-fold-scope-v1",
                        "task_id": task_id,
                        "model_family": family,
                        "outer_fold_id": outer_fold,
                        "train_participant_ids": details["bundle"][
                            "train_participant_ids"
                        ],
                        "inner_calibration_participant_ids": details["bundle"][
                            "inner_calibration_participant_ids"
                        ],
                        "test_participant_ids": details["bundle"][
                            "test_participant_ids"
                        ],
                        "test_target_window_ids": sorted(
                            fold_oof["target_window_id"].tolist()
                        ),
                    },
                )
                family_summary["outer_folds"].append(
                    {
                        "outer_fold_id": outer_fold,
                        "train_sample_count": len(outer_train),
                        "test_sample_count": len(fold_oof),
                        "train_participant_count": int(
                            outer_train.global_participant_id.nunique()
                        ),
                        "test_participant_count": int(
                            fold_oof.global_participant_id.nunique()
                        ),
                        "selected_params": selected,
                        "threshold": float(fold_oof.fold_threshold.iloc[0]),
                        "inner_calibration_participant_count": int(
                            details["inner_predictions"].global_participant_id.nunique()
                        ),
                    }
                )
                print(
                    json.dumps(
                        {
                            "event": "outer_fold_complete",
                            "task_id": task_id,
                            "model_family": family,
                            "outer_fold_id": outer_fold,
                            "selected_params": selected,
                            "threshold": float(fold_oof.fold_threshold.iloc[0]),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    flush=True,
                )
            combined = pd.concat(family_oof, ignore_index=True)
            if combined.target_window_id.duplicated().any():
                raise RuntimeError(
                    f"{task_id}/{family} outer OOF has duplicate windows"
                )
            weights = participant_equal_weights(
                combined.rename(columns={"y_true": "future_binary_target"})
            )
            combined["evaluation_weight"] = weights
            combined.to_parquet(family_root / "oof_predictions.parquet", index=False)
            all_oof.append(combined)
            task_summary["models"][family] = family_summary
        task_summary["positive_window_count"] = int(samples.future_binary_target.sum())
        task_summary["positive_participant_count"] = int(
            samples.loc[
                samples.future_binary_target.eq(1), "global_participant_id"
            ].nunique()
        )
        training_summary["tasks"][task_id] = task_summary
    oof = pd.concat(all_oof, ignore_index=True)
    oof = oof.sort_values(
        ["task_id", "model_family", "target_window_id"], kind="mergesort"
    ).reset_index(drop=True)
    oof.to_parquet(OUTPUT_ROOT / "oof_predictions.parquet", index=False)
    environment_lock = json.loads(
        (CONFIG_ROOT / "mood_social_forecast_v3_4_environment.lock").read_text(
            encoding="utf-8"
        )
    )
    _write_json(
        OUTPUT_ROOT / "training_manifest.json",
        {
            "schema_version": "mood-social-forecast-v3.4-training-manifest-v1",
            "experiment_version": EXPERIMENT_VERSION,
            "context_manifest_sha256": sha256_file(OUTPUT_ROOT / "manifest.json"),
            "inner_split_manifest_sha256": sha256_file(
                OUTPUT_ROOT / "inner_split_manifest.json"
            ),
            "search_space_sha256": sha256_file(
                CONFIG_ROOT / "mood_social_forecast_v3_4_search_space.json"
            ),
            "lightgbm_candidates_sha256": sha256_file(
                CONFIG_ROOT / "mood_social_forecast_v3_4_lightgbm_candidates.json"
            ),
            "environment_lock_sha256": sha256_file(
                CONFIG_ROOT / "mood_social_forecast_v3_4_environment.lock"
            ),
            "environment": environment_lock,
            "random_seed": SEED,
            "outer_fold_count": 5,
            "inner_fold_count": 5,
            "model_families": list(MODEL_FAMILIES),
            "oof_row_count": len(oof),
            "tasks": training_summary["tasks"],
            "release_status": "experimental",
            "execution_mode": "offline_only",
            "decision_authority": "shadow_only",
            "product_visible": False,
        },
    )
    return training_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary", action="store_true", help="print a compact JSON summary"
    )
    args = parser.parse_args()
    summary = run_training()
    print(
        json.dumps(
            {"status": "pass", "summary": summary}
            if args.summary
            else {"status": "pass"},
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
