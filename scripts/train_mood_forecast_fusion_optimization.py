"""Run strict FORECAST-OPT-001D fusion, calibration, and outer OOF."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml


ALGORITHM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ALGORITHM_ROOT / "src"))

from elderly_monitoring.modules.mental_health.mood_social.forecast_experiment import (  # noqa: E402
    sha256_file,
)
from elderly_monitoring.modules.mental_health.mood_social.forecast_fusion_optimization import (  # noqa: E402
    COMPONENT_FAMILIES,
    compare_calibrators,
    component_oof_table,
    freeze_calibration_and_threshold,
    search_blend_weights,
    select_family_components,
)
from elderly_monitoring.modules.mental_health.mood_social.forecast_optimization import (  # noqa: E402
    fit_binary_candidate,
    strict_auxiliary_outer_views,
)


CONFIG_PATH = (
    ALGORITHM_ROOT
    / "configs"
    / "experiments"
    / "mood_social_forecast_v3_4_opt_fusion.yaml"
)
BASELINE_ROOT = (
    ALGORITHM_ROOT / "artifacts" / "mental_health" / "mood_social" / "forecast_v3.4"
)
CANDIDATE_ROOT = BASELINE_ROOT / "optimization_candidates"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_sums(root: Path) -> None:
    rows = [
        f"{sha256_file(path)}  {path.relative_to(root).as_posix()}"
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "SHA256SUMS"
    ]
    (root / "SHA256SUMS").write_text(
        "\n".join(rows) + "\n", encoding="utf-8", newline="\n"
    )


def _append_attempt(root: Path, status: str, detail: str) -> None:
    path = root / "attempts.jsonl"
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "stage": "FORECAST-OPT-001D",
                    "status": status,
                    "detail": detail,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )


def _load_config(run_id: str) -> tuple[dict[str, Any], Path]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["task_id"] != "FORECAST-OPT-001D":
        raise RuntimeError("001D config task identity changed")
    if config["upstream_run_id"] != run_id:
        raise RuntimeError("001D must use its frozen upstream run")
    if (
        config["backend_change"]
        or config["api_change"]
        or config["v3_3_package_change"]
    ):
        raise RuntimeError("001D isolation flags were relaxed")
    root = CANDIDATE_ROOT / run_id
    if root.parent != CANDIDATE_ROOT or not root.is_dir():
        raise RuntimeError("upstream candidate path is invalid")
    for relative, expected in (
        ("selection_manifest.json", config["upstream_selection_manifest_sha256"]),
        (
            "../../../configs/experiments/mood_social_forecast_v3_4_opt.yaml",
            config["upstream_config_sha256"],
        ),
    ):
        path = (
            root / relative
            if not relative.startswith("../")
            else ALGORITHM_ROOT
            / "configs"
            / "experiments"
            / "mood_social_forecast_v3_4_opt.yaml"
        )
        if sha256_file(path) != expected:
            raise RuntimeError(f"frozen upstream hash changed: {path.name}")
    manifest = json.loads(
        (root / "selection_manifest.json").read_text(encoding="utf-8")
    )
    if (
        manifest.get("status") != "selection_completed"
        or manifest.get("outer_test_used_for_selection") is not False
        or manifest.get("outer_oof_generated") is not False
    ):
        raise RuntimeError("001C selection boundary changed")
    return config, root


def _inner_assignments() -> dict[int, dict[str, int]]:
    payload = json.loads(
        (BASELINE_ROOT / "inner_split_manifest.json").read_text(encoding="utf-8")
    )
    assignments = {fold: {} for fold in range(5)}
    for row in payload["rows"]:
        assignments[int(row["outer_fold_id"])][str(row["global_participant_id"])] = int(
            row["inner_fold_id"]
        )
    return assignments


def _feature_groups(root: Path) -> dict[str, tuple[str, ...]]:
    payload = json.loads(
        (root / "context" / "feature_manifest.json").read_text(encoding="utf-8")
    )
    groups = {name: tuple(values) for name, values in payload["feature_groups"].items()}
    if {name: len(values) for name, values in groups.items()} != {
        "anchor_only": 54,
        "anchor_delta": 162,
        "anchor_delta_rolling": 432,
    }:
        raise RuntimeError("001B feature groups changed")
    return groups


def _fit_outer_components(
    outer_train: pd.DataFrame,
    outer_test: pd.DataFrame,
    selected: pd.DataFrame,
    assignments: dict[str, int],
    groups: dict[str, tuple[str, ...]],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    output = outer_test[
        [
            "global_participant_id",
            "target_window_id",
            "future_binary_target",
            "outer_fold_id",
        ]
    ].copy()
    audits: list[dict[str, Any]] = []
    auxiliary_cache: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for row in selected.to_dict(orient="records"):
        family = str(row["model_family"])
        feature_names = groups[str(row["feature_group"])]
        auxiliary_mode = str(row["auxiliary_mode"])
        train = outer_train
        test = outer_test
        model_features = feature_names
        if auxiliary_mode != "none":
            if auxiliary_mode not in auxiliary_cache:
                auxiliary_cache[auxiliary_mode] = strict_auxiliary_outer_views(
                    outer_train,
                    outer_test,
                    assignments,
                    groups["anchor_delta_rolling"],
                    auxiliary_mode,
                )
            train, test = auxiliary_cache[auxiliary_mode]
            model_features = feature_names + ("__auxiliary_prediction",)
        probability = fit_binary_candidate(
            family,
            train,
            test,
            model_features,
            json.loads(str(row["params_json"])),
        )
        output[f"p_{family}"] = probability
        audits.append(
            {
                "model_family": family,
                "candidate_id": str(row["candidate_id"]),
                "feature_group": str(row["feature_group"]),
                "feature_count": len(model_features),
                "auxiliary_mode": auxiliary_mode,
                "outer_train_row_count": len(outer_train),
                "outer_test_row_count": len(outer_test),
                "auxiliary_outer_train_policy": (
                    "strict_inner_fold_oof"
                    if auxiliary_mode != "none"
                    else "not_applicable"
                ),
                "auxiliary_outer_test_policy": (
                    "fit_complete_outer_train_only"
                    if auxiliary_mode != "none"
                    else "not_applicable"
                ),
            }
        )
    return output, audits


def _completed_fold(fold_root: Path, outer_test: pd.DataFrame) -> pd.DataFrame | None:
    scope_path = fold_root / "fusion_scope.json"
    prediction_path = fold_root / "outer_test_predictions.parquet"
    if not scope_path.is_file() or not prediction_path.is_file():
        return None
    scope = json.loads(scope_path.read_text(encoding="utf-8"))
    prediction = pd.read_parquet(prediction_path)
    if (
        scope.get("status") != "complete"
        or scope.get("outer_test_labels_used_for_selection") is not False
        or len(prediction) != len(outer_test)
        or set(prediction["target_window_id"]) != set(outer_test["target_window_id"])
    ):
        raise RuntimeError(f"invalid completed 001D fold: {fold_root}")
    return prediction


def _run_fold(
    root: Path,
    task_id: str,
    outer_fold: int,
    samples: pd.DataFrame,
    assignments: dict[str, int],
    groups: dict[str, tuple[str, ...]],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    fold_root = root / "fusion" / task_id / f"outer_fold_{outer_fold}"
    fold_root.mkdir(parents=True, exist_ok=True)
    outer_train = samples.loc[samples["outer_fold_id"].ne(outer_fold)].copy()
    outer_test = samples.loc[samples["outer_fold_id"].eq(outer_fold)].copy()
    completed = _completed_fold(fold_root, outer_test)
    if completed is not None:
        print(
            json.dumps(
                {
                    "event": "fusion_fold_resumed",
                    "task_id": task_id,
                    "outer_fold_id": outer_fold,
                }
            ),
            flush=True,
        )
        return completed, []

    selection_root = root / "selection" / task_id / f"outer_fold_{outer_fold}"
    search = pd.read_parquet(selection_root / "candidate_search.parquet")
    all_oof = pd.read_parquet(selection_root / "all_candidate_inner_oof.parquet")
    selected = select_family_components(search)
    components = component_oof_table(all_oof, selected)
    if set(components["global_participant_id"]).intersection(
        outer_test["global_participant_id"]
    ):
        raise RuntimeError("outer-test participant entered fusion selection")

    selected_weights, weight_search, blended_inner = search_blend_weights(
        components, float(config["weight_grid"]["step"])
    )
    if len(weight_search) != int(config["weight_grid"]["expected_count"]):
        raise RuntimeError("weight grid count changed")
    selected_method, calibration_search, calibration_oof = compare_calibrators(
        components, blended_inner
    )
    calibrator, threshold, threshold_audit = freeze_calibration_and_threshold(
        components,
        blended_inner,
        selected_method,
        calibration_oof["p_calibrated_selected"],
    )

    base_outer, model_audits = _fit_outer_components(
        outer_train, outer_test, selected, assignments, groups
    )
    outer_matrix = base_outer[[f"p_{name}" for name in COMPONENT_FAMILIES]].to_numpy(
        dtype="float64"
    )
    weight_vector = np.asarray(
        [selected_weights[name] for name in COMPONENT_FAMILIES], dtype="float64"
    )
    outer_blended = outer_matrix @ weight_vector
    outer_calibrated = calibrator.predict(outer_blended)
    prediction = base_outer.copy()
    prediction["p_blended"] = outer_blended
    prediction["p_calibrated"] = outer_calibrated
    prediction["threshold"] = threshold
    prediction["predicted_class"] = (outer_calibrated >= threshold).astype("int8")
    prediction["selected_calibration_method"] = selected_method

    selected.to_parquet(fold_root / "family_components.parquet", index=False)
    components.to_parquet(fold_root / "component_inner_oof.parquet", index=False)
    weight_search.to_parquet(fold_root / "weight_search.parquet", index=False)
    _write_json(
        fold_root / "selected_weights.json",
        {
            "weights": selected_weights,
            "sum": float(sum(selected_weights.values())),
            "selection_source": "outer_train_inner_oof_only",
            "selection_metric": config["weight_grid"]["selection_metric"],
            "tie_breaker": config["weight_grid"]["tie_breaker"],
        },
    )
    calibration_search.to_parquet(
        fold_root / "calibration_candidates.parquet", index=False
    )
    calibration_oof.to_parquet(
        fold_root / "calibration_crossfit_inner_oof.parquet", index=False
    )
    joblib.dump(calibrator, fold_root / "calibrator.joblib")
    _write_json(
        fold_root / "calibrator.json",
        {
            **calibrator.audit(),
            "fit_source": "complete_outer_train_inner_oof_blended_probability",
            "selection_source": "inner_fold_cross_fitted_calibration_predictions",
        },
    )
    _write_json(
        fold_root / "threshold.json",
        {
            **threshold_audit,
            "fit_source": "selected_cross_fitted_inner_oof_calibrated_probability",
        },
    )
    base_outer.to_parquet(
        fold_root / "base_model_outer_test_predictions.parquet", index=False
    )
    prediction.to_parquet(fold_root / "outer_test_predictions.parquet", index=False)
    _write_json(
        fold_root / "fusion_scope.json",
        {
            "schema_version": "mood-social-forecast-v3.4-opt-fusion-scope-v1",
            "status": "complete",
            "task_id": task_id,
            "outer_fold_id": outer_fold,
            "outer_train_participant_ids": sorted(
                outer_train["global_participant_id"].unique().tolist()
            ),
            "outer_test_participant_ids": sorted(
                outer_test["global_participant_id"].unique().tolist()
            ),
            "outer_test_target_window_ids": sorted(
                outer_test["target_window_id"].tolist()
            ),
            "outer_test_labels_used_for_component_selection": False,
            "outer_test_labels_used_for_weight_selection": False,
            "outer_test_labels_used_for_calibration_selection": False,
            "outer_test_labels_used_for_threshold_selection": False,
            "outer_test_labels_used_for_selection": False,
            "selected_component_ids": dict(
                zip(selected["model_family"], selected["candidate_id"], strict=True)
            ),
            "selected_weights": selected_weights,
            "selected_calibration_method": selected_method,
            "selected_threshold": threshold,
            "model_fit_audits": model_audits,
        },
    )
    print(
        json.dumps(
            {
                "event": "fusion_fold_complete",
                "task_id": task_id,
                "outer_fold_id": outer_fold,
                "weights": selected_weights,
                "calibrator": selected_method,
                "threshold": threshold,
            }
        ),
        flush=True,
    )
    return prediction, []


def run_fusion(run_id: str) -> dict[str, Any]:
    config, root = _load_config(run_id)
    manifest_path = root / "fusion_manifest.json"
    if manifest_path.exists():
        raise FileExistsError("FORECAST-OPT-001D already completed for this run")
    _append_attempt(root, "started", "Strict inner-OOF fusion and calibration started.")
    assignments = _inner_assignments()
    groups = _feature_groups(root)
    all_predictions: list[pd.DataFrame] = []
    failures: list[dict[str, Any]] = []
    for task_id in config["forecast_tasks"]:
        samples = pd.read_parquet(
            root / "context" / task_id / "samples_all_features.parquet"
        )
        task_predictions: list[pd.DataFrame] = []
        for outer_fold in range(int(config["outer_folds"])):
            try:
                prediction, fold_failures = _run_fold(
                    root,
                    task_id,
                    outer_fold,
                    samples,
                    assignments[outer_fold],
                    groups,
                    config,
                )
                task_predictions.append(prediction)
                failures.extend(fold_failures)
                _write_sums(root)
            except Exception as exc:
                failure = {
                    "task_id": task_id,
                    "outer_fold_id": outer_fold,
                    "stage": "fusion_fold",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                failures.append(failure)
                pd.DataFrame(failures).to_parquet(
                    root / "fusion" / "failed_folds.parquet", index=False
                )
                raise
        task_oof = pd.concat(task_predictions, ignore_index=True).sort_values(
            "target_window_id", kind="mergesort"
        )
        if (
            len(task_oof) != len(samples)
            or task_oof["target_window_id"].duplicated().any()
            or set(task_oof["target_window_id"]) != set(samples["target_window_id"])
        ):
            raise RuntimeError(f"strict outer OOF coverage failed: {task_id}")
        task_oof.insert(0, "task_id", task_id)
        task_oof.to_parquet(
            root / "fusion" / task_id / "outer_oof.parquet", index=False
        )
        all_predictions.append(task_oof)

    failure_columns = [
        "task_id",
        "outer_fold_id",
        "stage",
        "error_type",
        "error",
        "traceback",
    ]
    pd.DataFrame(failures, columns=failure_columns).to_parquet(
        root / "fusion" / "failed_folds.parquet", index=False
    )
    combined = pd.concat(all_predictions, ignore_index=True)
    combined.to_parquet(root / "fusion" / "outer_oof.parquet", index=False)
    manifest = {
        "schema_version": "mood-social-forecast-v3.4-opt-fusion-manifest-v1",
        "task_id": "FORECAST-OPT-001D",
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "fusion_completed",
        "config_sha256": sha256_file(CONFIG_PATH),
        "selection_manifest_sha256": sha256_file(root / "selection_manifest.json"),
        "training_code_sha256": sha256_file(Path(__file__).resolve()),
        "fusion_module_sha256": sha256_file(
            ALGORITHM_ROOT
            / "src"
            / "elderly_monitoring"
            / "modules"
            / "mental_health"
            / "mood_social"
            / "forecast_fusion_optimization.py"
        ),
        "task_row_counts": {
            task_id: int(len(frame))
            for task_id, frame in combined.groupby("task_id", sort=False)
        },
        "outer_test_used_for_selection": False,
        "strict_outer_oof_generated": True,
        "bootstrap_performed": False,
        "promotion_decision_performed": False,
        "product_integration_performed": False,
        "failed_fold_count": len(failures),
        "release_status": config["release_status"],
        "execution_mode": config["execution_mode"],
        "decision_authority": config["decision_authority"],
        "product_visible": config["product_visible"],
    }
    _write_json(manifest_path, manifest)
    _append_attempt(root, "completed", "All ten strict outer folds completed.")
    _write_sums(root)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    root = CANDIDATE_ROOT / args.run_id
    try:
        result = run_fusion(args.run_id)
    except Exception as exc:
        if root.is_dir():
            _append_attempt(root, "failed", f"{type(exc).__name__}: {exc}")
            _write_json(
                root / "fusion_failure.json",
                {
                    "run_id": args.run_id,
                    "stage": "FORECAST-OPT-001D",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            _write_sums(root)
        raise
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
