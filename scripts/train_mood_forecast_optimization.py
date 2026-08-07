"""Select V3.4 Forecast optimization candidates with strict inner OOF only."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ALGORITHM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ALGORITHM_ROOT / "src"))

from elderly_monitoring.modules.mental_health.mood_social.forecast_experiment import (  # noqa: E402
    sha256_file,
)
from elderly_monitoring.modules.mental_health.mood_social.forecast_optimization import (  # noqa: E402
    FEATURE_GROUPS,
    OPTIMIZATION_VERSION,
    SelectionCandidate,
    evaluate_inner_candidate,
    strict_auxiliary_inner_views,
)


CONFIG_ROOT = ALGORITHM_ROOT / "configs" / "experiments"
CONFIG_PATH = CONFIG_ROOT / "mood_social_forecast_v3_4_opt.yaml"
SEARCH_PATH = CONFIG_ROOT / "mood_social_forecast_v3_4_opt_search_space.json"
ENVIRONMENT_PATH = CONFIG_ROOT / "mood_social_forecast_v3_4_opt_environment.lock"
BASELINE_ROOT = (
    ALGORITHM_ROOT / "artifacts" / "mental_health" / "mood_social" / "forecast_v3.4"
)
CANDIDATE_ROOT = BASELINE_ROOT / "optimization_candidates"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_sums(root: Path) -> None:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            rows.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    (root / "SHA256SUMS").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _append_attempt(root: Path, status: str, detail: str) -> None:
    path = root / "selection_attempts.json"
    payload = (
        json.loads(path.read_text(encoding="utf-8"))
        if path.exists()
        else {
            "schema_version": "mood-social-forecast-v3.4-opt-selection-attempts-v1",
            "run_id": root.name,
            "attempts": [],
        }
    )
    payload["attempts"].append(
        {
            "event_index": len(payload["attempts"]) + 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "detail": detail,
        }
    )
    _write_json(path, payload)


def _verify_environment() -> dict[str, Any]:
    lock = json.loads(ENVIRONMENT_PATH.read_text(encoding="utf-8"))
    actual_python = ".".join(str(value) for value in sys.version_info[:3])
    if actual_python != lock["python"]:
        raise RuntimeError(
            f"selection requires Python {lock['python']}, got {actual_python}"
        )
    distributions = {
        "numpy": "numpy",
        "pandas": "pandas",
        "pyarrow": "pyarrow",
        "scikit-learn": "scikit-learn",
        "lightgbm": "lightgbm",
        "catboost": "catboost",
        "PyYAML": "PyYAML",
        "pytest": "pytest",
    }
    mismatches = []
    actual_packages = {}
    for key, distribution in distributions.items():
        actual = importlib.metadata.version(distribution)
        actual_packages[key] = actual
        expected = str(lock["packages"][key])
        if actual != expected:
            mismatches.append(f"{key}={actual} expected={expected}")
    if mismatches:
        raise RuntimeError("environment mismatch: " + ", ".join(mismatches))
    return {"python": actual_python, "packages": actual_packages}


def _verify_candidate_context(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("optimization context manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "context_built":
        raise RuntimeError("optimization context is not ready")
    if manifest.get("optimization_version") != OPTIMIZATION_VERSION:
        raise RuntimeError("optimization context version mismatch")
    frozen = {
        "config_sha256": sha256_file(CONFIG_PATH),
        "search_space_sha256": sha256_file(SEARCH_PATH),
        "environment_lock_sha256": sha256_file(ENVIRONMENT_PATH),
        "inner_fold_manifest_sha256": config["inner_fold_manifest_sha256"],
        "baseline_context_manifest_sha256": config["baseline_context_manifest_sha256"],
    }
    for key, expected in frozen.items():
        if manifest.get(key) != expected:
            raise RuntimeError(f"optimization context frozen hash changed: {key}")
    sums = root / "SHA256SUMS"
    for line in sums.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        if relative != "manifest.json" and not relative.startswith("context/"):
            continue
        if sha256_file(root / relative) != digest:
            raise RuntimeError(f"candidate context hash mismatch: {relative}")
    return manifest


def _current_baseline_snapshot() -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for path in sorted(BASELINE_ROOT.rglob("*")):
        if not path.is_file() or CANDIDATE_ROOT in path.parents:
            continue
        snapshot[path.relative_to(BASELINE_ROOT).as_posix()] = sha256_file(path)
    return snapshot


def _verify_baseline(root: Path) -> None:
    expected = json.loads(
        (root / "context" / "baseline_snapshot.json").read_text(encoding="utf-8")
    )
    actual = _current_baseline_snapshot()
    if actual != expected:
        changed = sorted(
            set(actual).symmetric_difference(expected)
            | {
                name
                for name in set(actual).intersection(expected)
                if actual[name] != expected[name]
            }
        )
        raise RuntimeError(f"immutable V3.4 baseline changed: {changed[:10]}")


def _inner_assignments(config: dict[str, Any]) -> dict[int, dict[str, int]]:
    path = ALGORITHM_ROOT / config["inner_fold_manifest"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = {outer_fold: {} for outer_fold in range(5)}
    for row in payload["rows"]:
        result[int(row["outer_fold_id"])][str(row["global_participant_id"])] = int(
            row["inner_fold_id"]
        )
    return result


def _candidates(
    search: dict[str, Any], feature_groups: tuple[str, ...]
) -> list[SelectionCandidate]:
    candidates: list[SelectionCandidate] = []
    for feature_group in feature_groups:
        for family in ("elasticnet_logistic", "lightgbm", "catboost"):
            settings = search[family]
            for params in settings:
                candidates.append(
                    SelectionCandidate(feature_group, family, dict(params), "none")
                )
            for auxiliary_mode in ("phq9_score", "phq9_category"):
                candidates.append(
                    SelectionCandidate(
                        feature_group,
                        family,
                        dict(settings[0]),
                        auxiliary_mode,
                    )
                )
    if len(candidates) != 36 or len({item.candidate_id for item in candidates}) != 36:
        raise RuntimeError("frozen candidate matrix must contain 36 unique candidates")
    return candidates


def _feature_groups(root: Path) -> dict[str, tuple[str, ...]]:
    payload = json.loads(
        (root / "context" / "feature_manifest.json").read_text(encoding="utf-8")
    )
    result = {name: tuple(payload["feature_groups"][name]) for name in FEATURE_GROUPS}
    if {name: len(values) for name, values in result.items()} != {
        "anchor_only": 54,
        "anchor_delta": 162,
        "anchor_delta_rolling": 432,
    }:
        raise RuntimeError("history feature group counts changed")
    return result


def _selection_record(
    candidate: SelectionCandidate,
    candidate_index: int,
    metrics: dict[str, float] | None,
    error: Exception | None,
) -> dict[str, Any]:
    return {
        "candidate_index": candidate_index,
        "candidate_id": candidate.candidate_id,
        "feature_group": candidate.feature_group,
        "model_family": candidate.model_family,
        "auxiliary_mode": candidate.auxiliary_mode,
        "params_json": json.dumps(
            dict(candidate.params), sort_keys=True, separators=(",", ":")
        ),
        "auprc": metrics["auprc"] if metrics else None,
        "brier": metrics["brier"] if metrics else None,
        "status": "passed" if error is None else "failed",
        "error_type": type(error).__name__ if error else None,
        "error": str(error) if error else None,
    }


def _precompute_auxiliary_cache(
    frame: pd.DataFrame,
    assignments: dict[str, int],
    feature_names: tuple[str, ...],
) -> dict[tuple[tuple[str, ...], str, int], tuple[pd.Series, pd.Series]]:
    cache: dict[tuple[tuple[str, ...], str, int], tuple[pd.Series, pd.Series]] = {}

    def build_one(
        mode: str, inner_fold: int
    ) -> tuple[tuple[tuple[str, ...], str, int], tuple[pd.Series, pd.Series]]:
        train, validation = strict_auxiliary_inner_views(
            frame, assignments, feature_names, mode, inner_fold
        )
        return (
            (feature_names, mode, inner_fold),
            (
                train["__auxiliary_prediction"].copy(),
                validation["__auxiliary_prediction"].copy(),
            ),
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(build_one, mode, inner_fold)
            for mode in ("phq9_score", "phq9_category")
            for inner_fold in range(5)
        ]
        for future in as_completed(futures):
            key, value = future.result()
            cache[key] = value
    if len(cache) != 10:
        raise RuntimeError("auxiliary cache is incomplete")
    return cache


def _run_candidate(
    candidate_index: int,
    candidate: SelectionCandidate,
    outer_train: pd.DataFrame,
    assignment: dict[str, int],
    groups: dict[str, tuple[str, ...]],
    auxiliary_cache: dict[
        tuple[tuple[str, ...], str, int], tuple[pd.Series, pd.Series]
    ],
) -> tuple[int, pd.DataFrame | None, dict[str, float] | None, Exception | None]:
    try:
        predictions, metrics = evaluate_inner_candidate(
            outer_train,
            assignment,
            groups[candidate.feature_group],
            candidate,
            auxiliary_cache,
            groups["anchor_delta_rolling"],
        )
        return candidate_index, predictions, metrics, None
    except Exception as exc:  # Returned to the ordered audit loop.
        return candidate_index, None, None, exc


def _better(record: dict[str, Any], best: dict[str, Any] | None) -> bool:
    if record["status"] != "passed":
        return False
    if best is None:
        return True
    return bool(
        record["auprc"] > best["auprc"] + 1e-12
        or (
            abs(record["auprc"] - best["auprc"]) <= 1e-12
            and record["brier"] < best["brier"] - 1e-12
        )
    )


def _load_completed_fold(
    fold_root: Path,
    task_id: str,
    outer_fold: int,
    outer_train: pd.DataFrame,
    outer_test_scope: pd.DataFrame,
    candidates: list[SelectionCandidate],
) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    required = {
        "selection": fold_root / "selection.json",
        "search": fold_root / "candidate_search.parquet",
        "all_oof": fold_root / "all_candidate_inner_oof.parquet",
        "selected_oof": fold_root / "selected_inner_oof.parquet",
    }
    if not all(path.is_file() for path in required.values()):
        return None
    scope = json.loads(required["selection"].read_text(encoding="utf-8"))
    search = pd.read_parquet(required["search"])
    all_oof = pd.read_parquet(required["all_oof"])
    selected_oof = pd.read_parquet(required["selected_oof"])
    candidate_ids = {candidate.candidate_id for candidate in candidates}
    if (
        scope.get("task_id") != task_id
        or int(scope.get("outer_fold_id", -1)) != outer_fold
        or scope.get("candidate_count") != len(candidates)
        or scope.get("outer_test_labels_read_for_selection") is not False
        or len(search) != len(candidates)
        or set(search["candidate_id"]) != candidate_ids
    ):
        raise RuntimeError(
            f"completed fold metadata is invalid: {task_id}/{outer_fold}"
        )
    passed_ids = set(search.loc[search["status"].eq("passed"), "candidate_id"])
    if set(all_oof["candidate_id"]) != passed_ids:
        raise RuntimeError(
            f"completed fold OOF candidate set changed: {task_id}/{outer_fold}"
        )
    counts = all_oof.groupby("candidate_id")["target_window_id"].size()
    if not counts.eq(len(outer_train)).all():
        raise RuntimeError(
            f"completed fold OOF coverage changed: {task_id}/{outer_fold}"
        )
    selected = scope["selected"]
    if (
        selected["candidate_id"] not in passed_ids
        or len(selected_oof) != len(outer_train)
        or set(selected_oof["target_window_id"]) != set(outer_train["target_window_id"])
        or set(scope["selection_participant_ids"])
        != set(outer_train["global_participant_id"])
        or set(scope["untouched_outer_test_participant_ids"])
        != set(outer_test_scope["global_participant_id"])
    ):
        raise RuntimeError(
            f"completed fold selection scope changed: {task_id}/{outer_fold}"
        )
    failures = search.loc[search["status"].eq("failed")].to_dict(orient="records")
    return scope, failures


def run_selection(run_id: str) -> dict[str, Any]:
    root = CANDIDATE_ROOT / run_id
    if not root.is_dir():
        raise FileNotFoundError(f"candidate context does not exist: {run_id}")
    if (root / "selection_manifest.json").exists():
        raise FileExistsError("selection already completed for this run")
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    environment = _verify_environment()
    context_manifest = _verify_candidate_context(root, config)
    _verify_baseline(root)
    _append_attempt(
        root,
        "started",
        "HistoryPreprocessor reuses explicit masks; auxiliary OOF uses the full 432-feature history cache shared by all candidates.",
    )
    inner_by_outer = _inner_assignments(config)
    groups = _feature_groups(root)
    search = json.loads(SEARCH_PATH.read_text(encoding="utf-8"))
    candidates = _candidates(search, FEATURE_GROUPS)
    selections: dict[str, Any] = {}
    failure_rows: list[dict[str, Any]] = []
    for task_id in config["task_ids"]:
        samples = pd.read_parquet(
            root / "context" / task_id / "samples_all_features.parquet"
        )
        task_selections = []
        for outer_fold in range(5):
            fold_root = root / "selection" / task_id / f"outer_fold_{outer_fold}"
            fold_root.mkdir(parents=True, exist_ok=True)
            outer_train = samples.loc[samples["outer_fold_id"].ne(outer_fold)].copy()
            outer_test_scope = samples.loc[
                samples["outer_fold_id"].eq(outer_fold),
                ["global_participant_id", "target_window_id"],
            ]
            assignment = inner_by_outer[outer_fold]
            train_ids = set(outer_train["global_participant_id"])
            if not train_ids.issubset(assignment):
                raise RuntimeError("inner manifest does not cover outer-train")
            completed = _load_completed_fold(
                fold_root,
                task_id,
                outer_fold,
                outer_train,
                outer_test_scope,
                candidates,
            )
            if completed is not None:
                scope, completed_failures = completed
                task_selections.append(scope)
                failure_rows.extend(
                    {
                        "task_id": task_id,
                        "outer_fold_id": outer_fold,
                        **record,
                    }
                    for record in completed_failures
                )
                print(
                    json.dumps(
                        {
                            "event": "selection_fold_resumed",
                            "task_id": task_id,
                            "outer_fold_id": outer_fold,
                            "selected": scope["selected"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                continue
            auxiliary_cache = _precompute_auxiliary_cache(
                outer_train,
                assignment,
                groups["anchor_delta_rolling"],
            )
            records: list[dict[str, Any]] = []
            all_predictions: list[pd.DataFrame] = []
            best_record: dict[str, Any] | None = None
            best_predictions: pd.DataFrame | None = None
            candidate_results = {}
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [
                    executor.submit(
                        _run_candidate,
                        candidate_index,
                        candidate,
                        outer_train,
                        assignment,
                        groups,
                        auxiliary_cache,
                    )
                    for candidate_index, candidate in enumerate(candidates)
                ]
                for future in as_completed(futures):
                    candidate_index, predictions, metrics, error = future.result()
                    candidate_results[candidate_index] = (predictions, metrics, error)
            for candidate_index, candidate in enumerate(candidates):
                predictions, metrics, error = candidate_results[candidate_index]
                if predictions is not None and metrics is not None:
                    predictions["candidate_index"] = candidate_index
                    predictions["candidate_id"] = candidate.candidate_id
                    predictions["feature_group"] = candidate.feature_group
                    predictions["model_family"] = candidate.model_family
                    predictions["auxiliary_mode"] = candidate.auxiliary_mode
                    all_predictions.append(predictions)
                record = _selection_record(candidate, candidate_index, metrics, error)
                records.append(record)
                if error is not None:
                    failure_rows.append(
                        {
                            "task_id": task_id,
                            "outer_fold_id": outer_fold,
                            **record,
                        }
                    )
                if _better(record, best_record):
                    best_record = record
                    best_predictions = predictions.copy()
            if best_record is None or best_predictions is None:
                raise RuntimeError(
                    f"no candidate passed for {task_id}/fold {outer_fold}"
                )
            records_frame = pd.DataFrame(records)
            records_frame.to_parquet(
                fold_root / "candidate_search.parquet", index=False
            )
            pd.concat(all_predictions, ignore_index=True).to_parquet(
                fold_root / "all_candidate_inner_oof.parquet", index=False
            )
            best_predictions.to_parquet(
                fold_root / "selected_inner_oof.parquet", index=False
            )
            scope = {
                "schema_version": "mood-social-forecast-v3.4-opt-selection-scope-v1",
                "task_id": task_id,
                "outer_fold_id": outer_fold,
                "selection_participant_ids": sorted(train_ids),
                "selection_target_window_ids": sorted(
                    outer_train["target_window_id"].tolist()
                ),
                "untouched_outer_test_participant_ids": sorted(
                    outer_test_scope["global_participant_id"].unique().tolist()
                ),
                "untouched_outer_test_target_window_ids": sorted(
                    outer_test_scope["target_window_id"].tolist()
                ),
                "outer_test_labels_read_for_selection": False,
                "candidate_count": len(records),
                "passed_candidate_count": int(
                    records_frame["status"].eq("passed").sum()
                ),
                "failed_candidate_count": int(
                    records_frame["status"].eq("failed").sum()
                ),
                "selected": best_record,
                "auxiliary_policy": "binary-fit rows use nested OOF auxiliary predictions; inner-validation rows use auxiliary models fitted only on binary-fit rows",
            }
            _write_json(fold_root / "selection.json", scope)
            task_selections.append(scope)
            _write_sums(root)
            print(
                json.dumps(
                    {
                        "event": "selection_fold_complete",
                        "task_id": task_id,
                        "outer_fold_id": outer_fold,
                        "selected": best_record,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        selections[task_id] = task_selections
    failures = pd.DataFrame(failure_rows)
    failures.to_parquet(root / "selection" / "failed_candidates.parquet", index=False)
    _verify_baseline(root)
    manifest = {
        "schema_version": "mood-social-forecast-v3.4-opt-selection-manifest-v1",
        "run_id": run_id,
        "optimization_version": OPTIMIZATION_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "selection_completed",
        "context_manifest_sha256": sha256_file(root / "manifest.json"),
        "config_sha256": sha256_file(CONFIG_PATH),
        "search_space_sha256": sha256_file(SEARCH_PATH),
        "environment_lock_sha256": sha256_file(ENVIRONMENT_PATH),
        "training_code_sha256": sha256_file(Path(__file__).resolve()),
        "optimization_module_sha256": sha256_file(
            ALGORITHM_ROOT
            / "src"
            / "elderly_monitoring"
            / "modules"
            / "mental_health"
            / "mood_social"
            / "forecast_optimization.py"
        ),
        "environment": environment,
        "candidate_count_per_outer_fold": len(candidates),
        "outer_test_used_for_selection": False,
        "final_fusion_performed": False,
        "final_calibration_performed": False,
        "outer_oof_generated": False,
        "failed_candidate_count": len(failure_rows),
        "selections": selections,
        "baseline_file_count": context_manifest["baseline_file_count"],
        "baseline_snapshot_verified_after_selection": True,
        "release_status": "experimental",
        "execution_mode": "offline_only",
        "decision_authority": "shadow_only",
        "product_visible": False,
    }
    _write_json(root / "selection_manifest.json", manifest)
    _append_attempt(root, "completed", "All ten task/outer-fold selections passed.")
    _write_sums(root)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    root = CANDIDATE_ROOT / args.run_id
    try:
        run_selection(args.run_id)
    except Exception as exc:
        if root.is_dir():
            _append_attempt(root, "failed", f"{type(exc).__name__}: {exc}")
            _write_json(
                root / "selection_failure.json",
                {
                    "run_id": args.run_id,
                    "stage": "FORECAST-OPT-001C",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            _write_sums(root)
        raise
    print(json.dumps({"status": "pass", "run_id": args.run_id}, ensure_ascii=False))


if __name__ == "__main__":
    main()
