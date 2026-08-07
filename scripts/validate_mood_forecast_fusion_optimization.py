"""Independently validate FORECAST-OPT-001D fusion artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
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
    CALIBRATION_METHODS,
    COMPONENT_FAMILIES,
)


BASELINE_ROOT = (
    ALGORITHM_ROOT / "artifacts" / "mental_health" / "mood_social" / "forecast_v3.4"
)
CANDIDATE_ROOT = BASELINE_ROOT / "optimization_candidates"
CONFIG_PATH = (
    ALGORITHM_ROOT
    / "configs"
    / "experiments"
    / "mood_social_forecast_v3_4_opt_fusion.yaml"
)
TASK_COUNTS = {"forecast_1m": 9393, "forecast_2m": 9280}


@dataclass
class Checks:
    count: int = 0

    def require(self, condition: bool, message: str) -> None:
        if not condition:
            raise RuntimeError(message)
        self.count += 1


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _verify_sums(checks: Checks, root: Path) -> None:
    sums = root / "SHA256SUMS"
    listed: set[str] = set()
    for line in sums.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        checks.require(relative not in listed, f"duplicate SHA entry: {relative}")
        listed.add(relative)
        path = root / relative
        checks.require(path.is_file(), f"missing SHA entry: {relative}")
        checks.require(sha256_file(path) == digest, f"SHA mismatch: {relative}")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    checks.require(actual == listed, "SHA256SUMS does not cover complete run")


def _validate_fold(
    checks: Checks,
    root: Path,
    task_id: str,
    outer_fold: int,
    samples: pd.DataFrame,
) -> dict[str, Any]:
    fold_root = root / "fusion" / task_id / f"outer_fold_{outer_fold}"
    scope = _read_json(fold_root / "fusion_scope.json")
    outer_test = samples.loc[samples["outer_fold_id"].eq(outer_fold)]
    inner = pd.read_parquet(fold_root / "component_inner_oof.parquet")
    selected = pd.read_parquet(fold_root / "family_components.parquet")
    weights = _read_json(fold_root / "selected_weights.json")
    weight_search = pd.read_parquet(fold_root / "weight_search.parquet")
    calibration = pd.read_parquet(fold_root / "calibration_candidates.parquet")
    calibration_oof = pd.read_parquet(
        fold_root / "calibration_crossfit_inner_oof.parquet"
    )
    calibrator_meta = _read_json(fold_root / "calibrator.json")
    calibrator = joblib.load(fold_root / "calibrator.joblib")
    threshold = _read_json(fold_root / "threshold.json")
    prediction = pd.read_parquet(fold_root / "outer_test_predictions.parquet")
    base = pd.read_parquet(fold_root / "base_model_outer_test_predictions.parquet")

    checks.require(scope["status"] == "complete", "fold is not complete")
    for key in (
        "outer_test_labels_used_for_component_selection",
        "outer_test_labels_used_for_weight_selection",
        "outer_test_labels_used_for_calibration_selection",
        "outer_test_labels_used_for_threshold_selection",
        "outer_test_labels_used_for_selection",
    ):
        checks.require(scope[key] is False, f"outer label policy changed: {key}")
    checks.require(
        set(scope["outer_test_participant_ids"])
        == set(outer_test["global_participant_id"]),
        "outer-test participant scope changed",
    )
    checks.require(
        set(scope["outer_train_participant_ids"]).isdisjoint(
            scope["outer_test_participant_ids"]
        ),
        "outer train/test participants overlap",
    )
    checks.require(
        set(selected["model_family"]) == set(COMPONENT_FAMILIES),
        "components incomplete",
    )
    checks.require(
        len(selected) == 3 and selected["candidate_id"].is_unique,
        "component selection invalid",
    )
    checks.require(
        set(inner["global_participant_id"]).isdisjoint(
            outer_test["global_participant_id"]
        ),
        "outer-test participants entered inner OOF",
    )
    checks.require(
        not inner["outer_fold_id"].eq(outer_fold).any(),
        "outer-test rows entered inner OOF",
    )
    checks.require(
        inner["target_window_id"].is_unique, "component inner OOF duplicated"
    )
    checks.require(
        len(inner) == len(samples) - len(outer_test), "component inner OOF incomplete"
    )
    checks.require(len(weight_search) == 66, "weight grid size changed")
    checks.require(
        np.all(
            weight_search[[f"weight_{name}" for name in COMPONENT_FAMILIES]].to_numpy()
            >= -1e-12
        ),
        "negative fusion weight found",
    )
    checks.require(
        np.allclose(
            weight_search[[f"weight_{name}" for name in COMPONENT_FAMILIES]].sum(
                axis=1
            ),
            1.0,
        ),
        "fusion weights do not sum to one",
    )
    selected_weights = weights["weights"]
    checks.require(
        set(selected_weights) == set(COMPONENT_FAMILIES)
        and all(float(value) >= 0.0 for value in selected_weights.values())
        and np.isclose(float(sum(selected_weights.values())), 1.0),
        "selected fusion weights are invalid",
    )
    checks.require(
        set(calibration["method"]) == set(CALIBRATION_METHODS),
        "calibration grid incomplete",
    )
    checks.require(
        calibration_oof["p_calibrated_selected"].notna().all(),
        "calibration OOF incomplete",
    )
    checks.require(
        calibration_oof["target_window_id"].is_unique, "calibration OOF duplicated"
    )
    checks.require(
        calibrator_meta["method"] in CALIBRATION_METHODS, "calibrator method invalid"
    )
    checks.require(
        calibrator.method == calibrator_meta["method"], "calibrator metadata mismatch"
    )
    selected_method = str(scope["selected_calibration_method"])
    checks.require(
        selected_method == calibrator_meta["method"], "selected calibrator mismatch"
    )
    checks.require(
        0.0 <= float(scope["selected_threshold"]) <= 1.0, "threshold out of range"
    )
    checks.require(
        threshold["fit_source"].startswith("selected_cross_fitted_inner_oof"),
        "threshold fit source changed",
    )
    checks.require(
        len(base) == len(outer_test) == len(prediction),
        "outer prediction count changed",
    )
    checks.require(
        set(prediction["target_window_id"]) == set(outer_test["target_window_id"]),
        "outer OOF window coverage changed",
    )
    for column in [f"p_{name}" for name in COMPONENT_FAMILIES] + [
        "p_blended",
        "p_calibrated",
    ]:
        checks.require(
            prediction[column].between(0.0, 1.0).all(),
            f"probability out of range: {column}",
        )
    checks.require(
        prediction["selected_calibration_method"].eq(selected_method).all(),
        "outer prediction calibrator label changed",
    )
    return {
        "task_id": task_id,
        "outer_fold_id": outer_fold,
        "selected_weights": selected_weights,
        "selected_calibration_method": selected_method,
        "selected_threshold": float(scope["selected_threshold"]),
        "outer_test_row_count": len(prediction),
    }


def validate(run_id: str) -> dict[str, Any]:
    root = CANDIDATE_ROOT / run_id
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    checks = Checks()
    checks.require(config["task_id"] == "FORECAST-OPT-001D", "task identity changed")
    checks.require(
        config["outer_test_policy"].startswith("untouched"), "outer policy changed"
    )
    checks.require(config["product_visible"] is False, "product visibility changed")
    manifest = _read_json(root / "fusion_manifest.json")
    checks.require(manifest["status"] == "fusion_completed", "fusion status changed")
    checks.require(
        manifest["strict_outer_oof_generated"] is True, "strict outer OOF missing"
    )
    checks.require(
        manifest["outer_test_used_for_selection"] is False,
        "outer test used for selection",
    )
    checks.require(manifest["failed_fold_count"] == 0, "failed folds present")
    checks.require(
        manifest["bootstrap_performed"] is False, "bootstrap boundary changed"
    )
    checks.require(
        manifest["promotion_decision_performed"] is False,
        "promotion decision boundary changed",
    )
    checks.require(
        manifest["product_integration_performed"] is False,
        "product integration boundary changed",
    )
    checks.require(
        sha256_file(root / "selection_manifest.json")
        == config["upstream_selection_manifest_sha256"],
        "upstream selection hash changed",
    )
    _verify_sums(checks, root)
    fold_results = []
    for task_id, expected_count in TASK_COUNTS.items():
        samples = pd.read_parquet(
            root / "context" / task_id / "samples_all_features.parquet"
        )
        checks.require(
            len(samples) == expected_count, f"sample count changed: {task_id}"
        )
        outer_oof = pd.read_parquet(root / "fusion" / task_id / "outer_oof.parquet")
        checks.require(
            len(outer_oof) == expected_count, f"outer OOF count changed: {task_id}"
        )
        checks.require(
            outer_oof["target_window_id"].is_unique, f"outer OOF duplicated: {task_id}"
        )
        for outer_fold in range(5):
            fold_results.append(
                _validate_fold(checks, root, task_id, outer_fold, samples)
            )
    combined = pd.read_parquet(root / "fusion" / "outer_oof.parquet")
    checks.require(
        len(combined) == sum(TASK_COUNTS.values()), "combined OOF count changed"
    )
    return {
        "status": "pass",
        "run_id": run_id,
        "checks_passed": checks.count,
        "folds": fold_results,
        "release_status": manifest["release_status"],
        "execution_mode": manifest["execution_mode"],
        "decision_authority": manifest["decision_authority"],
        "product_visible": manifest["product_visible"],
        "validation_code_sha256": sha256_file(Path(__file__).resolve()),
    }


def _write_sums(root: Path) -> None:
    rows = [
        f"{sha256_file(path)}  {path.relative_to(root).as_posix()}"
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "SHA256SUMS"
    ]
    (root / "SHA256SUMS").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    root = CANDIDATE_ROOT / args.run_id
    result = validate(args.run_id)
    if args.write_report:
        _write_json(root / "fusion_validation_report.json", result)
        _write_sums(root)
        result = validate(args.run_id)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
