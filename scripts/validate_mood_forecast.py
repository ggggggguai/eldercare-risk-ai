"""Independently validate V3.4 forecast artifacts and V3.3 isolation."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ALGORITHM_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ALGORITHM_ROOT.parents[1]
sys.path.insert(0, str(ALGORITHM_ROOT / "src"))

from elderly_monitoring.modules.mental_health.mood_social.forecast_experiment import (  # noqa: E402
    sha256_file,
)


OUTPUT_ROOT = (
    ALGORITHM_ROOT / "artifacts" / "mental_health" / "mood_social" / "forecast_v3.4"
)
TASK_COUNTS = {"forecast_1m": 9393, "forecast_2m": 9280}
MODEL_FAMILIES = ("dummy", "elasticnet_logistic", "lightgbm")


class Checks:
    def __init__(self) -> None:
        self.count = 0

    def require(self, condition: bool, message: str) -> None:
        self.count += 1
        if not condition:
            raise AssertionError(message)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"JSON root is not an object: {path}")
    return value


def _verify_sums(checks: Checks, root: Path, sums_path: Path) -> None:
    seen = set()
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        checks.require(relative not in seen, f"duplicate checksum entry: {relative}")
        seen.add(relative)
        path = root / Path(relative)
        checks.require(path.is_file(), f"checksummed artifact is missing: {relative}")
        checks.require(
            sha256_file(path) == digest, f"artifact hash mismatch: {relative}"
        )


def _verify_v33(checks: Checks) -> dict[str, Any]:
    acceptance_path = (
        ALGORITHM_ROOT
        / "models"
        / "mental_health"
        / "mood_social"
        / "v3.3.3"
        / "acceptance.json"
    )
    acceptance = _read_json(acceptance_path)
    checks.require(
        acceptance.get("status") == "passed", "V3.3 acceptance is not passed"
    )
    checks.require(
        acceptance.get("model_version") == "mood-fusion-v3.3.3",
        "online model version changed",
    )
    for path_key, hash_key in (
        ("v3_3_config_path", "v3_3_config_sha256"),
        ("split_manifest_path", "split_manifest_sha256"),
        ("package_manifest_path", "package_manifest_sha256"),
        ("package_checksums_path", "package_checksums_sha256"),
        ("test_summary_path", "test_summary_sha256"),
    ):
        path = ALGORITHM_ROOT / acceptance[path_key]
        checks.require(path.is_file(), f"V3.3 frozen file missing: {path_key}")
        checks.require(
            sha256_file(path) == acceptance[hash_key],
            f"V3.3 frozen hash changed: {path_key}",
        )
    package_root = ALGORITHM_ROOT / acceptance["package_path"]
    _verify_sums(
        checks, package_root, ALGORITHM_ROOT / acceptance["package_checksums_path"]
    )
    forecast_files = [
        path
        for path in (
            ALGORITHM_ROOT / "models" / "mental_health" / "mood_social" / "v3.3.3"
        ).rglob("*")
        if path.is_file() and "forecast" in path.name.lower()
    ]
    checks.require(
        not forecast_files, "forecast artifact entered the V3.3 production model tree"
    )
    return acceptance


def _verify_backend_isolation(checks: Checks) -> None:
    backend = WORKSPACE_ROOT / "backend"
    tokens = ("forecast_1m", "forecast_2m", "mood_forecast", "forecast_v3.4")
    hits = []
    for path in backend.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {
            ".py",
            ".json",
            ".yaml",
            ".yml",
            ".md",
        }:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeError:
            continue
        if any(token in text for token in tokens):
            hits.append(str(path.relative_to(backend)))
    checks.require(not hits, f"backend contains forecast experiment logic: {hits}")
    pipeline = (
        ALGORITHM_ROOT
        / "src"
        / "elderly_monitoring"
        / "modules"
        / "mental_health"
        / "mood_social"
        / "pipeline.py"
    ).read_text(encoding="utf-8")
    checks.require(
        not any(token in pipeline for token in tokens),
        "MoodSocialPipeline references forecast artifacts",
    )


def validate() -> dict[str, Any]:
    checks = Checks()
    acceptance = _verify_v33(checks)
    _verify_backend_isolation(checks)
    required = [
        "oof_predictions.parquet",
        "training_manifest.json",
        "metrics.json",
        "experiment_manifest.json",
        "SHA256SUMS",
        "V3.4结果报告.md",
        "feature_contributions.parquet",
        "feature_missingness.parquet",
        "failure_samples.parquet",
        "calibration_curves.parquet",
        "fold_metrics.parquet",
    ]
    for name in required:
        checks.require((OUTPUT_ROOT / name).is_file(), f"missing V3.4 artifact: {name}")
    oof = pd.read_parquet(OUTPUT_ROOT / "oof_predictions.parquet")
    checks.require(
        len(oof) == 3 * sum(TASK_COUNTS.values()), "combined OOF row count changed"
    )
    for task, expected in TASK_COUNTS.items():
        samples = pd.read_parquet(OUTPUT_ROOT / task / "samples.parquet")
        checks.require(len(samples) == expected, f"{task} sample count changed")
        for family in MODEL_FAMILIES:
            selected = oof[oof.task_id.eq(task) & oof.model_family.eq(family)]
            checks.require(
                len(selected) == expected, f"{task}/{family} OOF coverage changed"
            )
            checks.require(
                selected.target_window_id.nunique() == expected,
                f"{task}/{family} OOF windows duplicate",
            )
            checks.require(
                np.isclose(selected.evaluation_weight.sum(), 1.0),
                f"{task}/{family} evaluation weight sum changed",
            )
            for fold in range(5):
                bundle_path = (
                    OUTPUT_ROOT / task / family / "folds" / f"outer_fold_{fold}.joblib"
                )
                search_path = (
                    OUTPUT_ROOT / task / family / "search" / f"outer_fold_{fold}.json"
                )
                inner_path = (
                    OUTPUT_ROOT
                    / task
                    / family
                    / "folds"
                    / f"outer_fold_{fold}_inner_oof.parquet"
                )
                threshold_path = (
                    OUTPUT_ROOT
                    / task
                    / family
                    / "folds"
                    / f"outer_fold_{fold}_threshold.json"
                )
                scope_path = (
                    OUTPUT_ROOT
                    / task
                    / family
                    / "folds"
                    / f"outer_fold_{fold}_scope.json"
                )
                checks.require(
                    bundle_path.is_file()
                    and search_path.is_file()
                    and inner_path.is_file()
                    and threshold_path.is_file()
                    and scope_path.is_file(),
                    f"{task}/{family}/fold{fold} artifact missing",
                )
                bundle = joblib.load(bundle_path)
                search = _read_json(search_path)
                train_ids = set(bundle["train_participant_ids"])
                test_ids = set(bundle["test_participant_ids"])
                calibration_ids = set(bundle["inner_calibration_participant_ids"])
                checks.require(
                    train_ids.isdisjoint(test_ids),
                    f"{task}/{family}/fold{fold} participant leakage",
                )
                checks.require(
                    calibration_ids == train_ids,
                    f"{task}/{family}/fold{fold} calibration scope changed",
                )
                checks.require(
                    bundle["raw_feature_names"]
                    == json.loads(
                        (OUTPUT_ROOT / "raw_feature_names.json").read_text(
                            encoding="utf-8"
                        )
                    )["raw_feature_names"],
                    f"{task}/{family}/fold{fold} feature order changed",
                )
                expected_candidates = {
                    "dummy": 1,
                    "elasticnet_logistic": 20,
                    "lightgbm": 64,
                }[family]
                checks.require(
                    search["candidate_count"] == expected_candidates,
                    f"{task}/{family}/fold{fold} search count changed",
                )
                if family == "dummy":
                    checks.require(
                        bundle["calibrator"] is None, "Dummy fitted a calibrator"
                    )
                    fold_rows = selected[selected.outer_fold_id.eq(fold)]
                    checks.require(
                        np.array_equal(
                            fold_rows.p_raw.to_numpy(),
                            fold_rows.p_calibrated.to_numpy(),
                        ),
                        "Dummy calibrated probabilities changed",
                    )
                    checks.require(
                        bundle["train_sample_weight_policy"]
                        == "participant_equal_prior_no_class_balance",
                        "Dummy training policy changed",
                    )
                else:
                    checks.require(
                        bundle["calibrator"] is not None, f"{family} calibrator missing"
                    )
    common = pd.read_parquet(OUTPUT_ROOT / "common_target_windows.parquet")
    checks.require(len(common) == 8494, "paired common-window count changed")
    metrics = _read_json(OUTPUT_ROOT / "metrics.json")
    checks.require(
        metrics.get("bootstrap_repetitions") == 2000,
        "bootstrap repetition count changed",
    )
    for task in TASK_COUNTS:
        for family in MODEL_FAMILIES:
            result = metrics["tasks"][task]["models"][family]
            checks.require(
                result["bootstrap"]["valid_repetitions"] == 2000,
                f"{task}/{family} bootstrap incomplete",
            )
            checks.require(
                set(result["primary_metrics"])
                >= {
                    "auprc",
                    "auroc",
                    "macro_f1",
                    "sensitivity",
                    "specificity",
                    "brier",
                    "ece",
                },
                f"{task}/{family} metrics incomplete",
            )
    for family in MODEL_FAMILIES:
        paired = metrics["paired_comparisons"][family]
        checks.require(
            paired["common_window_count"] == 8494
            and paired["valid_repetitions"] == 2000,
            f"{family} paired bootstrap incomplete",
        )
    experiment = _read_json(OUTPUT_ROOT / "experiment_manifest.json")
    for key, value in {
        "release_status": "experimental",
        "execution_mode": "offline_only",
        "decision_authority": "shadow_only",
        "product_visible": False,
    }.items():
        checks.require(
            experiment.get(key) == value, f"experiment boundary changed: {key}"
        )
    _verify_sums(checks, OUTPUT_ROOT, OUTPUT_ROOT / "SHA256SUMS")
    report = (OUTPUT_ROOT / "V3.4结果报告.md").read_text(encoding="utf-8")
    prohibited = ("预测抑郁症", "老人患病概率", "已经进入家属端", "固定天数预测")
    checks.require(
        not any(text in report for text in prohibited),
        "result report contains a prohibited claim",
    )
    return {
        "status": "pass",
        "check_count": checks.count,
        "online_model_version": acceptance["model_version"],
        "forecast_execution_mode": "offline_only",
    }


def main() -> None:
    result = validate()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
