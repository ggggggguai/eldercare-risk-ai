"""Independently validate FORECAST-OPT-001F evaluation artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ALGORITHM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ALGORITHM_ROOT / "src"))

from elderly_monitoring.modules.mental_health.mood_social.forecast_experiment import (  # noqa: E402
    sha256_file,
)


CONFIG_PATH = (
    ALGORITHM_ROOT
    / "configs"
    / "experiments"
    / "mood_social_forecast_v3_4_evaluation.yaml"
)
ROOT = ALGORITHM_ROOT / "artifacts" / "mental_health" / "mood_social" / "forecast_v3.4"
UPSTREAM_ROOT = ROOT / "optimization_candidates" / "MH-20260807-FOPT-002"
EVALUATION_ROOT = UPSTREAM_ROOT / "evaluation"
TASKS = ("forecast_1m", "forecast_2m")
METRICS = ("auprc", "auroc", "macro_f1", "sensitivity", "specificity", "brier", "ece")


@dataclass
class Checks:
    count: int = 0

    def require(self, condition: bool, message: str) -> None:
        if not condition:
            raise RuntimeError(message)
        self.count += 1


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value


def _write_json(path: Path, payload: Any) -> None:
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
    (root / "SHA256SUMS").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _verify_sums(checks: Checks, root: Path) -> None:
    listed: set[str] = set()
    for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
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
    checks.require(actual == listed, "evaluation SHA256SUMS is incomplete")


def validate() -> dict[str, Any]:
    checks = Checks()
    manifest = _read_json(EVALUATION_ROOT / "evaluation_manifest.json")
    checks.require(manifest["task_id"] == "FORECAST-OPT-001F", "task identity changed")
    checks.require(
        manifest["status"] == "evaluation_completed", "evaluation incomplete"
    )
    checks.require(manifest["bootstrap_performed"] is True, "bootstrap missing")
    checks.require(manifest["bootstrap_repetitions"] == 2000, "bootstrap count changed")
    checks.require(
        manifest["common_window_count"] == 8494, "common window count changed"
    )
    checks.require(
        manifest["outer_test_used_for_selection"] is False,
        "outer labels used for selection",
    )
    checks.require(
        manifest["outer_test_used_for_evaluation"] is True, "evaluation scope changed"
    )
    checks.require(
        manifest["promotion_decision_performed"] is True, "promotion decision missing"
    )
    checks.require(
        manifest["product_integration_performed"] is False,
        "product integration enabled",
    )
    checks.require(manifest["product_visible"] is False, "product visibility changed")
    for key, expected in (
        ("release_status", "experimental"),
        ("execution_mode", "offline_only"),
        ("decision_authority", "shadow_only"),
        ("failure_count", 0),
    ):
        checks.require(
            manifest[key] == expected, f"permission or failure boundary changed: {key}"
        )
    checks.require(
        manifest["config_sha256"] == sha256_file(CONFIG_PATH),
        "evaluation config changed",
    )
    checks.require(
        manifest["evaluation_code_sha256"]
        == sha256_file(
            ALGORITHM_ROOT / "scripts" / "evaluate_mood_forecast_optimization.py"
        ),
        "evaluation code changed",
    )
    checks.require(
        manifest["fusion_oof_sha256"]
        == sha256_file(UPSTREAM_ROOT / "fusion" / "outer_oof.parquet"),
        "001D OOF changed",
    )
    checks.require(
        manifest["baseline_oof_sha256"]
        == sha256_file(ROOT / "oof_predictions.parquet"),
        "baseline OOF changed",
    )
    checks.require(
        manifest["conscious_proxy_sha256"]
        == sha256_file(
            UPSTREAM_ROOT
            / "external_auxiliary"
            / "MH-20260807-FOPT-AUX-001"
            / "conscious_psyche_proxy.parquet"
        ),
        "Conscious proxy changed",
    )
    _verify_sums(checks, EVALUATION_ROOT)

    metrics = pd.read_parquet(EVALUATION_ROOT / "task_metrics.parquet")
    checks.require(len(metrics) == 8, "task metrics row count changed")
    checks.require(set(metrics["task_id"]) == set(TASKS), "task metrics tasks changed")
    checks.require(
        set(metrics["scope"]) == {"full", "common"}, "task metrics scopes changed"
    )
    checks.require(
        set(metrics["variant"]) == {"baseline", "candidate"},
        "task metrics variants changed",
    )
    checks.require(
        np.isfinite(metrics[list(METRICS)]).all().all(),
        "task metrics contain non-finite values",
    )

    bootstrap = pd.read_parquet(EVALUATION_ROOT / "bootstrap_deltas.parquet")
    checks.require(len(bootstrap) == 8000, "bootstrap row count changed")
    checks.require(
        bootstrap.groupby(["task_id", "scope"], sort=False).size().eq(2000).all(),
        "bootstrap coverage changed",
    )
    checks.require(
        bootstrap["replicate"].between(0, 1999).all(),
        "bootstrap replicate range changed",
    )
    checks.require(
        np.isfinite(
            bootstrap.filter(regex="^(baseline|candidate|delta)_").to_numpy()
        ).all(),
        "bootstrap contains non-finite values",
    )

    fold = pd.read_parquet(EVALUATION_ROOT / "fold_stability.parquet")
    checks.require(len(fold) == 10, "fold stability row count changed")
    checks.require(
        fold.groupby("task_id").size().eq(5).all(), "fold stability coverage changed"
    )
    checks.require(
        set(fold["outer_fold_id"]) == set(range(5)), "outer fold coverage changed"
    )

    ablation = pd.read_parquet(EVALUATION_ROOT / "ablation_metrics.parquet")
    checks.require(len(ablation) == 10, "ablation row count changed")
    checks.require(
        ablation["eligible_for_promotion"].eq(False).all(), "ablation became promotable"
    )
    checks.require(
        np.isfinite(ablation[list(METRICS)]).all().all(),
        "ablation contains non-finite values",
    )

    proxy = _read_json(EVALUATION_ROOT / "conscious_proxy_metrics.json")
    checks.require(len(proxy["rows"]) == 2, "Conscious proxy task coverage changed")
    checks.require(
        all(row["eligible_for_promotion"] is False for row in proxy["rows"]),
        "proxy became promotable",
    )

    decision = _read_json(EVALUATION_ROOT / "promotion_decision.json")
    checks.require(
        decision["status"] == "retain_v3_4_baseline", "promotion decision changed"
    )
    checks.require(
        decision["product_integration_performed"] is False,
        "decision enabled integration",
    )
    for task in TASKS:
        item = decision["per_task"][task]
        expected = all(item["criteria"].values())
        checks.require(
            item["status"] == ("promote_candidate" if expected else "retain_baseline"),
            f"promotion criteria mismatch: {task}",
        )
    return {
        "status": "pass",
        "task_id": "FORECAST-OPT-001F",
        "run_id": manifest["run_id"],
        "checks_passed": checks.count,
        "promotion_status": decision["status"],
        "bootstrap_repetitions": manifest["bootstrap_repetitions"],
        "common_window_count": manifest["common_window_count"],
        "validation_code_sha256": sha256_file(Path(__file__).resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    result = validate()
    if args.write_report:
        _write_json(EVALUATION_ROOT / "validation_report.json", result)
        _write_sums(EVALUATION_ROOT)
        _write_sums(UPSTREAM_ROOT)
        result = validate()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
