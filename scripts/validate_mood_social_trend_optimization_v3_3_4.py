"""Independently validate OPT-TREND-001 candidate artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from elderly_monitoring.modules.mental_health.mood_social.trend_optimization import (
    BRANCHES,
    CANDIDATES,
    RUN_ID,
    TASK_ID,
    TrendOptimizationError,
    load_trend_optimization_config,
    load_trend_optimization_inputs,
)


class ValidationFailure(RuntimeError):
    pass


class Checks:
    def __init__(self) -> None:
        self.count = 0

    def require(self, condition: bool, message: str) -> None:
        self.count += 1
        if not condition:
            raise ValidationFailure(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training/mood_social_trend_optimization_v3_3_4.yaml"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    path = args.config if args.config.is_absolute() else root / args.config
    checks = Checks()
    try:
        config = load_trend_optimization_config(path, repository_root=root)
        upstream, assignments, protection = load_trend_optimization_inputs(config)
        result = validate(config, upstream, assignments, protection, checks)
    except (
        TrendOptimizationError,
        ValidationFailure,
        OSError,
        KeyError,
        ValueError,
    ) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 1
    print(
        json.dumps(
            {"status": "pass", "checks": checks.count, **result},
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def validate(
    config: Any,
    upstream: pd.DataFrame,
    assignments: pd.DataFrame,
    protection: dict[str, Any],
    checks: Checks,
) -> dict[str, Any]:
    report = config.report_directory
    required = (
        "candidate_search.parquet",
        "oof_predictions.parquet",
        "overall_metrics.parquet",
        "outer_fold_stability.parquet",
        "playback_stability.parquet",
        "branch_reliability.parquet",
        "summary.json",
        "upstream_protection.json",
        "config.json",
        "run.json",
        "model_card.json",
        "model_card.md",
        "artifacts.json",
    )
    for name in required:
        checks.require((report / name).is_file(), f"missing trend artifact: {name}")
    checks.require(config.model_path.is_file(), "trend candidate model missing")
    checks.require(config.manifest_path.is_file(), "trend candidate manifest missing")
    oof = pd.read_parquet(report / "oof_predictions.parquet")
    checks.require(len(oof) == len(upstream), "trend OOF row count changed")
    checks.require(set(oof["branch"]) == set(BRANCHES), "trend branch scope changed")
    checks.require(
        set(oof.loc[oof["available"], "selected_candidate"]).issubset(set(CANDIDATES)),
        "trend selected candidate changed",
    )
    checks.require(
        oof.loc[oof["available"], "selected_probability"].notna().all(),
        "available trend OOF is incomplete",
    )
    checks.require(
        oof.loc[~oof["available"], "selected_probability"].isna().all(),
        "unavailable trend row received probability",
    )
    search = pd.read_parquet(report / "candidate_search.parquet")
    checks.require(
        set(search["candidate_id"]) == set(CANDIDATES), "trend candidate grid changed"
    )
    checks.require(
        set(search["outer_fold"]) == set(range(5)), "trend outer fold coverage changed"
    )
    overall = pd.read_parquet(report / "overall_metrics.parquet")
    checks.require(
        set(overall["branch"]) == set(BRANCHES), "trend overall branch scope changed"
    )
    checks.require(
        np.isfinite(overall[["auprc", "auroc", "brier"]]).all().all(),
        "trend metrics contain non-finite values",
    )
    playback = pd.read_parquet(report / "playback_stability.parquet")
    checks.require(
        playback["smoothing_non_increasing_jitter"].all(),
        "three-day smoothing increased playback jitter",
    )
    reliability = pd.read_parquet(report / "branch_reliability.parquet")
    social = reliability[reliability["branch"].eq("social")].iloc[0]
    checks.require(
        float(social["maximum_optimized_reliability"]) <= 0.25 + 1e-12,
        "social reliability cap changed",
    )
    summary = _json(report / "summary.json")
    checks.require(
        summary["task_id"] == TASK_ID and summary["run_id"] == RUN_ID,
        "trend summary identity changed",
    )
    checks.require(
        summary["synthetic_calls_as_supervision"] is False,
        "synthetic calls entered trend supervision",
    )
    checks.require(
        summary["social_evidence_scope"]
        == "engineering_proxy_no_direct_s10_phq9_validation",
        "social trend scope changed",
    )
    manifest = _json(config.manifest_path)
    checks.require(
        manifest["strict_oof"] is True and manifest["production"] is False,
        "trend manifest boundary changed",
    )
    checks.require(
        manifest["dataset_id_as_model_input"] is False, "dataset_id entered trend model"
    )
    checks.require(
        manifest["synthetic_calls_as_supervision"] is False,
        "synthetic calls entered model",
    )
    checks.require(
        _sha256_file(config.model_path) == manifest["model_sha256"],
        "trend model hash changed",
    )
    checks.require(
        _json(report / "upstream_protection.json") == protection,
        "trend upstream protection changed",
    )
    models = joblib.load(config.model_path)
    checks.require(
        set(models) == set(BRANCHES), "serialized trend branch scope changed"
    )
    _validate_artifacts(report, checks)
    _validate_checksums(config.model_path.parent, checks)
    return {
        "task_id": TASK_ID,
        "run_id": RUN_ID,
        "selected_candidates": summary["selected_candidates"],
        "model_sha256": _sha256_file(config.model_path),
        "manifest_sha256": _sha256_file(config.manifest_path),
    }


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValidationFailure(f"JSON object expected: {path}")
    return value


def _validate_artifacts(report: Path, checks: Checks) -> None:
    payload = _json(report / "artifacts.json")
    checks.require(isinstance(payload.get("artifacts"), list), "artifact list missing")
    for item in payload["artifacts"]:
        path = report / item["path"]
        checks.require(path.is_file(), f"artifact missing: {item['path']}")
        checks.require(
            path.stat().st_size == item["bytes"],
            f"artifact size changed: {item['path']}",
        )
        checks.require(
            _sha256_file(path) == item["sha256"],
            f"artifact hash changed: {item['path']}",
        )


def _validate_checksums(directory: Path, checks: Checks) -> None:
    path = directory / "SHA256SUMS"
    checks.require(path.is_file(), "SHA256SUMS missing")
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", maxsplit=1)
        checks.require(
            _sha256_file(directory / name) == digest, f"trend checksum changed: {name}"
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
