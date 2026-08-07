"""Independently validate OPT-EXPERT-001 candidate artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from elderly_monitoring.modules.mental_health.mood_social.experts.optimization import (
    CANDIDATES,
    EXPERTS,
    RUN_ID,
    TASK_ID,
    ExpertOptimizationError,
    load_expert_optimization_config,
    load_expert_optimization_inputs,
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
        default=Path("configs/training/mood_social_expert_optimization_v3_3_4.yaml"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    path = args.config if args.config.is_absolute() else root / args.config
    checks = Checks()
    try:
        config = load_expert_optimization_config(path, repository_root=root)
        predictions, frames, assignments, protections = load_expert_optimization_inputs(
            config
        )
        result = validate(config, predictions, frames, assignments, protections, checks)
    except (
        ExpertOptimizationError,
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
    predictions: dict[str, pd.DataFrame],
    frames: Any,
    assignments: pd.DataFrame,
    protections: dict[str, Any],
    checks: Checks,
) -> dict[str, Any]:
    report = config.report_directory
    for name in (
        "candidate_search.parquet",
        "oof_predictions.parquet",
        "overall_metrics.parquet",
        "outer_fold_stability.parquet",
        "source_robustness.parquet",
        "ablation.parquet",
        "summary.json",
        "upstream_protection.json",
        "config.json",
        "run.json",
        "model_card.json",
        "model_card.md",
        "artifacts.json",
    ):
        checks.require(
            (report / name).is_file(), f"missing expert optimization artifact: {name}"
        )
    checks.require(config.model_path.is_file(), "expert candidate model missing")
    checks.require(config.manifest_path.is_file(), "expert candidate manifest missing")
    oof = pd.read_parquet(report / "oof_predictions.parquet")
    checks.require(set(oof["expert"]) == set(EXPERTS), "expert OOF scope changed")
    checks.require(
        set(oof["selected_candidate"]).issubset(set(CANDIDATES) | {""}),
        "selected candidate grid changed",
    )
    search = pd.read_parquet(report / "candidate_search.parquet")
    checks.require(
        set(search["candidate_id"]) == set(CANDIDATES), "candidate search grid changed"
    )
    checks.require(set(search["expert"]) == set(EXPERTS), "search expert scope changed")
    checks.require(
        set(search["status"]).issubset({"pass", "unavailable", "failed"}),
        "failure status changed",
    )
    overall = pd.read_parquet(report / "overall_metrics.parquet")
    checks.require(
        set(overall["expert"]) == set(EXPERTS), "overall expert scope changed"
    )
    checks.require(
        np.isfinite(overall[["auprc", "auroc", "brier"]].dropna()).all().all(),
        "overall metrics contain non-finite values",
    )
    outer = pd.read_parquet(report / "outer_fold_stability.parquet")
    checks.require(
        set(outer["outer_fold"]) == set(range(5)), "outer-fold coverage changed"
    )
    source = pd.read_parquet(report / "source_robustness.parquet")
    checks.require(
        set(source["expert"]) == set(EXPERTS), "source robustness scope changed"
    )
    ablation = pd.read_parquet(report / "ablation.parquet")
    checks.require(
        set(ablation["ablation"])
        == {"base_calibrated", "temporal_engineered_candidate"},
        "ablation scope changed",
    )
    for expert, frame in predictions.items():
        checks.require(
            len(frame) == len(oof[oof["expert"].eq(expert)]),
            f"{expert} OOF row count changed",
        )
        checks.require(
            frame["expert_mask"].isin([0, 1]).all(), f"{expert} mask is invalid"
        )
    summary = _json(report / "summary.json")
    checks.require(
        summary["task_id"] == TASK_ID and summary["run_id"] == RUN_ID,
        "summary identity changed",
    )
    checks.require(
        set(summary["selected_candidates"]) == set(EXPERTS),
        "selected expert scope changed",
    )
    checks.require(
        summary["promotion_status"] == "report_only_candidates_not_production",
        "production boundary changed",
    )
    checks.require(
        isinstance(summary["failures"], list), "failure runs were not retained"
    )
    manifest = _json(config.manifest_path)
    checks.require(
        manifest["strict_oof"] is True and manifest["production"] is False,
        "manifest boundary changed",
    )
    checks.require(
        manifest["dataset_id_as_model_input"] is False, "dataset_id entered model"
    )
    checks.require(
        manifest["feature_masks_as_risk_inputs"] is False, "feature masks entered model"
    )
    checks.require(
        manifest["feature_coverage_as_risk_input"] is False, "coverage entered model"
    )
    checks.require(
        _sha256_file(config.model_path) == manifest["model_sha256"],
        "candidate model hash changed",
    )
    checks.require(
        _json(report / "upstream_protection.json") == protections,
        "upstream protection changed",
    )
    models = joblib.load(config.model_path)
    checks.require(set(models) == set(EXPERTS), "serialized expert scope changed")
    for expert, model in models.items():
        # The model contract is checked by the persisted feature names and finite estimator output.
        checks.require(
            model.feature_names
            and len(model.feature_names) == model.imputer.statistics_.shape[0],
            f"{expert} feature contract invalid",
        )
        checks.require(
            model.calibrator.classes_.tolist() == [0, 1], f"{expert} calibrator invalid"
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
            _sha256_file(directory / name) == digest, f"model checksum changed: {name}"
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
