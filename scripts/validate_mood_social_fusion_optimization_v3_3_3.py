"""Independently validate the OPT-FUSION-001 candidate and reports."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from elderly_monitoring.modules.mental_health.mood_social.fusion.optimization import (
    CALIBRATION_METHODS,
    RUN_ID,
    TASK_ID,
    V2_VARIANTS,
    FusionOptimizationError,
    load_optimization_config,
    load_optimization_inputs,
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
    parser.add_argument("--config", type=Path, default=Path("configs/training/mood_social_fusion_optimization_v3_3_3.yaml"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    path = args.config if args.config.is_absolute() else root / args.config
    checks = Checks()
    try:
        config = load_optimization_config(path, repository_root=root)
        table, assignments, protection = load_optimization_inputs(config)
        result = validate(config, table, assignments, protection, checks)
    except (FusionOptimizationError, ValidationFailure, OSError, KeyError, ValueError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps({"status": "pass", "checks": checks.count, **result}, ensure_ascii=False, sort_keys=True))
    return 0


def validate(config: Any, table: pd.DataFrame, assignments: pd.DataFrame, protection: dict[str, Any], checks: Checks) -> dict[str, Any]:
    report = config.report_directory
    required = (
        "calibration_comparison.parquet",
        "fusion_v2_candidates.parquet",
        "fusion_v2_oof_predictions.parquet",
        "source_robustness.parquet",
        "weak_branch_ablation.parquet",
        "sensitivity.json",
        "summary.json",
        "calibration_payload.json",
        "upstream_protection.json",
        "config.json",
        "run.json",
        "model_card.json",
        "artifacts.json",
    )
    for name in required:
        checks.require((report / name).is_file(), f"missing OPT-FUSION-001 artifact: {name}")
    checks.require(config.model_path.is_file(), "candidate model missing")
    checks.require(config.manifest_path.is_file(), "candidate manifest missing")
    checks.require(len(table) == 22191, "FUSION-001 row count changed")
    checks.require(len(assignments) == 15361, "DATA-007 participant count changed")

    calibration = pd.read_parquet(report / "calibration_comparison.parquet")
    checks.require(set(calibration["method"]) == set(CALIBRATION_METHODS), "calibration candidates changed")
    checks.require(set(calibration["weight_variant"]) == {"natural", "frozen_four_level"}, "calibration weight variants changed")
    candidates = pd.read_parquet(report / "fusion_v2_candidates.parquet")
    checks.require(set(candidates["variant"]) == set(V2_VARIANTS), "Fusion v2 candidates changed")
    checks.require(set(candidates["outer_fold"]) == set(range(5)), "Fusion v2 outer-fold coverage changed")
    oof = pd.read_parquet(report / "fusion_v2_oof_predictions.parquet")
    checks.require(len(oof) == 22188, "Fusion v2 OOF row count changed")
    checks.require(not oof[["dataset_id", "canonical_row_index"]].duplicated().any(), "Fusion v2 OOF identities duplicated")
    checks.require(oof["fusion_v2_probability"].between(0, 1).all(), "Fusion v2 OOF probability range changed")
    checks.require(set(oof["outer_fold"]) == set(range(5)), "Fusion v2 OOF fold coverage changed")
    ablation = pd.read_parquet(report / "weak_branch_ablation.parquet")
    checks.require(set(ablation["ablation"]) == {"none", "physiology", "social_context", "trend_social", "all_weak"}, "weak-branch ablation grid changed")
    source = pd.read_parquet(report / "source_robustness.parquet")
    checks.require(set(source["held_out_source"]).issubset(set(table["dataset_id"])), "unknown source in robustness report")

    sensitivity = _json(report / "sensitivity.json")
    checks.require(sensitivity["interpretation"] == "offline_sensitivity_only_no_production_prior_or_threshold", "sensitivity boundary changed")
    checks.require(all(row["label"] == "offline_sensitivity" for row in sensitivity["target_prior"]), "prior rows lost offline label")
    checks.require(all(row["label"] == "offline_sensitivity" for row in sensitivity["cost_work_points"]), "cost rows lost offline label")
    summary = _json(report / "summary.json")
    checks.require(summary["promotion_status"] == "review_required_art_candidate_not_production", "candidate promotion boundary changed")
    checks.require(summary["best_variant"] in V2_VARIANTS, "selected candidate variant invalid")

    model = joblib.load(config.model_path)
    model.validate()
    checks.require(model.task_id == TASK_ID and model.run_id == RUN_ID, "candidate model identity changed")
    manifest = _json(config.manifest_path)
    checks.require(manifest["strict_oof"] is True, "strict OOF flag changed")
    checks.require(manifest["offline_only"] is True, "offline-only flag changed")
    checks.require(manifest["production_boundary"]["select_threshold"] is False, "threshold boundary changed")
    checks.require(manifest["production_boundary"]["change_http_behavior"] is False, "HTTP boundary changed")
    checks.require(manifest["production_boundary"]["include_model006_predictions"] is False, "MODEL-006 entered fusion")
    checks.require(_sha256_file(config.model_path) == manifest["model_sha256"], "candidate model hash changed")
    checks.require(_sha256_file(report / "fusion_v2_oof_predictions.parquet") == manifest["oof_sha256"], "candidate OOF hash changed")
    checks.require(_json(report / "upstream_protection.json") == protection, "upstream protection changed")
    _validate_artifacts(report, checks)
    return {
        "task_id": TASK_ID,
        "run_id": RUN_ID,
        "selected_candidate_variant": summary["best_variant"],
        "model_sha256": _sha256_file(config.model_path),
        "manifest_sha256": _sha256_file(config.manifest_path),
        "report_core_sha256": _report_core_sha256(report),
    }


def _validate_artifacts(report: Path, checks: Checks) -> None:
    payload = _json(report / "artifacts.json")
    checks.require(isinstance(payload.get("artifacts"), list), "artifact list missing")
    for row in payload["artifacts"]:
        path = report / row["path"]
        checks.require(path.is_file(), f"artifact missing: {row['path']}")
        checks.require(path.stat().st_size == row["bytes"], f"artifact size changed: {row['path']}")
        checks.require(_sha256_file(path) == row["sha256"], f"artifact hash changed: {row['path']}")


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValidationFailure(f"JSON object expected: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _report_core_sha256(report: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((item for item in report.rglob("*") if item.is_file() and item.name not in {"run.json", "artifacts.json"}), key=lambda item: item.relative_to(report).as_posix()):
        digest.update(path.relative_to(report).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
