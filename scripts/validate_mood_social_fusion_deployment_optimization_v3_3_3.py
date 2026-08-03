"""Independently validate OPT-FUSION-002 artifacts and promotion evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from elderly_monitoring.modules.mental_health.mood_social.fusion.deployment_optimization import (
    CANDIDATE_IDS,
    RUN_ID,
    TASK_ID,
    DeploymentOptimizationError,
    load_deployment_optimization_config,
    load_deployment_optimization_inputs,
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
        default=Path(
            "configs/training/mood_social_fusion_deployment_optimization_v3_3_3.yaml"
        ),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    path = args.config if args.config.is_absolute() else root / args.config
    checks = Checks()
    try:
        config = load_deployment_optimization_config(path, repository_root=root)
        frame, _, protection = load_deployment_optimization_inputs(config)
        result = validate(config, frame, protection, checks)
    except (
        DeploymentOptimizationError,
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
    config: Any, frame: pd.DataFrame, protection: dict[str, Any], checks: Checks
) -> dict[str, Any]:
    report = config.report_directory
    required = (
        "candidate_search.parquet",
        "oof_predictions.parquet",
        "overall_metrics.parquet",
        "outer_fold_stability.parquet",
        "leave_one_source_stability.parquet",
        "promotion.json",
        "upstream_protection.json",
        "config.json",
        "run.json",
        "model_card.json",
        "model_card.md",
        "artifacts.json",
    )
    for name in required:
        checks.require(
            (report / name).is_file(), f"missing OPT-FUSION-002 artifact: {name}"
        )
    checks.require(config.model_path.is_file(), "deployment candidate model missing")
    checks.require(
        config.manifest_path.is_file(), "deployment candidate manifest missing"
    )
    checks.require(len(frame) == 22188, "available strict OOF row count changed")

    search = pd.read_parquet(report / "candidate_search.parquet")
    outer_search = search[search["outer_fold"].ge(0)]
    checks.require(
        set(outer_search["outer_fold"]) == set(range(5)), "outer search folds changed"
    )
    checks.require(
        set(outer_search["candidate_id"]) == set(CANDIDATE_IDS),
        "candidate search grid changed",
    )
    checks.require(
        len(outer_search) == 5 * len(CANDIDATE_IDS),
        "candidate search row count changed",
    )
    checks.require(
        outer_search.groupby("outer_fold")["selected"].sum().eq(1).all(),
        "outer selection is not unique",
    )

    oof = pd.read_parquet(report / "oof_predictions.parquet")
    checks.require(len(oof) == 22188, "deployment OOF row count changed")
    checks.require(
        not oof[["dataset_id", "canonical_row_index"]].duplicated().any(),
        "deployment OOF identities duplicated",
    )
    checks.require(
        oof["deployment_candidate_probability"].between(0, 1).all(),
        "deployment OOF probability range changed",
    )
    checks.require(
        set(oof["outer_fold"]) == set(range(5)), "deployment OOF outer coverage changed"
    )
    checks.require(
        set(oof["selected_candidate_id"]).issubset(set(CANDIDATE_IDS)),
        "unknown selected candidate",
    )

    overall = pd.read_parquet(report / "overall_metrics.parquet").set_index("model")
    checks.require(
        set(overall.index)
        == {
            "fusion_002_baseline",
            "opt_fusion_001_offline_reference",
            "deployment_candidate",
        },
        "overall model set changed",
    )
    target = oof["binary_target"].to_numpy(dtype=int)
    for model_name, column in (
        ("fusion_002_baseline", "baseline_probability"),
        ("opt_fusion_001_offline_reference", "offline_reference_probability"),
        ("deployment_candidate", "deployment_candidate_probability"),
    ):
        probability = oof[column].to_numpy(dtype=float)
        auprc = _average_precision(target, probability)
        brier = float(((probability - target) ** 2).mean())
        checks.require(
            abs(float(overall.loc[model_name, "auprc"]) - auprc) <= 1e-12,
            f"{model_name} AUPRC changed",
        )
        checks.require(
            abs(float(overall.loc[model_name, "brier"]) - brier) <= 1e-9,
            f"{model_name} Brier changed",
        )

    outer = pd.read_parquet(report / "outer_fold_stability.parquet")
    checks.require(
        len(outer) == 5 and set(outer["outer_fold"]) == set(range(5)),
        "outer stability changed",
    )
    source = pd.read_parquet(report / "leave_one_source_stability.parquet")
    checks.require(
        set(source["outer_fold"]).issubset(set(range(5))), "source outer folds changed"
    )
    checks.require(
        set(source["held_out_source"]).issubset(set(frame["dataset_id"])),
        "source report contains unknown source",
    )

    promotion = _json(report / "promotion.json")
    expected_checks = {
        "natural_auprc_delta",
        "natural_auroc_non_degraded",
        "natural_brier_within_tolerance",
        "outer_fold_stability",
        "leave_one_source_stability",
    }
    checks.require(
        set(promotion["checks"]) == expected_checks, "promotion checks changed"
    )
    checks.require(
        promotion["promotion_passed"] is all(promotion["checks"].values()),
        "promotion result does not match checks",
    )
    expected_default = (
        "deployment_candidate"
        if promotion["promotion_passed"]
        else "fusion_002_baseline"
    )
    checks.require(
        promotion["art001_default"] == expected_default, "ART-001 default changed"
    )

    model = joblib.load(config.model_path)
    model.validate()
    checks.require(model.candidate_id in CANDIDATE_IDS, "model candidate ID changed")
    checks.require(
        not model.coverage_calibrators
        or all("dataset" not in key.lower() for key in model.coverage_calibrators),
        "dataset calibration entered model",
    )
    manifest = _json(config.manifest_path)
    checks.require(
        manifest["task_id"] == TASK_ID and manifest["run_id"] == RUN_ID,
        "manifest identity changed",
    )
    checks.require(manifest["strict_oof"] is True, "strict OOF flag changed")
    checks.require(
        manifest["uses_public_dataset_id"] is False, "dataset ID boundary changed"
    )
    checks.require(
        manifest["promotion_passed"] is promotion["promotion_passed"],
        "manifest promotion result changed",
    )
    checks.require(
        manifest["art001_default"] == expected_default, "manifest ART default changed"
    )
    checks.require(
        _sha256_file(config.model_path) == manifest["model_sha256"],
        "model hash changed",
    )
    checks.require(
        _sha256_file(report / "oof_predictions.parquet") == manifest["oof_sha256"],
        "OOF hash changed",
    )
    checks.require(
        _report_core_sha256(report) == manifest["report_core_sha256"],
        "report core hash changed",
    )
    checks.require(
        _json(report / "upstream_protection.json") == protection,
        "upstream protection changed",
    )
    checks.require(
        manifest["production_boundary"]["select_production_threshold"] is False,
        "threshold boundary changed",
    )
    checks.require(
        manifest["production_boundary"]["include_model006_predictions"] is False,
        "MODEL-006 entered fusion",
    )
    checks.require(
        manifest["production_boundary"]["change_http_behavior"] is False,
        "HTTP behavior changed",
    )
    _validate_artifacts(report, checks)
    return {
        "task_id": TASK_ID,
        "run_id": RUN_ID,
        "selected_candidate_id": model.candidate_id,
        "promotion_passed": promotion["promotion_passed"],
        "art001_default": promotion["art001_default"],
        "model_sha256": _sha256_file(config.model_path),
        "manifest_sha256": _sha256_file(config.manifest_path),
        "report_core_sha256": _report_core_sha256(report),
    }


def _average_precision(target: Any, probability: Any) -> float:
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(target, probability))


def _validate_artifacts(report: Path, checks: Checks) -> None:
    payload = _json(report / "artifacts.json")
    checks.require(isinstance(payload.get("artifacts"), list), "artifact list missing")
    for row in payload["artifacts"]:
        path = report / row["path"]
        checks.require(path.is_file(), f"artifact missing: {row['path']}")
        checks.require(
            path.stat().st_size == row["bytes"], f"artifact size changed: {row['path']}"
        )
        checks.require(
            _sha256_file(path) == row["sha256"], f"artifact hash changed: {row['path']}"
        )


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
    for path in sorted(
        (
            item
            for item in report.iterdir()
            if item.is_file() and item.name not in {"run.json", "artifacts.json"}
        ),
        key=lambda item: item.name.encode("utf-8"),
    ):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
