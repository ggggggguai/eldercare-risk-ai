"""Independently validate the corrected OPT-FUSION-003 artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from elderly_monitoring.modules.mental_health.mood_social.fusion.optimization_v3_3_4 import (
    RUN_ID,
    TASK_ID,
    FusionV334OptimizationError,
    load_v334_config,
    load_v334_inputs,
)
from elderly_monitoring.modules.mental_health.mood_social.fusion.calibration_optimization import (
    _ece,
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
    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs/training/mood_social_fusion_optimization_v3_3_4.yaml"
    checks = Checks()
    try:
        config = load_v334_config(config_path, repository_root=root)
        frame, assignments, protection = load_v334_inputs(config)
        result = validate(config, frame, assignments, protection, checks)
    except (
        FusionV334OptimizationError,
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
    frame: pd.DataFrame,
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
        "leave_one_source_stability.parquet",
        "device_combination_metrics.parquet",
        "missing_modality_metrics.parquet",
        "weak_branch_ablation.parquet",
        "promotion.json",
        "summary.json",
        "upstream_protection.json",
        "config.json",
        "run.json",
        "model_card.json",
        "model_card.md",
        "artifacts.json",
    )
    for name in required:
        checks.require((report / name).is_file(), f"missing fusion artifact: {name}")
    checks.require(config.model_path.is_file(), "fusion candidate model missing")
    checks.require(config.manifest_path.is_file(), "fusion candidate manifest missing")
    checks.require(len(frame) == 22188, "strict OOF row count changed")
    checks.require(len(assignments) == 15361, "participant assignment count changed")
    checks.require(
        protection["baseline_probability_column"] == "deployment_candidate_probability",
        "current online baseline column changed",
    )

    oof = pd.read_parquet(report / "oof_predictions.parquet")
    checks.require(len(oof) == len(frame), "published OOF row count changed")
    checks.require(
        not oof[["dataset_id", "canonical_row_index"]].duplicated().any(),
        "published OOF identities duplicated",
    )
    for column in ("baseline_probability", "deployment_candidate_probability"):
        values = oof[column].to_numpy(dtype=float)
        checks.require(
            np.isfinite(values).all(), f"{column} contains non-finite values"
        )
        checks.require(
            bool(np.all((values > 0.0) & (values < 1.0))),
            f"{column} is outside probability bounds",
        )

    target = frame["binary_target"].to_numpy(dtype=int)
    baseline = _metrics(target, oof["baseline_probability"].to_numpy(dtype=float))
    checks.require(
        abs(baseline["auprc"] - 0.2387837904) < 1e-10, "online baseline AUPRC changed"
    )
    checks.require(
        abs(baseline["auroc"] - 0.6519567616) < 1e-10, "online baseline AUROC changed"
    )
    checks.require(
        abs(baseline["brier"] - 0.2399408103) < 1e-10, "online baseline Brier changed"
    )

    overall = pd.read_parquet(report / "overall_metrics.parquet").set_index("model")
    checks.require(
        set(overall.index)
        == {
            "current_online_baseline",
            "corrected_calibration_reference",
            "v334_candidate",
        },
        "overall model scope changed",
    )
    for metric, value in baseline.items():
        checks.require(
            abs(float(overall.loc["current_online_baseline", metric]) - value) < 1e-12,
            f"published baseline {metric} changed",
        )
    checks.require(
        abs(
            float(overall.loc["current_online_baseline", "ece"])
            - _ece(target, oof["baseline_probability"].to_numpy(dtype=float))
        )
        < 1e-12,
        "published baseline ECE changed",
    )
    checks.require(
        np.isfinite(overall["ece"]).all(),
        "published ECE contains non-finite values",
    )

    outer = pd.read_parquet(report / "outer_fold_stability.parquet")
    checks.require(
        set(outer["outer_fold"]) == set(range(5)), "outer-fold coverage changed"
    )
    checks.require(
        int(outer["auprc_delta"].ge(0.0).sum()) == 1, "outer-fold gate evidence changed"
    )
    checks.require(
        not pd.read_parquet(report / "leave_one_source_stability.parquet").empty,
        "source report is empty",
    )
    checks.require(
        not pd.read_parquet(report / "device_combination_metrics.parquet").empty,
        "device report is empty",
    )
    checks.require(
        not pd.read_parquet(report / "missing_modality_metrics.parquet").empty,
        "missing-modality report is empty",
    )
    ablation = pd.read_parquet(report / "weak_branch_ablation.parquet")
    checks.require(
        set(ablation["removed_branch"])
        == {"physiology", "social_context", "trend_social"},
        "weak-branch ablation scope changed",
    )

    promotion = _json(report / "promotion.json")
    checks.require(
        promotion["promotion_passed"] is False, "failed candidate was promoted"
    )
    checks.require(
        promotion["checks"]
        == {
            "leave_one_source_stability": False,
            "natural_auprc_delta": False,
            "natural_auroc_non_degraded": False,
            "natural_brier_within_tolerance": True,
            "outer_fold_stability": False,
        },
        "promotion gate result changed",
    )
    summary = _json(report / "summary.json")
    checks.require(
        summary["task_id"] == TASK_ID and summary["run_id"] == RUN_ID,
        "summary identity changed",
    )
    checks.require(
        summary["package_default"] == "MH-20260802-013", "fallback package changed"
    )

    manifest = _json(config.manifest_path)
    checks.require(manifest["strict_oof"] is True, "strict OOF flag changed")
    checks.require(
        manifest["uses_public_dataset_id"] is False, "dataset_id entered inference"
    )
    checks.require(manifest["promotion_passed"] is False, "manifest promotion changed")
    checks.require(
        _sha256_file(config.model_path) == manifest["model_sha256"],
        "model hash changed",
    )
    checks.require(
        _json(report / "upstream_protection.json") == protection,
        "upstream protection changed",
    )
    model = joblib.load(config.model_path)
    model.validate()
    prediction = model.predict_frame(frame.iloc[:128])
    checks.require(
        prediction["available"].all(), "loaded model lost available predictions"
    )
    checks.require(
        np.isfinite(prediction["fusion_probability"]).all(),
        "loaded model prediction is invalid",
    )
    _validate_artifacts(report, checks)
    _validate_checksums(config.model_path.parent, checks)
    return {
        "task_id": TASK_ID,
        "run_id": RUN_ID,
        "selected_candidate_id": summary["selected_candidate_id"],
        "promotion_passed": False,
        "model_sha256": _sha256_file(config.model_path),
        "manifest_sha256": _sha256_file(config.manifest_path),
        "report_core_sha256": manifest["report_core_sha256"],
    }


def _metrics(target: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    return {
        "auprc": float(average_precision_score(target, probability)),
        "auroc": float(roc_auc_score(target, probability)),
        "brier": float(brier_score_loss(target, probability)),
    }


def _validate_artifacts(report: Path, checks: Checks) -> None:
    payload = _json(report / "artifacts.json")
    for row in payload["artifacts"]:
        path = report / row["path"]
        checks.require(path.is_file(), f"artifact missing: {row['path']}")
        checks.require(
            path.stat().st_size == row["bytes"], f"artifact size changed: {row['path']}"
        )
        checks.require(
            _sha256_file(path) == row["sha256"], f"artifact hash changed: {row['path']}"
        )


def _validate_checksums(directory: Path, checks: Checks) -> None:
    path = directory / "SHA256SUMS"
    checks.require(path.is_file(), "SHA256SUMS missing")
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", maxsplit=1)
        checks.require(
            _sha256_file(directory / name) == digest, f"checksum changed: {name}"
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


if __name__ == "__main__":
    raise SystemExit(main())
