"""Independently validate OPT-CALIB-001 artifacts and calibration stages."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from elderly_monitoring.modules.mental_health.mood_social.fusion.calibration_optimization import (
    AUDIT_VARIANTS,
    RUN_ID,
    SELECTABLE_VARIANTS,
    TASK_ID,
    CalibrationOptimizationError,
    load_calibration_config,
    load_calibration_inputs,
)


class ValidationFailure(RuntimeError):
    """Raised when an independent artifact check fails."""


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
            "configs/training/mood_social_calibration_optimization_v3_3_4.yaml"
        ),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    path = args.config if args.config.is_absolute() else root / args.config
    checks = Checks()
    try:
        config = load_calibration_config(path, repository_root=root)
        frame, assignments, protection = load_calibration_inputs(config)
        result = validate(config, frame, assignments, protection, checks)
    except (
        CalibrationOptimizationError,
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
        "threshold_curve.parquet",
        "calibration_curve.parquet",
        "summary.json",
        "upstream_protection.json",
        "config.json",
        "run.json",
        "model_card.json",
        "model_card.md",
        "artifacts.json",
    )
    for name in required:
        checks.require(
            (report / name).is_file(), f"missing calibration artifact: {name}"
        )
    checks.require(config.model_path.is_file(), "calibration candidate model missing")
    checks.require(config.manifest_path.is_file(), "calibration manifest missing")
    checks.require(len(frame) == 22188, "strict calibration OOF row count changed")
    checks.require(len(assignments) == 15361, "DATA-007 participant count changed")

    search = pd.read_parquet(report / "candidate_search.parquet")
    checks.require(
        set(search.loc[search["outer_fold"].eq(-1), "variant_id"])
        == set(SELECTABLE_VARIANTS),
        "aggregate selectable candidate grid changed",
    )
    checks.require(
        not set(AUDIT_VARIANTS)
        & set(search.loc[search["outer_fold"].eq(-1), "variant_id"]),
        "audit-only legacy mismatch entered selection",
    )
    oof = pd.read_parquet(report / "oof_predictions.parquet")
    checks.require(len(oof) == len(frame), "calibration OOF row count changed")
    checks.require(
        not oof[["dataset_id", "canonical_row_index"]].duplicated().any(),
        "calibration OOF identities duplicated",
    )
    for variant in (*SELECTABLE_VARIANTS, *AUDIT_VARIANTS):
        values = oof[f"probability__{variant}"].to_numpy(dtype=float)
        checks.require(np.isfinite(values).all(), f"{variant} OOF is incomplete")
        checks.require(
            bool(np.all((values > 0.0) & (values < 1.0))),
            f"{variant} OOF is outside probability bounds",
        )

    overall = pd.read_parquet(report / "overall_metrics.parquet")
    checks.require(
        set(overall["model"])
        == {
            "current_online_baseline",
            "fusion_002_raw",
            *SELECTABLE_VARIANTS,
            *AUDIT_VARIANTS,
        },
        "overall metric model grid changed",
    )
    checks.require(
        np.isfinite(overall[["auprc", "auroc", "brier", "ece"]]).all().all(),
        "overall metrics contain non-finite values",
    )
    outer = pd.read_parquet(report / "outer_fold_stability.parquet")
    checks.require(set(outer["outer_fold"]) == set(range(5)), "outer folds missing")
    checks.require(
        set(outer["model"]) == set(overall["model"]),
        "outer-fold model grid changed",
    )
    threshold = pd.read_parquet(report / "threshold_curve.parquet")
    checks.require(not threshold.empty, "threshold curve is empty")
    calibration = pd.read_parquet(report / "calibration_curve.parquet")
    checks.require(len(calibration) == 10, "calibration curve bin count changed")

    summary = _json(report / "summary.json")
    checks.require(summary["task_id"] == TASK_ID, "summary task identity changed")
    checks.require(summary["run_id"] == RUN_ID, "summary run identity changed")
    checks.require(
        summary["selected_variant"] in SELECTABLE_VARIANTS,
        "selected calibration candidate is not selectable",
    )
    checks.require(
        summary["promotion_status"] == "report_only_candidate_not_production",
        "offline candidate boundary changed",
    )
    checks.require(
        summary["baseline_model"]
        == (
            "MH-20260802-012 deployment_candidate "
            "(aggregate package coverage_calibrated)"
        ),
        "current online baseline identity changed",
    )

    model = joblib.load(config.model_path)
    model.validate()
    checks.require(
        model.task_id == TASK_ID and model.run_id == RUN_ID, "model identity changed"
    )
    checks.require(
        model.variant_id == summary["selected_variant"], "selected model changed"
    )
    if model.coverage_calibrators:
        checks.require(
            model.fit_input_stage == "global",
            "selected coverage calibrator does not fit and infer on p_global",
        )
    sample = frame.iloc[: min(128, len(frame))]
    prediction = model.predict_frame(sample)
    checks.require(
        np.isfinite(prediction["final_probability"]).all(),
        "loaded candidate does not predict finite probabilities",
    )

    manifest = _json(config.manifest_path)
    checks.require(manifest["strict_oof"] is True, "strict OOF flag changed")
    checks.require(manifest["production"] is False, "candidate became production")
    checks.require(
        manifest["public_dataset_id_as_model_input"] is False,
        "dataset_id entered calibration inference",
    )
    checks.require(
        manifest["model006_online"] is False, "MODEL-006 entered online scope"
    )
    checks.require(
        _sha256_file(config.model_path) == manifest["model_sha256"],
        "candidate model hash changed",
    )
    checks.require(
        _json(report / "upstream_protection.json") == protection,
        "protected upstream inputs changed",
    )
    _validate_nested_assignments(frame, assignments, checks)
    _validate_artifacts(report, checks)
    _validate_sha256sums(config.model_path.parent, checks)
    return {
        "task_id": TASK_ID,
        "run_id": RUN_ID,
        "selected_variant": summary["selected_variant"],
        "model_sha256": _sha256_file(config.model_path),
        "manifest_sha256": _sha256_file(config.manifest_path),
        "report_core_sha256": _report_core_sha256(report),
    }


def _validate_nested_assignments(
    frame: pd.DataFrame,
    assignments: pd.DataFrame,
    checks: Checks,
) -> None:
    for row in frame[["global_participant_id", "outer_fold"]].itertuples(index=False):
        mapping = assignments.loc[
            str(row.global_participant_id), "inner_validation_fold_by_outer_fold"
        ]
        checks.require(
            mapping.get(str(int(row.outer_fold))) is None,
            "outer-test participant has an inner validation assignment",
        )
        inner = [
            value for key, value in mapping.items() if int(key) != int(row.outer_fold)
        ]
        checks.require(
            all(value in range(5) for value in inner),
            "inner validation assignment is outside the frozen fold grid",
        )


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


def _validate_sha256sums(directory: Path, checks: Checks) -> None:
    checksum_path = directory / "SHA256SUMS"
    checks.require(checksum_path.is_file(), "SHA256SUMS missing")
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", maxsplit=1)
        path = directory / name
        checks.require(path.is_file(), f"checksummed artifact missing: {name}")
        checks.require(
            _sha256_file(path) == digest, f"checksummed artifact changed: {name}"
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
            for item in report.rglob("*")
            if item.is_file() and item.name not in {"run.json", "artifacts.json"}
        ),
        key=lambda item: item.relative_to(report).as_posix(),
    ):
        digest.update(path.relative_to(report).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
