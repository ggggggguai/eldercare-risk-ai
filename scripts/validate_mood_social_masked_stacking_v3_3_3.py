"""Independently validate FUSION-002 inputs, model and published reports."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from elderly_monitoring.modules.mental_health.mood_social.fusion.masked_stacking import (
    BUNDLE_VERSION,
    C_VALUES,
    DEVICE_COMBINATION_ORDER,
    INPUT_FEATURES,
    MANIFEST_VERSION,
    OOF_VERSION,
    RUN_ID,
    TASK_ID,
    MaskedStackingError,
    evidence_combination,
    effective_evidence,
    load_masked_stacking_config,
    load_masked_stacking_inputs,
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
        default=Path("configs/training/mood_social_masked_stacking_v3_3_3.yaml"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    path = args.config if args.config.is_absolute() else root / args.config
    checks = Checks()
    try:
        config = load_masked_stacking_config(path, repository_root=root)
        inputs = load_masked_stacking_inputs(config)
        result = validate(config, inputs, checks)
    except (
        MaskedStackingError,
        ValidationFailure,
        OSError,
        ValueError,
        KeyError,
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


def validate(config: Any, inputs: Any, checks: Checks) -> dict[str, Any]:
    report = config.report_directory
    required = (
        "oof_predictions.parquet",
        "metrics_summary.parquet",
        "calibration_curve.parquet",
        "source_metrics.parquet",
        "coverage_metrics.parquet",
        "combination_metrics.parquet",
        "metrics.json",
        "calibration_results.json",
        "coefficient_summary.json",
        "warnings.json",
        "search_results.json",
        "weights.json",
        "upstream_protection.json",
        "training_manifest.json",
        "model_card.md",
        "artifacts.json",
        "run.json",
    )
    for name in required:
        checks.require(
            (report / name).is_file(), f"missing FUSION-002 artifact: {name}"
        )
    checks.require(config.model_path.is_file(), "FUSION-002 model is missing")
    checks.require(config.manifest_path.is_file(), "FUSION-002 manifest is missing")

    oof = pd.read_parquet(report / "oof_predictions.parquet")
    checks.require(len(oof) == len(inputs.table), "OOF row count changed")
    checks.require(oof["oof_version"].eq(OOF_VERSION).all(), "OOF version changed")
    checks.require(
        not oof[["dataset_id", "canonical_row_index"]].duplicated().any(),
        "OOF canonical identity duplicated",
    )
    checks.require(
        oof["fusion_probability"]
        .notna()
        .eq(oof["fusion_available"].astype(bool))
        .all(),
        "availability gate changed",
    )
    checks.require(
        oof.loc[~oof["fusion_available"].astype(bool), "fusion_logit"].isna().all(),
        "gated rows have a logit",
    )
    checks.require(
        oof.loc[~oof["fusion_available"].astype(bool), "fusion_confidence"].eq(0).all(),
        "gated rows have confidence",
    )
    checks.require(
        oof.loc[oof["fusion_available"].astype(bool), "fusion_probability"]
        .between(0, 1)
        .all(),
        "probability range changed",
    )
    checks.require(
        oof["evidence_combination"].isin(DEVICE_COMBINATION_ORDER).all(),
        "evidence combination changed",
    )
    key = ["dataset_id", "canonical_row_index"]
    expected_frame = inputs.table.set_index(key)
    observed_order = oof.set_index(key).index
    expected_combination = (
        evidence_combination(expected_frame)
        .reindex(observed_order)
        .reset_index(drop=True)
    )
    observed_combination = (
        oof.set_index(key)["evidence_combination"]
        .reindex(observed_order)
        .reset_index(drop=True)
    )
    checks.require(
        observed_combination.equals(expected_combination),
        "evidence combination is not deterministic",
    )
    expected_evidence = (
        pd.Series(effective_evidence(expected_frame), index=expected_frame.index)
        .reindex(observed_order)
        .to_numpy()
    )
    observed_evidence = (
        oof.set_index(key)["fusion_effective_evidence"]
        .reindex(observed_order)
        .to_numpy()
    )
    checks.require(
        np.allclose(observed_evidence, expected_evidence, equal_nan=False),
        "effective evidence changed",
    )

    model = joblib.load(config.model_path)
    model.validate()
    checks.require(
        model.bundle_version == BUNDLE_VERSION, "model bundle version changed"
    )
    checks.require(model.input_features == INPUT_FEATURES, "model input order changed")
    checks.require(model.selected_c in C_VALUES, "selected C changed")
    checks.require(
        model.probability_representation == "current_calibrated",
        "probability representation changed",
    )

    manifest = _json(config.manifest_path)
    checks.require(manifest["version"] == MANIFEST_VERSION, "manifest version changed")
    checks.require(
        manifest["task_id"] == TASK_ID and manifest["run_id"] == RUN_ID,
        "manifest identity changed",
    )
    checks.require(manifest["strict_oof"] is True, "strict OOF flag changed")
    checks.require(
        manifest["posthoc_calibrator_selected"] is False,
        "posthoc calibrator boundary changed",
    )
    checks.require(
        manifest["production_boundary"]["select_threshold"] is False,
        "threshold boundary changed",
    )
    checks.require(
        manifest["production_boundary"]["change_http_behavior"] is False,
        "HTTP boundary changed",
    )
    checks.require(
        _sha256_file(config.model_path) == manifest["model_sha256"],
        "model hash changed",
    )
    checks.require(
        _sha256_file(report / "oof_predictions.parquet") == manifest["oof_sha256"],
        "OOF hash changed",
    )
    _validate_artifacts(report, checks)
    protection = _json(report / "upstream_protection.json")
    checks.require(
        protection["model006_enters_fusion"] is False, "MODEL-006 entered fusion"
    )
    checks.require(
        protection["fusion_oof_table_sha256"]
        == config.payload["input"]["fusion_oof_table_sha256"],
        "upstream table protection changed",
    )
    checks.require(
        protection["fusion_oof_report_core_sha256"]
        == config.payload["input"]["fusion_oof_report_core_sha256"],
        "upstream report protection changed",
    )
    return {
        "run_id": RUN_ID,
        "row_count": len(oof),
        "available_row_count": int(oof["fusion_available"].sum()),
        "model_sha256": _sha256_file(config.model_path),
        "manifest_sha256": _sha256_file(config.manifest_path),
        "report_core_sha256": _report_core_sha256(report),
    }


def _validate_artifacts(report: Path, checks: Checks) -> None:
    artifacts = _json(report / "artifacts.json")
    checks.require(
        isinstance(artifacts.get("artifacts"), list), "artifact hashes missing"
    )
    for item in artifacts["artifacts"]:
        path = report / str(item["path"])
        checks.require(path.is_file(), f"artifact missing: {item['path']}")
        checks.require(
            _sha256_file(path) == item["sha256"],
            f"artifact hash changed: {item['path']}",
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
