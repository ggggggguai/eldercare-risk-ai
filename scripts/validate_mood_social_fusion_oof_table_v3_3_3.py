"""Independently validate FUSION-001 strict OOF alignment and artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from elderly_monitoring.modules.mental_health.mood_social.fusion.oof_table import (
    BASELINE_PROTECTION_SHA256,
    EVALUATION_CORE_SHA256,
    EXPERT_ORDER,
    FEATURE_SCHEMA_SHA256,
    MANIFEST_VERSION,
    RUN_ID,
    SPLIT_ID,
    SPLIT_SHA256,
    TABLE_VERSION,
    TREND_ORDER,
    FusionOOFError,
    build_fusion_oof_table,
    load_fusion_oof_config,
    load_fusion_oof_inputs,
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
        default=Path("configs/training/mood_social_fusion_oof_table_v3_3_3.yaml"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    path = args.config if args.config.is_absolute() else root / args.config
    checks = Checks()
    try:
        config = load_fusion_oof_config(path, repository_root=root)
        inputs = load_fusion_oof_inputs(config)
        result = validate(config, inputs, checks)
    except (FusionOOFError, ValidationFailure, OSError, ValueError, KeyError) as exc:
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
        "fusion_oof_table.parquet",
        "coverage_summary.parquet",
        "confidence_reliability.parquet",
        "alignment_diagnostics.json",
        "warnings.json",
        "upstream_protection.json",
        "fusion_table_manifest.json",
        "fusion_table_config.yaml",
        "artifacts.json",
        "run.json",
    )
    for name in required:
        checks.require(
            (report / name).is_file(), f"missing FUSION-001 artifact: {name}"
        )
    table = pd.read_parquet(report / "fusion_oof_table.parquet")
    expected, diagnostics, coverage, reliability = build_fusion_oof_table(inputs)
    checks.require(len(table) == 22191, "fusion row count changed")
    checks.require(
        table.equals(expected), "fusion table is not a deterministic rebuild"
    )
    checks.require(
        table["fusion_schema_version"].eq(TABLE_VERSION).all(), "table version drifted"
    )
    checks.require(table["fusion_row_id"].is_unique, "fusion row IDs are duplicated")
    checks.require(
        table["window_key"].equals(table["fusion_row_id"]), "window identity changed"
    )
    checks.require(
        table["global_participant_id"].nunique() == 15361,
        "participant coverage changed",
    )
    checks.require(int(table["binary_target"].sum()) == 4079, "target coverage changed")
    checks.require(
        table.groupby("global_participant_id")["outer_fold"].nunique().max() == 1,
        "participant crosses outer folds",
    )
    checks.require(
        set(table["outer_fold"].astype(int)) == set(range(5)),
        "outer fold coverage changed",
    )
    checks.require(
        table[["dataset_id", "canonical_row_index"]].duplicated().sum() == 0,
        "canonical identity duplicated",
    )
    checks.require(
        table["feature_mask_json"].notna().all(), "feature mask layer is missing"
    )
    checks.require(
        table["day_mask_semantics"].eq("aggregate_public_window_observed_proxy").all(),
        "day mask semantics changed",
    )
    checks.require(
        not diagnostics["model006_columns_consumed"],
        "MODEL-006 columns entered diagnostics",
    )
    checks.require(diagnostics["strict_oof"] is True, "strict OOF flag is false")
    checks.require(
        diagnostics["stacking_fitted"] is False, "stacking was fitted in FUSION-001"
    )
    checks.require(
        diagnostics["threshold_selected"] is False,
        "threshold was selected in FUSION-001",
    )
    checks.require(
        diagnostics["expected_row_count"] == len(table),
        "diagnostic row count disagrees",
    )
    checks.require(
        diagnostics["observed_row_count"] == len(table), "observed row count disagrees"
    )
    checks.require(
        diagnostics["coverage_pattern_count"] == table["mask_pattern"].nunique(),
        "mask pattern count disagrees",
    )
    checks.require(
        _json(report / "alignment_diagnostics.json") == diagnostics,
        "alignment diagnostics changed",
    )
    checks.require(
        pd.read_parquet(report / "coverage_summary.parquet").equals(coverage),
        "coverage summary changed",
    )
    checks.require(
        pd.read_parquet(report / "confidence_reliability.parquet").equals(reliability),
        "reliability summary changed",
    )
    _validate_manifest(report, config, table, checks)
    _validate_protection(report, checks)
    _validate_expert_columns(table, checks)
    _validate_trend_columns(table, checks)
    _validate_masks_and_probabilities(table, checks)
    _validate_upstream_row_keys(table, inputs, checks)
    return {
        "run_id": RUN_ID,
        "row_count": len(table),
        "participant_count": int(table["global_participant_id"].nunique()),
        "positive_row_count": int(table["binary_target"].sum()),
        "table_sha256": _sha256_file(report / "fusion_oof_table.parquet"),
        "report_core_sha256": _report_core_sha256(report),
    }


def _validate_manifest(
    report: Path, config: Any, table: pd.DataFrame, checks: Checks
) -> None:
    manifest = _json(report / "fusion_table_manifest.json")
    checks.require(manifest["version"] == MANIFEST_VERSION, "manifest version drifted")
    checks.require(manifest["task_id"] == "FUSION-001", "manifest task drifted")
    checks.require(manifest["run_id"] == RUN_ID, "manifest run drifted")
    checks.require(
        manifest["config_sha256"] == config.config_sha256,
        "manifest config hash drifted",
    )
    checks.require(manifest["split_id"] == SPLIT_ID, "manifest split ID drifted")
    checks.require(
        manifest["split_sha256"] == SPLIT_SHA256, "manifest split hash drifted"
    )
    checks.require(
        manifest["feature_schema_sha256"] == FEATURE_SCHEMA_SHA256,
        "manifest schema hash drifted",
    )
    checks.require(manifest["row_count"] == len(table), "manifest row count drifted")
    checks.require(
        manifest["participant_count"] == table["global_participant_id"].nunique(),
        "manifest participant count drifted",
    )
    checks.require(
        manifest["positive_row_count"] == int(table["binary_target"].sum()),
        "manifest positive count drifted",
    )
    checks.require(manifest["strict_oof"] is True, "manifest strict OOF flag drifted")
    checks.require(
        manifest["model006_included"] is False, "manifest includes MODEL-006"
    )
    checks.require(
        isinstance(manifest.get("artifacts"), dict), "manifest artifact hashes missing"
    )
    for name, item in manifest["artifacts"].items():
        path = report / name
        checks.require(path.is_file(), f"manifest artifact missing: {name}")
        checks.require(
            item["sha256"] == _sha256_file(path), f"artifact hash drifted: {name}"
        )


def _validate_protection(report: Path, checks: Checks) -> None:
    protection = _json(report / "upstream_protection.json")
    checks.require(
        protection["split_sha256"] == SPLIT_SHA256, "upstream split protection drifted"
    )
    checks.require(
        protection["feature_schema_sha256"] == FEATURE_SCHEMA_SHA256,
        "upstream schema protection drifted",
    )
    checks.require(
        protection["baseline_protection_sha256"] == BASELINE_PROTECTION_SHA256,
        "baseline protection drifted",
    )
    checks.require(
        protection["evaluation_core_sha256"] == EVALUATION_CORE_SHA256,
        "evaluation core protection drifted",
    )
    checks.require(
        protection["model006"]["enters_fusion"] is False,
        "MODEL-006 protection boundary drifted",
    )


def _validate_expert_columns(table: pd.DataFrame, checks: Checks) -> None:
    for expert in EXPERT_ORDER:
        prefix = f"expert_{expert}_"
        for column in (
            "raw_probability",
            "current_probability",
            "model_reliability",
            "feature_coverage",
            "confidence",
            "feature_mask",
            "day_mask",
            "expert_mask",
        ):
            checks.require(
                f"{prefix}{column}" in table.columns,
                f"missing expert column: {prefix}{column}",
            )
        applicable = table[f"{prefix}applicable"].astype(int)
        expected = table["dataset_id"].isin(
            {"psyche_d", "resilient", "nhanes"}
            if expert != "physiology" and expert != "social_context"
            else (
                {"resilient"}
                if expert == "physiology"
                else {"resilient", "nhanes", "shenzhen_elderly", "nhanes_ssq_2005_2008"}
            )
        )
        checks.require(
            applicable.eq(expected.astype(int)).all(), f"{expert} applicability changed"
        )


def _validate_trend_columns(table: pd.DataFrame, checks: Checks) -> None:
    for branch in TREND_ORDER:
        prefix = f"trend_{branch}_"
        for column in (
            "personal_change_evidence",
            "reliability",
            "feature_mask",
            "day_mask",
            "personal_change_mask",
            "valid_history_days",
        ):
            checks.require(
                f"{prefix}{column}" in table.columns,
                f"missing trend column: {prefix}{column}",
            )
        expected = table["dataset_id"].eq(
            "psyche_d" if branch != "social" else "shenzhen_elderly"
        )
        checks.require(
            table[f"{prefix}applicable"].astype(int).eq(expected.astype(int)).all(),
            f"{branch} applicability changed",
        )


def _validate_masks_and_probabilities(table: pd.DataFrame, checks: Checks) -> None:
    for expert in EXPERT_ORDER:
        prefix = f"expert_{expert}_"
        mask = table[f"{prefix}expert_mask"].fillna(0).astype(int)
        current = table[f"{prefix}current_probability"]
        raw = table[f"{prefix}raw_probability"]
        confidence = table[f"{prefix}confidence"].fillna(0.0)
        coverage = table[f"{prefix}feature_coverage"].fillna(0.0)
        reliability = table[f"{prefix}model_reliability"].fillna(0.0)
        checks.require(mask.isin([0, 1]).all(), f"{expert} expert mask is non-binary")
        checks.require(
            current[mask.eq(0)].isna().all(),
            f"{expert} unavailable current probability is non-null",
        )
        checks.require(
            raw[mask.eq(0)].isna().all(),
            f"{expert} unavailable raw probability is non-null",
        )
        checks.require(
            current[mask.eq(1)].between(0, 1).all(),
            f"{expert} current probability is invalid",
        )
        checks.require(
            raw[mask.eq(1)].between(0, 1).all(), f"{expert} raw probability is invalid"
        )
        checks.require(
            (confidence[mask.eq(1)] - reliability[mask.eq(1)] * coverage[mask.eq(1)])
            .abs()
            .lt(1e-12)
            .all(),
            f"{expert} confidence formula changed",
        )
        checks.require(
            confidence[mask.eq(0)].eq(0).all(),
            f"{expert} unavailable confidence is nonzero",
        )
        for column in ("feature_mask",):
            checks.require(
                table[f"{prefix}{column}"].isin([0, 1]).all(),
                f"{expert} feature mask is non-binary",
            )
    for branch in TREND_ORDER:
        prefix = f"trend_{branch}_"
        mask = table[f"{prefix}personal_change_mask"].fillna(0).astype(int)
        evidence = table[f"{prefix}personal_change_evidence"]
        reliability = table[f"{prefix}reliability"]
        checks.require(
            mask.isin([0, 1]).all(), f"{branch} personal change mask is non-binary"
        )
        checks.require(
            evidence[mask.eq(0)].isna().all(),
            f"{branch} unavailable evidence is non-null",
        )
        checks.require(
            evidence[mask.eq(1)].between(0, 1).all(), f"{branch} evidence is invalid"
        )
        checks.require(
            reliability.between(0, 1).all(), f"{branch} reliability is invalid"
        )


def _validate_upstream_row_keys(
    table: pd.DataFrame, inputs: Any, checks: Checks
) -> None:
    for expert, frame in inputs.expert_frames.items():
        prefix = f"expert_{expert}_"
        applicable = table[f"{prefix}applicable"].eq(1)
        keys = set(
            zip(
                table.loc[applicable, "dataset_id"],
                table.loc[applicable, "canonical_row_index"],
                strict=True,
            )
        )
        expected = set(
            zip(
                frame["dataset_id"].astype(str),
                frame["canonical_row_index"].astype(int),
                strict=True,
            )
        )
        checks.require(keys == expected, f"{expert} upstream key set changed")
        checks.require(
            table.loc[applicable, "outer_fold"].astype(int).isin(set(range(5))).all(),
            f"{expert} outer fold invalid",
        )
    for branch in TREND_ORDER:
        frame = inputs.trend_frame[inputs.trend_frame["branch"].astype(str).eq(branch)]
        applicable = table[f"trend_{branch}_applicable"].eq(1)
        keys = set(
            zip(
                table.loc[applicable, "dataset_id"],
                table.loc[applicable, "canonical_row_index"],
                strict=True,
            )
        )
        expected = set(
            zip(
                frame["dataset_id"].astype(str),
                frame["canonical_row_index"].astype(int),
                strict=True,
            )
        )
        checks.require(
            keys == expected, f"PersonalTrend {branch} upstream key set changed"
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
    names = (
        "alignment_diagnostics.json",
        "confidence_reliability.parquet",
        "coverage_summary.parquet",
        "fusion_oof_table.parquet",
        "fusion_table_config.yaml",
        "fusion_table_manifest.json",
        "upstream_protection.json",
        "warnings.json",
    )
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update((report / name).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
