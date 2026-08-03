"""Independently validate TREND-001 artifacts, strict OOF, and protection."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from elderly_monitoring.modules.mental_health.mood_social.personal_trend import (
    BRANCH_ORDER,
    BUNDLE_VERSION,
    COMPONENT_ORDER,
    C_VALUES,
    FEATURE_SCHEMA_SHA256,
    MANIFEST_VERSION,
    METRICS_VERSION,
    MODEL006_MANIFEST_SHA256,
    MODEL006_MODEL_SHA256,
    MODEL006_REPORT_CORE_SHA256,
    OOF_VERSION,
    OUTER_FOLD_COUNT,
    RUN_ID,
    SEARCH_VERSION,
    SPLIT_SHA256,
    TASK_ID,
    TRAINING_VERSION,
    PersonalTrendBundle,
    PersonalTrendError,
    TrendComponents,
    load_personal_trend_config,
    load_personal_trend_inputs,
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
        default=Path("configs/training/mood_social_personal_trend_v3_3_3.yaml"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else root / args.config
    checks = Checks()
    try:
        config = load_personal_trend_config(config_path, repository_root=root)
        inputs = load_personal_trend_inputs(config)
        result = validate(config, inputs, checks)
    except (PersonalTrendError, ValidationFailure, OSError, ValueError) as exc:
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
    checks.require(config.model_path.is_file(), "TREND-001 bundle is missing")
    checks.require(config.manifest_path.is_file(), "TREND-001 manifest is missing")
    checks.require(config.report_directory.is_dir(), "TREND-001 report is missing")
    manifest = _read_json(config.manifest_path)
    checks.require(
        manifest.get("version") == MANIFEST_VERSION, "manifest version drifted"
    )
    checks.require(
        manifest.get("training_version") == TRAINING_VERSION, "training version drifted"
    )
    checks.require(
        manifest.get("bundle_version") == BUNDLE_VERSION, "bundle version drifted"
    )
    checks.require(manifest.get("task_id") == TASK_ID, "task ID drifted")
    checks.require(manifest.get("run_id") == RUN_ID, "run ID drifted")
    checks.require(
        manifest.get("config_sha256") == config.config_sha256, "config hash drifted"
    )
    checks.require(manifest.get("split_sha256") == SPLIT_SHA256, "split hash drifted")
    checks.require(
        manifest.get("feature_schema_sha256") == FEATURE_SCHEMA_SHA256,
        "schema hash drifted",
    )
    checks.require(
        tuple(manifest.get("component_order", ())) == COMPONENT_ORDER,
        "component order drifted",
    )
    checks.require(
        tuple(manifest.get("branch_order", ())) == BRANCH_ORDER, "branch order drifted"
    )
    checks.require(
        not any(manifest.get("production_boundary", {}).values()),
        "production boundary changed",
    )
    checks.require(
        manifest.get("model_sha256") == _sha256_file(config.model_path),
        "model hash drifted",
    )

    bundle = joblib.load(config.model_path)
    checks.require(isinstance(bundle, PersonalTrendBundle), "bundle type drifted")
    bundle.validate()
    checks.require(
        tuple(bundle.branches) == BRANCH_ORDER, "bundle branch order drifted"
    )
    for branch, model in bundle.branches.items():
        checks.require(model.branch == branch, f"{branch} identity drifted")
        checks.require(model.selected_c in C_VALUES, f"{branch} selected C drifted")
        checks.require(model.training_row_count > 0, f"{branch} has no training rows")
        checks.require(
            not any(
                token in feature.lower()
                for feature in model.input_features
                for token in ("phq", "grade", "target", "mask", "model006")
            ),
            f"{branch} contains a forbidden risk input",
        )
        masked = TrendComponents(
            values=(0.0,) * len(COMPONENT_ORDER),
            personal_change_mask=0,
            reliability=0.0,
            valid_history_days=0,
            stable_baseline_ready=False,
            accepted_history_positions=(),
            current_feature_count=0,
            baseline_feature_count=0,
            details={},
        )
        checks.require(
            model.predict_probability(masked) is None,
            f"{branch} mask-zero prediction is non-null",
        )
        available = TrendComponents(
            values=(0.5,) * len(COMPONENT_ORDER),
            personal_change_mask=1,
            reliability=0.5,
            valid_history_days=7,
            stable_baseline_ready=True,
            accepted_history_positions=tuple(float(value) for value in range(7)),
            current_feature_count=2,
            baseline_feature_count=2,
            details={},
        )
        probability = model.predict_probability(available)
        checks.require(
            probability is not None and 0 <= probability <= 1,
            f"{branch} available prediction is invalid",
        )
    checks.require(
        bundle.production_activity_ecdf.training_participant_count == 4036,
        "production activity ECDF participant count drifted",
    )

    report = config.report_directory
    artifacts = _read_json(report / "artifacts.json")
    rows = artifacts.get("artifacts")
    checks.require(isinstance(rows, list), "artifact list is missing")
    checks.require(
        artifacts.get("artifact_count") == len(rows), "artifact count drifted"
    )
    for row in rows:
        path = report / str(row["path"])
        checks.require(path.is_file(), f"artifact missing: {row['path']}")
        checks.require(
            path.stat().st_size == int(row["bytes"]),
            f"artifact size drifted: {row['path']}",
        )
        checks.require(
            _sha256_file(path) == row["sha256"], f"artifact hash drifted: {row['path']}"
        )
    expected_core = _rows_sha256(rows)
    checks.require(
        artifacts.get("report_core_sha256") == expected_core, "report core drifted"
    )
    checks.require(
        manifest.get("report_core_sha256") == expected_core,
        "manifest report core drifted",
    )

    oof = pd.read_parquet(report / "oof_predictions.parquet")
    required = {
        "oof_version",
        "branch",
        "dataset_id",
        "prediction_id",
        "canonical_row_index",
        "global_participant_id",
        "outer_fold",
        "target",
        "available",
        "personal_change_mask",
        "reliability",
        "valid_history_days",
        "timescale_semantics",
        *COMPONENT_ORDER,
        "probability",
    }
    checks.require(required.issubset(oof.columns), "OOF columns are incomplete")
    checks.require(set(oof["oof_version"]) == {OOF_VERSION}, "OOF version drifted")
    checks.require(len(oof) == 27059, "OOF total row count drifted")
    checks.require(not oof["prediction_id"].duplicated().any(), "OOF IDs repeat")
    checks.require(
        set(oof["outer_fold"].astype(int)) == set(range(5)), "OOF outer folds drifted"
    )
    checks.require(
        oof["personal_change_mask"].isin([0, 1]).all(), "OOF mask is invalid"
    )
    checks.require(
        oof["available"].astype(int).eq(oof["personal_change_mask"]).all(),
        "OOF mask/availability mismatch",
    )
    checks.require(
        oof.loc[oof["available"], "probability"].notna().all(),
        "available OOF probability is null",
    )
    checks.require(
        oof.loc[~oof["available"], "probability"].isna().all(),
        "unavailable OOF probability is non-null",
    )
    checks.require(
        np.isfinite(oof.loc[:, list(COMPONENT_ORDER)].to_numpy()).all(),
        "OOF components are non-finite",
    )
    checks.require(
        (
            (oof.loc[:, list(COMPONENT_ORDER)] >= 0)
            & (oof.loc[:, list(COMPONENT_ORDER)] <= 1)
        )
        .all()
        .all(),
        "OOF components leave 0..1",
    )

    metrics_payload = _read_json(report / "metrics.json")
    checks.require(
        metrics_payload.get("version") == METRICS_VERSION, "metrics version drifted"
    )
    metrics = {row["branch"]: row for row in metrics_payload["branches"]}
    expected = {
        "activity": (10866, 1390, 379, "psyche_d"),
        "sleep": (10866, 1352, 370, "psyche_d"),
        "social": (5327, 5327, 186, "shenzhen_elderly"),
    }
    for branch in BRANCH_ORDER:
        branch_oof = oof[oof["branch"].eq(branch)]
        row_count, available_count, positive_count, dataset_id = expected[branch]
        checks.require(len(branch_oof) == row_count, f"{branch} OOF row count drifted")
        checks.require(
            int(branch_oof["available"].sum()) == available_count,
            f"{branch} availability drifted",
        )
        checks.require(
            set(branch_oof["dataset_id"]) == {dataset_id}, f"{branch} source drifted"
        )
        available = branch_oof[branch_oof["available"]]
        checks.require(
            int(available["target"].sum()) == positive_count,
            f"{branch} positives drifted",
        )
        weight = _weights(available)
        y = available["target"].to_numpy(dtype="int64")
        p = available["probability"].to_numpy(dtype="float64")
        recalculated = {
            "auroc": roc_auc_score(y, p),
            "auprc": average_precision_score(y, p),
            "brier": brier_score_loss(y, p),
            "weighted_auroc": roc_auc_score(y, p, sample_weight=weight),
            "weighted_auprc": average_precision_score(y, p, sample_weight=weight),
            "weighted_brier": np.average(np.square(p - y), weights=weight),
        }
        for name, value in recalculated.items():
            checks.require(
                math.isclose(float(metrics[branch][name]), float(value), abs_tol=1e-12),
                f"{branch} {name} drifted",
            )

    assignments = inputs.assignments.set_index("global_participant_id")
    for row in oof.itertuples(index=False):
        expected_fold = int(
            assignments.loc[str(row.global_participant_id), "outer_fold"]
        )
        checks.require(
            int(row.outer_fold) == expected_fold,
            f"OOF participant fold drifted: {row.prediction_id}",
        )

    search = _read_json(report / "search_results.json")
    checks.require(search.get("version") == SEARCH_VERSION, "search version drifted")
    checks.require(
        search.get("outer_test_fold_role") == "evaluation_only",
        "outer test role drifted",
    )
    checks.require(
        tuple(item["branch"] for item in search["branches"]) == BRANCH_ORDER,
        "search branch order drifted",
    )
    for branch in search["branches"]:
        checks.require(
            len(branch["outer_searches"]) == OUTER_FOLD_COUNT,
            "outer search count drifted",
        )
        for outer in branch["outer_searches"]:
            checks.require(
                len(outer["candidates"]) == len(C_VALUES), "C grid is incomplete"
            )
            checks.require(
                outer["outer_test_role"] == "evaluation_only",
                "outer fold used for selection",
            )

    warnings = _read_json(report / "warnings.json")["warnings"]
    warning_codes = {item["code"] for item in warnings}
    checks.require(
        "NO_DIRECT_S10_PHQ9_VALIDATION" in warning_codes, "S10 proxy warning missing"
    )
    checks.require(
        "PSYCHE_D_NOMINAL_MONTH_TIMESCALE_PROXY" in warning_codes,
        "timescale warning missing",
    )
    baseline_policy = _read_json(report / "baseline_policy.json")
    checks.require(
        baseline_policy["runtime"]["walking_speed_normalization"]
        == "per_scene_clip_q10_q90_then_same_day_non_null_median",
        "walking-speed normalization policy drifted",
    )
    checks.require(
        baseline_policy["runtime"]["walking_speed_denominator_epsilon"] == 1.0e-6,
        "walking-speed denominator epsilon drifted",
    )
    protection = _read_json(report / "upstream_protection.json")
    checks.require(
        protection["model006_model_sha256"] == MODEL006_MODEL_SHA256,
        "MODEL-006 model binding drifted",
    )
    checks.require(
        protection["model006_manifest_sha256"] == MODEL006_MANIFEST_SHA256,
        "MODEL-006 manifest binding drifted",
    )
    checks.require(
        protection["model006_report_core_sha256"] == MODEL006_REPORT_CORE_SHA256,
        "MODEL-006 report binding drifted",
    )
    return {
        "run_id": RUN_ID,
        "model_sha256": _sha256_file(config.model_path),
        "manifest_sha256": _sha256_file(config.manifest_path),
        "report_core_sha256": expected_core,
        "oof_rows": len(oof),
        "available_rows": int(oof["available"].sum()),
    }


def _weights(frame: pd.DataFrame) -> np.ndarray:
    y = frame["target"].astype(int).to_numpy()
    participants = frame["global_participant_id"].astype(str).to_numpy()
    output = np.zeros(len(frame), dtype="float64")
    classes = sorted(set(y.tolist()))
    for target in classes:
        indices = np.flatnonzero(y == target)
        ids, counts = np.unique(participants[indices], return_counts=True)
        count_by_id = dict(zip(ids, counts, strict=True))
        for index in indices:
            output[index] = (
                (1 / len(classes)) / len(ids) / count_by_id[participants[index]]
            )
    return output * (len(output) / output.sum())


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValidationFailure(f"JSON root is not an object: {path}")
    return value


def _rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = "".join(
        f"{row['path']}\0{row['bytes']}\0{row['sha256']}\n" for row in rows
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
