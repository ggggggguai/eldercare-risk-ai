"""Independently validate EVAL-001 outputs and frozen read-only boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
import yaml


DEFAULT_CONFIG = Path("configs/evaluation/mood_social_expert_audit_v3_3_3.yaml")
DEFAULT_REPORT = Path("reports/mental_health/mood_social/MH-20260801-006")
EXPERTS = ("activity", "sleep", "joint", "physiology", "social_context")
LOSO_EXPERTS = ("activity", "sleep", "joint", "social_context")
BASE_VARIANTS = ("raw", "active_calibrated")
CALIBRATION_VARIANTS = tuple(
    f"cf_{method}__{suffix}"
    for method in ("platt", "isotonic")
    for suffix in (
        "frozen_four_level",
        "natural_sample",
        "target_prior_0p05",
        "target_prior_0p10",
        "target_prior_0p20",
        "target_prior_0p30",
    )
)
VARIANTS = (*BASE_VARIANTS, *CALIBRATION_VARIANTS)
LOSO_SOURCE_COUNTS = {"activity": 3, "sleep": 3, "joint": 3, "social_context": 4}
REQUIRED_REPORT_FILES = {
    "artifact.json",
    "artifacts.json",
    "baseline_protection.json",
    "calibration_candidates.parquet",
    "calibration_curves.parquet",
    "calibration_fits.json",
    "evaluation_config.yaml",
    "figures/calibration_brier_comparison.png",
    "figures/leave_one_source_auprc.png",
    "figures/source_auprc.png",
    "figures/threshold_tradeoff.png",
    "leave_one_source_metrics.json",
    "leave_one_source_predictions.parquet",
    "leave_one_source_preprocessing.json",
    "metric_summary.parquet",
    "metrics.json",
    "physiology_uncertainty.json",
    "report.md",
    "run.json",
    "source_metrics.parquet",
    "threshold_curves.parquet",
}


class ValidationError(RuntimeError):
    """Raised without exposing participant-level values."""


def validate(root: Path, config_path: Path, report_dir: Path) -> dict[str, Any]:
    checks: list[str] = []
    repository = root.resolve()
    report = report_dir.resolve()
    config = _read_yaml(config_path.resolve())

    _check(config.get("task_id") == "EVAL-001", "config_task", checks)
    _check(config.get("run_id") == "MH-20260801-006", "config_run", checks)
    _check(
        tuple(config.get("baseline_protection", {})) == EXPERTS,
        "config_experts",
        checks,
    )
    _check(
        tuple(config.get("leave_one_source_out", {}).get("experts", ()))
        == LOSO_EXPERTS,
        "config_loso_experts",
        checks,
    )
    _check(
        not any(
            config["reporting"].get(name)
            for name in (
                "choose_production_threshold",
                "modify_attention_level_boundaries",
                "publish_diagnostic_models",
                "overwrite_active_artifacts",
            )
        ),
        "config_read_only",
        checks,
    )
    actual_files = {
        path.relative_to(report).as_posix()
        for path in report.rglob("*")
        if path.is_file()
    }
    _check(actual_files == REQUIRED_REPORT_FILES, "report_file_inventory", checks)
    _validate_artifact_manifest(report, checks)
    _validate_frozen_inputs(repository, config, report, checks)
    _validate_run(report, config, checks)

    predictions = pd.read_parquet(report / "calibration_candidates.parquet")
    metrics = pd.read_parquet(report / "metric_summary.parquet")
    source_metrics = pd.read_parquet(report / "source_metrics.parquet")
    calibration_curves = pd.read_parquet(report / "calibration_curves.parquet")
    thresholds = pd.read_parquet(report / "threshold_curves.parquet")
    fits = _read_json(report / "calibration_fits.json")
    _validate_calibration_outputs(
        repository,
        config,
        predictions,
        metrics,
        source_metrics,
        calibration_curves,
        thresholds,
        fits,
        checks,
    )
    loso_predictions = pd.read_parquet(report / "leave_one_source_predictions.parquet")
    loso_metrics = _read_json(report / "leave_one_source_metrics.json")
    loso_audit = _read_json(report / "leave_one_source_preprocessing.json")
    _validate_loso(loso_predictions, loso_metrics, loso_audit, checks)
    _validate_physiology(report, checks)
    _validate_reader_artifact(report, checks)
    _validate_figures(report, checks)

    return {
        "status": "pass",
        "check_count": len(checks),
        "run_id": config["run_id"],
        "expert_count": len(EXPERTS),
        "probability_variant_count_per_expert": len(VARIANTS),
        "calibration_prediction_row_count": len(predictions),
        "metric_row_count": len(metrics),
        "source_metric_row_count": len(source_metrics),
        "threshold_row_count": len(thresholds),
        "loso_prediction_row_count": len(loso_predictions),
        "report_core_sha256": _read_json(report / "artifacts.json")["core_sha256"],
    }


def _validate_artifact_manifest(report: Path, checks: list[str]) -> None:
    manifest = _read_json(report / "artifacts.json")
    rows = manifest.get("artifacts")
    _check(
        isinstance(rows, list) and len(rows) == 20, "artifact_manifest_count", checks
    )
    expected_paths = REQUIRED_REPORT_FILES - {"artifacts.json"}
    _check(
        {row.get("path") for row in rows} == expected_paths, "artifact_paths", checks
    )
    recomputed: list[dict[str, Any]] = []
    for row in rows:
        path = report / str(row["path"])
        expected = {
            "path": str(row["path"]),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        _check(row == expected, f"artifact_hash_{row['path']}", checks)
        recomputed.append(expected)
    _check(
        manifest.get("core_sha256")
        == _artifact_rows_sha256(
            [row for row in recomputed if row["path"] != "run.json"]
        ),
        "artifact_core_hash",
        checks,
    )


def _validate_frozen_inputs(
    root: Path,
    config: Mapping[str, Any],
    report: Path,
    checks: list[str],
) -> None:
    split = config["split"]
    split_path = root / str(split["relative_path"])
    _check(_sha256(split_path) == split["sha256"], "split_hash", checks)
    split_payload = _read_json(split_path)
    schema_path = root / str(
        split_payload["feature_schema_binding"]["snapshot_relative_path"]
    )
    _check(
        _sha256(schema_path) == split["feature_schema_sha256"],
        "schema_hash",
        checks,
    )
    protection = _read_json(report / "baseline_protection.json")
    for expert in EXPERTS:
        binding = config["baseline_protection"][expert]
        _check(
            _sha256(root / binding["model_path"]) == binding["model_sha256"],
            f"{expert}_model_hash",
            checks,
        )
        _check(
            _sha256(root / binding["manifest_path"]) == binding["manifest_sha256"],
            f"{expert}_manifest_hash",
            checks,
        )
        protected_report = root / binding["report_path"]
        for filename, field in (
            ("predictions.parquet", "predictions_sha256"),
            ("metrics.json", "metrics_sha256"),
            ("search_results.json", "search_results_sha256"),
            ("training_config.yaml", "training_config_sha256"),
        ):
            _check(
                _sha256(protected_report / filename) == binding[field],
                f"{expert}_{filename}_hash",
                checks,
            )
        _check(
            protection["experts"][expert]["model"]["sha256"] == binding["model_sha256"],
            f"{expert}_protection_record",
            checks,
        )


def _validate_run(report: Path, config: Mapping[str, Any], checks: list[str]) -> None:
    run = _read_json(report / "run.json")
    _check(
        run.get("status") == "completed"
        and run.get("run_id") == config["run_id"]
        and run.get("task_id") == "EVAL-001",
        "run_identity",
        checks,
    )
    _check(
        run.get("calibration_candidate_count_per_expert") == 12,
        "run_candidates",
        checks,
    )
    _check(
        run.get("probability_variant_count_per_expert") == 14, "run_variants", checks
    )
    _check(
        run.get("production_threshold_selected") is False, "run_no_threshold", checks
    )
    _check(
        run.get("attention_level_boundaries_modified") is False,
        "run_no_boundary_change",
        checks,
    )
    _check(run.get("diagnostic_models_published") is False, "run_no_models", checks)
    _check(
        run.get("baseline_protection_before_sha256")
        == run.get("baseline_protection_after_sha256"),
        "run_protection_stable",
        checks,
    )


def _validate_calibration_outputs(
    root: Path,
    config: Mapping[str, Any],
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    source_metrics: pd.DataFrame,
    calibration_curves: pd.DataFrame,
    thresholds: pd.DataFrame,
    fits: Mapping[str, Any],
    checks: list[str],
) -> None:
    identity = ["expert", "prediction_id"]
    _check(
        not predictions.duplicated(identity).any(), "candidate_identity_unique", checks
    )
    _check(set(fits) == set(EXPERTS), "fit_experts", checks)
    _check(len(metrics) == len(EXPERTS) * len(VARIANTS), "metric_row_count", checks)
    _check(
        not metrics.duplicated(["expert", "probability_variant"]).any(),
        "metric_identity_unique",
        checks,
    )
    for expert in EXPERTS:
        group = predictions.loc[predictions["expert"].eq(expert)].copy()
        probability_columns = tuple(column for column in group if column in VARIANTS)
        _check(set(probability_columns) == set(VARIANTS), f"{expert}_variants", checks)
        probability = group.loc[:, VARIANTS].to_numpy(dtype="float64")
        _check(
            np.isfinite(probability).all()
            and np.all((probability >= 0.0) & (probability <= 1.0)),
            f"{expert}_probability_bounds",
            checks,
        )
        binding = config["baseline_protection"][expert]
        published = pd.read_parquet(
            root / binding["report_path"] / "predictions.parquet"
        )
        published = published.loc[published["expert_mask"].eq(1)].sort_values(
            "prediction_id", kind="stable"
        )
        candidate = group.sort_values("prediction_id", kind="stable")
        _check(
            candidate["prediction_id"].tolist() == published["prediction_id"].tolist(),
            f"{expert}_published_coverage",
            checks,
        )
        _check(
            np.array_equal(
                candidate["raw"].to_numpy(),
                published["raw_probability"].to_numpy(),
            ),
            f"{expert}_raw_exact",
            checks,
        )
        _check(
            np.array_equal(
                candidate["active_calibrated"].to_numpy(),
                published["calibrated_probability"].to_numpy(),
            ),
            f"{expert}_active_exact",
            checks,
        )
        participant_fold_counts = published.groupby("global_participant_id")[
            "outer_fold"
        ].nunique()
        _check(
            participant_fold_counts.eq(1).all(), f"{expert}_participant_fold", checks
        )
        _check(
            set(group["outer_fold"]) == {0, 1, 2, 3, 4}, f"{expert}_five_folds", checks
        )
        _check(
            set(fits[expert]) == set(CALIBRATION_VARIANTS),
            f"{expert}_fit_variants",
            checks,
        )
        for variant, fit in fits[expert].items():
            folds = fit.get("folds", [])
            _check(
                [item.get("outer_fold") for item in folds] == [0, 1, 2, 3, 4],
                f"{expert}_{variant}_fit_folds",
                checks,
            )
            _check(
                sum(int(item["test_row_count"]) for item in folds) == len(group),
                f"{expert}_{variant}_fit_coverage",
                checks,
            )
            for item in folds:
                _check(
                    item["train_participant_sha256"] != item["test_participant_sha256"],
                    f"{expert}_{variant}_fold_hash_separation",
                    checks,
                )
                expected_prior = _variant_prior(variant)
                if expected_prior is not None:
                    _check(
                        np.isclose(item["weighted_fit_prevalence"], expected_prior),
                        f"{expert}_{variant}_weighted_prior",
                        checks,
                    )
                elif "frozen_four_level" in variant:
                    _check(
                        np.isclose(item["weighted_fit_prevalence"], 0.5),
                        f"{expert}_{variant}_four_level_prior",
                        checks,
                    )
        for variant in VARIANTS:
            row = metrics.loc[
                metrics["expert"].eq(expert)
                & metrics["probability_variant"].eq(variant)
            ].iloc[0]
            y = group["binary_target"].to_numpy(dtype="int64")
            score = group[variant].to_numpy(dtype="float64")
            _check(
                np.isclose(row["natural_auprc"], average_precision_score(y, score)),
                f"{expert}_{variant}_auprc",
                checks,
            )
            _check(
                np.isclose(row["natural_auroc"], roc_auc_score(y, score)),
                f"{expert}_{variant}_auroc",
                checks,
            )
            _check(
                np.isclose(row["natural_brier_score"], np.mean((score - y) ** 2)),
                f"{expert}_{variant}_brier",
                checks,
            )
            curve = thresholds.loc[
                thresholds["expert"].eq(expert)
                & thresholds["probability_variant"].eq(variant)
            ]
            expected_count = len(np.unique(score))
            expected_count += int(score.max() < 1.0) + int(score.min() > 0.0)
            _check(
                len(curve) == expected_count,
                f"{expert}_{variant}_threshold_count",
                checks,
            )
            _check(
                curve["threshold"].is_monotonic_decreasing,
                f"{expert}_{variant}_threshold_order",
                checks,
            )
            _check(
                curve["true_positive"].is_monotonic_increasing,
                f"{expert}_{variant}_tp_order",
                checks,
            )
            _check(
                curve["false_positive"].is_monotonic_increasing,
                f"{expert}_{variant}_fp_order",
                checks,
            )
        _check(
            int(
                source_metrics.loc[source_metrics["expert"].eq(expert)]
                .groupby("probability_variant")["row_count"]
                .sum()
                .min()
            )
            == len(group),
            f"{expert}_source_metric_coverage",
            checks,
        )
        _check(
            int(
                calibration_curves.loc[calibration_curves["expert"].eq(expert)]
                .groupby("probability_variant")["row_count"]
                .sum()
                .min()
            )
            == len(group),
            f"{expert}_calibration_curve_coverage",
            checks,
        )


def _validate_loso(
    predictions: pd.DataFrame,
    metrics: Mapping[str, Any],
    audit: Mapping[str, Any],
    checks: list[str],
) -> None:
    _check(
        not predictions.duplicated(["expert", "prediction_id"]).any(),
        "loso_identity_unique",
        checks,
    )
    _check(
        np.isfinite(predictions["raw_probability"]).all()
        and predictions["raw_probability"].between(0.0, 1.0).all(),
        "loso_probability_bounds",
        checks,
    )
    _check(
        predictions["dataset_id"].eq(predictions["heldout_source"]).all(),
        "loso_source_identity",
        checks,
    )
    for expert in LOSO_EXPERTS:
        expert_rows = predictions.loc[predictions["expert"].eq(expert)]
        _check(
            expert_rows["heldout_source"].nunique() == LOSO_SOURCE_COUNTS[expert],
            f"{expert}_loso_source_count",
            checks,
        )
        _check(
            set(expert_rows["outer_fold"]) == {0, 1, 2, 3, 4},
            f"{expert}_loso_folds",
            checks,
        )
        records = audit["experts"][expert]
        _check(
            len(records) == LOSO_SOURCE_COUNTS[expert] * 5,
            f"{expert}_loso_audit_count",
            checks,
        )
        _check(
            set(metrics["experts"][expert]) == set(expert_rows["heldout_source"]),
            f"{expert}_loso_metric_sources",
            checks,
        )
        for record in records:
            heldout = record["heldout_source"]
            _check(
                heldout not in record["training_source_ids"],
                f"{expert}_loso_training_excludes_source",
                checks,
            )
            _check(
                record["training_participant_sha256"]
                != record["test_participant_sha256"],
                f"{expert}_loso_fold_hash_separation",
                checks,
            )
            adaptation = record["target_source_adaptation"]
            _check(
                adaptation.get("labels_used") is False,
                f"{expert}_loso_no_target_labels",
                checks,
            )
            if expert in {"activity", "joint"}:
                _check(
                    adaptation.get("test_participants_excluded") is True,
                    f"{expert}_loso_ecdf_test_exclusion",
                    checks,
                )


def _validate_physiology(report: Path, checks: list[str]) -> None:
    payload = _read_json(report / "physiology_uncertainty.json")
    _check(payload.get("source_count") == 1, "physiology_one_source", checks)
    _check(
        payload.get("bootstrap_replicates") == 1000,
        "physiology_bootstrap_count",
        checks,
    )
    _check(
        payload.get("participant_count") == 71, "physiology_participant_count", checks
    )
    for variant in ("raw", "active_calibrated"):
        item = payload["variants"][variant]
        _check(
            item["valid_bootstrap_replicates"] > 950,
            f"physiology_{variant}_valid_bootstrap",
            checks,
        )
        for metric in ("auprc", "auroc", "brier_score"):
            interval = item["intervals_95"][metric]
            _check(
                interval["lower"] <= interval["upper"],
                f"physiology_{variant}_{metric}_interval",
                checks,
            )


def _validate_reader_artifact(report: Path, checks: list[str]) -> None:
    artifact = _read_json(report / "artifact.json")
    manifest = artifact["manifest"]
    snapshot = artifact["snapshot"]
    _check(artifact.get("surface") == "report", "reader_surface", checks)
    _check(snapshot.get("status") == "ready", "reader_status", checks)
    _check(len(snapshot["datasets"]) == 4, "reader_dataset_count", checks)
    _check(
        manifest["blocks"][0].get("type") == "markdown"
        and manifest["blocks"][0].get("body") == f"# {manifest['title']}",
        "reader_visible_title",
        checks,
    )
    _check(
        any(block.get("type") == "chart" for block in manifest["blocks"]),
        "reader_chart_block",
        checks,
    )
    _check(len(manifest.get("sources", [])) == 2, "reader_sources", checks)
    _check(
        all(source.get("query", {}).get("sql") for source in manifest["sources"]),
        "reader_source_queries",
        checks,
    )
    _check(
        sum(len(rows) for rows in snapshot["datasets"].values()) <= 2000,
        "reader_bounded_rows",
        checks,
    )


def _validate_figures(report: Path, checks: list[str]) -> None:
    for path in sorted((report / "figures").glob("*.png")):
        _check(
            path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", f"png_{path.stem}", checks
        )
        _check(path.stat().st_size > 10_000, f"png_size_{path.stem}", checks)


def _variant_prior(variant: str) -> float | None:
    marker = "target_prior_"
    if marker not in variant:
        return None
    return float(variant.split(marker, 1)[1].replace("p", "."))


def _check(condition: bool, name: str, checks: list[str]) -> None:
    if not bool(condition):
        raise ValidationError(f"EVAL-001 validation failed: {name}")
    checks.append(name)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValidationError("EVAL-001 configuration is not a mapping")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _artifact_rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = "".join(
        f"{row['path']}\0{row['bytes']}\0{row['sha256']}\n" for row in rows
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--report-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = (
        args.repository_root.resolve()
        if args.repository_root is not None
        else Path(__file__).resolve().parents[1]
    )
    config = args.config.resolve() if args.config else root / DEFAULT_CONFIG
    report = args.report_dir.resolve() if args.report_dir else root / DEFAULT_REPORT
    try:
        result = validate(root, config, report)
    except (ValidationError, OSError, ValueError, KeyError, TypeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
