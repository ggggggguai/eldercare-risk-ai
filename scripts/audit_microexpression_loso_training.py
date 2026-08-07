from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any

import torch

from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.dataset import (
    load_artifact_manifest,
)
from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.metrics import (
    classification_metrics,
)
from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.model import (
    load_model_checkpoint,
)
from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.splits import (
    validate_smic_loso_splits,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit MODEL-ME-001/002 artifacts.")
    parser.add_argument(
        "--package-manifest",
        type=Path,
        default=PROJECT_ROOT
        / "models/mental_health/facial_affect/mhssa_tgcn_smic_loso_v1/manifest.json",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=PROJECT_ROOT
        / "data/manifests/microexpression/smic_subject_loso_splits_v1.json",
    )
    parser.add_argument(
        "--artifact-manifest",
        type=Path,
        default=PROJECT_ROOT
        / "data/manifests/microexpression/flow_artifact_manifest_smic_hs_classification_v1.jsonl",
    )
    parser.add_argument(
        "--smoke-report",
        type=Path,
        default=PROJECT_ROOT
        / "data/manifests/microexpression/model_smoke_test_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "data/manifests/microexpression/smic_loso_training_audit_v1.json",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def add_issue(issues: list[dict[str, Any]], code: str, **detail: Any) -> None:
    issues.append({"code": code, **detail})


def main() -> int:
    args = parse_args()
    args.package_manifest = args.package_manifest.resolve()
    args.split_manifest = args.split_manifest.resolve()
    args.artifact_manifest = args.artifact_manifest.resolve()
    args.smoke_report = args.smoke_report.resolve()
    args.output = args.output.resolve()
    issues: list[dict[str, Any]] = []

    package = load_json(args.package_manifest)
    splits = load_json(args.split_manifest)
    smoke = load_json(args.smoke_report)
    records = load_artifact_manifest(args.artifact_manifest)
    validate_smic_loso_splits(splits)
    artifact_by_id = {str(record["sample_id"]): record for record in records}
    if smoke.get("status") != "pass":
        add_issue(issues, "model_smoke_not_passed")
    if package.get("protocol_status") != "completed":
        add_issue(issues, "protocol_not_completed")
    if package.get("test_used_for_selection") is not False:
        add_issue(issues, "package_test_selection_violation")
    if package.get("artifact_manifest_sha256") != file_sha256(args.artifact_manifest):
        add_issue(issues, "package_artifact_manifest_hash_mismatch")
    if package.get("split_manifest_sha256") != file_sha256(args.split_manifest):
        add_issue(issues, "package_split_manifest_hash_mismatch")

    metrics_path = Path(str(package["metrics"]["path"]))
    if not metrics_path.is_file() or file_sha256(metrics_path) != package["metrics"]["sha256"]:
        add_issue(issues, "metrics_hash_mismatch")
        metrics_report: dict[str, Any] = {}
    else:
        metrics_report = load_json(metrics_path)
    if metrics_report.get("status") != "completed":
        add_issue(issues, "metrics_protocol_not_completed")
    if metrics_report.get("test_used_for_selection") is not False:
        add_issue(issues, "metrics_test_selection_violation")
    if metrics_report.get("fold_count_completed") != 16:
        add_issue(issues, "unexpected_completed_fold_count")
    if metrics_report.get("sample_count_evaluated") != 164:
        add_issue(issues, "unexpected_evaluated_sample_count")
    if metrics_report.get("failures"):
        add_issue(issues, "training_failures_present")

    predictions_path = Path(str(metrics_report.get("predictions", {}).get("path", "")))
    predictions = []
    if predictions_path.is_file():
        predictions = [
            json.loads(line)
            for line in predictions_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if file_sha256(predictions_path) != metrics_report["predictions"]["sha256"]:
            add_issue(issues, "predictions_hash_mismatch")
    else:
        add_issue(issues, "predictions_missing")
    prediction_ids = [str(row["sample_id"]) for row in predictions]
    if len(prediction_ids) != 164 or len(set(prediction_ids)) != 164:
        add_issue(issues, "predictions_not_unique_complete", count=len(prediction_ids))
    if set(prediction_ids) != set(artifact_by_id):
        add_issue(issues, "prediction_artifact_id_set_mismatch")

    test_subject_by_sample: dict[str, str] = {}
    for fold in splits["folds"]:
        for sample_id in fold["test_sample_ids"]:
            test_subject_by_sample[str(sample_id)] = str(fold["test_subject"])
    for row in predictions:
        sample_id = str(row["sample_id"])
        if row["subject_id"] != test_subject_by_sample.get(sample_id):
            add_issue(issues, "prediction_test_subject_mismatch", sample_id=sample_id)
        if sample_id in artifact_by_id and int(row["true_label"]) != int(artifact_by_id[sample_id]["label"]):
            add_issue(issues, "prediction_true_label_mismatch", sample_id=sample_id)

    if predictions:
        recomputed = classification_metrics(
            [int(row["true_label"]) for row in predictions],
            [int(row["predicted_label"]) for row in predictions],
        )
        reported = metrics_report["aggregate_metrics"]
        for metric in ("accuracy", "macro_f1", "uf1", "macro_recall", "uar", "balanced_accuracy"):
            if not math.isclose(
                float(recomputed[metric]),
                float(reported[metric]),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                add_issue(issues, "metric_recompute_mismatch", metric=metric)
        if recomputed["confusion_matrix"] != reported["confusion_matrix"]:
            add_issue(issues, "confusion_matrix_recompute_mismatch")
    else:
        recomputed = None

    split_by_id = {str(fold["fold_id"]): fold for fold in splits["folds"]}
    checkpoint_fold_ids: list[str] = []
    checkpoint_bytes = 0
    for checkpoint_record in package.get("checkpoints", []):
        checkpoint_path = Path(str(checkpoint_record["path"]))
        if not checkpoint_path.is_file():
            add_issue(issues, "checkpoint_missing", path=str(checkpoint_path))
            continue
        checkpoint_bytes += checkpoint_path.stat().st_size
        if file_sha256(checkpoint_path) != checkpoint_record["sha256"]:
            add_issue(issues, "checkpoint_hash_mismatch", path=str(checkpoint_path))
            continue
        model, checkpoint = load_model_checkpoint(str(checkpoint_path), map_location="cpu")
        fold_id = str(checkpoint["fold_id"])
        checkpoint_fold_ids.append(fold_id)
        if fold_id not in split_by_id:
            add_issue(issues, "checkpoint_unknown_fold", fold_id=fold_id)
            continue
        fold = split_by_id[fold_id]
        if checkpoint.get("test_used_for_selection") is not False:
            add_issue(issues, "checkpoint_test_selection_violation", fold_id=fold_id)
        if checkpoint.get("selection_metric") != "validation_macro_f1_then_validation_loss":
            add_issue(issues, "checkpoint_selection_metric_mismatch", fold_id=fold_id)
        if checkpoint.get("test_subject") != fold["test_subject"]:
            add_issue(issues, "checkpoint_test_subject_mismatch", fold_id=fold_id)
        if sorted(checkpoint.get("validation_subjects", [])) != sorted(fold["validation_subjects"]):
            add_issue(issues, "checkpoint_validation_subject_mismatch", fold_id=fold_id)
        if checkpoint["artifact_manifest"]["sha256"] != file_sha256(args.artifact_manifest):
            add_issue(issues, "checkpoint_artifact_hash_mismatch", fold_id=fold_id)
        if checkpoint["split_manifest"]["sha256"] != file_sha256(args.split_manifest):
            add_issue(issues, "checkpoint_split_hash_mismatch", fold_id=fold_id)
        if not all(torch.isfinite(value).all().item() for value in model.state_dict().values()):
            add_issue(issues, "checkpoint_non_finite_weights", fold_id=fold_id)

    if sorted(checkpoint_fold_ids) != sorted(split_by_id):
        add_issue(issues, "checkpoint_fold_set_mismatch", folds=checkpoint_fold_ids)
    if len(package.get("checkpoints", [])) != 16:
        add_issue(issues, "unexpected_checkpoint_count")

    reported_folds = {str(fold["fold_id"]): fold for fold in metrics_report.get("folds", [])}
    for fold_id, fold in reported_folds.items():
        log_path = Path(str(fold["log"]["path"]))
        if not log_path.is_file() or file_sha256(log_path) != fold["log"]["sha256"]:
            add_issue(issues, "epoch_log_hash_mismatch", fold_id=fold_id)
            continue
        epoch_rows = [
            json.loads(line)
            for line in log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not epoch_rows:
            add_issue(issues, "empty_epoch_log", fold_id=fold_id)
            continue
        if any("test" in key.lower() for row in epoch_rows for key in row):
            add_issue(issues, "test_data_present_in_epoch_log", fold_id=fold_id)
        if any(row.get("selection_source") != "validation_only" for row in epoch_rows):
            add_issue(issues, "non_validation_selection_source", fold_id=fold_id)
        best_rows = [row for row in epoch_rows if row["epoch"] == fold["best_epoch"]]
        if len(best_rows) != 1 or best_rows[0].get("checkpoint_improved") is not True:
            add_issue(issues, "best_epoch_not_checkpointed", fold_id=fold_id)

    model_dir = args.package_manifest.parent
    for required in (model_dir / "MODEL_CARD.md", model_dir / "SHA256SUMS"):
        if not required.is_file():
            add_issue(issues, "model_package_file_missing", path=str(required))
    confusion_path = metrics_path.parent / "confusion_matrix.png"
    failure_analysis_path = Path(str(metrics_report.get("failure_analysis", {}).get("path", "")))
    if not confusion_path.is_file():
        add_issue(issues, "confusion_matrix_missing")
    if not failure_analysis_path.is_file():
        add_issue(issues, "failure_analysis_missing")

    result = {
        "schema_version": "smic_loso_training_audit_v1",
        "task_ids": ["MODEL-ME-001", "MODEL-ME-002"],
        "status": "pass" if not issues else "fail",
        "issue_count": len(issues),
        "issues": issues,
        "sample_count": len(predictions),
        "subject_count": len(splits["subjects"]),
        "fold_count": len(splits["folds"]),
        "checkpoint_count": len(package.get("checkpoints", [])),
        "checkpoint_bytes": checkpoint_bytes,
        "metrics": recomputed,
        "hashes": {
            "artifact_manifest": file_sha256(args.artifact_manifest),
            "split_manifest": file_sha256(args.split_manifest),
            "package_manifest": file_sha256(args.package_manifest),
            "metrics": file_sha256(metrics_path) if metrics_path.is_file() else None,
            "smoke_report": file_sha256(args.smoke_report),
        },
        "invariants": {
            "subject_level_loso": True,
            "test_used_for_selection": False,
            "all_checkpoints_strict_loadable": not any(
                issue["code"].startswith("checkpoint_") for issue in issues
            ),
            "all_predictions_recomputed": recomputed is not None,
            "backend_in_scope": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    output_sha256 = file_sha256(args.output)
    print(
        json.dumps(
            {**result, "output": args.output.as_posix(), "output_sha256": output_sha256},
            ensure_ascii=True,
            indent=2,
        )
    )
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
