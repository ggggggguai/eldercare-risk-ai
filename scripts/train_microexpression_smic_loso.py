from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import time
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.dataset import (
    load_artifact_manifest,
)
from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.metrics import (
    LABEL_NAMES,
    classification_metrics,
    misclassification_rows,
)
from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.model import (
    MHSSATGCNConfig,
    load_model_checkpoint,
)
from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.splits import (
    build_smic_loso_splits,
    validate_smic_loso_splits,
    write_split_manifest,
)
from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.training import (
    TrainingConfig,
    train_loso_fold,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
    return file_sha256(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the SMIC subject-level LOSO baseline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT
        / "configs/training/microexpression_mhssa_tgcn_smic_loso_v1.yaml",
    )
    parser.add_argument(
        "--artifact-manifest",
        type=Path,
        default=PROJECT_ROOT
        / "data/manifests/microexpression/flow_artifact_manifest_smic_hs_classification_v1.jsonl",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=PROJECT_ROOT
        / "data/manifests/microexpression/smic_subject_loso_splits_v1.json",
    )
    parser.add_argument("--run-id")
    parser.add_argument("--fold", action="append")
    parser.add_argument("--max-folds", type=int)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--min-epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--report-dir", type=Path)
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def load_configuration(args: argparse.Namespace) -> tuple[dict[str, Any], MHSSATGCNConfig, TrainingConfig]:
    payload = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model_values = dict(payload["model"])
    training_values = dict(payload["training"])
    if args.hidden_dim is not None:
        model_values["hidden_dim"] = args.hidden_dim
    if args.max_epochs is not None:
        training_values["max_epochs"] = args.max_epochs
    if args.min_epochs is not None:
        training_values["min_epochs"] = args.min_epochs
    if args.patience is not None:
        training_values["patience"] = args.patience
    payload["model"] = model_values
    payload["training"] = training_values
    if args.run_id:
        payload["run_id"] = args.run_id
    return payload, MHSSATGCNConfig.from_mapping(model_values), TrainingConfig.from_mapping(training_values)


def save_confusion_matrix(matrix: list[list[int]], output_path: Path) -> None:
    values = np.asarray(matrix, dtype=np.int64)
    figure, axis = plt.subplots(figsize=(5.2, 4.5))
    image = axis.imshow(values, cmap="Blues")
    axis.set_xticks(range(3), [LABEL_NAMES[index] for index in range(3)])
    axis.set_yticks(range(3), [LABEL_NAMES[index] for index in range(3)])
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title("SMIC subject-level LOSO confusion matrix")
    for row in range(3):
        for column in range(3):
            axis.text(column, row, str(values[row, column]), ha="center", va="center")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def compact_fold_result(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "fold_id": result["fold_id"],
        "status": result["status"],
        "test_subject": result["test_subject"],
        "validation_subjects": result["validation_subjects"],
        "best_epoch": result["best_epoch"],
        "selection_metric": result["selection_metric"],
        "test_used_for_selection": result["test_used_for_selection"],
        "validation": result["validation"],
        "test": {
            "loss": result["test"]["loss"],
            "metrics": result["test"]["metrics"],
        },
        "checkpoint": result["checkpoint"],
        "log": result["log"],
        "duration_seconds": result["duration_seconds"],
        "peak_gpu_memory_bytes": result["peak_gpu_memory_bytes"],
    }


def build_model_card(
    *,
    run_id: str,
    status: str,
    metrics: Mapping[str, Any],
    fold_results: list[Mapping[str, Any]],
    artifact_manifest_sha256: str,
    split_manifest_sha256: str,
    resolved_config_sha256: str,
    output_path: Path,
) -> None:
    rows = []
    for result in fold_results:
        test_metrics = result["test"]["metrics"]
        rows.append(
            f"| {result['test_subject']} | {result['best_epoch']} | "
            f"{test_metrics['uf1']:.4f} | {test_metrics['uar']:.4f} | "
            f"{test_metrics['balanced_accuracy']:.4f} |"
        )
    text = f"""# MHSSA-TGCN SMIC LOSO Model Card

> Run: `{run_id}`  
> Status: `{status}`  
> Release: `experimental / offline_only`  
> Annotation: `estimated / engineering_only`

## Scope

This package contains subject-level LOSO fold checkpoints for SMIC three-class micro-expression classification. It is not a psychological diagnosis model and is not connected to S10, backend alerts, or the psychological observation table.

## Aggregate Metrics

| Metric | Value |
|---|---:|
| Samples | {metrics['sample_count']} |
| UF1 / Macro-F1 | {metrics['uf1']:.6f} |
| UAR / Macro Recall | {metrics['uar']:.6f} |
| Balanced Accuracy | {metrics['balanced_accuracy']:.6f} |
| Accuracy | {metrics['accuracy']:.6f} |

## Fold Results

| Test subject | Best epoch | UF1 | UAR | Balanced Accuracy |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

## Reproducibility

- Artifact manifest SHA-256: `{artifact_manifest_sha256}`
- Split manifest SHA-256: `{split_manifest_sha256}`
- Resolved config SHA-256: `{resolved_config_sha256}`
- Checkpoint selection: validation Macro-F1, then validation loss
- Test fold used for selection: false

## Limitations

- SMIC apex frames are engineering estimates based on aligned motion energy.
- The dataset is small and class/subject distributions are imbalanced.
- Fold checkpoints are evaluation artifacts, not production deployment weights.
- Results do not claim reproduction of the paper's final metrics.
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.config = args.config.resolve()
    args.artifact_manifest = args.artifact_manifest.resolve()
    args.split_manifest = args.split_manifest.resolve()
    source_config, model_config, training_config = load_configuration(args)
    run_id = str(source_config["run_id"])
    output_dir = (args.output_dir or (
        PROJECT_ROOT / "models/mental_health/facial_affect/mhssa_tgcn_smic_loso_v1"
    )).resolve()
    report_dir = (
        args.report_dir or (PROJECT_ROOT / "reports/microexpression" / run_id)
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    records = load_artifact_manifest(args.artifact_manifest)
    artifact_manifest_sha256 = file_sha256(args.artifact_manifest)

    validation_subject_count = int(source_config["split"]["validation_subject_count"])
    split_manifest = build_smic_loso_splits(
        records, validation_subject_count=validation_subject_count
    )
    split_manifest_sha256 = write_split_manifest(split_manifest, args.split_manifest)
    validate_smic_loso_splits(split_manifest)

    resolved_config = {
        **source_config,
        "source_config": {
            "path": args.config.resolve().as_posix(),
            "sha256": file_sha256(args.config),
        },
        "artifact_manifest": {
            "path": args.artifact_manifest.resolve().as_posix(),
            "sha256": artifact_manifest_sha256,
        },
        "split_manifest": {
            "path": args.split_manifest.resolve().as_posix(),
            "sha256": split_manifest_sha256,
        },
        "device": str(device),
    }
    resolved_config_path = output_dir / "resolved_config.json"
    resolved_config_sha256 = atomic_json(resolved_config_path, resolved_config)

    folds = list(split_manifest["folds"])
    if args.fold:
        requested = set(args.fold)
        folds = [fold for fold in folds if fold["fold_id"] in requested or fold["test_subject"] in requested]
        unknown = requested - {
            value
            for fold in folds
            for value in (fold["fold_id"], fold["test_subject"])
        }
        if unknown:
            raise ValueError(f"Unknown folds: {sorted(unknown)}")
    if args.max_folds is not None:
        folds = folds[: args.max_folds]
    if not folds:
        raise ValueError("No folds selected")

    started = time.perf_counter()
    fold_results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for fold_index, fold in enumerate(folds):
        print(
            json.dumps(
                {
                    "event": "fold_start",
                    "fold_id": fold["fold_id"],
                    "index": fold_index + 1,
                    "total": len(folds),
                }
            ),
            flush=True,
        )
        try:
            result = train_loso_fold(
                records=records,
                fold=fold,
                model_config=model_config,
                training_config=training_config,
                device=device,
                output_dir=output_dir,
                artifact_manifest_path=args.artifact_manifest,
                artifact_manifest_sha256=artifact_manifest_sha256,
                split_manifest_path=args.split_manifest,
                split_manifest_sha256=split_manifest_sha256,
                config_path=resolved_config_path,
                config_sha256=resolved_config_sha256,
                fold_index=split_manifest["subjects"].index(fold["test_subject"]),
            )
            fold_results.append(result)
            atomic_json(report_dir / f"{fold['fold_id']}.json", compact_fold_result(result))
            print(
                json.dumps(
                    {
                        "event": "fold_complete",
                        "fold_id": fold["fold_id"],
                        "best_epoch": result["best_epoch"],
                        "test_uf1": result["test"]["metrics"]["uf1"],
                        "test_uar": result["test"]["metrics"]["uar"],
                    }
                ),
                flush=True,
            )
        except Exception as exc:
            failure = {
                "fold_id": str(fold["fold_id"]),
                "error": f"{type(exc).__name__}: {exc}",
            }
            failures.append(failure)
            atomic_json(report_dir / f"{fold['fold_id']}_failed.json", failure)
            print(json.dumps({"event": "fold_failed", **failure}), flush=True)

    labels: list[int] = []
    predictions: list[int] = []
    probabilities: list[list[float]] = []
    sample_ids: list[str] = []
    subject_ids: list[str] = []
    for result in fold_results:
        test = result["test"]
        labels.extend(test["labels"])
        predictions.extend(test["predictions"])
        probabilities.extend(test["probabilities"])
        sample_ids.extend(test["sample_ids"])
        subject_ids.extend(test["subject_ids"])
    aggregate_metrics = (
        classification_metrics(labels, predictions) if labels else None
    )
    misclassified = (
        misclassification_rows(
            sample_ids=sample_ids,
            subject_ids=subject_ids,
            y_true=labels,
            y_pred=predictions,
            probabilities=probabilities,
        )
        if labels
        else []
    )
    pair_counts = Counter(
        f"{row['true_label_name']}->{row['predicted_label_name']}"
        for row in misclassified
    )
    predictions_path = report_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in zip(
            sample_ids,
            subject_ids,
            labels,
            predictions,
            probabilities,
            strict=True,
        ):
            handle.write(
                json.dumps(
                    {
                        "sample_id": row[0],
                        "subject_id": row[1],
                        "true_label": row[2],
                        "predicted_label": row[3],
                        "probabilities": row[4],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    failure_analysis = {
        "misclassified_count": len(misclassified),
        "error_rate": len(misclassified) / max(len(labels), 1),
        "confusion_pairs": dict(sorted(pair_counts.items())),
        "rows": misclassified,
        "training_failures": failures,
    }
    atomic_json(report_dir / "failure_analysis.json", failure_analysis)

    all_folds_completed = len(fold_results) == len(split_manifest["folds"]) and not failures
    status = "completed" if all_folds_completed else "partial_or_failed"
    report = {
        "schema_version": "smic_loso_result_v1",
        "task_id": "MODEL-ME-002",
        "run_id": run_id,
        "status": status,
        "device": str(device),
        "duration_seconds": time.perf_counter() - started,
        "fold_count_requested": len(folds),
        "fold_count_completed": len(fold_results),
        "full_protocol_fold_count": len(split_manifest["folds"]),
        "sample_count_evaluated": len(labels),
        "aggregate_metrics": aggregate_metrics,
        "folds": [compact_fold_result(result) for result in fold_results],
        "failures": failures,
        "failure_analysis": {
            "path": (report_dir / "failure_analysis.json").resolve().as_posix(),
            "sha256": file_sha256(report_dir / "failure_analysis.json"),
        },
        "predictions": {
            "path": predictions_path.resolve().as_posix(),
            "sha256": file_sha256(predictions_path),
        },
        "artifact_manifest_sha256": artifact_manifest_sha256,
        "split_manifest_sha256": split_manifest_sha256,
        "resolved_config_sha256": resolved_config_sha256,
        "selection_authority": "validation_only",
        "test_used_for_selection": False,
        "frame_annotation_source": "estimated",
        "evaluation_scope": "engineering_only",
        "paper_reproduction_claim": False,
    }
    metrics_path = report_dir / "metrics.json"
    metrics_sha256 = atomic_json(metrics_path, report)
    if aggregate_metrics is not None:
        save_confusion_matrix(
            aggregate_metrics["confusion_matrix"], report_dir / "confusion_matrix.png"
        )

    checkpoints = [result["checkpoint"] for result in fold_results]
    package_manifest = {
        "schema_version": "mhssa_tgcn_smic_loso_package_v1",
        "model_id": "mhssa_tgcn_smic_loso_v1",
        "run_id": run_id,
        "status": "experimental" if fold_results else "failed",
        "protocol_status": status,
        "checkpoints": checkpoints,
        "artifact_manifest_sha256": artifact_manifest_sha256,
        "split_manifest_sha256": split_manifest_sha256,
        "resolved_config_sha256": resolved_config_sha256,
        "metrics": {
            "path": metrics_path.resolve().as_posix(),
            "sha256": metrics_sha256,
        },
        "test_used_for_selection": False,
    }
    package_manifest_path = output_dir / "manifest.json"
    atomic_json(package_manifest_path, package_manifest)
    sums_path = output_dir / "SHA256SUMS"
    sums_path.write_text(
        "\n".join(
            f"{checkpoint['sha256']}  {Path(checkpoint['path']).relative_to(output_dir).as_posix()}"
            for checkpoint in checkpoints
        )
        + ("\n" if checkpoints else ""),
        encoding="ascii",
    )
    if aggregate_metrics is not None:
        build_model_card(
            run_id=run_id,
            status=status,
            metrics=aggregate_metrics,
            fold_results=fold_results,
            artifact_manifest_sha256=artifact_manifest_sha256,
            split_manifest_sha256=split_manifest_sha256,
            resolved_config_sha256=resolved_config_sha256,
            output_path=output_dir / "MODEL_CARD.md",
        )

    verification_issues: list[str] = []
    for result in fold_results:
        checkpoint_path = Path(result["checkpoint"]["path"])
        if file_sha256(checkpoint_path) != result["checkpoint"]["sha256"]:
            verification_issues.append(f"hash mismatch: {result['fold_id']}")
            continue
        _, checkpoint = load_model_checkpoint(str(checkpoint_path), map_location="cpu")
        if checkpoint.get("test_used_for_selection") is not False:
            verification_issues.append(f"test selection violation: {result['fold_id']}")
        if checkpoint["artifact_manifest"]["sha256"] != artifact_manifest_sha256:
            verification_issues.append(f"artifact manifest mismatch: {result['fold_id']}")
        if checkpoint["split_manifest"]["sha256"] != split_manifest_sha256:
            verification_issues.append(f"split manifest mismatch: {result['fold_id']}")
        if checkpoint["config"]["sha256"] != resolved_config_sha256:
            verification_issues.append(f"config mismatch: {result['fold_id']}")
    verification = {
        "status": "pass" if not verification_issues and fold_results else "fail",
        "checkpoint_count": len(fold_results),
        "issues": verification_issues,
    }
    atomic_json(report_dir / "checkpoint_audit.json", verification)
    print(
        json.dumps(
            {
                "event": "run_complete",
                "status": status,
                "folds": len(fold_results),
                "samples": len(labels),
                "metrics": aggregate_metrics,
                "verification": verification,
                "report": metrics_path.resolve().as_posix(),
            },
            ensure_ascii=True,
            indent=2,
        ),
        flush=True,
    )
    if failures or verification["status"] != "pass":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
