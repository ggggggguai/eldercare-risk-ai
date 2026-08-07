"""Freeze V3.3 Platt calibration, then execute the official Test exactly once."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader

from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.model import (
    CognitiveChangeClueModel,
)

try:
    from .common import ALGORITHM_ROOT, sha256_file, workspace_relative
    from .dataset import (
        EXPECTED_INPUT_SHA256,
        MODALITY_ORDER,
        CognitiveFeatureDataset,
        MocaStatistics,
        cognitive_collate,
        default_input_paths,
        verify_frozen_input_hashes,
    )
    from .evaluate import COMBINATION_MODALITIES, predict_validation, subject_metrics
    from .train import configure_determinism, load_and_validate_config, write_json, write_jsonl
except ImportError:
    from common import ALGORITHM_ROOT, sha256_file, workspace_relative
    from dataset import (  # type: ignore[no-redef]
        EXPECTED_INPUT_SHA256,
        MODALITY_ORDER,
        CognitiveFeatureDataset,
        MocaStatistics,
        cognitive_collate,
        default_input_paths,
        verify_frozen_input_hashes,
    )
    from evaluate import COMBINATION_MODALITIES, predict_validation, subject_metrics
    from train import configure_determinism, load_and_validate_config, write_json, write_jsonl


MODEL_VERSION = "cognitive-mm-v3.3.0"
EXPECTED_CHECKPOINT_SHA256 = "b7077898ebb56218dd2a1b0a975efa7b9ca3422e1711ab62223598c5c06b73f8"
REPORT_ROOT = ALGORITHM_ROOT / "reports" / "cognitive_change_clue" / "v3.3.0"
DEFAULT_CHECKPOINT = (
    REPORT_ROOT / "runs" / "COG-20260804-002" / "best_checkpoint.pt"
)
DEFAULT_RUN_ID = "COG-20260804-003"
CALIBRATION_PATH = REPORT_ROOT / "calibration.json"
CALIBRATION_HASH_PATH = REPORT_ROOT / "calibration.sha256"
TEST_GUARD_PATH = REPORT_ROOT / "official_test_execution.json"


class CognitiveCalibrationError(RuntimeError):
    pass


def fit_platt_calibration(
    records: Sequence[Mapping[str, Any]], *, checkpoint_sha256: str
) -> dict[str, Any]:
    if not records:
        raise CognitiveCalibrationError("Validation calibration records are empty")
    logits = np.asarray(
        [float(row["hc_vs_non_hc"]["raw_logit"]) for row in records], dtype=np.float64
    ).reshape(-1, 1)
    labels = np.asarray(
        [int(row["hc_vs_non_hc"]["label"]) for row in records], dtype=np.int64
    )
    if not np.isfinite(logits).all() or sorted(np.unique(labels).tolist()) != [0, 1]:
        raise CognitiveCalibrationError("Validation calibration data is invalid")
    calibrator = LogisticRegression(
        solver="lbfgs", C=1e6, max_iter=1000, fit_intercept=True
    )
    calibrator.fit(logits, labels)
    a = float(calibrator.coef_[0, 0])
    b = float(calibrator.intercept_[0])
    if not math.isfinite(a) or not math.isfinite(b):
        raise CognitiveCalibrationError("Platt parameters are not finite")
    return {
        "schema_version": "platt_calibration_v1",
        "input": "task_raw_logit",
        "a": a,
        "b": b,
        "validation_complete_task_count": int(len(records)),
        "checkpoint_sha256": checkpoint_sha256,
    }


def calibrated_probability(raw_logit: float, calibration: Mapping[str, Any]) -> float:
    value = float(calibration["a"]) * float(raw_logit) + float(calibration["b"])
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def calibrated_subject_report(
    records: Sequence[Mapping[str, Any]], calibration: Mapping[str, Any]
) -> dict[str, Any]:
    base = subject_metrics(records)
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[str(row["subject_id"])].append(row)
    labels: list[int] = []
    probabilities: list[float] = []
    subject_rows: list[dict[str, Any]] = []
    for subject_id, rows in sorted(grouped.items()):
        subject_labels = {int(row["hc_vs_non_hc"]["label"]) for row in rows}
        if len(subject_labels) != 1:
            raise CognitiveCalibrationError(f"Test label conflict for {subject_id}")
        task_probabilities = [
            calibrated_probability(row["hc_vs_non_hc"]["raw_logit"], calibration)
            for row in rows
        ]
        label = next(iter(subject_labels))
        probability = float(np.mean(task_probabilities))
        labels.append(label)
        probabilities.append(probability)
        subject_rows.append(
            {
                "subject_id": subject_id,
                "diagnosis_label": str(rows[0]["diagnosis_label"]),
                "task_count": len(rows),
                "label": label,
                "mean_calibrated_probability": probability,
                "prediction_at_0_5": int(probability >= 0.5),
            }
        )
    predictions = [int(value >= 0.5) for value in probabilities]
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    main = {
        "subject_count": len(labels),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "balanced_accuracy_at_0_5": float(balanced_accuracy_score(labels, predictions)),
        "sensitivity_at_0_5": float(tp / (tp + fn)) if tp + fn else None,
        "specificity_at_0_5": float(tn / (tn + fp)) if tn + fp else None,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "aggregation": "calibrate_each_task_then_mean_probability_per_subject",
    }
    return {
        "task_count": len(records),
        "subject_count": len(subject_rows),
        "hc_vs_non_hc": main,
        "mci_vs_hc": base["mci_vs_hc"],
        "ad_mci_hc": base["ad_mci_hc"],
        "moca": base["moca"],
        "subjects": subject_rows,
    }


def _loader(dataset: CognitiveFeatureDataset, batch_size: int) -> DataLoader[Any]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        collate_fn=cognitive_collate,
    )


def _load_model(checkpoint_path: Path, device: torch.device) -> tuple[CognitiveChangeClueModel, Mapping[str, Any], MocaStatistics]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != "cognitive_training_checkpoint_v1":
        raise CognitiveCalibrationError("checkpoint schema version mismatch")
    if payload.get("input_hashes") != EXPECTED_INPUT_SHA256:
        raise CognitiveCalibrationError("checkpoint frozen input hashes mismatch")
    structure = payload["model_structure"]
    model = CognitiveChangeClueModel(
        projection_dim=int(structure["projection_dim"]),
        fusion_hidden_dims=tuple(structure["fusion_hidden_dims"]),
        dropout=float(structure["dropout"]),
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval().to(device)
    stats = payload["moca_statistics"]
    moca = MocaStatistics(
        mean=float(stats["mean"]),
        std=float(stats["std"]),
        subject_count=int(stats["subject_count"]),
    )
    return model, payload, moca


def _coverage_report(dataset: CognitiveFeatureDataset) -> dict[str, Any]:
    combinations: Counter[str] = Counter()
    asr_statuses: Counter[str] = Counter()
    subjects_with_complete: set[str] = set()
    all_subjects: set[str] = set()
    for record in dataset.records:
        available = [
            name for name, missing in zip(MODALITY_ORDER, record["missing_mask"]) if not missing
        ]
        combination = "+".join(name[0].upper() for name in available) or "none"
        combinations[combination] += 1
        asr_statuses[str(record.get("asr_status", "unknown"))] += 1
        all_subjects.add(str(record["subject_id"]))
        if len(available) == 3:
            subjects_with_complete.add(str(record["subject_id"]))
    return {
        "task_count": len(dataset),
        "subject_count": len(all_subjects),
        "modality_combination_task_counts": dict(sorted(combinations.items())),
        "asr_status_task_counts": dict(sorted(asr_statuses.items())),
        "subjects_with_complete_task": len(subjects_with_complete),
        "subjects_without_complete_task": len(all_subjects - subjects_with_complete),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_and_validate_config(args.config)
    configure_determinism(config)
    input_hashes = verify_frozen_input_hashes(default_input_paths(), EXPECTED_INPUT_SHA256)
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise CognitiveCalibrationError(
            f"checkpoint SHA-256 mismatch: {checkpoint_sha256}"
        )
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    model, checkpoint, moca_statistics = _load_model(checkpoint_path, device)
    run_dir = REPORT_ROOT / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if CALIBRATION_PATH.exists() or CALIBRATION_HASH_PATH.exists():
        raise CognitiveCalibrationError("frozen calibration already exists; refusing to overwrite")
    validation = CognitiveFeatureDataset(
        split_name="validation", complete_only=True, verify_hashes=True
    )
    validation_predictions = predict_validation(
        model,
        _loader(validation, int(config["training"]["micro_batch_size"])),
        device=device,
        moca_statistics=moca_statistics,
        checkpoint_sha256=checkpoint_sha256,
    )
    calibration = fit_platt_calibration(
        validation_predictions, checkpoint_sha256=checkpoint_sha256
    )
    write_json(CALIBRATION_PATH, calibration)
    calibration_sha256 = sha256_file(CALIBRATION_PATH)
    CALIBRATION_HASH_PATH.write_text(
        f"{calibration_sha256}  calibration.json\n", encoding="ascii"
    )
    write_jsonl(run_dir / "validation_calibration_predictions.jsonl", validation_predictions)
    calibration_report = {
        "schema_version": "cognitive_calibration_report_v1",
        "model_version": MODEL_VERSION,
        "run_id": args.run_id,
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "checkpoint": workspace_relative(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "calibration": calibration,
        "calibration_sha256": calibration_sha256,
        "input_hashes": input_hashes,
        "validation_subject_metrics_uncalibrated": subject_metrics(validation_predictions),
        "validation_subject_metrics_calibrated": calibrated_subject_report(
            validation_predictions, calibration
        ),
        "official_test_read": False,
    }
    write_json(run_dir / "calibration_report.json", calibration_report)

    if TEST_GUARD_PATH.exists():
        raise CognitiveCalibrationError("official Test execution guard already exists")
    guard = {
        "schema_version": "cognitive_official_test_execution_v1",
        "run_id": args.run_id,
        "status": "started",
        "started_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "checkpoint_sha256": checkpoint_sha256,
        "calibration_sha256": calibration_sha256,
        "attempt_number": 1,
        "policy": "official Test is evaluated once and never used for tuning",
    }
    write_json(TEST_GUARD_PATH, guard)

    try:
        test_all = CognitiveFeatureDataset(
            split_name="test", complete_only=False, verify_hashes=True, evaluation_only=True
        )
        test_complete = CognitiveFeatureDataset(
            split_name="test", complete_only=True, verify_hashes=True, evaluation_only=True
        )
        loader = _loader(test_complete, int(config["training"]["micro_batch_size"]))
        test_predictions = predict_validation(
            model,
            loader,
            device=device,
            moca_statistics=moca_statistics,
            checkpoint_sha256=checkpoint_sha256,
        )
        for row in test_predictions:
            row["schema_version"] = "cognitive_test_prediction_v1"
            row["hc_vs_non_hc"]["calibrated_probability"] = calibrated_probability(
                row["hc_vs_non_hc"]["raw_logit"], calibration
            )
        write_jsonl(run_dir / "test_predictions.jsonl", test_predictions)
        primary = calibrated_subject_report(test_predictions, calibration)

        ablations: dict[str, Any] = {}
        for combination in COMBINATION_MODALITIES:
            rows = predict_validation(
                model,
                _loader(test_complete, int(config["training"]["micro_batch_size"])),
                device=device,
                moca_statistics=moca_statistics,
                forced_combination=combination,
                checkpoint_sha256=checkpoint_sha256,
            )
            ablations[combination] = calibrated_subject_report(rows, calibration)
        ablation_report = {
            "schema_version": "cognitive_test_modality_ablation_v1",
            "checkpoint_sha256": checkpoint_sha256,
            "calibration_sha256": calibration_sha256,
            "comparison_population": "official_test_original_complete_audio_text_face_tasks",
            "combinations": ablations,
            "used_for_model_selection_or_tuning": False,
        }
        write_json(run_dir / "test_modality_ablation.json", ablation_report)
        report = {
            "schema_version": "cognitive_test_subject_metrics_v1",
            "model_version": MODEL_VERSION,
            "run_id": args.run_id,
            "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
            "official_test_execution": "first_and_only",
            "used_for_training_selection_tuning_or_fallback": False,
            "checkpoint_sha256": checkpoint_sha256,
            "calibration_sha256": calibration_sha256,
            "input_hashes": input_hashes,
            "coverage": _coverage_report(test_all),
            "complete_task_subject_metrics": primary,
            "modality_ablation_report": workspace_relative(
                run_dir / "test_modality_ablation.json"
            ),
        }
        write_json(run_dir / "test_subject_metrics.json", report)
        write_json(REPORT_ROOT / "test_subject_metrics.json", report)
        guard.update(
            {
                "status": "completed",
                "completed_at": datetime.now(timezone.utc).astimezone().isoformat(),
                "test_subject_metrics": workspace_relative(
                    run_dir / "test_subject_metrics.json"
                ),
                "test_subject_metrics_sha256": sha256_file(
                    run_dir / "test_subject_metrics.json"
                ),
            }
        )
        write_json(TEST_GUARD_PATH, guard)
        calibration_report["official_test_read"] = True
        calibration_report["official_test_execution"] = "first_and_only"
        calibration_report["official_test_report"] = workspace_relative(
            run_dir / "test_subject_metrics.json"
        )
        write_json(run_dir / "calibration_report.json", calibration_report)
        return report
    except Exception as exc:
        guard.update(
            {
                "status": "failed",
                "failed_at": datetime.now(timezone.utc).astimezone().isoformat(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        write_json(TEST_GUARD_PATH, guard)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument(
        "--config",
        default=str(ALGORITHM_ROOT / "configs" / "modules" / "cognitive_change_clue_v3_3.yaml"),
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> int:
    report = run(parse_args())
    print(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CognitiveCalibrationError",
    "calibrated_probability",
    "calibrated_subject_report",
    "fit_platt_calibration",
    "main",
    "run",
]
