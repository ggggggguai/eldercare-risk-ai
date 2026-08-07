"""Reproducible official-Train-only V3.4 cross-validation training."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import yaml
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.model import (
    CognitiveChangeClueModel,
)

try:
    from .build_subject_cv_v34 import DEFAULT_AUDIT_PATH, DEFAULT_OUTPUT_PATH
    from .common import ALGORITHM_ROOT, WORKSPACE_ROOT, sha256_file, workspace_relative
    from .dataset import (
        MODALITY_ORDER,
        MocaStatistics,
        TrainingWeights,
        cognitive_collate,
        compute_moca_statistics,
        compute_training_weights,
    )
    from .evaluate import (
        checkpoint_is_better,
        filter_valid_rows,
        move_batch_to_device,
        predict_validation,
        subject_metrics,
    )
    from .train import (
        _new_grad_scaler,
        append_jsonl,
        atomic_torch_save,
        build_masking_plan,
        capture_rng_state,
        configure_determinism,
        recover_grad_scaler_after_recorded_overflow,
        restore_rng_state,
        train_one_epoch,
        write_json,
        write_jsonl,
    )
    from .v34_data import (
        SubjectFeatureDataset,
        build_fold_datasets,
        load_fold,
        load_official_train_pool,
    )
    from .v34_metrics import (
        aggregate_subjects,
        binary_metrics,
        sigmoid,
        temporary_platt_evaluation,
    )
except ImportError:
    from build_subject_cv_v34 import (  # type: ignore[no-redef]
        DEFAULT_AUDIT_PATH,
        DEFAULT_OUTPUT_PATH,
    )
    from common import (  # type: ignore[no-redef]
        ALGORITHM_ROOT,
        WORKSPACE_ROOT,
        sha256_file,
        workspace_relative,
    )
    from dataset import (  # type: ignore[no-redef]
        MODALITY_ORDER,
        MocaStatistics,
        TrainingWeights,
        cognitive_collate,
        compute_moca_statistics,
        compute_training_weights,
    )
    from evaluate import (  # type: ignore[no-redef]
        checkpoint_is_better,
        filter_valid_rows,
        move_batch_to_device,
        predict_validation,
        subject_metrics,
    )
    from train import (  # type: ignore[no-redef]
        _new_grad_scaler,
        append_jsonl,
        atomic_torch_save,
        build_masking_plan,
        capture_rng_state,
        configure_determinism,
        recover_grad_scaler_after_recorded_overflow,
        restore_rng_state,
        train_one_epoch,
        write_json,
        write_jsonl,
    )
    from v34_data import (  # type: ignore[no-redef]
        SubjectFeatureDataset,
        build_fold_datasets,
        load_fold,
        load_official_train_pool,
    )
    from v34_metrics import (  # type: ignore[no-redef]
        aggregate_subjects,
        binary_metrics,
        sigmoid,
        temporary_platt_evaluation,
    )


CONFIG_PATH = ALGORITHM_ROOT / "configs" / "modules" / "cognitive_change_clue_v3_4.yaml"
REPORT_ROOT = ALGORITHM_ROOT / "reports" / "cognitive_change_clue" / "v3.4.0"
PREDICTION_FILES = {
    "inner_validation": "inner_validation_raw_predictions.jsonl",
    "inner_calibration": "inner_calibration_raw_predictions.jsonl",
    "outer_evaluation": "outer_evaluation_raw_predictions.jsonl",
}
PREDICTION_FIELDS = {
    "schema_version",
    "candidate_id",
    "outer_fold",
    "split_role",
    "sample_id",
    "subject_id",
    "diagnosis",
    "label_hc_vs_non_hc",
    "raw_logit",
    "audio_logit",
    "text_logit",
    "face_logit",
    "q_audio",
    "q_text",
    "q_face",
    "audio_missing_mask",
    "text_missing_mask",
    "face_missing_mask",
}
CANDIDATES = {
    "V34-A0": {
        "modalities": ("audio", "text", "face"),
        "validation_modalities": ("audio", "text", "face"),
        "selection_modalities": ("audio", "text", "face"),
        "heads": "four_head",
    },
    "V34-A1": {
        "modalities": ("audio", "text"),
        "validation_modalities": ("audio", "text", "face"),
        "selection_modalities": ("audio", "text", "face"),
        "heads": "four_head",
    },
    "V34-A2": {
        "modalities": ("audio", "text"),
        "validation_modalities": ("audio", "text", "face"),
        "selection_modalities": ("audio", "text", "face"),
        "heads": "main_head",
    },
    "V34-A3-Audio": {
        "modalities": ("audio",),
        "validation_modalities": ("audio",),
        "selection_modalities": ("audio",),
        "heads": "main_head",
    },
    "V34-A3-Text": {
        "modalities": ("text",),
        "validation_modalities": ("text",),
        "selection_modalities": ("text",),
        "heads": "main_head",
    },
    "V34-A3": {
        "kind": "late_fusion",
        "modalities": ("audio", "text"),
        "validation_modalities": ("audio", "text", "face"),
        "selection_modalities": ("audio", "text", "face"),
        "heads": "main_head",
    },
    "V34-Face": {
        "modalities": ("face",),
        "validation_modalities": ("face",),
        "selection_modalities": ("face",),
        "heads": "main_head",
    },
}


class V34TrainingError(RuntimeError):
    pass


def load_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise V34TrainingError("V3.4 config must be a mapping")
    expected = {
        "schema_version": "cognitive_change_clue_config_v3_4",
        "model_version": "cognitive-mm-v3.4.0",
        "seed": 20260805,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise V34TrainingError(f"V3.4 config mismatch for {key}")
    if config.get("outputs", {}).get("official_test_enabled") is not False:
        raise V34TrainingError("V3.4 config must disable official Test")
    cv = config.get("cross_validation", {})
    if (
        cv.get("outer_folds") != 5
        or cv.get("outer_seed") != 20260805
        or cv.get("inner_validation_seed_base") != 20260905
    ):
        raise V34TrainingError("V3.4 cross-validation protocol mismatch")
    return config


def allocate_run_id(report_root: Path = REPORT_ROOT) -> str:
    date = datetime.now().strftime("%Y%m%d")
    run_root = report_root / "runs"
    used = {path.name for path in run_root.glob(f"COG-{date}-*") if path.is_dir()}
    index = 1
    while f"COG-{date}-{index:03d}" in used:
        index += 1
    return f"COG-{date}-{index:03d}"


def _git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=WORKSPACE_ROOT,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        ).stdout.strip()

    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def _device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    value = torch.device(requested)
    if value.type == "cuda" and not torch.cuda.is_available():
        raise V34TrainingError("CUDA was requested but is unavailable")
    return value


def _loader(
    dataset: Dataset[Any], config: Mapping[str, Any], *, train: bool, epoch: int = 0
) -> DataLoader[Any]:
    training = config["training"]
    generator = torch.Generator()
    generator.manual_seed(int(training["seed"]) + int(epoch))
    return DataLoader(
        dataset,
        batch_size=int(training["micro_batch_size"]),
        shuffle=bool(training["train_shuffle"] if train else training["validation_shuffle"]),
        drop_last=bool(training["train_drop_last"] if train else training["validation_drop_last"]),
        num_workers=int(training["num_workers"]),
        collate_fn=cognitive_collate,
        generator=generator,
    )


def _checkpoint(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    scaler: Any,
    epoch: int,
    candidate_id: str,
    outer_fold: int,
    run_id: str,
    config: Mapping[str, Any],
    input_hashes: Mapping[str, str],
    best_state: Mapping[str, Any],
    early_state: Mapping[str, Any],
    weights: TrainingWeights,
    moca_statistics: MocaStatistics,
) -> dict[str, Any]:
    return {
        "schema_version": "cognitive_v34_training_checkpoint_v1",
        "candidate_id": candidate_id,
        "outer_fold": int(outer_fold),
        "run_id": run_id,
        "epoch": int(epoch),
        "next_epoch": int(epoch) + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "rng_state": capture_rng_state(),
        "config": copy.deepcopy(dict(config)),
        "input_hashes": dict(input_hashes),
        "checkpoint_selection_state": dict(best_state),
        "early_stopping_state": dict(early_state),
        "training_weights": weights.to_dict(),
        "moca_statistics": moca_statistics.to_dict(),
    }


def _restore_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    scaler: Any,
    candidate_id: str,
    outer_fold: int,
    run_id: str,
    input_hashes: Mapping[str, str],
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "schema_version": "cognitive_v34_training_checkpoint_v1",
        "candidate_id": candidate_id,
        "outer_fold": int(outer_fold),
        "run_id": run_id,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise V34TrainingError(f"checkpoint mismatch for {key}")
    if payload.get("input_hashes") != dict(input_hashes):
        raise V34TrainingError("checkpoint input hashes mismatch")
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    scheduler.load_state_dict(payload["scheduler_state_dict"])
    scaler.load_state_dict(payload["scaler_state_dict"])
    restore_rng_state(payload["rng_state"])
    return payload


def _candidate_missing_mask(original: torch.Tensor, modalities: Sequence[str]) -> torch.Tensor:
    selected = set(modalities)
    artificial = torch.tensor(
        [name not in selected for name in MODALITY_ORDER],
        dtype=torch.bool,
        device=original.device,
    )
    return original.bool() | artificial.unsqueeze(0)


def _candidate_records(
    records: Sequence[Mapping[str, Any]], modalities: Sequence[str]
) -> list[dict[str, Any]]:
    """Return training views whose missing masks enforce the candidate boundary."""
    selected = set(modalities)
    artificial = np.asarray(
        [name not in selected for name in MODALITY_ORDER], dtype=np.bool_
    )
    result: list[dict[str, Any]] = []
    for source in records:
        row = copy.copy(dict(source))
        row["missing_mask"] = np.asarray(source["missing_mask"], dtype=np.bool_) | artificial
        result.append(row)
    return result


def _candidate_loss_weights(
    base_weights: Mapping[str, float], heads: str
) -> dict[str, float]:
    weights = {name: float(value) for name, value in base_weights.items()}
    if heads == "four_head":
        return weights
    if heads != "main_head":
        raise V34TrainingError(f"unsupported candidate head policy: {heads}")
    return {
        "hc_vs_non_hc": weights["hc_vs_non_hc"],
        "mci_vs_hc": 0.0,
        "ad_mci_hc": 0.0,
        "moca": 0.0,
    }


def _candidate_masking_plan(
    records: Sequence[Mapping[str, Any]],
    *,
    modalities: Sequence[str],
    epoch: int,
    seed_base: int,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Build deterministic dropout without ever introducing an excluded modality."""
    selected = tuple(modalities)
    if not selected:
        raise V34TrainingError("candidate must select at least one modality")
    if len(selected) == len(MODALITY_ORDER):
        return build_masking_plan(records, epoch=epoch, seed_base=seed_base)
    rng = np.random.Generator(np.random.PCG64(int(seed_base) + int(epoch)))
    plan: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda item: int(item["manifest_index"])):
        manifest_index = int(record["manifest_index"])
        if len(selected) == 1 or manifest_index % 10 <= 6:
            target = selected
        else:
            target = (selected[int(rng.integers(0, len(selected)))],)
        target_set = set(target)
        artificial = np.asarray(
            [name not in target_set for name in MODALITY_ORDER], dtype=np.bool_
        )
        original = np.asarray(record["missing_mask"], dtype=np.bool_)
        final = original | artificial
        effective = "none" if bool(final.all()) else "_".join(
            name for name, missing in zip(MODALITY_ORDER, final) if not missing
        )
        payload = {
            "sample_id": str(record["sample_id"]),
            "global_manifest_index": manifest_index,
            "target_combination": "_".join(target),
            "final_missing_mask": [int(value) for value in final],
            "effective_combination": effective,
        }
        plan[str(record["sample_id"])] = payload
        rows.append(payload)
    return plan, rows


@torch.no_grad()
def predict_raw(
    model: nn.Module,
    dataset: Dataset[Any],
    *,
    config: Mapping[str, Any],
    device: torch.device,
    candidate_id: str,
    outer_fold: int,
    split_role: str,
) -> list[dict[str, Any]]:
    if candidate_id not in CANDIDATES:
        raise V34TrainingError(f"unsupported V3.4 candidate: {candidate_id}")
    model.eval()
    modalities = CANDIDATES[candidate_id]["modalities"]
    rows: dict[str, dict[str, Any]] = {}
    for raw_batch in _loader(dataset, config, train=False):
        batch = move_batch_to_device(raw_batch, device)
        natural = batch["missing_mask"].bool()
        candidate_missing = _candidate_missing_mask(natural, modalities)
        quality = batch["quality"].detach().cpu().numpy()
        natural_cpu = natural.detach().cpu().numpy()
        labels = batch["hc_vs_non_hc"].detach().cpu().numpy()
        for index, sample_id in enumerate(batch["sample_id"]):
            rows[str(sample_id)] = {
                "schema_version": str(config["outputs"]["prediction_schema_version"]),
                "candidate_id": candidate_id,
                "outer_fold": int(outer_fold),
                "split_role": split_role,
                "sample_id": str(sample_id),
                "subject_id": str(batch["subject_id"][index]),
                "diagnosis": str(batch["diagnosis_label"][index]),
                "label_hc_vs_non_hc": int(labels[index]),
                "raw_logit": None,
                "audio_logit": None,
                "text_logit": None,
                "face_logit": None,
                "q_audio": float(quality[index, 0]),
                "q_text": float(quality[index, 1]),
                "q_face": float(quality[index, 2]),
                "audio_missing_mask": int(natural_cpu[index, 0]),
                "text_missing_mask": int(natural_cpu[index, 1]),
                "face_missing_mask": int(natural_cpu[index, 2]),
            }

        filtered, valid = filter_valid_rows(batch, candidate_missing)
        if bool(valid.any()):
            output = model(filtered["features"], filtered["quality"], filtered["missing_mask"])
            for sample_id, logit in zip(
                filtered["sample_id"], output.hc_vs_non_hc_logit.detach().cpu().tolist()
            ):
                rows[str(sample_id)]["raw_logit"] = float(logit)

        for modality_index, modality in enumerate(MODALITY_ORDER):
            if modality not in modalities:
                continue
            single_missing = torch.ones_like(natural, dtype=torch.bool)
            single_missing[:, modality_index] = natural[:, modality_index]
            filtered, valid = filter_valid_rows(batch, single_missing)
            if not bool(valid.any()):
                continue
            output = model(filtered["features"], filtered["quality"], filtered["missing_mask"])
            for sample_id, logit in zip(
                filtered["sample_id"], output.hc_vs_non_hc_logit.detach().cpu().tolist()
            ):
                rows[str(sample_id)][f"{modality}_logit"] = float(logit)
    result = [rows[key] for key in sorted(rows)]
    _validate_prediction_rows(
        result,
        candidate_id=candidate_id,
        outer_fold=outer_fold,
        split_role=split_role,
    )
    return result


def _validate_prediction_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate_id: str,
    outer_fold: int,
    split_role: str,
) -> None:
    if not rows:
        raise V34TrainingError("prediction file cannot be empty")
    sample_ids: set[str] = set()
    for row in rows:
        if set(row) != PREDICTION_FIELDS:
            raise V34TrainingError("prediction row schema mismatch")
        if (
            row["candidate_id"] != candidate_id
            or int(row["outer_fold"]) != int(outer_fold)
            or row["split_role"] != split_role
        ):
            raise V34TrainingError("prediction identity mismatch")
        sample_id = str(row["sample_id"])
        if sample_id in sample_ids:
            raise V34TrainingError(f"duplicate prediction sample: {sample_id}")
        sample_ids.add(sample_id)
        for field in (
            "raw_logit",
            "audio_logit",
            "text_logit",
            "face_logit",
            "q_audio",
            "q_text",
            "q_face",
        ):
            value = row[field]
            if value is not None and not math.isfinite(float(value)):
                raise V34TrainingError(f"non-finite prediction value: {field}")


def verify_prediction_bundle(
    candidate_dir: str | Path,
    *,
    expected_split_sha256: str | None = None,
) -> dict[str, Any]:
    candidate_dir = Path(candidate_dir)
    manifest_path = candidate_dir / "prediction_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "cognitive_v34_prediction_manifest_v1":
        raise V34TrainingError("prediction manifest schema mismatch")
    if expected_split_sha256 and manifest.get("split_sha256") != expected_split_sha256:
        raise V34TrainingError("prediction manifest split hash mismatch")
    candidate_id = str(manifest["candidate_id"])
    outer_fold = int(manifest["outer_fold"])
    for split_role, file_name in PREDICTION_FILES.items():
        record = manifest["files"].get(split_role)
        path = candidate_dir / file_name
        if record is None or not path.is_file():
            raise V34TrainingError(f"prediction file is missing: {split_role}")
        if sha256_file(path) != record.get("sha256"):
            raise V34TrainingError(f"prediction hash mismatch: {split_role}")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        _validate_prediction_rows(
            rows,
            candidate_id=candidate_id,
            outer_fold=outer_fold,
            split_role=split_role,
        )
        if len(rows) != int(record.get("row_count", -1)):
            raise V34TrainingError(f"prediction row count mismatch: {split_role}")
    return manifest


def _complete_rows(rows: Sequence[Mapping[str, Any]], modalities: Sequence[str]) -> list[dict[str, Any]]:
    required = [f"{name}_missing_mask" for name in modalities]
    return [
        dict(row)
        for row in rows
        if row.get("raw_logit") is not None and all(int(row[field]) == 0 for field in required)
    ]


def _raw_metric_report(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scored = [row for row in rows if row.get("raw_logit") is not None]
    task_labels = [int(row["label_hc_vs_non_hc"]) for row in scored]
    task_probabilities = [sigmoid(float(row["raw_logit"])) for row in scored]
    subjects = aggregate_subjects(scored)
    return {
        "task": binary_metrics(task_labels, task_probabilities),
        "subject": binary_metrics(
            [int(row["label_hc_vs_non_hc"]) for row in subjects],
            [float(row["probability"]) for row in subjects],
        ),
        "subject_rows": subjects,
    }


def _coverage(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    task_counts: Counter[str] = Counter()
    subjects: dict[str, set[str]] = {}
    for row in rows:
        available = [
            name
            for name in MODALITY_ORDER
            if int(row[f"{name}_missing_mask"]) == 0
        ]
        combination = "+".join(name[0].upper() for name in available) if available else "none"
        task_counts[combination] += 1
        subjects.setdefault(combination, set()).add(str(row["subject_id"]))
    return {
        "task_counts": dict(sorted(task_counts.items())),
        "subject_counts": {key: len(value) for key, value in sorted(subjects.items())},
    }


def _fold_summary(
    *,
    candidate_dir: Path,
    candidate_id: str,
    outer_fold: int,
    checkpoint_sha256: str,
    split_sha256: str,
    best_epoch: int,
    prediction_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    duration_seconds: float,
) -> dict[str, Any]:
    modalities = CANDIDATES[candidate_id]["selection_modalities"]
    complete = {
        role: _complete_rows(rows, modalities) for role, rows in prediction_rows.items()
    }
    raw = {
        role: _raw_metric_report(rows) for role, rows in complete.items()
    }
    temporary = temporary_platt_evaluation(
        complete["inner_calibration"], complete["outer_evaluation"]
    )
    summary = {
        "schema_version": "cognitive_v34_fold_metrics_v1",
        "candidate_id": candidate_id,
        "outer_fold": int(outer_fold),
        "selection_population": "same_original_complete_modalities_tasks",
        "selection_modalities": list(modalities),
        "best_epoch": int(best_epoch),
        "checkpoint_sha256": checkpoint_sha256,
        "split_sha256": split_sha256,
        "complete_task_counts": {role: len(rows) for role, rows in complete.items()},
        "raw_metrics": {
            role: {key: value for key, value in report.items() if key != "subject_rows"}
            for role, report in raw.items()
        },
        "temporary_calibration": temporary,
        "coverage": {role: _coverage(rows) for role, rows in prediction_rows.items()},
        "duration_seconds": float(duration_seconds),
        "official_test_evaluated": False,
    }
    write_json(candidate_dir / "fold_metrics.json", summary)
    return summary


def _write_prediction_bundle(
    *,
    candidate_dir: Path,
    candidate_id: str,
    outer_fold: int,
    split_sha256: str,
    checkpoint_path: Path,
    input_hashes: Mapping[str, str],
    prediction_rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for split_role, file_name in PREDICTION_FILES.items():
        path = candidate_dir / file_name
        rows = list(prediction_rows[split_role])
        write_jsonl(path, rows)
        files[split_role] = {
            "path": file_name,
            "row_count": len(rows),
            "scored_row_count": sum(row.get("raw_logit") is not None for row in rows),
            "sha256": sha256_file(path),
        }
    manifest = {
        "schema_version": "cognitive_v34_prediction_manifest_v1",
        "candidate_id": candidate_id,
        "outer_fold": int(outer_fold),
        "split_sha256": split_sha256,
        "checkpoint_path": workspace_relative(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "input_hashes": dict(input_hashes),
        "files": files,
        "official_test_evaluated": False,
    }
    write_json(candidate_dir / "prediction_manifest.json", manifest)
    verify_prediction_bundle(candidate_dir, expected_split_sha256=split_sha256)
    return manifest


def _record_failure(candidate_dir: Path, error: BaseException, *, epoch: int | None) -> None:
    diagnostic = {
            "schema_version": "cognitive_v34_training_failure_v1",
            "time": datetime.now(timezone.utc).astimezone().isoformat(),
            "epoch": epoch,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "latest_checkpoint": (
                workspace_relative(candidate_dir / "latest_checkpoint.pt")
                if (candidate_dir / "latest_checkpoint.pt").is_file()
                else None
            ),
        }
    write_json(candidate_dir / "failure_diagnostic.json", diagnostic)
    append_jsonl(candidate_dir / "failure_history.jsonl", diagnostic)


def train_fold(
    *,
    candidate_id: str,
    outer_fold: int,
    run_id: str,
    run_dir: Path,
    base_config: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    frozen_input_hashes: Mapping[str, str],
    device: torch.device,
    input_dims: Mapping[str, int] | None = None,
    feature_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if candidate_id not in CANDIDATES:
        raise V34TrainingError(f"unsupported candidate: {candidate_id}")
    if CANDIDATES[candidate_id].get("kind", "neural") != "neural":
        raise V34TrainingError(
            f"candidate {candidate_id} must be built by its post-hoc training module"
        )
    candidate_dir = run_dir / f"fold_{outer_fold}" / candidate_id
    candidate_dir.mkdir(parents=True, exist_ok=True)
    feature_audit_path: Path | None = None
    if feature_audit is not None:
        feature_audit_path = candidate_dir / "feature_view_audit.json"
        write_json(feature_audit_path, dict(feature_audit))
    fold = load_fold(outer_fold)
    split_sha256 = str(fold["split_sha256"])
    input_hashes = {
        **dict(frozen_input_hashes),
        "v34_split": split_sha256,
        "v34_split_audit": sha256_file(DEFAULT_AUDIT_PATH),
        "v34_config": sha256_file(CONFIG_PATH),
    }
    if feature_audit_path is not None:
        input_hashes["feature_view_audit"] = sha256_file(feature_audit_path)
    existing_summary = candidate_dir / "fold_metrics.json"
    if existing_summary.is_file() and (candidate_dir / "prediction_manifest.json").is_file():
        verify_prediction_bundle(candidate_dir, expected_split_sha256=split_sha256)
        return json.loads(existing_summary.read_text(encoding="utf-8"))

    config = copy.deepcopy(dict(base_config))
    config["training"] = copy.deepcopy(dict(base_config["training"]))
    config["training"]["seed"] = int(base_config["seed"]) + int(outer_fold)
    config["masking"] = copy.deepcopy(dict(base_config["masking"]))
    config["masking"]["seed_base"] = int(base_config["masking"]["seed_base"]) + 1000 * int(outer_fold)
    configure_determinism(config)
    candidate = CANDIDATES[candidate_id]
    config["candidate"] = {
        "candidate_id": candidate_id,
        "modalities": list(candidate["modalities"]),
        "validation_modalities": list(candidate["validation_modalities"]),
        "selection_modalities": list(candidate["selection_modalities"]),
        "heads": candidate["heads"],
        "input_dims": dict(input_dims) if input_dims is not None else None,
        "feature_variant": (
            str(feature_audit.get("variant")) if feature_audit is not None else None
        ),
    }
    config["model"] = copy.deepcopy(dict(base_config["model"]))
    config["model"]["loss_weights"] = _candidate_loss_weights(
        base_config["model"]["loss_weights"], str(candidate["heads"])
    )
    datasets = build_fold_datasets(
        records,
        fold,
        selection_modalities=candidate["validation_modalities"],
        available_modalities=candidate["modalities"],
    )
    full_validation_dataset = SubjectFeatureDataset(
        records, fold["inner_validation_subjects"]
    )
    weights = compute_training_weights(datasets["inner_train"].records)
    moca_statistics = compute_moca_statistics(datasets["inner_train"].records)
    validation_loader = _loader(datasets["inner_validation"], config, train=False)

    model = CognitiveChangeClueModel(
        projection_dim=int(config["model"]["projection_dim"]),
        fusion_hidden_dims=tuple(config["model"]["fusion_hidden_dims"]),
        dropout=float(config["model"]["dropout"]),
        input_dims=input_dims,
    ).to(device)
    training = config["training"]
    optimizer = AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        betas=tuple(float(value) for value in training["adam_betas"]),
        eps=float(training["adam_eps"]),
    )
    scheduler = ReduceLROnPlateau(optimizer, **dict(training["scheduler"]))
    amp_enabled = bool(training["amp"]) and device.type == "cuda"
    scaler = _new_grad_scaler(amp_enabled)
    best_state: dict[str, Any] = {"auc": float("-inf"), "balanced_accuracy": float("-inf"), "epoch": None}
    early_state: dict[str, Any] = {"best_auc": float("-inf"), "epochs_without_improvement": 0}
    latest_path = candidate_dir / "latest_checkpoint.pt"
    best_path = candidate_dir / "best_checkpoint.pt"
    start_epoch = 1
    amp_recovery: dict[str, Any] | None = None
    if latest_path.is_file():
        restored = _restore_checkpoint(
            latest_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            candidate_id=candidate_id,
            outer_fold=outer_fold,
            run_id=run_id,
            input_hashes=input_hashes,
        )
        start_epoch = int(restored["next_epoch"])
        best_state = dict(restored["checkpoint_selection_state"])
        early_state = dict(restored["early_stopping_state"])
        amp_recovery = recover_grad_scaler_after_recorded_overflow(
            scaler,
            run_dir=candidate_dir,
            next_epoch=start_epoch,
            amp_enabled=amp_enabled,
        )

    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).astimezone().isoformat()
    previous_recoveries: list[dict[str, Any]] = []
    existing_run_manifest = candidate_dir / "run_manifest.json"
    if existing_run_manifest.is_file():
        previous = json.loads(existing_run_manifest.read_text(encoding="utf-8"))
        previous_recoveries = list(previous.get("recoveries", ()))
    write_json(
        candidate_dir / "run_manifest.json",
        {
            "schema_version": "cognitive_v34_fold_run_v1",
            "status": "running",
            "run_id": run_id,
            "candidate_id": candidate_id,
            "outer_fold": outer_fold,
            "started_at": started_at,
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "numpy": np.__version__,
            },
            "device": {
                "resolved": str(device),
                "name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                "amp_enabled": amp_enabled,
            },
            "git": _git_state(),
            "input_hashes": input_hashes,
            "subjects": {name: len(dataset.subject_ids) for name, dataset in datasets.items()},
            "tasks": {name: len(dataset) for name, dataset in datasets.items()},
            "resume_from_epoch": start_epoch if start_epoch > 1 else None,
            "recoveries": [
                *previous_recoveries,
                *([amp_recovery] if amp_recovery is not None else []),
            ],
            "official_test_evaluated": False,
        },
    )
    write_json(candidate_dir / "config_snapshot.json", config)
    completed_epoch = start_epoch - 1
    try:
        epoch_range = range(start_epoch, int(training["max_epochs"]) + 1)
        if int(early_state["epochs_without_improvement"]) >= int(
            config["selection"]["early_stopping_patience"]
        ):
            epoch_range = range(0)
        for epoch in epoch_range:
            epoch_started = time.perf_counter()
            masking_plan, masking_rows = _candidate_masking_plan(
                datasets["inner_train"].records,
                modalities=candidate["modalities"],
                epoch=epoch,
                seed_base=int(config["masking"]["seed_base"]),
            )
            write_jsonl(candidate_dir / "masking" / f"epoch_{epoch:03d}.jsonl", masking_rows)
            train_metrics = train_one_epoch(
                model,
                _loader(datasets["inner_train"], config, train=True, epoch=epoch),
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                masking_plan=masking_plan,
                weights=weights,
                moca_statistics=moca_statistics,
                loss_weights=config["model"]["loss_weights"],
                accumulation_steps=int(training["gradient_accumulation_steps"]),
                max_grad_norm=float(training["max_grad_norm"]),
                amp_enabled=amp_enabled,
            )
            validation_predictions = predict_validation(
                model,
                validation_loader,
                device=device,
                moca_statistics=moca_statistics,
            )
            validation = subject_metrics(validation_predictions)
            main = validation["hc_vs_non_hc"]
            auc = main["roc_auc"]
            balanced = main["balanced_accuracy_at_0_5"]
            if auc is None or balanced is None:
                raise V34TrainingError("inner Validation main metrics are not computable")
            better = checkpoint_is_better(
                float(auc),
                float(balanced),
                float(best_state["auc"]),
                float(best_state["balanced_accuracy"]),
                auc_tolerance=float(config["selection"]["checkpoint_auc_tolerance"]),
            )
            if better:
                best_state = {"auc": float(auc), "balanced_accuracy": float(balanced), "epoch": epoch}
            scheduler.step(float(auc))
            if float(auc) > float(early_state["best_auc"]) + float(
                config["selection"]["early_stopping_min_delta"]
            ):
                early_state = {"best_auc": float(auc), "epochs_without_improvement": 0}
            else:
                early_state["epochs_without_improvement"] = int(
                    early_state["epochs_without_improvement"]
                ) + 1
            checkpoint = _checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                candidate_id=candidate_id,
                outer_fold=outer_fold,
                run_id=run_id,
                config=config,
                input_hashes=input_hashes,
                best_state=best_state,
                early_state=early_state,
                weights=weights,
                moca_statistics=moca_statistics,
            )
            atomic_torch_save(latest_path, checkpoint)
            if better:
                atomic_torch_save(best_path, checkpoint)
            epoch_record = {
                "schema_version": "cognitive_v34_epoch_metrics_v1",
                "candidate_id": candidate_id,
                "outer_fold": outer_fold,
                "epoch": epoch,
                "train": train_metrics,
                "inner_validation": {key: value for key, value in validation.items() if key != "subjects"},
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "checkpoint_selected": better,
                "best": dict(best_state),
                "early_stopping": dict(early_state),
                "duration_seconds": time.perf_counter() - epoch_started,
            }
            epoch_path = candidate_dir / "epoch_metrics.jsonl"
            existing = [
                json.loads(line)
                for line in epoch_path.read_text(encoding="utf-8").splitlines()
                if line
            ] if epoch_path.is_file() else []
            existing = [row for row in existing if int(row["epoch"]) < epoch]
            write_jsonl(epoch_path, [*existing, epoch_record])
            print(json.dumps(epoch_record, ensure_ascii=False, allow_nan=False), flush=True)
            completed_epoch = epoch
            if int(early_state["epochs_without_improvement"]) >= int(
                config["selection"]["early_stopping_patience"]
            ):
                break
    except Exception as exc:
        _record_failure(candidate_dir, exc, epoch=completed_epoch + 1)
        raise

    if not best_path.is_file():
        raise V34TrainingError("fold training produced no best checkpoint")
    best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    model.to(device).eval()
    prediction_datasets: dict[str, Dataset[Any]] = {
        "inner_validation": full_validation_dataset,
        "inner_calibration": SubjectFeatureDataset(
            records, fold["inner_calibration_subjects"]
        ),
        "outer_evaluation": SubjectFeatureDataset(
            records, fold["outer_evaluation_subjects"]
        ),
    }
    prediction_rows = {
        role: predict_raw(
            model,
            dataset,
            config=config,
            device=device,
            candidate_id=candidate_id,
            outer_fold=outer_fold,
            split_role=role,
        )
        for role, dataset in prediction_datasets.items()
    }
    repeat_outer = predict_raw(
        model,
        prediction_datasets["outer_evaluation"],
        config=config,
        device=device,
        candidate_id=candidate_id,
        outer_fold=outer_fold,
        split_role="outer_evaluation",
    )
    if prediction_rows["outer_evaluation"] != repeat_outer:
        raise V34TrainingError("repeat outer-evaluation inference is not deterministic")
    prediction_manifest = _write_prediction_bundle(
        candidate_dir=candidate_dir,
        candidate_id=candidate_id,
        outer_fold=outer_fold,
        split_sha256=split_sha256,
        checkpoint_path=best_path,
        input_hashes=input_hashes,
        prediction_rows=prediction_rows,
    )
    summary = _fold_summary(
        candidate_dir=candidate_dir,
        candidate_id=candidate_id,
        outer_fold=outer_fold,
        checkpoint_sha256=prediction_manifest["checkpoint_sha256"],
        split_sha256=split_sha256,
        best_epoch=int(best_checkpoint["epoch"]),
        prediction_rows=prediction_rows,
        duration_seconds=time.perf_counter() - started,
    )
    manifest_path = candidate_dir / "run_manifest.json"
    run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_manifest.update(
        {
            "status": "completed",
            "ended_at": datetime.now(timezone.utc).astimezone().isoformat(),
            "completed_epochs": completed_epoch,
            "best_epoch": int(best_checkpoint["epoch"]),
            "best_checkpoint": workspace_relative(best_path),
            "best_checkpoint_sha256": prediction_manifest["checkpoint_sha256"],
            "prediction_manifest": workspace_relative(candidate_dir / "prediction_manifest.json"),
        }
    )
    write_json(manifest_path, run_manifest)
    return summary


def aggregate_candidate(run_dir: Path, *, candidate_id: str) -> dict[str, Any]:
    folds = [
        json.loads((run_dir / f"fold_{fold}" / candidate_id / "fold_metrics.json").read_text(encoding="utf-8"))
        for fold in range(5)
    ]
    raw_auc = [float(fold["raw_metrics"]["outer_evaluation"]["subject"]["roc_auc"]) for fold in folds]
    calibrated_auc = [
        float(fold["temporary_calibration"]["outer_evaluation_subject_metrics"]["roc_auc"])
        for fold in folds
    ]
    raw_task_auc = [
        float(fold["raw_metrics"]["outer_evaluation"]["task"]["roc_auc"])
        for fold in folds
    ]
    calibrated_brier = [
        float(fold["temporary_calibration"]["outer_evaluation_subject_metrics"]["brier"])
        for fold in folds
    ]
    calibrated_ece = [
        float(
            fold["temporary_calibration"]["outer_evaluation_subject_metrics"][
                "ece_10_equal_width"
            ]
        )
        for fold in folds
    ]
    confusion = {key: 0 for key in ("tp", "fn", "tn", "fp")}
    false_positive_counts: Counter[str] = Counter()
    outer_coverage_tasks: Counter[str] = Counter()
    for fold in folds:
        matrix = fold["temporary_calibration"]["outer_evaluation_subject_metrics"]["confusion_matrix"]
        for key in confusion:
            confusion[key] += int(matrix[key])
        false_positive_counts.update(fold["temporary_calibration"]["false_positive_subjects"])
        outer_coverage_tasks.update(fold["coverage"]["outer_evaluation"]["task_counts"])
    sensitivity = confusion["tp"] / (confusion["tp"] + confusion["fn"])
    specificity = confusion["tn"] / (confusion["tn"] + confusion["fp"])
    selection_modalities = tuple(CANDIDATES[candidate_id]["selection_modalities"])
    selection_population = "same_original_complete_" + "_".join(selection_modalities) + "_tasks"
    report = {
        "schema_version": "cognitive_v34_candidate_cv_summary_v1",
        "candidate_id": candidate_id,
        "run_id": run_dir.name,
        "fold_count": 5,
        "development_subject_count": 459,
        "official_test_evaluated": False,
        "selection_population": selection_population,
        "candidate_count_to_date": 1,
        "best_epochs": [int(fold["best_epoch"]) for fold in folds],
        "task_raw_auc": {
            "folds": raw_task_auc,
            "mean": mean(raw_task_auc),
            "population_std": pstdev(raw_task_auc),
            "worst": min(raw_task_auc),
        },
        "subject_raw_auc": {
            "folds": raw_auc,
            "mean": mean(raw_auc),
            "population_std": pstdev(raw_auc),
            "worst": min(raw_auc),
        },
        "subject_temporary_platt_auc": {
            "folds": calibrated_auc,
            "mean": mean(calibrated_auc),
            "population_std": pstdev(calibrated_auc),
            "worst": min(calibrated_auc),
        },
        "subject_temporary_platt_brier": {
            "folds": calibrated_brier,
            "mean": mean(calibrated_brier),
            "population_std": pstdev(calibrated_brier),
        },
        "subject_temporary_platt_ece": {
            "folds": calibrated_ece,
            "mean": mean(calibrated_ece),
            "population_std": pstdev(calibrated_ece),
        },
        "pooled_temporary_workpoint": {
            "confusion_matrix": confusion,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "balanced_accuracy": (sensitivity + specificity) / 2.0,
        },
        "outer_evaluation_modality_coverage_task_counts": dict(
            sorted(outer_coverage_tasks.items())
        ),
        "false_positive_analysis": {
            "population": "outer_evaluation_complete_"
            + "_".join(selection_modalities)
            + "_hc_subjects",
            "false_positive_count": sum(false_positive_counts.values()),
            "unique_false_positive_subject_count": len(false_positive_counts),
            "subjects": [
                {"subject_id": subject_id, "fold_occurrences": count}
                for subject_id, count in sorted(false_positive_counts.items())
            ],
        },
        "folds": [
            {
                "outer_fold": int(fold["outer_fold"]),
                "best_epoch": int(fold["best_epoch"]),
                "complete_outer_task_count": int(
                    fold["complete_task_counts"]["outer_evaluation"]
                ),
                "raw_task_metrics": fold["raw_metrics"]["outer_evaluation"]["task"],
                "raw_subject_metrics": fold["raw_metrics"]["outer_evaluation"]["subject"],
                "temporary_workpoint": fold["temporary_calibration"]["workpoint"],
                "temporary_outer_task_metrics": fold["temporary_calibration"]
                ["outer_evaluation_task_metrics"],
                "temporary_outer_subject_metrics": fold["temporary_calibration"]
                ["outer_evaluation_subject_metrics"],
            }
            for fold in folds
        ],
        "limitations": [
            "This is five-fold development evidence on CogPic official Train only.",
            "It is not an independent external Test result.",
        ],
    }
    write_json(run_dir / f"{candidate_id}_cv_summary.json", report)
    write_json(
        run_dir / f"{candidate_id}_false_positive_analysis.json",
        {
            "schema_version": "cognitive_v34_false_positive_analysis_v1",
            "candidate_id": candidate_id,
            "run_id": run_dir.name,
            "official_test_evaluated": False,
            **report["false_positive_analysis"],
        },
    )
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    write_json(
        REPORT_ROOT / f"{candidate_id}_baseline_report.json",
        {
            **report,
            "run_summary_path": workspace_relative(
                run_dir / f"{candidate_id}_cv_summary.json"
            ),
            "run_summary_sha256": sha256_file(
                run_dir / f"{candidate_id}_cv_summary.json"
            ),
            "false_positive_analysis_path": workspace_relative(
                run_dir / f"{candidate_id}_false_positive_analysis.json"
            ),
            "false_positive_analysis_sha256": sha256_file(
                run_dir / f"{candidate_id}_false_positive_analysis.json"
            ),
        },
    )
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    candidate_id = str(args.candidate)
    if candidate_id not in CANDIDATES:
        raise V34TrainingError(f"unsupported candidate: {candidate_id}")
    if not DEFAULT_OUTPUT_PATH.is_file() or not DEFAULT_AUDIT_PATH.is_file():
        raise V34TrainingError("V3.4 split and audit must exist before training")
    audit = json.loads(DEFAULT_AUDIT_PATH.read_text(encoding="utf-8"))
    if audit.get("status") != "passed" or int(audit.get("official_test_overlap_count", -1)) != 0:
        raise V34TrainingError("V3.4 split audit did not pass")
    device = _device(args.device)
    records, frozen_hashes = load_official_train_pool(verify_hashes=True)
    run_id = args.run_id or allocate_run_id()
    run_dir = REPORT_ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    requested_folds = list(range(5)) if args.folds == "all" else [int(value) for value in args.folds.split(",")]
    if any(value not in range(5) for value in requested_folds):
        raise V34TrainingError("folds must be all or a comma-separated subset of 0..4")
    write_json(
        run_dir / "run_manifest.json",
        {
            "schema_version": "cognitive_v34_candidate_run_v1",
            "status": "running",
            "run_id": run_id,
            "candidate_id": candidate_id,
            "requested_folds": requested_folds,
            "pid": os.getpid(),
            "device": str(device),
            "started_or_resumed_at": datetime.now(timezone.utc).astimezone().isoformat(),
            "split_sha256": sha256_file(DEFAULT_OUTPUT_PATH),
            "config_sha256": sha256_file(args.config),
            "official_test_evaluated": False,
        },
    )
    fold_summaries = []
    for outer_fold in requested_folds:
        fold_summaries.append(
            train_fold(
                candidate_id=candidate_id,
                outer_fold=outer_fold,
                run_id=run_id,
                run_dir=run_dir,
                base_config=config,
                records=records,
                frozen_input_hashes=frozen_hashes,
                device=device,
            )
        )
    all_completed = all(
        (run_dir / f"fold_{fold}" / candidate_id / "fold_metrics.json").is_file()
        for fold in range(5)
    )
    result: dict[str, Any] = {
        "run_id": run_id,
        "candidate_id": candidate_id,
        "completed_folds": sorted(
            int(path.parent.parent.name.split("_")[-1])
            for path in run_dir.glob(f"fold_*/{candidate_id}/fold_metrics.json")
        ),
        "official_test_evaluated": False,
    }
    if all_completed:
        result["cv_summary"] = aggregate_candidate(run_dir, candidate_id=candidate_id)
        manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        manifest["status"] = "completed"
        manifest["ended_at"] = datetime.now(timezone.utc).astimezone().isoformat()
        manifest["summary"] = workspace_relative(run_dir / f"{candidate_id}_cv_summary.json")
        write_json(run_dir / "run_manifest.json", manifest)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--candidate", default="V34-A0", choices=sorted(CANDIDATES))
    parser.add_argument("--folds", default="all")
    parser.add_argument("--run-id")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> int:
    try:
        result = run(parse_args())
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
        raise
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CANDIDATES",
    "PREDICTION_FIELDS",
    "PREDICTION_FILES",
    "V34TrainingError",
    "aggregate_candidate",
    "load_config",
    "predict_raw",
    "run",
    "train_fold",
    "verify_prediction_bundle",
]
