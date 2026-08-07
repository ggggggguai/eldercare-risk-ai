"""Train, package, and verify the cognitive-mm-v3.4.0 deployment candidate."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

import torch
from torch.optim import AdamW

from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.model import (
    CognitiveChangeClueModel,
    model_state_is_finite,
)

try:
    from .common import ALGORITHM_ROOT, sha256_file, workspace_relative
    from .dataset import compute_moca_statistics, compute_training_weights
    from .evaluate import move_batch_to_device
    from .train import (
        _new_grad_scaler,
        atomic_torch_save,
        capture_rng_state,
        configure_determinism,
        record_failure,
        recover_grad_scaler_after_recorded_overflow,
        restore_rng_state,
        train_one_epoch,
        write_json,
        write_jsonl,
    )
    from .train_v34 import (
        CANDIDATES,
        REPORT_ROOT,
        _candidate_loss_weights,
        _candidate_masking_plan,
        _device,
        _git_state,
        _loader,
        load_config,
        predict_raw,
    )
    from .v34_data import SubjectFeatureDataset, load_official_train_pool
    from .v34_metrics import (
        aggregate_subjects,
        apply_calibrator,
        binary_metrics,
        fit_calibrator,
        select_workpoint,
    )
except ImportError:
    from common import ALGORITHM_ROOT, sha256_file, workspace_relative  # type: ignore[no-redef]
    from dataset import compute_moca_statistics, compute_training_weights  # type: ignore[no-redef]
    from evaluate import move_batch_to_device  # type: ignore[no-redef]
    from train import (  # type: ignore[no-redef]
        _new_grad_scaler,
        atomic_torch_save,
        capture_rng_state,
        configure_determinism,
        record_failure,
        recover_grad_scaler_after_recorded_overflow,
        restore_rng_state,
        train_one_epoch,
        write_json,
        write_jsonl,
    )
    from train_v34 import (  # type: ignore[no-redef]
        CANDIDATES,
        REPORT_ROOT,
        _candidate_loss_weights,
        _candidate_masking_plan,
        _device,
        _git_state,
        _loader,
        load_config,
        predict_raw,
    )
    from v34_data import SubjectFeatureDataset, load_official_train_pool  # type: ignore[no-redef]
    from v34_metrics import (  # type: ignore[no-redef]
        aggregate_subjects,
        apply_calibrator,
        binary_metrics,
        fit_calibrator,
        select_workpoint,
    )


MODEL_VERSION = "cognitive-mm-v3.4.0"
CANDIDATE_ID = "V34-A2"
FINAL_COMPONENT = "V34-A2-Final"
V33_PACKAGE_ROOT = (
    ALGORITHM_ROOT / "models" / "mental_health" / "cognitive_change_clue" / "v3.3.0"
)
PACKAGE_ROOT = (
    ALGORITHM_ROOT / "models" / "mental_health" / "cognitive_change_clue" / "v3.4.0"
)
FIXED_SPLIT = (
    ALGORITHM_ROOT
    / "data"
    / "processed"
    / "cognitive_change_clue"
    / "v3.3.0"
    / "cogpic_subject_split_v33.json"
)
CONFIG_PATH = ALGORITHM_ROOT / "configs" / "modules" / "cognitive_change_clue_v3_4.yaml"
CANDIDATE_CONFIG = REPORT_ROOT / "cognitive_v34_candidate_config.json"
FEATURE_SELECTION = REPORT_ROOT / "OPT-COG-003_selection_report.json"
CALIBRATION_REPORT = REPORT_ROOT / "opt_cog_004" / "OPT-COG-004_report.json"
A2_SUMMARY = REPORT_ROOT / "runs" / "COG-20260806-001" / "V34-A2_cv_summary.json"
LOCKFILE = ALGORITHM_ROOT / "requirements-cognitive-v3.3.lock"


class V34PackageError(RuntimeError):
    pass


def fixed_epoch_median(best_epochs: Sequence[int]) -> int:
    values = [int(value) for value in best_epochs]
    if len(values) != 5 or any(value <= 0 for value in values):
        raise V34PackageError("exactly five positive best epochs are required")
    return int(median(values))


def _load_split() -> dict[str, Any]:
    payload = json.loads(FIXED_SPLIT.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "cogpic_subject_split_v33":
        raise V34PackageError("fixed deployment split schema mismatch")
    splits = payload.get("splits") or {}
    train = {str(value) for value in splits.get("train", ())}
    validation = {str(value) for value in splits.get("validation", ())}
    test = {str(value) for value in splits.get("test", ())}
    if len(train) != 367 or len(validation) != 92 or len(test) != 115:
        raise V34PackageError("fixed deployment split counts mismatch")
    if train & validation or train & test or validation & test:
        raise V34PackageError("fixed deployment split leaks subjects")
    return payload


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _directory_hash(path: Path) -> str:
    rows = [
        f"{sha256_file(child)}  {child.relative_to(path).as_posix()}\n"
        for child in sorted(item for item in path.rglob("*") if item.is_file())
    ]
    return hashlib.sha256("".join(rows).encode("utf-8")).hexdigest()


def _copy_file(source: Path, target: Path) -> None:
    if not source.is_file():
        raise V34PackageError(f"required source is missing: {source}")
    if target.is_file() and sha256_file(source) == sha256_file(target):
        return
    if target.exists():
        raise V34PackageError(f"refusing to overwrite divergent package file: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def _copy_directory(source: Path, target: Path) -> None:
    if not source.is_dir():
        raise V34PackageError(f"required source directory is missing: {source}")
    if target.is_dir():
        if _directory_hash(source) != _directory_hash(target):
            raise V34PackageError(f"refusing to overwrite divergent package directory: {target}")
        return
    shutil.copytree(source, target)


def _training_checkpoint(
    *,
    model: CognitiveChangeClueModel,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    epoch: int,
    run_id: str,
    fixed_epochs: int,
    config: Mapping[str, Any],
    input_hashes: Mapping[str, str],
    weights: Any,
    moca_statistics: Any,
) -> dict[str, Any]:
    return {
        "schema_version": "cognitive_v34_final_checkpoint_v1",
        "model_version": MODEL_VERSION,
        "candidate_id": CANDIDATE_ID,
        "component_id": FINAL_COMPONENT,
        "run_id": run_id,
        "epoch": int(epoch),
        "next_epoch": int(epoch) + 1,
        "fixed_epochs": int(fixed_epochs),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "rng_state": capture_rng_state(),
        "config": copy.deepcopy(dict(config)),
        "input_hashes": dict(input_hashes),
        "training_weights": weights.to_dict(),
        "moca_statistics": moca_statistics.to_dict(),
    }


def _restore_training_checkpoint(
    path: Path,
    *,
    model: CognitiveChangeClueModel,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    run_id: str,
    fixed_epochs: int,
    input_hashes: Mapping[str, str],
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "schema_version": "cognitive_v34_final_checkpoint_v1",
        "model_version": MODEL_VERSION,
        "candidate_id": CANDIDATE_ID,
        "component_id": FINAL_COMPONENT,
        "run_id": run_id,
        "fixed_epochs": int(fixed_epochs),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise V34PackageError(f"final checkpoint mismatch for {key}")
    if payload.get("input_hashes") != dict(input_hashes):
        raise V34PackageError("final checkpoint input hashes mismatch")
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    scaler.load_state_dict(payload["scaler_state_dict"])
    restore_rng_state(payload["rng_state"])
    return payload


def _calibrate(
    rows: Sequence[Mapping[str, Any]], checkpoint_sha256: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    scored = [dict(row) for row in rows if row.get("raw_logit") is not None]
    parameters = fit_calibrator(
        "platt",
        [float(row["raw_logit"]) for row in scored],
        [int(row["label_hc_vs_non_hc"]) for row in scored],
    )
    probabilities = apply_calibrator(
        "platt", [float(row["raw_logit"]) for row in scored], parameters
    )
    enriched = [
        {**row, "calibrated_probability": float(probability)}
        for row, probability in zip(scored, probabilities, strict=True)
    ]
    subjects = aggregate_subjects(enriched, probability_field="calibrated_probability")
    subject_workpoint = select_workpoint(subjects)
    task_workpoint = select_workpoint(
        [
            {
                "label_hc_vs_non_hc": int(row["label_hc_vs_non_hc"]),
                "probability": float(row["calibrated_probability"]),
            }
            for row in enriched
        ]
    )
    subject_metrics = binary_metrics(
        [int(row["label_hc_vs_non_hc"]) for row in subjects],
        [float(row["probability"]) for row in subjects],
        threshold=float(subject_workpoint["threshold"]),
    )
    task_metrics = binary_metrics(
        [int(row["label_hc_vs_non_hc"]) for row in enriched],
        [float(row["calibrated_probability"]) for row in enriched],
        threshold=float(task_workpoint["threshold"]),
    )
    calibration = {
        "schema_version": "cognitive_calibration_v2",
        "model_version": MODEL_VERSION,
        "method": "platt",
        "a": float(parameters["a"]),
        "b": float(parameters["b"]),
        "checkpoint_sha256": checkpoint_sha256,
        "fit_scope": "fixed_v33_validation_92_task_raw_logits",
        "fit_task_count": len(scored),
        "fit_subject_count": len(subjects),
        "subject_probability_aggregation": "calibrate_each_task_then_mean_probability",
        "official_test_evaluated": False,
    }
    report = {
        "schema_version": "cognitive_v34_final_calibration_report_v1",
        "calibration": calibration,
        "subject_workpoint": subject_workpoint,
        "task_workpoint_supplemental": task_workpoint,
        "validation_subject_metrics": subject_metrics,
        "validation_task_metrics": task_metrics,
        "validation_role": "final_package_implementation_record_not_independent_test",
        "official_test_evaluated": False,
    }
    return calibration, report


def _package_sums(package_root: Path = PACKAGE_ROOT) -> str:
    rows = []
    for path in sorted(
        item
        for item in package_root.rglob("*")
        if item.is_file() and item.name != "sha256sums.txt"
    ):
        rows.append(f"{sha256_file(path)}  {path.relative_to(package_root).as_posix()}\n")
    return "".join(rows)


def verify_package(package_root: Path = PACKAGE_ROOT, *, load_model: bool = True) -> dict[str, Any]:
    manifest_path = package_root / "manifest.json"
    sums_path = package_root / "sha256sums.txt"
    if not manifest_path.is_file() or not sums_path.is_file():
        raise V34PackageError("V3.4 package manifest or checksums are missing")
    if sums_path.read_text(encoding="ascii") != _package_sums(package_root):
        raise V34PackageError("V3.4 package checksums do not match package files")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("model_version") != MODEL_VERSION:
        raise V34PackageError("V3.4 package model version mismatch")
    if manifest.get("network_access") != "forbidden":
        raise V34PackageError("V3.4 package must forbid network access")
    checkpoint_path = package_root / "fusion.pt"
    checkpoint_sha = sha256_file(checkpoint_path)
    calibration = json.loads((package_root / "calibration.json").read_text(encoding="utf-8"))
    feature_stats = json.loads((package_root / "feature_stats.json").read_text(encoding="utf-8"))
    if calibration.get("checkpoint_sha256") != checkpoint_sha:
        raise V34PackageError("V3.4 calibration checkpoint hash mismatch")
    if feature_stats.get("checkpoint_sha256") != checkpoint_sha:
        raise V34PackageError("V3.4 feature statistics checkpoint hash mismatch")
    deterministic = None
    if load_model:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = CognitiveChangeClueModel(projection_dim=256, fusion_hidden_dims=(128, 64), dropout=0.2)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        if not model_state_is_finite(model):
            raise V34PackageError("V3.4 fusion checkpoint contains non-finite values")
        features = {
            "audio": torch.zeros((1, 768), dtype=torch.float32),
            "text": torch.zeros((1, 768), dtype=torch.float32),
            "face": torch.zeros((1, 512), dtype=torch.float32),
        }
        quality = torch.tensor([[0.8, 0.7, 0.0]], dtype=torch.float32)
        missing = torch.tensor([[False, False, True]])
        with torch.inference_mode():
            first = model(features, quality, missing).hc_vs_non_hc_logit
            second = model(features, quality, missing).hc_vs_non_hc_logit
        if not torch.isfinite(first).all() or not torch.equal(first, second):
            raise V34PackageError("V3.4 CPU inference is non-finite or non-deterministic")
        deterministic = float(first.item())
    return {
        "schema_version": "cognitive_v34_package_verification_v1",
        "status": "passed",
        "model_version": MODEL_VERSION,
        "package_root": workspace_relative(package_root),
        "manifest_sha256": sha256_file(manifest_path),
        "sha256sums_sha256": sha256_file(sums_path),
        "file_count": len([path for path in package_root.rglob("*") if path.is_file()]),
        "cpu_deterministic_logit": deterministic,
        "network_access": "forbidden",
    }


def _build_package(
    *,
    final_checkpoint: Path,
    calibration_path: Path,
    calibration_report_path: Path,
    training_report_path: Path,
    moca_statistics: Any,
    fixed_epochs: int,
    run_id: str,
) -> dict[str, Any]:
    PACKAGE_ROOT.mkdir(parents=True, exist_ok=True)
    _copy_file(final_checkpoint, PACKAGE_ROOT / "fusion.pt")
    _copy_file(calibration_path, PACKAGE_ROOT / "calibration.json")
    _copy_file(calibration_report_path, PACKAGE_ROOT / "calibration_report.json")
    _copy_file(training_report_path, PACKAGE_ROOT / "training_report.json")
    _copy_file(FIXED_SPLIT, PACKAGE_ROOT / "cogpic_subject_split_v33.json")
    _copy_file(CONFIG_PATH, PACKAGE_ROOT / "config_snapshot.yaml")
    _copy_file(CANDIDATE_CONFIG, PACKAGE_ROOT / "candidate_config.json")
    _copy_file(LOCKFILE, PACKAGE_ROOT / "requirements-cognitive-v3.3.lock")
    _copy_file(V33_PACKAGE_ROOT / "asr_package_ref.json", PACKAGE_ROOT / "asr_package_ref.json")
    _copy_directory(V33_PACKAGE_ROOT / "encoders", PACKAGE_ROOT / "encoders")
    _copy_directory(V33_PACKAGE_ROOT / "detectors", PACKAGE_ROOT / "detectors")

    checkpoint_sha = sha256_file(PACKAGE_ROOT / "fusion.pt")
    feature_stats = {
        "schema_version": "cognitive_feature_stats_v1",
        "feature_version": "cognitive_features_v3.3.0",
        "text_quality_version": "cognitive_text_quality_v3.3.1",
        "moca_train_subject_mean": float(moca_statistics.mean),
        "moca_train_subject_std": float(moca_statistics.std),
        "moca_train_subject_count": int(moca_statistics.subject_count),
        "moca_std_fallback_used": bool(float(moca_statistics.std) == 1.0),
        "checkpoint_sha256": checkpoint_sha,
    }
    _atomic_bytes(PACKAGE_ROOT / "feature_stats.json", _json_bytes(feature_stats))
    manifest = {
        "schema_version": "cognitive_model_manifest_v2",
        "model_version": MODEL_VERSION,
        "feature_version": "cognitive_features_v3.3.0",
        "text_quality_version": "cognitive_text_quality_v3.3.1",
        "source_checkpoint": workspace_relative(final_checkpoint),
        "source_checkpoint_sha256": checkpoint_sha,
        "candidate": {
            "id": CANDIDATE_ID,
            "feature_variant": "baseline",
            "heads": "main_head",
            "primary_modalities": ["audio", "text"],
            "face_in_primary_score": False,
            "calibration_method": "platt",
        },
        "training": {
            "dataset": "CogPic",
            "split_file": "cogpic_subject_split_v33.json",
            "split_sha256": sha256_file(PACKAGE_ROOT / "cogpic_subject_split_v33.json"),
            "train_subjects": 367,
            "calibration_subjects": 92,
            "official_test_subjects_used": 0,
            "fixed_epochs": int(fixed_epochs),
            "run_id": run_id,
        },
        "model_structure": {
            "projection_dim": 256,
            "fusion_hidden_dims": [128, 64],
            "dropout": 0.2,
            "input_dims": {"audio": 768, "text": 768, "face": 512},
        },
        "supported_api_modalities": ["audio+text+face", "audio+text", "audio+face", "audio"],
        "product_level_boundaries": {"normal": 40.0, "high_attention": 70.0},
        "artifacts": {
            "fusion": {"path": "fusion.pt", "sha256": checkpoint_sha},
            "calibration": {
                "path": "calibration.json",
                "sha256": sha256_file(PACKAGE_ROOT / "calibration.json"),
            },
            "feature_stats": {
                "path": "feature_stats.json",
                "sha256": sha256_file(PACKAGE_ROOT / "feature_stats.json"),
            },
            "candidate_config": {
                "path": "candidate_config.json",
                "sha256": sha256_file(PACKAGE_ROOT / "candidate_config.json"),
            },
        },
        "network_access": "forbidden",
        "official_test_evaluated": False,
    }
    _atomic_bytes(PACKAGE_ROOT / "manifest.json", _json_bytes(manifest))
    _atomic_bytes(PACKAGE_ROOT / "sha256sums.txt", _package_sums().encode("ascii"))
    return verify_package()


def run_package(*, run_id: str, device_name: str) -> dict[str, Any]:
    candidate_config = json.loads(CANDIDATE_CONFIG.read_text(encoding="utf-8"))
    if (
        candidate_config.get("model_version") != MODEL_VERSION
        or candidate_config.get("audio_text_candidate_id") != CANDIDATE_ID
        or candidate_config.get("feature_variant") != "baseline"
        or candidate_config.get("face_in_primary_score") is not False
        or candidate_config.get("calibration_method") != "platt"
        or candidate_config.get("official_test_evaluated") is not False
    ):
        raise V34PackageError("V3.4 candidate config is not the frozen final candidate")
    calibration_decision = json.loads(CALIBRATION_REPORT.read_text(encoding="utf-8"))
    if calibration_decision.get("status") != "passed" or calibration_decision.get(
        "official_test_evaluated"
    ) is not False:
        raise V34PackageError("OPT-COG-004 report did not pass")
    a2_summary = json.loads(A2_SUMMARY.read_text(encoding="utf-8"))
    fixed_epochs = fixed_epoch_median(a2_summary["best_epochs"])
    split = _load_split()
    config = load_config(CONFIG_PATH)
    config = copy.deepcopy(config)
    config["training"] = copy.deepcopy(config["training"])
    config["training"]["seed"] = int(config["seed"])
    config["training"]["fixed_epochs"] = fixed_epochs
    config["model"] = copy.deepcopy(config["model"])
    config["model"]["loss_weights"] = _candidate_loss_weights(
        config["model"]["loss_weights"], "main_head"
    )
    config["candidate"] = {
        "candidate_id": CANDIDATE_ID,
        "modalities": list(CANDIDATES[CANDIDATE_ID]["modalities"]),
        "heads": "main_head",
        "feature_variant": "baseline",
    }
    configure_determinism(config)
    device = _device(device_name)
    records, frozen_hashes = load_official_train_pool(verify_hashes=True)
    train_subjects = split["splits"]["train"]
    validation_subjects = split["splits"]["validation"]
    train_dataset = SubjectFeatureDataset(
        records, train_subjects, available_modalities=("audio", "text")
    )
    validation_dataset = SubjectFeatureDataset(
        records, validation_subjects, available_modalities=("audio", "text")
    )
    if len(train_dataset) != 1101 or len(validation_dataset) != 276:
        raise V34PackageError("fixed deployment task counts mismatch")
    weights = compute_training_weights(train_dataset.records)
    moca_statistics = compute_moca_statistics(train_dataset.records)
    input_hashes = {
        **frozen_hashes,
        "fixed_split": sha256_file(FIXED_SPLIT),
        "v34_config": sha256_file(CONFIG_PATH),
        "candidate_config": sha256_file(CANDIDATE_CONFIG),
        "feature_selection": sha256_file(FEATURE_SELECTION),
        "calibration_decision": sha256_file(CALIBRATION_REPORT),
        "a2_cv_summary": sha256_file(A2_SUMMARY),
    }
    run_dir = REPORT_ROOT / "runs" / run_id / FINAL_COMPONENT
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "run_manifest.json"
    final_checkpoint = run_dir / "final_checkpoint.pt"
    calibration_path = run_dir / "calibration.json"
    calibration_report_path = run_dir / "calibration_report.json"
    training_report_path = run_dir / "training_report.json"
    if final_checkpoint.is_file() and calibration_path.is_file() and training_report_path.is_file():
        return _build_package(
            final_checkpoint=final_checkpoint,
            calibration_path=calibration_path,
            calibration_report_path=calibration_report_path,
            training_report_path=training_report_path,
            moca_statistics=moca_statistics,
            fixed_epochs=fixed_epochs,
            run_id=run_id,
        )

    model = CognitiveChangeClueModel(projection_dim=256, fusion_hidden_dims=(128, 64), dropout=0.2).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
        betas=tuple(float(value) for value in config["training"]["adam_betas"]),
        eps=float(config["training"]["adam_eps"]),
    )
    amp_enabled = bool(config["training"]["amp"]) and device.type == "cuda"
    scaler = _new_grad_scaler(amp_enabled)
    latest = run_dir / "latest_checkpoint.pt"
    start_epoch = 1
    recoveries: list[dict[str, Any]] = []
    if latest.is_file():
        restored = _restore_training_checkpoint(
            latest,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            run_id=run_id,
            fixed_epochs=fixed_epochs,
            input_hashes=input_hashes,
        )
        start_epoch = int(restored["next_epoch"])
        recovery = recover_grad_scaler_after_recorded_overflow(
            scaler,
            run_dir=run_dir,
            next_epoch=start_epoch,
            amp_enabled=amp_enabled,
        )
        if recovery:
            recoveries.append(recovery)
    write_json(
        manifest_path,
        {
            "schema_version": "cognitive_v34_final_run_v1",
            "status": "running",
            "task_id": "PKG-COG-V34",
            "run_id": run_id,
            "component_id": FINAL_COMPONENT,
            "device": str(device),
            "amp_enabled": amp_enabled,
            "fixed_epochs": fixed_epochs,
            "resume_from_epoch": start_epoch if start_epoch > 1 else None,
            "recoveries": recoveries,
            "subjects": {"train": 367, "calibration": 92, "official_test": 0},
            "tasks": {"train": 1101, "calibration": 276, "official_test": 0},
            "input_hashes": input_hashes,
            "git": _git_state(),
            "started_or_resumed_at": datetime.now(timezone.utc).astimezone().isoformat(),
            "official_test_evaluated": False,
        },
    )
    started = time.perf_counter()
    completed_epoch = start_epoch - 1
    try:
        for epoch in range(start_epoch, fixed_epochs + 1):
            epoch_started = time.perf_counter()
            masking_plan, masking_rows = _candidate_masking_plan(
                train_dataset.records,
                modalities=("audio", "text"),
                epoch=epoch,
                seed_base=int(config["masking"]["seed_base"]),
            )
            write_jsonl(run_dir / "masking" / f"epoch_{epoch:03d}.jsonl", masking_rows)
            metrics = train_one_epoch(
                model,
                _loader(train_dataset, config, train=True, epoch=epoch),
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                masking_plan=masking_plan,
                weights=weights,
                moca_statistics=moca_statistics,
                loss_weights=config["model"]["loss_weights"],
                accumulation_steps=int(config["training"]["gradient_accumulation_steps"]),
                max_grad_norm=float(config["training"]["max_grad_norm"]),
                amp_enabled=amp_enabled,
            )
            checkpoint = _training_checkpoint(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                run_id=run_id,
                fixed_epochs=fixed_epochs,
                config=config,
                input_hashes=input_hashes,
                weights=weights,
                moca_statistics=moca_statistics,
            )
            atomic_torch_save(latest, checkpoint)
            record = {
                "schema_version": "cognitive_v34_final_epoch_v1",
                "epoch": epoch,
                "train": metrics,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "duration_seconds": time.perf_counter() - epoch_started,
            }
            existing = (
                [json.loads(line) for line in (run_dir / "epoch_metrics.jsonl").read_text(encoding="utf-8").splitlines() if line]
                if (run_dir / "epoch_metrics.jsonl").is_file()
                else []
            )
            write_jsonl(
                run_dir / "epoch_metrics.jsonl",
                [*([row for row in existing if int(row["epoch"]) < epoch]), record],
            )
            print(json.dumps(record, ensure_ascii=False, allow_nan=False), flush=True)
            completed_epoch = epoch
    except Exception as exc:
        record_failure(run_dir, manifest_path, epoch=completed_epoch + 1, error=exc)
        raise

    final_payload = torch.load(latest, map_location="cpu", weights_only=False)
    atomic_torch_save(final_checkpoint, final_payload)
    checkpoint_sha = sha256_file(final_checkpoint)
    model.load_state_dict(final_payload["model_state_dict"])
    model.to(device).eval()
    validation_rows = predict_raw(
        model,
        validation_dataset,
        config=config,
        device=device,
        candidate_id=CANDIDATE_ID,
        outer_fold=-1,
        split_role="final_validation",
    )
    write_jsonl(run_dir / "final_validation_raw_predictions.jsonl", validation_rows)
    calibration, calibration_report = _calibrate(validation_rows, checkpoint_sha)
    write_json(calibration_path, calibration)
    write_json(calibration_report_path, calibration_report)
    training_report = {
        "schema_version": "cognitive_v34_final_training_report_v1",
        "status": "passed",
        "task_id": "PKG-COG-V34",
        "run_id": run_id,
        "model_version": MODEL_VERSION,
        "candidate_id": CANDIDATE_ID,
        "fixed_epoch_source": workspace_relative(A2_SUMMARY),
        "five_fold_best_epochs": [int(value) for value in a2_summary["best_epochs"]],
        "fixed_epochs": fixed_epochs,
        "train_subjects": 367,
        "train_tasks": 1101,
        "calibration_subjects": 92,
        "calibration_tasks": 276,
        "checkpoint_sha256": checkpoint_sha,
        "calibration_sha256": sha256_file(calibration_path),
        "input_hashes": input_hashes,
        "duration_seconds": time.perf_counter() - started,
        "official_test_evaluated": False,
        "limitations": [
            "The fixed 92-subject split is a final package implementation record, not a new independent Test.",
            "No official Test media or predictions were read or generated.",
        ],
    }
    write_json(training_report_path, training_report)
    run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_manifest.update(
        {
            "status": "completed",
            "completed_at": datetime.now(timezone.utc).astimezone().isoformat(),
            "final_checkpoint_sha256": checkpoint_sha,
            "training_report_sha256": sha256_file(training_report_path),
            "calibration_report_sha256": sha256_file(calibration_report_path),
        }
    )
    write_json(manifest_path, run_manifest)
    return _build_package(
        final_checkpoint=final_checkpoint,
        calibration_path=calibration_path,
        calibration_report_path=calibration_report_path,
        training_report_path=training_report_path,
        moca_statistics=moca_statistics,
        fixed_epochs=fixed_epochs,
        run_id=run_id,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.verify and not args.run_id:
        raise V34PackageError("--run-id is required when building the V3.4 package")
    result = (
        verify_package()
        if args.verify
        else run_package(run_id=str(args.run_id), device_name=args.device)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "V34PackageError",
    "fixed_epoch_median",
    "main",
    "run_package",
    "verify_package",
]
