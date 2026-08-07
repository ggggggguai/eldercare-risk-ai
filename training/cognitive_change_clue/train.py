"""Deterministic CogPic training entry point for MODEL-COG-001."""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import math
import os
import platform
import random
import subprocess
import sys
import tempfile
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import yaml
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.model import (
    CognitiveChangeClueModel,
    model_state_is_finite,
)

try:
    from .common import (
        ALGORITHM_ROOT,
        DEFAULT_PROCESSED_ROOT,
        FEATURE_VERSION,
        WORKSPACE_ROOT,
        atomic_write_bytes,
        sha256_file,
        workspace_relative,
    )
    from .dataset import (
        EXPECTED_INPUT_SHA256,
        MODALITY_ORDER,
        TEXT_QUALITY_VERSION,
        CognitiveFeatureDataset,
        MocaStatistics,
        TrainingWeights,
        cognitive_collate,
        compute_moca_statistics,
        compute_training_weights,
        default_input_paths,
        verify_frozen_input_hashes,
    )
    from .evaluate import (
        COMBINATION_MODALITIES,
        checkpoint_is_better,
        filter_valid_rows,
        move_batch_to_device,
        predict_validation,
        subject_metrics,
    )
except ImportError:
    from common import (  # type: ignore[no-redef]
        ALGORITHM_ROOT,
        DEFAULT_PROCESSED_ROOT,
        FEATURE_VERSION,
        WORKSPACE_ROOT,
        atomic_write_bytes,
        sha256_file,
        workspace_relative,
    )
    from dataset import (  # type: ignore[no-redef]
        EXPECTED_INPUT_SHA256,
        MODALITY_ORDER,
        TEXT_QUALITY_VERSION,
        CognitiveFeatureDataset,
        MocaStatistics,
        TrainingWeights,
        cognitive_collate,
        compute_moca_statistics,
        compute_training_weights,
        default_input_paths,
        verify_frozen_input_hashes,
    )
    from evaluate import (  # type: ignore[no-redef]
        COMBINATION_MODALITIES,
        checkpoint_is_better,
        filter_valid_rows,
        move_batch_to_device,
        predict_validation,
        subject_metrics,
    )


CONFIG_PATH = ALGORITHM_ROOT / "configs" / "modules" / "cognitive_change_clue_v3_3.yaml"
REPORT_ROOT = ALGORITHM_ROOT / "reports" / "cognitive_change_clue" / "v3.3.0"
MODEL_SCHEMA_VERSION = "cognitive_training_checkpoint_v1"
RUN_SCHEMA_VERSION = "cognitive_training_run_v1"

FROZEN_CONFIG = {
    "feature_version": FEATURE_VERSION,
    "model.projection_dim": 256,
    "model.fusion_hidden_dims": [128, 64],
    "model.dropout": 0.2,
    "model.loss_weights": {
        "hc_vs_non_hc": 1.0,
        "mci_vs_hc": 0.3,
        "ad_mci_hc": 0.3,
        "moca": 0.1,
    },
    "training.seed": 20260802,
    "training.deterministic": True,
    "training.cublas_workspace_config": ":4096:8",
    "training.micro_batch_size": 4,
    "training.gradient_accumulation_steps": 4,
    "training.num_workers": 0,
    "training.train_shuffle": True,
    "training.train_drop_last": False,
    "training.validation_shuffle": False,
    "training.validation_drop_last": False,
    "training.max_epochs": 50,
    "training.learning_rate": 1e-4,
    "training.weight_decay": 1e-4,
    "training.adam_betas": [0.9, 0.999],
    "training.adam_eps": 1e-8,
    "training.max_grad_norm": 1.0,
    "training.scheduler.mode": "max",
    "training.scheduler.factor": 0.5,
    "training.scheduler.patience": 3,
    "training.scheduler.threshold": 1e-4,
    "training.scheduler.threshold_mode": "abs",
    "training.scheduler.min_lr": 1e-6,
    "masking.seed_base": 20260802,
    "masking.pair_combinations": ["audio_text", "audio_face", "text_face"],
    "masking.single_combinations": ["audio", "text"],
    "validation.checkpoint_auc_tolerance": 1e-6,
    "validation.early_stopping_min_delta": 1e-4,
    "validation.early_stopping_patience": 8,
    "validation.ablation_combinations": list(COMBINATION_MODALITIES),
}


class CognitiveTrainingError(RuntimeError):
    pass


def load_and_validate_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise CognitiveTrainingError("training config must be a mapping")
    for dotted, expected in FROZEN_CONFIG.items():
        actual: Any = config
        for part in dotted.split("."):
            if not isinstance(actual, Mapping) or part not in actual:
                raise CognitiveTrainingError(f"missing frozen config field: {dotted}")
            actual = actual[part]
        if isinstance(expected, float):
            matches = isinstance(actual, (int, float)) and math.isclose(
                float(actual), expected, rel_tol=0.0, abs_tol=1e-12
            )
        else:
            matches = actual == expected
        if not matches:
            raise CognitiveTrainingError(
                f"frozen config mismatch for {dotted}: {actual!r} != {expected!r}"
            )
    return config


def configure_determinism(config: Mapping[str, Any]) -> None:
    training = config["training"]
    seed = int(training["seed"])
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = str(training["cublas_workspace_config"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(bool(training["deterministic"]))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def build_masking_plan(
    records: Sequence[Mapping[str, Any]], *, epoch: int, seed_base: int
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    if epoch < 1:
        raise ValueError("epoch is 1-based and must be positive")
    rng = np.random.Generator(np.random.PCG64(int(seed_base) + int(epoch)))
    plan: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    pair = ("audio_text", "audio_face", "text_face")
    single = ("audio", "text")
    for record in sorted(records, key=lambda item: int(item["manifest_index"])):
        manifest_index = int(record["manifest_index"])
        remainder = manifest_index % 10
        if remainder <= 6:
            combination = "audio_text_face"
        elif remainder <= 8:
            combination = pair[int(rng.integers(0, 3))]
        else:
            combination = single[int(rng.integers(0, 2))]
        selected = COMBINATION_MODALITIES[combination]
        artificial = np.asarray(
            [name not in selected for name in MODALITY_ORDER], dtype=np.bool_
        )
        original = np.asarray(record["missing_mask"], dtype=np.bool_)
        final = original | artificial
        effective = "none" if bool(final.all()) else "_".join(
            name for name, missing in zip(MODALITY_ORDER, final) if not missing
        )
        payload = {
            "sample_id": str(record["sample_id"]),
            "global_manifest_index": manifest_index,
            "target_combination": combination,
            "final_missing_mask": [int(value) for value in final],
            "effective_combination": effective,
        }
        plan[str(record["sample_id"])] = payload
        rows.append(payload)
    return plan, rows


def apply_masking_plan(batch: Mapping[str, Any], plan: Mapping[str, Mapping[str, Any]]) -> Tensor:
    masks = []
    for sample_id in batch["sample_id"]:
        if str(sample_id) not in plan:
            raise CognitiveTrainingError(f"masking plan lacks sample: {sample_id}")
        masks.append(plan[str(sample_id)]["final_missing_mask"])
    return torch.tensor(masks, dtype=torch.bool, device=batch["missing_mask"].device)


def compute_multitask_loss(
    output: Any,
    batch: Mapping[str, Any],
    *,
    weights: TrainingWeights,
    moca_statistics: MocaStatistics,
    loss_weights: Mapping[str, float],
) -> tuple[Tensor, dict[str, Tensor | None]]:
    device = output.hc_vs_non_hc_logit.device
    main_loss = nn.functional.binary_cross_entropy_with_logits(
        output.hc_vs_non_hc_logit,
        batch["hc_vs_non_hc"],
        pos_weight=torch.tensor(weights.hc_vs_non_hc_pos_weight, device=device),
        reduction="mean",
    )
    components: dict[str, Tensor | None] = {"hc_vs_non_hc": main_loss}

    mci_mask = batch["mci_vs_hc_mask"].bool()
    components["mci_vs_hc"] = (
        nn.functional.binary_cross_entropy_with_logits(
            output.mci_vs_hc_logit[mci_mask],
            batch["mci_vs_hc"][mci_mask],
            pos_weight=torch.tensor(weights.mci_vs_hc_pos_weight, device=device),
            reduction="mean",
        )
        if bool(mci_mask.any())
        else None
    )
    components["ad_mci_hc"] = nn.functional.cross_entropy(
        output.ad_mci_hc_logits,
        batch["ad_mci_hc"],
        weight=torch.tensor(weights.ad_mci_hc_class_weights, device=device),
        reduction="mean",
    )
    moca_mask = batch["moca_mask"].bool()
    components["moca"] = (
        nn.functional.smooth_l1_loss(
            output.moca_standardized[moca_mask],
            (batch["moca"][moca_mask] - moca_statistics.mean) / moca_statistics.std,
            beta=1.0,
            reduction="mean",
        )
        if bool(moca_mask.any())
        else None
    )
    total = main_loss.new_zeros(())
    for name, value in components.items():
        if value is not None:
            total = total + float(loss_weights[name]) * value
    return total, components


def train_one_epoch(
    model: nn.Module,
    loader: Iterable[Mapping[str, Any]],
    *,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    masking_plan: Mapping[str, Mapping[str, Any]],
    weights: TrainingWeights,
    moca_statistics: MocaStatistics,
    loss_weights: Mapping[str, float],
    accumulation_steps: int,
    max_grad_norm: float,
    amp_enabled: bool,
) -> dict[str, Any]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    component_values: dict[str, list[float]] = defaultdict(list)
    skipped_rows = 0
    nonempty_micro_batches = 0
    optimizer_steps = 0
    pending = 0
    batches = list(loader)
    for batch_index, raw_batch in enumerate(batches):
        batch = move_batch_to_device(raw_batch, device)
        final_missing = apply_masking_plan(batch, masking_plan)
        filtered, valid = filter_valid_rows(batch, final_missing)
        skipped_rows += int((~valid).sum().item())
        if not bool(valid.any()):
            continue
        nonempty_micro_batches += 1
        pending += 1
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if amp_enabled
            else contextlib.nullcontext()
        )
        with autocast:
            output = model(filtered["features"], filtered["quality"], filtered["missing_mask"])
            _require_finite_outputs(output)
            loss, components = compute_multitask_loss(
                output,
                filtered,
                weights=weights,
                moca_statistics=moca_statistics,
                loss_weights=loss_weights,
            )
        if not bool(torch.isfinite(loss)):
            raise CognitiveTrainingError(f"non-finite loss at micro-batch {batch_index}")
        losses.append(float(loss.detach().cpu()))
        for name, value in components.items():
            if value is not None:
                component_values[name].append(float(value.detach().cpu()))
        scaler.scale(loss / accumulation_steps).backward()

        is_last_loader_batch = batch_index + 1 == len(batches)
        if pending == accumulation_steps or is_last_loader_batch:
            scaler.unscale_(optimizer)
            if pending < accumulation_steps:
                scale = accumulation_steps / pending
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(scale)
            _require_finite_gradients(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            _require_finite_gradients(model)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if not model_state_is_finite(model):
                raise CognitiveTrainingError("model parameters became non-finite")
            optimizer_steps += 1
            pending = 0
    if pending:
        scaler.unscale_(optimizer)
        scale = accumulation_steps / pending
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(scale)
        _require_finite_gradients(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        _require_finite_gradients(model)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if not model_state_is_finite(model):
            raise CognitiveTrainingError("model parameters became non-finite")
        optimizer_steps += 1
    if not losses:
        raise CognitiveTrainingError("epoch contained no trainable micro-batches")
    return {
        "loss": float(np.mean(losses)),
        "head_losses": {
            name: (float(np.mean(values)) if values else None)
            for name, values in sorted(component_values.items())
        },
        "nonempty_micro_batches": nonempty_micro_batches,
        "optimizer_steps": optimizer_steps,
        "skipped_all_missing_rows": skipped_rows,
    }


def _require_finite_outputs(output: Any) -> None:
    for name in (
        "hc_vs_non_hc_logit",
        "mci_vs_hc_logit",
        "ad_mci_hc_logits",
        "moca_standardized",
        "gate_weights",
        "fused",
    ):
        if not bool(torch.isfinite(getattr(output, name)).all()):
            raise CognitiveTrainingError(f"model output is non-finite: {name}")


def _require_finite_gradients(model: nn.Module) -> None:
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
            raise CognitiveTrainingError(f"gradient is non-finite: {name}")


def _new_grad_scaler(enabled: bool) -> Any:
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def recover_grad_scaler_after_recorded_overflow(
    scaler: Any,
    *,
    run_dir: Path,
    next_epoch: int,
    amp_enabled: bool,
) -> dict[str, Any] | None:
    diagnostic_path = run_dir / "failure_diagnostic.json"
    if not amp_enabled or not diagnostic_path.is_file():
        return None
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    error = str(diagnostic.get("error", ""))
    if int(diagnostic.get("epoch") or -1) != int(next_epoch):
        return None
    if not error.startswith("gradient is non-finite:"):
        return None
    previous_scale = float(scaler.get_scale())
    new_scale = max(1.0, previous_scale * 0.5)
    state = scaler.state_dict() if hasattr(scaler, "state_dict") else None
    if state and "scale" in state:
        state["scale"] = new_scale
        scaler.load_state_dict(state)
    else:
        scaler.update(new_scale)
    return {
        "time": datetime.now(timezone.utc).astimezone().isoformat(),
        "reason": error,
        "rerun_epoch": int(next_epoch),
        "previous_scale": previous_scale,
        "new_scale": new_scale,
        "checkpoint_epoch": int(next_epoch) - 1,
    }


def atomic_torch_save(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        torch.save(value, temporary)
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        for attempt in range(20):
            try:
                os.replace(temporary, target)
                break
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(min(1.0, 0.05 * (attempt + 1)))
    finally:
        temporary.unlink(missing_ok=True)


def _json_bytes(value: Any, *, indent: int | None = 2) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=indent,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def write_json(path: str | Path, value: Any) -> None:
    atomic_write_bytes(path, _json_bytes(value))


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    payload = b"".join(_json_bytes(dict(row), indent=None) for row in rows)
    atomic_write_bytes(path, payload)


def append_jsonl(path: str | Path, row: Mapping[str, Any]) -> None:
    target = Path(path)
    existing = target.read_bytes() if target.is_file() else b""
    atomic_write_bytes(target, existing + _json_bytes(dict(row), indent=None))


def record_failure(
    run_dir: Path,
    run_manifest_path: Path,
    *,
    epoch: int | None,
    error: BaseException,
) -> None:
    diagnostic = {
        "schema_version": "cognitive_training_failure_v1",
        "time": datetime.now(timezone.utc).astimezone().isoformat(),
        "epoch": epoch,
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
        "latest_checkpoint": (
            workspace_relative(run_dir / "latest_checkpoint.pt")
            if (run_dir / "latest_checkpoint.pt").is_file()
            else None
        ),
    }
    write_json(run_dir / "failure_diagnostic.json", diagnostic)
    if run_manifest_path.is_file():
        manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        manifest["status"] = "failed"
        manifest["ended_at"] = diagnostic["time"]
        manifest["failed_epoch"] = epoch
        manifest["failure_diagnostic"] = workspace_relative(
            run_dir / "failure_diagnostic.json"
        )
        write_json(run_manifest_path, manifest)


def allocate_run_id(report_root: Path = REPORT_ROOT) -> str:
    date = datetime.now().strftime("%Y%m%d")
    run_root = report_root / "runs"
    used = {
        path.name
        for path in run_root.glob(f"COG-{date}-*")
        if path.is_dir()
    } if run_root.is_dir() else set()
    index = 1
    while f"COG-{date}-{index:03d}" in used:
        index += 1
    return f"COG-{date}-{index:03d}"


def _git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=WORKSPACE_ROOT, check=True, capture_output=True, text=True,
            encoding="utf-8", errors="replace"
        ).stdout.strip()
    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


def _device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise CognitiveTrainingError("CUDA was requested but is unavailable")
    return device


def _make_loader(
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


def _checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    scaler: Any,
    epoch: int,
    best_state: Mapping[str, Any],
    early_state: Mapping[str, Any],
    run_id: str,
    run_dir: Path,
    config: Mapping[str, Any],
    input_hashes: Mapping[str, str],
    weights: TrainingWeights,
    moca_statistics: MocaStatistics,
) -> dict[str, Any]:
    return {
        "schema_version": MODEL_SCHEMA_VERSION,
        "run_id": run_id,
        "epoch": epoch,
        "next_epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "checkpoint_selection_state": dict(best_state),
        "early_stopping_state": dict(early_state),
        "scheduler_selection_state": {
            "best": scheduler.best,
            "num_bad_epochs": scheduler.num_bad_epochs,
        },
        "best_checkpoint_path": workspace_relative(run_dir / "best_checkpoint.pt"),
        "rng_state": capture_rng_state(),
        "config": copy.deepcopy(dict(config)),
        "input_hashes": dict(input_hashes),
        "feature_version": FEATURE_VERSION,
        "text_quality_version": TEXT_QUALITY_VERSION,
        "input_dimensions": {"audio": 768, "text": 768, "face": 512},
        "training_weights": weights.to_dict(),
        "moca_statistics": moca_statistics.to_dict(),
        "model_structure": {
            "projection_dim": int(config["model"]["projection_dim"]),
            "fusion_hidden_dims": list(config["model"]["fusion_hidden_dims"]),
            "dropout": float(config["model"]["dropout"]),
        },
        "optimizer_config": {
            "name": "AdamW",
            "learning_rate": float(config["training"]["learning_rate"]),
            "weight_decay": float(config["training"]["weight_decay"]),
            "betas": list(config["training"]["adam_betas"]),
            "eps": float(config["training"]["adam_eps"]),
        },
    }


def _load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    scaler: Any,
    expected_run_id: str,
    expected_hashes: Mapping[str, str],
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != MODEL_SCHEMA_VERSION:
        raise CognitiveTrainingError("checkpoint schema version mismatch")
    if payload.get("run_id") != expected_run_id:
        raise CognitiveTrainingError("checkpoint run ID mismatch")
    if payload.get("input_hashes") != dict(expected_hashes):
        raise CognitiveTrainingError("checkpoint frozen input hashes mismatch")
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    scheduler.load_state_dict(payload["scheduler_state_dict"])
    scaler.load_state_dict(payload["scaler_state_dict"])
    restore_rng_state(payload["rng_state"])
    return payload


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    config = load_and_validate_config(args.config)
    configure_determinism(config)
    device = _device(args.device)
    amp_enabled = bool(config["training"]["amp"]) and device.type == "cuda"
    input_paths = default_input_paths(DEFAULT_PROCESSED_ROOT)
    input_hashes = verify_frozen_input_hashes(input_paths, EXPECTED_INPUT_SHA256)

    train_dataset = CognitiveFeatureDataset(split_name="train", verify_hashes=True)
    validation_dataset = CognitiveFeatureDataset(
        split_name="validation", complete_only=True, verify_hashes=True
    )
    weights = compute_training_weights(train_dataset.records)
    moca_statistics = compute_moca_statistics(train_dataset.records)

    resume_path = Path(args.resume).resolve() if args.resume else None
    if resume_path:
        run_dir = resume_path.parent
        run_id = args.run_id or run_dir.name
    else:
        run_id = args.run_id or allocate_run_id()
        run_dir = REPORT_ROOT / "runs" / run_id
        if run_dir.exists():
            raise CognitiveTrainingError(f"run directory already exists: {run_dir}")
        run_dir.mkdir(parents=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    config_snapshot = run_dir / "config_snapshot.yaml"
    if not config_snapshot.exists():
        atomic_write_bytes(
            config_snapshot,
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False).encode("utf-8"),
        )
    run_manifest_path = run_dir / "run_manifest.json"
    started_at = datetime.now(timezone.utc).astimezone().isoformat()
    run_manifest = {
        "schema_version": RUN_SCHEMA_VERSION,
        "run_id": run_id,
        "status": "running",
        "started_at": started_at,
        "ended_at": None,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "git": _git_state(),
        "device": {
            "requested": args.device,
            "resolved": str(device),
            "name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        },
        "determinism": {
            "enabled": bool(config["training"]["deterministic"]),
            "seed": int(config["training"]["seed"]),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "amp_enabled": amp_enabled,
        },
        "input_hashes": input_hashes,
        "code_hashes": {
            workspace_relative(path): sha256_file(path)
            for path in (
                Path(__file__),
                Path(__file__).with_name("dataset.py"),
                Path(__file__).with_name("evaluate.py"),
                ALGORITHM_ROOT / "src" / "elderly_monitoring" / "modules" / "mental_health" / "submodules" / "cognitive_change_clue" / "fusion.py",
                ALGORITHM_ROOT / "src" / "elderly_monitoring" / "modules" / "mental_health" / "submodules" / "cognitive_change_clue" / "model.py",
                Path(args.config),
            )
        },
        "data": {
            "train_tasks": len(train_dataset),
            "train_subjects": len({row["subject_id"] for row in train_dataset.records}),
            "validation_complete_tasks": len(validation_dataset),
            "validation_complete_subjects": len(
                {row["subject_id"] for row in validation_dataset.records}
            ),
        },
    }
    if run_manifest_path.is_file() and resume_path:
        previous = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        run_manifest["started_at"] = previous.get("started_at", started_at)
        run_manifest["resumed_at"] = started_at
    write_json(run_manifest_path, run_manifest)

    model = CognitiveChangeClueModel(
        projection_dim=int(config["model"]["projection_dim"]),
        fusion_hidden_dims=tuple(config["model"]["fusion_hidden_dims"]),
        dropout=float(config["model"]["dropout"]),
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
    scaler = _new_grad_scaler(amp_enabled)
    best_state: dict[str, Any] = {
        "auc": float("-inf"),
        "balanced_accuracy": float("-inf"),
        "epoch": None,
    }
    early_state: dict[str, Any] = {
        "best_auc": float("-inf"),
        "epochs_without_improvement": 0,
    }
    start_epoch = 1
    if resume_path:
        payload = _load_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            expected_run_id=run_id,
            expected_hashes=input_hashes,
        )
        start_epoch = int(payload["next_epoch"])
        best_state = dict(payload["checkpoint_selection_state"])
        early_state = dict(payload["early_stopping_state"])
        amp_recovery = recover_grad_scaler_after_recorded_overflow(
            scaler,
            run_dir=run_dir,
            next_epoch=start_epoch,
            amp_enabled=amp_enabled,
        )
        if amp_recovery is not None:
            run_manifest.setdefault("recoveries", []).append(amp_recovery)
            write_json(run_manifest_path, run_manifest)

    validation_loader = _make_loader(validation_dataset, config, train=False)
    epoch_log = run_dir / "epoch_metrics.jsonl"
    latest_path = run_dir / "latest_checkpoint.pt"
    best_path = run_dir / "best_checkpoint.pt"
    training_started = time.perf_counter()
    completed_epoch = start_epoch - 1
    for epoch in range(start_epoch, int(training["max_epochs"]) + 1):
        epoch_started = time.perf_counter()
        masking_plan, masking_rows = build_masking_plan(
            train_dataset.records,
            epoch=epoch,
            seed_base=int(config["masking"]["seed_base"]),
        )
        write_jsonl(run_dir / "masking" / f"epoch_{epoch:03d}.jsonl", masking_rows)
        train_loader = _make_loader(train_dataset, config, train=True, epoch=epoch)
        try:
            train_metrics = train_one_epoch(
                model,
                train_loader,
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
            predictions = predict_validation(
                model,
                validation_loader,
                device=device,
                moca_statistics=moca_statistics,
            )
            validation_metrics = subject_metrics(predictions)
        except Exception as exc:
            record_failure(
                run_dir, run_manifest_path, epoch=epoch, error=exc
            )
            raise
        main = validation_metrics["hc_vs_non_hc"]
        auc = main["roc_auc"]
        balanced = main["balanced_accuracy_at_0_5"]
        if auc is None or balanced is None or not math.isfinite(float(auc)):
            raise CognitiveTrainingError("main Validation AUC is not computable")
        checkpoint_better = checkpoint_is_better(
            float(auc),
            float(balanced),
            float(best_state["auc"]),
            float(best_state["balanced_accuracy"]),
            auc_tolerance=float(config["validation"]["checkpoint_auc_tolerance"]),
        )
        if checkpoint_better:
            best_state = {
                "auc": float(auc),
                "balanced_accuracy": float(balanced),
                "epoch": epoch,
            }

        scheduler.step(float(auc))
        min_delta = float(config["validation"]["early_stopping_min_delta"])
        if float(auc) > float(early_state["best_auc"]) + min_delta:
            early_state = {"best_auc": float(auc), "epochs_without_improvement": 0}
        else:
            early_state["epochs_without_improvement"] = (
                int(early_state["epochs_without_improvement"]) + 1
            )

        checkpoint = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_state=best_state,
            early_state=early_state,
            run_id=run_id,
            run_dir=run_dir,
            config=config,
            input_hashes=input_hashes,
            weights=weights,
            moca_statistics=moca_statistics,
        )
        atomic_torch_save(latest_path, checkpoint)
        if checkpoint_better:
            atomic_torch_save(best_path, checkpoint)

        target_counts: dict[str, int] = defaultdict(int)
        effective_counts: dict[str, int] = defaultdict(int)
        for row in masking_rows:
            target_counts[row["target_combination"]] += 1
            effective_counts[row["effective_combination"]] += 1
        epoch_record = {
            "schema_version": "cognitive_epoch_metrics_v1",
            "run_id": run_id,
            "epoch": epoch,
            "train": train_metrics,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "masking": {
                "target_counts": dict(sorted(target_counts.items())),
                "effective_counts": dict(sorted(effective_counts.items())),
            },
            "validation": {
                key: value for key, value in validation_metrics.items() if key != "subjects"
            },
            "checkpoint_selected": checkpoint_better,
            "best": dict(best_state),
            "scheduler": {
                "best": float(scheduler.best),
                "num_bad_epochs": int(scheduler.num_bad_epochs),
            },
            "early_stopping": dict(early_state),
            "duration_seconds": time.perf_counter() - epoch_started,
        }
        append_jsonl(epoch_log, epoch_record)
        completed_epoch = epoch
        run_manifest["latest_completed_epoch"] = epoch
        run_manifest["latest_checkpoint"] = workspace_relative(latest_path)
        run_manifest["latest_checkpoint_sha256"] = sha256_file(latest_path)
        run_manifest["best_epoch"] = best_state["epoch"]
        run_manifest["best_checkpoint"] = workspace_relative(best_path)
        run_manifest["best_checkpoint_sha256"] = sha256_file(best_path)
        write_json(run_manifest_path, run_manifest)
        print(json.dumps(epoch_record, ensure_ascii=False, allow_nan=False), flush=True)
        if int(early_state["epochs_without_improvement"]) >= int(
            config["validation"]["early_stopping_patience"]
        ):
            break

    if not best_path.is_file():
        raise CognitiveTrainingError("training did not produce best_checkpoint.pt")
    try:
        final = finalize_run(
            model=model,
            best_path=best_path,
            validation_dataset=validation_dataset,
            validation_loader=validation_loader,
            device=device,
            moca_statistics=moca_statistics,
            run_id=run_id,
            run_dir=run_dir,
            config=config,
            input_hashes=input_hashes,
            weights=weights,
            duration_seconds=max(
                time.perf_counter() - training_started,
                (
                    datetime.now(timezone.utc).astimezone()
                    - datetime.fromisoformat(str(run_manifest["started_at"]))
                ).total_seconds(),
            ),
            completed_epoch=completed_epoch,
        )
    except Exception as exc:
        record_failure(
            run_dir, run_manifest_path, epoch=completed_epoch, error=exc
        )
        raise
    run_manifest["status"] = "completed"
    run_manifest["ended_at"] = datetime.now(timezone.utc).astimezone().isoformat()
    run_manifest["best_checkpoint_sha256"] = final["best_checkpoint_sha256"]
    run_manifest["best_epoch"] = final["best_epoch"]
    write_json(run_manifest_path, run_manifest)
    return final


def finalize_run(
    *,
    model: nn.Module,
    best_path: Path,
    validation_dataset: Dataset[Any],
    validation_loader: Iterable[Mapping[str, Any]],
    device: torch.device,
    moca_statistics: MocaStatistics,
    run_id: str,
    run_dir: Path,
    config: Mapping[str, Any],
    input_hashes: Mapping[str, str],
    weights: TrainingWeights,
    duration_seconds: float,
    completed_epoch: int,
) -> dict[str, Any]:
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    checkpoint_sha256 = sha256_file(best_path)
    predictions = predict_validation(
        model,
        validation_loader,
        device=device,
        moca_statistics=moca_statistics,
        checkpoint_sha256=checkpoint_sha256,
    )
    repeat_predictions = predict_validation(
        model,
        validation_loader,
        device=device,
        moca_statistics=moca_statistics,
        checkpoint_sha256=checkpoint_sha256,
    )
    reload_deterministic = _json_bytes(predictions) == _json_bytes(repeat_predictions)
    if not reload_deterministic:
        raise CognitiveTrainingError("best checkpoint repeat inference is not deterministic")
    prediction_path = run_dir / "validation_predictions.jsonl"
    write_jsonl(prediction_path, predictions)
    metrics = subject_metrics(predictions)
    subject_report = {
        "schema_version": "cognitive_validation_subject_metrics_v1",
        "run_id": run_id,
        "checkpoint_sha256": checkpoint_sha256,
        "best_epoch": int(checkpoint["epoch"]),
        "aggregation": "arithmetic_mean_task_raw_logits_per_subject",
        "selection_population": "original_complete_audio_text_face_validation_tasks",
        "metrics": metrics,
        "limitations": [
            "Validation probabilities are uncalibrated.",
            "Official Test was not loaded or evaluated by MODEL-COG-001.",
        ],
    }
    subject_path = run_dir / "validation_subject_metrics.json"
    write_json(subject_path, subject_report)

    ablations: dict[str, Any] = {}
    for combination in config["validation"]["ablation_combinations"]:
        rows = predict_validation(
            model,
            validation_loader,
            device=device,
            moca_statistics=moca_statistics,
            forced_combination=str(combination),
            checkpoint_sha256=checkpoint_sha256,
        )
        combination_metrics = subject_metrics(rows)["hc_vs_non_hc"]
        ablations[str(combination)] = {
            "task_count": len(rows),
            "subject_count": combination_metrics["subject_count"],
            "roc_auc": combination_metrics["roc_auc"],
            "balanced_accuracy_at_0_5": combination_metrics[
                "balanced_accuracy_at_0_5"
            ],
            "reason": combination_metrics["reason"],
        }
    ablation_path = run_dir / "validation_modality_ablation.json"
    write_json(
        ablation_path,
        {
            "schema_version": "cognitive_validation_modality_ablation_v1",
            "run_id": run_id,
            "checkpoint_sha256": checkpoint_sha256,
            "population": "same_original_complete_validation_tasks",
            "participates_in_checkpoint_selection": False,
            "combinations": ablations,
        },
    )

    summary = {
        "schema_version": "cognitive_train_metrics_v1",
        "run_id": run_id,
        "status": "completed",
        "feature_version": FEATURE_VERSION,
        "text_quality_version": TEXT_QUALITY_VERSION,
        "input_hashes": dict(input_hashes),
        "training_weights": weights.to_dict(),
        "moca_statistics": moca_statistics.to_dict(),
        "completed_epochs": completed_epoch,
        "best_epoch": int(checkpoint["epoch"]),
        "best_checkpoint": workspace_relative(best_path),
        "best_checkpoint_sha256": checkpoint_sha256,
        "validation_predictions": workspace_relative(prediction_path),
        "validation_predictions_sha256": sha256_file(prediction_path),
        "validation_subject_metrics": workspace_relative(subject_path),
        "validation_subject_metrics_sha256": sha256_file(subject_path),
        "validation_modality_ablation": workspace_relative(ablation_path),
        "validation_modality_ablation_sha256": sha256_file(ablation_path),
        "validation_main": metrics["hc_vs_non_hc"],
        "reload_deterministic": reload_deterministic,
        "duration_seconds": duration_seconds,
        "official_test_evaluated": False,
        "platt_calibration_fitted": False,
    }
    run_summary_path = run_dir / "train_metrics.json"
    write_json(run_summary_path, summary)
    canonical = {
        "schema_version": "cognitive_train_metrics_pointer_v1",
        "run_id": run_id,
        "run_summary": workspace_relative(run_summary_path),
        "run_summary_sha256": sha256_file(run_summary_path),
        "best_checkpoint": workspace_relative(best_path),
        "best_checkpoint_sha256": checkpoint_sha256,
    }
    write_json(REPORT_ROOT / "train_metrics.json", canonical)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-id")
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def main() -> int:
    try:
        result = run_training(parse_args())
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
        raise
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CognitiveTrainingError",
    "allocate_run_id",
    "apply_masking_plan",
    "atomic_torch_save",
    "build_masking_plan",
    "capture_rng_state",
    "compute_multitask_loss",
    "configure_determinism",
    "finalize_run",
    "load_and_validate_config",
    "restore_rng_state",
    "recover_grad_scaler_after_recorded_overflow",
    "run_training",
    "train_one_epoch",
]
