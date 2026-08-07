from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.causalnet import (  # noqa: E402
    ARCHITECTURE_CORRECTION,
    CausalNet,
    CausalNetConfig,
    code_hash,
    load_causalnet_checkpoint,
    save_causalnet_checkpoint,
)
from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.causalnet_dataset import (  # noqa: E402
    CausalNetArtifactDataset,
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def finite_gradients(model: torch.nn.Module) -> tuple[bool, float]:
    maximum = 0.0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        if not torch.isfinite(parameter.grad).all():
            return False, maximum
        maximum = max(maximum, float(parameter.grad.detach().abs().max().cpu()))
    return True, maximum


def small_cpu_smoke() -> dict[str, Any]:
    config = CausalNetConfig(
        dim=32,
        heads=2,
        block_repeats=(1, 1, 1),
        cross_heads=4,
        cross_depth=1,
        mlp_mult=2,
    )
    model = CausalNet(config)
    inputs = torch.randn(2, 4, 3, 28, 28)
    labels = torch.tensor((0, 1), dtype=torch.long)
    model.train()
    logits = model(inputs)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()
    gradients_finite, gradient_max = finite_gradients(model)
    if tuple(logits.shape) != (2, 3) or not torch.isfinite(loss) or not gradients_finite:
        raise RuntimeError("Small CPU CausalNet smoke failed")
    return {
        "status": "pass",
        "config": config.as_dict(),
        "input_shape": list(inputs.shape),
        "logits_shape": list(logits.shape),
        "loss": float(loss.detach()),
        "gradient_max": gradient_max,
        "device": "cpu",
    }


def real_artifact_smoke(
    manifest_path: Path,
    *,
    device: torch.device,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, CausalNet]:
    records = load_manifest(manifest_path)
    dataset = CausalNetArtifactDataset(records, sample_ids=[str(records[0]["sample_id"])])
    item = dataset[0]
    inputs = item["inputs"].unsqueeze(0)
    labels = item["label"].reshape(1)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    model = CausalNet().to(device)
    model.train()
    logits = model(inputs.to(device))
    loss = torch.nn.functional.cross_entropy(logits, labels.to(device))
    loss.backward()
    gradients_finite, gradient_max = finite_gradients(model)
    if tuple(logits.shape) != (1, 3) or not torch.isfinite(loss) or not gradients_finite:
        raise RuntimeError("Production real-artifact CausalNet smoke failed")
    result = {
        "status": "pass",
        "sample_id": str(item["sample_id"]),
        "input_shape": list(inputs.shape),
        "logits_shape": list(logits.shape),
        "loss": float(loss.detach().cpu()),
        "gradient_max": gradient_max,
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cuda_total_memory_bytes": (
            torch.cuda.get_device_properties(device).total_memory
            if device.type == "cuda"
            else None
        ),
        "cuda_peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        "cuda_peak_reserved_bytes": (
            torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
        ),
    }
    return result, inputs, labels, model


def checkpoint_round_trip(
    model: CausalNet,
    inputs: torch.Tensor,
    path: Path,
    *,
    source_manifest_hash: str,
    preprocessing_config_hash: str,
) -> dict[str, Any]:
    source_hash = code_hash()
    metadata = save_causalnet_checkpoint(
        path,
        model,
        training_config={"smoke_only": True, "training_started": False, "batch_size": 1},
        source_manifest_hash=source_manifest_hash,
        preprocessing_config_hash=preprocessing_config_hash,
        source_code_hash=source_hash,
    )
    reloaded, payload = load_causalnet_checkpoint(path, map_location="cpu")
    model_cpu = model.to("cpu").eval()
    with torch.no_grad():
        first = model_cpu(inputs.cpu())
        second = reloaded.eval()(inputs.cpu())
    max_difference = float((first - second).abs().max())
    if max_difference != 0.0:
        raise RuntimeError(f"Checkpoint round trip changed logits: max diff {max_difference}")
    required = {
        "model_schema_version", "model_config", "model_state_dict", "training_config",
        "upstream_repository", "upstream_commit", "upstream_license", "architecture_correction",
        "upstream_bit_exact", "paper_reproduction_claim", "source_manifest_hash",
        "preprocessing_config_hash", "code_hash",
    }
    if required - set(payload):
        raise RuntimeError("Checkpoint metadata is incomplete")
    return {
        "status": "pass",
        "path": path.resolve().as_posix(),
        "sha256": sha256_file(path),
        "max_logit_difference": max_difference,
        "metadata": metadata,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CausalNet MODEL-ME-008 smoke checks")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "data/manifests/microexpression/causalnet_v1/causalnet_artifact_manifest_smic_hs_source_compatible_roi_v1.jsonl")
    parser.add_argument("--preprocess-summary", type=Path, default=PROJECT_ROOT / "data/manifests/microexpression/causalnet_v1/causalnet_preprocess_summary_smic_hs_source_compatible_roi_v1.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports/microexpression/causalnet_v1/MODEL-ME-008-CausalNet-GPU-SMOKE")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "schema_version": "causalnet_model_smoke_v1",
        "task_id": "MODEL-ME-008",
        "training_started": False,
        "final_metrics_claim": False,
        "system_integration_started": False,
        "upstream_repository": "https://github.com/tony19980810/CausalNet",
        "upstream_commit": "7bdf163face030eeae4358fc7638cf538305acce",
        "upstream_license": "MIT",
        "architecture_correction": ARCHITECTURE_CORRECTION,
        "upstream_bit_exact": False,
        "paper_reproduction_claim": False,
    }
    result["small_cpu_smoke"] = small_cpu_smoke()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; production GPU smoke is required")
    device = torch.device("cuda")
    real_result, inputs, labels, model = real_artifact_smoke(args.manifest, device=device)
    result["production_real_artifact_gpu_smoke"] = real_result
    summary = json.loads(args.preprocess_summary.read_text(encoding="utf-8"))
    checkpoint = checkpoint_round_trip(
        model,
        inputs,
        args.output_dir / "causalnet_model_me_008_smoke.pt",
        source_manifest_hash=str(summary["manifest_sha256"]),
        preprocessing_config_hash=sha256_file(PROJECT_ROOT / "configs/preprocessing/microexpression_causalnet_v1.yaml"),
    )
    result["checkpoint_smoke"] = checkpoint
    result["manifest"] = {
        "path": args.manifest.resolve().as_posix(),
        "sha256": sha256_file(args.manifest),
        "record_count": len(load_manifest(args.manifest)),
    }
    result["status"] = "pass"
    report_path = args.output_dir / "model_me_008_smoke_report.json"
    result["report_path"] = report_path.resolve().as_posix()
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
