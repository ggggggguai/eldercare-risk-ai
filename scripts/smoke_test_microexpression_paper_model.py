from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch
from torch.utils.data import DataLoader
import yaml


ALGORITHM_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ALGORITHM_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.manifold_graph import (  # noqa: E402
    sha256_path,
)
from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.paper_dataset import (  # noqa: E402
    PaperV2ArtifactDataset,
    collate_paper_v2,
)
from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.paper_mhssa_tgcn import (  # noqa: E402
    PaperMHSSATGCN,
    PaperMHSSATGCNConfig,
    load_paper_model_checkpoint,
)
from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.paper_training import (  # noqa: E402
    PaperTrainingConfig,
    build_paper_training_components,
    save_paper_model_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MODEL-ME-005 GPU smoke audit.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ALGORITHM_ROOT
        / "configs/training/microexpression_paper_v2/mhssa_tgcn_smic_paper_exact.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ALGORITHM_ROOT
        / "reports/microexpression/paper_v2/model_me_005_smoke",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def tensor_sha256(tensor: torch.Tensor) -> str:
    return sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("MODEL-ME-005 requires a real CUDA smoke run")
    config_path = args.config.resolve()
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    manifest_path = (ALGORITHM_ROOT / raw_config["data"]["artifact_manifest"]).resolve()
    au_prior_path = (ALGORITHM_ROOT / raw_config["graph_prior"]["path"]).resolve()
    records = load_jsonl(manifest_path)
    au_audit = json.loads(au_prior_path.read_text(encoding="utf-8"))
    au_adjacency = torch.tensor(au_audit["adjacency"], dtype=torch.float32)
    model_config = PaperMHSSATGCNConfig.from_mapping(raw_config["model"])
    training_config = PaperTrainingConfig.from_mapping(raw_config["training"])

    selected_records = records[: min(training_config.batch_size, 8)]
    dataset = PaperV2ArtifactDataset(selected_records, preload=True)
    loader = DataLoader(
        dataset,
        batch_size=len(dataset),
        shuffle=False,
        collate_fn=collate_paper_v2,
    )
    batch = next(iter(loader))
    device = torch.device("cuda")
    model = PaperMHSSATGCN(au_adjacency, model_config).to(device)
    optimizer, scheduler, loss_function = build_paper_training_components(
        model, training_config
    )
    patches = batch["patches"].to(device)
    masks = batch["masks"].to(device)
    keypoints = batch["keypoints"].to(device)
    labels = batch["labels"].to(device)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits, aux = model(patches, masks, keypoints, return_aux=True)
    loss = loss_function(logits, labels)
    loss.backward()
    gradients_finite = all(
        parameter.grad is None or torch.isfinite(parameter.grad).all().item()
        for parameter in model.parameters()
    )
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), training_config.gradient_clip_norm
    )
    optimizer.step()
    scheduler.step()
    torch.cuda.synchronize(device)
    duration_seconds = time.perf_counter() - started
    if not gradients_finite or not torch.isfinite(loss):
        raise RuntimeError("GPU smoke produced non-finite gradients or loss")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = (
        ALGORITHM_ROOT
        / "models/mental_health/facial_affect/paper_v2/model_me_005_smoke"
        / "paper_mhssa_smoke.pt"
    ).resolve()
    checkpoint_sha256 = save_paper_model_checkpoint(
        checkpoint_path,
        model=model,
        training_config=training_config,
        metadata={
            "task_id": "MODEL-ME-005",
            "run_type": "gpu_smoke_only",
            "final_metrics_claim": False,
            "artifact_manifest_sha256": sha256_path(manifest_path),
            "au_prior_sha256": sha256_path(au_prior_path),
            "config_sha256": sha256_path(config_path),
        },
    )
    loaded_model, checkpoint = load_paper_model_checkpoint(
        str(checkpoint_path), map_location=device
    )
    model.eval()
    loaded_model = loaded_model.to(device).eval()
    with torch.no_grad():
        expected = model(patches, masks, keypoints)
        actual = loaded_model(patches, masks, keypoints)
    strict_reload_verified = torch.equal(expected, actual)
    if not strict_reload_verified:
        raise RuntimeError("Strict checkpoint reload changed model outputs")

    source_paths = [
        ALGORITHM_ROOT
        / "src/elderly_monitoring/modules/mental_health/submodules"
        / "facial_affect_clue/paper_dataset.py",
        ALGORITHM_ROOT
        / "src/elderly_monitoring/modules/mental_health/submodules"
        / "facial_affect_clue/manifold_graph.py",
        ALGORITHM_ROOT
        / "src/elderly_monitoring/modules/mental_health/submodules"
        / "facial_affect_clue/paper_mhssa_tgcn.py",
        ALGORITHM_ROOT
        / "src/elderly_monitoring/modules/mental_health/submodules"
        / "facial_affect_clue/paper_training.py",
    ]
    audit = {
        "schema_version": "model_me_005_gpu_smoke_v2",
        "task_id": "MODEL-ME-005",
        "status": "pass",
        "scope": "implementation_smoke_not_final_evaluation",
        "final_metrics_claim": False,
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "sample_ids": batch["sample_ids"],
        "input_shape": list(patches.shape),
        "logits_shape": list(logits.shape),
        "loss": float(loss.detach().cpu()),
        "gradients_finite": gradients_finite,
        "adjacency_finite": bool(torch.isfinite(aux["adjacency"]).all().item()),
        "batch_graphs_independent": bool(
            not torch.equal(aux["adjacency"][0], aux["adjacency"][1])
        ),
        "alpha": float(aux["alpha"].detach().cpu()),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "duration_seconds": duration_seconds,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "output_tensor_sha256": tensor_sha256(actual),
        "checkpoint": {
            "path": checkpoint_path.as_posix(),
            "sha256": checkpoint_sha256,
            "strict_reload_verified": strict_reload_verified,
            "schema_version": checkpoint["model_schema_version"],
        },
        "inputs": {
            "artifact_manifest": {
                "path": manifest_path.as_posix(),
                "sha256": sha256_path(manifest_path),
            },
            "au_prior": {
                "path": au_prior_path.as_posix(),
                "sha256": sha256_path(au_prior_path),
                "source": au_audit["source"],
            },
            "config": {
                "path": config_path.as_posix(),
                "sha256": sha256_path(config_path),
            },
        },
        "source_sha256": {
            path.name: sha256_path(path.resolve()) for path in source_paths
        },
    }
    audit_path = output_dir / "audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "audit": audit_path.as_posix(),
                "audit_sha256": sha256_path(audit_path),
                "checkpoint": checkpoint_path.as_posix(),
                "checkpoint_sha256": checkpoint_sha256,
                "status": "pass",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
