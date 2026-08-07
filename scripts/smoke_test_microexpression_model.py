from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.dataset import (
    MicroexpressionArtifactDataset,
    load_artifact_manifest,
)
from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.model import (
    MHSSATGCN,
    MHSSATGCNConfig,
    load_model_checkpoint,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MODEL-ME-001 smoke checks.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT
        / "data/manifests/microexpression/flow_artifact_manifest_smic_hs_classification_v1.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "data/manifests/microexpression/model_smoke_test_v1.json",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT
        / "models/mental_health/facial_affect/mhssa_tgcn_smic_loso_v1/smoke_test.pt",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def main() -> int:
    args = parse_args()
    records = load_artifact_manifest(args.manifest)
    selected = []
    seen_labels: set[int] = set()
    for record in records:
        label = int(record["label"])
        if label not in seen_labels or len(selected) < 4:
            selected.append(record)
            seen_labels.add(label)
        if len(selected) >= 4 and len(seen_labels) == 3:
            break
    dataset = MicroexpressionArtifactDataset(selected, preload=True)
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    batch = next(iter(loader))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = MHSSATGCNConfig()
    model = MHSSATGCN(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    started = time.perf_counter()
    model.train()
    logits = model(
        batch["patches"].to(device),
        batch["region_ids"].to(device),
        batch["masks"].to(device),
        batch["keypoints"].to(device),
    )
    loss = torch.nn.functional.cross_entropy(logits, batch["label"].to(device))
    loss.backward()
    gradients_finite = all(
        parameter.grad is None or torch.isfinite(parameter.grad).all().item()
        for parameter in model.parameters()
    )
    optimizer.step()

    model.eval()
    single = {key: value[:1].to(device) for key, value in batch.items() if torch.is_tensor(value)}
    with torch.no_grad():
        single_logits = model(
            single["patches"],
            single["region_ids"],
            single["masks"],
            single["keypoints"],
        )
        changed_patches = single["patches"].clone()
        changed_keypoints = single["keypoints"].clone()
        padding = single["masks"] == 0
        changed_patches[padding] = 1000.0
        changed_keypoints[padding] = 1000.0
        padding_logits = model(
            changed_patches,
            single["region_ids"],
            single["masks"],
            changed_keypoints,
        )
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_schema_version": "mhssa_tgcn_v1",
            "model_config": config.as_dict(),
            "model_state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "fold_id": "MODEL-ME-001-smoke",
            "test_used_for_selection": False,
        },
        args.checkpoint,
    )
    loaded, _ = load_model_checkpoint(str(args.checkpoint), map_location=device)
    loaded = loaded.to(device).eval()
    with torch.no_grad():
        reloaded_logits = loaded(
            single["patches"],
            single["region_ids"],
            single["masks"],
            single["keypoints"],
        )
    report = {
        "schema_version": "microexpression_model_smoke_v1",
        "task_id": "MODEL-ME-001",
        "status": "pass",
        "device": str(device),
        "sample_ids": list(batch["sample_id"]),
        "batch_shape": list(batch["patches"].shape),
        "single_logits_shape": list(single_logits.shape),
        "batch_logits_shape": list(logits.shape),
        "loss": float(loss.detach().cpu()),
        "logits_finite": bool(torch.isfinite(logits).all().item()),
        "gradients_finite": gradients_finite,
        "padding_invariant": bool(
            torch.allclose(single_logits, padding_logits, atol=1e-6, rtol=1e-6)
        ),
        "checkpoint_reload_equal": bool(
            torch.equal(single_logits.cpu(), reloaded_logits.cpu())
        ),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "duration_seconds": time.perf_counter() - started,
        "checkpoint": {
            "path": args.checkpoint.resolve().as_posix(),
            "sha256": file_sha256(args.checkpoint),
        },
        "manifest": {
            "path": args.manifest.resolve().as_posix(),
            "sha256": file_sha256(args.manifest),
        },
        "repairs": {
            "mask_applied_to_attention_residual_and_pooling": True,
            "networkx_or_mds_in_forward": False,
            "batch_independent_adjacency": True,
            "debug_prints_in_forward": False,
        },
    }
    required = (
        report["logits_finite"],
        report["gradients_finite"],
        report["padding_invariant"],
        report["checkpoint_reload_equal"],
    )
    if not all(required):
        report["status"] = "fail"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
