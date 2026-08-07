from __future__ import annotations

import argparse
import gc
from hashlib import sha256
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable

import numpy as np
import torch


ALGORITHM_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ALGORITHM_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.dstm import (  # noqa: E402
    load_dstm_checkpoint,
)
from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.dstm_dataset import (  # noqa: E402
    DSTMArtifactCollator,
    DSTMArtifactDataset,
)
from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.fold_reducers import (  # noqa: E402
    FoldOnlyReducer,
)
from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.paper_dataset import (  # noqa: E402
    PaperV2ArtifactDataset,
    PaperV2Collator,
)
from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.paper_mhssa_tgcn import (  # noqa: E402
    load_paper_model_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark offline forward latency for paper-v2 MHSSA and DSTM."
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output",
        type=Path,
        default=ALGORITHM_ROOT
        / "reports/microexpression/paper_v2"
        / "MODEL-ME-006-DSTM-SMIC-NESTED-LOSO-APEX-20260807"
        / "latency_benchmark.json",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def benchmark_forward(
    function: Callable[[], torch.Tensor],
    *,
    device: torch.device,
    batch_size: int,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    if warmup < 1 or repeats < 1:
        raise ValueError("warmup and repeats must both be positive")
    with torch.inference_mode():
        for _ in range(warmup):
            logits = function()
            if logits.shape != (batch_size, 3) or not torch.isfinite(logits).all():
                raise RuntimeError("Latency benchmark produced invalid logits")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        durations_ms: list[float] = []
        for _ in range(repeats):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            logits = function()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            durations_ms.append((time.perf_counter() - started) * 1000.0)
    values = np.asarray(durations_ms, dtype=np.float64)
    return {
        "batch_size": batch_size,
        "warmup_iterations": warmup,
        "measured_iterations": repeats,
        "mean_batch_latency_ms": float(values.mean()),
        "std_batch_latency_ms": float(values.std()),
        "p50_batch_latency_ms": float(np.percentile(values, 50)),
        "p95_batch_latency_ms": float(np.percentile(values, 95)),
        "min_batch_latency_ms": float(values.min()),
        "max_batch_latency_ms": float(values.max()),
        "mean_per_sample_latency_ms": float(values.mean() / batch_size),
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
    }


def release_device_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA latency benchmark requested but CUDA is unavailable")

    spatial_manifest = (
        ALGORITHM_ROOT
        / "data/manifests/microexpression/paper_v2"
        / "flow_artifact_manifest_smic_hs_classification_combined_v2.jsonl"
    ).resolve()
    temporal_manifest = (
        ALGORITHM_ROOT
        / "data/manifests/microexpression/paper_v2"
        / "dstm_temporal_flow_artifact_manifest_smic_hs_v2.jsonl"
    ).resolve()
    au_prior_path = (
        ALGORITHM_ROOT
        / "data/manifests/microexpression/paper_v2"
        / "au_prior_casme_samm_reconstructed_v2.json"
    ).resolve()
    mhssa_checkpoint = (
        ALGORITHM_ROOT
        / "reports/microexpression/paper_v2"
        / "EVAL-ME-002-SMIC-NESTED-LOSO-20260806"
        / "final/seed_20260806/loso_s01/best.pt"
    ).resolve()
    dstm_root = (
        ALGORITHM_ROOT
        / "reports/microexpression/paper_v2"
        / "MODEL-ME-006-DSTM-SMIC-NESTED-LOSO-APEX-20260807"
    ).resolve()
    dstm_checkpoint = (
        dstm_root / "folds/loso_s01/final/seed_20260806/best.pt"
    ).resolve()
    reducer_path = (dstm_root / "folds/loso_s01/reducer/reducer.joblib").resolve()

    spatial_records = load_jsonl(spatial_manifest)
    temporal_records = load_jsonl(temporal_manifest)
    batch_size = min(args.batch_size, len(spatial_records))
    selected_records = spatial_records[:batch_size]
    selected_ids = [str(record["sample_id"]) for record in selected_records]
    temporal_by_id = {str(record["sample_id"]): record for record in temporal_records}
    selected_temporal = [temporal_by_id[sample_id] for sample_id in selected_ids]
    au_prior = json.loads(au_prior_path.read_text(encoding="utf-8"))
    au_adjacency = torch.as_tensor(au_prior["adjacency"], dtype=torch.float32)

    paper_dataset = PaperV2ArtifactDataset(selected_records, preload=True)
    paper_batch = PaperV2Collator()(
        [paper_dataset[index] for index in range(len(paper_dataset))]
    )
    paper_model, _ = load_paper_model_checkpoint(
        str(mhssa_checkpoint), map_location=device
    )
    paper_model = paper_model.to(device).eval()
    paper_patches = paper_batch["patches"].to(device)
    paper_masks = paper_batch["masks"].to(device)
    paper_keypoints = paper_batch["keypoints"].to(device)
    mhssa_result = benchmark_forward(
        lambda: paper_model(paper_patches, paper_masks, paper_keypoints),
        device=device,
        batch_size=batch_size,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    mhssa_result.update(
        {
            "model": "paper_v2_mhssa_tgcn",
            "parameter_count": sum(parameter.numel() for parameter in paper_model.parameters()),
            "checkpoint_path": mhssa_checkpoint.as_posix(),
            "checkpoint_sha256": sha256_file(mhssa_checkpoint),
            "input_contract": "43x75 spatial patches only",
        }
    )
    paper_model = None
    paper_dataset = None
    paper_batch = None
    paper_patches = None
    paper_masks = None
    paper_keypoints = None
    release_device_memory()

    dstm_dataset = DSTMArtifactDataset(
        selected_records, selected_temporal, sample_ids=selected_ids, preload=True
    )
    dstm_batch = DSTMArtifactCollator()(
        [dstm_dataset[index] for index in range(len(dstm_dataset))]
    )
    dstm_model, _ = load_dstm_checkpoint(
        dstm_checkpoint, au_adjacency=au_adjacency, map_location=device
    )
    dstm_model = dstm_model.to(device).eval()
    reducer = FoldOnlyReducer.load(reducer_path)
    dstm_patches = dstm_batch["patches"].to(device)
    dstm_masks = dstm_batch["masks"].to(device)
    flow_sequences = dstm_batch["flow_sequences"].to(device)
    temporal_mask = dstm_batch["temporal_mask"].to(device)
    dstm_result = benchmark_forward(
        lambda: dstm_model(
            dstm_patches,
            dstm_masks,
            flow_sequences=flow_sequences,
            temporal_mask=temporal_mask,
            temporal_reducer=reducer,
        ),
        device=device,
        batch_size=batch_size,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    dstm_result.update(
        {
            "model": "paper_v2_dstm",
            "parameter_count": sum(parameter.numel() for parameter in dstm_model.parameters()),
            "checkpoint_path": dstm_checkpoint.as_posix(),
            "checkpoint_sha256": sha256_file(dstm_checkpoint),
            "reducer_path": reducer_path.as_posix(),
            "reducer_sha256": sha256_file(reducer_path),
            "input_contract": "43x75 spatial patches + padded onset-to-apex flow sequence + fold-only UMAP",
            "max_sequence_length": int(flow_sequences.shape[1]),
        }
    )

    output = {
        "schema_version": "microexpression_paper_v2_latency_benchmark_v2",
        "task_id": "MODEL-ME-006",
        "status": "pass",
        "scope": "offline_artifact_forward_latency_not_s10_fps",
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "sample_ids": selected_ids,
        "inputs": {
            "spatial_manifest": {
                "path": spatial_manifest.as_posix(),
                "sha256": sha256_file(spatial_manifest),
            },
            "temporal_manifest": {
                "path": temporal_manifest.as_posix(),
                "sha256": sha256_file(temporal_manifest),
            },
        },
        "models": [mhssa_result, dstm_result],
        "limitations": [
            "DSTM includes GPU motion CNN, CPU UMAP transform, temporal Transformer, and spatial stream.",
            "This benchmark uses already-generated SMIC artifacts and does not include video decode, face alignment, apex spotting, or optical-flow generation.",
            "Results are not an S10 real-time FPS claim.",
        ],
    }
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    output_sha256 = sha256_file(output_path)
    output_path.with_suffix(output_path.suffix + ".sha256").write_text(
        output_sha256 + "\n", encoding="ascii"
    )
    print(
        json.dumps(
            {
                "output": output_path.as_posix(),
                "output_sha256": output_sha256,
                "mhssa": mhssa_result,
                "dstm": dstm_result,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
