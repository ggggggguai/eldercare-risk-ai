from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml


ALGORITHM_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ALGORITHM_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.dstm import (  # noqa: E402
    DSTM,
    DSTMConfig,
    DSTMTemporalMotionCNN,
)
from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.dstm_dataset import (  # noqa: E402
    DSTMArtifactDataset,
)
from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.dstm_training import (  # noqa: E402
    DSTMTrainingConfig,
    _loader,
    evaluate_loader,
    set_dstm_seed,
    train_dstm_fold,
)
from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.fold_reducers import (  # noqa: E402
    FoldOnlyReducer,
    FoldReducerConfig,
    sha256_file,
)
from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.paper_evaluation import (  # noqa: E402
    aggregate_seed_predictions,
)


DSTM_EVALUATION_SCHEMA_VERSION = "dstm_nested_loso_evaluation_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run strict Nested LOSO for paper-chapter-4 DSTM.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ALGORITHM_ROOT / "configs/training/microexpression_paper_v2/dstm_smic_paper_exact.yaml",
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--run-id", default="MODEL-ME-006-DSTM-SMIC-NESTED-LOSO-20260807")
    parser.add_argument("--fold-limit", type=int)
    parser.add_argument("--max-epochs-override", type=int)
    parser.add_argument("--bootstrap-iterations", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
    return sha256_file(path)


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)
    return sha256_file(path)


def sha256_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def load_temporal_sequence(record: Mapping[str, Any]) -> np.ndarray:
    with np.load(Path(str(record["artifact_path"])), allow_pickle=False) as artifact:
        sequence = np.asarray(artifact["flow_sequence"]).copy()
    if sequence.ndim != 4 or sequence.shape[1:] != (3, 32, 32):
        raise ValueError(f"Invalid temporal sequence for {record['sample_id']}: {sequence.shape}")
    return sequence.astype(np.float32)


def extract_motion_features(
    records: Sequence[Mapping[str, Any]],
    *,
    encoder: DSTMTemporalMotionCNN,
    device: torch.device,
    batch_frames: int = 256,
) -> dict[str, np.ndarray]:
    sequences = [(str(record["sample_id"]), load_temporal_sequence(record)) for record in records]
    flattened: list[torch.Tensor] = []
    lengths: list[int] = []
    for _, sequence in sequences:
        flattened.append(torch.from_numpy(sequence))
        lengths.append(int(sequence.shape[0]))
    all_frames = torch.cat(flattened, dim=0)
    outputs: list[torch.Tensor] = []
    encoder = encoder.to(device).eval()
    with torch.no_grad():
        for start in range(0, len(all_frames), batch_frames):
            outputs.append(encoder(all_frames[start : start + batch_frames].to(device)).cpu())
    all_features = torch.cat(outputs, dim=0).numpy().astype(np.float32)
    result: dict[str, np.ndarray] = {}
    offset = 0
    for (sample_id, _), length in zip(sequences, lengths, strict=True):
        result[sample_id] = all_features[offset : offset + length].copy()
        offset += length
    if offset != len(all_features):
        raise RuntimeError("Motion feature slicing did not cover all frames")
    return result


def fit_fold_reducer(
    *,
    fold: Mapping[str, Any],
    all_records: Mapping[str, Mapping[str, Any]],
    motion_features: Mapping[str, np.ndarray],
    reducer_config: FoldReducerConfig,
    output_dir: Path,
) -> tuple[FoldOnlyReducer, dict[str, np.ndarray], dict[str, Any]]:
    train_ids = [str(value) for value in fold["train_sample_ids"]]
    validation_ids = [str(value) for value in fold["validation_sample_ids"]]
    test_ids = [str(value) for value in fold["test_sample_ids"]]
    train_features = np.concatenate([motion_features[sample_id] for sample_id in train_ids], axis=0)
    train_sample_ids = [
        sample_id
        for sample_id in train_ids
        for _ in range(motion_features[sample_id].shape[0])
    ]
    train_subject_ids = [
        str(all_records[sample_id]["subject_id"])
        for sample_id in train_ids
        for _ in range(motion_features[sample_id].shape[0])
    ]
    test_subject_ids = [str(all_records[sample_id]["subject_id"]) for sample_id in test_ids]
    reducer = FoldOnlyReducer(reducer_config)
    reducer.fit(
        train_features,
        sample_ids=train_sample_ids,
        subject_ids=train_subject_ids,
        fold_id=str(fold["fold_id"]),
        forbidden_sample_ids=validation_ids + test_ids,
        forbidden_subject_ids=test_subject_ids,
    )
    transformed: dict[str, np.ndarray] = {}
    for sample_id, features in motion_features.items():
        transformed[sample_id] = reducer.transform(features)
    reducer_metadata = reducer.save(output_dir / "reducer.joblib")
    return reducer, transformed, reducer_metadata


def write_feature_store(
    path: Path,
    features: Mapping[str, np.ndarray],
    *,
    reducer_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    sample_ids = sorted(features)
    offsets = [0]
    arrays: list[np.ndarray] = []
    for sample_id in sample_ids:
        value = np.asarray(features[sample_id], dtype=np.float32)
        arrays.append(value)
        offsets.append(offsets[-1] + value.shape[0])
    concatenated = np.concatenate(arrays, axis=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            features=concatenated,
            offsets=np.asarray(offsets, dtype=np.int64),
            sample_ids=np.asarray(sample_ids),
        )
    temporary.replace(path)
    metadata = {
        "schema_version": "dstm_fold_temporal_feature_store_v1",
        "sample_count": len(sample_ids),
        "frame_count": int(concatenated.shape[0]),
        "feature_dim": int(concatenated.shape[1]),
        "sample_ids": sample_ids,
        "reducer_metadata": dict(reducer_metadata),
        "feature_store_path": path.resolve().as_posix(),
        "feature_store_sha256": sha256_file(path),
    }
    write_json(path.with_suffix(path.suffix + ".json"), metadata)
    return metadata


def release_decision(seed_summaries: Sequence[Mapping[str, Any]]) -> str:
    uf1_values = [float(summary["pooled"]["uf1"]) for summary in seed_summaries]
    uar_values = [float(summary["pooled"]["uar"]) for summary in seed_summaries]
    mean_uf1 = float(np.mean(uf1_values))
    mean_uar = float(np.mean(uar_values))
    no_zero_folds = all(not summary["zero_score_folds"] for summary in seed_summaries)
    if mean_uf1 < 0.55 or mean_uar < 0.55:
        return "rejected"
    if (
        all(value >= 0.70 for value in uf1_values)
        and all(value >= 0.70 for value in uar_values)
        and float(np.std(uf1_values)) <= 0.05
        and float(np.std(uar_values)) <= 0.05
    ):
        return "candidate"
    if mean_uf1 >= 0.60 and mean_uar >= 0.60 and no_zero_folds:
        return "research_baseline"
    return "experimental"


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_id = str(args.run_id)
    output_root = (args.output_root or ALGORITHM_ROOT / "reports/microexpression/paper_v2" / run_id).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    previous_metrics = output_root / "aggregate/metrics.json"
    if args.resume and previous_metrics.is_file():
        previous_payload = json.loads(previous_metrics.read_text(encoding="utf-8"))
        if previous_payload.get("status") != "completed":
            previous_hash = sha256_file(previous_metrics)
            failed_copy = previous_metrics.with_name(
                f"metrics.failed.{previous_hash[:12]}.json"
            )
            if not failed_copy.exists():
                failed_copy.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(previous_metrics, failed_copy)
    resolved_config = dict(raw_config)
    resolved_config["execution_overrides"] = {
        "fold_limit": args.fold_limit,
        "max_epochs_override": args.max_epochs_override,
        "bootstrap_iterations": args.bootstrap_iterations,
        "device": args.device,
        "resume": args.resume,
    }
    write_json(output_root / "resolved_config.json", resolved_config)

    spatial_records = load_jsonl(ALGORITHM_ROOT / raw_config["inputs"]["spatial_manifest"])
    temporal_records = load_jsonl(ALGORITHM_ROOT / raw_config["inputs"]["temporal_manifest"])
    split_manifest = json.loads((ALGORITHM_ROOT / raw_config["inputs"]["split_manifest"]).read_text(encoding="utf-8"))
    au_prior = json.loads((ALGORITHM_ROOT / raw_config["inputs"]["au_prior"]).read_text(encoding="utf-8"))
    if len(spatial_records) != 164 or len(temporal_records) != 164:
        raise RuntimeError("MODEL-ME-006 primary SMIC input must contain 164 samples in both streams")
    spatial_by_id = {str(record["sample_id"]): record for record in spatial_records}
    temporal_by_id = {str(record["sample_id"]): record for record in temporal_records}
    if set(spatial_by_id) != set(temporal_by_id):
        raise RuntimeError("Spatial and temporal sample coverage differs")
    all_records = {
        sample_id: {**spatial_by_id[sample_id], "temporal_artifact_path": temporal_by_id[sample_id]["artifact_path"]}
        for sample_id in spatial_by_id
    }
    model_config = DSTMConfig.from_mapping(raw_config["model"])
    training_config = DSTMTrainingConfig.from_mapping(raw_config["training"])
    if args.max_epochs_override is not None:
        training_config = replace(
            training_config,
            max_epochs=args.max_epochs_override,
            patience=min(training_config.patience, args.max_epochs_override),
        )
    reducer_config = FoldReducerConfig(**raw_config["reducer"])
    seeds = [int(value) for value in raw_config["protocol"]["final_seeds"]]
    bootstrap_iterations = int(args.bootstrap_iterations or raw_config["protocol"]["bootstrap_iterations"])
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        raise RuntimeError("DSTM formal evaluation requested CUDA but CUDA is unavailable")

    # The motion CNN is frozen and seeded independently of labels.  Its frame
    # representations can therefore be computed once globally; each fold still
    # fits its reducer on train-subject rows only.
    set_dstm_seed(reducer_config.random_state)
    motion_encoder = DSTMTemporalMotionCNN(model_config.temporal_cnn_feature_dim).eval()
    for parameter in motion_encoder.parameters():
        parameter.requires_grad_(False)
    motion_features = extract_motion_features(
        list(temporal_by_id.values()), encoder=motion_encoder, device=device
    )
    motion_encoder_hash = sha256_bytes(
        b"".join(value.detach().cpu().numpy().tobytes() for value in motion_encoder.state_dict().values())
    )
    write_json(
        output_root / "motion_encoder.json",
        {
            "schema_version": "dstm_frozen_motion_encoder_v1",
            "feature_dim": model_config.temporal_cnn_feature_dim,
            "state_hash": motion_encoder_hash,
            "fit_data_used": False,
            "label_used": False,
        },
    )

    folds = list(split_manifest["folds"])
    if args.fold_limit is not None:
        folds = folds[: args.fold_limit]
    final_results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    started = time.perf_counter()
    au_adjacency = torch.as_tensor(au_prior["adjacency"], dtype=torch.float32)

    for fold_index, fold in enumerate(folds):
        fold_id = str(fold["fold_id"])
        fold_dir = output_root / "folds" / fold_id
        reducer_dir = fold_dir / "reducer"
        reducer_path = reducer_dir / "reducer.joblib"
        try:
            cached_result_paths = [
                fold_dir / "final" / f"seed_{seed}" / "result.json"
                for seed in seeds
            ]
            cached_prediction_paths = [
                fold_dir / "final" / f"seed_{seed}" / "test_predictions.jsonl"
                for seed in seeds
            ]
            if args.resume and all(path.is_file() for path in cached_result_paths + cached_prediction_paths):
                final_results.extend(
                    json.loads(path.read_text(encoding="utf-8"))
                    for path in cached_result_paths
                )
                continue
            if args.resume and reducer_path.is_file():
                reducer = FoldOnlyReducer.load(reducer_path)
                transformed = {
                    sample_id: reducer.transform(features)
                    for sample_id, features in motion_features.items()
                }
                reducer_metadata = reducer.metadata()
            else:
                reducer, transformed, reducer_metadata = fit_fold_reducer(
                    fold=fold,
                    all_records=all_records,
                    motion_features=motion_features,
                    reducer_config=reducer_config,
                    output_dir=reducer_dir,
                )
            feature_store = write_feature_store(
                fold_dir / "temporal_features.npz",
                transformed,
                reducer_metadata=reducer_metadata,
            )
            write_json(
                fold_dir / "fold_contract.json",
                {
                    "task_id": "MODEL-ME-006",
                    "fold_id": fold_id,
                    "train_subjects": fold["train_subjects"],
                    "validation_subjects": fold["validation_subjects"],
                    "test_subjects": fold["test_subjects"],
                    "train_sample_ids": fold["train_sample_ids"],
                    "validation_sample_ids": fold["validation_sample_ids"],
                    "test_sample_ids": fold["test_sample_ids"],
                    "test_used_for_reducer_fit": False,
                    "test_used_for_selection": False,
                    "reducer_metadata": reducer_metadata,
                    "feature_store": feature_store,
                },
            )
            for seed in seeds:
                seed_dir = fold_dir / "final" / f"seed_{seed}"
                result_path = seed_dir / "result.json"
                test_prediction_path = seed_dir / "test_predictions.jsonl"
                if args.resume and result_path.is_file() and test_prediction_path.is_file():
                    final_results.append(json.loads(result_path.read_text(encoding="utf-8")))
                    continue
                set_dstm_seed(seed)
                model = DSTM(au_adjacency, model_config)
                model.temporal_motion_cnn.load_state_dict(motion_encoder.state_dict(), strict=True)
                training_result, best_model = train_dstm_fold(
                    model=model,
                    au_adjacency=au_adjacency,
                    spatial_records=spatial_records,
                    temporal_records=temporal_records,
                    train_sample_ids=fold["train_sample_ids"],
                    validation_sample_ids=fold["validation_sample_ids"],
                    temporal_features=transformed,
                    training_config=training_config,
                    device=device,
                    seed=seed,
                    output_dir=seed_dir,
                    fold_id=fold_id,
                    reducer_metadata=reducer_metadata,
                )
                test_dataset = DSTMArtifactDataset(
                    spatial_records,
                    temporal_records,
                    sample_ids=fold["test_sample_ids"],
                    temporal_features=transformed,
                    preload=True,
                )
                test_loader = _loader(
                    test_dataset, config=training_config, shuffle=False, seed=seed
                )
                test = evaluate_loader(best_model, test_loader, device=device)
                test_prediction_sha256 = write_jsonl(test_prediction_path, test["predictions"])
                result = {
                    **training_result,
                    "test": {
                        "loss": test["loss"],
                        "metrics": test["metrics"],
                        "selection_score": test["selection_score"],
                        "mean_alignment_loss": test["mean_alignment_loss"],
                        "predictions": test["predictions"],
                        "prediction_path": test_prediction_path.resolve().as_posix(),
                        "prediction_sha256": test_prediction_sha256,
                        "evaluation_count": 1,
                        "test_used_for_selection": False,
                    },
                    "test_used_for_selection": False,
                    "fold_id": fold_id,
                    "seed": seed,
                    "model_parameter_count": sum(parameter.numel() for parameter in best_model.parameters()),
                }
                result["result_sha256"] = write_json(result_path, result)
                final_results.append(result)
                print(
                    json.dumps(
                        {
                            "fold_id": fold_id,
                            "seed": seed,
                            "best_epoch": result["best_epoch"],
                            "test_uf1": result["test"]["metrics"]["uf1"],
                            "test_uar": result["test"]["metrics"]["uar"],
                            "duration_seconds": result["duration_seconds"],
                        },
                        ensure_ascii=False,
                    )
                )
        except Exception as error:  # noqa: BLE001
            failures.append({"fold_id": fold_id, "error": repr(error)})
            write_json(fold_dir / "failure.json", {"task_id": "MODEL-ME-006", "fold_id": fold_id, "error": repr(error)})
            print(f"FAILED {fold_id}: {error}")

    expected_sample_ids = {
        str(sample_id)
        for fold in folds
        for sample_id in fold["test_sample_ids"]
    }
    seed_summaries: list[dict[str, Any]] = []
    for seed in seeds:
        rows = [
            row
            for result in final_results
            if int(result["seed"]) == seed
            for row in result["test"]["predictions"]
        ]
        sample_ids = [str(row["sample_id"]) for row in rows]
        if len(sample_ids) != len(set(sample_ids)) or set(sample_ids) != expected_sample_ids:
            failures.append({"seed": seed, "error": "test_prediction_coverage_mismatch"})
            continue
        prediction_path = output_root / "aggregate" / f"predictions_seed_{seed}.jsonl"
        prediction_sha256 = write_jsonl(prediction_path, rows)
        summary = aggregate_seed_predictions(rows, seed=seed, bootstrap_iterations=bootstrap_iterations)
        summary["prediction_path"] = prediction_path.resolve().as_posix()
        summary["prediction_sha256"] = prediction_sha256
        summary["fold_count"] = len([result for result in final_results if int(result["seed"]) == seed])
        seed_summaries.append(summary)
    aggregate = {
        "schema_version": DSTM_EVALUATION_SCHEMA_VERSION,
        "task_id": "MODEL-ME-006",
        "status": "completed" if not failures and len(seed_summaries) == len(seeds) else "failed",
        "run_id": run_id,
        "sample_count": len(spatial_records),
        "subject_count": len(split_manifest["subjects"]),
        "fold_count": len(folds),
        "seeds": seeds,
        "bootstrap_iterations": bootstrap_iterations,
        "seed_summaries": seed_summaries,
        "seed_statistics": {
            metric: {
                "mean": float(np.mean([summary["pooled"][metric] for summary in seed_summaries])) if seed_summaries else None,
                "std": float(np.std([summary["pooled"][metric] for summary in seed_summaries])) if seed_summaries else None,
                "values": [summary["pooled"][metric] for summary in seed_summaries],
            }
            for metric in ("uf1", "uar", "accuracy", "balanced_accuracy")
        },
        "release_decision": release_decision(seed_summaries) if len(seed_summaries) == len(seeds) else "failed",
        "deployment_eligible": False,
        "paper_reproduction_claim": False,
        "failures": failures,
        "resources": {
            "model_parameter_count": sum(parameter.numel() for parameter in DSTM(au_adjacency, model_config).parameters()),
            "motion_encoder_hash": motion_encoder_hash,
            "duration_seconds": time.perf_counter() - started,
            "device": str(device),
        },
    }
    aggregate_path = output_root / "aggregate" / "metrics.json"
    aggregate_sha256 = write_json(aggregate_path, aggregate)
    (aggregate_path.with_suffix(aggregate_path.suffix + ".sha256")).write_text(
        aggregate_sha256 + "\n", encoding="ascii"
    )
    aggregate["aggregate_sha256"] = aggregate_sha256
    audit_payload = {
        "task_id": "MODEL-ME-006",
        "status": aggregate["status"],
        "selection_results": 0,
        "final_results": len(final_results),
        "final_failures": len(failures),
        "test_prediction_coverage": {
            str(seed): len(
                [
                    row
                    for result in final_results
                    if int(result["seed"]) == seed
                    for row in result["test"]["predictions"]
                ]
            )
            for seed in seeds
        },
        "all_test_evaluation_count_one": all(
            result["test"]["evaluation_count"] == 1 for result in final_results
        ),
        "test_used_for_selection": any(result.get("test_used_for_selection") for result in final_results),
        "issues": failures,
    }
    write_json(output_root / "independent_audit.json", audit_payload)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    return 0 if aggregate["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
