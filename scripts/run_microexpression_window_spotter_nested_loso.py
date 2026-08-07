from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import platform
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

from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.microexpression_spotter import (  # noqa: E402
    SpotterModelConfig,
)
from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.spotting_evaluation import (  # noqa: E402
    SpotterTrainingConfig,
    binary_metrics,
    build_spotting_splits,
    sha256_file,
    train_spotting_fold,
)


DEFAULT_CONFIG = ALGORITHM_ROOT / "configs/evaluation/microexpression_window_spotter_nested_loso.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SPOT-ME-002 strict SMIC window evaluation.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument("--folds", nargs="*")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return sha256_file(path)


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(path)
    return sha256_file(path)


def bundle_hash(paths: Sequence[Path]) -> str:
    digest = sha256()
    for path in paths:
        digest.update(path.resolve().as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def bootstrap_subject_ci(
    rows: Sequence[Mapping[str, Any]], *, seed: int, iterations: int
) -> dict[str, list[float]]:
    subjects = sorted({str(row["subject_id"]) for row in rows})
    by_subject = {subject: [row for row in rows if str(row["subject_id"]) == subject] for subject in subjects}
    rng = np.random.default_rng(seed)
    values = {"macro_f1": [], "balanced_accuracy": []}
    for _ in range(iterations):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        selected = [row for subject in sampled for row in by_subject[str(subject)]]
        metrics = binary_metrics(
            [int(row["label"]) for row in selected],
            [int(row["prediction"]) for row in selected],
        )
        values["macro_f1"].append(float(metrics["macro_f1"]))
        values["balanced_accuracy"].append(float(metrics["balanced_accuracy"]))
    return {
        name: [float(np.quantile(scores, 0.025)), float(np.quantile(scores, 0.975))]
        for name, scores in values.items()
    }


def outcome(seed_summaries: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> str:
    mean_f1 = float(np.mean([summary["pooled"]["macro_f1"] for summary in seed_summaries]))
    mean_balanced = float(np.mean([summary["pooled"]["balanced_accuracy"] for summary in seed_summaries]))
    interpretation = config["interpretation"]
    if min(mean_f1, mean_balanced) < float(interpretation["rejected_below"]):
        return "rejected"
    if min(mean_f1, mean_balanced) >= float(interpretation["short_window_baseline_at"]):
        return "short_window_baseline"
    return "weak_window_baseline"


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_root = (
        args.run_root
        or ALGORITHM_ROOT / "reports/microexpression/spotter_v1" / str(raw["run_id"])
    ).resolve()
    input_paths = {key: (ALGORITHM_ROOT / value).resolve() for key, value in raw["inputs"].items()}
    input_hashes = {f"{key}_sha256": sha256_file(path) for key, path in input_paths.items()}
    records = load_jsonl(input_paths["feature_manifest"])
    if len(records) != 328 or len({str(record["sample_id"]) for record in records}) != 328:
        raise ValueError("SPOT-ME-002 requires 328 unique feature records")
    split_manifest = build_spotting_splits(
        records,
        validation_subject_count=int(raw["protocol"]["validation_subject_count"]),
    )
    split_path = ALGORITHM_ROOT / "data/manifests/microexpression/spotter_v1_subject_loso_splits.json"
    split_hash = write_json(split_path, split_manifest)
    input_hashes["split_manifest_sha256"] = split_hash
    seeds = args.seeds or [int(value) for value in raw["protocol"]["final_seeds"]]
    selected_folds = set(args.folds or [])
    folds = [
        fold for fold in split_manifest["folds"] if not selected_folds or fold["fold_id"] in selected_folds
    ]
    if not folds:
        raise ValueError("No spotting folds selected")
    model_config = SpotterModelConfig(**raw["model"])
    training_config = SpotterTrainingConfig(**raw["training"])
    device = torch.device(args.device)
    source_files = [
        SRC_ROOT / "elderly_monitoring/modules/mental_health/submodules/facial_affect_clue/microexpression_spotter.py",
        SRC_ROOT / "elderly_monitoring/modules/mental_health/submodules/facial_affect_clue/spotting_evaluation.py",
        Path(__file__).resolve(),
    ]
    evaluation_bundle_hash = bundle_hash(source_files)
    input_hashes["evaluation_bundle_sha256"] = evaluation_bundle_hash
    resolved = deepcopy(raw)
    resolved["execution"] = {
        "device": str(device),
        "seeds": seeds,
        "fold_ids": [fold["fold_id"] for fold in folds],
        "formal_protocol": len(seeds) == 3 and len(folds) == 16,
    }
    resolved["evidence"] = {"input_hashes": input_hashes}
    run_root.mkdir(parents=True, exist_ok=True)
    resolved_path = run_root / "resolved_config.yaml"
    resolved_path.write_text(yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True), encoding="utf-8")
    environment = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "cuda_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    write_json(run_root / "environment.json", environment)
    start = time.perf_counter()
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for seed in seeds:
        for fold in folds:
            output_dir = run_root / "final" / f"seed_{seed}" / str(fold["fold_id"])
            try:
                result = train_spotting_fold(
                    records=records,
                    train_sample_ids=fold["train_sample_ids"],
                    validation_sample_ids=fold["validation_sample_ids"],
                    test_sample_ids=fold["test_sample_ids"],
                    model_config=model_config,
                    training_config=training_config,
                    seed=seed,
                    device=device,
                    output_dir=output_dir,
                    fold_id=str(fold["fold_id"]),
                    input_hashes=input_hashes,
                )
                results.append(result)
                print(f"completed seed={seed} fold={fold['fold_id']} f1={result['test']['metrics']['macro_f1']:.4f}")
            except Exception as error:  # noqa: BLE001
                failures.append({"seed": seed, "fold_id": fold["fold_id"], "error": repr(error)})
                print(f"failed seed={seed} fold={fold['fold_id']}: {error}", file=sys.stderr)
    run_index = {
        "schema_version": "smic_window_spotter_run_index_v1",
        "task_id": "SPOT-ME-002",
        "run_id": raw["run_id"],
        "results": [
            {
                "seed": result["seed"],
                "fold_id": result["fold_id"],
                "result_path": str((run_root / "final" / f"seed_{result['seed']}" / result["fold_id"] / "result.json").resolve()),
            }
            for result in results
        ],
        "failure_count": len(failures),
        "failures": failures,
    }
    write_json(run_root / "run_index.json", run_index)
    expected = len(seeds) * len(folds)
    if failures or len(results) != expected:
        raise RuntimeError(f"SPOT-ME-002 completed {len(results)}/{expected} runs")

    seed_summaries: list[dict[str, Any]] = []
    expected_test_ids = {
        str(sample_id) for fold in folds for sample_id in fold["test_sample_ids"]
    }
    for seed in seeds:
        seed_results = [result for result in results if int(result["seed"]) == seed]
        rows = [row for result in seed_results for row in result["test"]["predictions"]]
        rows.sort(key=lambda row: str(row["sample_id"]))
        if (
            len(rows) != len(expected_test_ids)
            or len({str(row["sample_id"]) for row in rows}) != len(expected_test_ids)
            or {str(row["sample_id"]) for row in rows} != expected_test_ids
        ):
            raise RuntimeError(f"Seed {seed} does not cover selected test windows")
        prediction_path = run_root / "aggregate" / f"predictions_seed_{seed}.jsonl"
        prediction_hash = write_jsonl(prediction_path, rows)
        pooled = binary_metrics(
            [int(row["label"]) for row in rows], [int(row["prediction"]) for row in rows]
        )
        fold_metrics = [result["test"]["metrics"] for result in seed_results]
        seed_summaries.append(
            {
                "seed": seed,
                "pooled": pooled,
                "mean_fold": {
                    key: float(np.mean([metrics[key] for metrics in fold_metrics]))
                    for key in ("accuracy", "macro_f1", "balanced_accuracy")
                },
                "bootstrap_subject_95_ci": bootstrap_subject_ci(
                    rows, seed=seed, iterations=int(raw["protocol"]["bootstrap_iterations"])
                ),
                "thresholds": [float(result["test"]["threshold"]) for result in seed_results],
                "prediction_path": str(prediction_path.resolve()),
                "prediction_sha256": prediction_hash,
            }
        )
    decision = outcome(seed_summaries, raw)
    aggregate = {
        "schema_version": "smic_window_spotter_aggregate_v1",
        "task_id": "SPOT-ME-002",
        "run_id": raw["run_id"],
        "seed_summaries": seed_summaries,
        "seed_statistics": {
            metric: {
                "mean": float(np.mean([summary["pooled"][metric] for summary in seed_summaries])),
                "std": float(np.std([summary["pooled"][metric] for summary in seed_summaries])),
            }
            for metric in ("macro_f1", "balanced_accuracy", "accuracy")
        },
        "outcome": decision,
        "deployment_eligible": False,
        "long_video_spotting_claim": False,
        "system_integration_eligible": False,
        "runtime_seconds": time.perf_counter() - start,
    }
    aggregate_path = run_root / "aggregate" / "metrics.json"
    aggregate_hash = write_json(aggregate_path, aggregate)
    audit = {
        "schema_version": "smic_window_spotter_protocol_audit_v1",
        "task_id": "SPOT-ME-002",
        "status": "pass",
        "run_count": len(results),
        "seed_count": len(seeds),
        "fold_count": len(folds),
        "samples_per_seed": len(expected_test_ids),
        "test_used_for_selection": False,
        "subject_leakage_count": 0,
        "long_video_spotting_claim": False,
        "deployment_eligible": False,
        "input_hashes": input_hashes,
        "aggregate_sha256": aggregate_hash,
    }
    audit_hash = write_json(run_root / "protocol_audit.json", audit)
    print(
        json.dumps(
            {
                "status": "pass",
                "run_id": raw["run_id"],
                "outcome": decision,
                "seed_statistics": aggregate["seed_statistics"],
                "aggregate_sha256": aggregate_hash,
                "protocol_audit_sha256": audit_hash,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
