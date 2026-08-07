from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import platform
import subprocess
import sys
import traceback
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml


ALGORITHM_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ALGORITHM_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.causalnet import (  # noqa: E402
    CausalNetConfig,
)
from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.causalnet_evaluation import (  # noqa: E402
    CAUSALNET_EVALUATION_SCHEMA_VERSION,
    CausalNetTrainingConfig,
    aggregate_final_results,
    run_nested_loso_fold,
    sha256_file,
)


DEFAULT_CONFIG = (
    ALGORITHM_ROOT
    / "configs/evaluation/microexpression_causalnet_nested_loso.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run EVAL-ME-004 strict subject-level CausalNet Nested-LOSO."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--fold-limit", type=int)
    parser.add_argument("--seed-limit", type=int)
    parser.add_argument("--max-epochs-override", type=int)
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
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return sha256_file(path)


def combined_hash(paths: Sequence[Path]) -> str:
    digest = sha256()
    for path in paths:
        digest.update(path.resolve().as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def git_head() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ALGORITHM_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def environment_snapshot(environment_lock: Path) -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cuda_available": cuda_available,
        "torch_cuda_version": torch.version.cuda,
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "environment_lock_path": environment_lock.resolve().as_posix(),
        "environment_lock_sha256": sha256_file(environment_lock),
        "git_head": git_head(),
        "git_worktree_clean_claim": False,
    }


def record_failure(
    output_dir: Path,
    *,
    fold_id: str,
    seed: int,
    error: Exception,
) -> dict[str, Any]:
    failure = {
        "schema_version": CAUSALNET_EVALUATION_SCHEMA_VERSION,
        "task_id": "EVAL-ME-004",
        "status": "failed",
        "fold_id": fold_id,
        "seed": seed,
        "error": repr(error),
        "traceback": traceback.format_exc(),
    }
    write_json(output_dir / "failure.json", failure)
    return failure


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if raw_config.get("task_id") != "EVAL-ME-004":
        raise ValueError("Evaluation config task_id must be EVAL-ME-004")
    input_paths = {
        key: (ALGORITHM_ROOT / value).resolve()
        for key, value in raw_config["inputs"].items()
    }
    missing = [key for key, path in input_paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing evaluation inputs: {missing}")
    completion_audit = json.loads(
        input_paths["completion_audit"].read_text(encoding="utf-8")
    )
    if completion_audit.get("status") != "pass" or completion_audit.get(
        "issue_count"
    ) != 0:
        raise RuntimeError("FLOW-ME-004/MODEL-ME-008 completion audit is not pass")
    if not torch.cuda.is_available():
        raise RuntimeError("Formal EVAL-ME-004 requires CUDA")

    run_id = args.run_id or str(raw_config["run_id"])
    has_override = any(
        value is not None
        for value in (args.fold_limit, args.seed_limit, args.max_epochs_override)
    )
    if has_override and args.output_root is None and args.run_id is None:
        run_id = f"{run_id}-SMOKE"
    output_root = (
        args.output_root
        or ALGORITHM_ROOT / "reports/microexpression/causalnet_v1" / run_id
    ).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    model_values = dict(raw_config["model"])
    model_values["block_repeats"] = tuple(model_values["block_repeats"])
    model_config = CausalNetConfig(**model_values)
    training_config = CausalNetTrainingConfig.from_mapping(raw_config["training"])
    if args.max_epochs_override is not None:
        if args.max_epochs_override < 1:
            raise ValueError("max-epochs-override must be positive")
        training_config = replace(
            training_config,
            max_epochs=args.max_epochs_override,
            min_epochs=min(training_config.min_epochs, args.max_epochs_override),
            patience=min(training_config.patience, args.max_epochs_override),
        )

    records = load_jsonl(input_paths["artifact_manifest"])
    split_manifest = json.loads(
        input_paths["split_manifest"].read_text(encoding="utf-8")
    )
    folds = list(split_manifest["folds"])
    seeds = [int(seed) for seed in raw_config["protocol"]["final_seeds"]]
    if args.fold_limit is not None:
        folds = folds[: args.fold_limit]
    if args.seed_limit is not None:
        seeds = seeds[: args.seed_limit]
    formal_protocol = not has_override
    if formal_protocol and (len(folds) != 16 or len(seeds) != 3):
        raise RuntimeError("Formal EVAL-ME-004 requires exactly 16 folds and 3 seeds")

    source_files = [
        SRC_ROOT
        / "elderly_monitoring/modules/mental_health/submodules/facial_affect_clue/causalnet.py",
        SRC_ROOT
        / "elderly_monitoring/modules/mental_health/submodules/facial_affect_clue/causalnet_dataset.py",
        SRC_ROOT
        / "elderly_monitoring/modules/mental_health/submodules/facial_affect_clue/causalnet_evaluation.py",
        Path(__file__).resolve(),
    ]
    evaluation_bundle_hash = combined_hash(source_files)
    input_hashes = {
        f"{key}_sha256": sha256_file(path) for key, path in input_paths.items()
    }
    environment = environment_snapshot(input_paths["environment_lock"])
    resolved_config = {
        **raw_config,
        "model": model_config.as_dict(),
        "training": training_config.as_dict(),
        "execution": {
            "run_id": run_id,
            "formal_protocol": formal_protocol,
            "fold_limit": args.fold_limit,
            "seed_limit": args.seed_limit,
            "max_epochs_override": args.max_epochs_override,
            "fold_ids": [str(fold["fold_id"]) for fold in folds],
            "seeds": seeds,
            "device": "cuda",
        },
        "evidence": {
            "config_source_path": config_path.as_posix(),
            "config_source_sha256": sha256_file(config_path),
            "evaluation_bundle_hash": evaluation_bundle_hash,
            "input_hashes": input_hashes,
        },
    }
    resolved_config_path = output_root / "resolved_config.yaml"
    resolved_config_path.write_text(
        yaml.safe_dump(resolved_config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    environment_path = output_root / "environment.json"
    environment_sha256 = write_json(environment_path, environment)

    final_results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    device = torch.device("cuda")
    run_index_path = output_root / "run_index.json"
    for seed in seeds:
        for fold_index, fold in enumerate(folds, start=1):
            fold_id = str(fold["fold_id"])
            final_dir = output_root / "final" / f"seed_{seed}" / fold_id
            try:
                result = run_nested_loso_fold(
                    records=records,
                    train_sample_ids=fold["train_sample_ids"],
                    validation_sample_ids=fold["validation_sample_ids"],
                    test_sample_ids=fold["test_sample_ids"],
                    model_config=model_config,
                    training_config=training_config,
                    seed=seed,
                    device=device,
                    output_dir=final_dir,
                    fold_id=fold_id,
                    input_hashes=input_hashes,
                    evaluation_bundle_hash=evaluation_bundle_hash,
                )
                final_results.append(result)
                print(
                    f"completed seed={seed} fold={fold_id} "
                    f"({fold_index}/{len(folds)}) epoch={result['best_epoch']}",
                    flush=True,
                )
            except Exception as error:  # noqa: BLE001
                failures.append(
                    record_failure(
                        final_dir,
                        fold_id=fold_id,
                        seed=seed,
                        error=error,
                    )
                )
                print(
                    f"failed seed={seed} fold={fold_id}: {error!r}",
                    flush=True,
                )
            write_json(
                run_index_path,
                {
                    "schema_version": CAUSALNET_EVALUATION_SCHEMA_VERSION,
                    "task_id": "EVAL-ME-004",
                    "run_id": run_id,
                    "formal_protocol": formal_protocol,
                    "expected_final_run_count": len(folds) * len(seeds),
                    "completed_final_run_count": len(final_results),
                    "failed_final_run_count": len(failures),
                    "results": [
                        {
                            "fold_id": result["fold_id"],
                            "seed": result["seed"],
                            "path": result["result_path"],
                            "sha256": result["result_sha256"],
                        }
                        for result in final_results
                    ],
                    "failures": failures,
                },
            )
    if failures:
        raise RuntimeError(f"EVAL-ME-004 has {len(failures)} failed final folds")

    expected_sample_ids = {
        str(sample_id)
        for fold in folds
        for sample_id in fold["test_sample_ids"]
    }
    aggregate = aggregate_final_results(
        final_results=final_results,
        expected_sample_ids=expected_sample_ids,
        seeds=seeds,
        bootstrap_iterations=int(raw_config["protocol"]["bootstrap_iterations"]),
        output_root=output_root,
    )
    aggregate.update(
        {
            "run_id": run_id,
            "formal_protocol": formal_protocol,
            "input_hashes": input_hashes,
            "evaluation_bundle_hash": evaluation_bundle_hash,
            "resolved_config": {
                "path": resolved_config_path.resolve().as_posix(),
                "sha256": sha256_file(resolved_config_path),
            },
            "environment": {
                "path": environment_path.resolve().as_posix(),
                "sha256": environment_sha256,
            },
        }
    )
    aggregate_path = output_root / "aggregate" / "metrics.json"
    aggregate_sha256 = write_json(aggregate_path, aggregate)
    protocol_audit = {
        "schema_version": CAUSALNET_EVALUATION_SCHEMA_VERSION,
        "task_id": "EVAL-ME-004",
        "run_id": run_id,
        "status": "pass",
        "formal_protocol": formal_protocol,
        "fold_count": len(folds),
        "seed_count": len(seeds),
        "final_run_count": len(final_results),
        "expected_final_run_count": len(folds) * len(seeds),
        "selection_source": "training_side_validation_subjects_only",
        "configuration_selection": "frozen_before_outer_test",
        "test_used_for_selection": False,
        "all_tests_evaluated_once": all(
            result["test"]["evaluation_count"] == 1 for result in final_results
        ),
        "failed_final_run_count": 0,
        "aggregate_metrics": {
            "path": aggregate_path.resolve().as_posix(),
            "sha256": aggregate_sha256,
        },
    }
    protocol_audit_path = output_root / "protocol_audit.json"
    protocol_audit_sha256 = write_json(protocol_audit_path, protocol_audit)
    print(
        json.dumps(
            {
                "status": "pass",
                "run_id": run_id,
                "formal_protocol": formal_protocol,
                "final_run_count": len(final_results),
                "release_decision": aggregate["release_decision"],
                "seed_statistics": aggregate["seed_statistics"],
                "aggregate_path": aggregate_path.resolve().as_posix(),
                "aggregate_sha256": aggregate_sha256,
                "protocol_audit_path": protocol_audit_path.resolve().as_posix(),
                "protocol_audit_sha256": protocol_audit_sha256,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
