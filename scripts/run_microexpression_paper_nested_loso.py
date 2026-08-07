from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
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

from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.manifold_graph import (  # noqa: E402
    sha256_path,
)
from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.paper_evaluation import (  # noqa: E402
    PAPER_EVALUATION_SCHEMA_VERSION,
    EvaluationCandidate,
    aggregate_seed_predictions,
    choose_candidate,
    run_candidate_training,
)
from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.paper_mhssa_tgcn import (  # noqa: E402
    PaperMHSSATGCNConfig,
)
from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.paper_training import (  # noqa: E402
    PaperTrainingConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run leakage-safe nested LOSO selection and final EVAL-ME-002 tests."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ALGORITHM_ROOT
        / "configs/evaluation/microexpression_paper_v2_nested_loso.yaml",
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--fold-limit", type=int)
    parser.add_argument("--max-epochs-override", type=int)
    parser.add_argument("--selection-only", action="store_true")
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
    return sha256_path(path)


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)
    return sha256_path(path)


def stage_candidates(
    stage: str,
    current: EvaluationCandidate,
    stage_config: Mapping[str, Any],
) -> list[EvaluationCandidate]:
    if stage == "preprocessing":
        return [
            replace(
                current,
                variant_id=f"{code.lower()}_farneback_optical_strain",
            )
            for code in stage_config[stage]
        ]
    if stage == "flow_channel":
        preprocess_code = current.variant_id.split("_", maxsplit=1)[0]
        return [
            replace(
                current,
                variant_id=f"{preprocess_code}_{flow}_{channel}",
            )
            for flow, channel in stage_config[stage]
        ]
    if stage == "order":
        return [replace(current, order_mode=value) for value in stage_config[stage]]
    if stage == "attention":
        return [
            replace(current, attention_variant=value)
            for value in stage_config[stage]
        ]
    if stage == "graph":
        return [
            replace(current, graph_mode=mode, fusion_formula=formula)
            for mode, formula in stage_config[stage]
        ]
    if stage == "loss":
        return [replace(current, loss_mode=value) for value in stage_config[stage]]
    raise ValueError(f"Unknown ablation stage: {stage}")


def load_variant_records(
    variant_id: str,
    variants_summary: Mapping[str, Any],
    cache: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    if variant_id not in cache:
        cache[variant_id] = load_jsonl(
            Path(variants_summary["variants"][variant_id]["manifest_path"])
        )
    summary = variants_summary["variants"][variant_id]
    return cache[variant_id], {
        "path": str(summary["manifest_path"]),
        "sha256": str(summary["manifest_sha256"]),
        "variant_id": variant_id,
    }


def record_failure(
    output_dir: Path,
    *,
    fold_id: str,
    stage: str,
    candidate: EvaluationCandidate,
    error: Exception,
) -> dict[str, Any]:
    failure = {
        "schema_version": PAPER_EVALUATION_SCHEMA_VERSION,
        "task_id": "EVAL-ME-002",
        "status": "failed",
        "fold_id": fold_id,
        "stage": stage,
        "candidate": candidate.as_dict(),
        "candidate_id": candidate.candidate_id,
        "error": repr(error),
        "traceback": traceback.format_exc(),
    }
    write_json(output_dir / "failure.json", failure)
    return failure


def aggregate_final_results(
    *,
    final_results: Sequence[Mapping[str, Any]],
    expected_sample_ids: set[str],
    seeds: Sequence[int],
    bootstrap_iterations: int,
    output_root: Path,
) -> dict[str, Any]:
    seed_summaries: list[dict[str, Any]] = []
    all_prediction_files: list[dict[str, Any]] = []
    for seed in seeds:
        seed_results = [result for result in final_results if result["seed"] == seed]
        prediction_rows = [
            row
            for result in seed_results
            for row in result["test"]["predictions"]
        ]
        sample_ids = [str(row["sample_id"]) for row in prediction_rows]
        if len(sample_ids) != len(set(sample_ids)):
            raise RuntimeError(f"Seed {seed} contains duplicate outer-test predictions")
        if set(sample_ids) != expected_sample_ids:
            raise RuntimeError(f"Seed {seed} outer-test sample coverage mismatch")
        prediction_path = output_root / "aggregate" / f"predictions_seed_{seed}.jsonl"
        prediction_sha256 = write_jsonl(prediction_path, prediction_rows)
        seed_summary = aggregate_seed_predictions(
            prediction_rows,
            seed=seed,
            bootstrap_iterations=bootstrap_iterations,
        )
        seed_summary["prediction_path"] = prediction_path.resolve().as_posix()
        seed_summary["prediction_sha256"] = prediction_sha256
        seed_summary["fold_count"] = len(seed_results)
        seed_summaries.append(seed_summary)
        all_prediction_files.append(
            {
                "seed": seed,
                "path": prediction_path.resolve().as_posix(),
                "sha256": prediction_sha256,
            }
        )

    metric_names = ("uf1", "uar", "accuracy", "balanced_accuracy")
    seed_statistics = {
        metric: {
            "mean": float(
                np.mean([summary["pooled"][metric] for summary in seed_summaries])
            ),
            "std": float(
                np.std([summary["pooled"][metric] for summary in seed_summaries])
            ),
            "values": [summary["pooled"][metric] for summary in seed_summaries],
        }
        for metric in metric_names
    }
    mean_uf1 = seed_statistics["uf1"]["mean"]
    mean_uar = seed_statistics["uar"]["mean"]
    no_zero_folds = all(not summary["zero_score_folds"] for summary in seed_summaries)
    all_candidate = all(
        summary["pooled"]["uf1"] >= 0.70
        and summary["pooled"]["uar"] >= 0.70
        for summary in seed_summaries
    )
    if mean_uf1 < 0.55 or mean_uar < 0.55:
        release_decision = "rejected"
    elif all_candidate and seed_statistics["uf1"]["std"] <= 0.05 and seed_statistics[
        "uar"
    ]["std"] <= 0.05:
        release_decision = "candidate"
    elif mean_uf1 >= 0.60 and mean_uar >= 0.60 and no_zero_folds:
        release_decision = "research_baseline"
    else:
        release_decision = "experimental"
    return {
        "schema_version": PAPER_EVALUATION_SCHEMA_VERSION,
        "task_id": "EVAL-ME-002",
        "status": "completed",
        "seed_count": len(seed_summaries),
        "bootstrap_iterations": bootstrap_iterations,
        "seed_summaries": seed_summaries,
        "seed_statistics": seed_statistics,
        "prediction_files": all_prediction_files,
        "release_decision": release_decision,
        "deployment_eligible": release_decision == "candidate",
        "paper_reproduction_claim": False,
    }


def main() -> int:
    args = parse_args()
    raw_config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    run_id = args.run_id or str(raw_config["run_id"])
    output_root = (
        args.output_root
        or ALGORITHM_ROOT / "reports/microexpression/paper_v2" / run_id
    ).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    resolved_config_path = output_root / "resolved_config.yaml"
    input_paths = {
        key: (ALGORITHM_ROOT / value).resolve()
        for key, value in raw_config["inputs"].items()
    }
    variants_summary = json.loads(
        input_paths["variants_summary"].read_text(encoding="utf-8")
    )
    variants_audit = json.loads(
        input_paths["variants_audit"].read_text(encoding="utf-8")
    )
    if variants_audit["status"] != "pass":
        raise RuntimeError("Evaluation input variants did not pass audit")
    split_manifest = json.loads(
        input_paths["split_manifest"].read_text(encoding="utf-8")
    )
    au_prior = json.loads(input_paths["au_prior"].read_text(encoding="utf-8"))
    au_adjacency = torch.tensor(au_prior["adjacency"], dtype=torch.float32)
    model_config = PaperMHSSATGCNConfig.from_mapping(raw_config["model"])
    training_config = PaperTrainingConfig.from_mapping(raw_config["training"])
    if args.max_epochs_override:
        training_config = replace(
            training_config,
            max_epochs=args.max_epochs_override,
            patience=min(training_config.patience, args.max_epochs_override),
        )
        raw_config["training"]["max_epochs"] = training_config.max_epochs
        raw_config["training"]["patience"] = training_config.patience
    raw_config["execution_overrides"] = {
        "fold_limit": args.fold_limit,
        "max_epochs_override": args.max_epochs_override,
        "selection_only": args.selection_only,
    }
    resolved_config_path.write_text(
        yaml.safe_dump(raw_config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    if not torch.cuda.is_available():
        raise RuntimeError("Formal EVAL-ME-002 training requires CUDA")
    device = torch.device("cuda")
    folds = list(split_manifest["folds"])
    if args.fold_limit:
        folds = folds[: args.fold_limit]
    stages = (
        "preprocessing",
        "flow_channel",
        "order",
        "attention",
        "graph",
        "loss",
    )
    selection_seed = int(raw_config["protocol"]["selection_seed"])
    records_cache: dict[str, list[dict[str, Any]]] = {}
    selected_by_fold: dict[str, EvaluationCandidate] = {}
    selection_results: list[dict[str, Any]] = []
    failure_results: list[dict[str, Any]] = []

    for fold_index, fold in enumerate(folds):
        fold_id = str(fold["fold_id"])
        current = EvaluationCandidate(order_seed=selection_seed)
        for stage_index, stage in enumerate(stages):
            candidates = stage_candidates(
                stage, current, raw_config["sequential_ablation"]
            )
            stage_results: list[dict[str, Any]] = []
            for candidate in candidates:
                candidate_dir = (
                    output_root
                    / "selection"
                    / fold_id
                    / stage
                    / candidate.candidate_id
                )
                records, source_manifest = load_variant_records(
                    candidate.variant_id, variants_summary, records_cache
                )
                try:
                    result = run_candidate_training(
                        records=records,
                        train_sample_ids=fold["train_sample_ids"],
                        validation_sample_ids=fold["validation_sample_ids"],
                        test_sample_ids=None,
                        candidate=candidate,
                        base_model_config=model_config,
                        training_config=training_config,
                        au_adjacency=au_adjacency,
                        seed=selection_seed + fold_index * 100 + stage_index,
                        device=device,
                        output_dir=candidate_dir,
                        fold_id=fold_id,
                        stage=stage,
                        source_manifest=source_manifest,
                    )
                except Exception as error:  # noqa: BLE001
                    result = record_failure(
                        candidate_dir,
                        fold_id=fold_id,
                        stage=stage,
                        candidate=candidate,
                        error=error,
                    )
                    failure_results.append(result)
                stage_results.append(result)
                if result.get("status") == "completed":
                    selection_results.append(result)
            selected_result = choose_candidate(stage_results)
            current = EvaluationCandidate(**selected_result["candidate"])
            write_json(
                output_root / "selection" / fold_id / stage / "stage_summary.json",
                {
                    "fold_id": fold_id,
                    "stage": stage,
                    "test_loaded": False,
                    "candidate_count": len(stage_results),
                    "selected_candidate": current.as_dict(),
                    "selected_candidate_id": current.candidate_id,
                    "results": [
                        {
                            "candidate_id": result["candidate_id"],
                            "status": result["status"],
                            "validation": result.get("validation"),
                        }
                        for result in stage_results
                    ],
                },
            )
        selected_by_fold[fold_id] = current
        write_json(
            output_root / "selection" / fold_id / "selected_config.json",
            {
                "fold_id": fold_id,
                "selection_source": "training_side_validation_only",
                "test_used_for_selection": False,
                "candidate": current.as_dict(),
                "candidate_id": current.candidate_id,
            },
        )
        print(f"selection completed {fold_id}: {current.candidate_id}", flush=True)

    selection_summary = {
        "schema_version": PAPER_EVALUATION_SCHEMA_VERSION,
        "task_id": "EVAL-ME-002",
        "fold_count": len(folds),
        "candidate_run_count": len(selection_results) + len(failure_results),
        "completed_candidate_count": len(selection_results),
        "failed_candidate_count": len(failure_results),
        "test_loaded_during_selection": False,
        "selected_by_fold": {
            fold_id: candidate.as_dict()
            for fold_id, candidate in selected_by_fold.items()
        },
    }
    write_json(output_root / "selection_summary.json", selection_summary)
    if args.selection_only:
        print(json.dumps(selection_summary, ensure_ascii=False, indent=2))
        return 0

    seeds = [int(seed) for seed in raw_config["protocol"]["final_seeds"]]
    final_results: list[dict[str, Any]] = []
    final_failures: list[dict[str, Any]] = []
    for seed in seeds:
        for fold_index, fold in enumerate(folds):
            fold_id = str(fold["fold_id"])
            candidate = selected_by_fold[fold_id]
            records, source_manifest = load_variant_records(
                candidate.variant_id, variants_summary, records_cache
            )
            final_dir = output_root / "final" / f"seed_{seed}" / fold_id
            try:
                result = run_candidate_training(
                    records=records,
                    train_sample_ids=fold["train_sample_ids"],
                    validation_sample_ids=fold["validation_sample_ids"],
                    test_sample_ids=fold["test_sample_ids"],
                    candidate=candidate,
                    base_model_config=model_config,
                    training_config=training_config,
                    au_adjacency=au_adjacency,
                    seed=seed,
                    device=device,
                    output_dir=final_dir,
                    fold_id=fold_id,
                    stage="final_outer_test",
                    source_manifest=source_manifest,
                )
                final_results.append(result)
            except Exception as error:  # noqa: BLE001
                failure = record_failure(
                    final_dir,
                    fold_id=fold_id,
                    stage="final_outer_test",
                    candidate=candidate,
                    error=error,
                )
                final_failures.append(failure)
            print(f"final seed={seed} fold={fold_id}", flush=True)
    if final_failures:
        raise RuntimeError(
            f"Final outer evaluation has {len(final_failures)} failed folds"
        )
    expected_sample_ids = {
        str(sample_id)
        for fold in folds
        for sample_id in fold["test_sample_ids"]
    }
    aggregate = aggregate_final_results(
        final_results=final_results,
        expected_sample_ids=expected_sample_ids,
        seeds=seeds,
        bootstrap_iterations=int(
            raw_config["protocol"]["bootstrap_iterations"]
        ),
        output_root=output_root,
    )
    aggregate["run_id"] = run_id
    aggregate["inputs"] = {
        key: {"path": path.as_posix(), "sha256": sha256_path(path)}
        for key, path in input_paths.items()
    }
    aggregate["config"] = {
        "path": resolved_config_path.resolve().as_posix(),
        "sha256": sha256_path(resolved_config_path),
    }
    aggregate_path = output_root / "aggregate" / "metrics.json"
    aggregate_sha256 = write_json(aggregate_path, aggregate)
    audit = {
        "schema_version": PAPER_EVALUATION_SCHEMA_VERSION,
        "task_id": "EVAL-ME-002",
        "status": "pass",
        "fold_count": len(folds),
        "seed_count": len(seeds),
        "final_run_count": len(final_results),
        "expected_final_run_count": len(folds) * len(seeds),
        "selection_test_loaded": False,
        "all_final_tests_used_once": all(
            result["test"]["evaluation_count"] == 1 for result in final_results
        ),
        "all_test_used_for_selection_false": all(
            not result["test_used_for_selection"]
            and not result["test"]["test_used_for_selection"]
            for result in final_results
        ),
        "failed_selection_runs_preserved": len(failure_results),
        "failed_final_runs": len(final_failures),
        "aggregate_metrics": {
            "path": aggregate_path.resolve().as_posix(),
            "sha256": aggregate_sha256,
        },
    }
    audit_path = output_root / "audit.json"
    audit_sha256 = write_json(audit_path, audit)
    print(
        json.dumps(
            {
                "status": "pass",
                "run_id": run_id,
                "release_decision": aggregate["release_decision"],
                "seed_statistics": aggregate["seed_statistics"],
                "aggregate_path": aggregate_path.resolve().as_posix(),
                "aggregate_sha256": aggregate_sha256,
                "audit_path": audit_path.resolve().as_posix(),
                "audit_sha256": audit_sha256,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
