from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import yaml


ALGORITHM_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ALGORITHM_ROOT / "src"


if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.metrics import (  # noqa: E402
    classification_metrics,
)


AUDIT_SCHEMA_VERSION = "microexpression_nested_loso_independent_audit_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sequence_sha256(values: Sequence[str]) -> str:
    encoded = json.dumps(list(values), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def resolve_algorithm_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ALGORITHM_ROOT / path).resolve()


def close_enough(left: float, right: float) -> bool:
    return abs(float(left) - float(right)) <= 1e-12


def expected_release_decision(seed_summaries: Sequence[Mapping[str, Any]]) -> str:
    uf1_values = [float(summary["pooled"]["uf1"]) for summary in seed_summaries]
    uar_values = [float(summary["pooled"]["uar"]) for summary in seed_summaries]
    uf1_mean = sum(uf1_values) / len(uf1_values)
    uar_mean = sum(uar_values) / len(uar_values)
    uf1_std = (sum((value - uf1_mean) ** 2 for value in uf1_values) / len(uf1_values)) ** 0.5
    uar_std = (sum((value - uar_mean) ** 2 for value in uar_values) / len(uar_values)) ** 0.5
    no_zero_folds = all(not summary["zero_score_folds"] for summary in seed_summaries)
    all_candidate = all(
        float(summary["pooled"]["uf1"]) >= 0.70
        and float(summary["pooled"]["uar"]) >= 0.70
        for summary in seed_summaries
    )
    if uf1_mean < 0.55 or uar_mean < 0.55:
        return "rejected"
    if all_candidate and uf1_std <= 0.05 and uar_std <= 0.05:
        return "candidate"
    if uf1_mean >= 0.60 and uar_mean >= 0.60 and no_zero_folds:
        return "research_baseline"
    return "experimental"


def audit_run(run_root: Path) -> dict[str, Any]:
    issues: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            issues.append(message)

    resolved_config_path = run_root / "resolved_config.yaml"
    if not resolved_config_path.is_file():
        return {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "status": "fail",
            "issues": [f"Missing resolved config: {resolved_config_path}"],
        }
    config = yaml.safe_load(resolved_config_path.read_text(encoding="utf-8"))
    require(
        config.get("task_id") == "EVAL-ME-002",
        "resolved config task_id is not EVAL-ME-002",
    )
    overrides = config.get("execution_overrides", {})
    require(overrides.get("fold_limit") is None, "formal run has fold_limit override")
    require(
        overrides.get("max_epochs_override") is None,
        "formal run has max_epochs override",
    )
    require(not overrides.get("selection_only"), "formal run is selection-only")

    split_path = resolve_algorithm_path(config["inputs"]["split_manifest"])
    variants_summary_path = resolve_algorithm_path(config["inputs"]["variants_summary"])
    variants_audit_path = resolve_algorithm_path(config["inputs"]["variants_audit"])
    split_manifest = load_json(split_path)
    variants_summary = load_json(variants_summary_path)
    variants_audit = load_json(variants_audit_path)
    require(variants_audit.get("status") == "pass", "evaluation variants audit failed")
    folds = {str(fold["fold_id"]): fold for fold in split_manifest["folds"]}
    expected_seeds = [int(seed) for seed in config["protocol"]["final_seeds"]]
    expected_candidate_counts = {
        "preprocessing": 4,
        "flow_channel": 4,
        "order": 3,
        "attention": 2,
        "graph": 4,
        "loss": 4,
    }

    selection_root = run_root / "selection"
    selected_configs: dict[str, dict[str, Any]] = {}
    selection_result_count = 0
    selection_failure_count = 0
    selection_test_file_count = 0
    for fold_id, fold in folds.items():
        selected_path = selection_root / fold_id / "selected_config.json"
        require(selected_path.is_file(), f"missing selected config: {fold_id}")
        if selected_path.is_file():
            selected_configs[fold_id] = load_json(selected_path)["candidate"]
        for stage, expected_count in expected_candidate_counts.items():
            stage_root = selection_root / fold_id / stage
            summary_path = stage_root / "stage_summary.json"
            require(summary_path.is_file(), f"missing stage summary: {fold_id}/{stage}")
            result_paths = list(stage_root.glob("*/result.json"))
            failure_paths = list(stage_root.glob("*/failure.json"))
            selection_result_count += len(result_paths)
            selection_failure_count += len(failure_paths)
            require(
                len(result_paths) + len(failure_paths) == expected_count,
                f"candidate count mismatch: {fold_id}/{stage}",
            )
            if summary_path.is_file():
                summary = load_json(summary_path)
                require(
                    int(summary.get("candidate_count", -1)) == expected_count,
                    f"summary candidate count mismatch: {fold_id}/{stage}",
                )
                require(
                    summary.get("test_loaded") is False,
                    f"selection summary claims test was loaded: {fold_id}/{stage}",
                )
            for result_path in result_paths:
                result = load_json(result_path)
                require(
                    result.get("status") == "completed",
                    f"selection result not completed: {result_path}",
                )
                variant_id = str(result.get("candidate", {}).get("variant_id", ""))
                require(
                    variant_id in variants_summary.get("variants", {}),
                    f"unknown evaluation variant: {variant_id}",
                )
                require(
                    result.get("test") is None,
                    f"selection result contains test output: {result_path}",
                )
                partitions = result.get("run_contract", {}).get("partitions", {})
                require(
                    partitions.get("test", {}).get("sample_count") == 0,
                    f"selection contract has test samples: {result_path}",
                )
                selection_test_file_count += len(
                    list(result_path.parent.glob("test_predictions.jsonl"))
                )

    require(selection_result_count == 336, "formal selection result count is not 336")
    require(selection_failure_count == 0, "formal selection contains failed candidates")
    require(selection_test_file_count == 0, "selection contains test prediction files")
    require(len(selected_configs) == len(folds) == 16, "not all outer folds selected")

    final_root = run_root / "final"
    final_paths = sorted(final_root.glob("seed_*/loso_*/result.json"))
    require(len(final_paths) == len(expected_seeds) * len(folds) == 48, "final run count is not 48")
    final_rows_by_seed: dict[int, list[dict[str, Any]]] = {seed: [] for seed in expected_seeds}
    final_failure_count = len(list(final_root.rglob("failure.json")))
    require(final_failure_count == 0, "final contains failed runs")

    for result_path in final_paths:
        result = load_json(result_path)
        seed = int(result["seed"])
        fold_id = str(result["fold_id"])
        require(seed in final_rows_by_seed, f"unexpected final seed: {seed}")
        require(fold_id in folds, f"unexpected final fold: {fold_id}")
        if seed not in final_rows_by_seed or fold_id not in folds:
            continue
        require(result.get("status") == "completed", f"final not completed: {result_path}")
        variant_id = str(result.get("candidate", {}).get("variant_id", ""))
        require(
            variant_id in variants_summary.get("variants", {}),
            f"unknown final evaluation variant: {variant_id}",
        )
        require(result.get("test_used_for_selection") is False, f"final selection leak flag: {result_path}")
        test_result = result.get("test")
        require(test_result is not None, f"missing final test result: {result_path}")
        if test_result is None:
            continue
        require(test_result.get("evaluation_count") == 1, f"test evaluated more than once: {result_path}")
        require(test_result.get("test_used_for_selection") is False, f"test selection flag: {result_path}")
        fold = folds[fold_id]
        expected_test_ids = [str(value) for value in fold["test_sample_ids"]]
        expected_test_hash = sequence_sha256(expected_test_ids)
        contract = result.get("run_contract", {})
        test_partition = contract.get("partitions", {}).get("test", {})
        require(test_partition.get("sample_count") == len(expected_test_ids), f"test partition count mismatch: {result_path}")
        require(test_partition.get("sample_ids_sha256") == expected_test_hash, f"test partition hash mismatch: {result_path}")
        prediction_rows = list(test_result.get("predictions", []))
        prediction_ids = [str(row["sample_id"]) for row in prediction_rows]
        require(prediction_ids == expected_test_ids, f"test prediction ordering/coverage mismatch: {result_path}")
        prediction_path = Path(str(test_result["prediction_path"]))
        require(prediction_path.is_file(), f"missing test prediction file: {prediction_path}")
        if prediction_path.is_file():
            require(sha256_file(prediction_path) == test_result["prediction_sha256"], f"test prediction hash mismatch: {prediction_path}")
            require(load_jsonl(prediction_path) == prediction_rows, f"test prediction content mismatch: {prediction_path}")
        checkpoint = Path(str(result["checkpoint"]["path"]))
        epoch_log = Path(str(result["epoch_log"]["path"]))
        require(checkpoint.is_file(), f"missing checkpoint: {checkpoint}")
        require(epoch_log.is_file(), f"missing epoch log: {epoch_log}")
        if checkpoint.is_file():
            require(sha256_file(checkpoint) == result["checkpoint"]["sha256"], f"checkpoint hash mismatch: {checkpoint}")
        if epoch_log.is_file():
            require(sha256_file(epoch_log) == result["epoch_log"]["sha256"], f"epoch log hash mismatch: {epoch_log}")
        require(result.get("candidate") == selected_configs.get(fold_id), f"final candidate differs from selected config: {result_path}")
        final_rows_by_seed[seed].extend(prediction_rows)

    for seed, rows in final_rows_by_seed.items():
        require(len(rows) == 164, f"seed {seed} does not contain 164 predictions")
        ids = [str(row["sample_id"]) for row in rows]
        require(len(ids) == len(set(ids)) == 164, f"seed {seed} has duplicate predictions")

    aggregate_path = run_root / "aggregate" / "metrics.json"
    require(aggregate_path.is_file(), "missing aggregate metrics")
    aggregate = load_json(aggregate_path) if aggregate_path.is_file() else {}
    require(aggregate.get("seed_count") == 3, "aggregate seed count is not 3")
    require(aggregate.get("bootstrap_iterations") == int(config["protocol"]["bootstrap_iterations"]), "bootstrap iteration count mismatch")
    if aggregate:
        require(aggregate.get("release_decision") == expected_release_decision(aggregate["seed_summaries"]), "release decision is inconsistent with aggregate metrics")
        for summary in aggregate.get("seed_summaries", []):
            seed = int(summary["seed"])
            recomputed = classification_metrics(
                [int(row["label"]) for row in final_rows_by_seed[seed]],
                [int(row["prediction"]) for row in final_rows_by_seed[seed]],
            )
            for metric in ("uf1", "uar", "accuracy", "balanced_accuracy"):
                require(close_enough(summary["pooled"][metric], recomputed[metric]), f"pooled {metric} mismatch for seed {seed}")

    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "task_id": "EVAL-ME-002",
        "status": "pass" if not issues else "fail",
        "issues": issues,
        "fold_count": len(folds),
        "selection_result_count": selection_result_count,
        "selection_failure_count": selection_failure_count,
        "selection_test_prediction_file_count": selection_test_file_count,
        "final_result_count": len(final_paths),
        "final_failure_count": final_failure_count,
        "final_test_prediction_file_count": sum(len(rows) for rows in final_rows_by_seed.values()) // 3,
        "aggregate_path": aggregate_path.resolve().as_posix(),
        "aggregate_sha256": sha256_file(aggregate_path) if aggregate_path.is_file() else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Independently audit EVAL-ME-002 nested LOSO outputs.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = audit_run(args.run_root.resolve())
    output = (args.output or args.run_root / "independent_audit.json").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result["audit_path"] = output.as_posix()
    result["audit_sha256"] = sha256_file(output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
