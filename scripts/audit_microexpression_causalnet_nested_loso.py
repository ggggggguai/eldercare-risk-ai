from __future__ import annotations

import argparse
import gc
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml


ALGORITHM_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ALGORITHM_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.causalnet import (  # noqa: E402
    ARCHITECTURE_CORRECTION,
    CAUSALNET_UPSTREAM_COMMIT,
    CAUSALNET_UPSTREAM_LICENSE,
    CAUSALNET_UPSTREAM_REPOSITORY,
    code_hash,
)
from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.causalnet_evaluation import (  # noqa: E402
    CAUSALNET_EVALUATION_SCHEMA_VERSION,
    aggregate_seed_predictions,
    release_decision,
    sha256_file,
    validate_subject_partitions,
)
from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.metrics import (  # noqa: E402
    classification_metrics,
)


DEFAULT_CONFIG = (
    ALGORITHM_ROOT
    / "configs/evaluation/microexpression_causalnet_nested_loso.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independently audit all EVAL-ME-004 final folds and predictions."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def add_issue(issues: list[dict[str, Any]], code: str, **details: Any) -> None:
    issues.append({"code": code, **details})


def close_enough(left: float, right: float) -> bool:
    return bool(np.isclose(left, right, rtol=0.0, atol=1e-12))


def audit_checkpoint(
    path: Path,
    *,
    result: Mapping[str, Any],
    expected_hashes: Mapping[str, str],
    expected_bundle_hash: str,
    issues: list[dict[str, Any]],
) -> None:
    fold_id = str(result["fold_id"])
    seed = int(result["seed"])
    expected_sha256 = str(result["checkpoint"]["sha256"])
    if not path.is_file() or sha256_file(path) != expected_sha256:
        add_issue(issues, "checkpoint_hash_mismatch", fold_id=fold_id, seed=seed)
        return
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected_metadata = {
        "upstream_repository": CAUSALNET_UPSTREAM_REPOSITORY,
        "upstream_commit": CAUSALNET_UPSTREAM_COMMIT,
        "upstream_license": CAUSALNET_UPSTREAM_LICENSE,
        "architecture_correction": ARCHITECTURE_CORRECTION,
        "upstream_bit_exact": False,
        "paper_reproduction_claim": False,
        "source_manifest_hash": expected_hashes["artifact_manifest_sha256"],
        "preprocessing_config_hash": expected_hashes[
            "preprocessing_config_sha256"
        ],
        "code_hash": code_hash(),
    }
    for key, expected in expected_metadata.items():
        if payload.get(key) != expected:
            add_issue(
                issues,
                "checkpoint_metadata_mismatch",
                fold_id=fold_id,
                seed=seed,
                field=key,
            )
    training = payload.get("training_config", {})
    required_training = {
        "task_id": "EVAL-ME-004",
        "evaluation_schema_version": CAUSALNET_EVALUATION_SCHEMA_VERSION,
        "fold_id": fold_id,
        "seed": seed,
        "selection_source": "training_side_validation_subjects_only",
        "configuration_selection": "frozen_before_outer_test",
        "test_used_for_selection": False,
        "evaluation_bundle_hash": expected_bundle_hash,
        "input_hashes": dict(expected_hashes),
    }
    for key, expected in required_training.items():
        if training.get(key) != expected:
            add_issue(
                issues,
                "checkpoint_training_contract_mismatch",
                fold_id=fold_id,
                seed=seed,
                field=key,
            )
    if "model_state_dict" not in payload:
        add_issue(issues, "checkpoint_missing_state", fold_id=fold_id, seed=seed)
    del payload
    gc.collect()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_root = (
        args.run_root
        or ALGORITHM_ROOT
        / "reports/microexpression/causalnet_v1"
        / str(raw_config["run_id"])
    ).resolve()
    output_path = (args.output or run_root / "independent_audit.json").resolve()
    issues: list[dict[str, Any]] = []
    input_paths = {
        key: (ALGORITHM_ROOT / value).resolve()
        for key, value in raw_config["inputs"].items()
    }
    expected_hashes = {
        f"{key}_sha256": sha256_file(path) for key, path in input_paths.items()
    }
    records = load_jsonl(input_paths["artifact_manifest"])
    records_by_id = {str(record["sample_id"]): record for record in records}
    split_manifest = load_json(input_paths["split_manifest"])
    folds = list(split_manifest["folds"])
    folds_by_id = {str(fold["fold_id"]): fold for fold in folds}
    seeds = [int(seed) for seed in raw_config["protocol"]["final_seeds"]]
    expected_sample_ids = {
        str(sample_id)
        for fold in folds
        for sample_id in fold["test_sample_ids"]
    }
    if len(records) != 164 or len(records_by_id) != 164:
        add_issue(issues, "artifact_manifest_coverage")
    if len(folds) != 16 or len(seeds) != 3:
        add_issue(issues, "formal_protocol_shape")

    resolved_config_path = run_root / "resolved_config.yaml"
    environment_path = run_root / "environment.json"
    aggregate_path = run_root / "aggregate" / "metrics.json"
    protocol_audit_path = run_root / "protocol_audit.json"
    run_index_path = run_root / "run_index.json"
    required_files = (
        resolved_config_path,
        environment_path,
        aggregate_path,
        protocol_audit_path,
        run_index_path,
    )
    for path in required_files:
        if not path.is_file():
            add_issue(issues, "missing_run_file", path=path.as_posix())
    if issues:
        result = {
            "schema_version": "causalnet_nested_loso_independent_audit_v1",
            "task_id": "EVAL-ME-004",
            "status": "fail",
            "issue_count": len(issues),
            "issues": issues,
        }
        write_json(output_path, result)
        print(json.dumps(result, ensure_ascii=True, indent=2))
        return 1

    resolved_config = yaml.safe_load(resolved_config_path.read_text(encoding="utf-8"))
    environment = load_json(environment_path)
    aggregate = load_json(aggregate_path)
    protocol_audit = load_json(protocol_audit_path)
    run_index = load_json(run_index_path)
    if resolved_config.get("execution", {}).get("formal_protocol") is not True:
        add_issue(issues, "resolved_config_not_formal")
    if resolved_config.get("execution", {}).get("fold_ids") != list(folds_by_id):
        add_issue(issues, "resolved_fold_order_mismatch")
    if resolved_config.get("execution", {}).get("seeds") != seeds:
        add_issue(issues, "resolved_seed_mismatch")
    if resolved_config.get("evidence", {}).get("input_hashes") != expected_hashes:
        add_issue(issues, "resolved_input_hash_mismatch")
    if not environment.get("cuda_available") or not environment.get("cuda_name"):
        add_issue(issues, "formal_environment_not_cuda")

    source_files = [
        SRC_ROOT
        / "elderly_monitoring/modules/mental_health/submodules/facial_affect_clue/causalnet.py",
        SRC_ROOT
        / "elderly_monitoring/modules/mental_health/submodules/facial_affect_clue/causalnet_dataset.py",
        SRC_ROOT
        / "elderly_monitoring/modules/mental_health/submodules/facial_affect_clue/causalnet_evaluation.py",
        ALGORITHM_ROOT / "scripts/run_microexpression_causalnet_nested_loso.py",
    ]
    expected_bundle_hash = combined_hash(source_files)
    if resolved_config.get("evidence", {}).get(
        "evaluation_bundle_hash"
    ) != expected_bundle_hash:
        add_issue(issues, "evaluation_bundle_hash_mismatch")
    forbidden_source = "\n".join(path.read_text(encoding="utf-8") for path in source_files)
    if "third_party" in forbidden_source or "test_fold" in forbidden_source:
        add_issue(issues, "forbidden_runtime_reference")

    expected_pairs = {(seed, fold_id) for seed in seeds for fold_id in folds_by_id}
    index_pairs = {
        (int(row["seed"]), str(row["fold_id"])) for row in run_index["results"]
    }
    if index_pairs != expected_pairs or len(run_index["results"]) != 48:
        add_issue(issues, "run_index_coverage")
    if run_index.get("failed_final_run_count") != 0:
        add_issue(issues, "run_index_failures")

    all_results: list[dict[str, Any]] = []
    predictions_by_seed: dict[int, list[dict[str, Any]]] = {seed: [] for seed in seeds}
    checkpoint_bytes = 0
    epoch_count = 0
    for seed, fold_id in sorted(expected_pairs):
        fold = folds_by_id[fold_id]
        result_path = run_root / "final" / f"seed_{seed}" / fold_id / "result.json"
        if not result_path.is_file():
            add_issue(issues, "missing_result", fold_id=fold_id, seed=seed)
            continue
        result = load_json(result_path)
        all_results.append(result)
        if (
            result.get("schema_version") != CAUSALNET_EVALUATION_SCHEMA_VERSION
            or result.get("task_id") != "EVAL-ME-004"
            or result.get("status") != "completed"
            or result.get("fold_id") != fold_id
            or result.get("seed") != seed
        ):
            add_issue(issues, "result_identity_mismatch", fold_id=fold_id, seed=seed)
        try:
            expected_partitions = validate_subject_partitions(
                records,
                train_sample_ids=fold["train_sample_ids"],
                validation_sample_ids=fold["validation_sample_ids"],
                test_sample_ids=fold["test_sample_ids"],
            )
        except (ValueError, KeyError) as error:
            add_issue(issues, "split_manifest_leakage", fold_id=fold_id, detail=str(error))
            continue
        contract = result.get("run_contract", {})
        if contract.get("partitions") != expected_partitions:
            add_issue(issues, "result_partition_mismatch", fold_id=fold_id, seed=seed)
        if contract.get("input_hashes") != expected_hashes:
            add_issue(issues, "result_input_hash_mismatch", fold_id=fold_id, seed=seed)
        if contract.get("evaluation_bundle_hash") != expected_bundle_hash:
            add_issue(issues, "result_code_hash_mismatch", fold_id=fold_id, seed=seed)
        if (
            result.get("test_used_for_selection") is not False
            or contract.get("test_used_for_selection") is not False
            or result.get("test", {}).get("test_used_for_selection") is not False
            or result.get("test", {}).get("evaluation_count") != 1
        ):
            add_issue(issues, "test_selection_or_count_violation", fold_id=fold_id, seed=seed)

        prediction_path = Path(result["test"]["prediction_path"])
        if (
            not prediction_path.is_file()
            or sha256_file(prediction_path) != result["test"]["prediction_sha256"]
        ):
            add_issue(issues, "prediction_hash_mismatch", fold_id=fold_id, seed=seed)
            continue
        predictions = load_jsonl(prediction_path)
        if predictions != result["test"]["predictions"]:
            add_issue(issues, "prediction_payload_mismatch", fold_id=fold_id, seed=seed)
        sample_ids = [str(row["sample_id"]) for row in predictions]
        if sample_ids != [str(value) for value in fold["test_sample_ids"]]:
            add_issue(issues, "fold_test_sample_mismatch", fold_id=fold_id, seed=seed)
        if any(
            row.get("seed") != seed
            or row.get("fold_id") != fold_id
            or row.get("test_used_for_selection") is not False
            or int(row["label"]) != int(records_by_id[str(row["sample_id"])]["label"])
            for row in predictions
        ):
            add_issue(issues, "prediction_metadata_mismatch", fold_id=fold_id, seed=seed)
        recomputed_fold = classification_metrics(
            [int(row["label"]) for row in predictions],
            [int(row["prediction"]) for row in predictions],
        )
        if recomputed_fold != result["test"]["metrics"]:
            add_issue(issues, "fold_metric_mismatch", fold_id=fold_id, seed=seed)
        predictions_by_seed[seed].extend(predictions)

        epoch_log_path = Path(result["epoch_log"]["path"])
        if (
            not epoch_log_path.is_file()
            or sha256_file(epoch_log_path) != result["epoch_log"]["sha256"]
        ):
            add_issue(issues, "epoch_log_hash_mismatch", fold_id=fold_id, seed=seed)
        else:
            epochs = load_jsonl(epoch_log_path)
            epoch_count += len(epochs)
            if len(epochs) != result["epoch_log"]["epoch_count"] or any(
                row.get("test_used_for_selection") is not False for row in epochs
            ):
                add_issue(issues, "epoch_log_contract", fold_id=fold_id, seed=seed)

        checkpoint_path = Path(result["checkpoint"]["path"])
        checkpoint_bytes += checkpoint_path.stat().st_size if checkpoint_path.is_file() else 0
        if (
            result["checkpoint"].get("strict_reload_verified") is not True
            or result["checkpoint"].get("strict_reload_max_logit_difference") != 0.0
        ):
            add_issue(issues, "strict_reload_not_proven", fold_id=fold_id, seed=seed)
        audit_checkpoint(
            checkpoint_path,
            result=result,
            expected_hashes=expected_hashes,
            expected_bundle_hash=expected_bundle_hash,
            issues=issues,
        )

    seed_recomputed: list[dict[str, Any]] = []
    for seed in seeds:
        rows = predictions_by_seed[seed]
        sample_ids = [str(row["sample_id"]) for row in rows]
        if len(rows) != 164 or len(set(sample_ids)) != 164 or set(sample_ids) != expected_sample_ids:
            add_issue(issues, "seed_prediction_coverage", seed=seed)
            continue
        aggregate_prediction_path = run_root / "aggregate" / f"predictions_seed_{seed}.jsonl"
        aggregate_rows = load_jsonl(aggregate_prediction_path)
        if rows != aggregate_rows:
            add_issue(issues, "aggregate_prediction_mismatch", seed=seed)
        summary = aggregate_seed_predictions(
            rows,
            seed=seed,
            bootstrap_iterations=int(raw_config["protocol"]["bootstrap_iterations"]),
        )
        recorded = next(
            item for item in aggregate["seed_summaries"] if int(item["seed"]) == seed
        )
        for metric in ("uf1", "uar", "accuracy", "balanced_accuracy"):
            if not close_enough(summary["pooled"][metric], recorded["pooled"][metric]):
                add_issue(issues, "aggregate_metric_mismatch", seed=seed, metric=metric)
            if not close_enough(summary["mean_fold"][metric], recorded["mean_fold"][metric]):
                add_issue(issues, "mean_fold_metric_mismatch", seed=seed, metric=metric)
        if summary["pooled"]["per_class"] != recorded["pooled"]["per_class"]:
            add_issue(issues, "per_class_metric_mismatch", seed=seed)
        if summary["pooled"]["confusion_matrix"] != recorded["pooled"]["confusion_matrix"]:
            add_issue(issues, "confusion_matrix_mismatch", seed=seed)
        if summary["bootstrap_95_ci"] != recorded["bootstrap_95_ci"]:
            add_issue(issues, "bootstrap_ci_mismatch", seed=seed)
        seed_recomputed.append(summary)

    if len(seed_recomputed) == 3:
        expected_decision = release_decision(seed_recomputed)
        if aggregate.get("release_decision") != expected_decision:
            add_issue(issues, "release_decision_mismatch")
        if aggregate.get("deployment_eligible") != (expected_decision == "candidate"):
            add_issue(issues, "deployment_flag_mismatch")
    if (
        protocol_audit.get("status") != "pass"
        or protocol_audit.get("final_run_count") != 48
        or protocol_audit.get("test_used_for_selection") is not False
    ):
        add_issue(issues, "protocol_audit_mismatch")

    result = {
        "schema_version": "causalnet_nested_loso_independent_audit_v1",
        "task_id": "EVAL-ME-004",
        "run_id": raw_config["run_id"],
        "status": "pass" if not issues else "fail",
        "issue_count": len(issues),
        "issues": issues,
        "fold_count": len(folds),
        "seed_count": len(seeds),
        "expected_final_run_count": 48,
        "observed_final_run_count": len(all_results),
        "expected_samples_per_seed": 164,
        "observed_samples_per_seed": {
            str(seed): len(predictions_by_seed[seed]) for seed in seeds
        },
        "test_used_for_selection": False,
        "leakage_issue_count": sum(
            "leakage" in str(issue["code"]) or "selection" in str(issue["code"])
            for issue in issues
        ),
        "checkpoint_count": sum(
            1
            for seed, fold_id in expected_pairs
            if (run_root / "final" / f"seed_{seed}" / fold_id / "best.pt").is_file()
        ),
        "checkpoint_bytes": checkpoint_bytes,
        "total_epoch_count": epoch_count,
        "input_hashes": expected_hashes,
        "evaluation_bundle_hash": expected_bundle_hash,
        "aggregate_metrics_sha256": sha256_file(aggregate_path),
        "release_decision": aggregate.get("release_decision"),
        "seed_statistics": aggregate.get("seed_statistics"),
        "environment": environment,
    }
    output_sha256 = write_json(output_path, result)
    print(
        json.dumps(
            {**result, "output_path": output_path.as_posix(), "output_sha256": output_sha256},
            ensure_ascii=True,
            indent=2,
        )
    )
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
