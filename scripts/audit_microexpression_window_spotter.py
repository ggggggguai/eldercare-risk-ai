from __future__ import annotations

import argparse
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

from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.spotting_evaluation import (  # noqa: E402
    SPOTTING_EVALUATION_SCHEMA_VERSION,
    binary_metrics,
    sha256_file,
    validate_spotting_splits,
)


DEFAULT_CONFIG = ALGORITHM_ROOT / "configs/evaluation/microexpression_window_spotter_nested_loso.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Independently audit SPOT-ME-002 outputs.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
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


def close_enough(left: float, right: float) -> bool:
    return bool(np.isclose(left, right, rtol=0.0, atol=1e-12))


def main() -> int:
    args = parse_args()
    raw = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
    run_root = (
        args.run_root
        or ALGORITHM_ROOT / "reports/microexpression/spotter_v1" / str(raw["run_id"])
    ).resolve()
    output_path = (args.output or run_root / "independent_audit.json").resolve()
    issues: list[dict[str, Any]] = []

    def issue(code: str, **details: Any) -> None:
        issues.append({"code": code, **details})

    input_paths = {key: (ALGORITHM_ROOT / value).resolve() for key, value in raw["inputs"].items()}
    expected_input_hashes = {f"{key}_sha256": sha256_file(path) for key, path in input_paths.items()}
    feature_records = load_jsonl(input_paths["feature_manifest"])
    records_by_id = {str(record["sample_id"]): record for record in feature_records}
    if len(feature_records) != 328 or len(records_by_id) != 328:
        issue("feature_manifest_coverage")
    labels = [int(record["label"]) for record in feature_records]
    if labels.count(0) != 164 or labels.count(1) != 164:
        issue("feature_manifest_label_balance")
    for record in feature_records:
        feature_path = Path(str(record["feature_path"]))
        if not feature_path.is_file() or sha256_file(feature_path) != record["feature_sha256"]:
            issue("feature_hash_mismatch", sample_id=record["sample_id"])
            continue
        with np.load(feature_path, allow_pickle=False) as data:
            features = np.asarray(data["features"])
        if features.shape != (int(record["feature_dim"]),) or not np.all(np.isfinite(features)):
            issue("feature_contract_mismatch", sample_id=record["sample_id"])
        if bool(record.get("long_video_spotting_claim")):
            issue("forbidden_long_video_claim", sample_id=record["sample_id"])

    split_path = ALGORITHM_ROOT / "data/manifests/microexpression/spotter_v1_subject_loso_splits.json"
    split_manifest = load_json(split_path)
    try:
        validate_spotting_splits(split_manifest)
    except ValueError as error:
        issue("split_leakage_or_coverage", detail=str(error))
    expected_input_hashes["split_manifest_sha256"] = sha256_file(split_path)
    source_files = [
        SRC_ROOT / "elderly_monitoring/modules/mental_health/submodules/facial_affect_clue/microexpression_spotter.py",
        SRC_ROOT / "elderly_monitoring/modules/mental_health/submodules/facial_affect_clue/spotting_evaluation.py",
        ALGORITHM_ROOT / "scripts/run_microexpression_window_spotter_nested_loso.py",
    ]
    expected_input_hashes["evaluation_bundle_sha256"] = bundle_hash(source_files)
    if any("third_party" in path.read_text(encoding="utf-8") for path in source_files):
        issue("runtime_imports_audit_only_reference")

    resolved_path = run_root / "resolved_config.yaml"
    aggregate_path = run_root / "aggregate/metrics.json"
    protocol_path = run_root / "protocol_audit.json"
    index_path = run_root / "run_index.json"
    for path in (resolved_path, aggregate_path, protocol_path, index_path, run_root / "environment.json"):
        if not path.is_file():
            issue("missing_run_file", path=str(path))
    if issues:
        result = {
            "schema_version": "smic_window_spotter_independent_audit_v1",
            "task_id": "SPOT-ME-002",
            "status": "fail",
            "issue_count": len(issues),
            "issues": issues,
        }
        write_json(output_path, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1

    resolved = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    aggregate = load_json(aggregate_path)
    protocol = load_json(protocol_path)
    run_index = load_json(index_path)
    if resolved.get("evidence", {}).get("input_hashes") != expected_input_hashes:
        issue("resolved_input_hash_mismatch")
    seeds = [int(value) for value in raw["protocol"]["final_seeds"]]
    folds = {str(fold["fold_id"]): fold for fold in split_manifest["folds"]}
    expected_pairs = {(seed, fold_id) for seed in seeds for fold_id in folds}
    observed_pairs = {(int(row["seed"]), str(row["fold_id"])) for row in run_index["results"]}
    if expected_pairs != observed_pairs or len(run_index["results"]) != 48:
        issue("run_index_coverage")
    if int(run_index.get("failure_count", -1)) != 0:
        issue("run_index_failures")

    predictions_by_seed: dict[int, list[dict[str, Any]]] = {seed: [] for seed in seeds}
    checkpoint_count = 0
    epoch_count = 0
    for seed, fold_id in sorted(expected_pairs):
        fold = folds[fold_id]
        directory = run_root / "final" / f"seed_{seed}" / fold_id
        result_path = directory / "result.json"
        if not result_path.is_file():
            issue("missing_result", seed=seed, fold_id=fold_id)
            continue
        result = load_json(result_path)
        if (
            result.get("schema_version") != SPOTTING_EVALUATION_SCHEMA_VERSION
            or result.get("task_id") != "SPOT-ME-002"
            or result.get("status") != "completed"
            or result.get("test_used_for_selection") is not False
            or result.get("test_evaluation_count") != 1
        ):
            issue("result_contract", seed=seed, fold_id=fold_id)
        partition_subjects = result.get("partitions", {})
        if set(partition_subjects.get("train", {}).get("subject_ids", [])) & set(
            partition_subjects.get("test", {}).get("subject_ids", [])
        ):
            issue("result_subject_leakage", seed=seed, fold_id=fold_id)
        if float(result["validation"]["threshold"]) != float(result["test"]["threshold"]):
            issue("threshold_not_from_validation", seed=seed, fold_id=fold_id)

        prediction_path = Path(str(result["test"]["prediction_path"]))
        if not prediction_path.is_file() or sha256_file(prediction_path) != result["test"]["prediction_sha256"]:
            issue("prediction_hash_mismatch", seed=seed, fold_id=fold_id)
            continue
        rows = load_jsonl(prediction_path)
        if [str(row["sample_id"]) for row in rows] != [str(value) for value in fold["test_sample_ids"]]:
            issue("test_sample_mismatch", seed=seed, fold_id=fold_id)
        if any(
            row.get("test_used_for_selection") is not False
            or int(row["label"]) != int(records_by_id[str(row["sample_id"])]["label"])
            for row in rows
        ):
            issue("prediction_contract", seed=seed, fold_id=fold_id)
        recomputed = binary_metrics(
            [int(row["label"]) for row in rows], [int(row["prediction"]) for row in rows]
        )
        if recomputed != result["test"]["metrics"]:
            issue("fold_metric_mismatch", seed=seed, fold_id=fold_id)
        predictions_by_seed[seed].extend(rows)

        checkpoint_path = Path(str(result["checkpoint"]["path"]))
        if not checkpoint_path.is_file() or sha256_file(checkpoint_path) != result["checkpoint"]["sha256"]:
            issue("checkpoint_hash_mismatch", seed=seed, fold_id=fold_id)
        else:
            checkpoint_count += 1
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if (
                checkpoint.get("test_used_for_selection") is not False
                or checkpoint.get("selection_source") != "training_side_validation_subjects_only"
                or checkpoint.get("threshold") != result["test"]["threshold"]
                or checkpoint.get("input_hashes") != expected_input_hashes
            ):
                issue("checkpoint_contract", seed=seed, fold_id=fold_id)
        if (
            result["checkpoint"].get("strict_reload_verified") is not True
            or result["checkpoint"].get("strict_reload_max_logit_difference") != 0.0
        ):
            issue("checkpoint_reload", seed=seed, fold_id=fold_id)
        epoch_rows = load_jsonl(Path(str(result["epoch_log"]["path"])))
        epoch_count += len(epoch_rows)
        if any(row.get("test_used_for_selection") is not False for row in epoch_rows):
            issue("epoch_selection_leakage", seed=seed, fold_id=fold_id)

    for seed in seeds:
        rows = predictions_by_seed[seed]
        sample_ids = [str(row["sample_id"]) for row in rows]
        if len(rows) != 328 or len(set(sample_ids)) != 328 or set(sample_ids) != set(records_by_id):
            issue("seed_prediction_coverage", seed=seed)
            continue
        recomputed = binary_metrics(
            [int(row["label"]) for row in rows], [int(row["prediction"]) for row in rows]
        )
        recorded = next(summary for summary in aggregate["seed_summaries"] if int(summary["seed"]) == seed)
        for metric in ("accuracy", "macro_f1", "balanced_accuracy"):
            if not close_enough(recomputed[metric], recorded["pooled"][metric]):
                issue("aggregate_metric_mismatch", seed=seed, metric=metric)
        if recomputed["confusion_matrix"] != recorded["pooled"]["confusion_matrix"]:
            issue("aggregate_confusion_mismatch", seed=seed)
    if (
        protocol.get("status") != "pass"
        or protocol.get("run_count") != 48
        or protocol.get("test_used_for_selection") is not False
        or protocol.get("long_video_spotting_claim") is not False
        or aggregate.get("deployment_eligible") is not False
        or aggregate.get("long_video_spotting_claim") is not False
    ):
        issue("aggregate_protocol_contract")

    result = {
        "schema_version": "smic_window_spotter_independent_audit_v1",
        "task_id": "SPOT-ME-002",
        "run_id": raw["run_id"],
        "status": "pass" if not issues else "fail",
        "issue_count": len(issues),
        "issues": issues,
        "feature_count": len(feature_records),
        "subject_count": len({str(record["subject_id"]) for record in feature_records}),
        "expected_run_count": 48,
        "observed_run_count": len(run_index["results"]),
        "observed_samples_per_seed": {str(seed): len(predictions_by_seed[seed]) for seed in seeds},
        "checkpoint_count": checkpoint_count,
        "total_epoch_count": epoch_count,
        "test_used_for_selection": False,
        "leakage_issue_count": sum(
            "leakage" in str(item["code"]) or "selection" in str(item["code"]) for item in issues
        ),
        "long_video_spotting_claim": False,
        "deployment_eligible": False,
        "input_hashes": expected_input_hashes,
        "aggregate_sha256": sha256_file(aggregate_path),
        "outcome": aggregate.get("outcome"),
        "seed_statistics": aggregate.get("seed_statistics"),
    }
    output_hash = write_json(output_path, result)
    print(json.dumps({**result, "output_sha256": output_hash}, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
