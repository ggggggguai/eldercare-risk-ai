from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Independently audit MODEL-ME-006 DSTM outputs.")
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("reports/microexpression/paper_v2/MODEL-ME-006-DSTM-SMIC-NESTED-LOSO-20260807"),
    )
    parser.add_argument(
        "--temporal-audit",
        type=Path,
        default=Path("data/manifests/microexpression/paper_v2/dstm_temporal_flow_artifact_audit_smic_hs_v2.json"),
    )
    args = parser.parse_args()
    root = args.run_root.resolve()
    checks: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []

    def check(name: str, condition: bool, detail: Any = None) -> None:
        checks.append({"name": name, "passed": bool(condition), "detail": detail})
        if not condition:
            issues.append({"check": name, "detail": detail})

    aggregate_path = root / "aggregate/metrics.json"
    check("aggregate_exists", aggregate_path.is_file())
    if not aggregate_path.is_file():
        raise SystemExit(1)
    aggregate = load_json(aggregate_path)
    temporal_audit = load_json(args.temporal_audit.resolve())
    check(
        "temporal_artifact_audit_pass",
        temporal_audit.get("status") == "pass"
        and temporal_audit.get("sample_count") == 164
        and temporal_audit.get("nonzero_sequence_count") == 164,
        {
            "status": temporal_audit.get("status"),
            "sample_count": temporal_audit.get("sample_count"),
            "nonzero_sequence_count": temporal_audit.get("nonzero_sequence_count"),
        },
    )
    folds_root = root / "folds"
    fold_dirs = sorted(path for path in folds_root.iterdir() if path.is_dir()) if folds_root.is_dir() else []
    expected_fold_count = int(aggregate.get("fold_count", 0))
    seeds = [int(value) for value in aggregate.get("seeds", [])]
    check("fold_directory_count", len(fold_dirs) == expected_fold_count, {"actual": len(fold_dirs), "expected": expected_fold_count})

    final_result_paths: list[Path] = []
    checkpoint_hashes: list[dict[str, Any]] = []
    expected_sample_ids_by_seed: dict[int, set[str]] = {seed: set() for seed in seeds}
    for fold_dir in fold_dirs:
        contract_path = fold_dir / "fold_contract.json"
        reducer_meta_path = fold_dir / "reducer/reducer.joblib.json"
        reducer_path = fold_dir / "reducer/reducer.joblib"
        feature_store_path = fold_dir / "temporal_features.npz"
        check(f"{fold_dir.name}_contract_exists", contract_path.is_file())
        check(f"{fold_dir.name}_reducer_exists", reducer_path.is_file() and reducer_meta_path.is_file())
        check(f"{fold_dir.name}_feature_store_exists", feature_store_path.is_file())
        if not contract_path.is_file() or not reducer_meta_path.is_file():
            continue
        contract = load_json(contract_path)
        reducer_meta = load_json(reducer_meta_path)
        train_ids = set(str(value) for value in contract.get("train_sample_ids", []))
        validation_ids = set(str(value) for value in contract.get("validation_sample_ids", []))
        test_ids = set(str(value) for value in contract.get("test_sample_ids", []))
        fit_ids = set(str(value) for value in reducer_meta.get("fit_sample_ids", []))
        fit_subjects = set(str(value) for value in reducer_meta.get("fit_subject_ids", []))
        test_subjects = set(str(value) for value in contract.get("test_subjects", []))
        check(f"{fold_dir.name}_reducer_fit_subset_train", fit_ids <= train_ids)
        check(f"{fold_dir.name}_reducer_no_validation", not fit_ids & validation_ids)
        check(f"{fold_dir.name}_reducer_no_test_samples", not fit_ids & test_ids)
        check(f"{fold_dir.name}_reducer_no_test_subjects", not fit_subjects & test_subjects)
        check(f"{fold_dir.name}_reducer_transform_only_contract", reducer_meta.get("fit_transform_called_in_forward") is False)
        if feature_store_path.is_file():
            feature_meta_path = feature_store_path.with_suffix(feature_store_path.suffix + ".json")
            check(f"{fold_dir.name}_feature_store_hash", feature_meta_path.is_file() and load_json(feature_meta_path).get("feature_store_sha256") == sha256_file(feature_store_path))
        for seed in seeds:
            result_path = fold_dir / "final" / f"seed_{seed}" / "result.json"
            prediction_path = fold_dir / "final" / f"seed_{seed}" / "test_predictions.jsonl"
            final_result_paths.append(result_path)
            check(f"{fold_dir.name}_{seed}_result_exists", result_path.is_file())
            check(f"{fold_dir.name}_{seed}_prediction_exists", prediction_path.is_file())
            if not result_path.is_file() or not prediction_path.is_file():
                continue
            result = load_json(result_path)
            predictions = load_jsonl(prediction_path)
            test = result.get("test", {})
            check(f"{fold_dir.name}_{seed}_completed", result.get("status") == "completed")
            check(f"{fold_dir.name}_{seed}_test_once", test.get("evaluation_count") == 1)
            check(f"{fold_dir.name}_{seed}_test_not_selection", test.get("test_used_for_selection") is False and result.get("test_used_for_selection") is False)
            check(f"{fold_dir.name}_{seed}_prediction_hash", result.get("test", {}).get("prediction_sha256") == sha256_file(prediction_path))
            checkpoint_path_value = result.get("checkpoint_path")
            checkpoint_path = Path(str(checkpoint_path_value)) if checkpoint_path_value else Path()
            checkpoint_exists = checkpoint_path.is_file()
            check(f"{fold_dir.name}_{seed}_checkpoint_exists", checkpoint_exists, str(checkpoint_path))
            if checkpoint_exists:
                checkpoint_hashes.append(
                    {
                        "fold_id": fold_dir.name,
                        "seed": seed,
                        "checkpoint_path": checkpoint_path.resolve().as_posix(),
                        "checkpoint_sha256": sha256_file(checkpoint_path),
                        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
                    }
                )
            prediction_ids = {str(row["sample_id"]) for row in predictions}
            check(f"{fold_dir.name}_{seed}_prediction_coverage", prediction_ids == test_ids and len(predictions) == len(test_ids))
            expected_sample_ids_by_seed[seed].update(prediction_ids)

    for seed in seeds:
        expected = set()
        for fold_dir in fold_dirs:
            contract_path = fold_dir / "fold_contract.json"
            if contract_path.is_file():
                expected.update(str(value) for value in load_json(contract_path).get("test_sample_ids", []))
        check(f"seed_{seed}_aggregate_prediction_exists", (root / f"aggregate/predictions_seed_{seed}.jsonl").is_file())
        if (root / f"aggregate/predictions_seed_{seed}.jsonl").is_file():
            rows = load_jsonl(root / f"aggregate/predictions_seed_{seed}.jsonl")
            check(f"seed_{seed}_aggregate_coverage", {str(row["sample_id"]) for row in rows} == expected and len(rows) == len(expected))

    check("aggregate_status_completed", aggregate.get("status") == "completed")
    aggregate_hash_path = aggregate_path.with_suffix(aggregate_path.suffix + ".sha256")
    expected_aggregate_hash = aggregate_hash_path.read_text(encoding="ascii").strip() if aggregate_hash_path.is_file() else ""
    check("aggregate_sha256", bool(expected_aggregate_hash) and expected_aggregate_hash == sha256_file(aggregate_path))
    check("no_test_selection", not any(result.get("test_used_for_selection") for result in [load_json(path) for path in final_result_paths if path.is_file()]))
    check(
        "checkpoint_hash_coverage",
        len(checkpoint_hashes) == expected_fold_count * len(seeds),
        {"actual": len(checkpoint_hashes), "expected": expected_fold_count * len(seeds)},
    )
    checkpoint_hashes_path = root / "checkpoint_hashes.json"
    checkpoint_hashes_path.write_text(
        json.dumps(
            {
                "schema_version": "dstm_checkpoint_hashes_v1",
                "task_id": "MODEL-ME-006",
                "count": len(checkpoint_hashes),
                "checkpoints": checkpoint_hashes,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    checkpoint_hashes_sha256 = sha256_file(checkpoint_hashes_path)
    (checkpoint_hashes_path.with_suffix(checkpoint_hashes_path.suffix + ".sha256")).write_text(
        checkpoint_hashes_sha256 + "\n", encoding="ascii"
    )
    audit = {
        "schema_version": "dstm_nested_loso_independent_audit_v1",
        "task_id": "MODEL-ME-006",
        "status": "pass" if not issues else "failed",
        "run_root": root.as_posix(),
        "selection_results": 0,
        "final_results": sum(path.is_file() for path in final_result_paths),
        "expected_final_results": expected_fold_count * len(seeds),
        "checkpoint_hashes_path": checkpoint_hashes_path.resolve().as_posix(),
        "checkpoint_hashes_sha256": checkpoint_hashes_sha256,
        "checks": checks,
        "issues": issues,
    }
    output_path = root / "independent_audit_v2.json"
    output_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if audit["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
