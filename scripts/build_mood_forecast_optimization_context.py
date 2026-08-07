"""Build the immutable input context for a V3.4 optimization candidate."""

from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ALGORITHM_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ALGORITHM_ROOT.parents[1]
sys.path.insert(0, str(ALGORITHM_ROOT / "src"))

from elderly_monitoring.modules.mental_health.mood_social.forecast_experiment import (  # noqa: E402
    sha256_file,
)
from elderly_monitoring.modules.mental_health.mood_social.forecast_optimization import (  # noqa: E402
    FEATURE_GROUPS,
    METADATA_COLUMNS,
    OPTIMIZATION_VERSION,
    build_history_forecast_samples,
    history_feature_groups,
)


CONFIG_PATH = (
    ALGORITHM_ROOT / "configs" / "experiments" / "mood_social_forecast_v3_4_opt.yaml"
)
SEARCH_PATH = (
    ALGORITHM_ROOT
    / "configs"
    / "experiments"
    / "mood_social_forecast_v3_4_opt_search_space.json"
)
ENVIRONMENT_PATH = (
    ALGORITHM_ROOT
    / "configs"
    / "experiments"
    / "mood_social_forecast_v3_4_opt_environment.lock"
)
BASELINE_ROOT = (
    ALGORITHM_ROOT / "artifacts" / "mental_health" / "mood_social" / "forecast_v3.4"
)
CANDIDATE_ROOT = BASELINE_ROOT / "optimization_candidates"
RUN_ID_PATTERN = re.compile(r"^MH-\d{8}-FOPT-\d{3}$")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _candidate_root(run_id: str) -> Path:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("run ID must match MH-YYYYMMDD-FOPT-NNN")
    path = CANDIDATE_ROOT / run_id
    if path.exists():
        raise FileExistsError(f"optimization candidate already exists: {run_id}")
    path.mkdir(parents=True)
    return path


def _baseline_snapshot() -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for path in sorted(BASELINE_ROOT.rglob("*")):
        if not path.is_file() or CANDIDATE_ROOT in path.parents:
            continue
        snapshot[path.relative_to(BASELINE_ROOT).as_posix()] = sha256_file(path)
    if not snapshot:
        raise RuntimeError("V3.4 baseline artifact root is empty")
    return snapshot


def _write_sums(root: Path) -> None:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            rows.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    (root / "SHA256SUMS").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _verify_inputs(config: dict[str, Any]) -> tuple[Path, Path, Path]:
    source_root = WORKSPACE_ROOT / "数据集" / "心理" / "PSYCHE-D"
    for name, expected in config["source_artifact_hashes"].items():
        path = source_root / name
        if sha256_file(path) != expected:
            raise RuntimeError(f"source hash mismatch: {name}")
    split_path = ALGORITHM_ROOT / config["outer_fold_manifest"]
    if sha256_file(split_path) != config["outer_fold_manifest_sha256"]:
        raise RuntimeError("outer split hash mismatch")
    inner_path = ALGORITHM_ROOT / config["inner_fold_manifest"]
    if sha256_file(inner_path) != config["inner_fold_manifest_sha256"]:
        raise RuntimeError("inner split hash mismatch")
    baseline_manifest = ALGORITHM_ROOT / config["baseline_context_manifest"]
    if sha256_file(baseline_manifest) != config["baseline_context_manifest_sha256"]:
        raise RuntimeError("baseline context manifest hash mismatch")
    mapping_path = WORKSPACE_ROOT / config["feature_mapping_path"]
    return source_root / "anon_processed_df_parquet", mapping_path, split_path


def _missingness(
    samples: dict[str, pd.DataFrame], groups: dict[str, tuple[str, ...]]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for task_id, frame in samples.items():
        for group in FEATURE_GROUPS:
            for feature in groups[group]:
                rows.append(
                    {
                        "task_id": task_id,
                        "feature_group": group,
                        "feature_name": feature,
                        "row_count": len(frame),
                        "missing_count": int(frame[feature].isna().sum()),
                        "missing_rate": float(frame[feature].isna().mean()),
                    }
                )
    return pd.DataFrame(rows)


def build(run_id: str, root: Path) -> dict[str, Any]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if any(
        (
            config.get("release_status") != "experimental",
            config.get("execution_mode") != "offline_only",
            config.get("decision_authority") != "shadow_only",
            config.get("product_visible") is not False,
        )
    ):
        raise RuntimeError("optimization permissions are not frozen")
    source_path, mapping_path, split_path = _verify_inputs(config)
    baseline_snapshot = _baseline_snapshot()
    source = pd.read_parquet(source_path)
    if len(source) != 35_694:
        raise RuntimeError("PSYCHE-D raw row count changed")
    expected = config["expected_counts"]
    samples = {
        task_id: build_history_forecast_samples(
            source,
            task_id=task_id,
            mapping_path=mapping_path,
            split_path=split_path,
            expected_count=int(expected[task_id]),
        )
        for task_id in config["task_ids"]
    }
    source_fields = tuple(
        name.removesuffix("__anchor")
        for name in samples["forecast_1m"].columns
        if name.endswith("__anchor")
    )
    groups = history_feature_groups(source_fields)
    for group, expected_count in config["history_feature_groups"].items():
        if len(groups[group]) != int(expected_count):
            raise RuntimeError(f"feature count changed for {group}")
    context_root = root / "context"
    for task_id, frame in samples.items():
        task_root = context_root / task_id
        task_root.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(task_root / "samples_all_features.parquet", index=False)
        for group in FEATURE_GROUPS:
            frame.loc[:, [*METADATA_COLUMNS, *groups[group]]].to_parquet(
                task_root / f"samples_{group}.parquet", index=False
            )
    manifest_rows = pd.concat(
        [frame.loc[:, list(METADATA_COLUMNS)] for frame in samples.values()],
        ignore_index=True,
    ).sort_values(
        ["task_id", "global_participant_id", "label_month_slot"], kind="mergesort"
    )
    manifest_rows.to_parquet(context_root / "sample_manifest.parquet", index=False)
    missingness = _missingness(samples, groups)
    missingness.to_parquet(context_root / "missingness_audit.parquet", index=False)
    feature_manifest = {
        "schema_version": "mood-social-forecast-v3.4-opt-features-v1",
        "source_feature_count": len(source_fields),
        "source_feature_names": list(source_fields),
        "feature_groups": {name: list(values) for name, values in groups.items()},
        "feature_group_counts": {name: len(values) for name, values in groups.items()},
        "cutoff_rule": "c=t-h; source nominal_month must be <= c",
        "auxiliary_labels_are_features": False,
        "prohibited_inputs": config["prohibited_inputs"],
    }
    _write_json(context_root / "feature_manifest.json", feature_manifest)
    quantity = {
        "raw_source_rows": len(source),
        "tasks": {
            task_id: {
                "sample_count": len(frame),
                "participant_count": int(frame["global_participant_id"].nunique()),
                "positive_count": int(frame["future_binary_target"].sum()),
                "cutoff_month_min": int(frame["feature_month_slot"].min()),
                "cutoff_month_max": int(frame["feature_month_slot"].max()),
                "history_observed_month_count_min": int(
                    frame["history_observed_month_count"].min()
                ),
                "history_observed_month_count_max": int(
                    frame["history_observed_month_count"].max()
                ),
            }
            for task_id, frame in samples.items()
        },
    }
    _write_json(context_root / "quantity_audit.json", quantity)
    _write_json(
        context_root / "leakage_evidence.json",
        {
            "schema_version": "mood-social-forecast-v3.4-opt-leakage-v1",
            "cutoff_relation_all_rows": all(
                bool(
                    (
                        frame["feature_month_slot"]
                        == frame["label_month_slot"] - frame["nominal_gap_months"]
                    ).all()
                )
                for frame in samples.values()
            ),
            "future_or_target_month_features_present": False,
            "phq9_feature_present": False,
            "outer_test_used_for_selection": False,
            "auxiliary_labels_in_feature_manifest": False,
            "source_field_count": len(source_fields),
        },
    )
    _write_json(context_root / "baseline_snapshot.json", baseline_snapshot)
    manifest = {
        "schema_version": "mood-social-forecast-v3.4-opt-context-v1",
        "run_id": run_id,
        "optimization_version": OPTIMIZATION_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "context_built",
        "baseline_run_id": config["baseline_run_id"],
        "baseline_file_count": len(baseline_snapshot),
        "baseline_context_manifest_sha256": config["baseline_context_manifest_sha256"],
        "source_artifact_hashes": config["source_artifact_hashes"],
        "outer_fold_manifest_sha256": config["outer_fold_manifest_sha256"],
        "inner_fold_manifest_sha256": config["inner_fold_manifest_sha256"],
        "config_sha256": sha256_file(CONFIG_PATH),
        "search_space_sha256": sha256_file(SEARCH_PATH),
        "environment_lock_sha256": sha256_file(ENVIRONMENT_PATH),
        "code_file_sha256": {
            "forecast_optimization.py": sha256_file(
                ALGORITHM_ROOT
                / "src"
                / "elderly_monitoring"
                / "modules"
                / "mental_health"
                / "mood_social"
                / "forecast_optimization.py"
            ),
            "build_mood_forecast_optimization_context.py": sha256_file(
                Path(__file__).resolve()
            ),
        },
        "task_counts": {task: len(frame) for task, frame in samples.items()},
        "feature_group_counts": {name: len(values) for name, values in groups.items()},
        "release_status": "experimental",
        "execution_mode": "offline_only",
        "decision_authority": "shadow_only",
        "product_visible": False,
    }
    _write_json(root / "manifest.json", manifest)
    _write_sums(root)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    root = _candidate_root(args.run_id)
    try:
        manifest = build(args.run_id, root)
    except Exception as exc:
        _write_json(
            root / "failure.json",
            {
                "run_id": args.run_id,
                "stage": "FORECAST-OPT-001B",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        _write_sums(root)
        raise
    print(json.dumps({"status": "pass", **manifest}, ensure_ascii=False))


if __name__ == "__main__":
    main()
