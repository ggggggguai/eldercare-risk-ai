"""Build and audit the V3.4 PSYCHE-D forecast context artifacts.

This script intentionally stops at auditable samples and split metadata. It
does not fit a model, touch the V3.3 package, or write backend data.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import ParameterSampler, StratifiedKFold

ALGORITHM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ALGORITHM_ROOT / "src"))

from elderly_monitoring.datasets.adapters.psyche_d import (  # noqa: E402
    LABEL_FIELDS,
    PSYCHE_D_SOURCE_FIELDS,
)
from elderly_monitoring.modules.mental_health.mood_social.forecast_experiment import (  # noqa: E402
    EXPERIMENT_VERSION,
    build_forecast_pair,
    sha256_file,
)


SEED = 20260728
EXPECTED_COUNTS = {"forecast_1m": 9393, "forecast_2m": 9280, "common": 8494}
EXPECTED_SOURCE_ROWS = 35694
EXPECTED_PARTICIPANTS = {"forecast_1m": 3635, "forecast_2m": 3593}
EXPECTED_CANDIDATE_SHA256 = (
    "6f9cc94f5b2dbf361d1622e62b65cc37d1edf3aaffa4e1af14310d20f9a81f1f"
)
EXPECTED_INNER_SPLIT_SHA256 = (
    "64f9e4a8d548f9c5152275b9d1fc8933814bbc9730ca74af5557284657a4e4ef"
)
WORKSPACE_ROOT = ALGORITHM_ROOT.parents[1]
SOURCE_ROOT = WORKSPACE_ROOT / "数据集" / "心理" / "PSYCHE-D"
OUTPUT_ROOT = (
    ALGORITHM_ROOT / "artifacts" / "mental_health" / "mood_social" / "forecast_v3.4"
)
CONFIG_ROOT = ALGORITHM_ROOT / "configs" / "experiments"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _git_snapshot() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=WORKSPACE_ROOT, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=WORKSPACE_ROOT,
                text=True,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unavailable", True
    code_files = {
        "src/elderly_monitoring/modules/mental_health/mood_social/forecast_experiment.py": sha256_file(
            ALGORITHM_ROOT
            / "src"
            / "elderly_monitoring"
            / "modules"
            / "mental_health"
            / "mood_social"
            / "forecast_experiment.py"
        ),
        "scripts/build_mood_forecast_context.py": sha256_file(Path(__file__).resolve()),
    }
    return {
        "code_commit": commit,
        "working_tree_dirty": dirty,
        "code_file_sha256": code_files,
    }


def _load_search_space() -> dict[str, Any]:
    path = CONFIG_ROOT / "mood_social_forecast_v3_4_search_space.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _write_candidates() -> Path:
    payload = _load_search_space()
    candidates = list(
        ParameterSampler(
            payload["lightgbm"],
            n_iter=payload["lightgbm_candidate_count"],
            random_state=SEED,
        )
    )
    path = CONFIG_ROOT / "mood_social_forecast_v3_4_lightgbm_candidates.json"
    if path.exists():
        if sha256_file(path) != EXPECTED_CANDIDATE_SHA256:
            raise RuntimeError("existing LightGBM candidate snapshot hash mismatch")
        return path
    _write_json(
        path,
        {
            "schema_version": "mood-social-forecast-v3.4-lightgbm-candidates-v1",
            "experiment_version": EXPERIMENT_VERSION,
            "random_seed": SEED,
            "sampling": "ParameterSampler_without_replacement",
            "candidate_count": len(candidates),
            "candidates": candidates,
        },
    )
    if sha256_file(path) != EXPECTED_CANDIDATE_SHA256:
        raise RuntimeError("new LightGBM candidate snapshot does not match frozen hash")
    return path


def _build_inner_split_manifest(one: pd.DataFrame, two: pd.DataFrame) -> Path:
    path = OUTPUT_ROOT / "inner_split_manifest.json"
    if path.exists():
        if sha256_file(path) != EXPECTED_INNER_SPLIT_SHA256:
            raise RuntimeError("existing inner split manifest hash mismatch")
        return path
    rows: list[dict[str, Any]] = []
    for outer_fold in range(5):
        eligible = pd.concat([one, two], ignore_index=True)
        eligible = eligible.loc[eligible["outer_fold_id"] != outer_fold]
        participant_labels = (
            eligible.groupby("global_participant_id", sort=True)["future_binary_target"]
            .max()
            .sort_index()
        )
        if participant_labels.nunique() != 2:
            raise RuntimeError(f"inner split outer fold {outer_fold} is single-class")
        splitter = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=SEED + outer_fold,
        )
        ids = participant_labels.index.to_numpy()
        labels = participant_labels.to_numpy()
        for inner_fold, (_, validation_indices) in enumerate(
            splitter.split(ids, labels)
        ):
            for index in validation_indices:
                rows.append(
                    {
                        "global_participant_id": str(ids[index]),
                        "outer_fold_id": outer_fold,
                        "inner_fold_id": inner_fold,
                        "participant_label": int(labels[index]),
                    }
                )
    rows.sort(
        key=lambda row: (
            row["outer_fold_id"],
            row["inner_fold_id"],
            row["global_participant_id"],
        )
    )
    _write_json(
        path,
        {
            "schema_version": "mood-social-forecast-v3.4-inner-split-v1",
            "experiment_version": EXPERIMENT_VERSION,
            "random_seed": SEED,
            "outer_fold_count": 5,
            "inner_fold_count": 5,
            "participant_label_rule": "max future_binary_target across the two horizon union",
            "rows": rows,
        },
    )
    if sha256_file(path) != EXPECTED_INNER_SPLIT_SHA256:
        raise RuntimeError("new inner split manifest does not match frozen hash")
    return path


def _write_timing_evidence() -> Path:
    path = OUTPUT_ROOT / "timing_evidence.json"
    _write_json(
        path,
        {
            "schema_version": "mood-social-forecast-v3.4-timing-evidence-v1",
            "experiment_version": EXPERIMENT_VERSION,
            "timing_evidence": "design_level",
            "natural_dates_available": False,
            "official_source": {
                "name": "PSYCHE-D: Prediction of Severity Change-Depression",
                "zenodo_version": "0.1",
                "doi": "10.5281/zenodo.5085146",
                "license_boundary": "CC BY-NC 4.0; non-commercial competition use with attribution",
            },
            "local_evidence_file": "数据集/心理/PSYCHE-D/数据集说明.md",
            "evidence_excerpt": [
                "公开矩阵没有真实日期，month只表示名义研究月份。",
                "官方构造确认标签月 wearable 数据来自 end PHQ-9 前 8–14 天。",
                "名义月份偏移实验只使用 m-1 或 m-2 月行，不称固定 30/60 天预测。",
            ],
            "allowed_claim": "名义提前约 1 个月或约 2 个月的公开数据条件实验",
            "prohibited_claims": [
                "固定30/60天预测",
                "逐行真实日期已验证",
                "目标老人域已验证",
            ],
        },
    )
    return path


def _write_context_artifacts(
    one: pd.DataFrame, two: pd.DataFrame, common: pd.DataFrame
) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    combined = pd.concat([one, two], ignore_index=True)
    combined = combined.sort_values(
        ["task_id", "global_participant_id", "label_month_slot"],
        kind="mergesort",
    ).reset_index(drop=True)
    combined.to_parquet(OUTPUT_ROOT / "sample_manifest.parquet", index=False)
    common.sort_values(
        ["global_participant_id", "label_month_slot"], kind="mergesort"
    ).to_parquet(OUTPUT_ROOT / "common_target_windows.parquet", index=False)
    for task, frame in (("forecast_1m", one), ("forecast_2m", two)):
        task_root = OUTPUT_ROOT / task
        task_root.mkdir(parents=True, exist_ok=True)
        frame.sort_values(
            ["global_participant_id", "label_month_slot"], kind="mergesort"
        ).to_parquet(task_root / "samples.parquet", index=False)
    _write_json(
        OUTPUT_ROOT / "raw_feature_names.json",
        {
            "schema_version": "mood-social-forecast-v3.4-feature-list-v1",
            "raw_feature_names": list(PSYCHE_D_SOURCE_FIELDS),
            "raw_feature_count": len(PSYCHE_D_SOURCE_FIELDS),
            "label_fields_excluded": list(LABEL_FIELDS),
        },
    )


def main() -> None:
    source_path = SOURCE_ROOT / "anon_processed_df_parquet"
    mapping_path = SOURCE_ROOT / "p0_feature_mapping_v1.yaml"
    split_path = (
        ALGORITHM_ROOT
        / "data"
        / "processed"
        / "mental_health"
        / "mood_social"
        / "v3.3.3"
        / "splits"
        / "split_manifest.json"
    )
    acceptance_path = (
        ALGORITHM_ROOT
        / "models"
        / "mental_health"
        / "mood_social"
        / "v3.3.3"
        / "acceptance.json"
    )
    source_hashes = {
        "anon_processed_df_parquet": "4bf1d7622d947f5bb62067c91789e1ef5de0411f439a772b8a7941282cc1cd78",
        "Feature explanations - anonymized data - Sheet1.tsv": "e346b57eb81c8c7d22a6ff0bda71214e527d8bed72eef1c2218d7e5826612e3e",
        "p0_feature_mapping_v1.yaml": "6aa7c562def0b9b4dabc0c1ad14fbbb3fdde7b47b6540d45810fa954e4a7caff",
    }
    for name, expected in source_hashes.items():
        actual = sha256_file(SOURCE_ROOT / name)
        if actual != expected:
            raise RuntimeError(f"source hash mismatch: {name}")
    if (
        sha256_file(split_path)
        != "e9915dbc590a6ea454c34d26558ac5866c57ee8a8e78f44cdcc79c859df77ee3"
    ):
        raise RuntimeError("V3.3 split manifest hash mismatch")
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    if acceptance.get("status") != "passed":
        raise RuntimeError("V3.3 acceptance status is not passed")
    source = pd.read_parquet(
        source_path, columns=[*PSYCHE_D_SOURCE_FIELDS, *LABEL_FIELDS]
    )
    if len(source) != EXPECTED_SOURCE_ROWS:
        raise RuntimeError(
            f"raw source row count {len(source)} != {EXPECTED_SOURCE_ROWS}"
        )
    one, two, common = build_forecast_pair(
        source,
        mapping_path=mapping_path,
        split_path=split_path,
        expected_1m=EXPECTED_COUNTS["forecast_1m"],
        expected_2m=EXPECTED_COUNTS["forecast_2m"],
        expected_common=EXPECTED_COUNTS["common"],
    )
    if one["global_participant_id"].nunique() != EXPECTED_PARTICIPANTS["forecast_1m"]:
        raise RuntimeError("forecast_1m participant count mismatch")
    if two["global_participant_id"].nunique() != EXPECTED_PARTICIPANTS["forecast_2m"]:
        raise RuntimeError("forecast_2m participant count mismatch")
    _write_context_artifacts(one, two, common)
    candidate_path = _write_candidates()
    inner_path = _build_inner_split_manifest(one, two)
    timing_path = _write_timing_evidence()
    split_reference = {
        "schema_version": "mood-social-forecast-v3.4-split-reference-v1",
        "source_manifest_path": "data/processed/mental_health/mood_social/v3.3.3/splits/split_manifest.json",
        "source_manifest_sha256": sha256_file(split_path),
        "source_manifest_version": "mood-social-nested-participant-split-manifest-v1",
        "assignment_key": "global_participant_id=psyche_d::participant_id",
        "outer_fold_field": "participant_assignments[].outer_fold",
        "random_seed": SEED,
        "psyche_d_participant_count": 4036,
    }
    _write_json(OUTPUT_ROOT / "split_manifest_reference.json", split_reference)
    manifest = {
        "schema_version": "mood-social-forecast-v3.4-context-manifest-v1",
        "experiment_version": EXPERIMENT_VERSION,
        "context_schema_version": "mood-social-forecast-context-v3.4-v1",
        "dataset_id": "psyche_d",
        "source_root_relative": "数据集/心理/PSYCHE-D",
        "source_artifact_hashes": source_hashes,
        "v3_3_acceptance_path": "models/mental_health/mood_social/v3.3.3/acceptance.json",
        "v3_3_acceptance_sha256": sha256_file(acceptance_path),
        "split_manifest_sha256": sha256_file(split_path),
        "raw_source_row_count": len(source),
        "task_counts": {
            "forecast_1m": len(one),
            "forecast_2m": len(two),
            "common": len(common),
        },
        "participant_counts": {
            "forecast_1m": int(one["global_participant_id"].nunique()),
            "forecast_2m": int(two["global_participant_id"].nunique()),
        },
        "inner_split_manifest_sha256": sha256_file(inner_path),
        "lightgbm_candidates_sha256": sha256_file(candidate_path),
        "timing_evidence_sha256": sha256_file(timing_path),
        "raw_feature_count": len(PSYCHE_D_SOURCE_FIELDS),
        "raw_feature_names": list(PSYCHE_D_SOURCE_FIELDS),
        "release_status": "experimental",
        "execution_mode": "offline_only",
        "decision_authority": "shadow_only",
        "product_visible": False,
        **_git_snapshot(),
    }
    _write_json(OUTPUT_ROOT / "manifest.json", manifest)
    sums: list[str] = []
    for path in sorted(
        p for p in OUTPUT_ROOT.rglob("*") if p.is_file() and p.name != "SHA256SUMS"
    ):
        sums.append(f"{sha256_file(path)}  {path.relative_to(OUTPUT_ROOT).as_posix()}")
    (OUTPUT_ROOT / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"status": "pass", "manifest": manifest}, ensure_ascii=False, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
