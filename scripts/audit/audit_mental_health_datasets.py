"""Audit the local PSYCHE-D, RESILIENT, and NHANES mental-health datasets.

This script is intentionally read-only with respect to source datasets. It
creates aggregate audit artefacts only: no participant-level labels, features,
or identifiers are written to the report directory.

Run from the algorithm repository:

    conda run -n eldercare-ai python scripts/audit/audit_mental_health_datasets.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import subprocess
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


AUDIT_DATE = date(2026, 7, 25)
PSYCHE_LABEL_COLUMNS = [
    "phq9_score_start",
    "phq9_score_end",
    "phq9_cat_start",
    "phq9_cat_end",
]
RESILIENT_SENSOR_FILES = [
    "ScanWatch_HR.csv",
    "ScanWatch_Steps.csv",
    "Sleep_physio.csv",
    "Sleep_state.csv",
]
RESILIENT_TIMESTAMP_COLUMNS = {
    "ScanWatch_HR.csv": "HR Timestamp",
    "ScanWatch_Steps.csv": "Steps Timestamp",
    "Sleep_physio.csv": "Timestamp",
    "Sleep_state.csv": "Start time",
}


@dataclass(frozen=True)
class AuditPaths:
    workspace_root: Path
    algorithm_root: Path
    psyche_root: Path
    resilient_root: Path
    nhanes_root: Path
    output_root: Path


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_collection_hash(rows: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: item["relative_path"]):
        line = (
            f'{row["relative_path"]}\0{row["bytes"]}\0{row["sha256"]}\n'
        ).encode("utf-8")
        digest.update(line)
    return digest.hexdigest()


def _find_workspace_root(start: Path) -> Path:
    candidates = [start.resolve(), *start.resolve().parents]
    for candidate in candidates:
        if (candidate / "数据集" / "心理").is_dir():
            return candidate
    raise FileNotFoundError("无法从当前路径定位包含“数据集/心理”的工作区根目录")


def _git_revision(algorithm_root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=algorithm_root,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=algorithm_root,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError, UnicodeError):
        return {"commit": None, "dirty": None}


def resolve_paths(
    workspace_root: Path | None,
    output_root: Path | None,
) -> AuditPaths:
    inferred_workspace = _find_workspace_root(
        workspace_root if workspace_root is not None else Path.cwd()
    )
    algorithm_root = (
        inferred_workspace / "algorithm" / "eldercare-risk-ai-main"
    ).resolve()
    mental_root = inferred_workspace / "数据集" / "心理"
    resolved_output = (
        output_root.resolve()
        if output_root is not None
        else algorithm_root
        / "reports"
        / "mental_health"
        / f"dataset_audit_{AUDIT_DATE.isoformat()}"
    )
    paths = AuditPaths(
        workspace_root=inferred_workspace,
        algorithm_root=algorithm_root,
        psyche_root=mental_root / "PSYCHE-D",
        resilient_root=mental_root / "RESILIENT",
        nhanes_root=mental_root / "NHANES",
        output_root=resolved_output,
    )
    for source_root in (
        paths.psyche_root,
        paths.resilient_root,
        paths.nhanes_root,
    ):
        if not source_root.is_dir():
            raise FileNotFoundError(f"缺少数据集目录：{source_root}")
    return paths


def build_file_manifest(paths: AuditPaths) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, source_root in (
        ("PSYCHE-D", paths.psyche_root),
        ("RESILIENT", paths.resilient_root),
        ("NHANES", paths.nhanes_root),
    ):
        for file_path in sorted(path for path in source_root.rglob("*") if path.is_file()):
            relative = file_path.relative_to(paths.workspace_root).as_posix()
            rows.append(
                {
                    "dataset": dataset,
                    "relative_path": relative,
                    "bytes": file_path.stat().st_size,
                    "sha256": _sha256(file_path),
                }
            )

    manifest = pd.DataFrame(rows).sort_values(
        ["dataset", "relative_path"], kind="stable"
    )
    collections: dict[str, Any] = {}
    for dataset, group in manifest.groupby("dataset", sort=True):
        records = group.to_dict(orient="records")
        collections[dataset] = {
            "file_count": len(records),
            "bytes": int(group["bytes"].sum()),
            "collection_sha256": _canonical_collection_hash(records),
        }
    return manifest, collections


def _parse_psyche_index(index: pd.Index) -> pd.DataFrame:
    participants: list[str] = []
    months: list[int] = []
    for raw_value in index.astype(str):
        participant, separator, month_text = raw_value.rpartition("_")
        if not separator or not participant or not month_text.isdigit():
            raise ValueError(f"PSYCHE-D 样本索引不符合 participant_month：{raw_value}")
        participants.append(participant)
        months.append(int(month_text))
    return pd.DataFrame(
        {"participant_id": participants, "month": months},
        index=index,
    )


def build_psyche_feature_mapping(dictionary: pd.DataFrame) -> pd.DataFrame:
    mapping_rows: list[dict[str, Any]] = []
    for row in dictionary.to_dict(orient="records"):
        feature = str(row["Feature name"])
        category = str(row["Category"])
        subcategory = str(row["Subcategory"])
        description = row.get("Description")

        if feature in {"phq9_score_end", "phq9_cat_end"}:
            role = "target_only"
            include_state = False
            include_forecast = False
            reason = "end PHQ-9 是目标，禁止进入模型输入"
        elif feature in {"phq9_score_start", "phq9_cat_start"}:
            role = "baseline_label_optional"
            include_state = False
            include_forecast = False
            reason = "只允许 start-aware 消融；被动 sensor-only 主模型排除"
        elif category in {"Sleep", "Steps"} and subcategory == "Statistics":
            role = "sensor_v1"
            include_state = True
            include_forecast = True
            reason = "字段字典有明确统计语义；按任务时点选择 m、m-1 或 m-2 行"
        elif category in {"Sleep", "Steps"} and subcategory == "Linear regression":
            role = "quarantine_undefined_derived"
            include_state = False
            include_forecast = False
            reason = "本地字典缺少回看窗口与计算定义，P0 冻结排除"
        elif category == "Demographic" and subcategory == "Static":
            role = "static_ablation_only"
            include_state = False
            include_forecast = False
            reason = "仅用于公平性/域偏移或静态协变量消融，不进入 sensor-only 主结果"
        elif category == "Demographic" and subcategory == "Dynamic":
            role = "exclude_temporally_ambiguous"
            include_state = False
            include_forecast = False
            reason = "同月问卷/治疗/生活方式字段时序不清且与部署模态不一致"
        else:
            role = "exclude_unmapped"
            include_state = False
            include_forecast = False
            reason = "未进入冻结的 P0 特征契约"

        mapping_rows.append(
            {
                "mapping_version": "psyche_sensor_semantic_v1",
                "feature": feature,
                "category": category,
                "subcategory": subcategory,
                "description": None if pd.isna(description) else str(description),
                "role": role,
                "include_state_sensor_v1": include_state,
                "include_forecast_sensor_v1": include_forecast,
                "reason": reason,
            }
        )
    return pd.DataFrame(mapping_rows)


def audit_psyche(paths: AuditPaths) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    parquet_path = paths.psyche_root / "anon_processed_df_parquet"
    dictionary_path = (
        paths.psyche_root / "Feature explanations - anonymized data - Sheet1.tsv"
    )
    data = pd.read_parquet(parquet_path)
    dictionary = pd.read_csv(dictionary_path, sep="\t")
    index_frame = _parse_psyche_index(data.index)
    participant = index_frame["participant_id"]
    month = index_frame["month"]

    complete_labels = data[PSYCHE_LABEL_COLUMNS].notna().all(axis=1)
    any_labels = data[PSYCHE_LABEL_COLUMNS].notna().any(axis=1)
    labeled = data.loc[complete_labels].copy()
    labeled_participant = participant.loc[complete_labels]
    labeled_month = month.loc[complete_labels]

    duplicate_ids = int(data.index.duplicated().sum())
    label_sync = bool((complete_labels == any_labels).all())
    repeated = labeled_participant.value_counts().sort_index()
    repeat_distribution = (
        repeated.value_counts()
        .sort_index()
        .rename_axis("complete_windows_per_participant")
        .reset_index(name="participants")
    )
    repeat_distribution["participant_share"] = (
        repeat_distribution["participants"] / repeated.size
    )

    month_distribution = (
        labeled_month.value_counts()
        .sort_index()
        .rename_axis("nominal_month")
        .reset_index(name="complete_windows")
    )

    interval_values: list[int] = []
    continuity_pairs = 0
    continuity_equal = 0
    labeled_indexed = labeled.assign(
        participant_id=labeled_participant.to_numpy(),
        month=labeled_month.to_numpy(),
    )
    for _, group in labeled_indexed.groupby("participant_id", sort=False):
        ordered = group.sort_values("month")
        months = ordered["month"].astype(int).to_numpy()
        if len(months) > 1:
            interval_values.extend(np.diff(months).astype(int).tolist())
        previous = ordered.set_index("month")
        for current_month in months:
            prior_month = int(current_month) - 3
            if prior_month not in previous.index:
                continue
            prior_row = previous.loc[prior_month]
            current_row = previous.loc[int(current_month)]
            continuity_pairs += 1
            if float(prior_row["phq9_score_end"]) == float(
                current_row["phq9_score_start"]
            ):
                continuity_equal += 1

    interval_distribution = pd.DataFrame(
        sorted(Counter(interval_values).items()),
        columns=["nominal_month_gap", "adjacent_window_pairs"],
    )
    interval_distribution["pair_share"] = (
        interval_distribution["adjacent_window_pairs"]
        / interval_distribution["adjacent_window_pairs"].sum()
    )

    end_category = (
        labeled["phq9_cat_end"]
        .astype(int)
        .value_counts()
        .reindex(range(5), fill_value=0)
    )
    start_category = (
        labeled["phq9_cat_start"]
        .astype(int)
        .value_counts()
        .reindex(range(5), fill_value=0)
    )
    category_distribution = pd.DataFrame(
        {
            "phq_category": range(5),
            "start_windows": start_category.to_numpy(),
            "end_windows": end_category.to_numpy(),
        }
    )

    label_flags = pd.DataFrame(
        {
            "end_phq_ge10": labeled["phq9_score_end"] >= 10,
            "start_phq_ge10": labeled["phq9_score_start"] >= 10,
            "any_score_increase": (
                labeled["phq9_score_end"] > labeled["phq9_score_start"]
            ),
            "score_increase_ge5": (
                labeled["phq9_score_end"] - labeled["phq9_score_start"] >= 5
            ),
            "category_worsened": (
                labeled["phq9_cat_end"] > labeled["phq9_cat_start"]
            ),
            "incident_phq_ge10": (
                (labeled["phq9_score_start"] < 10)
                & (labeled["phq9_score_end"] >= 10)
            ),
        },
        index=labeled.index,
    )
    label_summary_rows: list[dict[str, Any]] = []
    for label_name in label_flags.columns:
        mask = label_flags[label_name]
        label_summary_rows.append(
            {
                "label": label_name,
                "positive_windows": int(mask.sum()),
                "positive_window_share": float(mask.mean()),
                "participants_with_positive": int(
                    labeled_participant.loc[mask].nunique()
                ),
            }
        )
    label_summary = pd.DataFrame(label_summary_rows)

    available_keys = set(data.index.astype(str))
    prior_availability_rows: list[dict[str, bool]] = []
    for sample_id, pid, end_month in zip(
        labeled.index.astype(str),
        labeled_participant,
        labeled_month,
        strict=True,
    ):
        del sample_id
        prior_availability_rows.append(
            {
                "has_m_minus_1": f"{pid}_{int(end_month) - 1}" in available_keys,
                "has_m_minus_2": f"{pid}_{int(end_month) - 2}" in available_keys,
            }
        )
    prior_availability = pd.DataFrame(prior_availability_rows)
    both_prior = (
        prior_availability["has_m_minus_1"]
        & prior_availability["has_m_minus_2"]
    )
    exactly_one_prior = (
        prior_availability["has_m_minus_1"]
        ^ prior_availability["has_m_minus_2"]
    )

    feature_mapping = build_psyche_feature_mapping(dictionary)
    mapped_features = set(feature_mapping["feature"])
    unmapped_columns = sorted(set(data.columns) - mapped_features)
    missing_dictionary_features = sorted(mapped_features - set(data.columns))
    sensor_v1_count = int(feature_mapping["include_state_sensor_v1"].sum())
    quarantined_count = int(
        (feature_mapping["role"] == "quarantine_undefined_derived").sum()
    )

    checks = [
        {
            "check": "sample_ids_unique",
            "passed": duplicate_ids == 0,
            "observed": duplicate_ids,
        },
        {
            "check": "four_label_fields_missing_in_sync",
            "passed": label_sync,
            "observed": int(complete_labels.sum()),
        },
        {
            "check": "quarterly_boundary_score_continuity",
            "passed": continuity_pairs > 0 and continuity_equal == continuity_pairs,
            "observed": f"{continuity_equal}/{continuity_pairs}",
        },
        {
            "check": "participant_group_split_feasible",
            "passed": labeled_participant.nunique() >= 1000,
            "observed": int(labeled_participant.nunique()),
        },
        {
            "check": "feature_dictionary_matches_matrix",
            "passed": not unmapped_columns and not missing_dictionary_features,
            "observed": {
                "unmapped_matrix_columns": unmapped_columns,
                "dictionary_only_features": missing_dictionary_features,
            },
        },
    ]

    summary = {
        "shape": {"rows": int(data.shape[0]), "columns": int(data.shape[1])},
        "all_participants": int(participant.nunique()),
        "complete_label_windows": int(complete_labels.sum()),
        "unlabeled_month_rows": int((~any_labels).sum()),
        "participants_with_complete_labels": int(labeled_participant.nunique()),
        "participants_without_complete_labels": int(
            participant.nunique() - labeled_participant.nunique()
        ),
        "label_fields_missing_in_sync": label_sync,
        "duplicate_sample_ids": duplicate_ids,
        "end_phq_ge10_windows": int(label_flags["end_phq_ge10"].sum()),
        "end_phq_ge10_participants": int(
            labeled_participant.loc[label_flags["end_phq_ge10"]].nunique()
        ),
        "category_worsened_windows": int(
            label_flags["category_worsened"].sum()
        ),
        "score_increase_ge5_windows": int(
            label_flags["score_increase_ge5"].sum()
        ),
        "incident_phq_ge10_windows": int(
            label_flags["incident_phq_ge10"].sum()
        ),
        "participants_with_two_or_more_windows": int((repeated >= 2).sum()),
        "participants_with_three_or_more_windows": int((repeated >= 3).sum()),
        "mean_windows_per_labeled_participant": float(repeated.mean()),
        "median_windows_per_labeled_participant": float(repeated.median()),
        "quarterly_continuity_pairs": continuity_pairs,
        "quarterly_continuity_equal_pairs": continuity_equal,
        "forecast_windows_with_both_prior_months": int(both_prior.sum()),
        "forecast_windows_with_exactly_one_prior_month": int(
            exactly_one_prior.sum()
        ),
        "forecast_windows_with_no_prior_month": int(
            (~prior_availability.any(axis=1)).sum()
        ),
        "sensor_v1_feature_count": sensor_v1_count,
        "quarantined_undefined_derived_feature_count": quarantined_count,
        "timing_claim": (
            "官方数据构造说明确认标签月 wearable PGHD 来自 end PHQ-9 前 "
            "8–14 天；公开矩阵没有真实日期，无法逐行重建精确时间戳。"
        ),
        "allowed_tasks": [
            "sensor_state：标签月传感器统计特征 -> end PHQ-9（近两周状态筛查）",
            "forecast_1m：m-1 月传感器统计特征 -> end PHQ-9/恶化",
            "forecast_2m：m-2 月传感器统计特征 -> end PHQ-9/恶化",
        ],
        "split_contract": (
            "按 participant_id 分组；固定约 800 人 untouched internal test，"
            "其余参与者采用 StratifiedGroupKFold。"
        ),
        "checks": checks,
    }
    tables = {
        "psyche_category_distribution.csv": category_distribution,
        "psyche_label_summary.csv": label_summary,
        "psyche_repeat_windows.csv": repeat_distribution,
        "psyche_interval_distribution.csv": interval_distribution,
        "psyche_feature_mapping_v1.csv": feature_mapping,
    }
    return summary, tables


def _resilient_phq_category(score: pd.Series) -> pd.Series:
    return pd.cut(
        score,
        bins=[-np.inf, 4, 9, 14, 19, np.inf],
        labels=["0-4", "5-9", "10-14", "15-19", "20-27"],
        right=True,
    )


def _count_resilient_ace_sensor_days(
    raw_root: Path,
    demographics: pd.DataFrame,
) -> tuple[int, list[dict[str, Any]]]:
    paired = demographics.dropna(
        subset=["ace_date", "ace_date_6months", "ace_total", "ace_total_6months"]
    ).copy()
    paired["ace_date"] = pd.to_datetime(
        paired["ace_date"], errors="coerce", format="%d/%m/%Y"
    )
    paired["ace_date_6months"] = pd.to_datetime(
        paired["ace_date_6months"], errors="coerce", format="%Y-%m-%d"
    )
    paired = paired.dropna(subset=["ace_date", "ace_date_6months"])

    participant_rows: list[dict[str, Any]] = []
    count_all_modalities_28_days = 0
    for row in paired.itertuples(index=False):
        participant = str(row.user_id)
        start_date = pd.Timestamp(row.ace_date).date()
        end_date = pd.Timestamp(row.ace_date_6months).date()
        per_file_days: dict[str, int] = {}
        participant_root = raw_root / participant
        for file_name in RESILIENT_SENSOR_FILES:
            file_path = participant_root / file_name
            timestamp_column = RESILIENT_TIMESTAMP_COLUMNS[file_name]
            if not file_path.exists() or file_path.stat().st_size == 0:
                per_file_days[file_name] = 0
                continue
            try:
                timestamps = pd.read_csv(
                    file_path,
                    usecols=[timestamp_column],
                    dtype={timestamp_column: "string"},
                )[timestamp_column]
            except pd.errors.EmptyDataError:
                per_file_days[file_name] = 0
                continue
            parsed = pd.to_datetime(timestamps, errors="coerce", utc=True)
            dates = parsed.dropna().dt.date
            valid_days = dates[(dates >= start_date) & (dates <= end_date)].nunique()
            per_file_days[file_name] = int(valid_days)
        minimum_days = min(per_file_days.values())
        if minimum_days >= 28:
            count_all_modalities_28_days += 1
        participant_rows.append(
            {
                "minimum_days_across_four_modalities": minimum_days,
                "all_four_modalities_ge28_days": minimum_days >= 28,
            }
        )
    return count_all_modalities_28_days, participant_rows


def audit_resilient(
    paths: AuditPaths,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    demographics_path = paths.resilient_root / "Demographics.csv"
    stats_path = paths.resilient_root / "summary_stats_per_participant.csv"
    raw_root = (
        paths.resilient_root
        / "Sleepmat_Watch_Data"
        / "Sleepmat_Watch_Data"
    )
    demographics = pd.read_csv(demographics_path, dtype={"user_id": "string"})
    stats = pd.read_csv(stats_path, dtype={"participant": "string"})

    phq_dates = pd.to_datetime(
        demographics["phq_date"], errors="coerce", format="%d/%m/%Y"
    )
    stats["earliest_date_parsed"] = pd.to_datetime(
        stats["earliest_date"], errors="coerce"
    )
    stats["latest_date_parsed"] = pd.to_datetime(
        stats["latest_date"], errors="coerce"
    )

    phq_distribution = (
        _resilient_phq_category(demographics["phq_total"])
        .value_counts(sort=False)
        .rename_axis("phq_score_band")
        .reset_index(name="participants")
    )
    phq_distribution["participant_share"] = (
        phq_distribution["participants"] / demographics["phq_total"].notna().sum()
    )

    earliest_wide = stats.pivot(
        index="participant",
        columns="file",
        values="earliest_date_parsed",
    )
    latest_wide = stats.pivot(
        index="participant",
        columns="file",
        values="latest_date_parsed",
    )
    phq_by_id = pd.Series(
        phq_dates.to_numpy(),
        index=demographics["user_id"],
        name="phq_date",
    )
    latest_modality_start = earliest_wide.max(axis=1, skipna=False)
    first_sensor_start = earliest_wide.min(axis=1, skipna=True)
    timing = pd.DataFrame(
        {
            "phq_date": phq_by_id.reindex(earliest_wide.index),
            "first_sensor_start": first_sensor_start,
            "all_modality_start": latest_modality_start,
            "all_modality_end": latest_wide.min(axis=1, skipna=False),
        }
    )
    timing["all_modality_start_offset_days"] = (
        timing["all_modality_start"] - timing["phq_date"]
    ).dt.days
    timing["first_sensor_start_offset_days"] = (
        timing["first_sensor_start"] - timing["phq_date"]
    ).dt.days

    offset_distribution = (
        timing["all_modality_start_offset_days"]
        .value_counts(dropna=False)
        .sort_index(na_position="last")
        .rename_axis("all_modality_start_offset_days")
        .reset_index(name="participants")
    )
    offset_distribution["timing_interpretation"] = offset_distribution[
        "all_modality_start_offset_days"
    ].map(
        lambda value: (
            "missing modality"
            if pd.isna(value)
            else ("same calendar date" if int(value) == 0 else "after PHQ date")
        )
    )

    modality_days = stats.pivot(index="participant", columns="file", values="num_days")
    all_modality_thresholds = {
        str(threshold): int((modality_days.min(axis=1) >= threshold).sum())
        for threshold in (7, 14, 28, 56)
    }

    ace_paired = demographics[
        ["ace_total", "ace_total_6months"]
    ].dropna()
    ace_decline = ace_paired["ace_total"] - ace_paired["ace_total_6months"]
    ace_all_four_28, ace_sensor_rows = _count_resilient_ace_sensor_days(
        raw_root, demographics
    )

    strict_past_counts: dict[str, int] = {}
    for days in (7, 14, 28):
        cutoff = timing["phq_date"] - pd.to_timedelta(days, unit="D")
        has_full_past_window = (
            timing["first_sensor_start"].notna()
            & timing["all_modality_end"].notna()
            & (timing["first_sensor_start"] <= cutoff)
            & (timing["all_modality_end"] < timing["phq_date"])
        )
        strict_past_counts[str(days)] = int(has_full_past_window.sum())

    checks = [
        {
            "check": "participant_key_unique",
            "passed": not demographics["user_id"].duplicated().any(),
            "observed": int(demographics["user_id"].nunique()),
        },
        {
            "check": "four_sensor_summary_rows_per_participant",
            "passed": bool((stats.groupby("participant").size() == 4).all()),
            "observed": stats.groupby("participant").size().value_counts().to_dict(),
        },
        {
            "check": "strict_past_only_windows_absent",
            "passed": all(value == 0 for value in strict_past_counts.values()),
            "observed": strict_past_counts,
        },
        {
            "check": "phq_positive_count_too_small_for_fixed_holdout",
            "passed": int((demographics["phq_total"] >= 10).sum()) < 20,
            "observed": int((demographics["phq_total"] >= 10).sum()),
        },
    ]

    summary = {
        "participants": int(demographics["user_id"].nunique()),
        "phq_complete": int(demographics["phq_total"].notna().sum()),
        "phq_ge10": int((demographics["phq_total"] >= 10).sum()),
        "phq_ge5": int((demographics["phq_total"] >= 5).sum()),
        "phq_median": float(demographics["phq_total"].median()),
        "gds_complete": int(demographics["gds_total"].notna().sum()),
        "gds_ge5": int((demographics["gds_total"] >= 5).sum()),
        "gds_ge6": int((demographics["gds_total"] >= 6).sum()),
        "gad_complete": int(demographics["gad_total"].notna().sum()),
        "gad_ge10": int((demographics["gad_total"] >= 10).sum()),
        "ace_baseline_complete": int(demographics["ace_total"].notna().sum()),
        "ace_paired_6month": int(len(ace_paired)),
        "ace_decline_ge3": int((ace_decline >= 3).sum()),
        "ace_decline_ge5": int((ace_decline >= 5).sum()),
        "ace_paired_with_all_four_modalities_ge28_days": int(ace_all_four_28),
        "sensor_files": int(len(stats)),
        "sensor_total_records": int(stats["total_records"].sum()),
        "participants_with_all_modalities_days": all_modality_thresholds,
        "all_modality_start_same_date_as_phq": int(
            (timing["all_modality_start_offset_days"] == 0).sum()
        ),
        "all_modality_start_after_phq": int(
            (timing["all_modality_start_offset_days"] > 0).sum()
        ),
        "all_modality_start_missing": int(
            timing["all_modality_start_offset_days"].isna().sum()
        ),
        "strict_past_only_windows": strict_past_counts,
        "timing_claim": (
            "没有任何参与者具备量表前 7/14/28 天四模态窗口；同日记录也没有"
            "日内先后时间，不能作为 PSYCHE-D past-only 任务的严格外部测试。"
        ),
        "recommended_protocol": [
            "先按字段语义冻结共同特征与 PSYCHE-D 模型",
            "在全部 73 人上做一次老人域跨队列、反向时间敏感性测试",
            "一次性测试后才解锁全部 73 人做嵌套交叉验证适配",
            "适配结果命名为 RESILIENT 内部适配验证，不再称外部测试",
        ],
        "checks": checks,
        "ace_sensor_coverage_rows_aggregated": len(ace_sensor_rows),
    }
    tables = {
        "resilient_phq_distribution.csv": phq_distribution,
        "resilient_sensor_start_offsets.csv": offset_distribution,
    }
    return summary, tables


def audit_nhanes(
    paths: AuditPaths,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    dpq_frames: list[pd.DataFrame] = []
    demo_frames: list[pd.DataFrame] = []
    local_schema_rows: list[dict[str, Any]] = []
    for cycle in ("G", "H"):
        dpq = pd.read_sas(paths.nhanes_root / f"DPQ_{cycle}.xpt", format="xport")
        demo = pd.read_sas(paths.nhanes_root / f"DEMO_{cycle}.xpt", format="xport")
        dpq["cycle"] = cycle
        demo["cycle"] = cycle
        dpq_frames.append(dpq)
        demo_frames.append(demo)
        local_schema_rows.append(
            {
                "file": f"DPQ_{cycle}.xpt",
                "rows": len(dpq),
                "columns": len(dpq.columns) - 1,
                "date_or_time_columns": ",".join(
                    column
                    for column in dpq.columns
                    if any(token in column.lower() for token in ("date", "time"))
                )
                or "(none)",
            }
        )
        local_schema_rows.append(
            {
                "file": f"DEMO_{cycle}.xpt",
                "rows": len(demo),
                "columns": len(demo.columns) - 1,
                "date_or_time_columns": ",".join(
                    column
                    for column in demo.columns
                    if any(token in column.lower() for token in ("date", "time"))
                )
                or "(none; RIDEXMON is season only)",
            }
        )

    dpq_all = pd.concat(dpq_frames, ignore_index=True)
    demo_all = pd.concat(demo_frames, ignore_index=True)
    day_path = paths.nhanes_root / "NHANES Preliminary Day Level Output.csv"
    day = pd.read_csv(day_path)
    day["calendar_date_parsed"] = pd.to_datetime(
        day["calendar_date"], errors="coerce"
    )
    day_years = sorted(
        int(value) for value in day["calendar_date_parsed"].dt.year.dropna().unique()
    )
    day_people = set(day["SEQN"].dropna().astype(int))
    dpq_people = set(dpq_all["SEQN"].dropna().astype(int))

    protocol_table = pd.DataFrame(
        [
            {
                "stage": 1,
                "event": "DPQ/PHQ-9",
                "location": "MEC 私密访谈",
                "relative_timing": "先",
                "interpretation": "回顾访谈前两周症状",
            },
            {
                "stage": 2,
                "event": "PAM 启动",
                "location": "MEC 检查会话结束",
                "relative_timing": "后",
                "interpretation": "设备在离开 MEC 后开始/继续记录",
            },
            {
                "stage": 3,
                "event": "PAM 佩戴",
                "location": "离开 MEC 后",
                "relative_timing": "随后约 7 天",
                "interpretation": "属于量表评估后的活动/睡眠窗口",
            },
        ]
    )
    local_schema = pd.DataFrame(local_schema_rows)
    local_schema = pd.concat(
        [
            local_schema,
            pd.DataFrame(
                [
                    {
                        "file": day_path.name,
                        "rows": len(day),
                        "columns": len(day.columns) - 1,
                        "date_or_time_columns": (
                            "calendar_date and derived sleep timestamps; "
                            "calendar dates are reset to year 2000"
                        ),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    dpq_timestamp_columns = [
        column
        for column in dpq_all.columns
        if any(token in column.lower() for token in ("date", "time"))
    ]
    checks = [
        {
            "check": "dpq_has_no_timestamp",
            "passed": not dpq_timestamp_columns,
            "observed": dpq_timestamp_columns,
        },
        {
            "check": "day_calendar_dates_are_reset_or_synthetic",
            "passed": day_years == [2000],
            "observed": {
                "years": day_years,
                "minimum": str(day["calendar_date_parsed"].min().date()),
                "maximum": str(day["calendar_date_parsed"].max().date()),
            },
        },
        {
            "check": "public_rows_cannot_establish_person_level_order",
            "passed": not dpq_timestamp_columns and day_years == [2000],
            "observed": "DPQ 无时间戳，PAM 日历日期被重置",
        },
    ]
    summary = {
        "dpq_rows": int(len(dpq_all)),
        "dpq_unique_participants": int(dpq_all["SEQN"].nunique()),
        "demo_rows": int(len(demo_all)),
        "day_rows": int(len(day)),
        "day_unique_participants": int(day["SEQN"].nunique()),
        "dpq_day_participant_overlap": int(len(dpq_people & day_people)),
        "day_calendar_date_min": str(day["calendar_date_parsed"].min().date()),
        "day_calendar_date_max": str(day["calendar_date_parsed"].max().date()),
        "day_calendar_years": day_years,
        "dpq_timestamp_columns": dpq_timestamp_columns,
        "row_level_order_verifiable": False,
        "protocol_order": "DPQ/PHQ-9 first; PAM starts at the end of the MEC session",
        "research_role": "post-assessment/post-outcome association",
        "allowed_uses": [
            "活动/睡眠特征方向验证",
            "辅助表征预训练",
            "2011–2012 与 2013–2014 跨周期稳健性分析",
        ],
        "prohibited_claim": "过去 7 天传感器数据预测随后测量的 PHQ-9",
        "checks": checks,
    }
    tables = {
        "nhanes_local_schema_audit.csv": local_schema,
        "nhanes_protocol_timing.csv": protocol_table,
    }
    return summary, tables


def build_decision_table(
    psyche: dict[str, Any],
    resilient: dict[str, Any],
    nhanes: dict[str, Any],
) -> pd.DataFrame:
    del psyche, resilient, nhanes
    return pd.DataFrame(
        [
            {
                "dataset": "PSYCHE-D",
                "primary_role": "主训练 + 参与者级内部验证",
                "time_direction": "传感器在 end PHQ-9 前 8–14 天（设计级确认）",
                "can_train_primary_model": "是",
                "can_be_strict_past_only_external_test": "否（主训练来源）",
                "p0_action": "冻结约 800 人内部测试；其余做 grouped CV",
            },
            {
                "dataset": "RESILIENT",
                "primary_role": "老人域一次性迁移敏感性测试；之后内部适配",
                "time_direction": "传感器同日或晚于基线量表",
                "can_train_primary_model": "否（73 人、PHQ>=10 仅 10 人）",
                "can_be_strict_past_only_external_test": "否",
                "p0_action": "全部 73 人先 one-shot；解锁后全体嵌套 CV，不切固定小测试集",
            },
            {
                "dataset": "NHANES",
                "primary_role": "量表后活动/睡眠关联与辅助表征",
                "time_direction": "DPQ 先，PAM 后约 7 天",
                "can_train_primary_model": "仅可作辅助关联/预训练",
                "can_be_strict_past_only_external_test": "否",
                "p0_action": "与 PSYCHE-D 主成绩分开报告，禁止前置筛查表述",
            },
            {
                "dataset": "项目自采前瞻队列",
                "primary_role": "真正老人域 past-only 外部验证",
                "time_direction": "先采 7–14 天设备数据，再做 PHQ-9/GDS",
                "can_train_primary_model": "后续可用于校准",
                "can_be_strict_past_only_external_test": "是（冻结模型后）",
                "p0_action": "现在冻结方案并启动采集；量表前窗口不得回填",
            },
        ]
    )


def write_outputs(
    paths: AuditPaths,
    manifest: pd.DataFrame,
    collections: dict[str, Any],
    psyche: dict[str, Any],
    psyche_tables: dict[str, pd.DataFrame],
    resilient: dict[str, Any],
    resilient_tables: dict[str, pd.DataFrame],
    nhanes: dict[str, Any],
    nhanes_tables: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    paths.output_root.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(
        paths.output_root / "dataset_file_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )
    for table_name, table in {
        **psyche_tables,
        **resilient_tables,
        **nhanes_tables,
    }.items():
        table.to_csv(
            paths.output_root / table_name,
            index=False,
            encoding="utf-8-sig",
        )

    decisions = build_decision_table(psyche, resilient, nhanes)
    decisions.to_csv(
        paths.output_root / "dataset_role_decisions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    sqlite_tables = {
        table_name.removesuffix(".csv"): table
        for table_name, table in {
            **psyche_tables,
            **resilient_tables,
            **nhanes_tables,
        }.items()
    }
    sqlite_tables["dataset_role_decisions"] = decisions
    with sqlite3.connect(paths.output_root / "audit_snapshot.sqlite") as connection:
        for table_name, table in sqlite_tables.items():
            table.to_sql(table_name, connection, if_exists="replace", index=False)

    all_checks = (
        [
            {"dataset": "PSYCHE-D", **check}
            for check in psyche["checks"]
        ]
        + [
            {"dataset": "RESILIENT", **check}
            for check in resilient["checks"]
        ]
        + [
            {"dataset": "NHANES", **check}
            for check in nhanes["checks"]
        ]
    )
    validation = {
        "passed": all(bool(check["passed"]) for check in all_checks),
        "checks": all_checks,
    }
    payload = {
        "audit_version": "mental-health-dataset-audit-v1",
        "audit_date": AUDIT_DATE.isoformat(),
        "source_data_read_only": True,
        "code": _git_revision(paths.algorithm_root),
        "collections": collections,
        "psyche_d": psyche,
        "resilient": resilient,
        "nhanes": nhanes,
        "validation": validation,
    }
    with (paths.output_root / "audit_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(_to_builtin(payload), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    with (paths.output_root / "validation_checks.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(_to_builtin(validation), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="审计本地心理健康公开数据集的样本、标签、时序和研究角色"
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=None,
        help="包含“数据集/心理”的工作区根目录；默认从当前路径向上搜索",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="聚合审计结果目录；默认写入 reports/mental_health/dataset_audit_2026-07-25",
    )
    parser.add_argument(
        "--skip-hashes",
        action="store_true",
        help="调试时跳过全文件 SHA-256；正式冻结不得使用",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = resolve_paths(args.workspace_root, args.output_root)
    if args.skip_hashes:
        manifest = pd.DataFrame(
            columns=["dataset", "relative_path", "bytes", "sha256"]
        )
        collections: dict[str, Any] = {
            dataset: {"file_count": None, "bytes": None, "collection_sha256": None}
            for dataset in ("PSYCHE-D", "RESILIENT", "NHANES")
        }
    else:
        manifest, collections = build_file_manifest(paths)

    psyche, psyche_tables = audit_psyche(paths)
    resilient, resilient_tables = audit_resilient(paths)
    nhanes, nhanes_tables = audit_nhanes(paths)
    payload = write_outputs(
        paths,
        manifest,
        collections,
        psyche,
        psyche_tables,
        resilient,
        resilient_tables,
        nhanes,
        nhanes_tables,
    )
    print(
        json.dumps(
            {
                "output_root": str(paths.output_root),
                "validation_passed": payload["validation"]["passed"],
                "psyche_complete_windows": psyche["complete_label_windows"],
                "resilient_participants": resilient["participants"],
                "nhanes_order": nhanes["protocol_order"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if payload["validation"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
