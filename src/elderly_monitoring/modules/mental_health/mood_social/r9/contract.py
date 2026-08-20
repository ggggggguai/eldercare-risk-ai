"""Frozen R9 data, time, feature, split and selection-lock contracts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from elderly_monitoring.modules.mental_health.mood_social.r4.data import (
    load_r4_development_frame,
)
from elderly_monitoring.modules.mental_health.mood_social.r4.features import (
    DEPLOYABLE_ACTIVITY_FEATURES,
    DEPLOYABLE_SLEEP_FEATURES,
)
from elderly_monitoring.modules.mental_health.mood_social.r5.contract import (
    build_split_assignment,
)
from elderly_monitoring.modules.mental_health.mood_social.r7.contract import (
    project_root,
    repository_root,
    sha256_file,
)
from elderly_monitoring.modules.mental_health.mood_social.r7.phq_history_expert import (
    HISTORY_FEATURES,
    build_psyche_history_frame,
)


R9_PROTOCOL_VERSION = "mood-social-v3.3.3-r9"
R9_FROZEN_DOCUMENT_SHA256 = (
    "BBB03EF95AA87A93AAFDCF5FECF2CED1CC3574F720A34F8C9807341B43DB8C5B"
)
R9_SPLIT_SEED = 20260819
R9_BOOTSTRAP_SEED = 20260820
R9_BOOTSTRAP_RESAMPLES = 2000
R9_OUTER_FOLDS = 5
R9_INNER_FOLDS = 5

DEFAULT_DATA_RELATIVE = Path(
    "data/processed/mental_health/mood_social/v3.3.3-r9"
)
DEFAULT_REPORT_RELATIVE = Path(
    "reports/mental_health/mood_social/v3.3.3-r9/OPT-V333-R9-000-protocol"
)
DEFAULT_CONFIG_RELATIVE = Path(
    "configs/training/mood_social_v3_3_3_r9.yaml"
)
FROZEN_DOCUMENT_RELATIVE = Path(
    "项目文档/开发协作/情绪与社交关注模块V3.3.3优化开发/"
    "00-V3.3.3-r9优化方案冻结说明.md"
)

ACTIVITY_FEATURES = tuple(DEPLOYABLE_ACTIVITY_FEATURES)
SLEEP_FEATURES = tuple(DEPLOYABLE_SLEEP_FEATURES)
R9_HISTORY_FEATURES = (
    *HISTORY_FEATURES,
    "history.freshness_weight",
    "history.decayed_last_score",
)

PERMANENT_DENYLIST = (
    "current_or_target_phq_items_total_severity_or_derived_label",
    "future_phq_future_behavior_or_future_label_derived_field",
    "participant_dataset_source_path_or_batch_identifier",
    "route_mask_coverage_or_fixed_missingness_as_risk_feature",
    "post_cutoff_or_late_backfilled_record",
    "concurrent_gad_isi_uls_or_other_active_questionnaire",
    "raw_video_url_frame_face_embedding_or_contact_hash",
)


def write_json(path: Path, payload: Any, *, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite R9 artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_state(root: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "head": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(run("status", "--porcelain")),
    }


def _environment(root: Path) -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in (
        "numpy",
        "pandas",
        "scikit-learn",
        "pyarrow",
        "lightgbm",
        "catboost",
        "joblib",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
        "git": _git_state(root),
        "network_download_performed": False,
    }


def _available(frame: pd.DataFrame, features: Iterable[str]) -> pd.Series:
    masks = [f"feature_mask.{name}" for name in features]
    missing = sorted(set(masks) - set(frame.columns))
    if missing:
        raise ValueError(f"R9 availability masks missing: {missing}")
    return frame[masks].fillna(0).max(axis=1).gt(0)


def _freshness_weight(age_days: pd.Series) -> pd.Series:
    age = pd.to_numeric(age_days, errors="raise").astype(float)
    weight = np.where(
        age <= 30.0,
        1.0,
        np.where(
            age <= 100.0,
            np.exp(-(age - 30.0) / 90.0),
            np.where(age <= 180.0, 0.0, 0.0),
        ),
    )
    return pd.Series(weight, index=age.index, dtype=float)


def build_signature_frames(
    frame: pd.DataFrame, split: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create only real-person A+S and S+strict-prior-PHQ signatures."""

    split_columns = [
        "r4_row_id",
        "outer_fold",
        "inner_validation_fold_by_outer_fold",
    ]
    joined = frame.drop(
        columns=["outer_fold", "inner_validation_fold_by_outer_fold"],
        errors="ignore",
    ).merge(split[split_columns], on="r4_row_id", how="left", validate="one_to_one")
    activity_available = _available(joined, ACTIVITY_FEATURES)
    sleep_available = _available(joined, SLEEP_FEATURES)
    activity_sleep = joined.loc[activity_available & sleep_available].copy()
    activity_sleep = activity_sleep.rename(
        columns={
            "phq9_ge5_r3_target": "phq9_ge5_target",
            "phq9_ge10_r3_target": "phq9_ge10_target",
        }
    )
    if (
        len(activity_sleep) != 2846
        or activity_sleep["global_participant_id"].nunique() != 2846
        or activity_sleep["r4_row_id"].duplicated().any()
    ):
        raise ValueError("R9 A+S signature count or identity drift")
    if set(activity_sleep["dataset_id"].unique()) != {"nhanes", "resilient"}:
        raise ValueError("R9 A+S source set drift")

    history = build_psyche_history_frame(frame)
    history["history.freshness_weight"] = _freshness_weight(
        history["history.age_days"]
    )
    history["history.decayed_last_score"] = (
        history["history.last_score"] * history["history.freshness_weight"]
    )
    sleep_columns = [
        "r4_row_id",
        "dataset_id",
        *SLEEP_FEATURES,
        *(f"feature_mask.{name}" for name in SLEEP_FEATURES),
    ]
    sleep_history = history.merge(
        joined[sleep_columns + split_columns[1:]],
        on="r4_row_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_source"),
    )
    if "dataset_id_source" in sleep_history:
        if not sleep_history["dataset_id_source"].eq("psyche_d").all():
            raise ValueError("R9 S+H source identity mismatch")
        sleep_history = sleep_history.drop(columns=["dataset_id_source"])
    sleep_history = sleep_history.loc[
        _available(sleep_history, SLEEP_FEATURES)
    ].copy()
    if (
        len(sleep_history) != 6743
        or sleep_history["global_participant_id"].nunique() != 3075
        or sleep_history["r4_row_id"].duplicated().any()
    ):
        raise ValueError("R9 S+H signature count or identity drift")
    if (sleep_history["history.age_days"] <= 0).any():
        raise ValueError("R9 history contains a non-prior assessment")
    return (
        activity_sleep.sort_values("r4_row_id", kind="stable").reset_index(drop=True),
        sleep_history.sort_values("r4_row_id", kind="stable").reset_index(drop=True),
    )


def _signature_summary(frame: pd.DataFrame) -> dict[str, Any]:
    participant_ids = sorted(frame["global_participant_id"].astype(str).unique())
    row_ids = sorted(frame["r4_row_id"].astype(str))
    return {
        "rows": int(len(frame)),
        "participants": int(len(participant_ids)),
        "sources": {
            str(key): int(value)
            for key, value in frame["dataset_id"].value_counts().sort_index().items()
        },
        "phq9_ge5_positive": int(frame["phq9_ge5_target"].sum()),
        "phq9_ge10_positive": int(frame["phq9_ge10_target"].sum()),
        "participant_id_sha256": hashlib.sha256(
            "\n".join(participant_ids).encode("utf-8")
        ).hexdigest(),
        "row_id_sha256": hashlib.sha256("\n".join(row_ids).encode("utf-8")).hexdigest(),
    }


def build_r9_protocol_artifacts(
    *,
    repository_root_value: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Materialize R9-000/001 contracts without opening candidate outer results."""

    root = repository_root(repository_root_value)
    project = project_root(root)
    frozen = project / FROZEN_DOCUMENT_RELATIVE
    actual_frozen_sha = sha256_file(frozen).upper()
    if actual_frozen_sha != R9_FROZEN_DOCUMENT_SHA256:
        raise ValueError(f"R9 frozen document hash drift: {actual_frozen_sha}")
    config = root / DEFAULT_CONFIG_RELATIVE
    if not config.is_file():
        raise FileNotFoundError(config)
    data = root / DEFAULT_DATA_RELATIVE
    report = root / DEFAULT_REPORT_RELATIVE
    protocol = data / "protocol"
    expected = [
        protocol / "r9_protocol_manifest.json",
        protocol / "split_manifest.json",
        protocol / "r9_split.parquet",
        data / "signatures/activity_sleep.parquet",
        data / "signatures/sleep_history.parquet",
        report / "data_exposure_ledger.json",
        report / "runtime_feature_contract.json",
        report / "paired_signature_manifest.json",
        report / "phq_history_freshness_contract.json",
        report / "risk_feature_denylist.json",
        report / "candidate_budget.json",
        report / "selection_lock_contract.json",
        report / "environment.json",
        report / "failed_runs.jsonl",
    ]
    if not overwrite:
        existing = [str(path) for path in expected if path.exists()]
        if existing:
            raise FileExistsError(f"refusing to overwrite R9 protocol: {existing}")

    frame = load_r4_development_frame(repository_root=root)
    split = build_split_assignment(frame, seed=R9_SPLIT_SEED).rename(
        columns={"r5_row_id": "r9_row_id"}
    )
    activity_sleep, sleep_history = build_signature_frames(frame, split)
    protocol.mkdir(parents=True, exist_ok=True)
    (data / "signatures").mkdir(parents=True, exist_ok=True)
    split.to_parquet(protocol / "r9_split.parquet", index=False)
    activity_sleep.to_parquet(data / "signatures/activity_sleep.parquet", index=False)
    sleep_history.to_parquet(data / "signatures/sleep_history.parquet", index=False)

    runtime_contract = {
        "protocol_version": R9_PROTOCOL_VERSION,
        "activity_features": list(ACTIVITY_FEATURES),
        "sleep_features": list(SLEEP_FEATURES),
        "history_features": list(R9_HISTORY_FEATURES),
        "source_route_mask_coverage": "weights/quality/routing/evaluation-only",
        "cross_modal_imputation": False,
        "training_serving_parity_required": True,
        "current_target_future_phq_in_sensor_experts": False,
    }
    freshness = {
        "protocol_version": R9_PROTOCOL_VERSION,
        "bands": {
            "0_13": "sensitivity-overlap-only",
            "14_30": "strong-voting",
            "31_100": "continuous-decay-voting",
            "101_180": "background-retest-no-vote",
            "over_180": "abstain",
        },
        "primary_evaluation_days": [14, 100],
        "constraints": [
            "assessment_time < target_date",
            "known_at <= inference_cutoff",
            "not_same_target_visit",
        ],
        "observed_age_days": {
            key: float(value)
            for key, value in sleep_history["history.age_days"].describe().items()
        },
        "runtime_previous_cutoff_days": 90,
        "train_serve_mismatch_fixed": True,
    }
    signatures = {
        "protocol_version": R9_PROTOCOL_VERSION,
        "activity_sleep": {
            **_signature_summary(activity_sleep),
            "same_person_same_target_compatible_window_asserted": True,
            "path": "signatures/activity_sleep.parquet",
            "sha256": sha256_file(data / "signatures/activity_sleep.parquet"),
        },
        "sleep_history": {
            **_signature_summary(sleep_history),
            "strict_prior_history_asserted": True,
            "path": "signatures/sleep_history.parquet",
            "sha256": sha256_file(data / "signatures/sleep_history.parquet"),
        },
        "complete_seven_domain": {
            "rows": 0,
            "training_allowed": False,
            "complete_r9_auprc_allowed": False,
        },
        "real_s10_video_current_phq": {
            "rows": 0,
            "training_allowed": False,
            "phq_video_metrics_allowed": False,
        },
    }
    exposure = {
        "protocol_version": R9_PROTOCOL_VERSION,
        "scorable_rows": int(len(frame)),
        "participants": int(frame["global_participant_id"].nunique()),
        "all_existing_subjects_previously_exposed": True,
        "evidence_status": "adaptive-development/reused-benchmark/locked-procedure estimate",
        "new_subjects": 0,
        "real_device_rows": 0,
        "real_phq_video_pairs": 0,
        "network_download_performed": False,
        "r5_production_unchanged": "MH-20260812-R5-001",
        "r8_candidate_unchanged": "MH-20260814-R8-001",
        "v3_4_forecast_changed": False,
    }
    budget = {
        "activity": 6,
        "sleep": 6,
        "activity_sleep": 4,
        "sleep_history": 3,
        "phq_history": 4,
        "social_supervised": "data-blocked/frozen-confirmation",
        "facial_phq": "data-blocked/no-paired-data",
        "complete_multimodal_stacking": "data-blocked/no-real-signature",
        "stop_rule": "two consecutive batches best inner delta AUPRC < 0.0015",
        "candidate_addition_after_selection_lock": False,
    }
    lock_contract = {
        "protocol_version": R9_PROTOCOL_VERSION,
        "inner_selection_only_before_lock": True,
        "outer_metrics_opened": False,
        "burned": False,
        "reopen_allowed": False,
        "lock_status": "unlocked",
        "candidate_budget_sha256": canonical_json_sha256(budget),
        "split_sha256": sha256_file(protocol / "r9_split.parquet"),
        "selection_lock_path": "reports/mental_health/mood_social/v3.3.3-r9/OPT-V333-R9-002-007-selection/selection_lock.json",
    }
    split_manifest = {
        "protocol_version": R9_PROTOCOL_VERSION,
        "seed": R9_SPLIT_SEED,
        "outer_folds": R9_OUTER_FOLDS,
        "inner_folds": R9_INNER_FOLDS,
        "participant_grouped": True,
        "rows": int(len(split)),
        "participants": int(split["global_participant_id"].nunique()),
        "participants_cross_outer_folds": int(
            split.groupby("global_participant_id")["outer_fold"].nunique().gt(1).sum()
        ),
        "path": "protocol/r9_split.parquet",
        "sha256": sha256_file(protocol / "r9_split.parquet"),
        "fold_rows": {
            str(key): int(value)
            for key, value in split["outer_fold"].value_counts().sort_index().items()
        },
    }
    write_json(report / "data_exposure_ledger.json", exposure, overwrite=overwrite)
    write_json(report / "runtime_feature_contract.json", runtime_contract, overwrite=overwrite)
    write_json(report / "paired_signature_manifest.json", signatures, overwrite=overwrite)
    write_json(report / "phq_history_freshness_contract.json", freshness, overwrite=overwrite)
    write_json(
        report / "risk_feature_denylist.json",
        {
            "protocol_version": R9_PROTOCOL_VERSION,
            "forbidden": list(PERMANENT_DENYLIST),
            "enforced_by_code": True,
        },
        overwrite=overwrite,
    )
    write_json(report / "candidate_budget.json", budget, overwrite=overwrite)
    write_json(report / "selection_lock_contract.json", lock_contract, overwrite=overwrite)
    write_json(report / "environment.json", _environment(root), overwrite=overwrite)
    failed = report / "failed_runs.jsonl"
    failed.parent.mkdir(parents=True, exist_ok=True)
    if failed.exists() and not overwrite:
        raise FileExistsError(failed)
    failed.write_text("", encoding="utf-8")
    write_json(protocol / "split_manifest.json", split_manifest, overwrite=overwrite)

    artifact_paths = [
        report / "data_exposure_ledger.json",
        report / "runtime_feature_contract.json",
        report / "paired_signature_manifest.json",
        report / "phq_history_freshness_contract.json",
        report / "risk_feature_denylist.json",
        report / "candidate_budget.json",
        report / "selection_lock_contract.json",
        report / "environment.json",
        protocol / "split_manifest.json",
    ]
    manifest = {
        "protocol_version": R9_PROTOCOL_VERSION,
        "status": "prepared-awaiting-paired-baseline",
        "tasks_completed": ["OPT-V333-R9-000-contract", "OPT-V333-R9-001"],
        "frozen_document_sha256": actual_frozen_sha,
        "config_sha256": sha256_file(config),
        "split_sha256": sha256_file(protocol / "r9_split.parquet"),
        "paired_signature_manifest_sha256": sha256_file(
            report / "paired_signature_manifest.json"
        ),
        "artifact_sha256": {
            path.relative_to(root).as_posix(): sha256_file(path)
            for path in artifact_paths
        },
        "candidate_outer_metrics_opened": False,
        "production_default_changed": False,
        "forecast_changed": False,
    }
    write_json(protocol / "r9_protocol_manifest.json", manifest, overwrite=overwrite)
    return manifest


__all__ = [
    "ACTIVITY_FEATURES",
    "DEFAULT_CONFIG_RELATIVE",
    "DEFAULT_DATA_RELATIVE",
    "DEFAULT_REPORT_RELATIVE",
    "PERMANENT_DENYLIST",
    "R9_BOOTSTRAP_RESAMPLES",
    "R9_BOOTSTRAP_SEED",
    "R9_FROZEN_DOCUMENT_SHA256",
    "R9_HISTORY_FEATURES",
    "R9_INNER_FOLDS",
    "R9_OUTER_FOLDS",
    "R9_PROTOCOL_VERSION",
    "R9_SPLIT_SEED",
    "SLEEP_FEATURES",
    "build_r9_protocol_artifacts",
    "build_signature_frames",
    "canonical_json_sha256",
    "write_json",
]
