"""Frozen protocol, exposure, split and selection-lock contracts for R10."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from elderly_monitoring.modules.mental_health.mood_social.r7.contract import (
    project_root,
    repository_root,
    sha256_file,
)
from elderly_monitoring.modules.mental_health.mood_social.r9.contract import (
    DEFAULT_DATA_RELATIVE as R9_DATA_RELATIVE,
)

from .features import (
    ACTIVITY_SLEEP_BASE_FEATURES,
    ACTIVITY_SLEEP_INTERACTION_FEATURES,
    SLEEP_HISTORY_FEATURES,
    add_activity_sleep_interactions,
)


R10_PROTOCOL_VERSION = "mood-social-v3.3.3-r10"
R10_FROZEN_DOCUMENT_SHA256 = (
    "95B07A4D1FA19D2BD680A2A804788B858D40F243A71AEE23BDAA5BDA31BC1ABF"
)
R10_REPEAT_SEEDS = (20260821, 20260822, 20260823)
R10_OUTER_FOLDS = 5
R10_INNER_FOLDS = 5
R10_BOOTSTRAP_SEED = 20260824
R10_BOOTSTRAP_RESAMPLES = 2000

DEFAULT_DATA_RELATIVE = Path("data/processed/mental_health/mood_social/v3.3.3-r10")
DEFAULT_REPORT_RELATIVE = Path(
    "reports/mental_health/mood_social/v3.3.3-r10/OPT-V333-R10-000-007"
)
DEFAULT_CONFIG_RELATIVE = Path("configs/training/mood_social_v3_3_3_r10.yaml")
FROZEN_DOCUMENT_RELATIVE = Path(
    "项目文档/开发协作/情绪与社交关注模块V3.3.3优化开发/"
    "00-V3.3.3-r10优化方案冻结说明.md"
)


def write_json(path: Path, value: Any, *, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _environment(root: Path) -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in ("numpy", "pandas", "scikit-learn", "pyarrow", "joblib"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
        "algorithm_git_head": git,
        "network_download_performed": False,
    }


def _repeated_split(frame: pd.DataFrame, node: str) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    strata = frame["phq9_ge5_target"].astype(str) + frame["phq9_ge10_target"].astype(str)
    groups = frame["global_participant_id"].astype(str)
    for repeat, seed in enumerate(R10_REPEAT_SEEDS):
        outer = StratifiedGroupKFold(
            n_splits=R10_OUTER_FOLDS, shuffle=True, random_state=seed
        )
        outer_assignment = pd.Series(-1, index=frame.index, dtype=int)
        inner_maps: dict[int, pd.Series] = {}
        for outer_fold, (train_index, test_index) in enumerate(
            outer.split(frame, strata, groups)
        ):
            outer_assignment.iloc[test_index] = outer_fold
            train = frame.iloc[train_index]
            inner = StratifiedGroupKFold(
                n_splits=R10_INNER_FOLDS,
                shuffle=True,
                random_state=seed + 100 + outer_fold,
            )
            assignment = pd.Series(-1, index=train.index, dtype=int)
            train_strata = strata.loc[train.index]
            train_groups = groups.loc[train.index]
            for inner_fold, (_, validation_position) in enumerate(
                inner.split(train, train_strata, train_groups)
            ):
                assignment.iloc[validation_position] = inner_fold
            if assignment.lt(0).any():
                raise ValueError(f"R10 incomplete inner split {node}/{repeat}/{outer_fold}")
            inner_maps[outer_fold] = assignment
        if outer_assignment.lt(0).any():
            raise ValueError(f"R10 incomplete outer split {node}/{repeat}")
        current = frame.copy()
        current["repeat"] = repeat
        current["split_seed"] = seed
        current["outer_fold"] = outer_assignment
        current["inner_validation_fold_by_outer_fold"] = [
            json.dumps(
                {
                    str(fold): (
                        None
                        if outer_assignment.loc[index] == fold
                        else int(inner_maps[fold].loc[index])
                    )
                    for fold in range(R10_OUTER_FOLDS)
                },
                sort_keys=True,
            )
            for index in frame.index
        ]
        rows.append(current)
    result = pd.concat(rows, ignore_index=True)
    crossed = (
        result.groupby(["repeat", "global_participant_id"])["outer_fold"]
        .nunique()
        .gt(1)
        .sum()
    )
    if crossed:
        raise ValueError(f"R10 participants cross outer folds: {crossed}")
    return result


def _summary(frame: pd.DataFrame) -> dict[str, Any]:
    unique = frame.loc[frame["repeat"].eq(0)]
    return {
        "rows_per_repeat": int(len(unique)),
        "rows_materialized": int(len(frame)),
        "participants": int(unique["global_participant_id"].nunique()),
        "positive_ge5": int(unique["phq9_ge5_target"].sum()),
        "positive_ge10": int(unique["phq9_ge10_target"].sum()),
        "sources": unique["dataset_id"].value_counts().sort_index().to_dict(),
        "repeats": len(R10_REPEAT_SEEDS),
    }


def prepare_r10_protocol(
    *, repository_root_value: Path | None = None, overwrite: bool = False
) -> dict[str, Any]:
    root = repository_root(repository_root_value)
    frozen = project_root(root) / FROZEN_DOCUMENT_RELATIVE
    actual = sha256_file(frozen).upper()
    if actual != R10_FROZEN_DOCUMENT_SHA256:
        raise ValueError(f"R10 frozen document hash drift: {actual}")
    config = root / DEFAULT_CONFIG_RELATIVE
    if not config.is_file():
        raise FileNotFoundError(config)
    r9_data = root / R9_DATA_RELATIVE / "signatures"
    activity_sleep = add_activity_sleep_interactions(
        pd.read_parquet(r9_data / "activity_sleep.parquet")
    )
    sleep_history_all = pd.read_parquet(r9_data / "sleep_history.parquet")
    # The frozen unlabeled interval audit permits only the observed 76-100 day band.
    sleep_history = sleep_history_all.loc[
        sleep_history_all["history.age_days"].between(76.0, 100.0, inclusive="both")
    ].copy()
    if len(activity_sleep) != 2846 or activity_sleep["global_participant_id"].nunique() != 2846:
        raise ValueError("R10 A+S signature identity drift")
    if len(sleep_history) != 6358 or sleep_history["global_participant_id"].nunique() != 2887:
        raise ValueError("R10 validated S+History interval identity drift")
    activity_sleep = _repeated_split(activity_sleep, "activity_sleep")
    sleep_history = _repeated_split(sleep_history, "sleep_history")
    data = root / DEFAULT_DATA_RELATIVE
    report = root / DEFAULT_REPORT_RELATIVE
    paths = {
        "activity_sleep": data / "signatures/activity_sleep_repeated.parquet",
        "sleep_history": data / "signatures/sleep_history_repeated.parquet",
    }
    for path in paths.values():
        if path.exists() and not overwrite:
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
    activity_sleep.to_parquet(paths["activity_sleep"], index=False)
    sleep_history.to_parquet(paths["sleep_history"], index=False)
    exposure = {
        "evidence_status": "adaptive-development/reused-benchmark/locked-procedure-estimate",
        "all_subjects_previously_exposed": True,
        "new_subjects": 0,
        "real_device_rows": 0,
        "complete_seven_domain_rows": 0,
        "paired_phq_video_rows": 0,
        "network_download_performed": False,
        "production_default": "MH-20260812-R5-001",
        "production_default_changed": False,
        "v3_4_changed": False,
    }
    history_audit = {
        "assessment_before_target_asserted_upstream": True,
        "known_at_cutoff_required_at_runtime": True,
        "same_visit_excluded": True,
        "observed_age_counts": {
            str(key): int(value)
            for key, value in sleep_history_all["history.age_days"].value_counts().sort_index().items()
        },
        "validated_probability_band_days": [76, 100],
        "validated_rows": 6358,
        "validated_participants": 2887,
        "other_bands": {
            "0_13": "overlap-sensitivity-only",
            "14_75": "rule-background/interval-validation-pending",
            "101_180": "background-only",
            "over_180": "abstain",
        },
    }
    budget = {
        "activity_sleep": [
            "r9_elastic_replay",
            "interaction_elastic",
            "spline_additive",
            "crossfit_logit_fusion",
            "ordinal_auxiliary",
        ],
        "sleep_history": [
            "r9_joint_elastic_replay",
            "residual_platt",
            "residual_beta",
            "residual_restricted_isotonic",
        ],
        "calibration_families": ["platt", "beta", "restricted_isotonic"],
        "repeats": 3,
        "outer_folds": 5,
        "inner_folds": 5,
        "stop_rule": "two consecutive batches best inner delta_auprc < 0.0015",
        "post_lock_candidate_addition": False,
    }
    denylist = {
        "forbidden": [
            "current/target/future PHQ as sensor risk input",
            "dataset/source/id/path/fold/route/mask/coverage as risk input",
            "concurrent questionnaire input",
            "post-cutoff or late-backfilled record",
            "cross-modal imputation or virtual participant",
        ],
        "features_checked": sorted(
            set(ACTIVITY_SLEEP_INTERACTION_FEATURES).union(SLEEP_HISTORY_FEATURES)
        ),
    }
    signature = {
        "activity_sleep": {**_summary(activity_sleep), "features": list(ACTIVITY_SLEEP_INTERACTION_FEATURES)},
        "sleep_history": {**_summary(sleep_history), "features": list(SLEEP_HISTORY_FEATURES)},
        "same_person_same_target_required": True,
        "cross_dataset_stacking": False,
    }
    for name, value in (
        ("data_exposure_ledger.json", exposure),
        ("history_interval_audit.json", history_audit),
        ("candidate_budget.json", budget),
        ("risk_feature_denylist.json", denylist),
        ("paired_signature_manifest.json", signature),
        ("environment.json", _environment(root)),
        ("failure_ledger.json", {"failures": [], "retry_policy": "no outer retuning"}),
    ):
        write_json(report / name, value, overwrite=overwrite)
    split_manifest = {
        "seeds": list(R10_REPEAT_SEEDS),
        "participant_grouped": True,
        "outer_folds": R10_OUTER_FOLDS,
        "inner_folds": R10_INNER_FOLDS,
        "activity_sleep_sha256": sha256_file(paths["activity_sleep"]),
        "sleep_history_sha256": sha256_file(paths["sleep_history"]),
    }
    write_json(report / "split_manifest.json", split_manifest, overwrite=overwrite)
    selection_contract = {
        "selection_scope": "outer-train-inner-oof-only",
        "outer_unopened": True,
        "lock_status": "awaiting-baseline",
        "candidate_budget_sha256": canonical_sha(budget),
        "split_manifest_sha256": sha256_file(report / "split_manifest.json"),
    }
    write_json(report / "selection_lock_contract.json", selection_contract, overwrite=overwrite)
    manifest = {
        "protocol_version": R10_PROTOCOL_VERSION,
        "status": "prepared-awaiting-paired-baseline",
        "frozen_document_sha256": actual,
        "config_sha256": sha256_file(config),
        "candidate_outer_opened": False,
        "selection_locked": False,
        "tasks_completed": ["OPT-V333-R10-000-protocol", "OPT-V333-R10-001", "OPT-V333-R10-002"],
        "production_default_changed": False,
    }
    write_json(data / "protocol/r10_protocol_manifest.json", manifest, overwrite=overwrite)
    return manifest


__all__ = [
    "DEFAULT_CONFIG_RELATIVE",
    "DEFAULT_DATA_RELATIVE",
    "DEFAULT_REPORT_RELATIVE",
    "R10_BOOTSTRAP_RESAMPLES",
    "R10_BOOTSTRAP_SEED",
    "R10_FROZEN_DOCUMENT_SHA256",
    "R10_PROTOCOL_VERSION",
    "R10_REPEAT_SEEDS",
    "canonical_sha",
    "prepare_r10_protocol",
    "repository_root",
    "sha256_file",
    "write_json",
]
