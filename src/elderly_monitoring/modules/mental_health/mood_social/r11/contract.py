"""Frozen R11 exposure, lineage, split, budget and no-reopen contracts."""

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
    ACTIVITY_FEATURES,
    R9_HISTORY_FEATURES,
    SLEEP_FEATURES,
)
from elderly_monitoring.modules.mental_health.mood_social.r10.baseline import (
    load_r10_frames,
)
from elderly_monitoring.modules.mental_health.mood_social.r10.features import (
    ACTIVITY_SLEEP_INTERACTION_FEATURES,
    SLEEP_HISTORY_FEATURES,
)


R11_PROTOCOL_VERSION = "mood-social-v3.3.3-r11"
R11_FROZEN_DOCUMENT_SHA256 = (
    "9A6E645F8CEF8346AA753EBCFDF96E8D2739D8881F492A973E30DF0BD099126D"
)
R11_REPEAT_SEEDS = (20261011, 20261012, 20261013)
R11_OUTER_FOLDS = 5
R11_INNER_FOLDS = 5
R11_BOOTSTRAP_SEED = 20261014
R11_BOOTSTRAP_RESAMPLES = 2000

DEFAULT_DATA_RELATIVE = Path("data/processed/mental_health/mood_social/v3.3.3-r11")
DEFAULT_REPORT_RELATIVE = Path(
    "reports/mental_health/mood_social/v3.3.3-r11/OPT-V333-R11-000-006"
)
DEFAULT_CONFIG_RELATIVE = Path("configs/training/mood_social_v3_3_3_r11.yaml")
FROZEN_DOCUMENT_RELATIVE = Path(
    "项目文档/开发协作/情绪与社交关注模块V3.3.3优化开发/"
    "00-V3.3.3-r11优化方案冻结说明.md"
)

R10_REPORT_RELATIVE = Path(
    "reports/mental_health/mood_social/v3.3.3-r10/OPT-V333-R10-000-007"
)
R10_COMPLETION_RELATIVE = Path(
    "reports/mental_health/mood_social/v3.3.3-r10/OPT-V333-R10-009-completion"
)
R9_PACKAGE_RELATIVE = Path(
    "models/mental_health/mood_social/v3.3.3-r9/packages/MH-20260814-R9-001"
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
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _environment(root: Path) -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in ("numpy", "pandas", "scikit-learn", "pyarrow", "joblib"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
        "algorithm_git_head": head,
        "network_download_performed": False,
    }


def _repeated_split(frame: pd.DataFrame, node: str) -> pd.DataFrame:
    base = frame.drop(
        columns=["repeat", "split_seed", "outer_fold", "inner_validation_fold_by_outer_fold"],
        errors="ignore",
    ).copy()
    strata = base["phq9_ge5_target"].astype(str) + base["phq9_ge10_target"].astype(str)
    groups = base["global_participant_id"].astype(str)
    materialized: list[pd.DataFrame] = []
    for repeat, seed in enumerate(R11_REPEAT_SEEDS):
        outer = StratifiedGroupKFold(
            n_splits=R11_OUTER_FOLDS, shuffle=True, random_state=seed
        )
        outer_assignment = pd.Series(-1, index=base.index, dtype=int)
        inner_maps: dict[int, pd.Series] = {}
        for outer_fold, (train_position, test_position) in enumerate(
            outer.split(base, strata, groups)
        ):
            outer_assignment.iloc[test_position] = outer_fold
            train = base.iloc[train_position]
            inner = StratifiedGroupKFold(
                n_splits=R11_INNER_FOLDS,
                shuffle=True,
                random_state=seed + 100 + outer_fold,
            )
            assignment = pd.Series(-1, index=train.index, dtype=int)
            for inner_fold, (_, validation_position) in enumerate(
                inner.split(train, strata.loc[train.index], groups.loc[train.index])
            ):
                assignment.iloc[validation_position] = inner_fold
            if assignment.lt(0).any():
                raise ValueError(f"R11 incomplete inner split: {node}/{repeat}/{outer_fold}")
            inner_maps[outer_fold] = assignment
        if outer_assignment.lt(0).any():
            raise ValueError(f"R11 incomplete outer split: {node}/{repeat}")
        current = base.copy()
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
                    for fold in range(R11_OUTER_FOLDS)
                },
                sort_keys=True,
            )
            for index in base.index
        ]
        materialized.append(current)
    result = pd.concat(materialized, ignore_index=True)
    crossed = (
        result.groupby(["repeat", "global_participant_id"])["outer_fold"]
        .nunique()
        .gt(1)
        .sum()
    )
    if crossed:
        raise ValueError(f"R11 participants crossed outer folds: {crossed}")
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
        "repeats": len(R11_REPEAT_SEEDS),
    }


def _lineage(root: Path) -> dict[str, Any]:
    paths = {
        "r10_frozen_document": project_root(root)
        / Path(
            "项目文档/开发协作/情绪与社交关注模块V3.3.3优化开发/"
            "00-V3.3.3-r10优化方案冻结说明.md"
        ),
        "r10_outer_event": root / R10_REPORT_RELATIVE / "evaluation/outer_open_event.json",
        "r10_evaluation": root / R10_REPORT_RELATIVE / "evaluation/evaluation_report.json",
        "r10_completion": root / R10_COMPLETION_RELATIVE / "completion_audit.json",
        "r9_package_manifest": root / R9_PACKAGE_RELATIVE / "manifest.json",
        "r9_package_checksums": root / R9_PACKAGE_RELATIVE / "SHA256SUMS",
        "production_runtime": root / "configs/runtime/mood_social_current_state_production.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"R11 lineage artifact missing: {missing}")
    return {
        name: {"path": path.relative_to(root if root in path.parents else project_root(root)).as_posix(), "sha256": sha256_file(path)}
        for name, path in paths.items()
    }


def prepare_r11_protocol(
    *, repository_root_value: Path | None = None, overwrite: bool = False
) -> dict[str, Any]:
    root = repository_root(repository_root_value)
    frozen = project_root(root) / FROZEN_DOCUMENT_RELATIVE
    actual = sha256_file(frozen).upper()
    if actual != R11_FROZEN_DOCUMENT_SHA256:
        raise ValueError(f"R11 frozen document hash drift: {actual}")
    config = root / DEFAULT_CONFIG_RELATIVE
    if not config.is_file():
        raise FileNotFoundError(config)
    r10 = load_r10_frames(root)
    activity_sleep = _repeated_split(r10["activity_sleep"].loc[lambda x: x.repeat.eq(0)], "activity_sleep")
    sleep_history = _repeated_split(r10["sleep_history"].loc[lambda x: x.repeat.eq(0)], "sleep_history")
    if (len(activity_sleep.loc[activity_sleep.repeat.eq(0)]), activity_sleep.loc[activity_sleep.repeat.eq(0), "global_participant_id"].nunique()) != (2846, 2846):
        raise ValueError("R11 A+S identity drift")
    if (len(sleep_history.loc[sleep_history.repeat.eq(0)]), sleep_history.loc[sleep_history.repeat.eq(0), "global_participant_id"].nunique()) != (6358, 2887):
        raise ValueError("R11 S+History identity drift")
    data = root / DEFAULT_DATA_RELATIVE
    report = root / DEFAULT_REPORT_RELATIVE
    outputs = {
        "activity_sleep": data / "signatures/activity_sleep_repeated.parquet",
        "sleep_history": data / "signatures/sleep_history_repeated.parquet",
    }
    for path, frame in ((outputs["activity_sleep"], activity_sleep), (outputs["sleep_history"], sleep_history)):
        if path.exists() and not overwrite:
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
    lineage = _lineage(root)
    exposure = {
        "protocol_version": R11_PROTOCOL_VERSION,
        "evidence_status": "adaptive-development/reused-benchmark/post-selection",
        "all_existing_participants_burned_for_independent_confirmation": True,
        "new_participants": 0,
        "complete_seven_domain_rows": 0,
        "elderly_social_phq_rows": 0,
        "real_device_phq_rows": 0,
        "s10_video_current_phq_rows": 0,
        "network_download_performed": False,
        "production_default": "MH-20260812-R5-001",
        "production_default_changed": False,
        "v3_4_changed": False,
    }
    budget = {
        "sleep_history": [
            "r9_joint_replay",
            "r10_unconditioned_residual",
            "prior_state_conditioned_residual",
            "four_state_causal_sleep_delta_conditional",
        ],
        "sleep_history_max": 4,
        "activity_sleep_max": 3,
        "activity_sleep_authorized_only_if_new_runtime_equivalent_information": True,
        "calibration_families": ["platt", "beta"],
        "post_lock_additions": False,
        "stop_rules": {
            "activity_sleep_first_batch_delta_auprc": 0.001,
            "sleep_history_overall_delta_auprc": 0.0004,
            "sleep_history_new_onset_delta_auprc": 0.003,
        },
    }
    time_contract = {
        "assessment_time_before_target_required": True,
        "known_at_at_or_before_inference_cutoff_required": True,
        "same_visit_excluded": True,
        "bands": {
            "14_75": "rule-or-research-only",
            "76_100": "validated-probability-path",
            "101_180": "background-only",
            "over_180": "no-vote",
        },
    }
    denylist = {
        "forbidden": [
            "current-target-future-PHQ-or-derived-label",
            "same-visit-active-questionnaire",
            "future-behavior-or-post-cutoff-record",
            "participant-dataset-source-route-mask-coverage-path-as-risk-input",
            "cross-modal-imputation-or-virtual-participant",
            "facial-negative-as-PHQ-probability",
        ],
        "allowed_probability_features": {
            "activity": list(ACTIVITY_FEATURES),
            "sleep": list(SLEEP_FEATURES),
            "history": list(R9_HISTORY_FEATURES),
            "activity_sleep_existing": list(ACTIVITY_SLEEP_INTERACTION_FEATURES),
            "sleep_history": list(SLEEP_HISTORY_FEATURES),
        },
        "quality_mask_coverage_use": "reliability-routing-explanation-only",
    }
    signature = {
        "activity_sleep": {**_summary(activity_sleep), "conditional_training": True},
        "sleep_history": {**_summary(sleep_history), "probability_band_days": [76, 100]},
        "same_person_same_target_required": True,
        "cross_dataset_stacking": False,
    }
    split_manifest = {
        "seeds": list(R11_REPEAT_SEEDS),
        "participant_grouped": True,
        "outer_folds": R11_OUTER_FOLDS,
        "inner_folds": R11_INNER_FOLDS,
        "r10_outer_reopened": False,
        "activity_sleep_sha256": sha256_file(outputs["activity_sleep"]),
        "sleep_history_sha256": sha256_file(outputs["sleep_history"]),
    }
    for name, value in (
        ("data_exposure_ledger.json", exposure),
        ("lineage_lock.json", lineage),
        ("candidate_budget.json", budget),
        ("history_time_contract.json", time_contract),
        ("risk_feature_denylist.json", denylist),
        ("paired_signature_manifest.json", signature),
        ("split_manifest.json", split_manifest),
        ("environment.json", _environment(root)),
        ("failure_ledger.json", {"failures": [], "skips": [], "retry_policy": "inner-only-before-lock; never reopen R10 outer"}),
    ):
        write_json(report / name, value, overwrite=overwrite)
    selection_contract = {
        "selection_scope": "R11-outer-train-inner-oof-only",
        "r10_outer_reopen_allowed": False,
        "r11_outer_unopened": True,
        "lock_status": "awaiting-runtime-feature-audit",
        "candidate_budget_sha256": canonical_sha(budget),
        "split_manifest_sha256": sha256_file(report / "split_manifest.json"),
        "lineage_lock_sha256": sha256_file(report / "lineage_lock.json"),
    }
    write_json(report / "selection_lock_contract.json", selection_contract, overwrite=overwrite)
    manifest = {
        "protocol_version": R11_PROTOCOL_VERSION,
        "status": "prepared-awaiting-runtime-feature-audit",
        "frozen_document_sha256": actual,
        "config_sha256": sha256_file(config),
        "r10_outer_reopened": False,
        "r11_candidate_outer_opened": False,
        "selection_locked": False,
        "tasks_completed": ["OPT-V333-R11-000"],
        "production_default_changed": False,
        "network_download_performed": False,
    }
    write_json(data / "protocol/r11_protocol_manifest.json", manifest, overwrite=overwrite)
    return manifest


__all__ = [
    "DEFAULT_CONFIG_RELATIVE",
    "DEFAULT_DATA_RELATIVE",
    "DEFAULT_REPORT_RELATIVE",
    "FROZEN_DOCUMENT_RELATIVE",
    "R11_BOOTSTRAP_RESAMPLES",
    "R11_BOOTSTRAP_SEED",
    "R11_FROZEN_DOCUMENT_SHA256",
    "R11_PROTOCOL_VERSION",
    "R11_REPEAT_SEEDS",
    "canonical_sha",
    "prepare_r11_protocol",
    "repository_root",
    "sha256_file",
    "write_json",
]

