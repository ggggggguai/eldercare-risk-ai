"""Frozen data, split, exposure and leakage contract for R7-000."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from elderly_monitoring.modules.mental_health.mood_social.r4.data import (
    load_r4_development_frame,
)
from elderly_monitoring.modules.mental_health.mood_social.r5.contract import (
    build_split_assignment as build_existing_split_assignment,
)
from elderly_monitoring.modules.mental_health.mood_social.r7.deprest import (
    SOCIAL_MAIN_FEATURES,
    SOCIAL_SENSITIVITY_FEATURES,
    build_calls_only_canonical,
    load_call_events,
    load_surveys,
    locate_deprest_root,
)


R7_PROTOCOL_VERSION = "mood-social-v3.3.3-r7"
R7_FROZEN_DOCUMENT_SHA256 = "E80C3AC90F021AE4AEEDEA14CFEE26B141FC689BF71CEA29A86232C9DE565274"
R7_EXISTING_EXPERT_SEEDS = (20260813, 20260814, 20260815)
R7_DEPREST_CONFIRMATION_SEED = 20260816
R7_DEPREST_NESTED_SEED = 20260817
R7_BOOTSTRAP_SEED = 20260818
R7_BOOTSTRAP_RESAMPLES = 2000
R7_CONFIRMATION_FRACTION = 0.20
R7_OUTER_FOLDS = 5
R7_INNER_FOLDS = 5

DEFAULT_DATA_RELATIVE = Path("data/processed/mental_health/mood_social/v3.3.3-r7")
DEFAULT_REPORT_RELATIVE = Path("reports/mental_health/mood_social/v3.3.3-r7/OPT-V333-R7-000")
DEFAULT_CONFIG_RELATIVE = Path("configs/training/mood_social_v3_3_3_r7.yaml")
FROZEN_DOCUMENT_RELATIVE = Path(
    "项目文档/开发协作/情绪与社交关注模块V3.3.3优化开发/"
    "00-V3.3.3-r7优化方案冻结说明.md"
)


def repository_root(value: Path | None = None) -> Path:
    return Path(value).resolve() if value is not None else Path(__file__).resolve().parents[6]


def project_root(root: Path) -> Path:
    result = Path(root).resolve().parents[1]
    if not (result / "项目文档").is_dir():
        raise FileNotFoundError("cannot locate project documentation root")
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any, *, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite frozen R7 artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def tree_manifest(root: Path) -> tuple[str, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    for path in sorted((p for p in Path(root).rglob("*") if p.is_file()), key=lambda p: p.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        value = sha256_file(path)
        rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": value})
        digest.update(relative.encode("utf-8") + b"\0" + value.encode("ascii") + b"\n")
    return digest.hexdigest(), rows


def _strata(frame: pd.DataFrame) -> pd.Series:
    age = frame["age_group"].astype(str).replace({"18-23": "18-39", "24-39": "18-39", "56+": "40+", "unknown": "any"})
    age = age.replace({"40-55": "40+"})
    labels = frame["phq9_ge5_target"].astype(str) + frame["phq9_ge10_target"].astype(str)
    strata = age + "|" + labels
    counts = strata.value_counts()
    rare = strata.map(counts).lt(2)
    for index in strata.index[rare]:
        label = labels.loc[index]
        compatible = strata.loc[labels.eq(label) & ~rare]
        if compatible.empty:
            raise ValueError(f"cannot stratify rare DepreST-CAT label combination {label}")
        strata.loc[index] = compatible.value_counts().index[0]
    if strata.value_counts().min() < 2:
        raise ValueError("DepreST-CAT stratification contains a singleton after collapse")
    return strata


def build_deprest_split(canonical: pd.DataFrame) -> pd.DataFrame:
    ordered = canonical.sort_values("participant_id", kind="stable").reset_index(drop=True)
    strata = _strata(ordered)
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=R7_CONFIRMATION_FRACTION,
        random_state=R7_DEPREST_CONFIRMATION_SEED,
    )
    development_index, confirmation_index = next(splitter.split(ordered, strata))
    assignment = ordered[["participant_id", "age_group", "phq9_ge5_target", "phq9_ge10_target"]].copy()
    assignment["partition"] = "development"
    assignment.loc[confirmation_index, "partition"] = "model_unseen_confirmation"
    assignment["outer_fold"] = pd.Series(pd.NA, index=assignment.index, dtype="Int64")
    inner_maps: list[dict[str, int]] = [dict() for _ in range(len(assignment))]
    development = assignment.iloc[development_index].copy()
    development_strata = _strata(development)
    outer = StratifiedKFold(n_splits=R7_OUTER_FOLDS, shuffle=True, random_state=R7_DEPREST_NESTED_SEED)
    for fold, (_, validation_position) in enumerate(outer.split(development, development_strata)):
        assignment.loc[development.index[validation_position], "outer_fold"] = fold
    for outer_fold in range(R7_OUTER_FOLDS):
        inner_indices = assignment.index[
            assignment["partition"].eq("development")
            & assignment["outer_fold"].ne(outer_fold)
        ]
        inner_frame = assignment.loc[inner_indices]
        inner_strata = _strata(inner_frame)
        splitter = StratifiedKFold(
            n_splits=R7_INNER_FOLDS,
            shuffle=True,
            random_state=R7_DEPREST_NESTED_SEED + 1000 + outer_fold,
        )
        for inner_fold, (_, validation_positions) in enumerate(splitter.split(inner_frame, inner_strata)):
            for index in inner_frame.index[validation_positions]:
                inner_maps[index][str(outer_fold)] = int(inner_fold)
    assignment["inner_validation_fold_by_outer_fold"] = [
        json.dumps(value, sort_keys=True, separators=(",", ":")) for value in inner_maps
    ]
    if assignment.loc[assignment["partition"].eq("development"), "outer_fold"].isna().any():
        raise ValueError("DepreST-CAT development outer folds are incomplete")
    if assignment.loc[assignment["partition"].eq("model_unseen_confirmation"), "outer_fold"].notna().any():
        raise ValueError("DepreST-CAT confirmation rows must not receive development folds")
    return assignment.sort_values("participant_id", kind="stable").reset_index(drop=True)


def _environment() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in ("numpy", "pandas", "scikit-learn", "pyarrow", "catboost", "lightgbm"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
    }


def _git_state(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
    return {"head": run("rev-parse", "HEAD"), "branch": run("branch", "--show-current"), "dirty": bool(run("status", "--porcelain"))}


def _partition_summary(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(frame)),
        "participants": int(frame["participant_id"].nunique()),
        "phq9_ge5_positive": int(frame["phq9_ge5_target"].sum()),
        "phq9_ge10_positive": int(frame["phq9_ge10_target"].sum()),
        "phq9_ge5_prevalence": float(frame["phq9_ge5_target"].mean()),
        "phq9_ge10_prevalence": float(frame["phq9_ge10_target"].mean()),
        "age_groups": {str(key): int(value) for key, value in frame["age_group"].value_counts(dropna=False).sort_index().items()},
    }


def build_r7_protocol_artifacts(
    *,
    repository_root_value: Path | None = None,
    output_directory: Path | None = None,
    report_directory: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Complete R7-000/001 without opening confirmation metrics or searching models."""

    root = repository_root(repository_root_value)
    project = project_root(root)
    frozen = project / FROZEN_DOCUMENT_RELATIVE
    actual_frozen_sha = sha256_file(frozen).upper()
    if actual_frozen_sha != R7_FROZEN_DOCUMENT_SHA256:
        raise ValueError(f"R7 frozen document hash drift: {actual_frozen_sha}")
    config = root / DEFAULT_CONFIG_RELATIVE
    if not config.is_file():
        raise FileNotFoundError(config)
    output = (Path(output_directory) if output_directory else root / DEFAULT_DATA_RELATIVE).resolve()
    report = (Path(report_directory) if report_directory else root / DEFAULT_REPORT_RELATIVE).resolve()
    expected = [
        output / "social/deprest_calls_only_canonical.parquet",
        output / "social/development.parquet",
        output / "social/model_unseen_confirmation.parquet",
        output / "protocol/deprest_split.parquet",
        output / "protocol/r7_protocol_manifest.json",
        report / "deprest_time_field_audit.json",
        report / "deprest_license_manifest.json",
        report / "data_exposure_ledger.json",
        report / "risk_feature_denylist.json",
        report / "environment.json",
        report / "social_confirmation_seal.json",
        *(output / "protocol/existing_splits" / f"seed-{seed}.parquet" for seed in R7_EXISTING_EXPERT_SEEDS),
    ]
    if not overwrite:
        existing = [str(path) for path in expected if path.exists()]
        if existing:
            raise FileExistsError(f"refusing to overwrite R7 protocol artifacts: {existing}")

    paths = locate_deprest_root(project)
    raw_tree_sha, raw_files = tree_manifest(paths.root)
    surveys = load_surveys(paths.survey)
    calls = load_call_events(paths.calls)
    canonical, time_audit = build_calls_only_canonical(surveys, calls)
    split = build_deprest_split(canonical)
    joined = canonical.merge(
        split[[
            "participant_id", "partition", "outer_fold",
            "inner_validation_fold_by_outer_fold",
        ]],
        on="participant_id", how="left", validate="one_to_one",
    )
    joined["global_participant_id"] = joined["participant_id"].astype(str)
    development = joined.loc[joined["partition"].eq("development")].copy()
    confirmation = joined.loc[joined["partition"].eq("model_unseen_confirmation")].copy()
    if len(development) + len(confirmation) != 369 or len(confirmation) != 74:
        raise ValueError("DepreST-CAT frozen split size drifted")

    for path, frame in (
        (output / "social/deprest_calls_only_canonical.parquet", canonical),
        (output / "social/development.parquet", development),
        (output / "social/model_unseen_confirmation.parquet", confirmation),
        (output / "protocol/deprest_split.parquet", split),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not overwrite:
            raise FileExistsError(path)
        frame.to_parquet(path, index=False)

    existing = load_r4_development_frame(repository_root=root)
    existing_splits: dict[str, Any] = {}
    for seed in R7_EXISTING_EXPERT_SEEDS:
        assignment = build_existing_split_assignment(existing, seed=seed).copy()
        path = output / "protocol/existing_splits" / f"seed-{seed}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        assignment.to_parquet(path, index=False)
        existing_splits[str(seed)] = {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
            "participants_cross_outer_folds": int(
                assignment.groupby("global_participant_id")["outer_fold"].nunique().gt(1).sum()
            ),
        }
    existing_by_source = {
        str(source): {"rows": int(len(part)), "participants": int(part["global_participant_id"].nunique())}
        for source, part in existing.groupby("dataset_id", sort=True)
    }
    environment_path = report / "environment.json"
    write_json(environment_path, _environment(), overwrite=overwrite)
    write_json(report / "deprest_time_field_audit.json", time_audit, overwrite=overwrite)
    license_manifest = {
        "dataset": "DepreST-CAT",
        "dataset_root_tree_sha256": raw_tree_sha,
        "files": raw_files,
        "readme_sha256": sha256_file(paths.readme),
        "license_statement": "academic use under CC BY-NC-SA",
        "citation_required": True,
        "commercial_or_default_model_release": "blocked_pending_legal_review",
        "raw_data_tracked_by_git": False,
        "derived_models_may_inherit_share_alike_constraints": "requires_legal_review",
    }
    write_json(report / "deprest_license_manifest.json", license_manifest, overwrite=overwrite)
    denylist = {
        "protocol_version": R7_PROTOCOL_VERSION,
        "sensor_experts": [
            "current/target PHQ total, items, severity and derived labels",
            "concurrent GAD/ISI/ULS/CES-D/GDS or other symptom questionnaires",
            "future behavior, future assessment or post-target event",
            "participant/dataset/source/file identifiers",
            "route/mask/coverage/fixed missingness as learned risk features",
            "appVersion/Group/control/stereotype threat/COVID/Remote/treatment history",
            "contact hash, absolute timestamp or calendar date",
        ],
        "historical_phq": "PHQHistoryExpert only; never activity/sleep/social/profile",
        "history_attention": "forbidden current-state risk input",
        "cross_modal_imputation": "forbidden",
        "social_main_features": list(SOCIAL_MAIN_FEATURES),
        "social_sensitivity_features": list(SOCIAL_SENSITIVITY_FEATURES),
    }
    write_json(report / "risk_feature_denylist.json", denylist, overwrite=overwrite)
    exposure = {
        "protocol_version": R7_PROTOCOL_VERSION,
        "existing_five_source_population": {"rows": int(len(existing)), "participants": int(existing["global_participant_id"].nunique()), "status": "adaptive-development/reused-benchmark", "by_source": existing_by_source},
        "deprest_cat": {"participants": 369, "summary_labels_already_read": True, "development": _partition_summary(development), "confirmation": {"rows": int(len(confirmation)), "participants": int(confirmation["participant_id"].nunique()), "model_metrics_opened": False, "evidence": "model-unseen confirmation; not fully label-blind"}},
        "new_elderly_subjects_available": False,
        "real_device_data_available": False,
        "network_dataset_or_dependency_download": False,
        "maximum_release_state": "offline-validated/integration-ready/device-validation-pending",
    }
    write_json(report / "data_exposure_ledger.json", exposure, overwrite=overwrite)
    seal = {
        "protocol_version": R7_PROTOCOL_VERSION,
        "confirmation_seed": R7_DEPREST_CONFIRMATION_SEED,
        "nested_seed": R7_DEPREST_NESTED_SEED,
        "confirmation_rows": int(len(confirmation)),
        "confirmation_id_sha256": hashlib.sha256("\n".join(sorted(confirmation["participant_id"].astype(str))).encode("utf-8")).hexdigest(),
        "confirmation_parquet_sha256": sha256_file(output / "social/model_unseen_confirmation.parquet"),
        "candidate_recipe_locked": False,
        "model_metrics_opened": False,
        "reopen_allowed": False,
    }
    write_json(report / "social_confirmation_seal.json", seal, overwrite=overwrite)
    manifest = {
        "protocol_version": R7_PROTOCOL_VERSION,
        "status": "pass",
        "tasks_completed": ["OPT-V333-R7-000", "OPT-V333-R7-001"],
        "frozen_document_sha256": actual_frozen_sha,
        "config_sha256": sha256_file(config),
        "deprest_raw_tree_sha256": raw_tree_sha,
        "canonical": {"path": (output / "social/deprest_calls_only_canonical.parquet").relative_to(root).as_posix(), "sha256": sha256_file(output / "social/deprest_calls_only_canonical.parquet"), "summary": _partition_summary(canonical)},
        "development": {"path": (output / "social/development.parquet").relative_to(root).as_posix(), "sha256": sha256_file(output / "social/development.parquet"), "summary": _partition_summary(development)},
        "confirmation": {"path": (output / "social/model_unseen_confirmation.parquet").relative_to(root).as_posix(), "sha256": sha256_file(output / "social/model_unseen_confirmation.parquet"), "metrics_opened": False},
        "split": {"path": (output / "protocol/deprest_split.parquet").relative_to(root).as_posix(), "sha256": sha256_file(output / "protocol/deprest_split.parquet"), "outer_folds": R7_OUTER_FOLDS, "inner_folds": R7_INNER_FOLDS},
        "existing_expert_splits": existing_splits,
        "seeds": {"existing_experts": list(R7_EXISTING_EXPERT_SEEDS), "deprest_confirmation": R7_DEPREST_CONFIRMATION_SEED, "deprest_nested": R7_DEPREST_NESTED_SEED, "bootstrap": R7_BOOTSTRAP_SEED},
        "git": _git_state(root),
        "raw_data_in_git": False,
        "confirmation_candidate_metrics_opened": False,
    }
    manifest_path = output / "protocol/r7_protocol_manifest.json"
    write_json(manifest_path, manifest, overwrite=overwrite)
    return manifest


__all__ = [
    "DEFAULT_CONFIG_RELATIVE", "DEFAULT_DATA_RELATIVE", "DEFAULT_REPORT_RELATIVE",
    "R7_BOOTSTRAP_RESAMPLES", "R7_BOOTSTRAP_SEED", "R7_DEPREST_CONFIRMATION_SEED",
    "R7_DEPREST_NESTED_SEED", "R7_EXISTING_EXPERT_SEEDS", "R7_FROZEN_DOCUMENT_SHA256",
    "R7_INNER_FOLDS", "R7_OUTER_FOLDS", "R7_PROTOCOL_VERSION", "build_deprest_split",
    "build_r7_protocol_artifacts", "project_root", "repository_root", "sha256_file", "write_json",
]
