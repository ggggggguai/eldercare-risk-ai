"""Frozen population, split, feature and budget contract for R6-000."""

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

from elderly_monitoring.modules.mental_health.mood_social.r3.data import (
    assert_participant_isolation,
)
from elderly_monitoring.modules.mental_health.mood_social.r4.data import (
    load_r4_development_frame,
)
from elderly_monitoring.modules.mental_health.mood_social.r5.contract import (
    R5_ALLOWED_FEATURES,
    R5_LABEL_TASKS,
    R5_PSYCHE_RESEARCH_SENSOR_FEATURES,
    build_split_assignment as build_r5_style_split_assignment,
    forbidden_r5_feature_reason,
)


R6_PROTOCOL_VERSION = "mood-social-v3.3.3-r6"
R6_FROZEN_DOCUMENT_SHA256 = (
    "94B977D6A12190D36260973F4EF11CE948E23AC3BFDDDDF3F44F17D00BBC0068"
)
R6_DEVELOPMENT_SEEDS = (20260831, 20260901, 20260902)
R6_CONFIRMATION_SEED = 20260905
R6_ALL_SEEDS = (*R6_DEVELOPMENT_SEEDS, R6_CONFIRMATION_SEED)
R6_LABEL_TASKS = dict(R5_LABEL_TASKS)
R6_ALLOWED_FEATURES = frozenset(R5_ALLOWED_FEATURES)
R6_PSYCHE_RESEARCH_SENSOR_FEATURES = tuple(R5_PSYCHE_RESEARCH_SENSOR_FEATURES)
R6_CANDIDATE_FAMILY_BUDGET = {
    "elasticnet_spline": 8,
    "lightgbm": 12,
    "catboost": 8,
    "hist_gradient_boosting": 4,
    "extra_trees": 4,
    "causal_deepsets": 6,
    "shrinkage_route_experts": 8,
}
R6_FIRST_ROUND_MAXIMUM = 46
R6_FULL_INNER_MAXIMUM = 10
R6_REPEAT_FINALIST_MAXIMUM = 6

DEFAULT_DATA_RELATIVE = Path(
    "data/processed/mental_health/mood_social/v3.3.3-r6/protocol"
)
DEFAULT_REPORT_RELATIVE = Path(
    "reports/mental_health/mood_social/v3.3.3-r6/OPT-V333-R6-000"
)
DEFAULT_CONFIG_RELATIVE = Path("configs/training/mood_social_v3_3_3_r6.yaml")
FROZEN_DOCUMENT_RELATIVE = Path(
    "项目文档/开发协作/情绪与社交关注模块V3.3.3优化开发/"
    "00-V3.3.3-r6优化方案冻结说明.md"
)
R5_POPULATION_RELATIVE = Path(
    "data/processed/mental_health/mood_social/v3.3.3-r5/protocol/"
    "r5_dual_label_population.parquet"
)


def repository_root(repository_root: Path | None = None) -> Path:
    if repository_root is not None:
        return Path(repository_root).resolve()
    return Path(__file__).resolve().parents[7]


def project_root(root: Path) -> Path:
    candidate = root.parents[1]
    if not (candidate / "项目文档").is_dir():
        raise FileNotFoundError("cannot locate project documentation root")
    return candidate


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_hash(root: Path, relative_paths: Iterable[Path]) -> tuple[str, list[dict[str, str]]]:
    rows: list[dict[str, str]] = []
    digest = hashlib.sha256()
    for relative in sorted({Path(value) for value in relative_paths}, key=lambda value: value.as_posix()):
        path = root / relative
        value = sha256_file(path)
        normalized = relative.as_posix()
        rows.append({"path": normalized, "sha256": value})
        digest.update(normalized.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest(), rows


def write_json(path: Path, value: Any, *, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite r6 artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def forbidden_r6_feature_reason(name: str) -> str | None:
    normalized = str(name).strip().lower()
    if normalized in {"window_count", "participant_window_count"}:
        return "repeated_measurement_pattern_shortcut"
    reason = forbidden_r5_feature_reason(normalized)
    if reason is not None:
        return reason
    if normalized.startswith(("r6_context.", "r6_stable.", "r6_interaction.")):
        return None
    return None


def assert_r6_research_feature_names(feature_names: Iterable[str]) -> tuple[str, ...]:
    names = tuple(str(name) for name in feature_names)
    if not names:
        raise ValueError("an r6 research model must have at least one feature")
    if len(names) != len(set(names)):
        raise ValueError("r6 research feature names contain duplicates")
    base = set(R6_ALLOWED_FEATURES) | set(R6_PSYCHE_RESEARCH_SENSOR_FEATURES)
    generated = ("r5_causal.", "psyche_causal.", "r6_context.", "r6_stable.", "r6_interaction.")
    unknown = sorted(name for name in names if name not in base and not name.startswith(generated))
    forbidden = {
        name: reason
        for name in names
        if name not in R6_PSYCHE_RESEARCH_SENSOR_FEATURES
        and not name.startswith(generated)
        and (reason := forbidden_r6_feature_reason(name)) is not None
    }
    if unknown or forbidden:
        raise ValueError(
            "r6 research feature contract violation: "
            + json.dumps({"unknown": unknown, "forbidden": forbidden}, ensure_ascii=False, sort_keys=True)
        )
    return names


def build_split_assignment(frame: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    if int(seed) not in R6_ALL_SEEDS:
        raise ValueError(f"unregistered r6 split seed: {seed}")
    split = build_r5_style_split_assignment(frame, seed=int(seed)).copy()
    split["r6_row_id"] = split["r4_row_id"].astype(str)
    split["r5_row_id"] = split["r4_row_id"].astype(str)
    ordered = [
        "r6_row_id",
        "r5_row_id",
        "r4_row_id",
        "dataset_id",
        "global_participant_id",
        "route_pattern",
        "phq9_ge5_r3_target",
        "phq9_ge10_r3_target",
        "split_seed",
        "outer_fold",
        "inner_validation_fold_by_outer_fold",
    ]
    output = split[ordered]
    if output["r6_row_id"].duplicated().any():
        raise ValueError("r6 split has duplicate row identifiers")
    if output.groupby("global_participant_id")["outer_fold"].nunique().max() != 1:
        raise ValueError("participant appears in multiple r6 outer folds")
    return output


def _label_summary(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    target = frame[column].astype(int)
    return {
        "rows": int(len(frame)),
        "participants": int(frame["global_participant_id"].nunique()),
        "positive_rows": int(target.sum()),
        "prevalence": float(target.mean()),
    }


def _environment() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for package in ("numpy", "pandas", "scikit-learn", "pyarrow", "catboost", "lightgbm", "torch"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
    }


def _git_state(root: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()

    return {
        "head": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(run("status", "--porcelain")),
    }


def build_r6_protocol_artifacts(
    *,
    repository_root_value: Path | None = None,
    output_directory: Path | None = None,
    report_directory: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Freeze the r6 reused population and all preregistered split assignments."""

    root = repository_root(repository_root_value)
    frozen_document = project_root(root) / FROZEN_DOCUMENT_RELATIVE
    frozen_hash = sha256_file(frozen_document).upper()
    if frozen_hash != R6_FROZEN_DOCUMENT_SHA256:
        raise ValueError(
            f"r6 frozen document hash drift: expected {R6_FROZEN_DOCUMENT_SHA256}, got {frozen_hash}"
        )
    config_path = root / DEFAULT_CONFIG_RELATIVE
    if not config_path.is_file():
        raise FileNotFoundError(f"missing frozen r6 config: {config_path}")
    base = load_r4_development_frame(repository_root=root).copy()
    assert_participant_isolation(base)
    if len(base) != 22070 or base["global_participant_id"].nunique() != 15327:
        raise ValueError("r6 scoring population size drifted")
    if base["dataset_id"].eq("hefei_elderly").any():
        raise ValueError("Hefei must remain label/auxiliary-only in r6")
    base["r6_row_id"] = base["r4_row_id"].astype(str)
    base["r5_row_id"] = base["r4_row_id"].astype(str)

    output = Path(output_directory) if output_directory else root / DEFAULT_DATA_RELATIVE
    report = Path(report_directory) if report_directory else root / DEFAULT_REPORT_RELATIVE
    if not output.is_absolute():
        output = root / output
    if not report.is_absolute():
        report = root / report
    population_path = output / "r6_dual_label_population.parquet"
    manifest_path = output / "r6_protocol_manifest.json"
    exposure_path = report / "data_exposure_ledger.json"
    denylist_path = report / "risk_feature_denylist.json"
    environment_path = report / "environment.json"
    for path in (population_path, manifest_path, exposure_path, denylist_path, environment_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite r6 protocol artifact: {path}")

    population_columns = [
        "r6_row_id", "r5_row_id", "r4_row_id", "r3_row_id", "dataset_id",
        "global_participant_id", "route_pattern", "nominal_month",
        "phq9_score_r3_target", "phq9_ge5_r3_target", "phq9_ge10_r3_target",
    ]
    population_path.parent.mkdir(parents=True, exist_ok=True)
    base[population_columns].to_parquet(population_path, index=False)
    split_artifacts: dict[str, Any] = {}
    for seed in R6_ALL_SEEDS:
        assignment = build_split_assignment(base, seed=seed)
        split_path = output / "splits" / f"seed-{seed}.parquet"
        if split_path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite r6 split: {split_path}")
        split_path.parent.mkdir(parents=True, exist_ok=True)
        assignment.to_parquet(split_path, index=False)
        split_artifacts[str(seed)] = {
            "path": split_path.relative_to(root).as_posix(),
            "sha256": sha256_file(split_path),
            "role": "development" if seed in R6_DEVELOPMENT_SEEDS else "sealed_confirmation",
            "candidate_metrics_opened": False,
            "folds": {
                str(fold): {
                    "rows": int(len(part)),
                    "participants": int(part["global_participant_id"].nunique()),
                    "ge5_positive": int(part["phq9_ge5_r3_target"].sum()),
                    "ge10_positive": int(part["phq9_ge10_r3_target"].sum()),
                }
                for fold, part in assignment.groupby("outer_fold", sort=True)
            },
        }

    core_paths = [
        Path("src/elderly_monitoring/modules/mental_health/mood_social/r6/__init__.py"),
        Path("src/elderly_monitoring/modules/mental_health/mood_social/r6/contract.py"),
        Path("src/elderly_monitoring/modules/mental_health/mood_social/r6/data.py"),
        Path("src/elderly_monitoring/modules/mental_health/mood_social/r6/baseline.py"),
        Path("src/elderly_monitoring/modules/mental_health/mood_social/r6/confirmation.py"),
        DEFAULT_CONFIG_RELATIVE,
    ]
    code_sha, code_files = tree_hash(root, core_paths)
    environment = _environment()
    write_json(environment_path, environment, overwrite=overwrite)
    write_json(
        denylist_path,
        {
            "protocol_version": R6_PROTOCOL_VERSION,
            "historical_phq": "V3.4-only",
            "current_phq": "target/loss-only",
            "concurrent_questionnaires": "forbidden",
            "future_data": "forbidden",
            "source_route_mask_coverage": "routing/confidence/abstention-only",
            "nominal_month": "ordering-only; forbidden risk input",
            "participant_window_count": "audit-only; forbidden risk input",
        },
        overwrite=overwrite,
    )
    exposure = {
        "protocol_version": R6_PROTOCOL_VERSION,
        "all_15327_participants_previously_exposed": True,
        "scorable_rows": int(len(base)),
        "participants": int(base["global_participant_id"].nunique()),
        "development_evidence": "adaptive-development",
        "confirmation_evidence": "sealed reused-cohort confirmation",
        "independent_blind_confirmation": False,
        "real_device_data_available": False,
        "maximum_release_state": "integration-ready/shadow-blocked",
        "local_data_only": True,
        "network_dataset_or_dependency_download": False,
        "r2_r3_r4_r5_artifacts_mutated": False,
        "r5_confirmation_seed_20260829_forbidden_for_r6_selection": True,
        "r6_confirmation_seed": R6_CONFIRMATION_SEED,
        "r6_confirmation_candidate_metrics_opened": False,
    }
    write_json(exposure_path, exposure, overwrite=overwrite)
    labels = {task: _label_summary(base, column) for task, column in R6_LABEL_TASKS.items()}
    manifest = {
        "protocol_version": R6_PROTOCOL_VERSION,
        "status": "pass",
        "evidence_level": "adaptive-development/sealed reused-cohort confirmation",
        "frozen_document": FROZEN_DOCUMENT_RELATIVE.as_posix(),
        "frozen_document_sha256": frozen_hash,
        "config": DEFAULT_CONFIG_RELATIVE.as_posix(),
        "config_sha256": sha256_file(config_path),
        "r5_population_sha256": sha256_file(root / R5_POPULATION_RELATIVE),
        "population": {
            "path": population_path.relative_to(root).as_posix(),
            "sha256": sha256_file(population_path),
            "rows": int(len(base)),
            "participants": int(base["global_participant_id"].nunique()),
        },
        "labels": labels,
        "by_source": {
            str(source): {
                "rows": int(len(part)),
                "participants": int(part["global_participant_id"].nunique()),
                "labels": {task: _label_summary(part, column) for task, column in R6_LABEL_TASKS.items()},
            }
            for source, part in base.groupby("dataset_id", sort=True)
        },
        "development_seeds": list(R6_DEVELOPMENT_SEEDS),
        "confirmation_seed": R6_CONFIRMATION_SEED,
        "r5_confirmation_seed_forbidden": 20260829,
        "splits": split_artifacts,
        "candidate_family_budget": R6_CANDIDATE_FAMILY_BUDGET,
        "first_round_maximum": R6_FIRST_ROUND_MAXIMUM,
        "full_inner_maximum": R6_FULL_INNER_MAXIMUM,
        "repeat_finalist_maximum_per_track": R6_REPEAT_FINALIST_MAXIMUM,
        "protocol_core_tree_sha256": code_sha,
        "protocol_core_files": code_files,
        "git": _git_state(root),
        "environment": {"path": environment_path.relative_to(root).as_posix(), "sha256": sha256_file(environment_path)},
        "denylist": {"path": denylist_path.relative_to(root).as_posix(), "sha256": sha256_file(denylist_path)},
        "exposure_ledger": {"path": exposure_path.relative_to(root).as_posix(), "sha256": sha256_file(exposure_path)},
        "hefei_policy": "label/auxiliary-only; excluded from passive-risk AUPRC",
        "confirmation_candidate_metrics_opened": False,
    }
    write_json(manifest_path, manifest, overwrite=overwrite)
    return manifest


__all__ = [
    "DEFAULT_CONFIG_RELATIVE", "DEFAULT_DATA_RELATIVE", "DEFAULT_REPORT_RELATIVE",
    "R6_ALL_SEEDS", "R6_ALLOWED_FEATURES", "R6_CANDIDATE_FAMILY_BUDGET",
    "R6_CONFIRMATION_SEED", "R6_DEVELOPMENT_SEEDS", "R6_FROZEN_DOCUMENT_SHA256",
    "R6_LABEL_TASKS", "R6_PROTOCOL_VERSION", "R6_PSYCHE_RESEARCH_SENSOR_FEATURES",
    "assert_r6_research_feature_names", "build_r6_protocol_artifacts",
    "build_split_assignment", "forbidden_r6_feature_reason", "repository_root",
    "sha256_file", "tree_hash", "write_json",
]
