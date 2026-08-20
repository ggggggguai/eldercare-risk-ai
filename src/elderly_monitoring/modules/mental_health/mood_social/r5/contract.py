"""Frozen population, split and feature contract for OPT-V333-R5-000."""

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
from sklearn.model_selection import StratifiedGroupKFold

from elderly_monitoring.modules.mental_health.mood_social.r3.contract import (
    R3_SPLIT_ID,
    R3_SPLIT_SHA256,
)
from elderly_monitoring.modules.mental_health.mood_social.r3.data import (
    assert_participant_isolation,
)
from elderly_monitoring.modules.mental_health.mood_social.r4 import contract as r4_contract
from elderly_monitoring.modules.mental_health.mood_social.r4.data import (
    load_r4_development_frame,
)


R5_PROTOCOL_VERSION = "mood-social-v3.3.3-r5"
R5_FROZEN_DOCUMENT_SHA256 = (
    "3AA1EE51D6FFE4658B3BE26926174B175DED9AF2CD2FEBFA2083BEB9E0D61238"
)
R5_DEVELOPMENT_SEEDS = (20260821, 20260822, 20260823)
R5_CONFIRMATION_SEED = 20260829
R5_ALL_SEEDS = (*R5_DEVELOPMENT_SEEDS, R5_CONFIRMATION_SEED)
R5_LABEL_TASKS = dict(r4_contract.R4_LABEL_TASKS)
R5_ALLOWED_FEATURES = frozenset(r4_contract.R4_ALLOWED_FEATURES)
R5_PSYCHE_RESEARCH_SENSOR_FEATURES = tuple(
    r4_contract.PSYCHE_RESEARCH_SENSOR_FEATURES
)
R5_CANDIDATE_FAMILY_BUDGET = {
    "catboost": 12,
    "lightgbm": 12,
    "extra_trees": 6,
    "elasticnet_spline": 6,
    "small_mlp": 4,
}
R5_DEEP_SEQUENCE_MODELS_ENABLED = False

DEFAULT_DATA_RELATIVE = Path(
    "data/processed/mental_health/mood_social/v3.3.3-r5/protocol"
)
DEFAULT_REPORT_RELATIVE = Path(
    "reports/mental_health/mood_social/v3.3.3-r5/OPT-V333-R5-000"
)
DEFAULT_CONFIG_RELATIVE = Path("configs/training/mood_social_v3_3_3_r5.yaml")
FROZEN_DOCUMENT_RELATIVE = Path(
    "项目文档/开发协作/情绪与社交关注模块V3.3.3优化开发/"
    "00-V3.3.3-r5优化方案冻结说明.md"
)
R4_POPULATION_RELATIVE = Path(
    "data/processed/mental_health/mood_social/v3.3.3-r4/protocol/"
    "r4_dual_label_population.parquet"
)


def _repository_root(repository_root: Path | None) -> Path:
    if repository_root is not None:
        return Path(repository_root).resolve()
    return Path(__file__).resolve().parents[7]


def _project_root(repository_root: Path) -> Path:
    candidate = repository_root.parents[1]
    if not (candidate / "项目文档").is_dir():
        raise FileNotFoundError("cannot locate project documentation root")
    return candidate


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_hash(root: Path, relative_paths: Iterable[Path]) -> tuple[str, list[dict[str, str]]]:
    rows: list[dict[str, str]] = []
    digest = hashlib.sha256()
    for relative in sorted((Path(value) for value in relative_paths), key=lambda value: value.as_posix()):
        path = root / relative
        value = sha256_file(path)
        normalized = relative.as_posix()
        rows.append({"path": normalized, "sha256": value})
        digest.update(normalized.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest(), rows


def _write_json(path: Path, value: Any, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite r5 protocol artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def forbidden_r5_feature_reason(name: str) -> str | None:
    """Return the frozen reason a field cannot enter an r5 risk vector."""

    reason = r4_contract.forbidden_r4_feature_reason(name)
    if reason is not None:
        return reason
    normalized = str(name).strip().lower()
    if normalized.startswith("r5_causal."):
        return None
    if normalized.startswith("psyche_causal."):
        return None
    if normalized in {"nominal_month", "wave", "timepoint"}:
        return "collection_schedule_shortcut"
    return None


def assert_r5_deployable_feature_names(feature_names: Iterable[str]) -> tuple[str, ...]:
    """Enforce runtime-equivalent deployable inputs for r5."""

    return r4_contract.assert_r4_feature_names(feature_names)


def assert_r5_research_feature_names(feature_names: Iterable[str]) -> tuple[str, ...]:
    """Allow exact audited PSYCHE fields plus generated causal features."""

    names = tuple(str(name) for name in feature_names)
    if not names:
        raise ValueError("an r5 research model must have at least one feature")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    base = set(R5_ALLOWED_FEATURES) | set(R5_PSYCHE_RESEARCH_SENSOR_FEATURES)
    unknown = sorted(
        name
        for name in names
        if name not in base
        and not name.startswith(("r5_causal.", "psyche_causal."))
    )
    forbidden = {
        name: reason
        for name in names
        if name not in R5_PSYCHE_RESEARCH_SENSOR_FEATURES
        and not name.startswith(("r5_causal.", "psyche_causal."))
        and (reason := forbidden_r5_feature_reason(name)) is not None
    }
    if duplicates or unknown or forbidden:
        raise ValueError(
            "r5 research feature contract violation: "
            + json.dumps(
                {"duplicates": duplicates, "unknown": unknown, "forbidden": forbidden},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    return names


def _stratification_key(frame: pd.DataFrame) -> pd.Series:
    severity = np.select(
        [
            frame["phq9_ge5_r3_target"].eq(0),
            frame["phq9_ge10_r3_target"].eq(0),
        ],
        ["lt5", "ge5_lt10"],
        default="ge10",
    )
    return frame["dataset_id"].astype(str) + "::" + pd.Series(
        severity, index=frame.index, dtype="object"
    )


def build_split_assignment(frame: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    """Create a deterministic participant-grouped nested 5x5 assignment."""

    required = {
        "r4_row_id",
        "dataset_id",
        "global_participant_id",
        "phq9_ge5_r3_target",
        "phq9_ge10_r3_target",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"r5 split frame missing columns: {missing}")
    if frame["r4_row_id"].duplicated().any():
        raise ValueError("r5 split frame has duplicate row identifiers")
    groups = frame["global_participant_id"].astype(str)
    strata = _stratification_key(frame)
    outer = np.full(len(frame), -1, dtype=int)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=int(seed))
    for fold, (_train, test) in enumerate(splitter.split(frame, strata, groups)):
        outer[test] = fold
    if (outer < 0).any():
        raise ValueError("r5 outer split assignment is incomplete")

    inner_maps: list[dict[str, int]] = [dict() for _ in range(len(frame))]
    for outer_fold in range(5):
        train_mask = outer != outer_fold
        train_positions = np.flatnonzero(train_mask)
        inner_splitter = StratifiedGroupKFold(
            n_splits=5,
            shuffle=True,
            random_state=int(seed) + 1000 + outer_fold,
        )
        for inner_fold, (_inner_train, validation_local) in enumerate(
            inner_splitter.split(
                frame.iloc[train_positions],
                strata.iloc[train_positions],
                groups.iloc[train_positions],
            )
        ):
            for position in train_positions[validation_local]:
                inner_maps[position][str(outer_fold)] = int(inner_fold)
    output = frame[
        [
            "r4_row_id",
            "dataset_id",
            "global_participant_id",
            "route_pattern",
            "phq9_ge5_r3_target",
            "phq9_ge10_r3_target",
        ]
    ].copy()
    output["r5_row_id"] = output["r4_row_id"].astype(str)
    output["split_seed"] = int(seed)
    output["outer_fold"] = outer
    output["inner_validation_fold_by_outer_fold"] = [
        json.dumps(value, sort_keys=True, separators=(",", ":")) for value in inner_maps
    ]
    output = output[
        [
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
    ]
    _assert_split_assignment(output)
    return output


def _extract_inner(value: str, outer_fold: int) -> float:
    parsed = json.loads(value)
    result = parsed.get(str(int(outer_fold)))
    return np.nan if result is None else float(result)


def _assert_split_assignment(frame: pd.DataFrame) -> None:
    if len(frame) == 0 or frame["r5_row_id"].duplicated().any():
        raise ValueError("r5 split must cover unique rows")
    folds_per_participant = frame.groupby("global_participant_id")["outer_fold"].nunique()
    if not folds_per_participant.eq(1).all():
        raise ValueError("participant appears in multiple r5 outer folds")
    for outer_fold in range(5):
        inner = frame["inner_validation_fold_by_outer_fold"].map(
            lambda value: _extract_inner(value, outer_fold)
        )
        is_test = frame["outer_fold"].eq(outer_fold)
        if inner[is_test].notna().any() or inner[~is_test].isna().any():
            raise ValueError(f"r5 inner assignment coverage failed for outer {outer_fold}")
        per_participant = (
            frame.loc[~is_test]
            .assign(_inner=inner.loc[~is_test].astype(int))
            .groupby("global_participant_id")["_inner"]
            .nunique()
        )
        if not per_participant.eq(1).all():
            raise ValueError(f"participant appears in multiple r5 inner folds for outer {outer_fold}")
        for task in ("phq9_ge5_r3_target", "phq9_ge10_r3_target"):
            counts = frame.loc[is_test, task].value_counts()
            if not {0, 1}.issubset(set(int(value) for value in counts.index)):
                raise ValueError(f"r5 outer fold {outer_fold} lacks both classes for {task}")


def _label_summary(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    target = frame[column].astype(int)
    return {
        "rows": int(len(frame)),
        "participants": int(frame["global_participant_id"].nunique()),
        "positive_rows": int(target.sum()),
        "prevalence": float(target.mean()),
    }


def _environment() -> dict[str, Any]:
    packages = {}
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


def build_r5_protocol_artifacts(
    *,
    repository_root: Path | None = None,
    output_directory: Path | None = None,
    report_directory: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Freeze r5's reused population and all pre-registered split assignments."""

    root = _repository_root(repository_root)
    project_root = _project_root(root)
    frozen_document = project_root / FROZEN_DOCUMENT_RELATIVE
    frozen_hash = sha256_file(frozen_document)
    if frozen_hash.upper() != R5_FROZEN_DOCUMENT_SHA256:
        raise ValueError(
            "r5 frozen document hash drift: "
            f"expected {R5_FROZEN_DOCUMENT_SHA256}, got {frozen_hash}"
        )
    config_path = root / DEFAULT_CONFIG_RELATIVE
    if not config_path.is_file():
        raise FileNotFoundError(f"missing frozen r5 config: {config_path}")
    base = load_r4_development_frame(repository_root=root).copy()
    assert_participant_isolation(base)
    base["r5_row_id"] = base["r4_row_id"].astype(str)
    if base["dataset_id"].eq("hefei_elderly").any():
        raise ValueError("Hefei must remain label/auxiliary-only in r5")

    output = Path(output_directory) if output_directory else root / DEFAULT_DATA_RELATIVE
    report = Path(report_directory) if report_directory else root / DEFAULT_REPORT_RELATIVE
    if not output.is_absolute():
        output = root / output
    if not report.is_absolute():
        report = root / report
    population_path = output / "r5_dual_label_population.parquet"
    manifest_path = output / "r5_protocol_manifest.json"
    exposure_path = report / "data_exposure_ledger.json"
    denylist_path = report / "risk_feature_denylist.json"
    environment_path = report / "environment.json"
    for path in (population_path, manifest_path, exposure_path, denylist_path, environment_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite r5 protocol artifact: {path}")

    population = base[
        [
            "r5_row_id",
            "r4_row_id",
            "r3_row_id",
            "dataset_id",
            "global_participant_id",
            "route_pattern",
            "nominal_month",
            "phq9_score_r3_target",
            "phq9_ge5_r3_target",
            "phq9_ge10_r3_target",
        ]
    ].copy()
    population_path.parent.mkdir(parents=True, exist_ok=True)
    population.to_parquet(population_path, index=False)
    split_artifacts: dict[str, Any] = {}
    for seed in R5_ALL_SEEDS:
        assignment = build_split_assignment(base, seed=seed)
        split_path = output / "splits" / f"seed-{seed}.parquet"
        if split_path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite r5 split: {split_path}")
        split_path.parent.mkdir(parents=True, exist_ok=True)
        assignment.to_parquet(split_path, index=False)
        split_artifacts[str(seed)] = {
            "path": split_path.relative_to(root).as_posix(),
            "sha256": sha256_file(split_path),
            "role": "development" if seed in R5_DEVELOPMENT_SEEDS else "sealed_confirmation",
            "predictions_opened": False,
            "folds": {
                str(fold): {
                    "rows": int(part.shape[0]),
                    "participants": int(part["global_participant_id"].nunique()),
                    "ge5_positive": int(part["phq9_ge5_r3_target"].sum()),
                    "ge10_positive": int(part["phq9_ge10_r3_target"].sum()),
                }
                for fold, part in assignment.groupby("outer_fold", sort=True)
            },
        }

    r5_protocol_files = (
        "__init__.py",
        "baseline.py",
        "confirmation.py",
        "contract.py",
        "data.py",
    )
    code_paths = [
        Path("src/elderly_monitoring/modules/mental_health/mood_social/r4") / path.name
        for path in (root / "src/elderly_monitoring/modules/mental_health/mood_social/r4").glob("*.py")
    ] + [
        Path("src/elderly_monitoring/modules/mental_health/mood_social/r5") / name
        for name in r5_protocol_files
    ]
    code_tree_sha, code_files = _tree_hash(root, code_paths)
    labels = {
        task: _label_summary(base, column) for task, column in R5_LABEL_TASKS.items()
    }
    by_source = {
        str(source): {
            "rows": int(len(part)),
            "participants": int(part["global_participant_id"].nunique()),
            "labels": {
                task: _label_summary(part, column)
                for task, column in R5_LABEL_TASKS.items()
            },
        }
        for source, part in base.groupby("dataset_id", sort=True)
    }
    git_state = _git_state(root)
    environment = _environment()
    _write_json(environment_path, environment, overwrite=overwrite)
    _write_json(
        denylist_path,
        {
            "protocol_version": R5_PROTOCOL_VERSION,
            "forbidden_exact": sorted(r4_contract.R4_FORBIDDEN_EXACT),
            "forbidden_prefixes": list(r4_contract.R4_FORBIDDEN_PREFIXES),
            "forbidden_semantic_tokens": list(r4_contract.R4_FORBIDDEN_TOKENS),
            "additional_forbidden": ["nominal_month", "wave", "timepoint"],
            "historical_phq": "V3.4-only",
            "current_phq": "target/loss-only",
            "route_mask_coverage": "routing/confidence/abstention-only",
        },
        overwrite=overwrite,
    )
    exposure = {
        "protocol_version": R5_PROTOCOL_VERSION,
        "r3_r4_population_exposed": True,
        "scorable_rows": int(len(base)),
        "participants": int(base["global_participant_id"].nunique()),
        "current_population_use": "adaptive-development",
        "confirmation_use": "sealed reused-cohort confirmation",
        "independent_blind_confirmation": False,
        "real_device_data_available": False,
        "maximum_release_state": "integration-ready/shadow-blocked",
        "local_data_only": True,
        "network_dataset_or_dependency_download": False,
        "r2_r3_r4_artifacts_mutated": False,
        "confirmation_seed": R5_CONFIRMATION_SEED,
        "confirmation_predictions_opened": False,
    }
    _write_json(exposure_path, exposure, overwrite=overwrite)
    manifest = {
        "protocol_version": R5_PROTOCOL_VERSION,
        "status": "pass",
        "evidence_level": "adaptive-development/reused-cohort confirmation",
        "frozen_document": FROZEN_DOCUMENT_RELATIVE.as_posix(),
        "frozen_document_sha256": frozen_hash,
        "config": DEFAULT_CONFIG_RELATIVE.as_posix(),
        "config_sha256": sha256_file(config_path),
        "inherited_split_id": R3_SPLIT_ID,
        "inherited_split_sha256": R3_SPLIT_SHA256,
        "r4_population_sha256": sha256_file(root / R4_POPULATION_RELATIVE),
        "population": {
            "path": population_path.relative_to(root).as_posix(),
            "sha256": sha256_file(population_path),
            "rows": int(len(base)),
            "participants": int(base["global_participant_id"].nunique()),
        },
        "labels": labels,
        "by_source": by_source,
        "development_seeds": list(R5_DEVELOPMENT_SEEDS),
        "confirmation_seed": R5_CONFIRMATION_SEED,
        "candidate_family_budget": R5_CANDIDATE_FAMILY_BUDGET,
        "finalist_maximum": 6,
        "deep_sequence_models_enabled": R5_DEEP_SEQUENCE_MODELS_ENABLED,
        "selection_scores": {
            "multisource": "0.55*natural_ap + 0.25*participant_ap + 0.20*macro_ap_at_10pct",
            "psyche": "0.65*natural_ap + 0.35*participant_ap",
        },
        "splits": split_artifacts,
        "code_tree_sha256": code_tree_sha,
        "code_tree_scope": "r4 inherited modules plus r5 protocol/split/baseline/confirmation core",
        "code_files": code_files,
        "git": git_state,
        "environment": {
            "path": environment_path.relative_to(root).as_posix(),
            "sha256": sha256_file(environment_path),
        },
        "denylist": {
            "path": denylist_path.relative_to(root).as_posix(),
            "sha256": sha256_file(denylist_path),
        },
        "exposure_ledger": {
            "path": exposure_path.relative_to(root).as_posix(),
            "sha256": sha256_file(exposure_path),
        },
        "hefei_policy": "label/auxiliary-only; excluded from passive-risk AUPRC",
        "historical_phq_policy": "V3.4-only; forbidden from V3.3.3-r5 inputs",
        "confirmation_predictions_opened": False,
    }
    _write_json(manifest_path, manifest, overwrite=overwrite)
    return manifest


__all__ = [
    "DEFAULT_CONFIG_RELATIVE",
    "DEFAULT_DATA_RELATIVE",
    "DEFAULT_REPORT_RELATIVE",
    "R5_ALLOWED_FEATURES",
    "R5_ALL_SEEDS",
    "R5_CONFIRMATION_SEED",
    "R5_CANDIDATE_FAMILY_BUDGET",
    "R5_DEEP_SEQUENCE_MODELS_ENABLED",
    "R5_DEVELOPMENT_SEEDS",
    "R5_FROZEN_DOCUMENT_SHA256",
    "R5_LABEL_TASKS",
    "R5_PROTOCOL_VERSION",
    "R5_PSYCHE_RESEARCH_SENSOR_FEATURES",
    "assert_r5_deployable_feature_names",
    "assert_r5_research_feature_names",
    "build_r5_protocol_artifacts",
    "build_split_assignment",
    "forbidden_r5_feature_reason",
    "sha256_file",
]
