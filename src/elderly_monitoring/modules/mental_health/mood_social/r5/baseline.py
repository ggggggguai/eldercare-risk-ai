"""Replay the frozen r4 recipe on r5 development splits for paired comparison."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from elderly_monitoring.modules.mental_health.mood_social.r4.baseline import (
    ap_context_metrics,
)
from elderly_monitoring.modules.mental_health.mood_social.r4.experiment import (
    _load_or_run,
    _run_deployable_fold,
    _run_psyche_fold,
)
from elderly_monitoring.modules.mental_health.mood_social.r4.selection import (
    enforce_dual_head_monotonicity,
    participant_equal_weights,
)
from elderly_monitoring.modules.mental_health.mood_social.r5.contract import (
    R5_CONFIRMATION_SEED,
    R5_DEVELOPMENT_SEEDS,
    R5_LABEL_TASKS,
    R5_PROTOCOL_VERSION,
    sha256_file,
)
from elderly_monitoring.modules.mental_health.mood_social.r5.data import (
    load_r5_development_frame,
)
from elderly_monitoring.modules.mental_health.mood_social.r5.confirmation import (
    confirmation_state,
)


DEFAULT_REPORT_RELATIVE = Path(
    "reports/mental_health/mood_social/v3.3.3-r5/"
    "OPT-V333-R5-000/r4-paired-baseline"
)


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _metric(frame: pd.DataFrame) -> dict[str, Any]:
    target = frame["binary_target"].to_numpy(int)
    probability = frame["baseline_probability"].to_numpy(float)
    weight = participant_equal_weights(frame)
    return {
        "rows": int(len(frame)),
        "participants": int(frame["global_participant_id"].nunique()),
        "positive_rows": int(target.sum()),
        "coverage": float(np.isfinite(probability).mean()),
        "natural": ap_context_metrics(target, probability),
        "participant_equal": ap_context_metrics(
            target, probability, sample_weight=weight
        ),
    }


def _project_dual_heads(oof: pd.DataFrame) -> pd.DataFrame:
    oof = oof.copy()
    oof["unprojected_baseline_probability"] = oof["candidate_probability"]
    parts: list[pd.DataFrame] = []
    for track, track_frame in oof.groupby("track", sort=True):
        early = track_frame.loc[
            track_frame["r4_task_id"].eq("phq_ge5_current")
        ].sort_values("r4_row_id", kind="stable").copy()
        elevated = track_frame.loc[
            track_frame["r4_task_id"].eq("phq_ge10_current")
        ].sort_values("r4_row_id", kind="stable").copy()
        if not early["r4_row_id"].reset_index(drop=True).equals(
            elevated["r4_row_id"].reset_index(drop=True)
        ):
            raise ValueError(f"r5 r4-replay dual-head rows differ for {track}")
        p5, p10 = enforce_dual_head_monotonicity(
            early["candidate_probability"].to_numpy(float),
            elevated["candidate_probability"].to_numpy(float),
        )
        early["baseline_probability"] = p5
        elevated["baseline_probability"] = p10
        parts.extend([early, elevated])
    result = pd.concat(parts, ignore_index=True)
    if (
        result["baseline_probability"].isna().any()
        or result["r4_row_id"].isna().any()
    ):
        raise ValueError("r5 r4-replay baseline projection is incomplete")
    return result


def run_r4_recipe_baseline(
    *,
    repository_root: Path,
    seed: int,
    report_directory: Path | None = None,
    overwrite: bool = False,
    allow_confirmation: bool = False,
) -> dict[str, Any]:
    """Train the sealed r4 candidate recipe on one r5 split.

    Confirmation predictions are protected until R5-006 explicitly opens the
    one-time queue.
    """

    seed = int(seed)
    if seed not in (*R5_DEVELOPMENT_SEEDS, R5_CONFIRMATION_SEED):
        raise ValueError(f"unregistered r5 baseline seed: {seed}")
    root = Path(repository_root).resolve()
    if seed == R5_CONFIRMATION_SEED and (
        not allow_confirmation or not confirmation_state(root)["opened"]
    ):
        raise PermissionError("r5 confirmation baseline cannot be opened before R5-006")
    base_output = Path(report_directory) if report_directory else root / DEFAULT_REPORT_RELATIVE
    if not base_output.is_absolute():
        base_output = root / base_output
    output = base_output / f"seed-{seed}"
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "_checkpoints"
    base = load_r5_development_frame(repository_root=root, seed=seed)
    psyche = base.loc[base["dataset_id"].eq("psyche_d")].copy()
    predictions: list[pd.DataFrame] = []
    searches: list[pd.DataFrame] = []
    audits: dict[str, Any] = {}
    for task_id in R5_LABEL_TASKS:
        for outer_fold in range(5):
            prediction, search, audit = _load_or_run(
                checkpoint,
                track="deployable",
                task_id=task_id,
                outer_fold=outer_fold,
                run=lambda task_id=task_id, outer_fold=outer_fold: _run_deployable_fold(
                    base,
                    task_id=task_id,
                    outer_fold=outer_fold,
                    seed=seed,
                ),
                overwrite=overwrite,
            )
            predictions.append(prediction)
            searches.append(search)
            audits[f"deployable:{task_id}:{outer_fold}"] = audit
            prediction, search, audit = _load_or_run(
                checkpoint,
                track="psyche_research",
                task_id=task_id,
                outer_fold=outer_fold,
                run=lambda task_id=task_id, outer_fold=outer_fold: _run_psyche_fold(
                    psyche,
                    task_id=task_id,
                    outer_fold=outer_fold,
                    seed=seed,
                ),
                overwrite=overwrite,
            )
            predictions.append(prediction)
            searches.append(search)
            audits[f"psyche_research:{task_id}:{outer_fold}"] = audit
    oof = _project_dual_heads(pd.concat(predictions, ignore_index=True))
    oof["r5_row_id"] = oof["r4_row_id"].astype(str)
    oof["r5_task_id"] = oof["r4_task_id"].astype(str)
    oof["split_seed"] = seed
    oof["baseline_recipe"] = "frozen_r4_inner_selected_top2_convex_calibrated"
    search = pd.concat(searches, ignore_index=True)
    metrics: dict[str, Any] = {}
    for (track, task_id), part in oof.groupby(["track", "r5_task_id"], sort=True):
        key = f"{track}:{task_id}"
        metrics[key] = {
            "overall": _metric(part),
            "by_source": {
                str(source): _metric(source_part)
                for source, source_part in part.groupby("dataset_id", sort=True)
            },
            "by_route": {
                str(route): _metric(route_part)
                for route, route_part in part.groupby("route_pattern", sort=True)
            },
            "by_outer_fold": {
                str(fold): _metric(fold_part)
                for fold, fold_part in part.groupby("outer_fold", sort=True)
            },
        }
    oof_path = output / "r4_recipe_paired_baseline_oof.parquet"
    search_path = output / "r4_recipe_inner_selection.parquet"
    metrics_path = output / "r4_recipe_baseline_metrics.json"
    audit_path = output / "r4_recipe_outer_isolation_audit.json"
    manifest_path = output / "artifact_manifest.json"
    for path in (oof_path, search_path, metrics_path, audit_path, manifest_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite r5 baseline artifact: {path}")
    oof.to_parquet(oof_path, index=False)
    search.to_parquet(search_path, index=False)
    _json_dump(metrics_path, metrics)
    _json_dump(audit_path, audits)
    manifest = {
        "protocol_version": R5_PROTOCOL_VERSION,
        "status": "pass",
        "evidence_level": (
            "sealed reused-cohort confirmation"
            if seed == R5_CONFIRMATION_SEED
            else "adaptive-development"
        ),
        "seed": seed,
        "r4_recipe_retrained_on_identical_r5_split": True,
        "historical_or_current_phq_feature_used": False,
        "confirmation_opened": seed == R5_CONFIRMATION_SEED,
        "artifacts": {
            "oof": {
                "path": oof_path.relative_to(root).as_posix(),
                "sha256": sha256_file(oof_path),
            },
            "search": {
                "path": search_path.relative_to(root).as_posix(),
                "sha256": sha256_file(search_path),
            },
            "metrics": {
                "path": metrics_path.relative_to(root).as_posix(),
                "sha256": sha256_file(metrics_path),
            },
            "audit": {
                "path": audit_path.relative_to(root).as_posix(),
                "sha256": sha256_file(audit_path),
            },
        },
    }
    _json_dump(manifest_path, manifest)
    return manifest


def summarize_r4_recipe_baselines(
    *,
    repository_root: Path,
    report_directory: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Validate and summarize all three development paired baselines."""

    root = Path(repository_root).resolve()
    base_output = Path(report_directory) if report_directory else root / DEFAULT_REPORT_RELATIVE
    if not base_output.is_absolute():
        base_output = root / base_output
    confirmation = base_output / f"seed-{R5_CONFIRMATION_SEED}"
    if confirmation.exists() and any(confirmation.glob("*.parquet")):
        raise PermissionError("r5 confirmation predictions were opened before R5-006")
    seed_rows: dict[str, Any] = {}
    oof_parts: list[pd.DataFrame] = []
    for seed in R5_DEVELOPMENT_SEEDS:
        directory = base_output / f"seed-{seed}"
        manifest_path = directory / "artifact_manifest.json"
        oof_path = directory / "r4_recipe_paired_baseline_oof.parquet"
        if not manifest_path.is_file() or not oof_path.is_file():
            raise FileNotFoundError(f"r5 paired baseline seed {seed} is incomplete")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "pass" or manifest.get("confirmation_opened"):
            raise ValueError(f"invalid r5 paired baseline manifest for seed {seed}")
        if sha256_file(oof_path) != manifest["artifacts"]["oof"]["sha256"]:
            raise ValueError(f"r5 paired baseline OOF hash drift for seed {seed}")
        oof = pd.read_parquet(oof_path)
        expected = load_r5_development_frame(repository_root=root, seed=seed)[
            ["r4_row_id", "outer_fold"]
        ].copy()
        observed = (
            oof.loc[oof["track"].eq("multisource_deployable")]
            .loc[oof["r5_task_id"].eq("phq_ge10_current"), ["r4_row_id", "outer_fold"]]
            .sort_values("r4_row_id")
            .reset_index(drop=True)
        )
        expected = expected.sort_values("r4_row_id").reset_index(drop=True)
        if not observed.equals(expected):
            raise ValueError(f"r5 paired baseline rows/folds differ for seed {seed}")
        seed_rows[str(seed)] = {
            "manifest_sha256": sha256_file(manifest_path),
            "oof_sha256": sha256_file(oof_path),
            "rows_per_task": int(len(expected)),
            "same_registered_rows_and_folds": True,
        }
        oof_parts.append(oof)
    combined = pd.concat(oof_parts, ignore_index=True)
    tables: dict[str, pd.DataFrame] = {
        "multisource_phq_ge10": combined.loc[
            combined["track"].eq("multisource_deployable")
            & combined["r5_task_id"].eq("phq_ge10_current")
        ],
        "multisource_phq_ge5": combined.loc[
            combined["track"].eq("multisource_deployable")
            & combined["r5_task_id"].eq("phq_ge5_current")
        ],
        "psyche_d_phq_ge10": combined.loc[
            combined["track"].eq("psyche_d_single_source_research")
            & combined["r5_task_id"].eq("phq_ge10_current")
        ],
        "psyche_d_phq_ge5": combined.loc[
            combined["track"].eq("psyche_d_single_source_research")
            & combined["r5_task_id"].eq("phq_ge5_current")
        ],
        "joint_101_111_phq_ge10": combined.loc[
            combined["track"].eq("multisource_deployable")
            & combined["r5_task_id"].eq("phq_ge10_current")
            & combined["route_pattern"].isin(["101", "111"])
        ],
    }
    summary_tables: dict[str, Any] = {}
    for name, table in tables.items():
        by_seed: dict[str, Any] = {}
        for seed, part in table.groupby("split_seed", sort=True):
            by_seed[str(int(seed))] = _metric(part)
        summary_tables[name] = {
            "by_seed": by_seed,
            "mean": {
                metric: float(np.mean([row["natural"][metric] for row in by_seed.values()]))
                for metric in ("auprc", "auroc", "brier", "ece", "normalized_ap")
            },
            "mean_participant_auprc": float(
                np.mean([row["participant_equal"]["auprc"] for row in by_seed.values()])
            ),
        }
    summary = {
        "protocol_version": R5_PROTOCOL_VERSION,
        "status": "pass",
        "evidence_level": "adaptive-development paired baseline",
        "recipe": "frozen_r4_inner_selected_top2_convex_calibrated",
        "development_seeds": list(R5_DEVELOPMENT_SEEDS),
        "confirmation_opened": False,
        "same_rows_participant_folds_weights_and_feature_availability": True,
        "seeds": seed_rows,
        "tables": summary_tables,
    }
    output_path = base_output / "development_paired_baseline_summary.json"
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite r5 baseline summary: {output_path}")
    _json_dump(output_path, summary)
    return summary


__all__ = ["run_r4_recipe_baseline", "summarize_r4_recipe_baselines"]
