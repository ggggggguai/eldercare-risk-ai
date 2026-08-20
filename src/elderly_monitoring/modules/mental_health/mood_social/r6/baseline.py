"""Replay the sealed r5 recipe on r6 splits for paired comparison."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from elderly_monitoring.modules.mental_health.mood_social.r4.baseline import (
    ap_context_metrics,
)
from elderly_monitoring.modules.mental_health.mood_social.r4.features import (
    route_signature,
)
from elderly_monitoring.modules.mental_health.mood_social.r4.selection import (
    participant_equal_weights,
)
from elderly_monitoring.modules.mental_health.mood_social.r5.competition import (
    _candidate_metrics,
)
from elderly_monitoring.modules.mental_health.mood_social.r5.features import (
    add_psyche_personal_change_features,
)
from elderly_monitoring.modules.mental_health.mood_social.r5.fusion import (
    nested_fusion_predictions,
)
from elderly_monitoring.modules.mental_health.mood_social.r5.selection import (
    R5CandidateSpec,
)
from elderly_monitoring.modules.mental_health.mood_social.r6.confirmation import (
    confirmation_state,
    locked_confirmation_recipe,
)
from elderly_monitoring.modules.mental_health.mood_social.r6.contract import (
    R6_CONFIRMATION_SEED,
    R6_DEVELOPMENT_SEEDS,
    R6_PROTOCOL_VERSION,
    sha256_file,
    write_json,
)
from elderly_monitoring.modules.mental_health.mood_social.r6.data import (
    load_r6_development_frame,
    r6_inner_fold_series,
)


Track = Literal["multisource_deployable", "psyche_d_single_source_research"]
TRACKS: tuple[Track, ...] = (
    "multisource_deployable",
    "psyche_d_single_source_research",
)
DEFAULT_REPORT_RELATIVE = Path(
    "reports/mental_health/mood_social/v3.3.3-r6/"
    "OPT-V333-R6-000/r5-paired-baseline"
)
R5_RECIPE_RELATIVE = Path(
    "reports/mental_health/mood_social/v3.3.3-r5/"
    "OPT-V333-R5-005-fusion/locked_recipe.json"
)
R5_STRUCTURE_RELATIVE = Path(
    "reports/mental_health/mood_social/v3.3.3-r5/"
    "OPT-V333-R5-003-structure/selected_structure.json"
)
R5_FEATURE_RELATIVE = Path(
    "reports/mental_health/mood_social/v3.3.3-r5/"
    "OPT-V333-R5-002-feature-ablation/retained_feature_set.json"
)


def _spec(value: dict[str, Any]) -> R5CandidateSpec:
    return R5CandidateSpec(
        candidate_id=str(value["candidate_id"]),
        family=str(value["family"]),  # type: ignore[arg-type]
        params=dict(value["params"]),
        seed=int(value["seed"]),
    )


def r5_locked_recipe(root: Path) -> dict[str, Any]:
    path = root / R5_RECIPE_RELATIVE
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "locked_before_confirmation":
        raise ValueError("r5 recipe is not sealed")
    return payload


def r5_locked_structure(root: Path) -> dict[str, str]:
    path = root / R5_STRUCTURE_RELATIVE
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("structure") != "conditional_ordered":
        raise ValueError("r5 objective structure drifted")
    return {
        "structure": str(payload["structure"]),
        "weight_scheme": str(payload["weight_scheme"]),
    }


def r5_retained_features(root: Path) -> tuple[str, ...]:
    payload = json.loads((root / R5_FEATURE_RELATIVE).read_text(encoding="utf-8"))
    names = tuple(str(value) for value in payload["retained_features"])
    if payload.get("status") != "pass" or len(names) != 486:
        raise ValueError("r5 retained feature set drifted")
    return names


def load_r6_track_frame(root: Path, *, seed: int, track: Track) -> tuple[pd.DataFrame, tuple[str, ...] | None]:
    base = load_r6_development_frame(repository_root_value=root, seed=int(seed))
    if track == "multisource_deployable":
        result = base.copy()
        result["feature_signature"] = result["route_pattern"].map(route_signature)
        if result["feature_signature"].eq("unsupported").any():
            raise ValueError("r6 deployable frame contains unsupported routes")
        return result, None
    psyche = base.loc[base["dataset_id"].eq("psyche_d")].copy()
    enriched, _batch1, _batch2 = add_psyche_personal_change_features(
        psyche, include_batch2=False
    )
    features = r5_retained_features(root)
    if not set(features).issubset(enriched.columns):
        raise ValueError("r6 cannot reproduce the r5 research feature set")
    return enriched, features


def _baseline_metric(frame: pd.DataFrame, probability: np.ndarray, target_column: str) -> dict[str, Any]:
    target = frame[target_column].to_numpy(int)
    weight = participant_equal_weights(frame)
    return {
        "rows": int(len(frame)),
        "participants": int(frame["global_participant_id"].nunique()),
        "positive_rows": int(target.sum()),
        "prevalence": float(target.mean()),
        "coverage": float(np.isfinite(probability).mean()),
        "natural": ap_context_metrics(target, probability),
        "participant_equal": ap_context_metrics(target, probability, sample_weight=weight),
    }


def run_r5_recipe_baseline(
    *,
    repository_root: Path,
    seed: int,
    report_directory: Path | None = None,
    overwrite: bool = False,
    allow_confirmation: bool = False,
) -> dict[str, Any]:
    """Retrain r5's complete nested graph on one registered r6 split."""

    root = Path(repository_root).resolve()
    seed = int(seed)
    if seed not in (*R6_DEVELOPMENT_SEEDS, R6_CONFIRMATION_SEED):
        raise ValueError(f"unregistered r6 baseline seed: {seed}")
    is_confirmation = seed == R6_CONFIRMATION_SEED
    if is_confirmation and (not allow_confirmation or not confirmation_state(root)["opened"]):
        raise PermissionError("r6 confirmation baseline cannot run before one-time opening")
    if is_confirmation:
        locked_confirmation_recipe(root)
    output_root = Path(report_directory) if report_directory else root / DEFAULT_REPORT_RELATIVE
    if not output_root.is_absolute():
        output_root = root / output_root
    output = output_root / f"seed-{seed}"
    manifest_path = output / "artifact_manifest.json"
    if manifest_path.is_file() and not overwrite:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    recipe = r5_locked_recipe(root)
    structure = r5_locked_structure(root)
    parts: list[pd.DataFrame] = []
    audits: dict[str, Any] = {}
    checkpoint = output / "_checkpoints"
    for track in TRACKS:
        frame, features = load_r6_track_frame(root, seed=seed, track=track)
        spec = _spec(recipe["tracks"][track]["stable_single_spec"])
        for outer_fold in range(5):
            prediction_path = checkpoint / track / f"outer-{outer_fold}.parquet"
            audit_path = checkpoint / track / f"outer-{outer_fold}.audit.json"
            if prediction_path.is_file() and audit_path.is_file() and not overwrite:
                prediction = pd.read_parquet(prediction_path)
                audit = json.loads(audit_path.read_text(encoding="utf-8"))
            else:
                train = frame.loc[frame["outer_fold"].ne(outer_fold)].copy()
                test = frame.loc[frame["outer_fold"].eq(outer_fold)].copy()
                inner = r6_inner_fold_series(train, outer_fold)
                prediction, audit = nested_fusion_predictions(
                    train, test, features, inner, [spec], track, structure
                )
                prediction["r6_row_id"] = prediction["r5_row_id"].astype(str)
                prediction["split_seed"] = seed
                prediction["outer_fold"] = outer_fold
                prediction["baseline_probability_ge5"] = prediction["probability_ge5"]
                prediction["baseline_probability_ge10"] = prediction["probability_ge10"]
                prediction_path.parent.mkdir(parents=True, exist_ok=True)
                prediction.to_parquet(prediction_path, index=False)
                write_json(audit_path, audit, overwrite=overwrite)
            parts.append(prediction)
            audits[f"{track}:{outer_fold}"] = audit
    combined = pd.concat(parts, ignore_index=True)
    expected_rows = 22070 + 10745
    if len(combined) != expected_rows or combined.duplicated(["track", "r6_row_id"]).any():
        raise ValueError("r6 paired r5 baseline OOF coverage is invalid")
    metrics: dict[str, Any] = {}
    for track, part in combined.groupby("track", sort=True):
        metrics[track] = {
            head: _baseline_metric(part, part[f"baseline_probability_{head}"].to_numpy(float), f"phq9_{head}_r3_target")
            for head in ("ge5", "ge10")
        }
        metrics[track]["joint_101_111_ge10"] = (
            _baseline_metric(
                part.loc[part["route_pattern"].isin(["101", "111"])],
                part.loc[part["route_pattern"].isin(["101", "111"]), "baseline_probability_ge10"].to_numpy(float),
                "phq9_ge10_r3_target",
            )
            if track == "multisource_deployable"
            else None
        )
    oof_path = output / "r5_recipe_paired_baseline_oof.parquet"
    metrics_path = output / "r5_recipe_baseline_metrics.json"
    audit_path = output / "r5_recipe_outer_isolation_audit.json"
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(oof_path, index=False)
    write_json(metrics_path, metrics, overwrite=overwrite)
    write_json(audit_path, audits, overwrite=overwrite)
    manifest = {
        "protocol_version": R6_PROTOCOL_VERSION,
        "status": "pass",
        "evidence_level": "sealed reused-cohort confirmation" if is_confirmation else "adaptive-development paired baseline",
        "seed": seed,
        "r5_recipe_retrained_on_identical_r6_split": True,
        "historical_or_current_phq_feature_used": False,
        "outer_test_labels_used_for_selection": False,
        "confirmation_opened": is_confirmation,
        "r5_recipe_sha256": sha256_file(root / R5_RECIPE_RELATIVE),
        "r5_structure_sha256": sha256_file(root / R5_STRUCTURE_RELATIVE),
        "r5_feature_set_sha256": sha256_file(root / R5_FEATURE_RELATIVE),
        "artifacts": {
            "oof": {"path": oof_path.relative_to(root).as_posix(), "sha256": sha256_file(oof_path)},
            "metrics": {"path": metrics_path.relative_to(root).as_posix(), "sha256": sha256_file(metrics_path)},
            "audit": {"path": audit_path.relative_to(root).as_posix(), "sha256": sha256_file(audit_path)},
        },
    }
    write_json(manifest_path, manifest, overwrite=overwrite)
    return manifest


def summarize_r5_recipe_baselines(
    *, repository_root: Path, report_directory: Path | None = None, overwrite: bool = False
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    output = Path(report_directory) if report_directory else root / DEFAULT_REPORT_RELATIVE
    if not output.is_absolute():
        output = root / output
    confirmation_directory = output / f"seed-{R6_CONFIRMATION_SEED}"
    if confirmation_directory.exists() and any(confirmation_directory.glob("*.parquet")):
        raise PermissionError("r6 confirmation baseline was opened before R6-006")
    rows: dict[str, Any] = {}
    table_metrics: dict[str, list[dict[str, Any]]] = {
        "multisource_ge5": [], "multisource_ge10": [], "psyche_ge5": [],
        "psyche_ge10": [], "joint_101_111_ge10": [],
    }
    for seed in R6_DEVELOPMENT_SEEDS:
        directory = output / f"seed-{seed}"
        manifest_path = directory / "artifact_manifest.json"
        metrics_path = directory / "r5_recipe_baseline_metrics.json"
        oof_path = directory / "r5_recipe_paired_baseline_oof.parquet"
        for path in (manifest_path, metrics_path, oof_path):
            if not path.is_file():
                raise FileNotFoundError(f"r6 paired baseline seed {seed} is incomplete: {path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "pass" or manifest.get("confirmation_opened"):
            raise ValueError(f"invalid r6 paired baseline manifest for seed {seed}")
        if manifest["artifacts"]["oof"]["sha256"] != sha256_file(oof_path):
            raise ValueError(f"r6 paired baseline OOF hash drift for seed {seed}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        table_metrics["multisource_ge5"].append(metrics["multisource_deployable"]["ge5"])
        table_metrics["multisource_ge10"].append(metrics["multisource_deployable"]["ge10"])
        table_metrics["psyche_ge5"].append(metrics["psyche_d_single_source_research"]["ge5"])
        table_metrics["psyche_ge10"].append(metrics["psyche_d_single_source_research"]["ge10"])
        table_metrics["joint_101_111_ge10"].append(metrics["multisource_deployable"]["joint_101_111_ge10"])
        rows[str(seed)] = {
            "manifest_sha256": sha256_file(manifest_path),
            "oof_sha256": sha256_file(oof_path),
            "same_registered_rows_and_folds": True,
        }
    summary_tables: dict[str, Any] = {}
    for name, values in table_metrics.items():
        summary_tables[name] = {
            "by_seed": {str(seed): value for seed, value in zip(R6_DEVELOPMENT_SEEDS, values)},
            "mean": {
                metric: float(np.mean([value["natural"][metric] for value in values]))
                for metric in ("auprc", "auroc", "brier", "ece", "normalized_ap")
            },
            "mean_participant_auprc": float(np.mean([value["participant_equal"]["auprc"] for value in values])),
        }
    summary = {
        "protocol_version": R6_PROTOCOL_VERSION,
        "status": "pass",
        "evidence_level": "adaptive-development paired r5 baseline",
        "recipe": "sealed_r5_stable_single_strict_nested_calibrated",
        "development_seeds": list(R6_DEVELOPMENT_SEEDS),
        "confirmation_opened": False,
        "same_rows_participant_folds_weights_and_feature_availability": True,
        "seeds": rows,
        "tables": summary_tables,
    }
    write_json(output / "development_paired_baseline_summary.json", summary, overwrite=overwrite)
    return summary


__all__ = [
    "DEFAULT_REPORT_RELATIVE", "TRACKS", "load_r6_track_frame", "r5_locked_recipe",
    "r5_locked_structure", "r5_retained_features", "run_r5_recipe_baseline",
    "summarize_r5_recipe_baselines",
]
