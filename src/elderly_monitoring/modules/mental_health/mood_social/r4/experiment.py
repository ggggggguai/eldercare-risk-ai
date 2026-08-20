"""Nested outer evaluation for the r4 deployable and PSYCHE research tracks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from elderly_monitoring.modules.mental_health.mood_social.r3.data import (
    inner_fold_series,
)
from elderly_monitoring.modules.mental_health.mood_social.r4.baseline import (
    ap_context_metrics,
)
from elderly_monitoring.modules.mental_health.mood_social.r4.calibration import (
    calibrate_outer_prediction,
)
from elderly_monitoring.modules.mental_health.mood_social.r4.contract import (
    PSYCHE_RESEARCH_SENSOR_FEATURES,
    R4_LABEL_TASKS,
    R4_PROTOCOL_VERSION,
)
from elderly_monitoring.modules.mental_health.mood_social.r4.data import (
    load_r4_development_frame,
    select_label_task,
)
from elderly_monitoring.modules.mental_health.mood_social.r4.features import (
    add_psyche_causal_features,
    deployable_features_for_signature,
    route_signature,
)
from elderly_monitoring.modules.mental_health.mood_social.r4.selection import (
    enforce_dual_head_monotonicity,
    participant_equal_weights,
    select_and_fit_predict,
)


DEFAULT_REPORT_RELATIVE = Path(
    "reports/mental_health/mood_social/v3.3.3-r4/OPT-V333-R4-003-007-selection"
)
BASELINE_RELATIVE = Path(
    "reports/mental_health/mood_social/v3.3.3-r4/OPT-V333-R4-001-baseline/"
    "strict_dual_label_baseline_oof.parquet"
)
R3_CANDIDATE_RELATIVE = Path(
    "reports/mental_health/mood_social/v3.3.3-r3/OPT-V333-007-formal-outer/"
    "formal_outer_oof.parquet"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _run_partition(
    frame: pd.DataFrame,
    *,
    outer_fold: int,
    features: tuple[str, ...],
    participant_equal: bool,
    seed: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    train = frame.loc[frame["outer_fold"].ne(outer_fold)].copy()
    test = frame.loc[frame["outer_fold"].eq(outer_fold)].copy()
    if train.empty or test.empty:
        raise ValueError("r4 partition requires non-empty outer train and test")
    inner = inner_fold_series(frame, outer_fold).loc[train.index].astype(int)
    raw_outer, raw_inner, rows, selection = select_and_fit_predict(
        train,
        test,
        features,
        inner,
        participant_equal=participant_equal,
        seed=seed,
    )
    calibration_frame = train[
        ["global_participant_id", "binary_target"]
    ].copy()
    calibration_frame["inner_fold"] = inner.to_numpy(int)
    calibrated, calibration = calibrate_outer_prediction(
        calibration_frame,
        raw_inner,
        raw_outer,
    )
    output = test[
        [
            "r4_row_id",
            "r3_row_id",
            "dataset_id",
            "global_participant_id",
            "route_pattern",
            "outer_fold",
            "binary_target",
            "r4_task_id",
        ]
    ].copy()
    output["raw_candidate_probability"] = raw_outer
    output["candidate_probability"] = calibrated
    output["calibration_method"] = calibration["selected_method"]
    audit = {
        "outer_fold": outer_fold,
        "outer_train_rows": int(len(train)),
        "outer_test_rows": int(len(test)),
        "outer_train_participants": int(train["global_participant_id"].nunique()),
        "outer_test_participants": int(test["global_participant_id"].nunique()),
        "participant_overlap": int(
            len(
                set(train["global_participant_id"].astype(str))
                & set(test["global_participant_id"].astype(str))
            )
        ),
        "selection": selection,
        "calibration": calibration,
    }
    if audit["participant_overlap"]:
        raise ValueError("r4 outer participant leakage")
    return output, rows, audit


def _run_deployable_fold(
    base: pd.DataFrame,
    *,
    task_id: str,
    outer_fold: int,
    seed: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    frame = select_label_task(base, task_id)
    frame["feature_signature"] = frame["route_pattern"].map(route_signature)
    outputs: list[pd.DataFrame] = []
    search: list[dict[str, Any]] = []
    audits: dict[str, Any] = {}
    for signature in ("profile", "sleep", "joint"):
        partition = frame.loc[frame["feature_signature"].eq(signature)].copy()
        output, rows, audit = _run_partition(
            partition,
            outer_fold=outer_fold,
            features=deployable_features_for_signature(signature),
            participant_equal=True,
            seed=seed,
        )
        output["track"] = "multisource_deployable"
        output["feature_signature"] = signature
        outputs.append(output)
        search.extend(
            {
                **row,
                "track": "multisource_deployable",
                "task_id": task_id,
                "outer_fold": outer_fold,
                "feature_signature": signature,
            }
            for row in rows
        )
        audits[signature] = audit
    result = pd.concat(outputs, ignore_index=True)
    expected = frame.loc[frame["outer_fold"].eq(outer_fold)]
    if len(result) != len(expected) or result["r4_row_id"].duplicated().any():
        raise ValueError("deployable r4 fold coverage is invalid")
    return result, search, audits


def _run_psyche_fold(
    psyche: pd.DataFrame,
    *,
    task_id: str,
    outer_fold: int,
    seed: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    frame = select_label_task(psyche, task_id)
    causal, generated = add_psyche_causal_features(frame)
    features = (*PSYCHE_RESEARCH_SENSOR_FEATURES, *generated)
    output, rows, audit = _run_partition(
        causal,
        outer_fold=outer_fold,
        features=features,
        participant_equal=True,
        seed=seed,
    )
    output["track"] = "psyche_d_single_source_research"
    output["feature_signature"] = "psyche_audited_sensor_plus_causal"
    search = [
        {
            **row,
            "track": "psyche_d_single_source_research",
            "task_id": task_id,
            "outer_fold": outer_fold,
            "feature_signature": "psyche_audited_sensor_plus_causal",
        }
        for row in rows
    ]
    audit["research_only"] = True
    audit["feature_count"] = len(features)
    audit["historical_phq_used"] = False
    return output, search, audit


def _checkpoint_paths(
    checkpoint: Path, track: str, task_id: str, outer_fold: int
) -> tuple[Path, Path, Path]:
    prefix = checkpoint / f"{track}__{task_id}__outer{outer_fold}"
    return (
        prefix.with_suffix(".prediction.parquet"),
        prefix.with_suffix(".search.parquet"),
        prefix.with_suffix(".audit.json"),
    )


def _load_or_run(
    checkpoint: Path,
    *,
    track: str,
    task_id: str,
    outer_fold: int,
    run: Any,
    overwrite: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    prediction_path, search_path, audit_path = _checkpoint_paths(
        checkpoint, track, task_id, outer_fold
    )
    if all(path.exists() for path in (prediction_path, search_path, audit_path)) and not overwrite:
        return (
            pd.read_parquet(prediction_path),
            pd.read_parquet(search_path),
            json.loads(audit_path.read_text(encoding="utf-8")),
        )
    prediction, search_rows, audit = run()
    search = pd.DataFrame(search_rows)
    for column in ("params",):
        if column in search:
            search[column] = search[column].map(
                lambda value: json.dumps(value, sort_keys=True) if isinstance(value, dict) else value
            )
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    prediction.to_parquet(prediction_path, index=False)
    search.to_parquet(search_path, index=False)
    _json_dump(audit_path, audit)
    return prediction, search, audit


def _metric_block(frame: pd.DataFrame) -> dict[str, Any]:
    target = frame["binary_target"].to_numpy(int)
    candidate = frame["candidate_probability"].to_numpy(float)
    baseline = frame["baseline_probability"].to_numpy(float)
    weight = participant_equal_weights(frame)
    candidate_natural = ap_context_metrics(target, candidate)
    baseline_natural = ap_context_metrics(target, baseline)
    candidate_participant = ap_context_metrics(target, candidate, sample_weight=weight)
    baseline_participant = ap_context_metrics(target, baseline, sample_weight=weight)
    return {
        "rows": int(len(frame)),
        "participants": int(frame["global_participant_id"].nunique()),
        "positive_rows": int(target.sum()),
        "coverage": float(np.isfinite(candidate).mean()),
        "candidate": candidate_natural,
        "baseline": baseline_natural,
        "delta": {
            key: float(candidate_natural[key] - baseline_natural[key])
            for key in ("auprc", "auroc", "brier", "ece", "normalized_ap")
        },
        "participant_equal": {
            "candidate": candidate_participant,
            "baseline": baseline_participant,
            "delta_auprc": float(
                candidate_participant["auprc"] - baseline_participant["auprc"]
            ),
            "delta_auroc": float(
                candidate_participant["auroc"] - baseline_participant["auroc"]
            ),
        },
    }


def run_screening_experiment(
    *,
    repository_root: Path,
    report_directory: Path | None = None,
    overwrite: bool = False,
    seed: int = 20260812,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    output = Path(report_directory) if report_directory else root / DEFAULT_REPORT_RELATIVE
    if not output.is_absolute():
        output = root / output
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "_checkpoints"
    base = load_r4_development_frame(repository_root=root)
    psyche = base.loc[base["dataset_id"].eq("psyche_d")].copy()
    predictions: list[pd.DataFrame] = []
    searches: list[pd.DataFrame] = []
    audits: dict[str, Any] = {}
    for task_id in R4_LABEL_TASKS:
        for outer_fold in range(5):
            prediction, search, audit = _load_or_run(
                checkpoint,
                track="deployable",
                task_id=task_id,
                outer_fold=outer_fold,
                run=lambda task_id=task_id, outer_fold=outer_fold: _run_deployable_fold(
                    base, task_id=task_id, outer_fold=outer_fold, seed=seed
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
                    psyche, task_id=task_id, outer_fold=outer_fold, seed=seed
                ),
                overwrite=overwrite,
            )
            predictions.append(prediction)
            searches.append(search)
            audits[f"psyche_research:{task_id}:{outer_fold}"] = audit
    oof = pd.concat(predictions, ignore_index=True)
    search = pd.concat(searches, ignore_index=True)

    # Apply the two-head order constraint within each track and row.  Raw and
    # unprojected calibrated probabilities remain in the artifact for audit.
    oof["unprojected_candidate_probability"] = oof["candidate_probability"]
    projected_parts: list[pd.DataFrame] = []
    for track, part in oof.groupby("track", sort=True):
        early = part.loc[part["r4_task_id"].eq("phq_ge5_current")].copy()
        elevated = part.loc[part["r4_task_id"].eq("phq_ge10_current")].copy()
        early = early.sort_values("r4_row_id", kind="stable")
        elevated = elevated.sort_values("r4_row_id", kind="stable")
        if not early["r4_row_id"].reset_index(drop=True).equals(
            elevated["r4_row_id"].reset_index(drop=True)
        ):
            raise ValueError(f"dual-head rows differ for {track}")
        p5, p10 = enforce_dual_head_monotonicity(
            early["candidate_probability"].to_numpy(float),
            elevated["candidate_probability"].to_numpy(float),
        )
        early["candidate_probability"] = p5
        elevated["candidate_probability"] = p10
        projected_parts.extend([early, elevated])
    oof = pd.concat(projected_parts, ignore_index=True)

    baseline = pd.read_parquet(root / BASELINE_RELATIVE)[
        ["r4_row_id", "r4_task_id", "baseline_probability"]
    ]
    oof = oof.merge(
        baseline,
        on=["r4_row_id", "r4_task_id"],
        how="left",
        validate="many_to_one",
    )
    if oof["baseline_probability"].isna().any():
        raise ValueError("r4 screening baseline merge incomplete")

    metrics: dict[str, Any] = {}
    for (track, task_id), part in oof.groupby(["track", "r4_task_id"], sort=True):
        key = f"{track}:{task_id}"
        metrics[key] = {
            "overall": _metric_block(part),
            "by_source": {
                str(source): _metric_block(source_part)
                for source, source_part in part.groupby("dataset_id", sort=True)
            },
            "by_route": {
                str(route): _metric_block(route_part)
                for route, route_part in part.groupby("route_pattern", sort=True)
            },
            "by_outer_fold": {
                str(fold): _metric_block(fold_part)
                for fold, fold_part in part.groupby("outer_fold", sort=True)
            },
        }

    oof_path = output / "screening_outer_oof.parquet"
    search_path = output / "inner_candidate_search.parquet"
    metrics_path = output / "screening_metrics.json"
    audit_path = output / "outer_isolation_and_selection_audit.json"
    manifest_path = output / "artifact_manifest.json"
    for path in (oof_path, search_path, metrics_path, audit_path, manifest_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite r4 screening artifact: {path}")
    oof.to_parquet(oof_path, index=False)
    search.to_parquet(search_path, index=False)
    _json_dump(metrics_path, metrics)
    _json_dump(audit_path, audits)
    manifest = {
        "protocol_version": R4_PROTOCOL_VERSION,
        "status": "pass",
        "evidence_level": "adaptive-development/reused-benchmark",
        "seed": seed,
        "outer_results_opened": True,
        "historical_phq_used": False,
        "artifacts": {
            "oof": {"path": oof_path.relative_to(root).as_posix(), "sha256": _sha256(oof_path)},
            "search": {"path": search_path.relative_to(root).as_posix(), "sha256": _sha256(search_path)},
            "metrics": {"path": metrics_path.relative_to(root).as_posix(), "sha256": _sha256(metrics_path)},
            "audit": {"path": audit_path.relative_to(root).as_posix(), "sha256": _sha256(audit_path)},
        },
    }
    _json_dump(manifest_path, manifest)
    return manifest


__all__ = ["run_screening_experiment"]
