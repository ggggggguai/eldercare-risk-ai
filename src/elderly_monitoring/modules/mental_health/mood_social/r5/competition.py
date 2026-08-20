"""Finite two-stage model competition for OPT-V333-R5-004."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd

from elderly_monitoring.modules.mental_health.mood_social.r4.baseline import (
    ap_context_metrics,
)
from elderly_monitoring.modules.mental_health.mood_social.r4.evaluation import (
    ap_at_prevalence,
)
from elderly_monitoring.modules.mental_health.mood_social.r4.features import (
    deployable_features_for_signature,
    route_signature,
)
from elderly_monitoring.modules.mental_health.mood_social.r4.selection import (
    participant_equal_weights,
)
from elderly_monitoring.modules.mental_health.mood_social.r5.contract import (
    R5_CANDIDATE_FAMILY_BUDGET,
    R5_DEVELOPMENT_SEEDS,
    R5_PROTOCOL_VERSION,
    sha256_file,
)
from elderly_monitoring.modules.mental_health.mood_social.r5.data import (
    load_r5_development_frame,
    r5_inner_fold_series,
)
from elderly_monitoring.modules.mental_health.mood_social.r5.features import (
    add_psyche_personal_change_features,
)
from elderly_monitoring.modules.mental_health.mood_social.r5.objectives import (
    conditional_ordered_heads,
    project_independent_heads,
)
from elderly_monitoring.modules.mental_health.mood_social.r5.selection import (
    R5CandidateSpec,
    fit_r5_model,
    predict_r5_model,
    r5_candidate_space,
)
from elderly_monitoring.modules.mental_health.mood_social.r5.weights import (
    r5_training_weights,
)


Track = Literal["multisource_deployable", "psyche_d_single_source_research"]
Stage = Literal["screening", "full_inner"]
TRACKS: tuple[Track, ...] = (
    "multisource_deployable",
    "psyche_d_single_source_research",
)
FAMILIES = tuple(R5_CANDIDATE_FAMILY_BUDGET)
SCREENING_SEED = R5_DEVELOPMENT_SEEDS[0]
SCREENING_OUTER_FOLD = 0
SCREENING_VALIDATION_FOLD = 0
REFERENCE_CANDIDATE_ID = "lgb_leaf15_min40_l27"
DEFAULT_REPORT_RELATIVE = Path(
    "reports/mental_health/mood_social/v3.3.3-r5/OPT-V333-R5-004-competition"
)
RETAINED_FEATURE_RELATIVE = Path(
    "reports/mental_health/mood_social/v3.3.3-r5/"
    "OPT-V333-R5-002-feature-ablation/retained_feature_set.json"
)
STRUCTURE_SELECTION_RELATIVE = Path(
    "reports/mental_health/mood_social/v3.3.3-r5/"
    "OPT-V333-R5-003-structure/selected_structure.json"
)
BASELINE_RELATIVE = Path(
    "reports/mental_health/mood_social/v3.3.3-r5/"
    "OPT-V333-R5-000/r4-paired-baseline"
)


def _write_json(path: Path, value: Any, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite r5 competition artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _spec_from_dict(value: dict[str, Any]) -> R5CandidateSpec:
    return R5CandidateSpec(
        candidate_id=str(value["candidate_id"]),
        family=str(value["family"]),  # type: ignore[arg-type]
        params=dict(value["params"]),
        seed=int(value["seed"]),
    )


def _retained_features(root: Path) -> tuple[str, ...]:
    payload = json.loads((root / RETAINED_FEATURE_RELATIVE).read_text(encoding="utf-8"))
    features = tuple(str(name) for name in payload["retained_features"])
    if payload.get("status") != "pass" or len(features) != 486:
        raise ValueError("r5 retained feature set is unavailable or drifted")
    return features


def _selected_structure(root: Path) -> dict[str, str]:
    payload = json.loads((root / STRUCTURE_SELECTION_RELATIVE).read_text(encoding="utf-8"))
    if payload.get("structure") != "conditional_ordered":
        raise ValueError("r5 competition expects the sealed conditional structure")
    if payload.get("weight_scheme") not in {
        "participant_equal",
        "threshold_near_mild_downweight",
        "extreme_severity_mild_strengthen",
    }:
        raise ValueError("r5 competition weight scheme drifted")
    return {
        "structure": str(payload["structure"]),
        "weight_scheme": str(payload["weight_scheme"]),
    }


def _load_track_frame(root: Path, seed: int, track: Track) -> tuple[pd.DataFrame, tuple[str, ...] | None]:
    base = load_r5_development_frame(repository_root=root, seed=int(seed))
    if track == "multisource_deployable":
        result = base.copy()
        result["feature_signature"] = result["route_pattern"].map(route_signature)
        if result["feature_signature"].eq("unsupported").any():
            raise ValueError("r5 deployable competition has unsupported route rows")
        return result, None
    psyche = base.loc[base["dataset_id"].eq("psyche_d")].copy()
    enriched, _batch1, _batch2 = add_psyche_personal_change_features(
        psyche, include_batch2=False
    )
    features = _retained_features(root)
    if not set(features).issubset(enriched.columns):
        raise ValueError("r5 competition cannot reproduce retained PSYCHE features")
    return enriched, features


def _fit_binary(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: tuple[str, ...],
    spec: R5CandidateSpec,
    *,
    target_column: str,
    weight_scheme: str,
) -> np.ndarray:
    fit = train.copy()
    fit["binary_target"] = fit[target_column].astype(int)
    if fit["binary_target"].nunique() != 2:
        raise ValueError(f"r5 candidate train lacks both classes for {target_column}")
    if weight_scheme == "participant_equal":
        weight = participant_equal_weights(fit)
    else:
        weight = r5_training_weights(fit, weight_scheme)  # type: ignore[arg-type]
    model = fit_r5_model(fit, features, spec, sample_weight=weight)
    return predict_r5_model(model, validation, features)


def _predict_multisource_split(
    train: pd.DataFrame, validation: pd.DataFrame, spec: R5CandidateSpec
) -> tuple[np.ndarray, np.ndarray]:
    p5 = pd.Series(np.nan, index=validation.index, dtype=float)
    p10 = pd.Series(np.nan, index=validation.index, dtype=float)
    for signature in ("profile", "sleep", "joint"):
        fit = train.loc[train["feature_signature"].eq(signature)]
        score = validation.loc[validation["feature_signature"].eq(signature)]
        if score.empty:
            continue
        if fit.empty:
            raise ValueError(f"r5 multisource train lacks {signature} support")
        features = deployable_features_for_signature(signature)
        p5.loc[score.index] = _fit_binary(
            fit,
            score,
            features,
            spec,
            target_column="phq9_ge5_r3_target",
            weight_scheme="participant_equal",
        )
        p10.loc[score.index] = _fit_binary(
            fit,
            score,
            features,
            spec,
            target_column="phq9_ge10_r3_target",
            weight_scheme="participant_equal",
        )
    if p5.isna().any() or p10.isna().any():
        raise ValueError("r5 multisource candidate prediction coverage is incomplete")
    return project_independent_heads(
        p5.loc[validation.index].to_numpy(float),
        p10.loc[validation.index].to_numpy(float),
    )


def _predict_psyche_split(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: tuple[str, ...],
    spec: R5CandidateSpec,
    weight_scheme: str,
) -> tuple[np.ndarray, np.ndarray]:
    p5 = _fit_binary(
        train,
        validation,
        features,
        spec,
        target_column="phq9_ge5_r3_target",
        weight_scheme=weight_scheme,
    )
    conditional_train = train.loc[train["phq9_ge5_r3_target"].eq(1)]
    conditional = _fit_binary(
        conditional_train,
        validation,
        features,
        spec,
        target_column="phq9_ge10_r3_target",
        weight_scheme=weight_scheme,
    )
    return conditional_ordered_heads(p5, conditional)


def _predict_split(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: tuple[str, ...] | None,
    spec: R5CandidateSpec,
    track: Track,
    structure: dict[str, str],
) -> tuple[np.ndarray, np.ndarray]:
    if track == "multisource_deployable":
        return _predict_multisource_split(train, validation, spec)
    if features is None:
        raise ValueError("r5 PSYCHE competition features are missing")
    return _predict_psyche_split(
        train, validation, features, spec, structure["weight_scheme"]
    )


def _metric(frame: pd.DataFrame, target_column: str, probability: np.ndarray, track: Track) -> dict[str, float]:
    target = frame[target_column].to_numpy(int)
    natural = ap_context_metrics(target, probability)
    participant = ap_context_metrics(
        target, probability, sample_weight=participant_equal_weights(frame)
    )
    macro_values: list[float] = []
    for _, positions in frame.groupby("dataset_id", sort=True).indices.items():
        location = np.asarray(positions, dtype=int)
        source_target = target[location]
        if np.unique(source_target).size == 2:
            macro_values.append(ap_at_prevalence(source_target, probability[location]))
    macro = float(np.mean(macro_values)) if macro_values else ap_at_prevalence(target, probability)
    selection_score = (
        0.55 * natural["auprc"] + 0.25 * participant["auprc"] + 0.20 * macro
        if track == "multisource_deployable"
        else 0.65 * natural["auprc"] + 0.35 * participant["auprc"]
    )
    return {
        **natural,
        "participant_auprc": participant["auprc"],
        "macro_ap_at_10pct": macro,
        "selection_score": selection_score,
    }


def _candidate_metrics(
    frame: pd.DataFrame, p5: np.ndarray, p10: np.ndarray, track: Track
) -> dict[str, Any]:
    violation = int(np.sum(np.asarray(p10) > np.asarray(p5) + 1.0e-12))
    if violation:
        raise ValueError(f"r5 candidate has {violation} monotonic violations")
    heads = {
        "ge5": _metric(frame, "phq9_ge5_r3_target", p5, track),
        "ge10": _metric(frame, "phq9_ge10_r3_target", p10, track),
    }
    return {
        "heads": heads,
        "joint_selection_score": float(
            np.mean([heads[head]["selection_score"] for head in ("ge5", "ge10")])
        ),
        "monotonic_violation_count": violation,
        "coverage": float(np.isfinite(p5).mean() * np.isfinite(p10).mean()),
    }


def _prediction_frame(
    frame: pd.DataFrame,
    p5: np.ndarray,
    p10: np.ndarray,
    *,
    candidate_id: str,
    track: Track,
    stage: str,
) -> pd.DataFrame:
    output = frame[
        [
            "r5_row_id",
            "global_participant_id",
            "dataset_id",
            "route_pattern",
            "outer_fold",
            "phq9_ge5_r3_target",
            "phq9_ge10_r3_target",
        ]
    ].copy()
    output["candidate_id"] = candidate_id
    output["track"] = track
    output["stage"] = stage
    output["probability_ge5"] = p5
    output["probability_ge10"] = p10
    return output


def _candidate_checkpoint(output: Path, stage: Stage, track: Track, candidate_id: str) -> Path:
    return output / stage / track / candidate_id


def run_competition_candidates(
    *,
    repository_root: Path,
    stage: Stage,
    track: Track,
    family: str | None = None,
    report_directory: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run one checkpointable screening/full-inner shard."""

    if stage not in {"screening", "full_inner"} or track not in TRACKS:
        raise ValueError("invalid r5 competition stage or track")
    if family is not None and family not in FAMILIES:
        raise ValueError(f"unknown r5 candidate family: {family}")
    root = Path(repository_root).resolve()
    output = Path(report_directory) if report_directory else root / DEFAULT_REPORT_RELATIVE
    if not output.is_absolute():
        output = root / output
    frame, features = _load_track_frame(root, SCREENING_SEED, track)
    frame = frame.loc[frame["outer_fold"].ne(SCREENING_OUTER_FOLD)].copy()
    inner = r5_inner_fold_series(frame, SCREENING_OUTER_FOLD).astype(int)
    structure = _selected_structure(root)
    if stage == "screening":
        specs = list(r5_candidate_space(SCREENING_SEED))
    else:
        selection_path = output / "screening" / track / "family_top2.json"
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        specs = [_spec_from_dict(value) for value in selection["candidates"]]
    if family is not None:
        specs = [spec for spec in specs if spec.family == family]
    completed = 0
    failed = 0
    for spec in specs:
        checkpoint = _candidate_checkpoint(output, stage, track, spec.candidate_id)
        manifest_path = checkpoint / "manifest.json"
        if manifest_path.exists() and not overwrite:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            completed += int(manifest.get("status") == "pass")
            failed += int(manifest.get("status") != "pass")
            continue
        try:
            if stage == "screening":
                train = frame.loc[inner.ne(SCREENING_VALIDATION_FOLD)]
                validation = frame.loc[inner.eq(SCREENING_VALIDATION_FOLD)]
                p5, p10 = _predict_split(
                    train, validation, features, spec, track, structure
                )
                scored = validation
                prediction = _prediction_frame(
                    validation,
                    p5,
                    p10,
                    candidate_id=spec.candidate_id,
                    track=track,
                    stage=stage,
                )
                prediction["inner_fold"] = SCREENING_VALIDATION_FOLD
            else:
                p5_series = pd.Series(np.nan, index=frame.index, dtype=float)
                p10_series = pd.Series(np.nan, index=frame.index, dtype=float)
                for fold in sorted(inner.unique()):
                    train = frame.loc[inner.ne(fold)]
                    validation = frame.loc[inner.eq(fold)]
                    fold_p5, fold_p10 = _predict_split(
                        train, validation, features, spec, track, structure
                    )
                    p5_series.loc[validation.index] = fold_p5
                    p10_series.loc[validation.index] = fold_p10
                if p5_series.isna().any() or p10_series.isna().any():
                    raise ValueError("r5 full inner candidate OOF is incomplete")
                p5 = p5_series.loc[frame.index].to_numpy(float)
                p10 = p10_series.loc[frame.index].to_numpy(float)
                scored = frame
                prediction = _prediction_frame(
                    frame,
                    p5,
                    p10,
                    candidate_id=spec.candidate_id,
                    track=track,
                    stage=stage,
                )
                prediction["inner_fold"] = inner.to_numpy(int)
            metrics = _candidate_metrics(scored, p5, p10, track)
            checkpoint.mkdir(parents=True, exist_ok=True)
            prediction_path = checkpoint / "inner_oof.parquet"
            metrics_path = checkpoint / "metrics.json"
            prediction.to_parquet(prediction_path, index=False)
            _write_json(metrics_path, metrics, overwrite=True)
            manifest = {
                "protocol_version": R5_PROTOCOL_VERSION,
                "status": "pass",
                "stage": stage,
                "track": track,
                "spec": asdict(spec),
                "outer_test_labels_read": False,
                "confirmation_opened": False,
                "prediction_sha256": sha256_file(prediction_path),
                "metrics_sha256": sha256_file(metrics_path),
            }
            _write_json(manifest_path, manifest, overwrite=True)
            completed += 1
        except Exception as error:
            checkpoint.mkdir(parents=True, exist_ok=True)
            _write_json(
                manifest_path,
                {
                    "protocol_version": R5_PROTOCOL_VERSION,
                    "status": "failed",
                    "stage": stage,
                    "track": track,
                    "spec": asdict(spec),
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "predictions_opened": False,
                    "confirmation_opened": False,
                },
                overwrite=True,
            )
            failed += 1
    return {
        "status": "pass" if completed else "failed",
        "stage": stage,
        "track": track,
        "family": family,
        "completed": completed,
        "failed": failed,
    }


def _load_stage_rows(output: Path, stage: Stage, track: Track) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    root = output / stage / track
    if not root.is_dir():
        return rows
    for manifest_path in sorted(root.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "pass":
            continue
        metrics_path = manifest_path.parent / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append({"spec": manifest["spec"], "metrics": metrics})
    return rows


def finalize_competition_stage(
    *,
    repository_root: Path,
    stage: Stage,
    track: Track,
    report_directory: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Select family top-2 or at most three track finalists."""

    root = Path(repository_root).resolve()
    output = Path(report_directory) if report_directory else root / DEFAULT_REPORT_RELATIVE
    if not output.is_absolute():
        output = root / output
    rows = _load_stage_rows(output, stage, track)
    expected = 40 if stage == "screening" else 10
    failed_manifests = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (output / stage / track).glob("*/manifest.json")
        if json.loads(path.read_text(encoding="utf-8")).get("status") != "pass"
    ]
    if len(rows) + len(failed_manifests) != expected:
        raise RuntimeError(
            f"r5 {stage} {track} is incomplete: {len(rows)} pass + "
            f"{len(failed_manifests)} failed != {expected}"
        )
    if stage == "screening":
        selected: list[dict[str, Any]] = []
        for family in FAMILIES:
            family_rows = [row for row in rows if row["spec"]["family"] == family]
            if len(family_rows) < 2:
                raise RuntimeError(f"r5 screening family {family} has fewer than two passes")
            family_rows.sort(
                key=lambda row: (
                    row["metrics"]["joint_selection_score"],
                    row["spec"]["candidate_id"],
                ),
                reverse=True,
            )
            family_selected = family_rows[:2]
            if family == "lightgbm" and not any(
                row["spec"]["candidate_id"] == REFERENCE_CANDIDATE_ID
                for row in family_selected
            ):
                reference = next(
                    row
                    for row in family_rows
                    if row["spec"]["candidate_id"] == REFERENCE_CANDIDATE_ID
                )
                family_selected = [family_rows[0], reference]
            selected.extend(row["spec"] for row in family_selected)
        payload = {
            "protocol_version": R5_PROTOCOL_VERSION,
            "status": "pass",
            "track": track,
            "stage": stage,
            "candidate_count": len(selected),
            "candidates": selected,
            "family_top2": True,
            "fixed_ablation_reference_included": REFERENCE_CANDIDATE_ID,
            "outer_test_labels_read": False,
            "confirmation_opened": False,
            "failed_runs": failed_manifests,
        }
        path = output / stage / track / "family_top2.json"
    else:
        reference = next(
            row for row in rows if row["spec"]["candidate_id"] == REFERENCE_CANDIDATE_ID
        )
        eligible: list[dict[str, Any]] = []
        for row in rows:
            delta = {
                head: row["metrics"]["heads"][head]["auprc"]
                - reference["metrics"]["heads"][head]["auprc"]
                for head in ("ge5", "ge10")
            }
            row["delta_vs_reference"] = delta
            row["screening_gate"] = bool(
                row is reference
                or (max(delta.values()) >= 0.001 and min(delta.values()) >= -0.002)
            )
            if row["screening_gate"]:
                eligible.append(row)
        eligible.sort(
            key=lambda row: (
                row["metrics"]["joint_selection_score"],
                min(row["metrics"]["heads"][head]["normalized_ap"] for head in ("ge5", "ge10")),
                row["spec"]["candidate_id"],
            ),
            reverse=True,
        )
        selected_rows: list[dict[str, Any]] = []
        family_counts: dict[str, int] = {}
        for row in eligible:
            family = row["spec"]["family"]
            if family_counts.get(family, 0) >= 2:
                continue
            selected_rows.append(row)
            family_counts[family] = family_counts.get(family, 0) + 1
            if len(selected_rows) == 3:
                break
        if not selected_rows:
            selected_rows = [reference]
        payload = {
            "protocol_version": R5_PROTOCOL_VERSION,
            "status": "pass",
            "track": track,
            "stage": stage,
            "reference_candidate_id": REFERENCE_CANDIDATE_ID,
            "candidate_count": len(selected_rows),
            "candidates": [
                {
                    "spec": row["spec"],
                    "metrics": row["metrics"],
                    "delta_vs_reference": row.get("delta_vs_reference", {"ge5": 0.0, "ge10": 0.0}),
                }
                for row in selected_rows
            ],
            "maximum_finalists_per_track": 3,
            "screening_gate": "one head delta AP >= +0.001 and other >= -0.002",
            "outer_test_labels_read": False,
            "confirmation_opened": False,
            "failed_runs": failed_manifests,
        }
        path = output / stage / track / "finalists.json"
    _write_json(path, payload, overwrite=overwrite)
    return payload


def _development_prediction_path(output: Path, track: Track, candidate_id: str, seed: int) -> Path:
    return output / "development" / track / candidate_id / f"seed-{seed}.parquet"


def run_development_repeat(
    *,
    repository_root: Path,
    track: Track,
    seed: int,
    report_directory: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Evaluate frozen track finalists on one registered development repeat."""

    if int(seed) not in R5_DEVELOPMENT_SEEDS:
        raise ValueError("r5 development repeat seed is not registered")
    root = Path(repository_root).resolve()
    output = Path(report_directory) if report_directory else root / DEFAULT_REPORT_RELATIVE
    if not output.is_absolute():
        output = root / output
    finalists_path = output / "full_inner" / track / "finalists.json"
    finalists = json.loads(finalists_path.read_text(encoding="utf-8"))["candidates"]
    frame, features = _load_track_frame(root, int(seed), track)
    structure = _selected_structure(root)
    completed = 0
    failed = 0
    for value in finalists:
        spec = _spec_from_dict(value["spec"])
        prediction_path = _development_prediction_path(output, track, spec.candidate_id, int(seed))
        manifest_path = prediction_path.with_suffix(".manifest.json")
        if manifest_path.exists() and not overwrite:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            completed += int(manifest.get("status") == "pass")
            failed += int(manifest.get("status") != "pass")
            continue
        try:
            parts: list[pd.DataFrame] = []
            for outer_fold in range(5):
                train = frame.loc[frame["outer_fold"].ne(outer_fold)]
                test = frame.loc[frame["outer_fold"].eq(outer_fold)]
                p5, p10 = _predict_split(train, test, features, spec, track, structure)
                part = _prediction_frame(
                    test,
                    p5,
                    p10,
                    candidate_id=spec.candidate_id,
                    track=track,
                    stage="development",
                )
                part["split_seed"] = int(seed)
                parts.append(part)
            prediction = pd.concat(parts, ignore_index=True)
            if len(prediction) != len(frame) or prediction["r5_row_id"].duplicated().any():
                raise ValueError("r5 development candidate OOF coverage is invalid")
            metrics = _candidate_metrics(
                prediction,
                prediction["probability_ge5"].to_numpy(float),
                prediction["probability_ge10"].to_numpy(float),
                track,
            )
            metrics["by_outer_fold"] = {
                str(int(fold)): _candidate_metrics(
                    part,
                    part["probability_ge5"].to_numpy(float),
                    part["probability_ge10"].to_numpy(float),
                    track,
                )
                for fold, part in prediction.groupby("outer_fold", sort=True)
            }
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            prediction.to_parquet(prediction_path, index=False)
            metrics_path = prediction_path.with_suffix(".metrics.json")
            _write_json(metrics_path, metrics, overwrite=True)
            _write_json(
                manifest_path,
                {
                    "protocol_version": R5_PROTOCOL_VERSION,
                    "status": "pass",
                    "track": track,
                    "seed": int(seed),
                    "spec": asdict(spec),
                    "evidence_level": "adaptive-development outer OOF",
                    "confirmation_opened": False,
                    "prediction_sha256": sha256_file(prediction_path),
                    "metrics_sha256": sha256_file(metrics_path),
                },
                overwrite=True,
            )
            completed += 1
        except Exception as error:
            _write_json(
                manifest_path,
                {
                    "protocol_version": R5_PROTOCOL_VERSION,
                    "status": "failed",
                    "track": track,
                    "seed": int(seed),
                    "spec": asdict(spec),
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "predictions_opened": False,
                    "confirmation_opened": False,
                },
                overwrite=True,
            )
            failed += 1
    return {"status": "pass" if completed else "failed", "track": track, "seed": int(seed), "completed": completed, "failed": failed}


def _baseline_track_frame(root: Path, track: Track, seed: int) -> pd.DataFrame:
    path = root / BASELINE_RELATIVE / f"seed-{seed}" / "r4_recipe_paired_baseline_oof.parquet"
    baseline = pd.read_parquet(path)
    baseline_track = (
        "multisource_deployable"
        if track == "multisource_deployable"
        else "psyche_d_single_source_research"
    )
    return baseline.loc[baseline["track"].eq(baseline_track)].copy()


def finalize_development(
    *, repository_root: Path, report_directory: Path | None = None, overwrite: bool = False
) -> dict[str, Any]:
    """Pair every development finalist against the matching r4 replay."""

    root = Path(repository_root).resolve()
    output = Path(report_directory) if report_directory else root / DEFAULT_REPORT_RELATIVE
    if not output.is_absolute():
        output = root / output
    tracks: dict[str, Any] = {}
    for track in TRACKS:
        finalists = json.loads(
            (output / "full_inner" / track / "finalists.json").read_text(encoding="utf-8")
        )["candidates"]
        candidate_rows: list[dict[str, Any]] = []
        for value in finalists:
            spec = _spec_from_dict(value["spec"])
            seed_rows: dict[str, Any] = {}
            for seed in R5_DEVELOPMENT_SEEDS:
                prediction_path = _development_prediction_path(output, track, spec.candidate_id, seed)
                manifest_path = prediction_path.with_suffix(".manifest.json")
                if not manifest_path.is_file():
                    raise FileNotFoundError(f"r5 development prediction missing: {prediction_path}")
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("status") != "pass":
                    seed_rows[str(seed)] = {"status": "failed", "manifest": manifest}
                    continue
                prediction = pd.read_parquet(prediction_path)
                baseline = _baseline_track_frame(root, track, seed)
                baseline_wide = baseline.pivot(
                    index="r5_row_id", columns="r5_task_id", values="baseline_probability"
                ).rename(
                    columns={
                        "phq_ge5_current": "baseline_ge5",
                        "phq_ge10_current": "baseline_ge10",
                    }
                )
                paired = prediction.merge(
                    baseline_wide.reset_index(), on="r5_row_id", how="left", validate="one_to_one"
                )
                if paired[["baseline_ge5", "baseline_ge10"]].isna().any().any():
                    raise ValueError("r5 development baseline pairing is incomplete")
                candidate_metrics = _candidate_metrics(
                    paired,
                    paired["probability_ge5"].to_numpy(float),
                    paired["probability_ge10"].to_numpy(float),
                    track,
                )
                baseline_metrics = _candidate_metrics(
                    paired,
                    paired["baseline_ge5"].to_numpy(float),
                    paired["baseline_ge10"].to_numpy(float),
                    track,
                )
                delta = {
                    head: {
                        metric: candidate_metrics["heads"][head][metric]
                        - baseline_metrics["heads"][head][metric]
                        for metric in ("auprc", "participant_auprc", "auroc", "brier", "ece")
                    }
                    for head in ("ge5", "ge10")
                }
                seed_rows[str(seed)] = {
                    "status": "pass",
                    "candidate": candidate_metrics,
                    "baseline": baseline_metrics,
                    "delta": delta,
                }
            successful = [row for row in seed_rows.values() if row["status"] == "pass"]
            if not successful:
                continue
            mean_delta = {
                head: {
                    metric: float(np.mean([row["delta"][head][metric] for row in successful]))
                    for metric in ("auprc", "participant_auprc", "auroc", "brier", "ece")
                }
                for head in ("ge5", "ge10")
            }
            development_gate = bool(
                len(successful) == 3
                and max(mean_delta[head]["auprc"] for head in ("ge5", "ge10")) >= 0.003
                and min(mean_delta[head]["auprc"] for head in ("ge5", "ge10")) >= -0.004
            )
            candidate_rows.append(
                {
                    "spec": asdict(spec),
                    "seeds": seed_rows,
                    "mean_delta": mean_delta,
                    "development_gate": development_gate,
                }
            )
        tracks[track] = {"candidates": candidate_rows}
    payload = {
        "protocol_version": R5_PROTOCOL_VERSION,
        "status": "pass",
        "evidence_level": "adaptive-development paired outer OOF",
        "confirmation_opened": False,
        "tracks": tracks,
    }
    path = output / "development" / "development_summary.json"
    _write_json(path, payload, overwrite=overwrite)
    return payload


__all__ = [
    "finalize_competition_stage",
    "finalize_development",
    "run_competition_candidates",
    "run_development_repeat",
]
