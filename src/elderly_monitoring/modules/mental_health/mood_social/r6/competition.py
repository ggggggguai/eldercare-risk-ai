"""R6-004 checkpointed three-track finite model competition."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from elderly_monitoring.modules.mental_health.mood_social.r4.baseline import ap_context_metrics
from elderly_monitoring.modules.mental_health.mood_social.r4.features import deployable_features_for_signature, route_signature
from elderly_monitoring.modules.mental_health.mood_social.r4.selection import participant_equal_weights
from elderly_monitoring.modules.mental_health.mood_social.r5.objectives import conditional_ordered_heads
from elderly_monitoring.modules.mental_health.mood_social.r5.selection import R5CandidateSpec
from elderly_monitoring.modules.mental_health.mood_social.r6.baseline import load_r6_track_frame, r5_locked_recipe
from elderly_monitoring.modules.mental_health.mood_social.r6.contract import R6_DEVELOPMENT_SEEDS, R6_PROTOCOL_VERSION, sha256_file, write_json
from elderly_monitoring.modules.mental_health.mood_social.r6.data import r6_inner_fold_series
from elderly_monitoring.modules.mental_health.mood_social.r6.objectives import r6_training_weights
from elderly_monitoring.modules.mental_health.mood_social.r6.selection import (
    R6CandidateSpec,
    fit_deepset_model,
    fit_regular_model,
    predict_regular_model,
    r6_candidate_registry,
)


TRACKS = ("psyche_d_single_source_research", "multisource_deployable", "joint_route_recovery")
DEFAULT_REPORT_RELATIVE = Path("reports/mental_health/mood_social/v3.3.3-r6/OPT-V333-R6-004-competition")


def _reference_spec(root: Path, track: str, seed: int) -> R6CandidateSpec:
    recipe_track = "psyche_d_single_source_research" if track == "psyche_d_single_source_research" else "multisource_deployable"
    value = r5_locked_recipe(root)["tracks"][recipe_track]["stable_single_spec"]
    return R6CandidateSpec(f"r5_reference_{track}", track, value["family"], dict(value["params"]), int(seed))  # type: ignore[arg-type]


def _track_frame(root: Path, seed: int, track: str) -> tuple[pd.DataFrame, tuple[str, ...] | None]:
    if track == "psyche_d_single_source_research":
        return load_r6_track_frame(root, seed=seed, track="psyche_d_single_source_research")
    frame, _ = load_r6_track_frame(root, seed=seed, track="multisource_deployable")
    if track == "joint_route_recovery":
        frame = frame.loc[frame["route_pattern"].isin(["101", "111"])].copy()
        frame["feature_signature"] = "joint"
    return frame, None


def _fit_predict_head(train: pd.DataFrame, validation: pd.DataFrame, features: tuple[str, ...], spec: R6CandidateSpec, target: str) -> np.ndarray:
    fit = train.copy(); fit["binary_target"] = fit[target].astype(int)
    weight = r6_training_weights(fit, "capped_class")
    if spec.family == "causal_deepsets":
        model = fit_deepset_model(fit, features, spec, target=target, sample_weight=weight)
        return model.predict_proba(validation, features)
    model = fit_regular_model(fit, features, spec, sample_weight=weight)
    probability = predict_regular_model(model, validation, features)
    if spec.family == "shrinkage_route_experts":
        alpha = float(spec.params["alpha"])
        probability = alpha * probability + (1.0 - alpha) * float(np.average(fit[target], weights=weight))
    return np.clip(probability, 1.0e-6, 1.0 - 1.0e-6)


def predict_candidate(train: pd.DataFrame, validation: pd.DataFrame, features: tuple[str, ...] | None, spec: R6CandidateSpec, track: str) -> tuple[np.ndarray, np.ndarray]:
    p5 = pd.Series(np.nan, index=validation.index, dtype=float)
    conditional = pd.Series(np.nan, index=validation.index, dtype=float)
    signatures = ("psyche",) if track == "psyche_d_single_source_research" else tuple(sorted(validation["feature_signature"].unique()))
    for signature in signatures:
        if signature == "psyche":
            fit = train; score = validation; selected = features
        else:
            fit = train.loc[train["feature_signature"].eq(signature)]; score = validation.loc[validation["feature_signature"].eq(signature)]; selected = deployable_features_for_signature(str(signature))
        if score.empty:
            continue
        if fit.empty or selected is None:
            raise ValueError(f"candidate lacks train support for {signature}")
        p5.loc[score.index] = _fit_predict_head(fit, score, tuple(selected), spec, "phq9_ge5_r3_target")
        conditional_fit = fit.loc[fit["phq9_ge5_r3_target"].eq(1)].copy()
        conditional.loc[score.index] = _fit_predict_head(conditional_fit, score, tuple(selected), spec, "phq9_ge10_r3_target")
    if p5.isna().any() or conditional.isna().any():
        raise ValueError("candidate prediction coverage incomplete")
    return conditional_ordered_heads(p5.to_numpy(float), conditional.to_numpy(float))


def _metric(frame: pd.DataFrame, target: str, probability: np.ndarray) -> dict[str, float]:
    y = frame[target].to_numpy(int)
    natural = ap_context_metrics(y, probability)
    participant = ap_context_metrics(y, probability, sample_weight=participant_equal_weights(frame))
    return {**natural, "participant_auprc": participant["auprc"], "selection_score": 0.65 * natural["auprc"] + 0.35 * participant["auprc"]}


def _metrics(frame: pd.DataFrame, p5: np.ndarray, p10: np.ndarray) -> dict[str, Any]:
    heads = {head: _metric(frame, f"phq9_{head}_r3_target", p) for head, p in (("ge5", p5), ("ge10", p10))}
    return {"heads": heads, "joint_score": float(np.mean([heads[h]["selection_score"] for h in ("ge5", "ge10")])), "monotonic_violation_count": int(np.sum(p10 > p5 + 1e-12)), "coverage": float(np.isfinite(p5).mean() * np.isfinite(p10).mean())}


def _prediction_frame(frame: pd.DataFrame, p5: np.ndarray, p10: np.ndarray, spec: R6CandidateSpec, stage: str, seed: int) -> pd.DataFrame:
    output = frame[["r6_row_id", "global_participant_id", "dataset_id", "route_pattern", "outer_fold", "phq9_ge5_r3_target", "phq9_ge10_r3_target"]].copy()
    output["candidate_id"] = spec.candidate_id; output["track"] = spec.track; output["stage"] = stage; output["seed"] = seed
    output["probability_ge5"] = p5; output["probability_ge10"] = p10
    return output


def _run_spec(frame: pd.DataFrame, features: tuple[str, ...] | None, spec: R6CandidateSpec, track: str, stage: str, inner: pd.Series | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    if stage == "screening":
        assert inner is not None
        train = frame.loc[inner.ne(0)]; validation = frame.loc[inner.eq(0)]
        p5, p10 = predict_candidate(train, validation, features, spec, track)
        return _prediction_frame(validation, p5, p10, spec, stage, spec.seed), _metrics(validation, p5, p10)
    p5 = pd.Series(np.nan, index=frame.index, dtype=float); p10 = pd.Series(np.nan, index=frame.index, dtype=float)
    fold_series = inner if stage == "full_inner" else frame["outer_fold"]
    assert fold_series is not None
    for fold in sorted(fold_series.dropna().astype(int).unique()):
        train = frame.loc[fold_series.ne(fold)]; validation = frame.loc[fold_series.eq(fold)]
        a, b = predict_candidate(train, validation, features, spec, track)
        p5.loc[validation.index] = a; p10.loc[validation.index] = b
    if p5.isna().any() or p10.isna().any():
        raise ValueError(f"{stage} OOF incomplete")
    return _prediction_frame(frame, p5.to_numpy(float), p10.to_numpy(float), spec, stage, spec.seed), _metrics(frame, p5.to_numpy(float), p10.to_numpy(float))


def _save_run(root: Path, directory: Path, spec: R6CandidateSpec, prediction: pd.DataFrame, metrics: dict[str, Any], *, overwrite: bool) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    prediction_path = directory / "oof.parquet"; metrics_path = directory / "metrics.json"; manifest_path = directory / "manifest.json"
    if prediction_path.exists() and not overwrite:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    prediction.to_parquet(prediction_path, index=False); write_json(metrics_path, metrics, overwrite=True)
    manifest = {"protocol_version": R6_PROTOCOL_VERSION, "status": "pass", "spec": asdict(spec), "confirmation_opened": False, "outer_test_used_for_selection": False, "oof_sha256": sha256_file(prediction_path), "metrics_sha256": sha256_file(metrics_path)}
    write_json(manifest_path, manifest, overwrite=True); return manifest


def run_r6_competition(*, repository_root: Path, output_directory: Path | None = None, overwrite: bool = False) -> dict[str, Any]:
    root = Path(repository_root).resolve(); output = Path(output_directory) if output_directory else root / DEFAULT_REPORT_RELATIVE
    if not output.is_absolute(): output = root / output
    seed = R6_DEVELOPMENT_SEEDS[0]; registry = r6_candidate_registry(seed)
    screening_rows: list[dict[str, Any]] = []; failures: list[dict[str, str]] = []; track_references: dict[str, dict[str, Any]] = {}
    for track in TRACKS:
        frame, features = _track_frame(root, seed, track); train_frame = frame.loc[frame["outer_fold"].ne(0)].copy(); inner = r6_inner_fold_series(train_frame, 0).astype(int)
        candidates = [spec for spec in registry if spec.track == track]
        reference = _reference_spec(root, track, seed)
        for spec in [reference, *candidates]:
            directory = output / "screening" / track / spec.candidate_id
            try:
                if (directory / "manifest.json").is_file() and not overwrite:
                    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
                else:
                    prediction, metrics = _run_spec(train_frame, features, spec, track, "screening", inner)
                    _save_run(root, directory, spec, prediction, metrics, overwrite=overwrite)
                row = {"track": track, "spec": asdict(spec), "metrics": metrics}
                if spec.candidate_id.startswith("r5_reference_"): track_references[track] = row
                else: screening_rows.append(row)
            except Exception as error:
                failures.append({"stage": "screening", "track": track, "candidate_id": spec.candidate_id, "error_type": type(error).__name__, "message": str(error)})
    if len(screening_rows) + len([f for f in failures if f["stage"] == "screening" and not f["candidate_id"].startswith("r5_reference_")]) != 46:
        raise RuntimeError("r6 first-round candidate accounting drifted")
    allocations = {"psyche_d_single_source_research": 4, "multisource_deployable": 4, "joint_route_recovery": 2}
    full_specs: list[R6CandidateSpec] = []
    for track, maximum in allocations.items():
        reference = track_references[track]["metrics"]
        ranked = []
        for row in [value for value in screening_rows if value["track"] == track]:
            delta = {h: row["metrics"]["heads"][h]["auprc"] - reference["heads"][h]["auprc"] for h in ("ge5", "ge10")}
            row["delta_vs_reference"] = delta; row["screening_gate"] = bool(max(delta.values()) >= 0.001 or (min(delta.values()) >= -0.002 and row["metrics"]["joint_score"] >= reference["joint_score"]))
            if row["screening_gate"]: ranked.append(row)
        ranked.sort(key=lambda row: (row["metrics"]["joint_score"], min(row["delta_vs_reference"].values()), row["spec"]["candidate_id"]), reverse=True)
        if len(ranked) < maximum:
            used = {row["spec"]["candidate_id"] for row in ranked}
            diagnostic_only = sorted(
                [row for row in screening_rows if row["track"] == track and row["spec"]["candidate_id"] not in used],
                key=lambda row: row["metrics"]["joint_score"], reverse=True,
            )
            ranked.extend(diagnostic_only[: maximum - len(ranked)])
        for row in ranked[:maximum]: full_specs.append(R6CandidateSpec(**row["spec"]))
    # Fill an under-subscribed track allocation by score, while retaining the
    # global maximum of ten complete inner candidates.
    if len(full_specs) < 10:
        used = {spec.candidate_id for spec in full_specs}
        remaining = sorted([row for row in screening_rows if row["spec"]["candidate_id"] not in used], key=lambda row: row["metrics"]["joint_score"], reverse=True)
        for row in remaining[: 10 - len(full_specs)]: full_specs.append(R6CandidateSpec(**row["spec"]))
    full_rows: list[dict[str, Any]] = []
    for spec in full_specs:
        frame, features = _track_frame(root, seed, spec.track); train_frame = frame.loc[frame["outer_fold"].ne(0)].copy(); inner = r6_inner_fold_series(train_frame, 0).astype(int)
        directory = output / "full_inner" / spec.track / spec.candidate_id
        try:
            if (directory / "manifest.json").is_file() and not overwrite:
                metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
            else:
                prediction, metrics = _run_spec(train_frame, features, spec, spec.track, "full_inner", inner)
                _save_run(root, directory, spec, prediction, metrics, overwrite=overwrite)
            full_rows.append({"track": spec.track, "spec": asdict(spec), "metrics": metrics})
        except Exception as error:
            failures.append({"stage": "full_inner", "track": spec.track, "candidate_id": spec.candidate_id, "error_type": type(error).__name__, "message": str(error)})
    finalists: list[R6CandidateSpec] = []
    for track in TRACKS:
        rows = [row for row in full_rows if row["track"] == track]
        rows.sort(key=lambda row: row["metrics"]["joint_score"], reverse=True)
        # Two candidates per track are sufficient to test residual diversity
        # and fusion while honoring the frozen <=6 ceiling and stop rule.
        finalists.extend(R6CandidateSpec(**row["spec"]) for row in rows[:2])
    repeated_rows: list[dict[str, Any]] = []
    for seed_value in R6_DEVELOPMENT_SEEDS:
        for original in finalists:
            spec = R6CandidateSpec(original.candidate_id, original.track, original.family, original.params, int(seed_value))
            frame, features = _track_frame(root, seed_value, spec.track); directory = output / "repeated_outer" / f"seed-{seed_value}" / spec.track / spec.candidate_id
            try:
                if (directory / "manifest.json").is_file() and not overwrite:
                    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
                else:
                    prediction, metrics = _run_spec(frame, features, spec, spec.track, "repeated_outer")
                    _save_run(root, directory, spec, prediction, metrics, overwrite=overwrite)
                repeated_rows.append({"seed": seed_value, "track": spec.track, "spec": asdict(spec), "metrics": metrics})
            except Exception as error:
                failures.append({"stage": "repeated_outer", "track": spec.track, "candidate_id": spec.candidate_id, "error_type": type(error).__name__, "message": str(error)})
    output.mkdir(parents=True, exist_ok=True)
    screening_path = output / "screening_results.json"; full_path = output / "full_inner_results.json"; repeated_path = output / "repeated_outer_results.json"; finalist_path = output / "finalists.json"; failure_path = output / "failed_runs.json"
    write_json(screening_path, {"protocol_version": R6_PROTOCOL_VERSION, "candidate_count": 46, "rows": screening_rows, "references": track_references}, overwrite=overwrite)
    write_json(full_path, {"protocol_version": R6_PROTOCOL_VERSION, "candidate_count": len(full_specs), "rows": full_rows}, overwrite=overwrite)
    write_json(repeated_path, {"protocol_version": R6_PROTOCOL_VERSION, "rows": repeated_rows}, overwrite=overwrite)
    write_json(finalist_path, {"protocol_version": R6_PROTOCOL_VERSION, "status": "development-finalists", "candidates": [asdict(spec) for spec in finalists], "maximum_per_track": 6, "confirmation_opened": False}, overwrite=overwrite)
    write_json(failure_path, failures, overwrite=overwrite)
    manifest = {"protocol_version": R6_PROTOCOL_VERSION, "status": "pass", "stage": "R6-004", "first_round_candidates": 46, "full_inner_candidates": len(full_specs), "finalists": len(finalists), "repeated_outer_completed": len(repeated_rows), "failed_runs": len(failures), "confirmation_opened": False, "artifacts": {p.name: sha256_file(p) for p in (screening_path, full_path, repeated_path, finalist_path, failure_path)}}
    write_json(output / "artifact_manifest.json", manifest, overwrite=overwrite); return manifest


__all__ = ["predict_candidate", "run_r6_competition"]
