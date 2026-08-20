"""R6-003 ordered objectives and capped training-weight comparison."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRanker, LGBMRegressor
from sklearn.impute import SimpleImputer

from elderly_monitoring.modules.mental_health.mood_social.r4.baseline import ap_context_metrics
from elderly_monitoring.modules.mental_health.mood_social.r4.selection import participant_equal_weights
from elderly_monitoring.modules.mental_health.mood_social.r5.objectives import conditional_ordered_heads, ordinal_heads, phq_severity_class
from elderly_monitoring.modules.mental_health.mood_social.r5.selection import R5CandidateSpec, fit_r5_model, predict_r5_model
from elderly_monitoring.modules.mental_health.mood_social.r6.baseline import load_r6_track_frame, r5_locked_recipe
from elderly_monitoring.modules.mental_health.mood_social.r6.contract import R6_DEVELOPMENT_SEEDS, R6_PROTOCOL_VERSION, sha256_file, write_json
from elderly_monitoring.modules.mental_health.mood_social.r6.data import r6_inner_fold_series


Objective = Literal["conditional_ordered", "ordinal_five_class", "huber_score_aux", "pairwise_ranking_aux"]
WeightScheme = Literal["participant_equal", "capped_class", "source_pow025_capped"]
OBJECTIVES: tuple[Objective, ...] = ("conditional_ordered", "ordinal_five_class", "huber_score_aux", "pairwise_ranking_aux")
WEIGHT_SCHEMES: tuple[WeightScheme, ...] = ("participant_equal", "capped_class", "source_pow025_capped")
DEFAULT_REPORT_RELATIVE = Path("reports/mental_health/mood_social/v3.3.3-r6/OPT-V333-R6-003-objectives")


def r6_training_weights(frame: pd.DataFrame, scheme: WeightScheme) -> np.ndarray:
    if scheme not in WEIGHT_SCHEMES:
        raise ValueError(f"unknown r6 weight scheme: {scheme}")
    base = participant_equal_weights(frame)
    modifier = np.ones(len(frame), dtype=float)
    if scheme == "capped_class":
        severity = phq_severity_class(frame["phq9_score_r3_target"].to_numpy(float))
        counts = pd.Series(severity).value_counts().to_dict()
        raw = np.asarray([(len(frame) / max(5 * counts[int(value)], 1)) ** 0.25 for value in severity], float)
        modifier = np.clip(raw, 0.75, 1.50)
    elif scheme == "source_pow025_capped":
        counts = frame["dataset_id"].value_counts().to_dict()
        raw = np.asarray([(len(frame) / max(counts[str(value)], 1)) ** 0.25 for value in frame["dataset_id"]], float)
        modifier = np.clip(raw, 0.75, 1.50)
    result = base * modifier
    result *= len(result) / result.sum()
    if not np.isfinite(result).all() or (result <= 0).any() or result.max() / result.min() > 20:
        raise ValueError("invalid capped r6 weights")
    return result


def _binary(train: pd.DataFrame, validation: pd.DataFrame, features: tuple[str, ...], target: str, spec: R5CandidateSpec, scheme: WeightScheme) -> np.ndarray:
    fit = train.copy()
    fit["binary_target"] = fit[target].astype(int)
    model = fit_r5_model(fit, features, spec, sample_weight=r6_training_weights(fit, scheme))
    return predict_r5_model(model, validation, features)


def _score_heads(score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Strictly monotone score-to-risk maps. They preserve ranking and guarantee
    # P(PHQ>=10) <= P(PHQ>=5); calibration is deferred to R6-005.
    value = np.asarray(score, float)
    p5 = 1.0 / (1.0 + np.exp(-np.clip((value - 5.0) / 3.0, -20, 20)))
    p10 = 1.0 / (1.0 + np.exp(-np.clip((value - 10.0) / 3.0, -20, 20)))
    return p5, p10


def _predict_objective(train: pd.DataFrame, validation: pd.DataFrame, features: tuple[str, ...], spec: R5CandidateSpec, objective: Objective, scheme: WeightScheme) -> tuple[np.ndarray, np.ndarray]:
    weight = r6_training_weights(train, scheme)
    if objective == "conditional_ordered":
        p5 = _binary(train, validation, features, "phq9_ge5_r3_target", spec, scheme)
        conditional_train = train.loc[train["phq9_ge5_r3_target"].eq(1)].copy()
        conditional = _binary(conditional_train, validation, features, "phq9_ge10_r3_target", spec, scheme)
        return conditional_ordered_heads(p5, conditional)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    x_train = imputer.fit_transform(train.loc[:, features])
    x_validation = imputer.transform(validation.loc[:, features])
    if objective == "ordinal_five_class":
        model = LGBMClassifier(
            objective="multiclass", num_class=5, n_estimators=300, learning_rate=0.03,
            num_leaves=15, min_child_samples=40, reg_lambda=7.0, reg_alpha=0.7,
            subsample=0.85, colsample_bytree=0.85, verbosity=-1, n_jobs=1,
            random_state=spec.seed,
        )
        model.fit(x_train, phq_severity_class(train["phq9_score_r3_target"].to_numpy(float)), sample_weight=weight)
        predicted = np.asarray(model.predict_proba(x_validation), float)
        aligned = np.zeros((len(validation), 5), float)
        aligned[:, np.asarray(model.classes_, int)] = predicted
        return ordinal_heads(aligned)
    if objective == "huber_score_aux":
        model = LGBMRegressor(
            objective="huber", alpha=0.8, n_estimators=350, learning_rate=0.025,
            num_leaves=15, min_child_samples=40, reg_lambda=8.0, reg_alpha=0.8,
            verbosity=-1, n_jobs=1, random_state=spec.seed,
        )
        model.fit(x_train, train["phq9_score_r3_target"].to_numpy(float), sample_weight=weight)
        return _score_heads(np.clip(model.predict(x_validation), 0, 27))
    ordered = train.assign(_position=np.arange(len(train))).sort_values(["global_participant_id", "nominal_month"], kind="stable")
    order = ordered["_position"].to_numpy(int)
    groups = ordered.groupby("global_participant_id", sort=False).size().to_numpy(int)
    model = LGBMRanker(
        objective="lambdarank", label_gain=list(range(5)), n_estimators=300,
        learning_rate=0.03, num_leaves=15, min_child_samples=30,
        reg_lambda=8.0, reg_alpha=0.8, verbosity=-1, n_jobs=1,
        random_state=spec.seed,
    )
    model.fit(x_train[order], phq_severity_class(train["phq9_score_r3_target"].to_numpy(float))[order], group=groups, sample_weight=weight[order])
    rank_score = np.asarray(model.predict(x_validation), float)
    # Map rank score to the train PHQ scale without using validation targets.
    train_rank = np.asarray(model.predict(x_train), float)
    if np.std(train_rank) <= 1.0e-9:
        score = np.repeat(float(train["phq9_score_r3_target"].median()), len(validation))
    else:
        coefficient = np.polyfit(train_rank, train["phq9_score_r3_target"].to_numpy(float), 1)
        score = np.polyval(coefficient, rank_score)
    return _score_heads(np.clip(score, 0, 27))


def _oof(frame: pd.DataFrame, features: tuple[str, ...], inner: pd.Series, spec: R5CandidateSpec, objective: Objective, scheme: WeightScheme) -> tuple[np.ndarray, np.ndarray]:
    p5 = pd.Series(np.nan, index=frame.index, dtype=float)
    p10 = pd.Series(np.nan, index=frame.index, dtype=float)
    for fold in sorted(inner.astype(int).unique()):
        train = frame.loc[inner.ne(fold)].copy()
        validation = frame.loc[inner.eq(fold)].copy()
        a, b = _predict_objective(train, validation, features, spec, objective, scheme)
        p5.loc[validation.index] = a
        p10.loc[validation.index] = b
    if p5.isna().any() or p10.isna().any() or (p10 > p5 + 1.0e-12).any():
        raise ValueError("r6 objective OOF is incomplete or unordered")
    return p5.to_numpy(float), p10.to_numpy(float)


def _metric(frame: pd.DataFrame, target: str, probability: np.ndarray) -> dict[str, float]:
    y = frame[target].to_numpy(int)
    natural = ap_context_metrics(y, probability)
    participant = ap_context_metrics(y, probability, sample_weight=participant_equal_weights(frame))
    return {**natural, "participant_auprc": participant["auprc"], "selection_score": 0.65 * natural["auprc"] + 0.35 * participant["auprc"]}


def run_r6_objective_comparison(*, repository_root: Path, output_directory: Path | None = None, overwrite: bool = False) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    output = Path(output_directory) if output_directory else root / DEFAULT_REPORT_RELATIVE
    if not output.is_absolute():
        output = root / output
    seed = R6_DEVELOPMENT_SEEDS[0]
    frame, features = load_r6_track_frame(root, seed=seed, track="psyche_d_single_source_research")
    assert features is not None
    frame = frame.loc[frame["outer_fold"].ne(0)].copy()
    inner = r6_inner_fold_series(frame, 0).astype(int)
    spec_value = r5_locked_recipe(root)["tracks"]["psyche_d_single_source_research"]["stable_single_spec"]
    spec = R5CandidateSpec(str(spec_value["candidate_id"]), str(spec_value["family"]), dict(spec_value["params"]), int(spec_value["seed"]))  # type: ignore[arg-type]
    rows: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []
    for objective in OBJECTIVES:
        for scheme in WEIGHT_SCHEMES:
            try:
                p5, p10 = _oof(frame, features, inner, spec, objective, scheme)
            except Exception as error:
                failures.append({"objective": objective, "weight_scheme": scheme, "error_type": type(error).__name__, "message": str(error)})
                continue
            metrics = {head: _metric(frame, f"phq9_{head}_r3_target", probability) for head, probability in (("ge5", p5), ("ge10", p10))}
            rows.append({"objective": objective, "weight_scheme": scheme, "metrics": metrics, "joint_score": float(np.mean([metrics[h]["selection_score"] for h in ("ge5", "ge10")])), "monotonic_violation_count": 0})
            part = frame[["r6_row_id", "global_participant_id", "nominal_month"]].copy()
            part["inner_fold"] = inner.to_numpy(int); part["objective"] = objective; part["weight_scheme"] = scheme
            part["target_ge5"] = frame["phq9_ge5_r3_target"].to_numpy(int); part["target_ge10"] = frame["phq9_ge10_r3_target"].to_numpy(int)
            part["probability_ge5"] = p5; part["probability_ge10"] = p10
            predictions.append(part)
    if not rows:
        raise RuntimeError("all r6 objectives failed")
    reference = next(row for row in rows if row["objective"] == "conditional_ordered" and row["weight_scheme"] == "participant_equal")
    for row in rows:
        row["delta_vs_reference"] = {h: row["metrics"][h]["auprc"] - reference["metrics"][h]["auprc"] for h in ("ge5", "ge10")}
        row["screening_gate"] = bool(max(row["delta_vs_reference"].values()) >= 0.001 or (min(row["delta_vs_reference"].values()) >= -0.002 and row["joint_score"] >= reference["joint_score"]))
    eligible = [row for row in rows if row["screening_gate"]]
    selected = max(eligible or [reference], key=lambda row: row["joint_score"])
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "objective_weight_comparison.json"; prediction_path = output / "objective_inner_oof.parquet"; selected_path = output / "selected_objective.json"; failure_path = output / "failed_runs.json"
    write_json(result_path, {"protocol_version": R6_PROTOCOL_VERSION, "status": "pass", "runs": rows}, overwrite=overwrite)
    pd.concat(predictions, ignore_index=True).to_parquet(prediction_path, index=False)
    write_json(selected_path, {"protocol_version": R6_PROTOCOL_VERSION, "status": "development-selected", "objective": selected["objective"], "weight_scheme": selected["weight_scheme"], "joint_score": selected["joint_score"], "delta_vs_reference": selected["delta_vs_reference"], "monotonic_violation_count": 0, "phq_used_as_target_only": True, "confirmation_opened": False}, overwrite=overwrite)
    write_json(failure_path, failures, overwrite=overwrite)
    manifest = {"protocol_version": R6_PROTOCOL_VERSION, "status": "pass", "stage": "R6-003", "completed_runs": len(rows), "failed_runs": len(failures), "selected": {"objective": selected["objective"], "weight_scheme": selected["weight_scheme"]}, "confirmation_opened": False, "artifacts": {p.name: sha256_file(p) for p in (result_path, prediction_path, selected_path, failure_path)}}
    write_json(output / "artifact_manifest.json", manifest, overwrite=overwrite)
    return manifest


__all__ = ["OBJECTIVES", "WEIGHT_SCHEMES", "r6_training_weights", "run_r6_objective_comparison"]
