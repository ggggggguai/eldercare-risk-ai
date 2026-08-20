"""Strict outer-train-only fusion, calibration and workpoints for r5.

The module intentionally rebuilds candidate inner OOF predictions inside every
outer fold.  No outer-test label is used to choose a member, fusion transform,
calibrator or operating threshold.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from elderly_monitoring.modules.mental_health.mood_social.r4.baseline import (
    ap_context_metrics,
)
from elderly_monitoring.modules.mental_health.mood_social.r4.calibration import (
    CalibrationMethod,
    FittedCalibration,
    fit_calibration,
)
from elderly_monitoring.modules.mental_health.mood_social.r4.selection import (
    participant_equal_weights,
)
from elderly_monitoring.modules.mental_health.mood_social.r5.competition import (
    BASELINE_RELATIVE,
    DEFAULT_REPORT_RELATIVE as COMPETITION_RELATIVE,
    TRACKS,
    Track,
    _baseline_track_frame,
    _candidate_metrics,
    _load_track_frame,
    _predict_split,
    _prediction_frame,
    _selected_structure,
    _spec_from_dict,
)
from elderly_monitoring.modules.mental_health.mood_social.r5.contract import (
    R5_CONFIRMATION_SEED,
    R5_DEVELOPMENT_SEEDS,
    R5_PROTOCOL_VERSION,
    sha256_file,
)
from elderly_monitoring.modules.mental_health.mood_social.r5.confirmation import (
    confirmation_state,
    verify_locked_confirmation_recipe,
)
from elderly_monitoring.modules.mental_health.mood_social.r5.data import (
    r5_inner_fold_series,
)
from elderly_monitoring.modules.mental_health.mood_social.r5.objectives import (
    project_independent_heads,
)
from elderly_monitoring.modules.mental_health.mood_social.r5.selection import (
    R5CandidateSpec,
)


EPSILON = 1.0e-6
CALIBRATION_METHODS: tuple[CalibrationMethod, ...] = (
    "none",
    "platt",
    "beta",
    "isotonic",
)
DEFAULT_REPORT_RELATIVE = Path(
    "reports/mental_health/mood_social/v3.3.3-r5/OPT-V333-R5-005-fusion"
)
MAX_RESIDUAL_CORRELATION = 0.985
STABLE_SINGLE_DIRECTORY = "development_stable_single"
ADAPTIVE_DIRECTORY = "development_adaptive_top3"


def _write_json(path: Path, value: Any, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite r5 fusion artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _clip(value: Sequence[float]) -> np.ndarray:
    return np.clip(np.asarray(value, dtype=float), EPSILON, 1.0 - EPSILON)


def _logit(value: Sequence[float]) -> np.ndarray:
    probability = _clip(value)
    return np.log(probability / (1.0 - probability))


def _sigmoid(value: Sequence[float]) -> np.ndarray:
    array = np.clip(np.asarray(value, dtype=float), -30.0, 30.0)
    return _clip(1.0 / (1.0 + np.exp(-array)))


def select_train_only_workpoints(
    target: Sequence[int], probability: Sequence[float]
) -> dict[str, Any]:
    """Select the frozen workpoints in O(n log n), including every unique score."""

    y = np.asarray(target, dtype=int)
    score = np.asarray(probability, dtype=float)
    if len(y) != len(score) or np.unique(y).size != 2 or not np.isfinite(score).all():
        raise ValueError("r5 workpoint selection requires finite binary observations")
    # Preserve the frozen candidate set exactly, including 501 quantile
    # thresholds that can choose a different numeric boundary while inducing
    # the same classification as a nearby observed score.
    thresholds = np.unique(
        np.concatenate(
            [
                np.asarray([0.0, 1.0]),
                score,
                np.quantile(score, np.linspace(0.0, 1.0, 501)),
            ]
        )
    )
    order = np.argsort(score, kind="stable")
    ordered_y = y[order]
    ordered_score = score[order]
    positive_suffix = np.r_[np.cumsum(ordered_y[::-1])[::-1], 0].astype(float)
    first_selected = np.searchsorted(ordered_score, thresholds, side="left")
    tp = positive_suffix[first_selected]
    selected_count = len(score) - first_selected
    fp = selected_count.astype(float) - tp
    positive = float(y.sum())
    negative = float(len(y) - y.sum())
    fn = positive - tp
    tn = negative - fp
    rows: list[dict[str, float]] = []
    for threshold, true_positive, false_positive, false_negative, true_negative in zip(
        thresholds, tp, fp, fn, tn
    ):
        sensitivity = true_positive / positive
        specificity = true_negative / negative
        precision = true_positive / max(true_positive + false_positive, 1.0)
        f1 = 2.0 * precision * sensitivity / max(precision + sensitivity, 1.0e-15)
        rows.append(
            {
                "threshold": float(threshold),
                "sensitivity": float(sensitivity),
                "specificity": float(specificity),
                "precision": float(precision),
                "f1": float(f1),
                "alert_rate": float((true_positive + false_positive) / len(y)),
            }
        )
    points = rows
    competition = max(
        points, key=lambda row: (row["f1"], row["precision"], row["threshold"])
    )
    safety = max(
        (row for row in points if row["sensitivity"] >= 0.80),
        key=lambda row: (row["precision"], row["specificity"], row["threshold"]),
    )
    sensitivity_at_specificity = {}
    for target_specificity in (0.80, 0.90):
        sensitivity_at_specificity[f"{target_specificity:.2f}"] = max(
            (row for row in points if row["specificity"] >= target_specificity),
            key=lambda row: (row["sensitivity"], row["precision"], -row["threshold"]),
        )
    return {
        "competition": competition,
        "safety": safety,
        "sensitivity_at_specificity": sensitivity_at_specificity,
        "threshold_candidate_count": len(points),
        "implementation": "exact_unique_score_sort_cumulative",
    }


def _ecdf_reference(reference: np.ndarray, value: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.asarray(reference, dtype=float))
    rank = np.searchsorted(ordered, np.asarray(value, dtype=float), side="right")
    return _clip((rank + 0.5) / (len(ordered) + 1.0))


def select_low_correlation_members(
    frame: pd.DataFrame,
    predictions: Mapping[str, tuple[np.ndarray, np.ndarray]],
    specs: Mapping[str, R5CandidateSpec],
    track: Track,
    *,
    maximum_members: int = 3,
    maximum_correlation: float = MAX_RESIDUAL_CORRELATION,
) -> tuple[list[str], dict[str, Any]]:
    """Choose the best member then add only materially distinct residuals."""

    target5 = frame["phq9_ge5_r3_target"].to_numpy(float)
    target10 = frame["phq9_ge10_r3_target"].to_numpy(float)
    scores = {
        candidate_id: _candidate_metrics(frame, values[0], values[1], track)[
            "joint_selection_score"
        ]
        for candidate_id, values in predictions.items()
    }
    ranked = sorted(scores, key=lambda key: (scores[key], key), reverse=True)
    residual = {
        key: np.concatenate(
            [target5 - predictions[key][0], target10 - predictions[key][1]]
        )
        for key in ranked
    }
    correlations: dict[str, float] = {}
    selected: list[str] = []
    for candidate_id in ranked:
        if not selected:
            selected.append(candidate_id)
            continue
        pair_values = []
        for previous in selected:
            correlation = float(np.corrcoef(residual[candidate_id], residual[previous])[0, 1])
            if not np.isfinite(correlation):
                correlation = 1.0
            correlations[f"{previous}__{candidate_id}"] = correlation
            pair_values.append(abs(correlation))
        # A second model from the same family is permitted only when its errors
        # genuinely differ; family identity alone is never treated as evidence.
        if max(pair_values) <= float(maximum_correlation):
            selected.append(candidate_id)
        if len(selected) >= int(maximum_members):
            break
    return selected, {
        "selection_scores": {key: float(value) for key, value in scores.items()},
        "ranked_candidates": ranked,
        "selected_members": selected,
        "selected_families": [specs[key].family for key in selected],
        "maximum_absolute_residual_correlation": float(maximum_correlation),
        "pairwise_residual_correlations": correlations,
    }


def _fuse(
    kind: str,
    weights: np.ndarray,
    inner_values: Sequence[np.ndarray],
    outer_values: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    if kind == "probability":
        return (
            _clip(sum(weight * value for weight, value in zip(weights, inner_values))),
            _clip(sum(weight * value for weight, value in zip(weights, outer_values))),
        )
    if kind == "logit":
        return (
            _sigmoid(sum(weight * _logit(value) for weight, value in zip(weights, inner_values))),
            _sigmoid(sum(weight * _logit(value) for weight, value in zip(weights, outer_values))),
        )
    if kind == "rank_normalized":
        inner_rank = [
            _ecdf_reference(reference, reference) for reference in inner_values
        ]
        outer_rank = [
            _ecdf_reference(reference, value)
            for reference, value in zip(inner_values, outer_values)
        ]
        return (
            _clip(sum(weight * value for weight, value in zip(weights, inner_rank))),
            _clip(sum(weight * value for weight, value in zip(weights, outer_rank))),
        )
    raise ValueError(f"unknown r5 fusion kind: {kind}")


def _weight_grid(member_count: int) -> list[np.ndarray]:
    if member_count == 1:
        return [np.asarray([1.0])]
    if member_count == 2:
        return [np.asarray([step / 20.0, 1.0 - step / 20.0]) for step in range(21)]
    # Coarse three-way simplex is the second and final fusion batch.
    result: list[np.ndarray] = []
    for first in range(5):
        for second in range(5 - first):
            third = 4 - first - second
            result.append(np.asarray([first, second, third], dtype=float) / 4.0)
    return result


def select_fusion(
    frame: pd.DataFrame,
    inner_predictions: Mapping[str, tuple[np.ndarray, np.ndarray]],
    outer_predictions: Mapping[str, tuple[np.ndarray, np.ndarray]],
    specs: Mapping[str, R5CandidateSpec],
    track: Track,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], dict[str, Any]]:
    members, correlation_audit = select_low_correlation_members(
        frame, inner_predictions, specs, track
    )
    candidates: list[dict[str, Any]] = []
    best_payload: tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] | None = None
    best_key: tuple[float, float, str] | None = None
    kinds = ("probability",) if len(members) == 1 else (
        "probability",
        "logit",
        "rank_normalized",
    )
    for kind in kinds:
        for weights in _weight_grid(len(members)):
            inner5, outer5 = _fuse(
                kind,
                weights,
                [inner_predictions[key][0] for key in members],
                [outer_predictions[key][0] for key in members],
            )
            inner10, outer10 = _fuse(
                kind,
                weights,
                [inner_predictions[key][1] for key in members],
                [outer_predictions[key][1] for key in members],
            )
            inner5, inner10 = project_independent_heads(inner5, inner10)
            outer5, outer10 = project_independent_heads(outer5, outer10)
            metrics = _candidate_metrics(frame, inner5, inner10, track)
            candidate_id = f"{kind}__" + "__".join(
                f"{member}:{weight:.2f}" for member, weight in zip(members, weights)
            )
            row = {
                "candidate_id": candidate_id,
                "kind": kind,
                "members": members,
                "weights": [float(value) for value in weights],
                "metrics": metrics,
            }
            candidates.append(row)
            key = (
                float(metrics["joint_selection_score"]),
                -float(np.sum(np.abs(weights - 1.0 / len(weights)))),
                candidate_id,
            )
            if best_key is None or key > best_key:
                best_key = key
                best_payload = ((inner5, inner10), (outer5, outer10))
                selected = row
    assert best_payload is not None
    candidates.sort(
        key=lambda row: (row["metrics"]["joint_selection_score"], row["candidate_id"]),
        reverse=True,
    )
    audit = {
        **correlation_audit,
        "fusion_batches": ["pairwise_0.05_grid", "optional_three_way_0.25_simplex"],
        "candidate_count": len(candidates),
        "selected": selected,
        "top_candidates": candidates[:10],
        "selection_used_outer_labels": False,
    }
    return best_payload[0], best_payload[1], audit


def _crossfit_calibration(
    frame: pd.DataFrame,
    probability: np.ndarray,
    target_column: str,
    method: CalibrationMethod,
) -> np.ndarray:
    calibrated = pd.Series(np.nan, index=frame.index, dtype=float)
    raw = pd.Series(np.asarray(probability, dtype=float), index=frame.index)
    for validation_fold in sorted(frame["inner_fold"].astype(int).unique()):
        train_mask = frame["inner_fold"].astype(int).ne(validation_fold)
        validation_mask = ~train_mask
        fit = frame.loc[train_mask].copy()
        fit["binary_target"] = fit[target_column].astype(int)
        model = fit_calibration(
            fit,
            raw.loc[train_mask].to_numpy(float),
            method,
        )
        calibrated.loc[validation_mask] = model.predict(
            raw.loc[validation_mask].to_numpy(float)
        )
    if calibrated.isna().any():
        raise ValueError("r5 cross-fitted calibration coverage is incomplete")
    return calibrated.loc[frame.index].to_numpy(float)


def select_calibration(
    frame: pd.DataFrame,
    raw_inner: np.ndarray,
    raw_outer: np.ndarray,
    target_column: str,
) -> tuple[np.ndarray, np.ndarray, FittedCalibration, dict[str, Any]]:
    candidates: dict[str, dict[str, float]] = {}
    crossfitted: dict[str, np.ndarray] = {}
    fit_frame = frame.copy()
    fit_frame["binary_target"] = fit_frame[target_column].astype(int)
    weight = participant_equal_weights(fit_frame)
    for method in CALIBRATION_METHODS:
        values = _crossfit_calibration(fit_frame, raw_inner, target_column, method)
        crossfitted[method] = values
        candidates[method] = ap_context_metrics(
            fit_frame["binary_target"].to_numpy(int), values, sample_weight=weight
        )
    selected = min(
        CALIBRATION_METHODS,
        key=lambda method: (
            candidates[method]["brier"] + 0.25 * candidates[method]["ece"],
            CALIBRATION_METHODS.index(method),
        ),
    )
    fitted = fit_calibration(fit_frame, raw_inner, selected)
    return (
        crossfitted[selected],
        fitted.predict(raw_outer),
        fitted,
        {
            "selected_method": selected,
            "selection_metric": "participant_equal_brier_plus_0.25_ece",
            "candidate_metrics": candidates,
            "selection_used_outer_labels": False,
        },
    )


def select_dual_calibration(
    frame: pd.DataFrame,
    raw_inner: tuple[np.ndarray, np.ndarray],
    raw_outer: tuple[np.ndarray, np.ndarray],
    *,
    maximum_inner_ap_loss: float = 0.001,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], dict[str, Any]]:
    """Select a calibration pair without sacrificing either task's ranking."""

    targets = {
        "ge5": frame["phq9_ge5_r3_target"].to_numpy(int),
        "ge10": frame["phq9_ge10_r3_target"].to_numpy(int),
    }
    raw = {"ge5": raw_inner[0], "ge10": raw_inner[1]}
    outer = {"ge5": raw_outer[0], "ge10": raw_outer[1]}
    crossfitted: dict[str, dict[CalibrationMethod, np.ndarray]] = {"ge5": {}, "ge10": {}}
    fullfit: dict[str, dict[CalibrationMethod, FittedCalibration]] = {"ge5": {}, "ge10": {}}
    for head, target_column in (
        ("ge5", "phq9_ge5_r3_target"),
        ("ge10", "phq9_ge10_r3_target"),
    ):
        fit_frame = frame.copy()
        fit_frame["binary_target"] = fit_frame[target_column].astype(int)
        for method in CALIBRATION_METHODS:
            crossfitted[head][method] = _crossfit_calibration(
                frame, raw[head], target_column, method
            )
            fullfit[head][method] = fit_calibration(fit_frame, raw[head], method)
    raw_metrics = {
        head: ap_context_metrics(targets[head], raw[head]) for head in ("ge5", "ge10")
    }
    candidates: list[dict[str, Any]] = []
    selected_payload: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
    selected_key: tuple[float, float, int, str] | None = None
    for method5 in CALIBRATION_METHODS:
        for method10 in CALIBRATION_METHODS:
            inner5, inner10 = project_independent_heads(
                crossfitted["ge5"][method5], crossfitted["ge10"][method10]
            )
            metrics = {
                "ge5": ap_context_metrics(targets["ge5"], inner5),
                "ge10": ap_context_metrics(targets["ge10"], inner10),
            }
            ap_loss = {
                head: float(metrics[head]["auprc"] - raw_metrics[head]["auprc"])
                for head in ("ge5", "ge10")
            }
            rank_gate = min(ap_loss.values()) >= -float(maximum_inner_ap_loss)
            objective = float(
                np.mean(
                    [metrics[head]["brier"] + 0.25 * metrics[head]["ece"] for head in ("ge5", "ge10")]
                )
            )
            row = {
                "methods": {"ge5": method5, "ge10": method10},
                "metrics": metrics,
                "delta_auprc_vs_raw": ap_loss,
                "rank_gate": rank_gate,
                "calibration_objective": objective,
            }
            candidates.append(row)
            if not rank_gate:
                continue
            key = (
                -objective,
                min(ap_loss.values()),
                -(CALIBRATION_METHODS.index(method5) + CALIBRATION_METHODS.index(method10)),
                f"{method5}:{method10}",
            )
            if selected_key is None or key > selected_key:
                outer5, outer10 = project_independent_heads(
                    fullfit["ge5"][method5].predict(outer["ge5"]),
                    fullfit["ge10"][method10].predict(outer["ge10"]),
                )
                selected_payload = (inner5, inner10, outer5, outer10)
                selected_key = key
                selected = row
    if selected_payload is None:
        raise RuntimeError("r5 dual calibration has no rank-safe candidate")
    candidates.sort(
        key=lambda row: (
            bool(row["rank_gate"]),
            -float(row["calibration_objective"]),
            min(row["delta_auprc_vs_raw"].values()),
        ),
        reverse=True,
    )
    return (
        (selected_payload[0], selected_payload[1]),
        (selected_payload[2], selected_payload[3]),
        {
            "selected": selected,
            "maximum_inner_ap_loss": float(maximum_inner_ap_loss),
            "raw_metrics": raw_metrics,
            "candidate_count": len(candidates),
            "candidate_pairs": candidates,
            "selection_metric": "rank gate then mean participant-natural brier + 0.25*ece",
            "selection_used_outer_labels": False,
        },
    )
def nested_fusion_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: tuple[str, ...] | None,
    inner_fold: pd.Series,
    specs: Sequence[R5CandidateSpec],
    track: Track,
    structure: dict[str, str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit the complete second-level graph within one outer-train partition."""

    inner = pd.Series(inner_fold, index=train.index).astype(int)
    inner_predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    outer_predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for spec in specs:
        p5 = pd.Series(np.nan, index=train.index, dtype=float)
        p10 = pd.Series(np.nan, index=train.index, dtype=float)
        for fold in sorted(inner.unique()):
            fit = train.loc[inner.ne(fold)]
            validation = train.loc[inner.eq(fold)]
            fold5, fold10 = _predict_split(
                fit, validation, features, spec, track, structure
            )
            p5.loc[validation.index] = fold5
            p10.loc[validation.index] = fold10
        if p5.isna().any() or p10.isna().any():
            raise ValueError(f"r5 candidate {spec.candidate_id} inner OOF is incomplete")
        inner_predictions[spec.candidate_id] = (
            p5.loc[train.index].to_numpy(float),
            p10.loc[train.index].to_numpy(float),
        )
        outer_predictions[spec.candidate_id] = _predict_split(
            train, test, features, spec, track, structure
        )
    spec_map = {spec.candidate_id: spec for spec in specs}
    raw_inner, raw_outer, fusion_audit = select_fusion(
        train, inner_predictions, outer_predictions, spec_map, track
    )
    calibration_frame = train.copy()
    calibration_frame["inner_fold"] = inner.to_numpy(int)
    (calibrated5, calibrated10), (outer5, outer10), calibration_audit = select_dual_calibration(
        calibration_frame, raw_inner, raw_outer
    )
    workpoints = {
        "ge5": select_train_only_workpoints(
            train["phq9_ge5_r3_target"].to_numpy(int), calibrated5
        ),
        "ge10": select_train_only_workpoints(
            train["phq9_ge10_r3_target"].to_numpy(int), calibrated10
        ),
    }
    prediction = _prediction_frame(
        test,
        outer5,
        outer10,
        candidate_id="r5_strict_fusion",
        track=track,
        stage="strict_nested_fusion",
    )
    prediction["raw_probability_ge5"] = raw_outer[0]
    prediction["raw_probability_ge10"] = raw_outer[1]
    for head in ("ge5", "ge10"):
        for point in ("competition", "safety"):
            threshold = float(workpoints[head][point]["threshold"])
            prediction[f"{head}_{point}_threshold"] = threshold
            prediction[f"{head}_{point}_positive"] = (
                prediction[f"probability_{head}"].to_numpy(float) >= threshold
            ).astype(int)
    train_participants = set(train["global_participant_id"].astype(str))
    test_participants = set(test["global_participant_id"].astype(str))
    audit = {
        "protocol_version": R5_PROTOCOL_VERSION,
        "track": track,
        "candidate_specs": [asdict(spec) for spec in specs],
        "fusion": fusion_audit,
        "calibration": calibration_audit,
        "workpoints": workpoints,
        "workpoint_source": "outer-train cross-fitted inner OOF only",
        "outer_test_labels_used_for_selection": False,
        "participant_overlap_count": len(train_participants & test_participants),
        "monotonic_violation_count": int(np.sum(outer10 > outer5 + 1.0e-12)),
    }
    return prediction, audit


def select_stable_single_specs(repository_root: Path) -> dict[Track, R5CandidateSpec]:
    """Lock one development-stable member per track after all three repeats."""

    root = Path(repository_root).resolve()
    summary_path = root / COMPETITION_RELATIVE / "development" / "development_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    selected: dict[Track, R5CandidateSpec] = {}
    for track in TRACKS:
        eligible = [
            row
            for row in summary["tracks"][track]["candidates"]
            if bool(row.get("development_gate"))
        ]
        if not eligible:
            raise RuntimeError(f"r5 has no development-stable candidate for {track}")
        best = max(
            eligible,
            key=lambda row: (
                np.mean(
                    [
                        row["mean_delta"][head]["auprc"]
                        for head in ("ge5", "ge10")
                    ]
                ),
                min(
                    row["mean_delta"][head]["auprc"]
                    for head in ("ge5", "ge10")
                ),
                row["spec"]["candidate_id"],
            ),
        )
        selected[track] = _spec_from_dict(best["spec"])
    return selected


def run_fusion_development_repeat(
    *,
    repository_root: Path,
    track: Track,
    seed: int,
    report_directory: Path | None = None,
    overwrite: bool = False,
    recipe_mode: str = "stable_single",
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    is_confirmation = int(seed) == R5_CONFIRMATION_SEED
    if int(seed) not in (*R5_DEVELOPMENT_SEEDS, R5_CONFIRMATION_SEED):
        raise ValueError("r5 fusion seed is not registered")
    if is_confirmation and not confirmation_state(root)["opened"]:
        raise PermissionError("r5 confirmation must be locked and opened before prediction")
    if is_confirmation and recipe_mode != "stable_single":
        raise PermissionError("r5 confirmation only permits the locked stable-single recipe")
    output = Path(report_directory) if report_directory else root / DEFAULT_REPORT_RELATIVE
    if not output.is_absolute():
        output = root / output
    if recipe_mode == "stable_single":
        if is_confirmation:
            locked = verify_locked_confirmation_recipe(root)
            specs = [_spec_from_dict(locked["recipe"]["candidates"][track])]
        else:
            specs = [select_stable_single_specs(root)[track]]
        development_directory = "candidate" if is_confirmation else STABLE_SINGLE_DIRECTORY
    elif recipe_mode == "adaptive_top3":
        finalists_path = root / COMPETITION_RELATIVE / "full_inner" / track / "finalists.json"
        finalists = json.loads(finalists_path.read_text(encoding="utf-8"))["candidates"]
        specs = [_spec_from_dict(value["spec"]) for value in finalists]
        development_directory = ADAPTIVE_DIRECTORY
    else:
        raise ValueError(f"unknown r5 fusion recipe mode: {recipe_mode}")
    frame, features = _load_track_frame(root, int(seed), track)
    structure = _selected_structure(root)
    prediction_path = output / development_directory / track / f"seed-{int(seed)}.parquet"
    manifest_path = prediction_path.with_suffix(".manifest.json")
    if manifest_path.exists() and not overwrite:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    parts: list[pd.DataFrame] = []
    audits: dict[str, Any] = {}
    checkpoint_root = output / "_checkpoints" / recipe_mode / track / f"seed-{int(seed)}"
    for outer_fold in range(5):
        fold_prediction_path = checkpoint_root / f"outer-{outer_fold}.parquet"
        fold_audit_path = checkpoint_root / f"outer-{outer_fold}.audit.json"
        fold_manifest_path = checkpoint_root / f"outer-{outer_fold}.manifest.json"
        if (
            fold_prediction_path.is_file()
            and fold_audit_path.is_file()
            and fold_manifest_path.is_file()
            and not overwrite
        ):
            fold_manifest = json.loads(fold_manifest_path.read_text(encoding="utf-8"))
            if fold_manifest.get("status") != "pass":
                raise RuntimeError(f"r5 fusion checkpoint failed: {fold_manifest_path}")
            prediction = pd.read_parquet(fold_prediction_path)
            audit = json.loads(fold_audit_path.read_text(encoding="utf-8"))
        else:
            train = frame.loc[frame["outer_fold"].ne(outer_fold)].copy()
            test = frame.loc[frame["outer_fold"].eq(outer_fold)].copy()
            inner = r5_inner_fold_series(train, outer_fold)
            prediction, audit = nested_fusion_predictions(
                train, test, features, inner, specs, track, structure
            )
            prediction["split_seed"] = int(seed)
            prediction["outer_fold"] = int(outer_fold)
            fold_prediction_path.parent.mkdir(parents=True, exist_ok=True)
            prediction.to_parquet(fold_prediction_path, index=False)
            _write_json(fold_audit_path, audit, overwrite=True)
            _write_json(
                fold_manifest_path,
                {
                    "protocol_version": R5_PROTOCOL_VERSION,
                    "status": "pass",
                    "track": track,
                    "seed": int(seed),
                    "outer_fold": int(outer_fold),
                    "outer_test_labels_used_for_selection": False,
                    "prediction_sha256": sha256_file(fold_prediction_path),
                    "audit_sha256": sha256_file(fold_audit_path),
                },
                overwrite=True,
            )
        parts.append(prediction)
        audits[str(outer_fold)] = audit
    result = pd.concat(parts, ignore_index=True)
    if len(result) != len(frame) or result["r5_row_id"].duplicated().any():
        raise ValueError("r5 strict fusion outer OOF coverage is invalid")
    metrics = _candidate_metrics(
        result,
        result["probability_ge5"].to_numpy(float),
        result["probability_ge10"].to_numpy(float),
        track,
    )
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(prediction_path, index=False)
    audit_path = prediction_path.with_suffix(".audit.json")
    metrics_path = prediction_path.with_suffix(".metrics.json")
    _write_json(audit_path, audits, overwrite=True)
    _write_json(metrics_path, metrics, overwrite=True)
    manifest = {
        "protocol_version": R5_PROTOCOL_VERSION,
        "status": "pass",
        "track": track,
        "seed": int(seed),
        "evidence_level": (
            "sealed reused-cohort confirmation"
            if is_confirmation
            else "adaptive-development strict nested outer OOF"
        ),
        "candidate_count": len(specs),
        "recipe_mode": recipe_mode,
        "confirmation_opened": is_confirmation,
        "prediction_sha256": sha256_file(prediction_path),
        "audit_sha256": sha256_file(audit_path),
        "metrics_sha256": sha256_file(metrics_path),
    }
    _write_json(manifest_path, manifest, overwrite=True)
    return manifest


def finalize_fusion_development(
    *, repository_root: Path, report_directory: Path | None = None, overwrite: bool = False
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    output = Path(report_directory) if report_directory else root / DEFAULT_REPORT_RELATIVE
    if not output.is_absolute():
        output = root / output
    tracks: dict[str, Any] = {}
    for track in TRACKS:
        seeds: dict[str, Any] = {}
        for seed in R5_DEVELOPMENT_SEEDS:
            prediction_path = output / STABLE_SINGLE_DIRECTORY / track / f"seed-{seed}.parquet"
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
            candidate = _candidate_metrics(
                paired,
                paired["probability_ge5"].to_numpy(float),
                paired["probability_ge10"].to_numpy(float),
                track,
            )
            reference = _candidate_metrics(
                paired,
                paired["baseline_ge5"].to_numpy(float),
                paired["baseline_ge10"].to_numpy(float),
                track,
            )
            delta = {
                head: {
                    metric: float(candidate["heads"][head][metric] - reference["heads"][head][metric])
                    for metric in ("auprc", "participant_auprc", "auroc", "brier", "ece")
                }
                for head in ("ge5", "ge10")
            }
            seeds[str(seed)] = {"candidate": candidate, "baseline": reference, "delta": delta}
        mean_delta = {
            head: {
                metric: float(np.mean([seeds[str(seed)]["delta"][head][metric] for seed in R5_DEVELOPMENT_SEEDS]))
                for metric in ("auprc", "participant_auprc", "auroc", "brier", "ece")
            }
            for head in ("ge5", "ge10")
        }
        tracks[track] = {
            "seeds": seeds,
            "mean_delta": mean_delta,
            "development_gate": bool(
                max(mean_delta[head]["auprc"] for head in ("ge5", "ge10")) >= 0.003
                and min(mean_delta[head]["auprc"] for head in ("ge5", "ge10")) >= -0.004
            ),
        }
    recipe = {
        "protocol_version": R5_PROTOCOL_VERSION,
        "status": "locked_before_confirmation",
        "tracks": {
            track: {"stable_single_spec": asdict(spec)}
            for track, spec in select_stable_single_specs(root).items()
        },
        "member_rule": {
            "maximum_members": 3,
            "maximum_absolute_residual_correlation": MAX_RESIDUAL_CORRELATION,
            "observed_all_pairwise_residual_correlations_above_limit": True,
            "final_choice": "one development-stable member per track",
        },
        "fusion_rule": {
            "methods": ["probability", "logit", "rank_normalized"],
            "pair_grid": 0.05,
            "three_way_grid": 0.25,
            "selection": "skipped for locked stable single; adaptive-top3 retained as batch-1 audit",
        },
        "calibration_rule": {
            "methods": list(CALIBRATION_METHODS),
            "selection": (
                "dual-head cross-fitted per-head delta AUPRC >= -0.001, "
                "then mean brier + 0.25*ece"
            ),
        },
        "workpoint_rule": "outer-train cross-fitted inner OOF competition and safety",
        "confirmation_seed": 20260829,
        "confirmation_opened": False,
    }
    recipe_path = output / "locked_recipe.json"
    _write_json(recipe_path, recipe, overwrite=overwrite)
    payload = {
        "protocol_version": R5_PROTOCOL_VERSION,
        "status": "pass",
        "evidence_level": "adaptive-development strict nested outer OOF",
        "confirmation_opened": False,
        "tracks": tracks,
        "locked_recipe_sha256": sha256_file(recipe_path),
    }
    _write_json(output / "development_summary.json", payload, overwrite=overwrite)
    return payload


__all__ = [
    "finalize_fusion_development",
    "nested_fusion_predictions",
    "run_fusion_development_repeat",
    "select_stable_single_specs",
    "select_train_only_workpoints",
    "select_calibration",
    "select_dual_calibration",
    "select_fusion",
    "select_low_correlation_members",
]
