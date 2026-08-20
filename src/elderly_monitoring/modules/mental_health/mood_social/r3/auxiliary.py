"""Strict inner-OOF PHQ auxiliary targets for OPT-V333-004."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from elderly_monitoring.modules.mental_health.mood_social.r3.contract import (
    R3_PROTOCOL_VERSION,
)
from elderly_monitoring.modules.mental_health.mood_social.r3.data import (
    eligible_rows,
    inner_fold_series,
    load_r3_training_frame,
)
from elderly_monitoring.modules.mental_health.mood_social.r3.modeling import (
    CandidateUnavailable,
    ModelSpec,
    WeightSpec,
    binary_metrics,
    fit_classifier_target,
    fit_regressor,
    predict_probability,
    predict_regression,
    sample_weights,
    weighted_binary_metrics,
)
from elderly_monitoring.modules.mental_health.mood_social.r3.selection import (
    EXPERT_DEFINITIONS,
)


THRESHOLDS = (5, 10, 15, 20)
AUXILIARY_VERSION = "mood-social-v3.3.3-r3-auxiliary-v1"
DEFAULT_SELECTION_RELATIVE = Path(
    "reports/mental_health/mood_social/v3.3.3-r3/OPT-V333-003-004-selection/selected_experts.json"
)
DEFAULT_REPORT_RELATIVE = Path(
    "reports/mental_health/mood_social/v3.3.3-r3/OPT-V333-004-auxiliary"
)
DEFAULT_PRIMARY_OOF_RELATIVE = Path(
    "reports/mental_health/mood_social/v3.3.3-r3/OPT-V333-003-004-selection/selected_expert_inner_oof.parquet"
)
PARTICIPANT_WEIGHT = WeightSpec("participant", participant_equal=True)
REGRESSION_FALLBACK = ModelSpec(
    "aux_hist_regression_fallback",
    "hist_gradient",
    {"max_leaf_nodes": 15, "l2_regularization": 1.0},
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def project_cumulative_probabilities(probability: np.ndarray) -> np.ndarray:
    """PAVA projection onto p(>=5) >= p(>=10) >= p(>=15) >= p(>=20)."""

    matrix = np.asarray(probability, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(THRESHOLDS):
        raise ValueError("cumulative probability matrix has invalid shape")
    result = np.empty_like(matrix)
    for row_index, row in enumerate(matrix):
        # Pool adjacent violators for a non-increasing four-element sequence.
        levels = [float(value) for value in row]
        weights = [1.0] * len(levels)
        starts = list(range(len(levels)))
        ends = list(range(len(levels)))
        cursor = 0
        while cursor < len(levels) - 1:
            if levels[cursor] >= levels[cursor + 1]:
                cursor += 1
                continue
            total = weights[cursor] + weights[cursor + 1]
            pooled = (
                levels[cursor] * weights[cursor]
                + levels[cursor + 1] * weights[cursor + 1]
            ) / total
            levels[cursor : cursor + 2] = [pooled]
            weights[cursor : cursor + 2] = [total]
            ends[cursor] = ends[cursor + 1]
            del ends[cursor + 1]
            del starts[cursor + 1]
            cursor = max(cursor - 1, 0)
        projected = np.empty(len(THRESHOLDS), dtype=float)
        for level, start, end in zip(levels, starts, ends, strict=True):
            projected[start : end + 1] = level
        result[row_index] = projected
    return np.clip(result, 1.0e-6, 1.0 - 1.0e-6)


def _model_spec(payload: Mapping[str, Any]) -> ModelSpec:
    return ModelSpec(
        candidate_id=str(payload["candidate_id"]),
        family=str(payload["family"]),  # type: ignore[arg-type]
        params=dict(payload["params"]),
        seed=int(payload.get("seed", 20260728)),
    )


def _blend(matrix: np.ndarray, kind: str) -> np.ndarray:
    if kind == "probability":
        return np.mean(matrix, axis=0)
    if kind == "raw_logit":
        logit = np.log(matrix / (1.0 - matrix)).mean(axis=0)
        return 1.0 / (1.0 + np.exp(-np.clip(logit, -30.0, 30.0)))
    # A train-reference rank ensemble is not batch invariant without the
    # training references. Auxiliary targets therefore use probability blend
    # when the selected primary expert was rank-blended.
    if kind == "rank":
        return np.mean(matrix, axis=0)
    raise ValueError(f"unsupported auxiliary blend: {kind}")


def _classification_prediction(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: Sequence[str],
    members: Sequence[ModelSpec],
    target_column: str,
    blend_kind: str,
) -> np.ndarray:
    predictions = []
    for member in members:
        model = fit_classifier_target(
            train,
            features,
            member,
            PARTICIPANT_WEIGHT,
            target_column,
        )
        predictions.append(predict_probability(model, validation, features))
    return _blend(np.vstack(predictions), blend_kind)


def _regression_prediction(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: Sequence[str],
    members: Sequence[ModelSpec],
) -> tuple[np.ndarray, list[str]]:
    regression_members = [
        member
        for member in members
        if member.family in {"hist_gradient", "lightgbm", "catboost", "xgboost"}
    ] or [REGRESSION_FALLBACK]
    predictions: list[np.ndarray] = []
    used: list[str] = []
    for member in regression_members:
        try:
            model = fit_regressor(
                train,
                features,
                member,
                PARTICIPANT_WEIGHT,
            )
        except CandidateUnavailable:
            continue
        predictions.append(predict_regression(model, validation, features))
        used.append(member.candidate_id)
    if not predictions:
        raise CandidateUnavailable("all preregistered PHQ regressors unavailable")
    return np.mean(np.vstack(predictions), axis=0), used


def auxiliary_inner_oof(
    frame: pd.DataFrame,
    *,
    outer_fold: int,
    expert_id: str,
    choice: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    definition = next(item for item in EXPERT_DEFINITIONS if item.expert_id == expert_id)
    outer_train = frame.loc[frame["outer_fold"].ne(int(outer_fold))].copy()
    outer_train["_inner_fold"] = inner_fold_series(frame, outer_fold).loc[
        outer_train.index
    ].astype(int)
    working = outer_train.loc[
        outer_train["route_pattern"].isin(definition.required_routes)
    ].copy()
    members = tuple(_model_spec(item) for item in choice["members"])
    blend_kind = str(choice.get("blend_kind", "probability"))
    parts: list[pd.DataFrame] = []
    regression_members_used: set[str] = set()
    violations_before = 0
    for validation_fold in sorted(working["_inner_fold"].unique()):
        train = working.loc[working["_inner_fold"].ne(validation_fold)]
        validation = working.loc[working["_inner_fold"].eq(validation_fold)]
        threshold_probability = np.column_stack(
            [
                _classification_prediction(
                    train,
                    validation,
                    definition.features,
                    members,
                    f"phq9_ge{threshold}_r3_target",
                    blend_kind,
                )
                for threshold in THRESHOLDS
            ]
        )
        violations_before += int(np.any(np.diff(threshold_probability, axis=1) > 0, axis=1).sum())
        projected = project_cumulative_probabilities(threshold_probability)
        regression, used = _regression_prediction(
            train, validation, definition.features, members
        )
        regression_members_used.update(used)
        part = validation[
            [
                "r3_row_id",
                "dataset_id",
                "global_participant_id",
                "binary_target",
                "phq9_score_r3_target",
                "route_pattern",
            ]
        ].copy()
        part["inner_fold"] = int(validation_fold)
        part["phq9_score_prediction"] = regression
        for index, threshold in enumerate(THRESHOLDS):
            part[f"phq9_ge{threshold}_probability_raw"] = threshold_probability[:, index]
            part[f"phq9_ge{threshold}_probability"] = projected[:, index]
        part["ordinal_expected_severity"] = projected.sum(axis=1)
        part["regression_risk_probability"] = 1.0 / (
            1.0 + np.exp(-np.clip((regression - 10.0) / 3.0, -30.0, 30.0))
        )
        parts.append(part)
    oof = pd.concat(parts, ignore_index=True)
    if len(oof) != len(working) or oof["r3_row_id"].duplicated().any():
        raise ValueError("auxiliary inner OOF coverage is invalid")
    cumulative = oof[[f"phq9_ge{value}_probability" for value in THRESHOLDS]].to_numpy()
    if np.any(np.diff(cumulative, axis=1) > 1.0e-12):
        raise ValueError("projected PHQ cumulative probabilities are not monotonic")
    score_error = oof["phq9_score_prediction"].to_numpy(float) - oof[
        "phq9_score_r3_target"
    ].to_numpy(float)
    audit = {
        "outer_fold": int(outer_fold),
        "expert_id": expert_id,
        "outer_results_opened": False,
        "row_count": int(len(oof)),
        "participant_count": int(oof["global_participant_id"].nunique()),
        "selected_primary_members": [member.candidate_id for member in members],
        "regression_members": sorted(regression_members_used),
        "blend_kind": blend_kind,
        "monotonic_violations_before_projection": int(violations_before),
        "monotonic_violations_after_projection": 0,
        "phq_score_mae": float(np.mean(np.abs(score_error))),
        "phq_score_rmse": float(np.sqrt(np.mean(np.square(score_error)))),
        "ge10_metrics": binary_metrics(
            oof["binary_target"], oof["phq9_ge10_probability"]
        ),
        "regression_risk_metrics": binary_metrics(
            oof["binary_target"], oof["regression_risk_probability"]
        ),
    }
    return oof, audit


def run_auxiliary_oof(
    *,
    repository_root: Path,
    selection_path: Path | None = None,
    report_directory: Path | None = None,
    outer_folds: Iterable[int] = range(5),
    overwrite: bool = False,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    selection_file = selection_path or root / DEFAULT_SELECTION_RELATIVE
    selection = json.loads(selection_file.read_text(encoding="utf-8"))
    output = Path(report_directory) if report_directory else root / DEFAULT_REPORT_RELATIVE
    if not output.is_absolute():
        output = root / output
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = output / "_checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    frame = eligible_rows(load_r3_training_frame(repository_root=root))
    all_oof: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    for outer_fold in tuple(int(value) for value in outer_folds):
        for definition in EXPERT_DEFINITIONS:
            stem = checkpoints / f"outer{outer_fold}_{definition.expert_id}"
            oof_path = stem.with_suffix(".parquet")
            audit_path = stem.with_suffix(".json")
            if oof_path.exists() and audit_path.exists() and not overwrite:
                oof = pd.read_parquet(oof_path)
                audit = json.loads(audit_path.read_text(encoding="utf-8"))
            else:
                choice = selection["outer_choices"][str(outer_fold)][definition.expert_id]
                oof, audit = auxiliary_inner_oof(
                    frame,
                    outer_fold=outer_fold,
                    expert_id=definition.expert_id,
                    choice=choice,
                )
                oof.insert(0, "expert_id", definition.expert_id)
                oof.insert(0, "outer_context", outer_fold)
                oof.to_parquet(oof_path, index=False)
                audit_path.write_text(
                    json.dumps(audit, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
                    + "\n",
                    encoding="utf-8",
                )
            all_oof.append(oof)
            audits.append(audit)
    table = pd.concat(all_oof, ignore_index=True)
    table_path = output / "auxiliary_inner_oof.parquet"
    audit_path = output / "auxiliary_audit.json"
    if not overwrite and (table_path.exists() or audit_path.exists()):
        raise FileExistsError("refusing to overwrite auxiliary final artifacts")
    table.to_parquet(table_path, index=False)
    audit_path.write_text(
        json.dumps(
            {
                "protocol_version": R3_PROTOCOL_VERSION,
                "auxiliary_version": AUXILIARY_VERSION,
                "outer_results_opened": False,
                "audits": audits,
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "protocol_version": R3_PROTOCOL_VERSION,
        "auxiliary_version": AUXILIARY_VERSION,
        "task_id": "OPT-V333-004",
        "status": "pass",
        "outer_results_opened": False,
        "selection_path": selection_file.relative_to(root).as_posix(),
        "selection_sha256": _sha256_file(selection_file),
        "oof_path": table_path.relative_to(root).as_posix(),
        "oof_sha256": _sha256_file(table_path),
        "oof_rows": int(len(table)),
        "audit_path": audit_path.relative_to(root).as_posix(),
        "audit_sha256": _sha256_file(audit_path),
    }
    manifest_path = output / "artifact_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return manifest


def evaluate_auxiliary_ablation(
    *,
    repository_root: Path,
    auxiliary_oof_path: Path | None = None,
    primary_oof_path: Path | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Decide, before fusion outer results, whether auxiliary signals promote."""

    root = Path(repository_root).resolve()
    aux_path = auxiliary_oof_path or root / DEFAULT_REPORT_RELATIVE / "auxiliary_inner_oof.parquet"
    primary_path = primary_oof_path or root / DEFAULT_PRIMARY_OOF_RELATIVE
    output = output_path or root / DEFAULT_REPORT_RELATIVE / "auxiliary_ablation.json"
    auxiliary = pd.read_parquet(aux_path)
    primary = pd.read_parquet(primary_path)[
        ["outer_context", "expert_id", "r3_row_id", "probability"]
    ].rename(columns={"probability": "primary_probability"})
    merged = auxiliary.merge(
        primary,
        on=["outer_context", "expert_id", "r3_row_id"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(auxiliary):
        raise ValueError("auxiliary and primary expert OOF rows do not align")
    records: list[dict[str, Any]] = []
    candidate_columns = {
        "primary": "primary_probability",
        "aux_ge10": "phq9_ge10_probability",
        "regression_risk": "regression_risk_probability",
    }
    for weight in (0.25, 0.5, 0.75):
        name = f"primary_regression_blend_{int(weight * 100):02d}"
        merged[name] = (
            weight * merged["primary_probability"]
            + (1.0 - weight) * merged["regression_risk_probability"]
        )
        candidate_columns[name] = name
    for (outer_context, expert_id), frame in merged.groupby(
        ["outer_context", "expert_id"], sort=True
    ):
        primary_metrics = binary_metrics(
            frame["binary_target"], frame["primary_probability"]
        )
        for candidate_id, column in candidate_columns.items():
            natural = binary_metrics(frame["binary_target"], frame[column])
            participant = weighted_binary_metrics(
                frame["binary_target"],
                frame[column],
                sample_weights(frame, PARTICIPANT_WEIGHT),
            )
            records.append(
                {
                    "outer_context": int(outer_context),
                    "expert_id": str(expert_id),
                    "candidate_id": candidate_id,
                    "natural_auprc": natural["auprc"],
                    "participant_auprc": participant["auprc"],
                    "natural_brier": natural["brier"],
                    "auprc_delta_vs_primary": float(
                        natural["auprc"] - primary_metrics["auprc"]
                    ),
                }
            )
    table = pd.DataFrame(records)
    best_by_context = (
        table.sort_values(
            [
                "outer_context",
                "expert_id",
                "natural_auprc",
                "participant_auprc",
                "natural_brier",
                "candidate_id",
            ],
            ascending=[True, True, False, False, True, True],
            kind="stable",
        )
        .groupby(["outer_context", "expert_id"], sort=True)
        .head(1)
    )
    promotion: dict[str, Any] = {}
    for expert_id, frame in best_by_context.groupby("expert_id", sort=True):
        deltas = frame["auprc_delta_vs_primary"].to_numpy(float)
        auxiliary_wins = frame["candidate_id"].ne("primary")
        passed = bool(
            auxiliary_wins.sum() >= 4
            and float(deltas.mean()) >= 0.005
            and float(deltas.min()) >= -0.002
        )
        promotion[str(expert_id)] = {
            "promoted": passed,
            "winning_contexts": int(auxiliary_wins.sum()),
            "mean_auprc_delta": float(deltas.mean()),
            "minimum_auprc_delta": float(deltas.min()),
            "best_candidates": frame[
                ["outer_context", "candidate_id", "auprc_delta_vs_primary"]
            ].to_dict(orient="records"),
        }
    payload = {
        "protocol_version": R3_PROTOCOL_VERSION,
        "auxiliary_version": AUXILIARY_VERSION,
        "outer_results_opened": False,
        "promotion_rule": (
            "auxiliary wins >=4/5 contexts, mean AUPRC delta >=0.005, "
            "and minimum delta >=-0.002 within each expert"
        ),
        "promotion": promotion,
        "any_auxiliary_promoted": any(item["promoted"] for item in promotion.values()),
        "records": table.to_dict(orient="records"),
    }
    output.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return payload


__all__ = [
    "AUXILIARY_VERSION",
    "THRESHOLDS",
    "auxiliary_inner_oof",
    "project_cumulative_probabilities",
    "evaluate_auxiliary_ablation",
    "run_auxiliary_oof",
]
