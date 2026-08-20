"""Strict r3 replay of the deployed V3.3.3 modelling recipe.

The replay deliberately keeps the legacy expert families and the interaction
logistic/coverage-Platt fusion recipe, while rebuilding every fitted object on
the frozen r2 split.  The second-level fusion OOF is generated from experts
that are re-cross-fitted inside each fusion fold; global expert OOF tables are
never reused as fusion validation features.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from elderly_monitoring.modules.mental_health.mood_social.r3.contract import (
    R3_PROTOCOL_VERSION,
)
from elderly_monitoring.modules.mental_health.mood_social.r3.data import (
    eligible_rows,
    inner_fold_series,
    load_r3_training_frame,
)
from elderly_monitoring.modules.mental_health.mood_social.r3.modeling import (
    LEGACY_ACTIVITY_SPEC,
    LEGACY_JOINT_SPEC,
    LEGACY_PROFILE_SPEC,
    LEGACY_SLEEP_SPEC,
    ModelSpec,
    WeightSpec,
    binary_metrics,
    fit_classifier,
    predict_probability,
    sample_weights,
    weighted_binary_metrics,
)


TASK_ID = "OPT-V333-002"
REPLAY_VERSION = "mood-social-v3.3.3-r3-strict-baseline-replay-v1"
EPSILON = 1.0e-6
BASE_WEIGHT = WeightSpec("participant", participant_equal=True)
DEFAULT_REPORT_RELATIVE = Path(
    "reports/mental_health/mood_social/v3.3.3-r3/OPT-V333-002-baseline-replay"
)


@dataclass(frozen=True)
class LegacyExpertDefinition:
    expert_id: str
    features: tuple[str, ...]
    routes: tuple[str, ...]
    model_spec: ModelSpec
    reliability: float


# These are the selected input fields and frozen runtime reliabilities in
# MH-20260802-013. Feature availability only gates evidence; it is not passed
# to an expert classifier as a learned missingness indicator.
LEGACY_EXPERTS: tuple[LegacyExpertDefinition, ...] = (
    LegacyExpertDefinition(
        "activity",
        (
            "activity.activity_volume_norm",
            "activity.relative_amplitude",
            "activity.interdaily_stability",
            "activity.intradaily_variability",
            "activity.activity_variability",
            "activity.valid_days",
        ),
        ("100", "101", "110", "111"),
        LEGACY_ACTIVITY_SPEC,
        0.1538915552195952,
    ),
    LegacyExpertDefinition(
        "sleep",
        (
            "sleep.sleep_duration_norm",
            "sleep.time_in_bed_norm",
            "sleep.sleep_efficiency",
            "sleep.sleep_onset_sin",
            "sleep.sleep_onset_cos",
            "sleep.wake_time_sin",
            "sleep.wake_time_cos",
            "sleep.sleep_midpoint_sin",
            "sleep.sleep_midpoint_cos",
            "sleep.sleep_fragmentation",
            "sleep.sleep_regularity",
            "sleep.valid_nights",
        ),
        ("010", "011", "110", "111"),
        LEGACY_SLEEP_SPEC,
        0.1947389358315479,
    ),
    LegacyExpertDefinition(
        "joint",
        (
            "activity.activity_volume_norm",
            "activity.relative_amplitude",
            "activity.interdaily_stability",
            "activity.intradaily_variability",
            "activity.activity_variability",
            "activity.valid_days",
            "sleep.sleep_duration_norm",
            "sleep.time_in_bed_norm",
            "sleep.sleep_efficiency",
            "sleep.sleep_onset_sin",
            "sleep.sleep_onset_cos",
            "sleep.wake_time_sin",
            "sleep.wake_time_cos",
            "sleep.sleep_midpoint_sin",
            "sleep.sleep_midpoint_cos",
            "sleep.sleep_fragmentation",
            "sleep.sleep_regularity",
            "sleep.valid_nights",
        ),
        ("110", "111"),
        LEGACY_JOINT_SPEC,
        0.27138228707079404,
    ),
    LegacyExpertDefinition(
        "profile",
        (
            "social_context.age_group",
            "social_context.sex",
            "social_context.marital_status",
            "social_context.self_rated_health",
            "social_context.education_level",
            "social_context.economic_status",
        ),
        ("001", "011", "101", "111"),
        LEGACY_PROFILE_SPEC,
        0.3524914741129761,
    ),
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _applicable(frame: pd.DataFrame, definition: LegacyExpertDefinition) -> pd.Series:
    return frame["route_pattern"].isin(definition.routes)


def _observed_fraction(
    frame: pd.DataFrame, definition: LegacyExpertDefinition
) -> np.ndarray:
    observed = frame.loc[:, definition.features].notna().sum(axis=1).to_numpy(float)
    return observed / float(len(definition.features))


def _fit_predict_expert(
    train: pd.DataFrame,
    predict: pd.DataFrame,
    definition: LegacyExpertDefinition,
) -> np.ndarray:
    train_work = train.loc[_applicable(train, definition)].copy()
    if train_work.empty or train_work["binary_target"].nunique() < 2:
        raise ValueError(f"legacy {definition.expert_id} training partition lacks classes")
    model = fit_classifier(
        train_work,
        definition.features,
        definition.model_spec,
        BASE_WEIGHT,
    )
    probability = np.full(len(predict), np.nan, dtype=float)
    available = _applicable(predict, definition).to_numpy()
    if available.any():
        probability[available] = predict_probability(
            model, predict.loc[available], definition.features
        )
    return probability


def _expert_raw_oof(
    frame: pd.DataFrame,
    fold: pd.Series,
) -> pd.DataFrame:
    """Cross-fit every legacy expert over the supplied participant folds."""

    fold = pd.Series(fold, index=frame.index).astype(int)
    result = frame[
        [
            "r3_row_id",
            "dataset_id",
            "global_participant_id",
            "binary_target",
            "route_pattern",
        ]
    ].copy()
    result["_fold"] = fold.to_numpy()
    for definition in LEGACY_EXPERTS:
        result[f"{definition.expert_id}_raw"] = np.nan
        available = _applicable(frame, definition).to_numpy()
        confidence = definition.reliability * _observed_fraction(frame, definition)
        result[f"{definition.expert_id}_confidence"] = np.where(
            available, confidence, 0.0
        )
    for validation_fold in sorted(fold.unique()):
        train = frame.loc[fold.ne(validation_fold)]
        validation = frame.loc[fold.eq(validation_fold)]
        for definition in LEGACY_EXPERTS:
            probability = _fit_predict_expert(train, validation, definition)
            result.loc[fold.eq(validation_fold), f"{definition.expert_id}_raw"] = (
                probability
            )
    for definition in LEGACY_EXPERTS:
        available = _applicable(frame, definition).to_numpy()
        values = result[f"{definition.expert_id}_raw"].to_numpy(float)
        if not np.isfinite(values[available]).all() or np.isfinite(values[~available]).any():
            raise ValueError(f"legacy {definition.expert_id} OOF coverage is invalid")
    return result


@dataclass(frozen=True)
class _ProbabilityCalibrator:
    method: str
    estimator: Any

    def predict(self, probability: Sequence[float]) -> np.ndarray:
        values = np.clip(np.asarray(probability, dtype=float), EPSILON, 1.0 - EPSILON)
        if self.method == "isotonic":
            return np.clip(self.estimator.predict(values), EPSILON, 1.0 - EPSILON)
        return self.estimator.predict_proba(_logit(values)[:, None])[:, 1]


def _fit_calibrator(frame: pd.DataFrame, column: str) -> _ProbabilityCalibrator:
    work = frame.loc[frame[column].notna()].copy()
    if work.empty or work["binary_target"].nunique() < 2:
        raise ValueError(f"calibrator {column} lacks both classes")
    probability = work[column].to_numpy(float)
    target = work["binary_target"].to_numpy(int)
    weight = sample_weights(work, BASE_WEIGHT)
    positive_participants = work.loc[
        work["binary_target"].eq(1), "global_participant_id"
    ].nunique()
    if positive_participants >= 50 and np.unique(probability).size >= 4:
        estimator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        estimator.fit(probability, target, sample_weight=weight)
        return _ProbabilityCalibrator("isotonic", estimator)
    estimator = LogisticRegression(
        solver="liblinear", C=1.0, max_iter=3000, random_state=20260728
    )
    estimator.fit(_logit(probability)[:, None], target, sample_weight=weight)
    return _ProbabilityCalibrator("platt", estimator)


def _crossfit_expert_calibration(raw_oof: pd.DataFrame) -> pd.DataFrame:
    result = raw_oof.copy()
    for definition in LEGACY_EXPERTS:
        raw_column = f"{definition.expert_id}_raw"
        calibrated_column = f"{definition.expert_id}_probability"
        result[calibrated_column] = np.nan
        for validation_fold in sorted(result["_fold"].unique()):
            calibrator = _fit_calibrator(
                result.loc[result["_fold"].ne(validation_fold)], raw_column
            )
            selected = result["_fold"].eq(validation_fold) & result[raw_column].notna()
            result.loc[selected, calibrated_column] = calibrator.predict(
                result.loc[selected, raw_column]
            )
    return result


def _expert_train_and_predict(
    train: pd.DataFrame,
    train_fold: pd.Series,
    predict: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return leakage-safe train features and full-train predictions."""

    raw_oof = _expert_raw_oof(train, train_fold)
    train_features = _crossfit_expert_calibration(raw_oof)
    predict_features = predict[
        [
            "r3_row_id",
            "dataset_id",
            "global_participant_id",
            "binary_target",
            "route_pattern",
        ]
    ].copy()
    for definition in LEGACY_EXPERTS:
        raw_column = f"{definition.expert_id}_raw"
        probability_column = f"{definition.expert_id}_probability"
        raw_probability = _fit_predict_expert(train, predict, definition)
        predict_features[raw_column] = raw_probability
        predict_features[probability_column] = np.nan
        available = _applicable(predict, definition).to_numpy()
        predict_features[f"{definition.expert_id}_confidence"] = np.where(
            available,
            definition.reliability * _observed_fraction(predict, definition),
            0.0,
        )
        calibrator = _fit_calibrator(raw_oof, raw_column)
        selected = np.isfinite(raw_probability)
        # A held-out source can contain none of an expert's routes (for
        # example profile-only NHANES-SSQ has no activity rows).  Scikit-learn
        # isotonic calibration rejects a zero-length vector, so preserve the
        # all-NaN unavailable branch without calling the estimator.
        if selected.any():
            predict_features.loc[selected, probability_column] = calibrator.predict(
                raw_probability[selected]
            )
    return train_features, predict_features


def _fusion_matrix(frame: pd.DataFrame) -> np.ndarray:
    evidence: list[np.ndarray] = []
    for definition in LEGACY_EXPERTS:
        probability = frame[f"{definition.expert_id}_probability"].to_numpy(float)
        available = np.isfinite(probability)
        safe = np.clip(np.nan_to_num(probability, nan=0.5), EPSILON, 1.0 - EPSILON)
        # The old runtime multiplied a branch logit by frozen model reliability
        # and the observed-feature fraction.  The latter is routing/confidence
        # metadata, not an independently learned feature.
        confidence = frame[f"{definition.expert_id}_confidence"].to_numpy(float)
        if not np.isfinite(confidence).all() or np.any(confidence < 0.0):
            raise ValueError(f"invalid {definition.expert_id} fusion confidence")
        evidence.append(confidence * _logit(safe))
    base = np.column_stack(evidence)
    interactions = np.column_stack(
        [
            base[:, 0] * base[:, 1],
            base[:, 0] * base[:, 2],
            base[:, 1] * base[:, 2],
        ]
    )
    return np.column_stack([base, interactions])


def _fit_fusion(frame: pd.DataFrame) -> LogisticRegression:
    model = LogisticRegression(
        solver="liblinear", C=10.0, max_iter=5000, random_state=20260728
    )
    model.fit(
        _fusion_matrix(frame),
        frame["binary_target"].to_numpy(int),
        sample_weight=sample_weights(frame, BASE_WEIGHT),
    )
    return model


def _fusion_probability(model: LogisticRegression, frame: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(_fusion_matrix(frame))[:, 1]


@dataclass(frozen=True)
class _FusionCalibration:
    global_calibrator: _ProbabilityCalibrator
    route_calibrators: Mapping[str, _ProbabilityCalibrator]

    def predict(self, raw_probability: np.ndarray, routes: pd.Series) -> np.ndarray:
        # This reproduces the deployed coverage_calibrated inference order.
        result = self.global_calibrator.predict(raw_probability)
        for route, calibrator in self.route_calibrators.items():
            selected = routes.astype(str).eq(route).to_numpy()
            if selected.any():
                result[selected] = calibrator.predict(result[selected])
        return np.clip(result, EPSILON, 1.0 - EPSILON)


def _fit_fusion_calibration(fusion_oof: pd.DataFrame) -> _FusionCalibration:
    global_calibrator = _fit_calibrator(fusion_oof, "fusion_raw_probability")
    route_calibrators: dict[str, _ProbabilityCalibrator] = {}
    for route in sorted(fusion_oof["route_pattern"].astype(str).unique()):
        work = fusion_oof.loc[fusion_oof["route_pattern"].astype(str).eq(route)]
        positives = int(work["binary_target"].sum())
        negatives = int(len(work) - positives)
        if len(work) >= 50 and positives >= 5 and negatives >= 5:
            # The deployed candidate fitted coverage Platt on the raw primary
            # probability but applied it after the global Platt; preserve that
            # recipe in the replay rather than silently correcting history.
            route_calibrators[route] = _fit_calibrator(
                work, "fusion_raw_probability"
            )
    return _FusionCalibration(global_calibrator, route_calibrators)


def _crossfit_fusion_calibration(fusion_oof: pd.DataFrame) -> np.ndarray:
    result = np.full(len(fusion_oof), np.nan, dtype=float)
    for validation_fold in sorted(fusion_oof["fusion_validation_fold"].unique()):
        train = fusion_oof.loc[
            fusion_oof["fusion_validation_fold"].ne(validation_fold)
        ]
        selected = fusion_oof["fusion_validation_fold"].eq(validation_fold).to_numpy()
        calibration = _fit_fusion_calibration(train)
        result[selected] = calibration.predict(
            fusion_oof.loc[selected, "fusion_raw_probability"].to_numpy(float),
            fusion_oof.loc[selected, "route_pattern"],
        )
    if not np.isfinite(result).all():
        raise ValueError("cross-fitted fusion calibration is incomplete")
    return result


def _second_level_fusion_oof(
    outer_train: pd.DataFrame,
    inner_fold: pd.Series,
) -> pd.DataFrame:
    """Build genuine second-level OOF without global-expert-OOF reuse."""

    parts: list[pd.DataFrame] = []
    inner_fold = pd.Series(inner_fold, index=outer_train.index).astype(int)
    for fusion_validation_fold in sorted(inner_fold.unique()):
        fusion_train = outer_train.loc[inner_fold.ne(fusion_validation_fold)].copy()
        fusion_validation = outer_train.loc[
            inner_fold.eq(fusion_validation_fold)
        ].copy()
        subfold = inner_fold.loc[fusion_train.index]
        train_features, validation_features = _expert_train_and_predict(
            fusion_train,
            subfold,
            fusion_validation,
        )
        fusion_model = _fit_fusion(train_features)
        validation_features["fusion_raw_probability"] = _fusion_probability(
            fusion_model, validation_features
        )
        validation_features["fusion_validation_fold"] = int(
            fusion_validation_fold
        )
        parts.append(validation_features)
    result = pd.concat(parts, ignore_index=True)
    if len(result) != len(outer_train) or result["r3_row_id"].duplicated().any():
        raise ValueError("second-level fusion OOF does not cover outer train once")
    return result


def legacy_train_and_predict(
    train: pd.DataFrame,
    train_fold: pd.Series,
    predict: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Strict legacy baseline features for a meta-train/meta-test partition."""

    second_level_oof = _second_level_fusion_oof(train, train_fold)
    train_output = second_level_oof[
        [
            "r3_row_id",
            "dataset_id",
            "global_participant_id",
            "binary_target",
            "route_pattern",
            "fusion_validation_fold",
            "fusion_raw_probability",
        ]
    ].copy()
    train_output["baseline_probability"] = _crossfit_fusion_calibration(
        second_level_oof
    )
    fusion_calibration = _fit_fusion_calibration(second_level_oof)
    train_features, predict_features = _expert_train_and_predict(
        train, train_fold, predict
    )
    fusion_model = _fit_fusion(train_features)
    predict_raw = _fusion_probability(fusion_model, predict_features)
    predict_probability = fusion_calibration.predict(
        predict_raw, predict_features["route_pattern"]
    )
    return train_output, predict_raw, predict_probability


def replay_outer_fold(frame: pd.DataFrame, outer_fold: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    outer_train = frame.loc[frame["outer_fold"].ne(int(outer_fold))].copy()
    outer_test = frame.loc[frame["outer_fold"].eq(int(outer_fold))].copy()
    inner = inner_fold_series(frame, outer_fold).loc[outer_train.index].astype(int)
    baseline_train, raw_probability, probability = legacy_train_and_predict(
        outer_train, inner, outer_test
    )
    output = outer_test[
        [
            "r3_row_id",
            "dataset_id",
            "global_participant_id",
            "binary_target",
            "route_pattern",
        ]
    ].copy()
    output["outer_fold"] = int(outer_fold)
    output["baseline_raw_probability"] = raw_probability
    output["baseline_probability"] = probability
    audit = {
        "outer_fold": int(outer_fold),
        "outer_train_rows": int(len(outer_train)),
        "outer_test_rows": int(len(outer_test)),
        "outer_train_participants": int(outer_train["global_participant_id"].nunique()),
        "outer_test_participants": int(outer_test["global_participant_id"].nunique()),
        "participant_overlap": int(
            len(
                set(outer_train["global_participant_id"].astype(str))
                & set(outer_test["global_participant_id"].astype(str))
            )
        ),
        "second_level_oof_rows": int(len(baseline_train)),
        "second_level_unique_rows": bool(
            not baseline_train["r3_row_id"].duplicated().any()
        ),
        "fusion_validation_folds": sorted(
            int(value) for value in baseline_train["fusion_validation_fold"].unique()
        ),
        "route_calibrators": "strict_cross_fitted_then_full_train",
    }
    if audit["participant_overlap"] != 0:
        raise ValueError("outer participant leakage in baseline replay")
    return output, audit


def _ece(target: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for index in range(bins):
        selected = (probability >= edges[index]) & (
            probability < edges[index + 1]
            if index < bins - 1
            else probability <= edges[index + 1]
        )
        if selected.any():
            total += float(selected.mean()) * abs(
                float(probability[selected].mean()) - float(target[selected].mean())
            )
    return float(total)


def replay_legacy_baseline(
    *,
    repository_root: Path,
    report_directory: Path | None = None,
    outer_folds: Iterable[int] = range(5),
    overwrite: bool = False,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    output = Path(report_directory) if report_directory else root / DEFAULT_REPORT_RELATIVE
    if not output.is_absolute():
        output = root / output
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = output / "_checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    frame = eligible_rows(load_r3_training_frame(repository_root=root))
    folds = tuple(int(value) for value in outer_folds)
    predictions: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    for outer_fold in folds:
        checkpoint_prediction = checkpoints / f"outer_fold_{outer_fold}.parquet"
        checkpoint_audit = checkpoints / f"outer_fold_{outer_fold}.json"
        if checkpoint_prediction.exists() and checkpoint_audit.exists() and not overwrite:
            prediction = pd.read_parquet(checkpoint_prediction)
            audit = json.loads(checkpoint_audit.read_text(encoding="utf-8"))
            expected_ids = set(
                frame.loc[frame["outer_fold"].eq(outer_fold), "r3_row_id"].astype(str)
            )
            if set(prediction["r3_row_id"].astype(str)) != expected_ids:
                raise ValueError(f"baseline checkpoint outer {outer_fold} row drift")
        else:
            prediction, audit = replay_outer_fold(frame, outer_fold)
            prediction.to_parquet(checkpoint_prediction, index=False)
            checkpoint_audit.write_text(
                json.dumps(
                    audit,
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
        predictions.append(prediction)
        audits.append(audit)
    oof = pd.concat(predictions, ignore_index=True).sort_values(
        ["dataset_id", "r3_row_id"], kind="stable"
    ).reset_index(drop=True)
    expected = frame[frame["outer_fold"].isin(folds)]
    if len(oof) != len(expected) or oof["r3_row_id"].duplicated().any():
        raise ValueError("baseline replay OOF coverage is invalid")
    target = oof["binary_target"].to_numpy(int)
    probability = oof["baseline_probability"].to_numpy(float)
    natural = binary_metrics(target, probability)
    participant_weight = sample_weights(
        oof, WeightSpec("participant_eval", participant_equal=True)
    )
    participant = weighted_binary_metrics(target, probability, participant_weight)
    metrics = {
        "protocol_version": R3_PROTOCOL_VERSION,
        "replay_version": REPLAY_VERSION,
        "task_id": TASK_ID,
        "historical_reference_auprc": 0.238784,
        "historical_reference_only": True,
        "formal_r3_paired_baseline": len(folds) == 5,
        "population": "common-support",
        "row_count": int(len(oof)),
        "participant_count": int(oof["global_participant_id"].nunique()),
        "positive_row_count": int(target.sum()),
        "prevalence": float(target.mean()),
        "prediction_coverage": 1.0,
        "natural": {**natural, "ece": _ece(target, probability)},
        "participant_equal": {
            **participant,
            "ece": _ece(target, probability),
        },
        "outer_fold_metrics": {
            str(fold): binary_metrics(
                oof.loc[oof["outer_fold"].eq(fold), "binary_target"],
                oof.loc[oof["outer_fold"].eq(fold), "baseline_probability"],
            )
            for fold in folds
        },
    }
    oof_path = output / "strict_baseline_oof.parquet"
    metrics_path = output / "baseline_metrics.json"
    audit_path = output / "outer_isolation_audit.json"
    for path in (oof_path, metrics_path, audit_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite baseline artifact: {path}")
    oof.to_parquet(oof_path, index=False)
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    audit_path.write_text(
        json.dumps(
            {
                "outer_results_opened": True,
                "scope": "frozen_legacy_baseline_only_no_r3_candidate",
                "outer_folds": audits,
                "all_participant_overlaps_zero": all(
                    row["participant_overlap"] == 0 for row in audits
                ),
                "all_second_level_oof_complete": all(
                    row["second_level_oof_rows"] == row["outer_train_rows"]
                    and row["second_level_unique_rows"]
                    for row in audits
                ),
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
        "replay_version": REPLAY_VERSION,
        "task_id": TASK_ID,
        "status": "pass",
        "outer_candidate_results_opened": False,
        "model_recipe": {
            "experts": [
                {
                    **asdict(definition),
                    "model_spec": definition.model_spec.to_dict(),
                }
                for definition in LEGACY_EXPERTS
            ],
            "expert_calibration": "strict cross-fitted legacy isotonic/platt",
            "fusion": "legacy C=10 interaction logistic",
            "fusion_calibration": "genuine second-level OOF global plus route Platt",
            "fit_weight": BASE_WEIGHT.to_dict(),
        },
        "artifacts": {
            "strict_baseline_oof": {
                "path": oof_path.relative_to(root).as_posix(),
                "sha256": _sha256_file(oof_path),
            },
            "baseline_metrics": {
                "path": metrics_path.relative_to(root).as_posix(),
                "sha256": _sha256_file(metrics_path),
            },
            "outer_isolation_audit": {
                "path": audit_path.relative_to(root).as_posix(),
                "sha256": _sha256_file(audit_path),
            },
        },
    }
    manifest_path = output / "artifact_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    manifest["artifact_manifest_sha256"] = _sha256_file(manifest_path)
    return manifest


def _logit(probability: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(probability, dtype=float), EPSILON, 1.0 - EPSILON)
    return np.log(value / (1.0 - value))


__all__ = [
    "DEFAULT_REPORT_RELATIVE",
    "LEGACY_EXPERTS",
    "REPLAY_VERSION",
    "replay_legacy_baseline",
    "replay_outer_fold",
    "legacy_train_and_predict",
]
