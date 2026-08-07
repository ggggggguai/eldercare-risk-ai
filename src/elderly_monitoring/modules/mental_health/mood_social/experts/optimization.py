"""Strict nested-OOF optimization for the three strong mood-social experts.

This module consumes frozen V3.3.3 expert OOF predictions and the corresponding
canonical feature rows.  It creates versioned, offline-only candidates for
Activity, Sleep and ActivitySleepJoint.  Dataset identifiers, feature masks and
coverage are used only for alignment/availability checks, never as risk inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
import yaml

from elderly_monitoring.modules.mental_health.mood_social.experts.activity import (
    EXPECTED_SPLIT_SHA256,
    load_activity_training_inputs,
)


TASK_ID = "OPT-EXPERT-001"
RUN_ID = "MH-20260804-015"
MODEL_VERSION = "mood-social-expert-optimization-candidate-v3.3.4-v1"
RANDOM_SEED = 20260728
OUTER_FOLDS = 5
INNER_FOLDS = 5
EPSILON = 1.0e-7
EXPERTS = ("activity", "sleep", "joint")
CANDIDATES = (
    "lgbm_temporal",
    "catboost_temporal",
    "xgboost_temporal",
    "lgbm_temporal_bagging3",
)
AVAILABLE_XGBOOST_CANDIDATE = "xgboost_temporal"
PREDICTION_COLUMNS = {
    "activity": "reports/mental_health/mood_social/MH-20260731-001/predictions.parquet",
    "sleep": "reports/mental_health/mood_social/MH-20260731-002/predictions.parquet",
    "joint": "reports/mental_health/mood_social/MH-20260731-003/predictions.parquet",
}
PREDICTION_SHA256 = {
    "activity": "c3c54f192785ea1336bddc01e7c4d0502dde3a22963ba69e3967a0d62a70275c",
    "sleep": "f8f4c0de0f51d768383840504724ecac27798a7854abc2006dfd1a0c8f1da685",
    "joint": "b55c53faabca77b48a1b630af9dfdbfcf3857975d01dd6d0623628119c3c0f64",
}


class ExpertOptimizationError(RuntimeError):
    """Raised when the frozen strong-expert protocol is violated."""


class CandidateUnavailable(ExpertOptimizationError):
    """Raised for an optional model family absent from the environment."""


@dataclass(frozen=True)
class ExpertOptimizationConfig:
    repository_root: Path
    payload: Mapping[str, Any]
    config_path: Path

    @property
    def report_directory(self) -> Path:
        return _resolve(
            self.repository_root, self.payload["output"]["report_directory"]
        )

    @property
    def model_path(self) -> Path:
        return _resolve(self.repository_root, self.payload["output"]["model_path"])

    @property
    def manifest_path(self) -> Path:
        return _resolve(self.repository_root, self.payload["output"]["manifest_path"])


@dataclass
class CandidateFit:
    candidate_id: str
    imputer: SimpleImputer
    estimator: Any
    calibrator: LogisticRegression
    feature_names: tuple[str, ...]

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        matrix = np.asarray(
            self.imputer.transform(features.loc[:, self.feature_names]),
            dtype=float,
        )
        raw = _clip_probability(self._predict_raw(matrix))
        calibrated = self.calibrator.predict_proba(_logit(raw)[:, None])[:, 1]
        return _clip_probability(calibrated)

    def _predict_raw(self, matrix: np.ndarray) -> np.ndarray:
        if isinstance(self.estimator, list):
            values = [model.predict_proba(matrix)[:, 1] for model in self.estimator]
            return np.mean(values, axis=0)
        return self.estimator.predict_proba(matrix)[:, 1]


def load_expert_optimization_config(
    path: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> ExpertOptimizationConfig:
    config_path = Path(path).resolve()
    root = Path(repository_root or config_path.parents[2]).resolve()
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ExpertOptimizationError("OPT-EXPERT-001 config is unreadable") from exc
    if not isinstance(payload, Mapping):
        raise ExpertOptimizationError("OPT-EXPERT-001 config must be a mapping")
    if payload.get("task_id") != TASK_ID or payload.get("run_id") != RUN_ID:
        raise ExpertOptimizationError("OPT-EXPERT-001 identity changed")
    if tuple(payload.get("experts", ())) != EXPERTS:
        raise ExpertOptimizationError("expert optimization scope changed")
    if tuple(payload.get("candidates", {}).get("ids", ())) != CANDIDATES:
        raise ExpertOptimizationError("expert candidate grid changed")
    if (
        payload.get("split", {}).get("outer_folds") != OUTER_FOLDS
        or payload.get("split", {}).get("inner_folds") != INNER_FOLDS
    ):
        raise ExpertOptimizationError("nested participant split changed")
    boundary = payload.get("production_boundary", {})
    for key in (
        "offline_only",
        "overwrite_base_models",
        "use_dataset_id_as_feature",
        "use_feature_masks_as_risk_inputs",
        "use_coverage_as_risk_input",
        "select_threshold",
    ):
        if boundary.get(key) is not False and key != "offline_only":
            raise ExpertOptimizationError(f"production boundary changed: {key}")
    if boundary.get("offline_only") is not True:
        raise ExpertOptimizationError("expert optimization must remain offline-only")
    return ExpertOptimizationConfig(root, payload, config_path)


def load_expert_optimization_inputs(
    config: ExpertOptimizationConfig,
) -> tuple[
    dict[str, pd.DataFrame], Mapping[str, pd.DataFrame], pd.DataFrame, dict[str, Any]
]:
    split_path = _resolve(config.repository_root, config.payload["split"]["path"])
    expected_split = str(config.payload["split"]["sha256"])
    if (
        _sha256_file(split_path) != expected_split
        or expected_split != EXPECTED_SPLIT_SHA256
    ):
        raise ExpertOptimizationError("DATA-007 split protection failed")
    loaded = load_activity_training_inputs(
        config.repository_root, expected_split_sha256=expected_split
    )
    predictions: dict[str, pd.DataFrame] = {}
    for expert in EXPERTS:
        path = _resolve(config.repository_root, PREDICTION_COLUMNS[expert])
        if _sha256_file(path) != PREDICTION_SHA256[expert]:
            raise ExpertOptimizationError(f"upstream {expert} OOF drifted")
        frame = pd.read_parquet(path).reset_index(drop=True)
        required = {
            "dataset_id",
            "global_participant_id",
            "canonical_row_index",
            "binary_target",
            "outer_fold",
            "raw_probability",
            "calibrated_probability",
            "expert_mask",
        }
        if required - set(frame.columns):
            raise ExpertOptimizationError(f"upstream {expert} OOF columns changed")
        if frame[["dataset_id", "canonical_row_index"]].duplicated().any():
            raise ExpertOptimizationError(
                f"upstream {expert} OOF identities duplicated"
            )
        predictions[expert] = frame
    assignments = loaded.assignments.copy()
    protections = {
        "split_sha256": expected_split,
        "prediction_sha256": dict(PREDICTION_SHA256),
        "canonical_input_bindings": {
            key: dict(value) for key, value in loaded.input_bindings.items()
        },
        "dataset_id_as_model_input": False,
        "feature_masks_as_risk_inputs": False,
        "feature_coverage_as_risk_input": False,
        "model006_online": False,
    }
    return predictions, loaded.frames, assignments, protections


def optimize_experts(
    config: ExpertOptimizationConfig,
    *,
    overwrite: bool = False,
    command: Sequence[str] = (),
) -> dict[str, Any]:
    predictions, frames, assignments, protections = load_expert_optimization_inputs(
        config
    )
    _check_outputs(config, overwrite=overwrite)
    all_oof: list[pd.DataFrame] = []
    all_search: list[dict[str, Any]] = []
    all_outer: list[dict[str, Any]] = []
    all_source: list[dict[str, Any]] = []
    all_ablation: list[dict[str, Any]] = []
    final_models: dict[str, CandidateFit] = {}
    summaries: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    for expert in EXPERTS:
        table, feature_frame = _prepare_expert_table(
            expert, predictions[expert], frames
        )
        result = _nested_expert_oof(expert, table, feature_frame, assignments, failures)
        all_oof.append(result["oof"])
        all_search.extend(result["search"])
        all_outer.extend(result["outer"])
        all_source.extend(result["source"])
        all_ablation.extend(result["ablation"])
        final_models[expert] = _fit_candidate(
            result["train_features"],
            result["train_target"],
            result["selected_candidate"],
            seed=RANDOM_SEED,
            candidate_label=expert,
        )
        summaries[expert] = result["summary"]
    oof = _concat_oof(all_oof)
    search = pd.DataFrame(all_search)
    outer = pd.DataFrame(all_outer)
    source = pd.DataFrame(all_source)
    ablation = pd.DataFrame(all_ablation)
    overall = _overall_metrics(oof)
    selected = {expert: summaries[expert]["selected_candidate"] for expert in EXPERTS}
    summary = {
        "task_id": TASK_ID,
        "run_id": RUN_ID,
        "status": "pass",
        "strict_oof": True,
        "strict_oof_rows_by_expert": {
            expert: int(predictions[expert]["expert_mask"].sum()) for expert in EXPERTS
        },
        "selected_candidates": selected,
        "promotion_status": "report_only_candidates_not_production",
        "model_families_compared": [
            "LightGBM",
            "CatBoost",
            "XGBoost_optional",
            "LightGBM_seed_bagging3",
        ],
        "failures": failures,
        "limitations": [
            "public strict OOF only",
            "no real household deployment labels",
            "Physiology remains excluded from this task",
        ],
    }
    _publish(
        config,
        final_models,
        search,
        oof,
        overall,
        outer,
        source,
        ablation,
        summary,
        protections,
        command,
    )
    return {
        "status": "pass",
        "task_id": TASK_ID,
        "run_id": RUN_ID,
        "selected_candidates": selected,
        "report_directory": str(config.report_directory),
        "model_path": str(config.model_path),
    }


def _prepare_expert_table(
    expert: str,
    predictions: pd.DataFrame,
    frames: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    for dataset_id, group in predictions.groupby("dataset_id", sort=True):
        source = frames.get(str(dataset_id))
        if source is None:
            raise ExpertOptimizationError(f"canonical source missing: {dataset_id}")
        indices = group["canonical_row_index"].to_numpy(dtype=int)
        if indices.max(initial=-1) >= len(source):
            raise ExpertOptimizationError(f"{expert} canonical row index out of range")
        selected = source.iloc[indices].reset_index(drop=True).copy()
        key = group[["dataset_id", "canonical_row_index"]].reset_index(drop=True)
        selected.insert(0, "_row_position", np.arange(len(selected)))
        selected["dataset_id"] = key["dataset_id"].to_numpy()
        selected["canonical_row_index"] = key["canonical_row_index"].to_numpy()
        selected["global_participant_id"] = key.assign(
            global_participant_id=group["global_participant_id"].to_numpy()
        )["global_participant_id"].to_numpy()
        rows.append(selected)
    canonical = (
        pd.concat(rows, ignore_index=True)
        .sort_values(["dataset_id", "canonical_row_index"], kind="stable")
        .reset_index(drop=True)
    )
    pred = predictions.sort_values(
        ["dataset_id", "canonical_row_index"], kind="stable"
    ).reset_index(drop=True)
    if not np.array_equal(
        canonical["global_participant_id"].astype(str),
        pred["global_participant_id"].astype(str),
    ):
        raise ExpertOptimizationError(
            f"{expert} canonical/prediction participant alignment changed"
        )
    features = _engineer_features(expert, canonical)
    base = pred[
        [
            "dataset_id",
            "canonical_row_index",
            "global_participant_id",
            "binary_target",
            "outer_fold",
            "raw_probability",
            "calibrated_probability",
            "expert_mask",
        ]
    ].copy()
    base["raw_logit"] = _logit(
        _clip_probability(base["raw_probability"].fillna(0.5).to_numpy(dtype=float))
    )
    features.index = base.index
    return base, features


def _engineer_features(expert: str, frame: pd.DataFrame) -> pd.DataFrame:
    prefixes = {
        "activity": ("activity.", "source__steps"),
        "sleep": ("sleep.", "source__sleep"),
        "joint": ("activity.", "sleep.", "source__steps", "source__sleep"),
    }[expert]
    columns = [
        column
        for column in frame.columns
        if any(column.startswith(prefix) for prefix in prefixes)
        and not column.startswith("feature_mask.")
    ]
    values = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    result = values.copy()
    result["temporal_summary_mean"] = values.mean(axis=1, skipna=True)
    result["temporal_summary_std"] = values.std(axis=1, skipna=True).fillna(0.0)
    result["temporal_summary_range"] = values.max(axis=1, skipna=True) - values.min(
        axis=1, skipna=True
    )
    result["temporal_summary_median"] = values.median(axis=1, skipna=True)
    if expert == "joint":
        interactions = {
            "activity_sleep_volume_efficiency": (
                "activity.activity_volume_norm",
                "sleep.sleep_efficiency",
            ),
            "sedentary_sleep_fragmentation": (
                "activity.sedentary_ratio",
                "sleep.sleep_fragmentation",
            ),
            "activity_sleep_regularity": (
                "activity.relative_amplitude",
                "sleep.sleep_regularity",
            ),
        }
        for name, (left, right) in interactions.items():
            if left in frame and right in frame:
                result[name] = pd.to_numeric(
                    frame[left], errors="coerce"
                ) * pd.to_numeric(frame[right], errors="coerce")
    return result.replace([np.inf, -np.inf], np.nan)


def _nested_expert_oof(
    expert: str,
    table: pd.DataFrame,
    features: pd.DataFrame,
    assignments: pd.DataFrame,
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    available = table["expert_mask"].astype(bool).to_numpy()
    if int(available.sum()) == 0:
        raise ExpertOptimizationError(f"{expert} has no available OOF rows")
    target = table["binary_target"].to_numpy(dtype=int)
    fold = table["outer_fold"].to_numpy(dtype=int)
    participant_to_inner = _inner_mapping(assignments)
    probability = {
        candidate: np.full(len(table), np.nan, dtype=float) for candidate in CANDIDATES
    }
    selected = np.full(len(table), "", dtype=object)
    search_rows: list[dict[str, Any]] = []
    outer_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    for outer_fold in range(OUTER_FOLDS):
        outer_test = available & (fold == outer_fold)
        outer_train = available & ~outer_test
        inner_rows: list[dict[str, Any]] = []
        candidate_fits: dict[str, CandidateFit] = {}
        for candidate in CANDIDATES:
            inner_pred = np.full(int(outer_train.sum()), np.nan, dtype=float)
            train_positions = np.flatnonzero(outer_train)
            try:
                for inner_fold in range(INNER_FOLDS):
                    validation = np.array(
                        [
                            participant_to_inner[str(value)][str(outer_fold)]
                            == inner_fold
                            for value in table.loc[
                                train_positions, "global_participant_id"
                            ]
                        ],
                        dtype=bool,
                    )
                    if not validation.any() or validation.all():
                        raise ExpertOptimizationError(
                            "inner participant fold coverage changed"
                        )
                    fit = _fit_candidate(
                        features.iloc[train_positions[~validation]],
                        target[train_positions[~validation]],
                        candidate,
                        seed=RANDOM_SEED + outer_fold * 31 + inner_fold,
                        candidate_label=f"{expert}_inner",
                    )
                    inner_pred[validation] = fit.predict(
                        features.iloc[train_positions[validation]]
                    )
                metrics = _metrics(target[train_positions], inner_pred)
                inner_rows.append(
                    {
                        "expert": expert,
                        "outer_fold": outer_fold,
                        "candidate_id": candidate,
                        "status": "pass",
                        **metrics,
                    }
                )
                fit = _fit_candidate(
                    features.iloc[train_positions],
                    target[train_positions],
                    candidate,
                    seed=RANDOM_SEED + outer_fold * 31,
                    candidate_label=f"{expert}_outer",
                )
                probability[candidate][outer_test] = fit.predict(
                    features.iloc[np.flatnonzero(outer_test)]
                )
                candidate_fits[candidate] = fit
            except CandidateUnavailable as exc:
                failures.append(
                    {
                        "expert": expert,
                        "outer_fold": outer_fold,
                        "candidate_id": candidate,
                        "stage": "nested_oof",
                        "error": str(exc),
                    }
                )
                inner_rows.append(
                    {
                        "expert": expert,
                        "outer_fold": outer_fold,
                        "candidate_id": candidate,
                        "status": "unavailable",
                        "error": str(exc),
                        "auprc": None,
                        "auroc": None,
                        "brier": None,
                    }
                )
            except (ExpertOptimizationError, ImportError, ValueError) as exc:
                failures.append(
                    {
                        "expert": expert,
                        "outer_fold": outer_fold,
                        "candidate_id": candidate,
                        "stage": "nested_oof",
                        "error": str(exc),
                    }
                )
                inner_rows.append(
                    {
                        "expert": expert,
                        "outer_fold": outer_fold,
                        "candidate_id": candidate,
                        "status": "failed",
                        "error": str(exc),
                        "auprc": None,
                        "auroc": None,
                        "brier": None,
                    }
                )
        valid = [row for row in inner_rows if row["status"] == "pass"]
        if not valid:
            raise ExpertOptimizationError(
                f"{expert} has no successful candidate in outer fold {outer_fold}"
            )
        chosen = sorted(
            valid, key=lambda row: (-row["auprc"], row["brier"], row["candidate_id"])
        )[0]["candidate_id"]
        selected[outer_test] = chosen
        for row in inner_rows:
            row["selected"] = row["candidate_id"] == chosen
            search_rows.append(row)
        chosen_probability = probability[chosen][outer_test]
        y_test = target[outer_test]
        outer_metrics = _metrics(y_test, chosen_probability)
        outer_rows.append(
            {
                "expert": expert,
                "outer_fold": outer_fold,
                "selected_candidate": chosen,
                **outer_metrics,
                "row_count": int(outer_test.sum()),
                "positive_row_count": int(y_test.sum()),
            }
        )
        for dataset_id, index in (
            table.loc[outer_test].groupby("dataset_id").groups.items()
        ):
            source_rows.append(
                {
                    "expert": expert,
                    "outer_fold": outer_fold,
                    "dataset_id": str(dataset_id),
                    "selected_candidate": chosen,
                    **_metrics(
                        target[np.asarray(index)],
                        chosen_probability[
                            np.searchsorted(
                                np.flatnonzero(outer_test), np.asarray(index)
                            )
                        ],
                    ),
                }
            )
        ablation_rows.append(
            {
                "expert": expert,
                "outer_fold": outer_fold,
                "ablation": "base_calibrated",
                **_metrics(
                    y_test,
                    table.loc[outer_test, "calibrated_probability"].to_numpy(
                        dtype=float
                    ),
                ),
            }
        )
        ablation_rows.append(
            {
                "expert": expert,
                "outer_fold": outer_fold,
                "ablation": "temporal_engineered_candidate",
                **outer_metrics,
            }
        )
    selected_available = np.array([value != "" for value in selected], dtype=bool)
    selected_probability = np.full(len(table), np.nan, dtype=float)
    for candidate in CANDIDATES:
        mask = selected == candidate
        selected_probability[mask] = probability[candidate][mask]
    oof = table.copy()
    oof.insert(0, "expert", expert)
    for candidate in CANDIDATES:
        oof[f"probability__{candidate}"] = probability[candidate]
    oof["selected_candidate"] = selected
    oof["selected_probability"] = selected_probability
    oof.loc[~selected_available, "selected_probability"] = np.nan
    valid_selection = [row for row in search_rows if row.get("status") == "pass"]
    counts = (
        pd.DataFrame(valid_selection)
        .groupby("candidate_id")["selected"]
        .sum()
        .to_dict()
    )
    selected_global = sorted(
        counts,
        key=lambda value: (-int(counts[value]), value),
    )[0]
    summary = {
        "expert": expert,
        "selected_candidate": selected_global,
        "outer_fold_selection_counts": {
            key: int(value) for key, value in counts.items()
        },
        "available_row_count": int(available.sum()),
        "overall_metrics": _metrics(target[available], selected_probability[available]),
    }
    train_positions = np.flatnonzero(available)
    return {
        "oof": oof,
        "search": search_rows,
        "outer": outer_rows,
        "source": source_rows,
        "ablation": ablation_rows,
        "summary": summary,
        "selected_candidate": selected_global,
        "train_features": features.iloc[train_positions],
        "train_target": target[train_positions],
    }


def _fit_candidate(
    features: pd.DataFrame,
    target: np.ndarray,
    candidate: str,
    *,
    seed: int,
    candidate_label: str,
) -> CandidateFit:
    if candidate not in CANDIDATES:
        raise ExpertOptimizationError(f"unknown expert candidate: {candidate}")
    names = tuple(str(value) for value in features.columns)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    matrix = np.asarray(imputer.fit_transform(features.loc[:, names]), dtype=float)
    if not np.isfinite(matrix).all() or set(np.unique(target)) != {0, 1}:
        raise ExpertOptimizationError("expert candidate training matrix is invalid")
    if candidate.startswith("lgbm"):
        from lightgbm import LGBMClassifier

        def build(local_seed: int) -> Any:
            return LGBMClassifier(
                n_estimators=120,
                learning_rate=0.035,
                num_leaves=15,
                max_depth=-1,
                min_child_samples=30,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                random_state=local_seed,
                n_jobs=1,
                verbosity=-1,
                force_col_wise=True,
            )

        estimator = (
            [build(seed + offset) for offset in range(3)]
            if candidate.endswith("bagging3")
            else build(seed)
        )
        models = estimator if isinstance(estimator, list) else [estimator]
        for model in models:
            model.fit(matrix, target)
    elif candidate == "catboost_temporal":
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:
            raise CandidateUnavailable("CatBoost is not installed") from exc
        estimator = CatBoostClassifier(
            iterations=140,
            depth=5,
            learning_rate=0.035,
            loss_function="Logloss",
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
            thread_count=1,
            l2_leaf_reg=3.0,
        )
        estimator.fit(matrix, target)
    elif candidate == "xgboost_temporal":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise CandidateUnavailable(
                "XGBoost is not installed in eldercare-ai"
            ) from exc
        estimator = XGBClassifier(
            n_estimators=120,
            max_depth=3,
            learning_rate=0.035,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=1,
        )
        estimator.fit(matrix, target)
    else:
        raise ExpertOptimizationError(f"unsupported expert candidate: {candidate}")
    if isinstance(estimator, list):
        raw_train = np.mean(
            [model.predict_proba(matrix)[:, 1] for model in estimator], axis=0
        )
    else:
        raw_train = estimator.predict_proba(matrix)[:, 1]
    calibrator = LogisticRegression(
        C=1.0, solver="lbfgs", max_iter=1000, random_state=seed
    )
    calibrator.fit(_logit(_clip_probability(raw_train))[:, None], target)
    return CandidateFit(candidate, imputer, estimator, calibrator, names)


def _metrics(target: Sequence[int], probability: Sequence[float]) -> dict[str, float]:
    y = np.asarray(target, dtype=int)
    p = _clip_probability(np.asarray(probability, dtype=float))
    if (
        len(y) != len(p)
        or len(y) == 0
        or not np.isfinite(p).all()
        or set(np.unique(y)) != {0, 1}
    ):
        raise ExpertOptimizationError("metric input is invalid")
    return {
        "auprc": float(average_precision_score(y, p)),
        "auroc": float(roc_auc_score(y, p)),
        "brier": float(np.mean((p - y) ** 2)),
    }


def _overall_metrics(oof: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for expert in EXPERTS:
        part = oof[oof["expert"].eq(expert) & oof["expert_mask"].eq(1)]
        y = part["binary_target"].to_numpy(dtype=int)
        rows.append(
            {
                "expert": expert,
                "candidate_id": "base_calibrated",
                **_metrics(y, part["calibrated_probability"]),
            }
        )
        for candidate in CANDIDATES:
            values = part[f"probability__{candidate}"].dropna().to_numpy(dtype=float)
            if len(values) == len(part):
                rows.append(
                    {"expert": expert, "candidate_id": candidate, **_metrics(y, values)}
                )
        selected = part["selected_probability"].dropna().to_numpy(dtype=float)
        rows.append(
            {
                "expert": expert,
                "candidate_id": "selected_candidate",
                **_metrics(y, selected),
            }
        )
    return pd.DataFrame(rows)


def _concat_oof(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    common = [
        "expert",
        "dataset_id",
        "global_participant_id",
        "canonical_row_index",
        "binary_target",
        "outer_fold",
        "raw_probability",
        "calibrated_probability",
        "expert_mask",
        "selected_candidate",
        "selected_probability",
    ]
    candidate_cols = [f"probability__{candidate}" for candidate in CANDIDATES]
    return pd.concat(
        [frame.loc[:, common + candidate_cols] for frame in frames], ignore_index=True
    )


def _inner_mapping(assignments: pd.DataFrame) -> dict[str, Mapping[str, Any]]:
    result = {}
    for row in assignments.to_dict(orient="records"):
        mapping = dict(row["inner_validation_fold_by_outer_fold"])
        if mapping.get(str(row["outer_fold"])) is not None:
            raise ExpertOptimizationError("outer-test participant entered inner fold")
        result[str(row["global_participant_id"])] = mapping
    return result


def _publish(
    config: ExpertOptimizationConfig,
    models: Mapping[str, CandidateFit],
    search: pd.DataFrame,
    oof: pd.DataFrame,
    overall: pd.DataFrame,
    outer: pd.DataFrame,
    source: pd.DataFrame,
    ablation: pd.DataFrame,
    summary: Mapping[str, Any],
    protections: Mapping[str, Any],
    command: Sequence[str],
) -> None:
    report = config.report_directory
    report.mkdir(parents=True, exist_ok=True)
    config.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(models), config.model_path, compress=3)
    manifest = {
        "task_id": TASK_ID,
        "run_id": RUN_ID,
        "model_version": MODEL_VERSION,
        "strict_oof": True,
        "candidate_scope": "offline_candidate",
        "production": False,
        "experts": list(EXPERTS),
        "candidates": list(CANDIDATES),
        "selected_candidates": dict(summary["selected_candidates"]),
        "dataset_id_as_model_input": False,
        "feature_masks_as_risk_inputs": False,
        "feature_coverage_as_risk_input": False,
        "model006_online": False,
        "model_sha256": _sha256_file(config.model_path),
    }
    _write_json(config.manifest_path, manifest)
    _write_parquet(report / "candidate_search.parquet", search)
    _write_parquet(report / "oof_predictions.parquet", oof)
    _write_parquet(report / "overall_metrics.parquet", overall)
    _write_parquet(report / "outer_fold_stability.parquet", outer)
    _write_parquet(report / "source_robustness.parquet", source)
    _write_parquet(report / "ablation.parquet", ablation)
    _write_json(report / "summary.json", summary)
    _write_json(report / "upstream_protection.json", protections)
    _write_json(report / "config.json", config.payload)
    _write_json(
        report / "run.json",
        {
            "task_id": TASK_ID,
            "run_id": RUN_ID,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "command": list(command),
        },
    )
    _write_json(
        report / "model_card.json",
        {
            "task_id": TASK_ID,
            "run_id": RUN_ID,
            "model_version": MODEL_VERSION,
            "production_status": "offline_candidate_not_production",
            "selected_candidates": dict(summary["selected_candidates"]),
            "feature_policy": "canonical temporal/rhythm summaries; no dataset_id, mask or coverage risk inputs",
        },
    )
    (report / "model_card.md").write_text(
        "# OPT-EXPERT-001 candidates\n\nOffline candidates for Activity, Sleep and Joint. Dataset identifiers and availability masks are alignment controls only.\n",
        encoding="utf-8",
    )
    artifacts = [
        {"path": path.name, "bytes": path.stat().st_size, "sha256": _sha256_file(path)}
        for path in sorted(report.iterdir(), key=lambda item: item.name.encode("utf-8"))
        if path.is_file() and path.name != "artifacts.json"
    ]
    _write_json(
        report / "artifacts.json",
        {
            "version": "mood-social-opt-expert-artifacts-v1",
            "artifacts": artifacts,
            "model": {
                "path": config.model_path.name,
                "sha256": _sha256_file(config.model_path),
            },
            "manifest": {
                "path": config.manifest_path.name,
                "sha256": _sha256_file(config.manifest_path),
            },
        },
    )
    sums = [
        f"{_sha256_file(path)}  {path.name}"
        for path in sorted(
            config.model_path.parent.iterdir(),
            key=lambda item: item.name.encode("utf-8"),
        )
        if path.is_file() and path.name != "SHA256SUMS"
    ]
    (config.model_path.parent / "SHA256SUMS").write_text(
        "\n".join(sums) + "\n", encoding="utf-8"
    )


def _check_outputs(config: ExpertOptimizationConfig, *, overwrite: bool) -> None:
    paths = (config.report_directory, config.model_path, config.manifest_path)
    if not overwrite and any(path.exists() for path in paths):
        raise ExpertOptimizationError("OPT-EXPERT-001 output exists; use --overwrite")


def _clip_probability(value: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(value, dtype=float), EPSILON, 1.0 - EPSILON)


def _logit(value: np.ndarray) -> np.ndarray:
    p = _clip_probability(value)
    return np.log(p / (1.0 - p))


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, indent=2, default=_json_default
        )
        + "\n",
        encoding="utf-8",
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    frame.to_parquet(path, index=False)


__all__ = [
    "CANDIDATES",
    "EXPERTS",
    "ExpertOptimizationConfig",
    "ExpertOptimizationError",
    "RUN_ID",
    "TASK_ID",
    "load_expert_optimization_config",
    "load_expert_optimization_inputs",
    "optimize_experts",
]
