"""Deployment-safe strict OOF optimization for OPT-FUSION-002."""

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
import yaml

from .masked_stacking import (
    INPUT_FEATURES,
    build_fusion_features,
    effective_evidence,
    evidence_combination,
    masked_stacking_sample_weights,
)


TASK_ID = "OPT-FUSION-002"
RUN_ID = "MH-20260802-012"
TRAINING_VERSION = "mood-social-fusion-deployment-optimization-v3.3.3-v1"
MODEL_VERSION = "mood-social-deployment-safe-fusion-candidate-v1"
MANIFEST_VERSION = "mood-social-deployment-safe-fusion-manifest-v1"
REPORT_VERSION = "mood-social-deployment-safe-fusion-report-v1"
RANDOM_SEED = 20260728
OUTER_FOLDS = 5
INNER_FOLDS = 5
EPSILON = 1.0e-6
MULTISEEDS = (20260728, 20260729, 20260730)
CANDIDATE_IDS = (
    "baseline_refit",
    "interactions",
    "reliability_gated",
    "global_calibrated",
    "coverage_calibrated",
    "probability_average_25",
    "probability_average_50",
    "probability_average_75",
    "rank_average",
    "multiseed_interactions",
)
BRANCH_NAMES = (
    "activity",
    "sleep",
    "joint",
    "physiology",
    "social_context",
    "trend_activity",
    "trend_sleep",
    "trend_social",
)
WEAK_BRANCH_SCALES = {
    "physiology": 0.25,
    "social_context": 0.50,
    "trend_social": 0.25,
}


class DeploymentOptimizationError(RuntimeError):
    """Raised when the frozen OPT-FUSION-002 protocol is violated."""


@dataclass(frozen=True)
class DeploymentOptimizationConfig:
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


@dataclass(frozen=True)
class DeploymentSafeFusionModel:
    """Source-independent fusion candidate suitable for one-row inference."""

    task_id: str
    run_id: str
    model_version: str
    candidate_id: str
    primary_kind: str
    primary_classifiers: tuple[LogisticRegression, ...]
    secondary_kind: str | None
    secondary_classifiers: tuple[LogisticRegression, ...]
    blend_kind: str
    primary_weight: float
    branch_scales: Mapping[str, float]
    global_calibrator: LogisticRegression | None
    coverage_calibrators: Mapping[str, LogisticRegression]
    primary_rank_reference: tuple[float, ...]
    secondary_rank_reference: tuple[float, ...]

    def validate(self) -> None:
        if self.task_id != TASK_ID or self.run_id != RUN_ID:
            raise DeploymentOptimizationError("deployment candidate identity changed")
        if (
            self.model_version != MODEL_VERSION
            or self.candidate_id not in CANDIDATE_IDS
        ):
            raise DeploymentOptimizationError("deployment candidate version changed")
        if self.primary_kind not in {"base", "interactions", "reliability_gated"}:
            raise DeploymentOptimizationError("primary feature kind is invalid")
        if not self.primary_classifiers:
            raise DeploymentOptimizationError("primary classifiers are missing")
        if any(
            not isinstance(model, LogisticRegression)
            for model in self.primary_classifiers
        ):
            raise DeploymentOptimizationError("unsupported primary classifier")
        if self.secondary_kind is None and self.secondary_classifiers:
            raise DeploymentOptimizationError("secondary classifier contract changed")
        if self.secondary_kind is not None and not self.secondary_classifiers:
            raise DeploymentOptimizationError("secondary classifiers are missing")
        if self.blend_kind not in {"none", "probability", "rank"}:
            raise DeploymentOptimizationError("blend kind is invalid")
        if not 0.0 <= self.primary_weight <= 1.0:
            raise DeploymentOptimizationError("blend weight is invalid")
        if set(self.branch_scales) - set(BRANCH_NAMES):
            raise DeploymentOptimizationError("unknown branch scale")
        if any(
            "dataset" in key.lower() or "source" in key.lower()
            for key in self.coverage_calibrators
        ):
            raise DeploymentOptimizationError(
                "public source calibration entered candidate"
            )

    def predict_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        self.validate()
        evidence = effective_evidence(frame)
        available = evidence > 0.0
        probability = np.full(len(frame), np.nan, dtype=float)
        if np.any(available):
            selected = frame.loc[available]
            primary = _component_probability(
                self.primary_classifiers,
                self.primary_kind,
                selected,
                self.branch_scales,
            )
            combined = primary
            if self.secondary_kind is not None:
                secondary = _component_probability(
                    self.secondary_classifiers,
                    self.secondary_kind,
                    selected,
                    {},
                )
                if self.blend_kind == "rank":
                    primary = _apply_rank_reference(
                        primary, self.primary_rank_reference
                    )
                    secondary = _apply_rank_reference(
                        secondary, self.secondary_rank_reference
                    )
                combined = (
                    self.primary_weight * primary
                    + (1.0 - self.primary_weight) * secondary
                )
            combined = np.clip(combined, EPSILON, 1.0 - EPSILON)
            if self.global_calibrator is not None:
                combined = self.global_calibrator.predict_proba(
                    _logit(combined)[:, None]
                )[:, 1]
            if self.coverage_calibrators:
                modes = evidence_combination(selected).astype(str).to_numpy()
                calibrated = combined.copy()
                for mode, calibrator in self.coverage_calibrators.items():
                    mask = modes == mode
                    if mask.any():
                        calibrated[mask] = calibrator.predict_proba(
                            _logit(combined[mask])[:, None]
                        )[:, 1]
                combined = calibrated
            probability[available] = np.clip(combined, EPSILON, 1.0 - EPSILON)
        return pd.DataFrame(
            {
                "fusion_probability": probability,
                "effective_evidence": evidence,
                "available": available,
            },
            index=frame.index,
        )


def load_deployment_optimization_config(
    path: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> DeploymentOptimizationConfig:
    config_path = Path(path).resolve()
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise DeploymentOptimizationError(
            "OPT-FUSION-002 config is unreadable"
        ) from exc
    if not isinstance(payload, Mapping):
        raise DeploymentOptimizationError("OPT-FUSION-002 config must be a mapping")
    if payload.get("task_id") != TASK_ID or payload.get("run_id") != RUN_ID:
        raise DeploymentOptimizationError("OPT-FUSION-002 task or run ID changed")
    if int(payload.get("random_seed", -1)) != RANDOM_SEED:
        raise DeploymentOptimizationError("OPT-FUSION-002 random seed changed")
    if tuple(payload.get("candidates", {}).get("ids", ())) != CANDIDATE_IDS:
        raise DeploymentOptimizationError("deployment candidate grid changed")
    if tuple(payload.get("candidates", {}).get("multiseed_seeds", ())) != MULTISEEDS:
        raise DeploymentOptimizationError("deployment multiseed grid changed")
    boundary = payload.get("production_boundary", {})
    required_false = (
        "consume_public_dataset_id",
        "select_production_threshold",
        "change_active_experts",
        "change_personal_trend",
        "include_model006_predictions",
        "change_http_behavior",
        "change_attention_levels",
        "overwrite_existing_models",
    )
    if any(boundary.get(key) is not False for key in required_false):
        raise DeploymentOptimizationError("OPT-FUSION-002 production boundary changed")
    if boundary.get("deployment_safe_candidate") is not True:
        raise DeploymentOptimizationError("deployment-safe candidate flag changed")
    root = Path(repository_root or config_path.parents[2]).resolve()
    return DeploymentOptimizationConfig(root, payload, config_path)


def load_deployment_optimization_inputs(
    config: DeploymentOptimizationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    input_config = config.payload["input"]
    table_path = _resolve(config.repository_root, input_config["fusion_oof_table_path"])
    baseline_path = _resolve(config.repository_root, input_config["baseline_oof_path"])
    optimization_path = _resolve(
        config.repository_root, input_config["optimization_oof_path"]
    )
    optimization_model_path = _resolve(
        config.repository_root,
        input_config["optimization_model_path"],
    )
    split_path = _resolve(config.repository_root, input_config["split"]["path"])
    protected = {
        table_path: str(input_config["fusion_oof_table_sha256"]),
        baseline_path: str(input_config["baseline_oof_sha256"]),
        optimization_path: str(input_config["optimization_oof_sha256"]),
        optimization_model_path: str(input_config["optimization_model_sha256"]),
        split_path: str(input_config["split"]["sha256"]),
    }
    for path, expected in protected.items():
        if not path.is_file() or _sha256_file(path) != expected:
            raise DeploymentOptimizationError(f"protected input drifted: {path.name}")
    table = pd.read_parquet(table_path).reset_index(drop=True)
    baseline = pd.read_parquet(baseline_path)
    optimization = pd.read_parquet(optimization_path)
    if len(table) != 22191 or table["global_participant_id"].nunique() != 15361:
        raise DeploymentOptimizationError("FUSION-001 row or participant count changed")
    keys = ["dataset_id", "canonical_row_index"]
    frame = table.merge(
        baseline[keys + ["fusion_probability"]],
        on=keys,
        how="left",
        validate="one_to_one",
    ).rename(columns={"fusion_probability": "baseline_probability"})
    frame = frame.merge(
        optimization[keys + ["fusion_v2_probability"]],
        on=keys,
        how="left",
        validate="one_to_one",
    ).rename(columns={"fusion_v2_probability": "offline_reference_probability"})
    frame = frame.loc[effective_evidence(frame) > 0].reset_index(drop=True)
    if (
        len(frame) != 22188
        or frame[["baseline_probability", "offline_reference_probability"]]
        .isna()
        .any()
        .any()
    ):
        raise DeploymentOptimizationError("strict OOF probability alignment changed")
    split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    assignments = pd.DataFrame(split_payload.get("participant_assignments", []))
    if assignments.empty or assignments["global_participant_id"].duplicated().any():
        raise DeploymentOptimizationError("DATA-007 assignments are invalid")
    assignments["global_participant_id"] = assignments["global_participant_id"].astype(
        str
    )
    assignments = assignments.set_index("global_participant_id")
    if set(frame["global_participant_id"].astype(str)) - set(assignments.index):
        raise DeploymentOptimizationError(
            "strict OOF participants are missing from DATA-007"
        )
    protection = {
        "fusion_oof_table_sha256": protected[table_path],
        "baseline_oof_sha256": protected[baseline_path],
        "optimization_oof_sha256": protected[optimization_path],
        "optimization_model_sha256": protected[optimization_model_path],
        "split_sha256": protected[split_path],
        "public_dataset_id_as_model_input": False,
        "model006_enters_fusion": False,
    }
    return frame, assignments, protection


def optimize_deployment_fusion(
    config: DeploymentOptimizationConfig,
    *,
    overwrite: bool = False,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    frame, assignments, protection = load_deployment_optimization_inputs(config)
    _validate_output_paths(config, overwrite=overwrite)
    search, oof, selected_id, model = _nested_search(frame, assignments)
    overall = _overall_metrics(frame, oof)
    outer = _outer_stability(frame, oof)
    source = _source_stability(frame, selected_id)
    promotion = _promotion_decision(config, overall, outer, source)
    _publish(
        config,
        frame,
        search,
        oof,
        overall,
        outer,
        source,
        promotion,
        model,
        protection,
        command or (),
    )
    return {
        "status": "pass",
        "task_id": TASK_ID,
        "run_id": RUN_ID,
        "selected_candidate_id": selected_id,
        "promotion_passed": bool(promotion["promotion_passed"]),
        "model_path": str(config.model_path),
        "report_directory": str(config.report_directory),
    }


def _nested_search(
    frame: pd.DataFrame,
    assignments: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, str, DeploymentSafeFusionModel]:
    rows: list[dict[str, Any]] = []
    selected_by_outer: list[str] = []
    prediction = np.full(len(frame), np.nan, dtype=float)
    selected_for_row = np.full(len(frame), "", dtype=object)
    for outer_fold in range(OUTER_FOLDS):
        outer_test = frame["outer_fold"].astype(int).to_numpy() == outer_fold
        outer_train = frame.loc[~outer_test]
        candidate_scores: list[dict[str, Any]] = []
        for candidate_id in CANDIDATE_IDS:
            inner_probability = np.full(len(outer_train), np.nan, dtype=float)
            for inner_fold in range(INNER_FOLDS):
                inner_test = np.array(
                    [
                        _inner_fold_for(assignments, participant, outer_fold)
                        == inner_fold
                        for participant in outer_train["global_participant_id"].astype(
                            str
                        )
                    ],
                    dtype=bool,
                )
                if not inner_test.any() or inner_test.all():
                    raise DeploymentOptimizationError(
                        "DATA-007 inner fold coverage changed"
                    )
                fitted = _fit_candidate(outer_train.loc[~inner_test], candidate_id)
                inner_probability[inner_test] = fitted.predict_frame(
                    outer_train.loc[inner_test]
                )["fusion_probability"].to_numpy(dtype=float)
            if not np.isfinite(inner_probability).all():
                raise DeploymentOptimizationError(
                    "inner OOF predictions are incomplete"
                )
            metrics = _metrics(
                outer_train["binary_target"].to_numpy(dtype=int),
                inner_probability,
            )
            candidate_scores.append({"candidate_id": candidate_id, **metrics})
        selected = sorted(
            candidate_scores,
            key=lambda row: (-row["auprc"], row["brier"], row["candidate_id"]),
        )[0]["candidate_id"]
        selected_by_outer.append(str(selected))
        fitted = _fit_candidate(outer_train, str(selected))
        prediction[outer_test] = fitted.predict_frame(frame.loc[outer_test])[
            "fusion_probability"
        ].to_numpy(dtype=float)
        selected_for_row[outer_test] = str(selected)
        rows.extend(
            {
                "outer_fold": outer_fold,
                **row,
                "selected": row["candidate_id"] == selected,
            }
            for row in candidate_scores
        )
    search = pd.DataFrame(rows)
    production_rows = []
    for candidate_id in CANDIDATE_IDS:
        subset = search[search["candidate_id"].eq(candidate_id)]
        production_rows.append(
            {
                "candidate_id": candidate_id,
                "selection_count": int(
                    sum(value == candidate_id for value in selected_by_outer)
                ),
                "mean_inner_auprc": float(subset["auprc"].mean()),
                "mean_inner_brier": float(subset["brier"].mean()),
            }
        )
    selected_id = sorted(
        production_rows,
        key=lambda row: (
            -row["selection_count"],
            -row["mean_inner_auprc"],
            row["mean_inner_brier"],
            row["candidate_id"],
        ),
    )[0]["candidate_id"]
    search = pd.concat(
        [
            search,
            pd.DataFrame(
                [
                    {
                        "outer_fold": -1,
                        **row,
                        "selected": row["candidate_id"] == selected_id,
                    }
                    for row in production_rows
                ]
            ),
        ],
        ignore_index=True,
        sort=False,
    )
    oof = frame[
        [
            "dataset_id",
            "canonical_row_index",
            "global_participant_id",
            "binary_target",
            "outer_fold",
            "baseline_probability",
            "offline_reference_probability",
        ]
    ].copy()
    oof["deployment_candidate_probability"] = prediction
    oof["selected_candidate_id"] = selected_for_row
    model = _fit_candidate(frame, selected_id)
    return search, oof, selected_id, model


def _fit_candidate(frame: pd.DataFrame, candidate_id: str) -> DeploymentSafeFusionModel:
    if candidate_id not in CANDIDATE_IDS:
        raise DeploymentOptimizationError(
            f"unknown deployment candidate: {candidate_id}"
        )
    target = frame["binary_target"].to_numpy(dtype=int)
    weight = masked_stacking_sample_weights(frame)
    primary_kind = "interactions"
    secondary_kind: str | None = None
    blend_kind = "none"
    primary_weight = 1.0
    branch_scales: dict[str, float] = {}
    seeds = (RANDOM_SEED,)
    if candidate_id == "baseline_refit":
        primary_kind = "base"
    elif candidate_id == "reliability_gated":
        primary_kind = "reliability_gated"
        branch_scales = dict(WEAK_BRANCH_SCALES)
    elif candidate_id.startswith("probability_average_"):
        secondary_kind = "base"
        blend_kind = "probability"
        primary_weight = int(candidate_id.rsplit("_", 1)[1]) / 100.0
    elif candidate_id == "rank_average":
        secondary_kind = "base"
        blend_kind = "rank"
        primary_weight = 0.5
    elif candidate_id == "multiseed_interactions":
        seeds = MULTISEEDS
    primary_matrix = _feature_matrix(frame, primary_kind, branch_scales)
    primary_models = tuple(
        _fit_logistic(primary_matrix, target, weight, seed, multiseed=len(seeds) > 1)
        for seed in seeds
    )
    secondary_models: tuple[LogisticRegression, ...] = ()
    if secondary_kind is not None:
        secondary_models = (
            _fit_logistic(
                _feature_matrix(frame, secondary_kind, {}),
                target,
                weight,
                RANDOM_SEED,
                multiseed=False,
            ),
        )
    primary_probability = _average_probability(primary_models, primary_matrix)
    combined = primary_probability.copy()
    primary_reference: tuple[float, ...] = ()
    secondary_reference: tuple[float, ...] = ()
    if secondary_kind is not None:
        secondary_probability = _average_probability(
            secondary_models,
            _feature_matrix(frame, secondary_kind, {}),
        )
        if blend_kind == "rank":
            primary_reference = tuple(
                sorted(float(value) for value in primary_probability)
            )
            secondary_reference = tuple(
                sorted(float(value) for value in secondary_probability)
            )
            primary_probability = _apply_rank_reference(
                primary_probability, primary_reference
            )
            secondary_probability = _apply_rank_reference(
                secondary_probability, secondary_reference
            )
        combined = (
            primary_weight * primary_probability
            + (1.0 - primary_weight) * secondary_probability
        )
    global_calibrator: LogisticRegression | None = None
    coverage_calibrators: dict[str, LogisticRegression] = {}
    if candidate_id in {"global_calibrated", "coverage_calibrated", "rank_average"}:
        global_calibrator = _fit_platt(combined, target, weight)
    if candidate_id == "coverage_calibrated":
        modes = evidence_combination(frame).astype(str)
        for mode in sorted(modes.unique(), key=lambda value: value.encode("utf-8")):
            selected = modes.eq(mode).to_numpy()
            positives = int(target[selected].sum())
            negatives = int(selected.sum() - positives)
            if selected.sum() >= 50 and positives >= 5 and negatives >= 5:
                coverage_calibrators[mode] = _fit_platt(
                    combined[selected],
                    target[selected],
                    weight[selected],
                )
    model = DeploymentSafeFusionModel(
        task_id=TASK_ID,
        run_id=RUN_ID,
        model_version=MODEL_VERSION,
        candidate_id=candidate_id,
        primary_kind=primary_kind,
        primary_classifiers=primary_models,
        secondary_kind=secondary_kind,
        secondary_classifiers=secondary_models,
        blend_kind=blend_kind,
        primary_weight=primary_weight,
        branch_scales=branch_scales,
        global_calibrator=global_calibrator,
        coverage_calibrators=coverage_calibrators,
        primary_rank_reference=primary_reference,
        secondary_rank_reference=secondary_reference,
    )
    model.validate()
    return model


def _feature_matrix(
    frame: pd.DataFrame,
    kind: str,
    branch_scales: Mapping[str, float],
) -> np.ndarray:
    base = build_fusion_features(frame)
    if kind == "base":
        return base
    if kind == "reliability_gated":
        base = base.copy()
        for index, name in enumerate(BRANCH_NAMES):
            base[:, index] *= float(branch_scales.get(name, 1.0))
    return _interaction_features(base)


def _interaction_features(base: np.ndarray) -> np.ndarray:
    interactions = np.column_stack(
        [
            base[:, 0] * base[:, 1],
            base[:, 0] * base[:, 2],
            base[:, 1] * base[:, 2],
            base[:, 0] * base[:, 5],
            base[:, 1] * base[:, 6],
            base[:, 4] * base[:, 7],
        ]
    )
    return np.column_stack([base, interactions])


def _fit_logistic(
    matrix: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    seed: int,
    *,
    multiseed: bool,
) -> LogisticRegression:
    model = LogisticRegression(
        C=10.0,
        solver="saga" if multiseed else "lbfgs",
        max_iter=5000,
        tol=1.0e-6 if multiseed else 1.0e-10,
        random_state=seed,
    )
    model.fit(matrix, target, sample_weight=weight)
    return model


def _fit_platt(
    probability: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
) -> LogisticRegression:
    model = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=2000,
        tol=1.0e-10,
        random_state=RANDOM_SEED,
    )
    model.fit(_logit(probability)[:, None], target, sample_weight=weight)
    return model


def _component_probability(
    models: Sequence[LogisticRegression],
    kind: str,
    frame: pd.DataFrame,
    scales: Mapping[str, float],
) -> np.ndarray:
    return _average_probability(models, _feature_matrix(frame, kind, scales))


def _average_probability(
    models: Sequence[LogisticRegression],
    matrix: np.ndarray,
) -> np.ndarray:
    return np.mean([model.predict_proba(matrix)[:, 1] for model in models], axis=0)


def _apply_rank_reference(
    probability: np.ndarray,
    reference: Sequence[float],
) -> np.ndarray:
    values = np.asarray(reference, dtype=float)
    if values.size == 0:
        raise DeploymentOptimizationError("rank reference is missing")
    rank = np.searchsorted(values, probability, side="right") / float(values.size)
    return np.clip(rank, EPSILON, 1.0 - EPSILON)


def _overall_metrics(frame: pd.DataFrame, oof: pd.DataFrame) -> pd.DataFrame:
    target = frame["binary_target"].to_numpy(dtype=int)
    rows = []
    for name, column in (
        ("fusion_002_baseline", "baseline_probability"),
        ("opt_fusion_001_offline_reference", "offline_reference_probability"),
        ("deployment_candidate", "deployment_candidate_probability"),
    ):
        rows.append(
            {"model": name, **_metrics(target, oof[column].to_numpy(dtype=float))}
        )
    return pd.DataFrame(rows)


def _outer_stability(frame: pd.DataFrame, oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for outer_fold in range(OUTER_FOLDS):
        selected = frame["outer_fold"].astype(int).eq(outer_fold).to_numpy()
        target = frame.loc[selected, "binary_target"].to_numpy(dtype=int)
        baseline = _metrics(
            target, oof.loc[selected, "baseline_probability"].to_numpy(dtype=float)
        )
        candidate = _metrics(
            target,
            oof.loc[selected, "deployment_candidate_probability"].to_numpy(dtype=float),
        )
        rows.append(
            {
                "outer_fold": outer_fold,
                "row_count": int(selected.sum()),
                "positive_rows": int(target.sum()),
                "selected_candidate_id": str(
                    oof.loc[selected, "selected_candidate_id"].iloc[0]
                ),
                "baseline_auprc": baseline["auprc"],
                "candidate_auprc": candidate["auprc"],
                "auprc_delta": candidate["auprc"] - baseline["auprc"],
                "baseline_auroc": baseline["auroc"],
                "candidate_auroc": candidate["auroc"],
                "baseline_brier": baseline["brier"],
                "candidate_brier": candidate["brier"],
            }
        )
    return pd.DataFrame(rows)


def _source_stability(frame: pd.DataFrame, candidate_id: str) -> pd.DataFrame:
    rows = []
    sources = sorted(
        frame["dataset_id"].astype(str).unique(),
        key=lambda value: value.encode("utf-8"),
    )
    for source in sources:
        for outer_fold in range(OUTER_FOLDS):
            train = frame[
                frame["outer_fold"].astype(int).ne(outer_fold)
                & frame["dataset_id"].astype(str).ne(source)
            ]
            test = frame[
                frame["outer_fold"].astype(int).eq(outer_fold)
                & frame["dataset_id"].astype(str).eq(source)
            ]
            if train.empty or test.empty or train["binary_target"].nunique() < 2:
                continue
            target = test["binary_target"].to_numpy(dtype=int)
            if np.unique(target).size < 2:
                continue
            baseline_model = _fit_candidate(train, "baseline_refit")
            candidate_model = _fit_candidate(train, candidate_id)
            baseline_probability = baseline_model.predict_frame(test)[
                "fusion_probability"
            ].to_numpy(dtype=float)
            candidate_probability = candidate_model.predict_frame(test)[
                "fusion_probability"
            ].to_numpy(dtype=float)
            baseline = _metrics(target, baseline_probability)
            candidate = _metrics(target, candidate_probability)
            rows.append(
                {
                    "held_out_source": source,
                    "outer_fold": outer_fold,
                    "row_count": int(len(test)),
                    "positive_rows": int(target.sum()),
                    "baseline_auprc": baseline["auprc"],
                    "candidate_auprc": candidate["auprc"],
                    "auprc_delta": candidate["auprc"] - baseline["auprc"],
                    "baseline_auroc": baseline["auroc"],
                    "candidate_auroc": candidate["auroc"],
                    "baseline_brier": baseline["brier"],
                    "candidate_brier": candidate["brier"],
                }
            )
    return pd.DataFrame(rows)


def _promotion_decision(
    config: DeploymentOptimizationConfig,
    overall: pd.DataFrame,
    outer: pd.DataFrame,
    source: pd.DataFrame,
) -> dict[str, Any]:
    gate = config.payload["promotion_gate"]
    baseline = overall.set_index("model").loc["fusion_002_baseline"]
    candidate = overall.set_index("model").loc["deployment_candidate"]
    eligible_source = source[
        source["positive_rows"].ge(int(gate["source_minimum_positive_rows"]))
    ]
    checks = {
        "natural_auprc_delta": bool(
            candidate["auprc"] - baseline["auprc"]
            >= float(gate["minimum_natural_auprc_delta"])
        ),
        "natural_auroc_non_degraded": bool(
            candidate["auroc"] - baseline["auroc"]
            >= float(gate["minimum_natural_auroc_delta"])
        ),
        "natural_brier_within_tolerance": bool(
            candidate["brier"] - baseline["brier"]
            <= float(gate["maximum_natural_brier_increase"])
        ),
        "outer_fold_stability": bool(
            outer["auprc_delta"].ge(-float(gate["outer_fold_auprc_tolerance"])).sum()
            >= int(gate["minimum_non_degraded_outer_folds"])
        ),
        "leave_one_source_stability": bool(
            eligible_source.empty
            or eligible_source["auprc_delta"]
            .ge(-float(gate["maximum_leave_one_source_auprc_drop"]))
            .all()
        ),
    }
    return {
        "version": "mood-social-deployment-promotion-gate-v1",
        "promotion_passed": all(checks.values()),
        "checks": checks,
        "thresholds": dict(gate),
        "observed": {
            "natural_auprc_delta": float(candidate["auprc"] - baseline["auprc"]),
            "natural_auroc_delta": float(candidate["auroc"] - baseline["auroc"]),
            "natural_brier_increase": float(candidate["brier"] - baseline["brier"]),
            "non_degraded_outer_folds": int(outer["auprc_delta"].ge(0.0).sum()),
            "eligible_leave_one_source_units": int(len(eligible_source)),
            "worst_eligible_leave_one_source_auprc_delta": (
                float(eligible_source["auprc_delta"].min())
                if not eligible_source.empty
                else None
            ),
        },
        "art001_default": "deployment_candidate"
        if all(checks.values())
        else "fusion_002_baseline",
        "offline_reference_only": "MH-20260802-011 source_calibrated",
    }


def _metrics(target: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    target = np.asarray(target, dtype=int)
    probability = np.clip(np.asarray(probability, dtype=float), EPSILON, 1.0 - EPSILON)
    return {
        "row_count": int(len(target)),
        "positive_rows": int(target.sum()),
        "auprc": float(average_precision_score(target, probability)),
        "auroc": float(roc_auc_score(target, probability))
        if np.unique(target).size == 2
        else None,
        "brier": float(np.mean((probability - target) ** 2)),
    }


def _inner_fold_for(
    assignments: pd.DataFrame, participant: str, outer_fold: int
) -> int:
    try:
        value = assignments.loc[str(participant), "inner_validation_fold_by_outer_fold"]
    except KeyError as exc:
        raise DeploymentOptimizationError(
            f"participant missing from DATA-007: {participant}"
        ) from exc
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
    if isinstance(value, Mapping):
        value = value.get(str(outer_fold), value.get(outer_fold))
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise DeploymentOptimizationError("DATA-007 inner fold is invalid") from exc
    if result not in range(INNER_FOLDS):
        raise DeploymentOptimizationError("DATA-007 inner fold is outside 0..4")
    return result


def _publish(
    config: DeploymentOptimizationConfig,
    frame: pd.DataFrame,
    search: pd.DataFrame,
    oof: pd.DataFrame,
    overall: pd.DataFrame,
    outer: pd.DataFrame,
    source: pd.DataFrame,
    promotion: Mapping[str, Any],
    model: DeploymentSafeFusionModel,
    protection: Mapping[str, Any],
    command: Sequence[str],
) -> None:
    report = config.report_directory
    report.mkdir(parents=True, exist_ok=True)
    config.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.model_path)
    _write_parquet(search, report / "candidate_search.parquet")
    _write_parquet(oof, report / "oof_predictions.parquet")
    _write_parquet(overall, report / "overall_metrics.parquet")
    _write_parquet(outer, report / "outer_fold_stability.parquet")
    _write_parquet(source, report / "leave_one_source_stability.parquet")
    _write_json(report / "promotion.json", promotion)
    _write_json(report / "upstream_protection.json", protection)
    _write_json(report / "config.json", config.payload)
    _write_json(
        report / "run.json",
        {
            "task_id": TASK_ID,
            "run_id": RUN_ID,
            "training_version": TRAINING_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "command": list(command),
            "strict_oof": True,
        },
    )
    model_card = {
        "version": REPORT_VERSION,
        "task_id": TASK_ID,
        "run_id": RUN_ID,
        "model_version": MODEL_VERSION,
        "candidate_id": model.candidate_id,
        "strict_oof": True,
        "deployment_safe": True,
        "uses_public_dataset_id": False,
        "promotion_passed": bool(promotion["promotion_passed"]),
        "art001_default": promotion["art001_default"],
        "medical_diagnosis": False,
        "production_threshold_selected": False,
    }
    _write_json(report / "model_card.json", model_card)
    (report / "model_card.md").write_text(
        "# OPT-FUSION-002 Model Card\n\n"
        f"- Run: `{RUN_ID}`\n"
        f"- Candidate: `{model.candidate_id}`\n"
        f"- Promotion passed: `{str(bool(promotion['promotion_passed'])).lower()}`\n"
        "- Public `dataset_id` is evaluation metadata only and is not consumed by inference.\n"
        "- No production threshold or medical diagnosis is selected.\n",
        encoding="utf-8",
        newline="\n",
    )
    manifest = {
        "version": MANIFEST_VERSION,
        "task_id": TASK_ID,
        "run_id": RUN_ID,
        "training_version": TRAINING_VERSION,
        "model_version": MODEL_VERSION,
        "strict_oof": True,
        "selected_candidate_id": model.candidate_id,
        "promotion_passed": bool(promotion["promotion_passed"]),
        "art001_default": promotion["art001_default"],
        "uses_public_dataset_id": False,
        "input_features": list(INPUT_FEATURES),
        "row_count": int(len(frame)),
        "positive_rows": int(frame["binary_target"].sum()),
        "model_sha256": _sha256_file(config.model_path),
        "oof_sha256": _sha256_file(report / "oof_predictions.parquet"),
        "report_core_sha256": _report_core_sha256(report),
        "upstream_protection": dict(protection),
        "production_boundary": {
            "consume_public_dataset_id": False,
            "select_production_threshold": False,
            "change_active_experts": False,
            "change_personal_trend": False,
            "include_model006_predictions": False,
            "change_http_behavior": False,
            "change_attention_levels": False,
            "overwrite_existing_models": False,
        },
    }
    _write_json(config.manifest_path, manifest)
    artifact_rows = []
    for path in sorted(report.iterdir(), key=lambda item: item.name.encode("utf-8")):
        if path.is_file() and path.name != "artifacts.json":
            artifact_rows.append(
                {
                    "path": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
    _write_json(
        report / "artifacts.json",
        {
            "version": "mood-social-deployment-optimization-artifacts-v1",
            "artifacts": artifact_rows,
        },
    )


def _validate_output_paths(
    config: DeploymentOptimizationConfig,
    *,
    overwrite: bool,
) -> None:
    targets = (config.model_path, config.manifest_path, config.report_directory)
    if not overwrite and any(path.exists() for path in targets):
        raise DeploymentOptimizationError(
            "OPT-FUSION-002 output already exists; use --overwrite"
        )


def _resolve(root: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (root / path).resolve()


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0)))


def _logit(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(value, dtype=float), EPSILON, 1.0 - EPSILON)
    return np.log(clipped / (1.0 - clipped))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _report_core_sha256(report: Path) -> str:
    digest = hashlib.sha256()
    excluded = {"run.json", "artifacts.json"}
    for path in sorted(
        (
            item
            for item in report.iterdir()
            if item.is_file() and item.name not in excluded
        ),
        key=lambda item: item.name.encode("utf-8"),
    ):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    frame.to_parquet(path, index=False, compression="zstd")
