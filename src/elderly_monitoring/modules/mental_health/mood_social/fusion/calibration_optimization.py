"""Strict nested-OOF calibration candidates for OPT-CALIB-001.

The current deployment candidate trained coverage calibrators on the primary
probability and applied them to the globally calibrated probability.  This
module keeps that legacy chain as an audit-only diagnostic, while producing
corrected candidates whose fit and inference inputs are identical.

This task is report-only.  It never mutates the V3.3.3 online package or HTTP
contract.  All candidate selection is participant-level nested OOF.
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
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
import yaml


TASK_ID = "OPT-CALIB-001"
RUN_ID = "MH-20260804-014"
TRAINING_VERSION = "mood-social-calibration-optimization-v3.3.4-v1"
MODEL_VERSION = "mood-social-calibration-candidate-v3.3.4-v1"
MANIFEST_VERSION = "mood-social-calibration-manifest-v1"
REPORT_VERSION = "mood-social-calibration-report-v1"
RANDOM_SEED = 20260728
OUTER_FOLDS = 5
INNER_FOLDS = 5
EPSILON = 1.0e-7
CALIBRATION_METHODS = ("platt", "beta", "temperature", "isotonic")
SELECTABLE_VARIANTS = (
    "global_platt",
    "global_beta",
    "global_temperature",
    "global_isotonic",
    "corrected_coverage_platt",
    "dual_raw_logit_platt",
)
AUDIT_VARIANTS = ("legacy_mismatched_coverage_platt",)
VARIANT_IDS = SELECTABLE_VARIANTS + AUDIT_VARIANTS
MIN_GROUP_ROWS = 50
MIN_GROUP_POSITIVE_ROWS = 5
MIN_GROUP_NEGATIVE_ROWS = 5


class CalibrationOptimizationError(RuntimeError):
    """Raised when the frozen OPT-CALIB-001 protocol is violated."""


@dataclass(frozen=True)
class CalibrationConfig:
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
class PlattCalibrator:
    coefficient: float
    intercept: float

    def predict(self, probability: np.ndarray) -> np.ndarray:
        z = self.coefficient * _logit(probability) + self.intercept
        return _sigmoid(z)


@dataclass(frozen=True)
class BetaCalibrator:
    coefficients: tuple[float, float]
    intercept: float

    def predict(self, probability: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(probability, dtype=float), EPSILON, 1.0 - EPSILON)
        matrix = np.column_stack([np.log(p), np.log1p(-p)])
        return _sigmoid(matrix @ np.asarray(self.coefficients) + self.intercept)


@dataclass(frozen=True)
class TemperatureCalibrator:
    temperature: float

    def predict(self, probability: np.ndarray) -> np.ndarray:
        return _sigmoid(_logit(probability) / float(self.temperature))


@dataclass(frozen=True)
class IsotonicCalibrator:
    x_thresholds: tuple[float, ...]
    y_thresholds: tuple[float, ...]

    def predict(self, probability: np.ndarray) -> np.ndarray:
        thresholds = np.asarray(self.x_thresholds, dtype=float)
        values = np.asarray(self.y_thresholds, dtype=float)
        if thresholds.size == 0 or values.size != thresholds.size:
            raise CalibrationOptimizationError(
                "isotonic calibrator thresholds are invalid"
            )
        return np.asarray(
            np.interp(np.asarray(probability, dtype=float), thresholds, values),
            dtype=float,
        )


Calibrator = (
    PlattCalibrator | BetaCalibrator | TemperatureCalibrator | IsotonicCalibrator
)


@dataclass(frozen=True)
class CalibrationCandidateModel:
    task_id: str
    run_id: str
    model_version: str
    variant_id: str
    global_method: str
    coverage_method: str | None
    global_calibrator: Calibrator | None
    coverage_calibrators: Mapping[str, Calibrator]
    fit_input_stage: str

    def validate(self) -> None:
        if self.task_id != TASK_ID or self.run_id != RUN_ID:
            raise CalibrationOptimizationError("calibration candidate identity changed")
        if self.model_version != MODEL_VERSION or self.variant_id not in VARIANT_IDS:
            raise CalibrationOptimizationError("calibration candidate version changed")
        if self.global_method not in {"none", *CALIBRATION_METHODS}:
            raise CalibrationOptimizationError("unknown global calibration method")
        if self.coverage_method not in {None, "platt"}:
            raise CalibrationOptimizationError("unknown coverage calibration method")
        if self.variant_id in AUDIT_VARIANTS and self.fit_input_stage != "primary":
            raise CalibrationOptimizationError(
                "legacy audit candidate input stage changed"
            )
        if any(
            "dataset" in key.lower() or "source" in key.lower()
            for key in self.coverage_calibrators
        ):
            raise CalibrationOptimizationError(
                "public source calibration entered candidate"
            )

    def predict_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        self.validate()
        required = {"fusion_logit", "evidence_combination"}
        missing = required - set(frame.columns)
        if missing:
            raise CalibrationOptimizationError(
                f"calibration input columns missing: {sorted(missing)}"
            )
        primary = _sigmoid(frame["fusion_logit"].to_numpy(dtype=float))
        if self.global_calibrator is None:
            global_probability = primary.copy()
        else:
            global_probability = self.global_calibrator.predict(primary)
        final_probability = global_probability.copy()
        if self.coverage_calibrators:
            modes = frame["evidence_combination"].astype(str).to_numpy()
            for mode, calibrator in self.coverage_calibrators.items():
                selected = modes == mode
                if not selected.any():
                    continue
                source = (
                    primary[selected]
                    if self.fit_input_stage == "primary"
                    else global_probability[selected]
                )
                final_probability[selected] = calibrator.predict(source)
        final_probability = np.clip(final_probability, EPSILON, 1.0 - EPSILON)
        return pd.DataFrame(
            {
                "primary_probability": primary,
                "global_probability": np.clip(
                    global_probability, EPSILON, 1.0 - EPSILON
                ),
                "final_probability": final_probability,
                "ranking_score": frame["fusion_logit"].to_numpy(dtype=float),
            },
            index=frame.index,
        )


def load_calibration_config(
    path: str | Path, *, repository_root: str | Path | None = None
) -> CalibrationConfig:
    config_path = Path(path).resolve()
    root = Path(repository_root or config_path.parents[2]).resolve()
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise CalibrationOptimizationError(
            "OPT-CALIB-001 config is unreadable"
        ) from exc
    if not isinstance(payload, Mapping):
        raise CalibrationOptimizationError("OPT-CALIB-001 config must be a mapping")
    if payload.get("task_id") != TASK_ID or payload.get("run_id") != RUN_ID:
        raise CalibrationOptimizationError("OPT-CALIB-001 task or run ID changed")
    if payload.get("training_version") != TRAINING_VERSION:
        raise CalibrationOptimizationError("OPT-CALIB-001 training version changed")
    if tuple(payload.get("calibration", {}).get("methods", ())) != CALIBRATION_METHODS:
        raise CalibrationOptimizationError("calibration method grid changed")
    if (
        tuple(payload.get("candidates", {}).get("selectable", ()))
        != SELECTABLE_VARIANTS
    ):
        raise CalibrationOptimizationError("calibration candidate grid changed")
    if tuple(payload.get("candidates", {}).get("audit_only", ())) != AUDIT_VARIANTS:
        raise CalibrationOptimizationError("calibration audit grid changed")
    if (
        payload.get("split", {}).get("outer_folds") != OUTER_FOLDS
        or payload.get("split", {}).get("inner_folds") != INNER_FOLDS
    ):
        raise CalibrationOptimizationError("nested fold contract changed")
    boundary = payload.get("production_boundary", {})
    required_false = (
        "overwrite_online_package",
        "change_http_behavior",
        "include_model006_predictions",
        "consume_public_dataset_id",
        "select_threshold",
    )
    if any(boundary.get(key) is not False for key in required_false):
        raise CalibrationOptimizationError("OPT-CALIB-001 production boundary changed")
    return CalibrationConfig(root, payload, config_path)


def load_calibration_inputs(
    config: CalibrationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    input_payload = config.payload["input"]
    baseline_path = _resolve(config.repository_root, input_payload["baseline_oof_path"])
    current_online_path = _resolve(
        config.repository_root, input_payload["current_online_oof_path"]
    )
    split_path = _resolve(config.repository_root, input_payload["split_path"])
    table_path = _resolve(
        config.repository_root, input_payload["fusion_oof_table_path"]
    )
    for path, expected in (
        (baseline_path, input_payload["baseline_oof_sha256"]),
        (current_online_path, input_payload["current_online_oof_sha256"]),
        (split_path, input_payload["split_sha256"]),
        (table_path, input_payload["fusion_oof_table_sha256"]),
    ):
        if not path.is_file() or _sha256_file(path) != str(expected):
            raise CalibrationOptimizationError(f"protected input drifted: {path.name}")
    baseline = pd.read_parquet(baseline_path).reset_index(drop=True)
    current_online = pd.read_parquet(current_online_path).reset_index(drop=True)
    table = pd.read_parquet(table_path).reset_index(drop=True)
    required = {
        "global_participant_id",
        "outer_fold",
        "binary_target",
        "fusion_logit",
        "fusion_probability",
        "fusion_available",
        "evidence_combination",
    }
    if required - set(baseline.columns):
        raise CalibrationOptimizationError(
            f"baseline columns missing: {sorted(required - set(baseline.columns))}"
        )
    if len(baseline) != 22191 or len(table) != 22191:
        raise CalibrationOptimizationError("strict OOF row count changed")
    available = (
        baseline.loc[baseline["fusion_available"].astype(bool)]
        .copy()
        .reset_index(drop=True)
    )
    if len(available) != 22188:
        raise CalibrationOptimizationError("available calibration row count changed")
    if available[["dataset_id", "canonical_row_index"]].duplicated().any():
        raise CalibrationOptimizationError("available calibration row identity changed")
    participant_folds = available.groupby("global_participant_id")[
        "outer_fold"
    ].nunique()
    if int(participant_folds.max()) != 1:
        raise CalibrationOptimizationError("participant crossed outer folds")
    current_required = {
        "dataset_id",
        "canonical_row_index",
        "global_participant_id",
        "binary_target",
        "outer_fold",
        "deployment_candidate_probability",
        "selected_candidate_id",
    }
    if current_required - set(current_online.columns) or len(current_online) != 22188:
        raise CalibrationOptimizationError("current online baseline OOF changed")
    active = current_online[
        [
            "dataset_id",
            "canonical_row_index",
            "global_participant_id",
            "binary_target",
            "outer_fold",
            "deployment_candidate_probability",
            "selected_candidate_id",
        ]
    ].rename(
        columns={
            "global_participant_id": "current_global_participant_id",
            "binary_target": "current_binary_target",
            "outer_fold": "current_outer_fold",
            "deployment_candidate_probability": "current_online_probability",
        }
    )
    available = available.merge(
        active,
        on=["dataset_id", "canonical_row_index"],
        how="left",
        validate="one_to_one",
    )
    if available["current_online_probability"].isna().any():
        raise CalibrationOptimizationError(
            "current online OOF does not align to calibration rows"
        )
    if not (
        available["global_participant_id"].astype(str)
        == available["current_global_participant_id"].astype(str)
    ).all():
        raise CalibrationOptimizationError(
            "current online participant alignment changed"
        )
    if (
        not (
            available["binary_target"].to_numpy(dtype=int)
            == available["current_binary_target"].to_numpy(dtype=int)
        ).all()
        or not (
            available["outer_fold"].to_numpy(dtype=int)
            == available["current_outer_fold"].to_numpy(dtype=int)
        ).all()
    ):
        raise CalibrationOptimizationError(
            "current online target or fold alignment changed"
        )
    if set(available["selected_candidate_id"].astype(str)) != {
        "coverage_calibrated",
        "multiseed_interactions",
    }:
        raise CalibrationOptimizationError("current online candidate identity changed")
    available = available.drop(
        columns=[
            "current_global_participant_id",
            "current_binary_target",
            "current_outer_fold",
        ]
    )
    if not np.isfinite(available["fusion_logit"].to_numpy(dtype=float)).all():
        raise CalibrationOptimizationError("fusion logits contain non-finite values")
    split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    assignments = pd.DataFrame(split_payload.get("participant_assignments", []))
    if assignments.empty or assignments["global_participant_id"].duplicated().any():
        raise CalibrationOptimizationError("DATA-007 assignments are invalid")
    assignments["global_participant_id"] = assignments["global_participant_id"].astype(
        str
    )
    assignments = assignments.set_index("global_participant_id")
    if set(available["global_participant_id"].astype(str)) - set(assignments.index):
        raise CalibrationOptimizationError(
            "calibration participants missing from DATA-007"
        )
    protection = {
        "baseline_oof_sha256": str(input_payload["baseline_oof_sha256"]),
        "current_online_oof_sha256": str(input_payload["current_online_oof_sha256"]),
        "fusion_oof_table_sha256": str(input_payload["fusion_oof_table_sha256"]),
        "split_sha256": str(input_payload["split_sha256"]),
        "strict_oof": True,
        "public_dataset_id_as_model_input": False,
        "model006_enters_fusion": False,
        "legacy_training_stage": "primary",
        "corrected_training_stage": "global",
    }
    return available, assignments, protection


def optimize_calibration(
    config: CalibrationConfig,
    *,
    overwrite: bool = False,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    frame, assignments, protection = load_calibration_inputs(config)
    _validate_outputs(config, overwrite=overwrite)
    search, oof, selected_id = _nested_oof(frame, assignments)
    overall = _overall_metrics(frame, oof)
    outer = _outer_metrics(frame, oof)
    selected_model = _fit_variant(frame, selected_id)
    thresholds = _threshold_curve(
        frame["binary_target"].to_numpy(dtype=int),
        oof[f"probability__{selected_id}"].to_numpy(dtype=float),
        selected_id,
    )
    calibration_curve = _calibration_curve(
        frame["binary_target"].to_numpy(dtype=int),
        oof[f"probability__{selected_id}"].to_numpy(dtype=float),
        selected_id,
    )
    baseline_metrics = overall.loc[overall["model"].eq("current_online_baseline")].iloc[
        0
    ]
    selected_metrics = overall.loc[overall["model"].eq(selected_id)].iloc[0]
    summary = {
        "task_id": TASK_ID,
        "run_id": RUN_ID,
        "status": "pass",
        "selected_variant": selected_id,
        "selection_rule": "inner_oof_auprc_desc_then_brier_asc_then_variant_id",
        "promotion_status": "report_only_candidate_not_production",
        "legacy_mismatch_audited": True,
        "strict_oof_rows": int(len(frame)),
        "baseline_model": (
            "MH-20260802-012 deployment_candidate "
            "(aggregate package coverage_calibrated)"
        ),
        "baseline_metrics": {
            key: float(baseline_metrics[key])
            for key in ("auprc", "auroc", "brier", "ece")
        },
        "selected_metrics": {
            key: float(selected_metrics[key])
            for key in ("auprc", "auroc", "brier", "ece")
        },
        "selected_minus_baseline": {
            "auprc": float(selected_metrics["auprc"] - baseline_metrics["auprc"]),
            "auroc": float(selected_metrics["auroc"] - baseline_metrics["auroc"]),
            "brier": float(selected_metrics["brier"] - baseline_metrics["brier"]),
            "ece": float(selected_metrics["ece"] - baseline_metrics["ece"]),
        },
    }
    _publish(
        config,
        frame,
        search,
        oof,
        overall,
        outer,
        thresholds,
        calibration_curve,
        summary,
        selected_model,
        protection,
        command or (),
    )
    return {
        "status": "pass",
        "task_id": TASK_ID,
        "run_id": RUN_ID,
        "selected_variant": selected_id,
        "model_path": str(config.model_path),
        "report_directory": str(config.report_directory),
    }


def _nested_oof(
    frame: pd.DataFrame, assignments: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    search_rows: list[dict[str, Any]] = []
    predictions = {
        variant: np.full(len(frame), np.nan, dtype=float) for variant in VARIANT_IDS
    }
    selected = np.full(len(frame), "", dtype=object)
    for outer_fold in range(OUTER_FOLDS):
        outer_test = frame["outer_fold"].to_numpy(dtype=int) == outer_fold
        outer_train = frame.loc[~outer_test].reset_index(drop=True)
        inner_scores: list[dict[str, Any]] = []
        for variant in SELECTABLE_VARIANTS:
            inner_probability = np.full(len(outer_train), np.nan, dtype=float)
            for inner_fold in range(INNER_FOLDS):
                validation = np.array(
                    [
                        _inner_fold(assignments, value, outer_fold) == inner_fold
                        for value in outer_train["global_participant_id"].astype(str)
                    ],
                    dtype=bool,
                )
                if not validation.any() or validation.all():
                    raise CalibrationOptimizationError(
                        "nested inner fold coverage changed"
                    )
                model = _fit_variant(
                    outer_train.loc[~validation].reset_index(drop=True), variant
                )
                inner_probability[validation] = (
                    model.predict_frame(outer_train.loc[validation])
                    .iloc[:, 2]
                    .to_numpy(dtype=float)
                )
            if not np.isfinite(inner_probability).all():
                raise CalibrationOptimizationError(
                    "inner calibration OOF is incomplete"
                )
            metrics = _metrics(
                outer_train["binary_target"].to_numpy(dtype=int), inner_probability
            )
            inner_scores.append(
                {"outer_fold": outer_fold, "variant_id": variant, **metrics}
            )
        chosen = sorted(
            inner_scores,
            key=lambda row: (-row["auprc"], row["brier"], row["variant_id"]),
        )[0]["variant_id"]
        selected[outer_test] = chosen
        for variant in VARIANT_IDS:
            model = _fit_variant(outer_train, variant)
            predicted = (
                model.predict_frame(frame.loc[outer_test])
                .iloc[:, 2]
                .to_numpy(dtype=float)
            )
            predictions[variant][outer_test] = predicted
        search_rows.extend(inner_scores)
        for row in inner_scores:
            row["selected"] = row["variant_id"] == chosen
    search = pd.DataFrame(search_rows)
    production = search.groupby("variant_id", as_index=False).agg(
        selection_count=("selected", "sum"),
        mean_inner_auprc=("auprc", "mean"),
        mean_inner_brier=("brier", "mean"),
    )
    production = production[production["variant_id"].isin(SELECTABLE_VARIANTS)].copy()
    chosen_global = sorted(
        production.to_dict("records"),
        key=lambda row: (
            -int(row["selection_count"]),
            -float(row["mean_inner_auprc"]),
            float(row["mean_inner_brier"]),
            row["variant_id"],
        ),
    )[0]["variant_id"]
    production["outer_fold"] = -1
    production["selected"] = production["variant_id"].eq(chosen_global)
    search = pd.concat(
        [
            search,
            production[
                [
                    "outer_fold",
                    "variant_id",
                    "mean_inner_auprc",
                    "mean_inner_brier",
                    "selection_count",
                    "selected",
                ]
            ].rename(
                columns={"mean_inner_auprc": "auprc", "mean_inner_brier": "brier"}
            ),
        ],
        ignore_index=True,
        sort=False,
    )
    oof = frame[
        [
            "dataset_id",
            "global_participant_id",
            "canonical_row_index",
            "binary_target",
            "outer_fold",
            "fusion_logit",
            "fusion_probability",
            "evidence_combination",
        ]
    ].copy()
    for variant, values in predictions.items():
        oof[f"probability__{variant}"] = values
    oof["selected_variant"] = selected
    return search, oof, str(chosen_global)


def _fit_variant(frame: pd.DataFrame, variant: str) -> CalibrationCandidateModel:
    if variant not in VARIANT_IDS:
        raise CalibrationOptimizationError(f"unknown calibration variant: {variant}")
    primary = _sigmoid(frame["fusion_logit"].to_numpy(dtype=float))
    target = frame["binary_target"].to_numpy(dtype=int)
    if variant.startswith("global_"):
        method = variant.removeprefix("global_")
        return CalibrationCandidateModel(
            TASK_ID,
            RUN_ID,
            MODEL_VERSION,
            variant,
            method,
            None,
            _fit_calibrator(primary, target, method),
            {},
            "global",
        )
    if variant == "dual_raw_logit_platt":
        return CalibrationCandidateModel(
            TASK_ID,
            RUN_ID,
            MODEL_VERSION,
            variant,
            "platt",
            None,
            _fit_calibrator(primary, target, "platt"),
            {},
            "primary",
        )
    if variant in {"corrected_coverage_platt", "legacy_mismatched_coverage_platt"}:
        global_calibrator = _fit_calibrator(primary, target, "platt")
        global_probability = global_calibrator.predict(primary)
        source = (
            global_probability if variant == "corrected_coverage_platt" else primary
        )
        calibrators: dict[str, Calibrator] = {}
        modes = frame["evidence_combination"].astype(str)
        for mode in sorted(modes.unique(), key=lambda value: value.encode("utf-8")):
            selected = modes.eq(mode).to_numpy()
            positives = int(target[selected].sum())
            negatives = int(selected.sum() - positives)
            if (
                selected.sum() >= MIN_GROUP_ROWS
                and positives >= MIN_GROUP_POSITIVE_ROWS
                and negatives >= MIN_GROUP_NEGATIVE_ROWS
            ):
                calibrators[mode] = _fit_calibrator(
                    source[selected], target[selected], "platt"
                )
        return CalibrationCandidateModel(
            TASK_ID,
            RUN_ID,
            MODEL_VERSION,
            variant,
            "platt",
            "platt",
            global_calibrator,
            calibrators,
            "global" if variant == "corrected_coverage_platt" else "primary",
        )
    raise CalibrationOptimizationError(f"unsupported calibration variant: {variant}")


def _fit_calibrator(
    probability: np.ndarray, target: np.ndarray, method: str
) -> Calibrator:
    p = np.clip(np.asarray(probability, dtype=float), EPSILON, 1.0 - EPSILON)
    y = np.asarray(target, dtype=int)
    if set(np.unique(y)) != {0, 1}:
        raise CalibrationOptimizationError("calibration fit requires both classes")
    if method == "platt":
        model = LogisticRegression(
            C=1.0, solver="lbfgs", max_iter=2000, random_state=RANDOM_SEED
        )
        model.fit(_logit(p)[:, None], y)
        return PlattCalibrator(float(model.coef_[0, 0]), float(model.intercept_[0]))
    if method == "beta":
        model = LogisticRegression(
            C=1.0, solver="lbfgs", max_iter=2000, random_state=RANDOM_SEED
        )
        matrix = np.column_stack([np.log(p), np.log1p(-p)])
        model.fit(matrix, y)
        return BetaCalibrator(
            (float(model.coef_[0, 0]), float(model.coef_[0, 1])),
            float(model.intercept_[0]),
        )
    if method == "temperature":
        logits = _logit(p)
        grid = np.exp(np.linspace(np.log(0.05), np.log(10.0), 401))
        losses = [
            log_loss(y, _sigmoid(logits / value), labels=[0, 1]) for value in grid
        ]
        return TemperatureCalibrator(float(grid[int(np.argmin(losses))]))
    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip").fit(p, y)
        return IsotonicCalibrator(
            tuple(model.X_thresholds_.tolist()), tuple(model.y_thresholds_.tolist())
        )
    raise CalibrationOptimizationError(f"unknown calibration method: {method}")


def _metrics(target: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    p = np.asarray(probability, dtype=float)
    y = np.asarray(target, dtype=int)
    return {
        "auprc": float(average_precision_score(y, p)),
        "auroc": float(roc_auc_score(y, p)),
        "brier": float(np.mean((p - y) ** 2)),
        "ece": float(_ece(y, p)),
    }


def _overall_metrics(frame: pd.DataFrame, oof: pd.DataFrame) -> pd.DataFrame:
    y = frame["binary_target"].to_numpy(dtype=int)
    rows = [
        {
            "model": "current_online_baseline",
            **_metrics(y, frame["current_online_probability"].to_numpy(dtype=float)),
        },
        {
            "model": "fusion_002_raw",
            **_metrics(y, frame["fusion_probability"].to_numpy(dtype=float)),
        },
    ]
    for variant in VARIANT_IDS:
        rows.append(
            {
                "model": variant,
                **_metrics(y, oof[f"probability__{variant}"].to_numpy(dtype=float)),
            }
        )
    return pd.DataFrame(rows)


def _outer_metrics(frame: pd.DataFrame, oof: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold in range(OUTER_FOLDS):
        selected = frame["outer_fold"].to_numpy(dtype=int) == fold
        y = frame.loc[selected, "binary_target"].to_numpy(dtype=int)
        for variant in ("current_online_baseline", "fusion_002_raw", *VARIANT_IDS):
            if variant == "current_online_baseline":
                p = frame.loc[selected, "current_online_probability"].to_numpy(
                    dtype=float
                )
            elif variant == "fusion_002_raw":
                p = frame.loc[selected, "fusion_probability"].to_numpy(dtype=float)
            else:
                p = oof.loc[selected, f"probability__{variant}"].to_numpy(dtype=float)
            rows.append({"outer_fold": fold, "model": variant, **_metrics(y, p)})
    return pd.DataFrame(rows)


def _threshold_curve(
    target: np.ndarray, probability: np.ndarray, model: str
) -> pd.DataFrame:
    thresholds = np.unique(np.concatenate(([0.0, 1.0], np.clip(probability, 0.0, 1.0))))
    rows = []
    for threshold in thresholds:
        predicted = probability >= threshold
        tp = int(np.sum(predicted & (target == 1)))
        fp = int(np.sum(predicted & (target == 0)))
        fn = int(np.sum(~predicted & (target == 1)))
        tn = int(np.sum(~predicted & (target == 0)))
        rows.append(
            {
                "model": model,
                "threshold": float(threshold),
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "true_negative": tn,
                "sensitivity": float(tp / (tp + fn)) if tp + fn else 0.0,
                "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
                "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _calibration_curve(
    target: np.ndarray, probability: np.ndarray, model: str, bins: int = 10
) -> pd.DataFrame:
    rows = []
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        selected = (probability >= lower) & (
            (probability < upper) if index < bins - 1 else (probability <= upper)
        )
        rows.append(
            {
                "model": model,
                "bin": index,
                "lower": lower,
                "upper": upper,
                "row_count": int(selected.sum()),
                "mean_probability": float(probability[selected].mean())
                if selected.any()
                else None,
                "positive_rate": float(target[selected].mean())
                if selected.any()
                else None,
            }
        )
    return pd.DataFrame(rows)


def _ece(target: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    curve = _calibration_curve(target, probability, "ece", bins)
    total = len(target)
    return float(
        sum(
            (row.row_count / total) * abs(row.mean_probability - row.positive_rate)
            for row in curve.itertuples()
            if row.row_count
            and row.mean_probability is not None
            and row.positive_rate is not None
        )
    )


def _inner_fold(assignments: pd.DataFrame, participant: str, outer_fold: int) -> int:
    value = assignments.loc[str(participant), "inner_validation_fold_by_outer_fold"]
    if not isinstance(value, Mapping):
        raise CalibrationOptimizationError("inner fold mapping is invalid")
    result = value.get(str(outer_fold))
    if result is None:
        raise CalibrationOptimizationError("outer-test participant entered inner fold")
    return int(result)


def _publish(
    config: CalibrationConfig,
    frame: pd.DataFrame,
    search: pd.DataFrame,
    oof: pd.DataFrame,
    overall: pd.DataFrame,
    outer: pd.DataFrame,
    thresholds: pd.DataFrame,
    calibration_curve: pd.DataFrame,
    summary: Mapping[str, Any],
    model: CalibrationCandidateModel,
    protection: Mapping[str, Any],
    command: Sequence[str],
) -> None:
    report = config.report_directory
    report.mkdir(parents=True, exist_ok=True)
    config.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.model_path)
    model.validate()
    manifest = {
        "version": MANIFEST_VERSION,
        "task_id": TASK_ID,
        "run_id": RUN_ID,
        "model_version": MODEL_VERSION,
        "selected_variant": model.variant_id,
        "strict_oof": True,
        "candidate_scope": "offline_candidate",
        "production": False,
        "public_dataset_id_as_model_input": False,
        "model006_online": False,
        "fit_input_stage": model.fit_input_stage,
        "model_sha256": _sha256_file(config.model_path),
    }
    _write_json(config.manifest_path, manifest)
    _write_parquet(report / "candidate_search.parquet", search)
    _write_parquet(report / "oof_predictions.parquet", oof)
    _write_parquet(report / "overall_metrics.parquet", overall)
    _write_parquet(report / "outer_fold_stability.parquet", outer)
    _write_parquet(report / "threshold_curve.parquet", thresholds)
    _write_parquet(report / "calibration_curve.parquet", calibration_curve)
    _write_json(report / "summary.json", dict(summary))
    _write_json(report / "upstream_protection.json", dict(protection))
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
    model_card = {
        "task_id": TASK_ID,
        "run_id": RUN_ID,
        "model_version": MODEL_VERSION,
        "variant": model.variant_id,
        "strict_oof_rows": int(len(frame)),
        "production_status": "offline_candidate_not_production",
        "limitations": [
            "public strict OOF only",
            "no real household deployment labels",
            "legacy p_primary/p_global mismatch retained only as audit",
        ],
    }
    _write_json(report / "model_card.json", model_card)
    (report / "model_card.md").write_text(
        "# OPT-CALIB-001 candidate\n\n- Scope: offline candidate only\n- Training and inference calibration stages are bound by the candidate manifest.\n- The legacy mismatched coverage chain is audit-only.\n",
        encoding="utf-8",
    )
    _write_json(
        config.report_directory / "artifacts.json",
        {
            "version": "mood-social-opt-calib-artifacts-v1",
            "artifacts": [
                {
                    "path": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
                for path in sorted(
                    report.iterdir(), key=lambda item: item.name.encode("utf-8")
                )
                if path.is_file() and path.name != "artifacts.json"
            ],
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
    checksum_lines = [
        f"{_sha256_file(path)}  {path.name}"
        for path in sorted(
            (config.model_path.parent).iterdir(),
            key=lambda item: item.name.encode("utf-8"),
        )
        if path.is_file() and path.name != "SHA256SUMS"
    ]
    (config.model_path.parent / "SHA256SUMS").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )


def _validate_outputs(config: CalibrationConfig, *, overwrite: bool) -> None:
    paths = [config.model_path, config.manifest_path, config.report_directory]
    if not overwrite and any(path.exists() for path in paths):
        raise CalibrationOptimizationError(
            "OPT-CALIB-001 output already exists; use --overwrite"
        )


def _sigmoid(value: np.ndarray | float) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(array, -40.0, 40.0)))


def _logit(value: np.ndarray | float) -> np.ndarray:
    p = np.clip(np.asarray(value, dtype=float), EPSILON, 1.0 - EPSILON)
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
    raise TypeError(f"unsupported JSON value: {type(value)!r}")


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    frame.to_parquet(path, index=False)


__all__ = [
    "AUDIT_VARIANTS",
    "CALIBRATION_METHODS",
    "CalibrationCandidateModel",
    "CalibrationConfig",
    "CalibrationOptimizationError",
    "MODEL_VERSION",
    "RUN_ID",
    "SELECTABLE_VARIANTS",
    "TASK_ID",
    "load_calibration_config",
    "load_calibration_inputs",
    "optimize_calibration",
]
