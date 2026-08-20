"""V3.3.3 missing-aware logistic stacking for MoodSocial.

The module consumes only the strict OOF evidence table published by
FUSION-001.  Masks gate evidence; they are never learned as independent risk
features.  The native logistic sigmoid is the model probability, while
calibration curves and ECE are reported as diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import pyarrow
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    log_loss,
    roc_auc_score,
)
import sklearn
import yaml

from .oof_table import (
    BASELINE_PROTECTION_SHA256,
    EVALUATION_CORE_SHA256,
    FEATURE_SCHEMA_SHA256,
    FusionOOFConfig,
    build_fusion_oof_table,
    load_fusion_oof_config,
    load_fusion_oof_inputs,
)


TASK_ID = "FUSION-002"
RUN_ID = "MH-20260802-010"
TRAINING_VERSION = "mood-social-masked-stacking-v3.3.3-v1"
BUNDLE_VERSION = "mood-social-masked-stacking-bundle-v1"
MANIFEST_VERSION = "mood-social-masked-stacking-manifest-v1"
OOF_VERSION = "mood-social-masked-stacking-oof-v1"
METRICS_VERSION = "mood-social-masked-stacking-metrics-v1"
SEARCH_VERSION = "mood-social-masked-stacking-search-v1"
CALIBRATION_VERSION = "mood-social-masked-stacking-calibration-v1"
ARTIFACT_VERSION = "mood-social-masked-stacking-artifacts-v1"
RANDOM_SEED = 20260728
OUTER_FOLD_COUNT = 5
INNER_FOLD_COUNT = 5
PROBABILITY_CLIP_EPSILON = 1.0e-6
C_VALUES = (0.01, 0.1, 1.0, 10.0)
INPUT_FEATURES = (
    "expert_activity_evidence",
    "expert_sleep_evidence",
    "expert_joint_evidence",
    "expert_physiology_evidence",
    "expert_social_context_evidence",
    "trend_activity_evidence",
    "trend_sleep_evidence",
    "trend_social_evidence",
)
EXPERT_BRANCHES = ("activity", "sleep", "joint", "physiology", "social_context")
TREND_BRANCHES = ("activity", "sleep", "social")
MASKED_EXPERT_COLUMNS = {
    "activity": ("expert_activity_expert_mask", "expert_activity_confidence"),
    "sleep": ("expert_sleep_expert_mask", "expert_sleep_confidence"),
    "joint": ("expert_joint_expert_mask", "expert_joint_confidence"),
    "physiology": (
        "expert_physiology_expert_mask",
        "expert_physiology_confidence",
    ),
    "social_context": (
        "expert_social_context_expert_mask",
        "expert_social_context_confidence",
    ),
}
MASKED_TREND_COLUMNS = {
    "activity": (
        "trend_activity_personal_change_mask",
        "trend_activity_reliability",
        "trend_activity_personal_change_evidence",
    ),
    "sleep": (
        "trend_sleep_personal_change_mask",
        "trend_sleep_reliability",
        "trend_sleep_personal_change_evidence",
    ),
    "social": (
        "trend_social_personal_change_mask",
        "trend_social_reliability",
        "trend_social_personal_change_evidence",
    ),
}
DEVICE_COMBINATION_ORDER = ("none", "C", "S", "C+S", "T", "C+T", "S+T", "C+S+T")


class MaskedStackingError(RuntimeError):
    """Raised when FUSION-002 input, training, or publication is invalid."""


@dataclass(frozen=True)
class MaskedStackingConfig:
    repository_root: Path
    payload: Mapping[str, Any]
    config_path: Path
    fusion_oof_config_path: Path

    @property
    def config_sha256(self) -> str:
        return _sha256_file(self.config_path)

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
class MaskedStackingModel:
    bundle_version: str
    training_version: str
    task_id: str
    run_id: str
    input_features: tuple[str, ...]
    probability_representation: str
    selected_c: float
    classifier: LogisticRegression

    def validate(self) -> None:
        if self.bundle_version != BUNDLE_VERSION:
            raise MaskedStackingError("MaskedStacking bundle version changed")
        if self.task_id != TASK_ID or self.run_id != RUN_ID:
            raise MaskedStackingError("MaskedStacking task identity changed")
        if self.input_features != INPUT_FEATURES:
            raise MaskedStackingError("MaskedStacking input order changed")
        if self.probability_representation != "current_calibrated":
            raise MaskedStackingError(
                "only current calibrated probabilities are allowed"
            )
        if self.selected_c not in C_VALUES:
            raise MaskedStackingError("MaskedStacking selected C is invalid")
        if not isinstance(self.classifier, LogisticRegression):
            raise MaskedStackingError("MaskedStacking classifier type changed")
        if self.classifier.penalty != "l2" or self.classifier.solver != "lbfgs":
            raise MaskedStackingError("MaskedStacking logistic protocol changed")
        if self.classifier.coef_.shape != (1, len(INPUT_FEATURES)):
            raise MaskedStackingError("MaskedStacking coefficient shape changed")

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        return build_fusion_features(frame)

    def predict_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        self.validate()
        matrix = self.transform(frame)
        evidence = effective_evidence(frame)
        available = evidence > 0.0
        probability = np.full(len(frame), np.nan, dtype="float64")
        logit = np.full(len(frame), np.nan, dtype="float64")
        if np.any(available):
            logit[available] = self.classifier.decision_function(matrix[available])
            probability[available] = _sigmoid(logit[available])
        result = pd.DataFrame(
            {
                "fusion_probability": probability,
                "fusion_logit": logit,
                "effective_evidence": evidence,
                "available": available,
            },
            index=frame.index,
        )
        result["fusion_confidence"] = confidence_from_coefficients(
            frame, self.classifier.coef_[0]
        )
        result.loc[~available, "fusion_confidence"] = 0.0
        return result


@dataclass(frozen=True)
class MaskedStackingInputs:
    config: MaskedStackingConfig
    fusion_oof_config: FusionOOFConfig
    table: pd.DataFrame
    assignments: pd.DataFrame
    fusion_oof_manifest: Mapping[str, Any]
    fusion_oof_core_sha256: str


def load_masked_stacking_config(
    path: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> MaskedStackingConfig:
    config_path = Path(path).resolve()
    root = Path(repository_root or config_path.parents[2]).resolve()
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise MaskedStackingError("FUSION-002 config is unreadable") from exc
    if not isinstance(payload, Mapping):
        raise MaskedStackingError("FUSION-002 config must be a mapping")
    if payload.get("task_id") != TASK_ID or payload.get("run_id") != RUN_ID:
        raise MaskedStackingError("FUSION-002 task or run ID changed")
    if payload.get("random_seed") != RANDOM_SEED:
        raise MaskedStackingError("FUSION-002 random seed changed")
    if tuple(payload.get("input_features", ())) != INPUT_FEATURES:
        raise MaskedStackingError("FUSION-002 input feature order changed")
    fusion_config_path = _resolve(root, payload["input"]["fusion_oof_config_path"])
    return MaskedStackingConfig(
        repository_root=root,
        payload=payload,
        config_path=config_path,
        fusion_oof_config_path=fusion_config_path,
    )


def load_masked_stacking_inputs(config: MaskedStackingConfig) -> MaskedStackingInputs:
    fusion_config = load_fusion_oof_config(
        config.fusion_oof_config_path,
        repository_root=config.repository_root,
    )
    if fusion_config.payload.get("task_id") != "FUSION-001":
        raise MaskedStackingError("FUSION-001 input task binding changed")
    fusion_inputs = load_fusion_oof_inputs(fusion_config)
    table_path = _resolve(
        config.repository_root,
        config.payload["input"]["fusion_oof_table_path"],
    )
    table = pd.read_parquet(table_path).reset_index(drop=True)
    expected, _, _, _ = build_fusion_oof_table(fusion_inputs)
    if not table.equals(expected):
        raise MaskedStackingError("FUSION-001 table is not a deterministic rebuild")
    expected_sha = str(config.payload["input"]["fusion_oof_table_sha256"])
    if _sha256_file(table_path) != expected_sha:
        raise MaskedStackingError("FUSION-001 table hash drifted")
    fusion_report = fusion_config.report_directory
    manifest = _read_json(fusion_report / "fusion_table_manifest.json")
    if (
        manifest.get("strict_oof") is not True
        or manifest.get("model006_included") is not False
    ):
        raise MaskedStackingError("FUSION-001 production boundary changed")
    report_core = _report_core_sha256(fusion_report)
    if report_core != str(config.payload["input"]["fusion_oof_report_core_sha256"]):
        raise MaskedStackingError("FUSION-001 report core drifted")
    split_path = _resolve(
        config.repository_root,
        config.payload["split"]["path"],
    )
    split = _read_json(split_path)
    assignments = pd.DataFrame(split.get("participant_assignments", []))
    required = {
        "global_participant_id",
        "dataset_id",
        "outer_fold",
        "inner_validation_fold_by_outer_fold",
    }
    if assignments.empty or not required.issubset(assignments.columns):
        raise MaskedStackingError("DATA-007 nested assignments are incomplete")
    if assignments["global_participant_id"].duplicated().any():
        raise MaskedStackingError("DATA-007 participant assignments are duplicated")
    if set(assignments["global_participant_id"].astype(str)) != set(
        table["global_participant_id"].astype(str)
    ):
        raise MaskedStackingError("FUSION-001 participants do not match DATA-007")
    assignment_outer = assignments.set_index("global_participant_id")["outer_fold"]
    table_outer = table.groupby("global_participant_id", sort=False)[
        "outer_fold"
    ].first()
    table_outer.index = table_outer.index.astype(str)
    if not table_outer.sort_index().equals(assignment_outer.astype(int).sort_index()):
        raise MaskedStackingError("FUSION-001 outer folds do not match DATA-007")
    _validate_table_contract(table)
    return MaskedStackingInputs(
        config=config,
        fusion_oof_config=fusion_config,
        table=table,
        assignments=assignments,
        fusion_oof_manifest=manifest,
        fusion_oof_core_sha256=report_core,
    )


def build_fusion_features(frame: pd.DataFrame) -> np.ndarray:
    """Build the eight formula-level evidence inputs in frozen order."""

    _validate_feature_columns(frame)
    values: list[np.ndarray] = []
    for branch in EXPERT_BRANCHES:
        mask_name, confidence_name = MASKED_EXPERT_COLUMNS[branch]
        probability_name = f"expert_{branch}_current_probability"
        probability = pd.to_numeric(frame[probability_name], errors="coerce").to_numpy(
            dtype="float64"
        )
        mask = pd.to_numeric(frame[mask_name], errors="raise").to_numpy(dtype="float64")
        confidence = pd.to_numeric(frame[confidence_name], errors="raise").to_numpy(
            dtype="float64"
        )
        safe = np.clip(
            np.nan_to_num(probability, nan=0.5),
            PROBABILITY_CLIP_EPSILON,
            1 - PROBABILITY_CLIP_EPSILON,
        )
        values.append(mask * confidence * _logit(safe))
    for branch in TREND_BRANCHES:
        mask_name, reliability_name, evidence_name = MASKED_TREND_COLUMNS[branch]
        mask = pd.to_numeric(frame[mask_name], errors="raise").to_numpy(dtype="float64")
        reliability = pd.to_numeric(frame[reliability_name], errors="raise").to_numpy(
            dtype="float64"
        )
        evidence = pd.to_numeric(frame[evidence_name], errors="coerce").to_numpy(
            dtype="float64"
        )
        values.append(mask * reliability * np.nan_to_num(evidence, nan=0.0))
    matrix = np.column_stack(values).astype("float64", copy=False)
    if not np.isfinite(matrix).all():
        raise MaskedStackingError(
            "FUSION-002 feature matrix contains non-finite values"
        )
    return matrix


def effective_evidence(frame: pd.DataFrame) -> np.ndarray:
    _validate_feature_columns(frame)
    result = np.zeros(len(frame), dtype="float64")
    for mask_name, confidence_name in MASKED_EXPERT_COLUMNS.values():
        result += pd.to_numeric(frame[mask_name], errors="raise").to_numpy(
            dtype="float64"
        ) * pd.to_numeric(frame[confidence_name], errors="raise").to_numpy(
            dtype="float64"
        )
    for mask_name, reliability_name, _ in MASKED_TREND_COLUMNS.values():
        result += pd.to_numeric(frame[mask_name], errors="raise").to_numpy(
            dtype="float64"
        ) * pd.to_numeric(frame[reliability_name], errors="raise").to_numpy(
            dtype="float64"
        )
    if not np.isfinite(result).all() or np.any(result < 0):
        raise MaskedStackingError("effective evidence is invalid")
    return result


def evidence_combination(frame: pd.DataFrame) -> pd.Series:
    """Return the descriptive C/S/T availability combination for reporting.

    C represents camera-derived activity evidence, S sleep-device evidence, and
    T the S10/social PersonalTrend evidence.  SocialContextExpert is profile
    context, not a fourth device, and therefore does not create a combination
    token.  The result is a reporting dimension only; it is never a model input.
    """

    _validate_feature_columns(frame)
    camera = np.zeros(len(frame), dtype=bool)
    camera |= (
        pd.to_numeric(frame["expert_activity_expert_mask"], errors="raise").to_numpy(
            dtype="float64"
        )
        > 0
    )
    camera |= (
        pd.to_numeric(frame["expert_joint_expert_mask"], errors="raise").to_numpy(
            dtype="float64"
        )
        > 0
    )
    camera |= (
        pd.to_numeric(
            frame["trend_activity_personal_change_mask"], errors="raise"
        ).to_numpy(dtype="float64")
        > 0
    )

    sleep = np.zeros(len(frame), dtype=bool)
    sleep |= (
        pd.to_numeric(frame["expert_sleep_expert_mask"], errors="raise").to_numpy(
            dtype="float64"
        )
        > 0
    )
    sleep |= (
        pd.to_numeric(frame["expert_physiology_expert_mask"], errors="raise").to_numpy(
            dtype="float64"
        )
        > 0
    )
    sleep |= (
        pd.to_numeric(
            frame["trend_sleep_personal_change_mask"], errors="raise"
        ).to_numpy(dtype="float64")
        > 0
    )

    s10 = (
        pd.to_numeric(
            frame["trend_social_personal_change_mask"], errors="raise"
        ).to_numpy(dtype="float64")
        > 0
    )
    values = np.full(len(frame), "none", dtype=object)
    for index in range(len(frame)):
        tokens = []
        if camera[index]:
            tokens.append("C")
        if sleep[index]:
            tokens.append("S")
        if s10[index]:
            tokens.append("T")
        values[index] = "+".join(tokens) if tokens else "none"
    result = pd.Series(values, index=frame.index, name="evidence_combination")
    if not result.isin(DEVICE_COMBINATION_ORDER).all():
        raise MaskedStackingError("FUSION-002 evidence combination is invalid")
    return result


def masked_stacking_sample_weights(frame: pd.DataFrame) -> np.ndarray:
    """Four-level equal weights: dataset, class, participant, repeated row."""

    required = {"dataset_id", "global_participant_id", "binary_target"}
    if not required.issubset(frame.columns) or frame.empty:
        raise MaskedStackingError("FUSION-002 sample-weight frame is incomplete")
    datasets = frame["dataset_id"].astype(str)
    participants = frame["global_participant_id"].astype(str)
    target = pd.to_numeric(frame["binary_target"], errors="raise").astype("int64")
    if not target.isin([0, 1]).all():
        raise MaskedStackingError("FUSION-002 target is not binary")
    weights = np.zeros(len(frame), dtype="float64")
    for dataset_id in sorted(set(datasets), key=lambda value: value.encode("utf-8")):
        dataset_mask = datasets.eq(dataset_id).to_numpy()
        classes = sorted(set(target[dataset_mask].tolist()))
        class_mass = 1.0 / len(classes)
        for class_value in classes:
            class_mask = dataset_mask & target.eq(class_value).to_numpy()
            units = participants[class_mask]
            unit_counts = units.value_counts(sort=False)
            unit_mass = class_mass / len(unit_counts)
            for row_index in np.flatnonzero(class_mask):
                weights[row_index] = unit_mass / int(
                    unit_counts[participants.iloc[row_index]]
                )
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise MaskedStackingError("FUSION-002 sample weights are invalid")
    weights *= len(weights) / weights.sum()
    return weights


def confidence_from_coefficients(
    frame: pd.DataFrame, coefficients: Sequence[float]
) -> np.ndarray:
    coeff = np.asarray(coefficients, dtype="float64")
    if coeff.shape != (len(INPUT_FEATURES),):
        raise MaskedStackingError("FUSION-002 coefficient vector shape changed")
    numerator = np.zeros(len(frame), dtype="float64")
    denominator = np.zeros(len(frame), dtype="float64")
    for index, branch in enumerate(EXPERT_BRANCHES):
        mask_name, confidence_name = MASKED_EXPERT_COLUMNS[branch]
        mask = pd.to_numeric(frame[mask_name], errors="raise").to_numpy(dtype="float64")
        confidence = pd.to_numeric(frame[confidence_name], errors="raise").to_numpy(
            dtype="float64"
        )
        weight = abs(float(coeff[index]))
        numerator += mask * weight * confidence
        denominator += mask * weight
    for offset, branch in enumerate(TREND_BRANCHES, start=len(EXPERT_BRANCHES)):
        mask_name, reliability_name, _ = MASKED_TREND_COLUMNS[branch]
        mask = pd.to_numeric(frame[mask_name], errors="raise").to_numpy(dtype="float64")
        reliability = pd.to_numeric(frame[reliability_name], errors="raise").to_numpy(
            dtype="float64"
        )
        weight = abs(float(coeff[offset]))
        numerator += mask * weight * reliability
        denominator += mask * weight
    result = np.divide(
        numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0
    )
    return np.clip(result, 0.0, 1.0)


def train_masked_stacking(
    repository_root: str | Path,
    config_path: str | Path,
    *,
    overwrite: bool = False,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    root = Path(repository_root).resolve()
    config = load_masked_stacking_config(config_path, repository_root=root)
    _validate_output_paths(config, overwrite=overwrite)
    inputs = load_masked_stacking_inputs(config)
    table = inputs.table
    feature_matrix = build_fusion_features(table)
    evidence = effective_evidence(table)
    available = evidence > 0.0
    if not np.any(available) or table.loc[available, "binary_target"].nunique() < 2:
        raise MaskedStackingError("FUSION-002 has no binary available training data")
    outer_results: list[pd.DataFrame] = []
    outer_searches: list[dict[str, Any]] = []
    weight_audits: list[dict[str, Any]] = []
    for outer_fold in range(OUTER_FOLD_COUNT):
        result, search, audit = _train_outer_fold(
            table,
            inputs.assignments,
            feature_matrix,
            evidence,
            outer_fold,
        )
        outer_results.append(result)
        outer_searches.append(search)
        weight_audits.append(audit)
    oof = (
        pd.concat(outer_results, ignore_index=True)
        .sort_values(["outer_fold", "canonical_row_index"], kind="stable")
        .reset_index(drop=True)
    )
    production_selection = _select_production_candidate(outer_searches)
    production_frame = table.loc[available].copy()
    production_matrix = feature_matrix[available]
    production_weights = masked_stacking_sample_weights(production_frame)
    production_classifier = _fit_classifier(
        production_matrix,
        production_frame["binary_target"].to_numpy(dtype="int64"),
        production_weights,
        c_value=float(production_selection["selected_c"]),
    )
    model = MaskedStackingModel(
        bundle_version=BUNDLE_VERSION,
        training_version=TRAINING_VERSION,
        task_id=TASK_ID,
        run_id=RUN_ID,
        input_features=INPUT_FEATURES,
        probability_representation="current_calibrated",
        selected_c=float(production_selection["selected_c"]),
        classifier=production_classifier,
    )
    model.validate()
    metrics_summary, metrics_payload = _metrics_payload(oof)
    calibration_table, calibration_payload = _calibration_payload(oof)
    combination_table = _grouped_metrics(oof, "evidence_combination")
    source_table = _grouped_metrics(oof, "dataset_id")
    coverage_table = _grouped_metrics(oof, "mask_pattern")
    coefficient_payload = _coefficient_payload(model)
    warnings = _warnings_payload(table, oof)
    result = _publish_training_run(
        config,
        inputs,
        model,
        oof,
        metrics_summary,
        metrics_payload,
        calibration_table,
        calibration_payload,
        source_table,
        coverage_table,
        combination_table,
        coefficient_payload,
        warnings,
        outer_searches,
        production_selection,
        weight_audits,
        started=started,
        ended=datetime.now(timezone.utc),
        overwrite=overwrite,
        command=command or (),
    )
    return result


def _train_outer_fold(
    table: pd.DataFrame,
    assignments: pd.DataFrame,
    feature_matrix: np.ndarray,
    evidence: np.ndarray,
    outer_fold: int,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    outer_train_mask = table["outer_fold"].astype(int).ne(outer_fold).to_numpy()
    outer_test_mask = ~outer_train_mask
    outer_train_available = outer_train_mask & (evidence > 0.0)
    outer_test_available = outer_test_mask & (evidence > 0.0)
    if not np.any(outer_train_available) or not np.any(outer_test_available):
        raise MaskedStackingError(f"outer fold {outer_fold} lacks available rows")
    normalized_assignments = assignments.copy()
    normalized_assignments["global_participant_id"] = normalized_assignments[
        "global_participant_id"
    ].astype(str)
    assignment_by_id = normalized_assignments.set_index("global_participant_id")
    participant_ids = table["global_participant_id"].astype(str)
    inner_values = np.full(len(table), -1, dtype="int64")
    inner_values[outer_train_mask] = (
        participant_ids.loc[outer_train_mask]
        .map(lambda value: _inner_fold_value(assignment_by_id.loc[value], outer_fold))
        .to_numpy(dtype="int64")
    )
    inner_oof_by_c: dict[float, list[pd.DataFrame]] = {value: [] for value in C_VALUES}
    inner_manifests: list[dict[str, Any]] = []
    for inner_fold in range(INNER_FOLD_COUNT):
        validation_mask = outer_train_mask & (inner_values == inner_fold)
        train_mask = outer_train_mask & ~validation_mask
        train_available = train_mask & (evidence > 0.0)
        validation_available = validation_mask & (evidence > 0.0)
        _require_binary(
            table.loc[train_available, "binary_target"],
            f"outer {outer_fold} inner {inner_fold} train",
        )
        _require_binary(
            table.loc[validation_available, "binary_target"],
            f"outer {outer_fold} inner {inner_fold} validation",
        )
        train_frame = table.loc[train_available]
        validation_frame = table.loc[validation_available]
        train_weight = masked_stacking_sample_weights(train_frame)
        for c_value in C_VALUES:
            classifier = _fit_classifier(
                feature_matrix[train_available],
                train_frame["binary_target"].to_numpy(dtype="int64"),
                train_weight,
                c_value=c_value,
            )
            prediction = _sigmoid(
                classifier.decision_function(feature_matrix[validation_available])
            )
            part = validation_frame[
                ["global_participant_id", "canonical_row_index", "binary_target"]
            ].copy()
            part["probability"] = prediction
            part["selected_c"] = c_value
            inner_oof_by_c[c_value].append(part)
        inner_manifests.append(
            {
                "outer_fold": outer_fold,
                "inner_fold": inner_fold,
                "train_participant_count": int(participant_ids[train_mask].nunique()),
                "validation_participant_count": int(
                    participant_ids[validation_mask].nunique()
                ),
                "train_available_rows": int(train_available.sum()),
                "validation_available_rows": int(validation_available.sum()),
                "train_weight_audit": _weight_audit(train_frame, train_weight),
            }
        )
    candidate_rows: list[dict[str, Any]] = []
    for c_value in C_VALUES:
        inner_oof = pd.concat(inner_oof_by_c[c_value], ignore_index=True)
        # Restore the table rows in exactly the validation-fold order used above.
        weight_frame = table.loc[outer_train_mask & (evidence > 0.0)].copy()
        weight_frame = (
            weight_frame.set_index(["global_participant_id", "canonical_row_index"])
            .loc[
                pd.MultiIndex.from_frame(
                    inner_oof[["global_participant_id", "canonical_row_index"]]
                )
            ]
            .reset_index()
        )
        weights = masked_stacking_sample_weights(weight_frame)
        scores = _binary_metric_values(
            inner_oof["binary_target"].to_numpy(dtype="int64"),
            inner_oof["probability"].to_numpy(dtype="float64"),
            weights,
        )
        candidate_rows.append(
            {
                "candidate_id": f"l2-c={c_value:g}",
                "c": c_value,
                "pooled_inner_oof_auprc": scores["auprc"],
                "pooled_inner_oof_brier": scores["brier"],
                "pooled_inner_available_rows": int(len(inner_oof)),
            }
        )
    selected = sorted(
        candidate_rows,
        key=lambda row: (
            -float(row["pooled_inner_oof_auprc"]),
            float(row["pooled_inner_oof_brier"]),
            str(row["candidate_id"]),
        ),
    )[0]
    outer_train_frame = table.loc[outer_train_available]
    outer_weight = masked_stacking_sample_weights(outer_train_frame)
    classifier = _fit_classifier(
        feature_matrix[outer_train_available],
        outer_train_frame["binary_target"].to_numpy(dtype="int64"),
        outer_weight,
        c_value=float(selected["c"]),
    )
    outer_table = table.loc[outer_test_mask].copy()
    outer_table["fusion_input_feature_matrix"] = list(feature_matrix[outer_test_mask])
    outer_table["fusion_effective_evidence"] = evidence[outer_test_mask]
    outer_table["fusion_available"] = outer_test_available[outer_test_mask]
    outer_table["fusion_logit"] = np.nan
    outer_table["fusion_probability"] = np.nan
    outer_table["fusion_confidence"] = 0.0
    if np.any(outer_test_available[outer_test_mask]):
        local = outer_test_available[outer_test_mask]
        logit = classifier.decision_function(feature_matrix[outer_test_mask][local])
        outer_table.loc[local, "fusion_logit"] = logit
        outer_table.loc[local, "fusion_probability"] = _sigmoid(logit)
    outer_table["fusion_confidence"] = confidence_from_coefficients(
        outer_table, classifier.coef_[0]
    )
    outer_table.loc[~outer_table["fusion_available"], "fusion_confidence"] = 0.0
    outer_table["selected_c"] = float(selected["c"])
    outer_table["probability_representation"] = "current_calibrated"
    outer_table["evidence_combination"] = evidence_combination(outer_table)
    outer_table = outer_table.drop(columns=["fusion_input_feature_matrix"])
    search = {
        "outer_fold": outer_fold,
        "outer_train_participant_count": int(
            participant_ids[outer_train_mask].nunique()
        ),
        "outer_test_participant_count": int(participant_ids[outer_test_mask].nunique()),
        "outer_train_available_rows": int(outer_train_available.sum()),
        "outer_test_available_rows": int(outer_test_available.sum()),
        "candidates": candidate_rows,
        "selected_candidate_id": selected["candidate_id"],
        "selected_c": selected["c"],
        "outer_test_role": "evaluation_only",
        "inner_folds": inner_manifests,
    }
    audit = {
        "outer_fold": outer_fold,
        "outer_train": _weight_audit(outer_train_frame, outer_weight),
    }
    return outer_table, search, audit


def _select_production_candidate(
    searches: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for c_value in C_VALUES:
        candidates = [
            row
            for search in searches
            for row in search["candidates"]
            if float(row["c"]) == c_value
        ]
        rows.append(
            {
                "candidate_id": f"l2-c={c_value:g}",
                "c": c_value,
                "mean_inner_oof_auprc": float(
                    np.mean([row["pooled_inner_oof_auprc"] for row in candidates])
                ),
                "mean_inner_oof_brier": float(
                    np.mean([row["pooled_inner_oof_brier"] for row in candidates])
                ),
                "outer_fold_selection_count": int(
                    sum(float(search["selected_c"]) == c_value for search in searches)
                ),
            }
        )
    selected = sorted(
        rows,
        key=lambda row: (
            -row["mean_inner_oof_auprc"],
            row["mean_inner_oof_brier"],
            row["candidate_id"],
        ),
    )[0]
    return {
        "selection_rule": "mean_inner_oof_auprc_desc_then_brier_asc_then_candidate_id",
        "candidates": rows,
        "selected_c": selected["c"],
        "selected_candidate_id": selected["candidate_id"],
    }


def _fit_classifier(
    matrix: np.ndarray, target: np.ndarray, weight: np.ndarray, *, c_value: float
) -> LogisticRegression:
    if c_value not in C_VALUES:
        raise MaskedStackingError("invalid FUSION-002 C")
    classifier = LogisticRegression(
        C=float(c_value),
        penalty="l2",
        solver="lbfgs",
        max_iter=5000,
        tol=1.0e-10,
        random_state=RANDOM_SEED,
    )
    classifier.fit(matrix, target, sample_weight=weight)
    return classifier


def _metrics_payload(oof: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    available = oof["fusion_available"].astype(bool)
    frame = oof.loc[available].copy()
    target = frame["binary_target"].to_numpy(dtype="int64")
    probability = frame["fusion_probability"].to_numpy(dtype="float64")
    weights = masked_stacking_sample_weights(frame)
    natural = _binary_metric_values(target, probability, np.ones(len(frame)))
    frozen = _binary_metric_values(target, probability, weights)
    summary = pd.DataFrame(
        [
            {
                "scope": "overall",
                "group": "all",
                "weight_variant": "natural",
                **natural,
            },
            {
                "scope": "overall",
                "group": "all",
                "weight_variant": "frozen_four_level",
                **frozen,
            },
        ]
    )
    payload = {
        "version": METRICS_VERSION,
        "task_id": TASK_ID,
        "run_id": RUN_ID,
        "available_row_count": int(len(frame)),
        "unavailable_row_count": int((~available).sum()),
        "positive_available_rows": int(frame["binary_target"].sum()),
        "threshold_status": "0.5_descriptive_only_not_production_selected",
        "natural": natural,
        "frozen_four_level": frozen,
    }
    return summary, payload


def _grouped_metrics(oof: pd.DataFrame, group: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, group_frame in oof.groupby(group, dropna=False, sort=True):
        available = group_frame["fusion_available"].astype(bool)
        available_frame = group_frame.loc[available]
        if available_frame.empty or available_frame["binary_target"].nunique() < 1:
            continue
        target = available_frame["binary_target"].to_numpy(dtype="int64")
        probability = available_frame["fusion_probability"].to_numpy(dtype="float64")
        natural = _binary_metric_values(
            target, probability, np.ones(len(available_frame))
        )
        weights = masked_stacking_sample_weights(available_frame)
        frozen = _binary_metric_values(target, probability, weights)
        for variant, metrics in (("natural", natural), ("frozen_four_level", frozen)):
            rows.append({"group": str(key), "weight_variant": variant, **metrics})
    return pd.DataFrame(rows)


def _binary_metric_values(
    target: np.ndarray, probability: np.ndarray, weight: np.ndarray
) -> dict[str, Any]:
    if len(target) == 0:
        return {
            "row_count": 0,
            "positive_rows": 0,
            "prevalence": None,
            "auprc": None,
            "auroc": None,
            "macro_f1": None,
            "sensitivity": None,
            "specificity": None,
            "brier": None,
            "log_loss": None,
            "ece": None,
        }
    score = np.clip(np.asarray(probability, dtype="float64"), 0.0, 1.0)
    weight = np.asarray(weight, dtype="float64")
    predicted = (score >= 0.5).astype("int64")
    tp = float(np.sum(weight[(target == 1) & (predicted == 1)]))
    fn = float(np.sum(weight[(target == 1) & (predicted == 0)]))
    tn = float(np.sum(weight[(target == 0) & (predicted == 0)]))
    fp = float(np.sum(weight[(target == 0) & (predicted == 1)]))
    try:
        auprc = float(average_precision_score(target, score, sample_weight=weight))
    except ValueError:
        auprc = None
    try:
        auroc = float(roc_auc_score(target, score, sample_weight=weight))
    except ValueError:
        auroc = None
    try:
        macro_f1 = float(
            f1_score(
                target,
                predicted,
                average="macro",
                sample_weight=weight,
                zero_division=0,
            )
        )
    except ValueError:
        macro_f1 = None
    try:
        loss = float(log_loss(target, score, sample_weight=weight, labels=[0, 1]))
    except ValueError:
        loss = None
    return {
        "row_count": int(len(target)),
        "positive_rows": int(target.sum()),
        "prevalence": float(np.average(target, weights=weight)),
        "auprc": auprc,
        "auroc": auroc,
        "macro_f1": macro_f1,
        "sensitivity": float(tp / (tp + fn)) if tp + fn else None,
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "brier": float(np.average((score - target) ** 2, weights=weight)),
        "log_loss": loss,
        "ece": _ece(target, score, weight),
    }


def _calibration_payload(oof: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = oof.loc[oof["fusion_available"].astype(bool)].copy()
    if frame.empty:
        columns = [
            "weight_variant",
            "bin",
            "lower",
            "upper",
            "row_count",
            "weight_sum",
            "mean_probability",
            "observed_rate",
            "absolute_gap",
        ]
        return pd.DataFrame(columns=columns), {
            "version": CALIBRATION_VERSION,
            "method": "native_logistic_sigmoid_no_posthoc_calibrator",
            "probability_representation": "current_calibrated",
            "ece": {"natural": None, "frozen_four_level": None},
            "posthoc_calibrator_selected": False,
            "threshold_selected": False,
        }
    rows: list[dict[str, Any]] = []
    for variant, weight in (
        ("natural", np.ones(len(frame))),
        ("frozen_four_level", masked_stacking_sample_weights(frame)),
    ):
        probability = frame["fusion_probability"].to_numpy(dtype="float64")
        target = frame["binary_target"].to_numpy(dtype="int64")
        for bin_index in range(10):
            lower = bin_index / 10.0
            upper = (bin_index + 1) / 10.0
            selected = (probability >= lower) & (
                (probability < upper) if bin_index < 9 else (probability <= upper)
            )
            if not np.any(selected):
                continue
            rows.append(
                {
                    "weight_variant": variant,
                    "bin": bin_index,
                    "lower": lower,
                    "upper": upper,
                    "row_count": int(selected.sum()),
                    "weight_sum": float(weight[selected].sum()),
                    "mean_probability": float(
                        np.average(probability[selected], weights=weight[selected])
                    ),
                    "observed_rate": float(
                        np.average(target[selected], weights=weight[selected])
                    ),
                    "absolute_gap": float(
                        abs(
                            np.average(probability[selected], weights=weight[selected])
                            - np.average(target[selected], weights=weight[selected])
                        )
                    ),
                }
            )
    table = pd.DataFrame(rows)
    ece_values: dict[str, float | None] = {}
    for variant in ("natural", "frozen_four_level"):
        selected_rows = table.loc[table["weight_variant"].eq(variant)]
        denominator = float(selected_rows["weight_sum"].sum())
        ece_values[variant] = (
            float(
                selected_rows["absolute_gap"].mul(selected_rows["weight_sum"]).sum()
                / denominator
            )
            if denominator > 0
            else None
        )
    payload = {
        "version": CALIBRATION_VERSION,
        "method": "native_logistic_sigmoid_no_posthoc_calibrator",
        "probability_representation": "current_calibrated",
        "ece": ece_values,
        "posthoc_calibrator_selected": False,
        "threshold_selected": False,
    }
    return table, payload


def _coefficient_payload(model: MaskedStackingModel) -> dict[str, Any]:
    coefficient = model.classifier.coef_[0].astype("float64")
    return {
        "input_features": list(INPUT_FEATURES),
        "intercept": float(model.classifier.intercept_[0]),
        "coefficients": {
            name: float(value)
            for name, value in zip(INPUT_FEATURES, coefficient, strict=True)
        },
        "absolute_coefficient_weights": {
            name: float(abs(value))
            for name, value in zip(INPUT_FEATURES, coefficient, strict=True)
        },
        "selected_c": model.selected_c,
        "probability_representation": model.probability_representation,
        "mask_values_are_risk_features": False,
    }


def _warnings_payload(table: pd.DataFrame, oof: pd.DataFrame) -> dict[str, Any]:
    warnings = [
        {
            "code": "NATIVE_LOGISTIC_NO_POSTHOC_CALIBRATOR",
            "message": "The logistic sigmoid is the published FUSION-002 probability; Platt/Isotonic are not silently adopted.",
        },
        {
            "code": "NO_PRODUCTION_THRESHOLD_SELECTED",
            "message": "Threshold 0.5 is descriptive only; no production work threshold was selected.",
        },
        {
            "code": "PSYCHE_D_NOMINAL_MONTH_TIMESCALE_PROXY",
            "message": "Activity and sleep PersonalTrend inputs remain a nominal-month to natural-day proxy transfer.",
        },
        {
            "code": "NO_DIRECT_S10_PHQ9_VALIDATION",
            "message": "The social PersonalTrend branch remains an engineering proxy without direct longitudinal S10/PHQ-9 validation.",
        },
    ]
    if int((~oof["fusion_available"].astype(bool)).sum()):
        warnings.append(
            {
                "code": "UNAVAILABLE_ROWS_HARD_GATED",
                "row_count": int((~oof["fusion_available"].astype(bool)).sum()),
                "message": "Rows with effective evidence at or below zero have null probability and do not run the intercept.",
            }
        )
    return {"version": "mood-social-masked-stacking-warnings-v1", "warnings": warnings}


def _publish_training_run(
    config: MaskedStackingConfig,
    inputs: MaskedStackingInputs,
    model: MaskedStackingModel,
    oof: pd.DataFrame,
    metrics_summary: pd.DataFrame,
    metrics_payload: Mapping[str, Any],
    calibration_table: pd.DataFrame,
    calibration_payload: Mapping[str, Any],
    source_table: pd.DataFrame,
    coverage_table: pd.DataFrame,
    combination_table: pd.DataFrame,
    coefficient_payload: Mapping[str, Any],
    warnings: Mapping[str, Any],
    outer_searches: Sequence[Mapping[str, Any]],
    production_selection: Mapping[str, Any],
    weight_audits: Sequence[Mapping[str, Any]],
    *,
    started: datetime,
    ended: datetime,
    overwrite: bool,
    command: Sequence[str],
) -> dict[str, Any]:
    root = config.repository_root
    stage_root = Path(tempfile.mkdtemp(prefix="fusion-002-stage-", dir=root))
    try:
        stage_model = stage_root / config.model_path.name
        stage_manifest = stage_root / config.manifest_path.name
        stage_report = stage_root / "report"
        stage_report.mkdir(parents=True)
        joblib.dump(model, stage_model, compress=0, protocol=4)
        loaded = joblib.load(stage_model)
        if not isinstance(loaded, MaskedStackingModel):
            raise MaskedStackingError("staged FUSION-002 model cannot be reloaded")
        loaded.validate()
        model_sha = _sha256_file(stage_model)
        published_oof = _publishable_oof(oof)
        _write_parquet(published_oof, stage_report / "oof_predictions.parquet")
        _write_parquet(metrics_summary, stage_report / "metrics_summary.parquet")
        _write_parquet(calibration_table, stage_report / "calibration_curve.parquet")
        _write_parquet(source_table, stage_report / "source_metrics.parquet")
        _write_parquet(coverage_table, stage_report / "coverage_metrics.parquet")
        _write_parquet(combination_table, stage_report / "combination_metrics.parquet")
        _write_json(stage_report / "metrics.json", metrics_payload)
        _write_json(stage_report / "calibration_results.json", calibration_payload)
        _write_json(stage_report / "coefficient_summary.json", coefficient_payload)
        _write_json(stage_report / "warnings.json", warnings)
        _write_json(
            stage_report / "search_results.json",
            {
                "version": SEARCH_VERSION,
                "task_id": TASK_ID,
                "run_id": RUN_ID,
                "outer_test_fold_role": "evaluation_only",
                "selection": production_selection["selection_rule"],
                "outer_searches": list(outer_searches),
                "production_selection": production_selection,
            },
        )
        _write_json(
            stage_report / "weights.json",
            {
                "definition": "dataset_equal_then_class_equal_then_participant_class_equal_then_repeat_window_equal",
                "outer_folds": list(weight_audits),
                "production": _weight_audit(
                    inputs.table.loc[
                        inputs.table["binary_target"].isin([0, 1])
                        & (effective_evidence(inputs.table) > 0.0)
                    ],
                    masked_stacking_sample_weights(
                        inputs.table.loc[
                            inputs.table["binary_target"].isin([0, 1])
                            & (effective_evidence(inputs.table) > 0.0)
                        ]
                    ),
                ),
            },
        )
        _write_json(
            stage_report / "upstream_protection.json",
            _upstream_protection_payload(config, inputs),
        )
        _write_json(
            stage_report / "training_manifest.json",
            _training_manifest(
                config,
                inputs,
                model,
                model_sha,
                published_oof,
                metrics_summary,
                calibration_table,
            ),
        )
        (stage_report / "model_card.md").write_text(
            _model_card(metrics_payload, calibration_payload, production_selection),
            encoding="utf-8",
            newline="\n",
        )
        report_core = _deterministic_report_core_sha256(stage_report)
        external_manifest = {
            **_training_manifest(
                config,
                inputs,
                model,
                model_sha,
                published_oof,
                metrics_summary,
                calibration_table,
            ),
            "report_core_sha256": report_core,
            "oof_sha256": _sha256_file(stage_report / "oof_predictions.parquet"),
            "metrics_sha256": _sha256_file(stage_report / "metrics.json"),
            "search_sha256": _sha256_file(stage_report / "search_results.json"),
            "calibration_sha256": _sha256_file(
                stage_report / "calibration_results.json"
            ),
        }
        _write_json(stage_manifest, external_manifest)
        manifest_sha = _sha256_file(stage_manifest)
        artifact_rows = _file_hash_rows(
            stage_report, exclude={"run.json", "artifacts.json"}
        )
        _write_json(
            stage_report / "artifacts.json",
            {
                "version": ARTIFACT_VERSION,
                "artifact_count": len(artifact_rows),
                "artifacts": artifact_rows,
                "report_core_sha256": report_core,
            },
        )
        _write_json(
            stage_report / "run.json",
            {
                "version": "mood-social-masked-stacking-run-v1",
                "task_id": TASK_ID,
                "run_id": RUN_ID,
                "started_at": started.isoformat(),
                "ended_at": ended.isoformat(),
                "duration_seconds": (ended - started).total_seconds(),
                "command": list(command),
                "python": sys.version,
                "platform": platform.platform(),
                "libraries": {
                    "numpy": np.__version__,
                    "pandas": pd.__version__,
                    "pyarrow": pyarrow.__version__,
                    "sklearn": sklearn.__version__,
                    "joblib": joblib.__version__,
                },
            },
        )
        _atomic_publish_file(stage_model, config.model_path, overwrite=overwrite)
        _atomic_publish_file(stage_manifest, config.manifest_path, overwrite=overwrite)
        _atomic_publish_directory(
            stage_report, config.report_directory, overwrite=overwrite
        )
        return {
            "task_id": TASK_ID,
            "run_id": RUN_ID,
            "model_path": config.model_path.as_posix(),
            "model_sha256": model_sha,
            "manifest_path": config.manifest_path.as_posix(),
            "manifest_sha256": manifest_sha,
            "report_directory": config.report_directory.as_posix(),
            "report_core_sha256": report_core,
            "oof_row_count": len(published_oof),
            "available_oof_row_count": int(published_oof["fusion_available"].sum()),
        }
    finally:
        shutil.rmtree(stage_root, ignore_errors=True)


def _training_manifest(
    config: MaskedStackingConfig,
    inputs: MaskedStackingInputs,
    model: MaskedStackingModel,
    model_sha: str,
    oof: pd.DataFrame,
    metrics_summary: pd.DataFrame,
    calibration_table: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "version": MANIFEST_VERSION,
        "task_id": TASK_ID,
        "run_id": RUN_ID,
        "training_version": TRAINING_VERSION,
        "bundle_version": BUNDLE_VERSION,
        "model_sha256": model_sha,
        "config_sha256": config.config_sha256,
        "input_fusion_oof_sha256": _sha256_file(
            _resolve(
                config.repository_root, config.payload["input"]["fusion_oof_table_path"]
            )
        ),
        "input_fusion_oof_report_core_sha256": inputs.fusion_oof_core_sha256,
        "split_id": str(config.payload["split"]["split_id"]),
        "split_sha256": str(config.payload["split"]["sha256"]),
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "input_features": list(INPUT_FEATURES),
        "probability_representation": "current_calibrated",
        "selected_c": model.selected_c,
        "classifier": {
            "family": "logistic_regression",
            "penalty": "l2",
            "solver": "lbfgs",
            "max_iter": 5000,
            "tol": 1.0e-10,
        },
        "native_probability": True,
        "posthoc_calibrator_selected": False,
        "strict_oof": True,
        "outer_folds": OUTER_FOLD_COUNT,
        "inner_folds": INNER_FOLD_COUNT,
        "row_count": int(len(oof)),
        "available_row_count": int(oof["fusion_available"].sum()),
        "positive_available_row_count": int(
            oof.loc[oof["fusion_available"], "binary_target"].sum()
        ),
        "input_metric_rows": int(len(metrics_summary)),
        "calibration_curve_rows": int(len(calibration_table)),
        "production_boundary": config.payload["production_boundary"],
    }


def _upstream_protection_payload(
    config: MaskedStackingConfig, inputs: MaskedStackingInputs
) -> dict[str, Any]:
    fusion_manifest = inputs.fusion_oof_manifest
    return {
        "version": "mood-social-masked-stacking-upstream-protection-v1",
        "fusion_oof_run_id": "MH-20260802-009",
        "fusion_oof_report_core_sha256": inputs.fusion_oof_core_sha256,
        "fusion_oof_table_sha256": _sha256_file(
            _resolve(
                config.repository_root, config.payload["input"]["fusion_oof_table_path"]
            )
        ),
        "split_sha256": str(config.payload["split"]["sha256"]),
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "baseline_protection_sha256": BASELINE_PROTECTION_SHA256,
        "evaluation_core_sha256": EVALUATION_CORE_SHA256,
        "model006_enters_fusion": False,
        "fusion_table_manifest_version": fusion_manifest.get("version"),
        "frozen_upstream_manifest": fusion_manifest.get("artifacts", {}),
    }


def _publishable_oof(oof: pd.DataFrame) -> pd.DataFrame:
    result = oof.copy()
    result.insert(0, "oof_version", OOF_VERSION)
    result["fusion_input_feature_names"] = json.dumps(
        list(INPUT_FEATURES), separators=(",", ":")
    )
    result["fusion_attention_level_descriptive"] = result["fusion_probability"].map(
        _descriptive_level
    )
    result["fusion_attention_score_descriptive"] = result["fusion_probability"].map(
        _score_half_up
    )
    return result


def _model_card(
    metrics: Mapping[str, Any],
    calibration: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> str:
    natural = metrics.get("natural", {})
    frozen = metrics.get("frozen_four_level", {})
    return "\n".join(
        [
            "# MoodFusionModel V3.3.3",
            "",
            "FUSION-002 publishes a missing-aware L2 Logistic stacking model trained only from the FUSION-001 strict OOF table.",
            "",
            f"- Run: `{RUN_ID}`",
            f"- Selected C: `{selection.get('selected_c')}`",
            "- Probability: native logistic sigmoid of the fused logit; no post-hoc calibrator was selected.",
            "- Mask policy: masks gate evidence and are not learned risk features.",
            "- Threshold: 0.5 is descriptive only; no production work threshold was selected.",
            f"- Natural AUPRC/AUROC/Brier: `{natural.get('auprc')}` / `{natural.get('auroc')}` / `{natural.get('brier')}`",
            f"- Frozen-weight AUPRC/AUROC/Brier: `{frozen.get('auprc')}` / `{frozen.get('auroc')}` / `{frozen.get('brier')}`",
            f"- Natural ECE: `{calibration.get('ece', {}).get('natural')}`",
            "",
            "This artifact is not a medical diagnosis and is not connected to HTTP inference. The PSYCHE-D and social PersonalTrend branches retain their documented proxy limitations.",
        ]
    )


def _validate_table_contract(table: pd.DataFrame) -> None:
    required = {
        "outer_fold",
        "binary_target",
        "global_participant_id",
        "canonical_row_index",
        "dataset_id",
        "mask_pattern",
    }
    missing = required - set(table.columns)
    if missing:
        raise MaskedStackingError(
            f"FUSION-001 table missing columns: {sorted(missing)}"
        )
    if table.empty or table[["dataset_id", "canonical_row_index"]].duplicated().any():
        raise MaskedStackingError("FUSION-001 canonical row identity is invalid")
    if not table["binary_target"].isin([0, 1]).all():
        raise MaskedStackingError("FUSION-001 target is not binary")
    if not table["outer_fold"].isin(range(OUTER_FOLD_COUNT)).all():
        raise MaskedStackingError("FUSION-001 outer fold is invalid")
    _validate_feature_columns(table)


def _validate_feature_columns(frame: pd.DataFrame) -> None:
    required: set[str] = set()
    for mask_name, confidence_name in MASKED_EXPERT_COLUMNS.values():
        required.update({mask_name, confidence_name})
    for mask_name, reliability_name, evidence_name in MASKED_TREND_COLUMNS.values():
        required.update({mask_name, reliability_name, evidence_name})
    for branch in EXPERT_BRANCHES:
        required.add(f"expert_{branch}_current_probability")
    missing = required - set(frame.columns)
    if missing:
        raise MaskedStackingError(
            f"FUSION-002 input columns missing: {sorted(missing)}"
        )


def _inner_fold_value(row: pd.Series, outer_fold: int) -> int:
    value = row["inner_validation_fold_by_outer_fold"]
    if isinstance(value, Mapping):
        value = value.get(str(outer_fold), value.get(outer_fold))
    if value is None or int(value) not in range(INNER_FOLD_COUNT):
        raise MaskedStackingError("DATA-007 inner fold value is invalid")
    return int(value)


def _require_binary(target: pd.Series, scope: str) -> None:
    if target.empty or set(pd.to_numeric(target, errors="raise").astype(int)) != {0, 1}:
        raise MaskedStackingError(f"{scope} does not contain both target classes")


def _weight_audit(frame: pd.DataFrame, weight: np.ndarray) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    target = pd.to_numeric(frame["binary_target"], errors="raise").astype(int)
    for dataset_id in sorted(
        set(frame["dataset_id"].astype(str)), key=lambda value: value.encode("utf-8")
    ):
        selected = frame["dataset_id"].astype(str).eq(dataset_id).to_numpy()
        rows[dataset_id] = {
            "total": float(weight[selected].sum()),
            "by_class": {
                str(value): float(weight[selected & target.eq(value).to_numpy()].sum())
                for value in sorted(set(target[selected].tolist()))
            },
        }
    return {"normalized_total": float(weight.sum()), "by_dataset": rows}


def _ece(
    target: np.ndarray, probability: np.ndarray, weight: np.ndarray
) -> float | None:
    if len(target) == 0 or float(weight.sum()) <= 0:
        return None
    value = 0.0
    for bin_index in range(10):
        lower = bin_index / 10.0
        upper = (bin_index + 1) / 10.0
        selected = (probability >= lower) & (
            (probability < upper) if bin_index < 9 else (probability <= upper)
        )
        if np.any(selected):
            bin_weight = float(weight[selected].sum())
            value += (
                bin_weight
                / float(weight.sum())
                * abs(
                    float(np.average(probability[selected], weights=weight[selected]))
                    - float(np.average(target[selected], weights=weight[selected]))
                )
            )
    return float(value)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype="float64")
    return 1.0 / (1.0 + np.exp(-np.clip(values, -700.0, 700.0)))


def _logit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype="float64")
    return np.log(values / (1.0 - values))


def _descriptive_level(value: Any) -> int | None:
    if value is None or not np.isfinite(float(value)):
        return None
    value = float(value)
    return 0 if value < 0.25 else 1 if value < 0.45 else 2 if value < 0.65 else 3


def _score_half_up(value: Any) -> int | None:
    if value is None or not np.isfinite(float(value)):
        return None
    return int(np.floor(float(value) * 100.0 + 0.5))


def _resolve(root: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (root / path).resolve()


def _validate_output_paths(config: MaskedStackingConfig, *, overwrite: bool) -> None:
    if not overwrite and (
        config.model_path.exists()
        or config.manifest_path.exists()
        or config.report_directory.exists()
    ):
        raise MaskedStackingError(
            "FUSION-002 output exists; pass overwrite=True for deterministic rebuild"
        )
    config.model_path.parent.mkdir(parents=True, exist_ok=True)
    config.manifest_path.parent.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise MaskedStackingError(f"JSON object expected: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _report_core_sha256(report: Path) -> str:
    names = [
        row["path"]
        for row in _file_hash_rows(report, exclude={"run.json", "artifacts.json"})
    ]
    digest = hashlib.sha256()
    for name in sorted(names):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update((report / name).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _deterministic_report_core_sha256(report: Path) -> str:
    return _report_core_sha256(report)


def _file_hash_rows(directory: Path, *, exclude: set[str]) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(
        (path for path in directory.rglob("*") if path.is_file()),
        key=lambda value: value.relative_to(directory).as_posix(),
    ):
        relative = path.relative_to(directory).as_posix()
        if relative in exclude:
            continue
        rows.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    frame.to_parquet(path, index=False, engine="pyarrow", compression="snappy")


def _atomic_publish_file(stage: Path, destination: Path, *, overwrite: bool) -> None:
    if destination.exists() and not overwrite:
        raise MaskedStackingError(f"output exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        temporary.unlink()
    shutil.copy2(stage, temporary)
    temporary.replace(destination)


def _atomic_publish_directory(
    stage: Path, destination: Path, *, overwrite: bool
) -> None:
    if destination.exists() and not overwrite:
        raise MaskedStackingError(f"report exists: {destination}")
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    shutil.copytree(stage, temporary)
    if destination.exists():
        shutil.rmtree(destination)
    temporary.replace(destination)


__all__ = [
    "C_VALUES",
    "INPUT_FEATURES",
    "MaskedStackingConfig",
    "MaskedStackingError",
    "MaskedStackingInputs",
    "MaskedStackingModel",
    "build_fusion_features",
    "confidence_from_coefficients",
    "effective_evidence",
    "evidence_combination",
    "load_masked_stacking_config",
    "load_masked_stacking_inputs",
    "masked_stacking_sample_weights",
    "train_masked_stacking",
]
