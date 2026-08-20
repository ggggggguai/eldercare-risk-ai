from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
from scipy.optimize import minimize
from sklearn.pipeline import Pipeline

from .casme2_traditional import TraditionalConfig, build_pipeline, feature_matrix
from .metrics import classification_metrics


@dataclass(frozen=True)
class TraditionalCandidate:
    candidate_id: str
    feature_groups: tuple[str, ...]
    kernel: str = "rbf"


TRADITIONAL_REGISTRY: dict[str, TraditionalCandidate] = {
    "T6": TraditionalCandidate("T6", ("dense_motion",)),
    "T7": TraditionalCandidate("T7", ("dense_motion", "landmark_features")),
}


@dataclass(frozen=True)
class CalibrationCandidate:
    calibration_id: str
    complexity_rank: int
    regularization: float = 0.0


CALIBRATION_REGISTRY: dict[str, CalibrationCandidate] = {
    "identity": CalibrationCandidate("identity", 0),
    "temperature": CalibrationCandidate("temperature", 1),
    "regularized_vector": CalibrationCandidate("regularized_vector", 2, 0.01),
}

FUSION_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
FUSION_WEIGHT_MEANING = "p=weight*p_causal+(1-weight)*p_traditional"


@dataclass
class FittedTraditional:
    candidate: TraditionalCandidate
    pipeline: Pipeline
    random_state: int
    fit_sample_count: int
    feature_dimension: int


def fit_traditional_from_matrix(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    candidate_id: str,
    random_state: int,
) -> FittedTraditional:
    if candidate_id not in TRADITIONAL_REGISTRY:
        raise ValueError(f"unknown traditional candidate: {candidate_id}")
    matrix = np.asarray(features, dtype=np.float32)
    target = np.asarray(labels, dtype=np.int64)
    if matrix.ndim != 2 or len(matrix) != len(target) or not np.isfinite(matrix).all():
        raise ValueError("traditional fit requires finite [samples,features] input")
    if set(target.tolist()) != {0, 1, 2}:
        raise ValueError("traditional fit requires all three classes in fold train")
    candidate = TRADITIONAL_REGISTRY[candidate_id]
    config = TraditionalConfig(
        feature_groups=candidate.feature_groups,
        pca_components=min(32, matrix.shape[1]),
        kernel=candidate.kernel,
        c=1.0,
        class_weight="balanced",
        random_state=random_state,
    )
    pipeline = build_pipeline(config, max_samples=len(matrix))
    pipeline.fit(matrix, target)
    probabilities = pipeline.predict_proba(matrix)
    if probabilities.shape != (len(matrix), 3) or not np.isfinite(probabilities).all():
        raise RuntimeError("traditional pipeline produced invalid training probabilities")
    return FittedTraditional(
        candidate=candidate,
        pipeline=pipeline,
        random_state=random_state,
        fit_sample_count=len(matrix),
        feature_dimension=matrix.shape[1],
    )


def fit_traditional_records(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate_id: str,
    random_state: int,
) -> tuple[FittedTraditional, np.ndarray]:
    candidate = TRADITIONAL_REGISTRY[candidate_id]
    artifact_rows = [
        {
            **dict(row),
            "artifact_path": str(row["traditional_resolved_path"]),
            "artifact_sha256": str(row["traditional_artifact_sha256"]),
        }
        for row in rows
    ]
    matrix = feature_matrix(artifact_rows, candidate.feature_groups)
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    return (
        fit_traditional_from_matrix(
            matrix,
            labels,
            candidate_id=candidate_id,
            random_state=random_state,
        ),
        matrix,
    )


def traditional_probability_rows(
    fitted: FittedTraditional,
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    artifact_rows = [
        {
            **dict(row),
            "artifact_path": str(row["traditional_resolved_path"]),
            "artifact_sha256": str(row["traditional_artifact_sha256"]),
        }
        for row in rows
    ]
    matrix = feature_matrix(artifact_rows, fitted.candidate.feature_groups)
    probabilities = fitted.pipeline.predict_proba(matrix)
    classes = [int(value) for value in fitted.pipeline.named_steps["classifier"].classes_]
    if classes != [0, 1, 2]:
        raise RuntimeError(f"unexpected traditional classifier classes: {classes}")
    return [
        {
            "sample_id": str(row["sample_id"]),
            "subject_id": str(row["subject_id"]),
            "label": int(row["label"]),
            "prediction": int(np.argmax(probability)),
            "probabilities": probability.tolist(),
        }
        for row, probability in zip(rows, probabilities, strict=True)
    ]


def save_me3_traditional_checkpoint(
    path: Path, fitted: FittedTraditional, *, metadata: Mapping[str, Any]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "schema_version": "causalnet_opt_me_003_traditional_checkpoint_v1",
            "candidate": asdict(fitted.candidate),
            "pipeline": fitted.pipeline,
            "random_state": fitted.random_state,
            "fit_sample_count": fitted.fit_sample_count,
            "feature_dimension": fitted.feature_dimension,
            "metadata": dict(metadata),
        },
        path,
    )


def load_me3_traditional_checkpoint(path: Path) -> FittedTraditional:
    payload = joblib.load(path)
    if payload.get("schema_version") != "causalnet_opt_me_003_traditional_checkpoint_v1":
        raise ValueError("unsupported ME3 traditional checkpoint schema")
    candidate_payload = payload.get("candidate", {})
    candidate_id = str(candidate_payload.get("candidate_id", ""))
    if candidate_id not in TRADITIONAL_REGISTRY:
        raise ValueError("traditional checkpoint candidate is outside frozen registry")
    candidate = TRADITIONAL_REGISTRY[candidate_id]
    if candidate_payload != asdict(candidate):
        raise ValueError("traditional checkpoint candidate contract drift")
    return FittedTraditional(
        candidate=candidate,
        pipeline=payload["pipeline"],
        random_state=int(payload["random_state"]),
        fit_sample_count=int(payload["fit_sample_count"]),
        feature_dimension=int(payload["feature_dimension"]),
    )


def _probability_matrix(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    probabilities = np.asarray([row["probabilities"] for row in rows], dtype=np.float64)
    if probabilities.shape != (len(rows), 3):
        raise ValueError("probability rows must contain exactly three classes")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0.0):
        raise ValueError("probabilities must be finite and non-negative")
    totals = probabilities.sum(axis=1, keepdims=True)
    if np.any(totals <= 0.0):
        raise ValueError("probability row has zero mass")
    return probabilities / totals


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def _nll(probabilities: np.ndarray, labels: np.ndarray) -> float:
    clipped = np.clip(probabilities, 1e-12, 1.0)
    return float(-np.mean(np.log(clipped[np.arange(len(labels)), labels])))


def _brier(probabilities: np.ndarray, labels: np.ndarray) -> float:
    targets = np.eye(3, dtype=np.float64)[labels]
    return float(np.mean(np.sum((probabilities - targets) ** 2, axis=1)))


@dataclass(frozen=True)
class FittedCalibrator:
    calibration_id: str
    parameters: tuple[float, ...]


def fit_calibrator(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    calibration_id: str,
) -> FittedCalibrator:
    if calibration_id not in CALIBRATION_REGISTRY:
        raise ValueError(f"unknown calibration candidate: {calibration_id}")
    matrix = np.asarray(probabilities, dtype=np.float64)
    target = np.asarray(labels, dtype=np.int64)
    if matrix.shape != (len(target), 3) or set(target.tolist()) != {0, 1, 2}:
        raise ValueError("calibration fit requires finite three-class training data")
    matrix = matrix / matrix.sum(axis=1, keepdims=True)
    logits = np.log(np.clip(matrix, 1e-12, 1.0))
    if calibration_id == "identity":
        return FittedCalibrator("identity", ())
    if calibration_id == "temperature":
        objective = lambda value: _nll(_softmax(logits / np.exp(value[0])), target)
        result = minimize(
            objective,
            np.zeros(1, dtype=np.float64),
            method="L-BFGS-B",
            bounds=[(np.log(0.05), np.log(10.0))],
        )
        if not result.success or not np.isfinite(result.fun):
            raise RuntimeError("temperature calibration optimization failed")
        return FittedCalibrator("temperature", (float(np.exp(result.x[0])),))
    regularization = CALIBRATION_REGISTRY[calibration_id].regularization

    def vector_objective(value: np.ndarray) -> float:
        log_scales = value[:3]
        biases = value[3:]
        calibrated = _softmax(logits * np.exp(log_scales)[None, :] + biases[None, :])
        penalty = regularization * float(np.sum(log_scales**2) + np.sum(biases**2))
        return _nll(calibrated, target) + penalty

    result = minimize(
        vector_objective,
        np.zeros(6, dtype=np.float64),
        method="L-BFGS-B",
        bounds=[(-2.0, 2.0)] * 3 + [(-2.0, 2.0)] * 3,
    )
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError("regularized vector calibration optimization failed")
    return FittedCalibrator(
        "regularized_vector", tuple(float(value) for value in result.x)
    )


def apply_calibrator(
    probabilities: np.ndarray, fitted: FittedCalibrator
) -> np.ndarray:
    matrix = np.asarray(probabilities, dtype=np.float64)
    matrix = matrix / matrix.sum(axis=1, keepdims=True)
    if fitted.calibration_id == "identity":
        return matrix
    logits = np.log(np.clip(matrix, 1e-12, 1.0))
    if fitted.calibration_id == "temperature":
        return _softmax(logits / fitted.parameters[0])
    if fitted.calibration_id != "regularized_vector" or len(fitted.parameters) != 6:
        raise ValueError("invalid fitted calibrator")
    values = np.asarray(fitted.parameters, dtype=np.float64)
    return _softmax(logits * np.exp(values[:3])[None, :] + values[3:][None, :])


def probability_metric_bundle(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("metric bundle requires rows")
    probabilities = _probability_matrix(rows)
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    predictions = probabilities.argmax(axis=1)
    metrics = classification_metrics(labels, predictions)
    subjects = sorted({str(row["subject_id"]) for row in rows})
    per_subject = {
        subject: classification_metrics(
            [int(row["label"]) for row in rows if str(row["subject_id"]) == subject],
            [
                int(np.argmax(row["probabilities"]))
                for row in rows
                if str(row["subject_id"]) == subject
            ],
        )
        for subject in subjects
    }
    zero_subjects = [
        subject
        for subject, value in per_subject.items()
        if float(value["uf1"]) <= 0.0 or float(value["uar"]) <= 0.0
    ]
    return {
        **metrics,
        "selection_score": min(float(metrics["uf1"]), float(metrics["uar"])),
        "nll": _nll(probabilities, labels),
        "brier": _brier(probabilities, labels),
        "per_subject": per_subject,
        "zero_score_subjects": zero_subjects,
        "zero_score_subject_count": len(zero_subjects),
    }


def _ranking_tuple(metrics: Mapping[str, Any], complexity: float = 0.0) -> tuple[float, ...]:
    positive = metrics["per_class"]["positive"]
    return (
        -float(metrics["selection_score"]),
        -float(positive["f1"]),
        -float(positive["recall"]),
        float(metrics["zero_score_subject_count"]),
        float(complexity),
    )


def crossfit_select_calibration(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    folds = sorted({str(row["fold_id"]) for row in rows})
    if len(folds) < 2:
        raise ValueError("cross-fitted calibration requires at least two folds")
    candidate_outputs: dict[str, Any] = {}
    for calibration_id, contract in CALIBRATION_REGISTRY.items():
        output_rows: list[dict[str, Any]] = []
        fold_parameters: dict[str, list[float]] = {}
        for fold_id in folds:
            fit_rows = [row for row in rows if str(row["fold_id"]) != fold_id]
            held_rows = [row for row in rows if str(row["fold_id"]) == fold_id]
            fitted = fit_calibrator(
                _probability_matrix(fit_rows),
                np.asarray([int(row["label"]) for row in fit_rows], dtype=np.int64),
                calibration_id=calibration_id,
            )
            calibrated = apply_calibrator(_probability_matrix(held_rows), fitted)
            fold_parameters[fold_id] = list(fitted.parameters)
            output_rows.extend(
                {
                    **dict(row),
                    "probabilities": probability.tolist(),
                    "prediction": int(np.argmax(probability)),
                    "calibration_id": calibration_id,
                }
                for row, probability in zip(held_rows, calibrated, strict=True)
            )
        metrics = probability_metric_bundle(output_rows)
        candidate_outputs[calibration_id] = {
            "calibration_id": calibration_id,
            "rows": output_rows,
            "metrics": metrics,
            "fold_parameters": fold_parameters,
            "complexity_rank": contract.complexity_rank,
        }
    selected_id = min(
        candidate_outputs,
        key=lambda value: (
            *_ranking_tuple(candidate_outputs[value]["metrics"]),
            float(candidate_outputs[value]["metrics"]["nll"]),
            float(candidate_outputs[value]["metrics"]["brier"]),
            candidate_outputs[value]["complexity_rank"],
            value,
        ),
    )
    return {
        "selected_calibration_id": selected_id,
        "selected_rows": candidate_outputs[selected_id]["rows"],
        "candidates": {
            key: {name: value for name, value in payload.items() if name != "rows"}
            for key, payload in candidate_outputs.items()
        },
    }


def crossfit_select_fusion(
    causal_rows: Sequence[Mapping[str, Any]],
    traditional_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    causal_by_id = {str(row["sample_id"]): row for row in causal_rows}
    traditional_by_id = {str(row["sample_id"]): row for row in traditional_rows}
    if causal_by_id.keys() != traditional_by_id.keys():
        raise ValueError("fusion component sample coverage mismatch")
    ordered = [causal_by_id[sample_id] for sample_id in sorted(causal_by_id)]
    folds = sorted({str(row["fold_id"]) for row in ordered})
    selected_weights: dict[str, float] = {}
    output_rows: list[dict[str, Any]] = []
    weight_diagnostics: dict[str, Any] = {}
    for held_fold in folds:
        fit_ids = [
            str(row["sample_id"]) for row in ordered if str(row["fold_id"]) != held_fold
        ]
        held_ids = [
            str(row["sample_id"]) for row in ordered if str(row["fold_id"]) == held_fold
        ]
        candidate_metrics: dict[str, Any] = {}
        for weight in FUSION_WEIGHTS:
            fit_rows = []
            for sample_id in fit_ids:
                causal = causal_by_id[sample_id]
                traditional = traditional_by_id[sample_id]
                probability = weight * np.asarray(causal["probabilities"]) + (
                    1.0 - weight
                ) * np.asarray(traditional["probabilities"])
                fit_rows.append({**dict(causal), "probabilities": probability.tolist()})
            candidate_metrics[str(weight)] = probability_metric_bundle(fit_rows)
        selected = min(
            FUSION_WEIGHTS,
            key=lambda weight: (
                *_ranking_tuple(candidate_metrics[str(weight)], abs(weight - 0.5)),
                abs(weight - 0.5),
                -weight,
            ),
        )
        selected_weights[held_fold] = selected
        weight_diagnostics[held_fold] = candidate_metrics
        for sample_id in held_ids:
            causal = causal_by_id[sample_id]
            traditional = traditional_by_id[sample_id]
            probability = selected * np.asarray(causal["probabilities"]) + (
                1.0 - selected
            ) * np.asarray(traditional["probabilities"])
            output_rows.append(
                {
                    **dict(causal),
                    "probabilities": probability.tolist(),
                    "prediction": int(np.argmax(probability)),
                    "fusion_weight": selected,
                    "fusion_weight_meaning": FUSION_WEIGHT_MEANING,
                }
            )
    return {
        "rows": output_rows,
        "metrics": probability_metric_bundle(output_rows),
        "selected_weights_by_held_fold": selected_weights,
        "weight_diagnostics": weight_diagnostics,
        "weight_meaning": FUSION_WEIGHT_MEANING,
    }


def average_probability_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["sample_id"]), []).append(row)
    output: list[dict[str, Any]] = []
    for sample_id in sorted(grouped):
        members = grouped[sample_id]
        labels = {int(row["label"]) for row in members}
        subjects = {str(row["subject_id"]) for row in members}
        if len(labels) != 1 or len(subjects) != 1:
            raise ValueError(f"repeated probability metadata mismatch: {sample_id}")
        probability = np.mean(_probability_matrix(members), axis=0)
        output.append(
            {
                "sample_id": sample_id,
                "subject_id": next(iter(subjects)),
                "label": next(iter(labels)),
                "prediction": int(np.argmax(probability)),
                "probabilities": probability.tolist(),
                "member_count": len(members),
            }
        )
    return output


def paired_subject_bootstrap(
    baseline_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    iterations: int = 2000,
    seed: int = 20260810,
) -> dict[str, Any]:
    baseline = {str(row["sample_id"]): row for row in baseline_rows}
    candidate = {str(row["sample_id"]): row for row in candidate_rows}
    if baseline.keys() != candidate.keys():
        raise ValueError("paired bootstrap sample coverage mismatch")
    subjects = sorted({str(row["subject_id"]) for row in baseline.values()})
    by_subject = {
        subject: sorted(
            sample_id
            for sample_id, row in baseline.items()
            if str(row["subject_id"]) == subject
        )
        for subject in subjects
    }
    generator = np.random.default_rng(seed)
    deltas = {"uf1": [], "uar": [], "selection_score": []}

    def bootstrap_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
        probabilities = _probability_matrix(rows)
        labels = [int(row["label"]) for row in rows]
        metrics = classification_metrics(labels, probabilities.argmax(axis=1))
        return {
            "uf1": float(metrics["uf1"]),
            "uar": float(metrics["uar"]),
            "selection_score": min(float(metrics["uf1"]), float(metrics["uar"])),
        }

    for _ in range(iterations):
        sampled_subjects = generator.choice(subjects, size=len(subjects), replace=True)
        baseline_sample: list[dict[str, Any]] = []
        candidate_sample: list[dict[str, Any]] = []
        for copy_index, subject in enumerate(sampled_subjects):
            for sample_id in by_subject[str(subject)]:
                baseline_sample.append(
                    {**dict(baseline[sample_id]), "sample_id": f"{copy_index}:{sample_id}"}
                )
                candidate_sample.append(
                    {**dict(candidate[sample_id]), "sample_id": f"{copy_index}:{sample_id}"}
                )
        baseline_metrics = bootstrap_metrics(baseline_sample)
        candidate_metrics = bootstrap_metrics(candidate_sample)
        for metric in deltas:
            deltas[metric].append(
                float(candidate_metrics[metric]) - float(baseline_metrics[metric])
            )
    return {
        "iterations": iterations,
        "seed": seed,
        "delta_95_ci": {
            metric: [
                float(np.quantile(values, 0.025)),
                float(np.quantile(values, 0.975)),
            ]
            for metric, values in deltas.items()
        },
        "delta_mean": {metric: float(np.mean(values)) for metric, values in deltas.items()},
    }
