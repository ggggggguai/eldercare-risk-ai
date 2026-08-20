from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression

from .metrics import classification_metrics


CALIBRATION_METHODS = ("identity", "temperature", "vector_scaling")
TEMPERATURE_GRID = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
FUSION_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
FINAL_SEEDS = (20260806, 20260817, 20260829)


def _normalize(probabilities: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-8, 1.0)
    return value / value.sum(axis=1, keepdims=True)


def multiclass_brier(labels: np.ndarray, probabilities: np.ndarray) -> float:
    target = np.eye(3, dtype=np.float64)[labels]
    return float(np.mean(np.sum((probabilities - target) ** 2, axis=1)))


def nll(labels: np.ndarray, probabilities: np.ndarray) -> float:
    return float(-np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1.0)).mean())


def selection_score(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, float, float]:
    metrics = classification_metrics(labels, np.argmax(probabilities, axis=1))
    harmonic = 2 * metrics["uf1"] * metrics["uar"] / max(metrics["uf1"] + metrics["uar"], 1e-12)
    return harmonic, metrics["per_class"]["positive"]["f1"], metrics["per_class"]["positive"]["recall"]


def fit_calibrator(method: str, labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    probabilities = _normalize(probabilities)
    logits = np.log(probabilities)
    if method == "identity":
        return {"method": method}
    if method == "temperature":
        losses = {temperature: nll(labels, _normalize(np.exp(logits / temperature))) for temperature in TEMPERATURE_GRID}
        temperature = min(losses, key=lambda value: (losses[value], value))
        return {"method": method, "temperature": float(temperature)}
    if method == "vector_scaling":
        model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=20260808)
        model.fit(logits, labels)
        return {"method": method, "coef": model.coef_.tolist(), "intercept": model.intercept_.tolist(), "classes": model.classes_.tolist()}
    raise ValueError(f"Unknown calibration method: {method}")


def apply_calibrator(calibrator: Mapping[str, Any], probabilities: np.ndarray) -> np.ndarray:
    probabilities = _normalize(probabilities)
    method = calibrator["method"]
    if method == "identity":
        return probabilities
    logits = np.log(probabilities)
    if method == "temperature":
        scaled = logits / float(calibrator["temperature"])
    elif method == "vector_scaling":
        scaled = logits @ np.asarray(calibrator["coef"], dtype=np.float64).T + np.asarray(calibrator["intercept"], dtype=np.float64)
    else:
        raise ValueError(f"Unknown calibration method: {method}")
    scaled -= scaled.max(axis=1, keepdims=True)
    return _normalize(np.exp(scaled))


def crossfit_calibration(predictions: Sequence[Mapping[str, Any]], method: str) -> tuple[np.ndarray, list[dict[str, Any]]]:
    labels = np.asarray([row["true_label"] for row in predictions], dtype=np.int64)
    probabilities = np.asarray([row["probabilities"] for row in predictions], dtype=np.float64)
    folds = np.asarray([row["inner_fold"] for row in predictions], dtype=np.int64)
    calibrated = np.zeros_like(probabilities)
    contracts = []
    for fold in sorted(set(folds.tolist())):
        train, validation = folds != fold, folds == fold
        calibrator = fit_calibrator(method, labels[train], probabilities[train])
        calibrated[validation] = apply_calibrator(calibrator, probabilities[validation])
        contracts.append({"held_out_inner_fold": int(fold), "calibrator": calibrator, "fit_sample_count": int(train.sum()), "apply_sample_count": int(validation.sum())})
    return calibrated, contracts


def crossfit_fusion(labels: np.ndarray, folds: np.ndarray, deep: np.ndarray, traditional: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
    fused = np.zeros_like(deep)
    contracts = []
    for fold in sorted(set(folds.tolist())):
        train, validation = folds != fold, folds == fold
        scores = {weight: selection_score(labels[train], _normalize(weight * deep[train] + (1 - weight) * traditional[train])) for weight in FUSION_WEIGHTS}
        weight = max(scores, key=lambda value: (scores[value], -abs(value - 0.5)))
        fused[validation] = _normalize(weight * deep[validation] + (1 - weight) * traditional[validation])
        contracts.append({"held_out_inner_fold": int(fold), "deep_weight": float(weight), "traditional_weight": float(1 - weight), "fit_sample_count": int(train.sum()), "apply_sample_count": int(validation.sum())})
    return fused, contracts


def equal_seed_probability_ensemble(probabilities: Sequence[np.ndarray]) -> np.ndarray:
    if len(probabilities) != 3 or any(np.asarray(value).shape != np.asarray(probabilities[0]).shape for value in probabilities):
        raise ValueError("Three aligned seed probability arrays are required")
    return _normalize(np.mean(np.stack(probabilities), axis=0))


def freeze_outer_contract(outer_result: Mapping[str, Any]) -> dict[str, Any]:
    deep_result = next(value for value in outer_result["deep_candidates"] if value["candidate"]["candidate_id"] == outer_result["selected_deep_candidate"])
    traditional_result = next(value for value in outer_result["traditional_candidates"] if value["candidate"]["candidate_id"] == outer_result["selected_traditional_candidate"])
    deep_by_id = {row["sample_id"]: row for row in deep_result["predictions"]}
    traditional_by_id = {row["sample_id"]: row for row in traditional_result["predictions"]}
    sample_ids = sorted(deep_by_id)
    if sample_ids != sorted(traditional_by_id):
        raise ValueError("Deep/traditional inner OOF population mismatch")
    aligned_deep = [deep_by_id[sample_id] for sample_id in sample_ids]
    aligned_traditional = [traditional_by_id[sample_id] for sample_id in sample_ids]
    labels = np.asarray([row["true_label"] for row in aligned_deep], dtype=np.int64)
    folds = np.asarray([row["inner_fold"] for row in aligned_deep], dtype=np.int64)
    raw_deep = np.asarray([row["probabilities"] for row in aligned_deep], dtype=np.float64)
    raw_traditional = np.asarray([row["probabilities"] for row in aligned_traditional], dtype=np.float64)
    calibration_results = {}
    for method in CALIBRATION_METHODS:
        probabilities, contracts = crossfit_calibration(aligned_deep, method)
        calibration_results[method] = {"probabilities": probabilities, "contracts": contracts, "brier": multiclass_brier(labels, probabilities), "nll": nll(labels, probabilities), "metrics": classification_metrics(labels, np.argmax(probabilities, axis=1))}
    selected_method = min(CALIBRATION_METHODS, key=lambda method: (calibration_results[method]["brier"], calibration_results[method]["nll"], CALIBRATION_METHODS.index(method)))
    calibrated_deep = calibration_results[selected_method]["probabilities"]
    fused, fusion_contracts = crossfit_fusion(labels, folds, calibrated_deep, raw_traditional)
    options = {
        "deep": calibrated_deep,
        "traditional": raw_traditional,
        "deep_traditional_fusion": fused,
    }
    option_metrics = {name: classification_metrics(labels, np.argmax(value, axis=1)) for name, value in options.items()}
    selected_option = max(options, key=lambda name: (selection_score(labels, options[name]), -list(options).index(name)))
    return {
        "outer_fold": outer_result["outer_fold"],
        "selected_deep_candidate": outer_result["selected_deep_candidate"],
        "selected_traditional_candidate": outer_result["selected_traditional_candidate"],
        "selected_calibration": selected_method,
        "calibration_comparison": {method: {key: value for key, value in result.items() if key != "probabilities"} for method, result in calibration_results.items()},
        "fusion_contracts": fusion_contracts,
        "option_metrics": option_metrics,
        "selected_output": selected_option,
        "selected_inner_validation_metrics": option_metrics[selected_option],
        "inner_oof_sample_count": len(sample_ids),
        "inner_oof_sample_ids_sha256": __import__("hashlib").sha256("\n".join(sample_ids).encode()).hexdigest(),
        "outer_test_predictions_generated": False,
    }


def audit_frozen_release(*, selection: Mapping[str, Any], freeze: Mapping[str, Any], split: Mapping[str, Any]) -> dict[str, Any]:
    errors = []
    if selection.get("outer_test_predictions_generated") is not False or freeze.get("outer_test_predictions_generated") is not False:
        errors.append({"code": "outer_test_access_flag"})
    if len(freeze.get("outer_contracts", [])) != len(selection.get("outer_results", [])):
        errors.append({"code": "outer_contract_count"})
    split_by_fold = {fold["outer_fold"]: fold for fold in split["outer_folds"]}
    for contract in freeze.get("outer_contracts", []):
        expected = len(split_by_fold[contract["outer_fold"]]["outer_train_sample_ids"])
        if contract["inner_oof_sample_count"] != expected:
            errors.append({"code": "inner_oof_coverage", "outer_fold": contract["outer_fold"], "actual": contract["inner_oof_sample_count"], "expected": expected})
    config = freeze.get("formal_evaluation_config", {})
    if config.get("final_seeds") != list(FINAL_SEEDS) or config.get("three_seed_ensemble") != "equal_probability_mean":
        errors.append({"code": "ensemble_contract"})
    return {"schema_version": "opt_me_002_freeze_audit_v1", "status": "passed" if not errors else "failed", "error_count": len(errors), "errors": errors, "outer_contracts": len(freeze.get("outer_contracts", [])), "guards": {"outer_test_predictions_generated": False, "outer_test_metrics_generated": False, "formal_evaluation_started": False}}
