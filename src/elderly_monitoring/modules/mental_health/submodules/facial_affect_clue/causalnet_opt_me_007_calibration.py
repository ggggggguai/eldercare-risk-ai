from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np

from .causalnet_opt_me_004_evaluation import metric_bundle


TEMPERATURE_GRID = (1.000, 1.025, 1.050, 1.075, 1.100, 1.125, 1.150)
CANDIDATE_TEMPERATURES: dict[str, float | None] = {
    "C0": 1.0,
    "C1": 1.05,
    "C2": 1.10,
    "C3": None,
}


def probability_rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = [
        {
            "sample_id": str(row["sample_id"]),
            "subject_id": str(row["subject_id"]),
            "label": int(row["label"]),
            "probabilities": [float(value) for value in row["probabilities"]],
        }
        for row in sorted(rows, key=lambda item: str(item["sample_id"]))
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def power_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    matrix = np.asarray(probabilities, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
        squeeze = True
    else:
        squeeze = False
    if matrix.ndim != 2 or matrix.shape[1] != 3:
        raise ValueError("probabilities must have shape [n,3] or [3]")
    if not np.isfinite(matrix).all() or np.any(matrix < 0.0):
        raise ValueError("probabilities must be finite and non-negative")
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    row_sums = matrix.sum(axis=1)
    if np.any(row_sums <= 0.0):
        raise ValueError("probability rows must have positive mass")
    normalized = matrix / row_sums[:, None]
    before = normalized.argmax(axis=1)
    log_values = np.log(np.clip(normalized, 1e-12, 1.0)) / float(temperature)
    log_values -= log_values.max(axis=1, keepdims=True)
    transformed = np.exp(log_values)
    transformed /= transformed.sum(axis=1, keepdims=True)
    if not np.isfinite(transformed).all() or not np.allclose(
        transformed.sum(axis=1), 1.0, rtol=0.0, atol=1e-12
    ):
        raise RuntimeError("temperature transform produced invalid probabilities")
    if not np.array_equal(before, transformed.argmax(axis=1)):
        raise RuntimeError("shared scalar temperature changed argmax")
    return transformed[0] if squeeze else transformed


def apply_temperature_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    temperature: float,
    candidate_id: str,
    calibration_group: int | str,
) -> list[dict[str, Any]]:
    if candidate_id not in CANDIDATE_TEMPERATURES:
        raise ValueError(f"unknown OPT-ME-007 calibration candidate: {candidate_id}")
    matrix = np.asarray([row["probabilities"] for row in rows], dtype=np.float64)
    transformed = power_temperature(matrix, temperature)
    output: list[dict[str, Any]] = []
    for source, values in zip(rows, transformed, strict=True):
        original = [float(value) for value in source["probabilities"]]
        result = dict(source)
        result.update(
            {
                "pre_temperature_probabilities": original,
                "probabilities": [float(value) for value in values],
                "prediction": int(np.argmax(values)),
                "calibration_candidate": candidate_id,
                "ensemble_temperature": float(temperature),
                "calibration_group": calibration_group,
                "argmax_preserved": int(np.argmax(original)) == int(np.argmax(values)),
            }
        )
        output.append(result)
    if not all(row["argmax_preserved"] for row in output):
        raise RuntimeError("temperature candidate changed a sample prediction")
    return output


def fixed_temperature_calibration(
    rows: Sequence[Mapping[str, Any]], candidate_id: str
) -> dict[str, Any]:
    if candidate_id not in ("C0", "C1", "C2"):
        raise ValueError("fixed calibration is only defined for C0-C2")
    temperature = float(CANDIDATE_TEMPERATURES[candidate_id])
    selected = apply_temperature_rows(
        rows,
        temperature=temperature,
        candidate_id=candidate_id,
        calibration_group="fixed_global",
    )
    return {
        "selected_rows": selected,
        "registry": [
            {
                "candidate_id": candidate_id,
                "mode": "fixed_scalar_power_temperature",
                "temperature": temperature,
                "fit_subject_ids": [],
                "held_subject_ids": sorted({str(row["subject_id"]) for row in rows}),
                "input_probability_sha256": probability_rows_sha256(rows),
                "output_probability_sha256": probability_rows_sha256(selected),
                "argmax_identity": all(row["argmax_preserved"] for row in selected),
            }
        ],
    }


def crossfit_subject_temperature(
    rows: Sequence[Mapping[str, Any]],
    subject_to_group: Mapping[str, int],
    *,
    temperature_grid: Sequence[float] = TEMPERATURE_GRID,
) -> dict[str, Any]:
    if tuple(float(value) for value in temperature_grid) != TEMPERATURE_GRID:
        raise ValueError("C3 temperature grid drift")
    subjects = sorted({str(row["subject_id"]) for row in rows})
    if not subjects or any(subject not in subject_to_group for subject in subjects):
        raise ValueError("every probability-row subject must have a calibration group")
    group_ids = sorted({int(subject_to_group[subject]) for subject in subjects})
    if len(group_ids) < 2:
        raise ValueError("C3 requires at least two populated calibration groups")
    output_by_sample: dict[str, dict[str, Any]] = {}
    registry: list[dict[str, Any]] = []
    for held_group in group_ids:
        held = [row for row in rows if int(subject_to_group[str(row["subject_id"])]) == held_group]
        fit = [row for row in rows if int(subject_to_group[str(row["subject_id"])]) != held_group]
        held_subjects = sorted({str(row["subject_id"]) for row in held})
        fit_subjects = sorted({str(row["subject_id"]) for row in fit})
        if not held or not fit or set(held_subjects) & set(fit_subjects):
            raise RuntimeError("C3 fit/held subject isolation failed")
        reference = metric_bundle(fit)
        trials: list[dict[str, Any]] = []
        for temperature in TEMPERATURE_GRID:
            candidate_rows = apply_temperature_rows(
                fit,
                temperature=temperature,
                candidate_id="C3",
                calibration_group=held_group,
            )
            metrics = metric_bundle(candidate_rows)
            valid = (
                float(metrics["brier"]) - float(reference["brier"]) <= 0.005 + 1e-12
                and float(metrics["nll"]) - float(reference["nll"]) <= 0.010 + 1e-12
                and float(metrics["high_confidence_error_rate"])
                - float(reference["high_confidence_error_rate"])
                <= 0.010 + 1e-12
            )
            trials.append(
                {
                    "temperature": float(temperature),
                    "valid": bool(valid),
                    "ece": float(metrics["ece"]),
                    "nll": float(metrics["nll"]),
                    "brier": float(metrics["brier"]),
                    "high_confidence_error_rate": float(metrics["high_confidence_error_rate"]),
                }
            )
        valid_trials = [trial for trial in trials if trial["valid"]]
        if not valid_trials:
            raise RuntimeError(f"C3 group {held_group} has no constraint-valid temperature")
        chosen = min(
            valid_trials,
            key=lambda trial: (
                trial["ece"],
                trial["nll"],
                trial["brier"],
                abs(float(trial["temperature"]) - 1.0),
                trial["temperature"],
            ),
        )
        selected = apply_temperature_rows(
            held,
            temperature=float(chosen["temperature"]),
            candidate_id="C3",
            calibration_group=held_group,
        )
        for row in selected:
            sample_id = str(row["sample_id"])
            if sample_id in output_by_sample:
                raise RuntimeError(f"C3 sample assigned twice: {sample_id}")
            output_by_sample[sample_id] = row
        registry.append(
            {
                "candidate_id": "C3",
                "mode": "held_subject_crossfit_scalar_power_temperature",
                "held_group": held_group,
                "temperature": float(chosen["temperature"]),
                "fit_subject_ids": fit_subjects,
                "held_subject_ids": held_subjects,
                "fit_sample_count": len(fit),
                "held_sample_count": len(held),
                "fit_held_subject_intersection": [],
                "input_probability_sha256": probability_rows_sha256(held),
                "output_probability_sha256": probability_rows_sha256(selected),
                "argmax_identity": all(row["argmax_preserved"] for row in selected),
                "trials": trials,
                "selection_order": ["ece", "nll", "brier", "abs_temperature_from_one", "temperature"],
            }
        )
    ordered = [output_by_sample[str(row["sample_id"])] for row in rows]
    if len(ordered) != len(rows):
        raise RuntimeError("C3 cross-fitted output coverage mismatch")
    return {"selected_rows": ordered, "registry": registry}


def calibrate_candidate(
    rows: Sequence[Mapping[str, Any]],
    candidate_id: str,
    *,
    subject_to_group: Mapping[str, int],
) -> dict[str, Any]:
    if candidate_id == "C3":
        return crossfit_subject_temperature(rows, subject_to_group)
    return fixed_temperature_calibration(rows, candidate_id)


__all__ = [
    "CANDIDATE_TEMPERATURES",
    "TEMPERATURE_GRID",
    "apply_temperature_rows",
    "calibrate_candidate",
    "crossfit_subject_temperature",
    "fixed_temperature_calibration",
    "power_temperature",
    "probability_rows_sha256",
]
