from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np


EQUAL_WEIGHT = 0.5
P0_CONFIDENCE_MARGIN = 0.05
A2_TEMPERATURE = 0.85


def _probability_vector(values: Sequence[float]) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all() or np.any(vector < 0.0):
        raise ValueError("expected finite non-negative three-class probability vector")
    total = float(vector.sum())
    if total <= 0.0:
        raise ValueError("probability vector has zero mass")
    return vector / total


def selective_gate(
    b0_probabilities: Sequence[float], p0_probabilities: Sequence[float]
) -> tuple[np.ndarray, str]:
    """Return the frozen A1 probability and a label-free decision reason."""
    b0 = _probability_vector(b0_probabilities)
    p0 = _probability_vector(p0_probabilities)
    agree = int(np.argmax(b0)) == int(np.argmax(p0))
    margin = float(np.max(p0) - np.max(b0))
    if agree:
        return EQUAL_WEIGHT * b0 + EQUAL_WEIGHT * p0, "agree_mix"
    if margin + 1e-12 >= P0_CONFIDENCE_MARGIN:
        return EQUAL_WEIGHT * b0 + EQUAL_WEIGHT * p0, "disagreement_mix"
    return b0.copy(), "fallback_B0"


def power_temperature(
    probabilities: Sequence[float], temperature: float = A2_TEMPERATURE
) -> np.ndarray:
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    vector = _probability_vector(probabilities)
    powered = np.power(np.clip(vector, 1e-12, 1.0), 1.0 / float(temperature))
    return powered / powered.sum()


def probability_rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = [
        {
            "sample_id": str(row["sample_id"]),
            "probabilities": [float(value) for value in row["probabilities"]],
        }
        for row in sorted(rows, key=lambda item: str(item["sample_id"]))
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def derive_selective_candidates(
    b0_rows: Sequence[Mapping[str, Any]],
    p0_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    b0 = {str(row["sample_id"]): dict(row) for row in b0_rows}
    p0 = {str(row["sample_id"]): dict(row) for row in p0_rows}
    if b0.keys() != p0.keys() or not b0:
        raise ValueError("B0/P0 sample coverage mismatch")
    a1_rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    counts = {"agree_mix": 0, "disagreement_mix": 0, "fallback_B0": 0}
    for sample_id in sorted(b0):
        left, right = b0[sample_id], p0[sample_id]
        if int(left["label"]) != int(right["label"]) or str(left["subject_id"]) != str(
            right["subject_id"]
        ):
            raise ValueError(f"B0/P0 metadata mismatch: {sample_id}")
        probability, reason = selective_gate(left["probabilities"], right["probabilities"])
        counts[reason] += 1
        a1_rows.append(
            {
                **left,
                "candidate_id": "A1",
                "prediction": int(np.argmax(probability)),
                "probabilities": probability.tolist(),
                "fusion_decision": reason,
            }
        )
        decisions.append(
            {
                "sample_id": sample_id,
                "decision": reason,
                "b0_prediction": int(np.argmax(_probability_vector(left["probabilities"]))),
                "p0_prediction": int(np.argmax(_probability_vector(right["probabilities"]))),
                "p0_minus_b0_max_confidence": float(
                    np.max(_probability_vector(right["probabilities"]))
                    - np.max(_probability_vector(left["probabilities"]))
                ),
            }
        )
    a2_rows: list[dict[str, Any]] = []
    for row in a1_rows:
        probability = power_temperature(row["probabilities"])
        if int(np.argmax(probability)) != int(row["prediction"]):
            raise RuntimeError("A2 changed A1 hard prediction")
        a2_rows.append(
            {
                **row,
                "candidate_id": "A2",
                "prediction": int(np.argmax(probability)),
                "probabilities": probability.tolist(),
                "temperature": A2_TEMPERATURE,
            }
        )
    total = len(a1_rows)
    registry = {
        "schema_version": "opt_me_008_fusion_registry_v1",
        "sample_count": total,
        "equal_weight": EQUAL_WEIGHT,
        "p0_confidence_margin": P0_CONFIDENCE_MARGIN,
        "a2_temperature": A2_TEMPERATURE,
        "decision_counts": counts,
        "decision_rates": {key: value / total for key, value in counts.items()},
        "a1_input_b0_sha256": probability_rows_sha256(b0_rows),
        "a1_input_p0_sha256": probability_rows_sha256(p0_rows),
        "a1_output_sha256": probability_rows_sha256(a1_rows),
        "a2_output_sha256": probability_rows_sha256(a2_rows),
        "a2_hard_identity_with_a1": all(
            int(left["prediction"]) == int(right["prediction"])
            for left, right in zip(a1_rows, a2_rows, strict=True)
        ),
        "label_used_by_gate": False,
        "subject_id_used_by_gate": False,
        "decisions": decisions,
    }
    return {"A1": a1_rows, "A2": a2_rows, "registry": registry}


__all__ = [
    "A2_TEMPERATURE",
    "EQUAL_WEIGHT",
    "P0_CONFIDENCE_MARGIN",
    "derive_selective_candidates",
    "power_temperature",
    "probability_rows_sha256",
    "selective_gate",
]
