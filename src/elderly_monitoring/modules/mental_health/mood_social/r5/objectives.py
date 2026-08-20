"""Ordered dual-head and ordinal probability transforms for R5-003."""

from __future__ import annotations

import numpy as np


def project_independent_heads(
    probability_ge5: np.ndarray, probability_ge10: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Minimal midpoint projection onto 0 <= p10 <= p5 <= 1."""

    p5 = np.clip(np.asarray(probability_ge5, dtype=float), 0.0, 1.0)
    p10 = np.clip(np.asarray(probability_ge10, dtype=float), 0.0, 1.0)
    if p5.shape != p10.shape:
        raise ValueError("r5 dual-head probabilities must have equal shape")
    violation = p10 > p5
    midpoint = 0.5 * (p5[violation] + p10[violation])
    p5[violation] = midpoint
    p10[violation] = midpoint
    return p5, p10


def conditional_ordered_heads(
    probability_ge5: np.ndarray, probability_ge10_given_ge5: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compose P(>=10)=P(>=5)*P(>=10|>=5)."""

    p5 = np.clip(np.asarray(probability_ge5, dtype=float), 0.0, 1.0)
    conditional = np.clip(
        np.asarray(probability_ge10_given_ge5, dtype=float), 0.0, 1.0
    )
    if p5.shape != conditional.shape:
        raise ValueError("r5 conditional-head probabilities must have equal shape")
    return p5, p5 * conditional


def ordinal_heads(class_probability: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Derive >=5 and >=10 heads from five PHQ severity classes."""

    probability = np.asarray(class_probability, dtype=float)
    if probability.ndim != 2 or probability.shape[1] != 5:
        raise ValueError("r5 ordinal probability must have shape (n, 5)")
    if (probability < 0).any():
        raise ValueError("r5 ordinal probabilities cannot be negative")
    row_sum = probability.sum(axis=1)
    if not np.allclose(row_sum, 1.0, atol=1.0e-6):
        raise ValueError("r5 ordinal probabilities must sum to one")
    p5 = probability[:, 1:].sum(axis=1)
    p10 = probability[:, 2:].sum(axis=1)
    return p5, p10


def phq_severity_class(score: np.ndarray) -> np.ndarray:
    """Map PHQ-9 score to 0-4/5-9/10-14/15-19/20-27."""

    values = np.asarray(score, dtype=float)
    if ((values < 0) | (values > 27) | ~np.isfinite(values)).any():
        raise ValueError("PHQ score must be finite and within 0..27")
    return np.digitize(values, bins=[5, 10, 15, 20], right=False).astype(int)


__all__ = [
    "conditional_ordered_heads",
    "ordinal_heads",
    "phq_severity_class",
    "project_independent_heads",
]
