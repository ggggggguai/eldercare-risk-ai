"""Anchorless public-trajectory shape normalization for wandering step 4.

This module deliberately implements only a robust center and one isotropic
scale.  Camera bbox-height compensation belongs to step 7 and must happen
before calling this shared shape core.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class AnchorlessNormalizationError(ValueError):
    """A trajectory cannot be safely represented in the shared shape space."""

    def __init__(self, message: str, *, reason_code: str = "insufficient_motion") -> None:
        super().__init__(message)
        self.reason_code = reason_code


@dataclass(frozen=True)
class AnchorlessNormalizationResult:
    """The normalized points plus the exact center and isotropic scale used."""

    points: np.ndarray
    center: np.ndarray
    scale: float


def robust_isotropic_normalize(
    points: np.ndarray,
    *,
    quantile_low: float = 0.05,
    quantile_high: float = 0.95,
    quantile_method: str = "linear",
    scale_epsilon: float = 1e-8,
) -> AnchorlessNormalizationResult:
    """Apply the frozen median/Q05-Q95 isotropic normalization formula.

    All arithmetic is float64.  No per-axis stretching, source-canvas
    inference, rotation, bbox-height correction, or epsilon-based amplification
    is performed here.
    """

    try:
        array = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise AnchorlessNormalizationError("points must be finite numeric XY pairs") from exc
    if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] != 2:
        raise AnchorlessNormalizationError("points must have shape [N,2] with N >= 2")
    if not np.isfinite(array).all():
        raise AnchorlessNormalizationError("points must be finite")
    if (
        isinstance(quantile_low, bool)
        or isinstance(quantile_high, bool)
        or not 0.0 <= float(quantile_low) < float(quantile_high) <= 1.0
    ):
        raise AnchorlessNormalizationError("quantiles must satisfy 0 <= low < high <= 1")
    if quantile_method != "linear":
        raise AnchorlessNormalizationError("quantile_method must remain 'linear'")
    if isinstance(scale_epsilon, bool) or float(scale_epsilon) <= 0.0:
        raise AnchorlessNormalizationError("scale_epsilon must be positive")

    center = np.median(array, axis=0)
    quantiles = np.quantile(
        array,
        [float(quantile_low), float(quantile_high)],
        axis=0,
        method=quantile_method,
    )
    spans = quantiles[1] - quantiles[0]
    scale = float(np.sqrt(float(np.square(spans).sum())))
    if not np.isfinite(scale) or scale <= float(scale_epsilon):
        raise AnchorlessNormalizationError(
            "trajectory isotropic scale is degenerate",
            reason_code="insufficient_motion",
        )
    normalized = (array - center) / scale
    if not np.isfinite(normalized).all():
        raise AnchorlessNormalizationError("normalized points are not finite")
    return AnchorlessNormalizationResult(
        points=normalized,
        center=center,
        scale=scale,
    )
