"""Deterministic curvature, reversal, winding, and revisit diagnostics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


class TopologyError(ValueError):
    """Invalid input or parameters for the frozen topology diagnostics."""


@dataclass(frozen=True)
class TurnGeometry:
    """Per-point displacement directions and wrapped direction changes."""

    headings: np.ndarray
    valid_displacement: np.ndarray
    turn_angles: np.ndarray
    valid_turn: np.ndarray
    abs_curvature: np.ndarray


def wrap_pi(angle: float | np.ndarray) -> float | np.ndarray:
    """Wrap radians into the frozen half-open interval [-pi, pi)."""

    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def compute_turn_geometry(
    points: np.ndarray,
    point_mask: np.ndarray,
    *,
    step_epsilon: float = 1e-8,
) -> TurnGeometry:
    """Compute heading and the section-6.3 dimensionless curvature proxy."""

    array, mask = _validated_points_and_mask(points, point_mask)
    if isinstance(step_epsilon, bool) or float(step_epsilon) <= 0.0:
        raise TopologyError("step_epsilon must be positive")
    epsilon = float(step_epsilon)
    count = len(array)
    headings = np.zeros(count, dtype=np.float64)
    valid_displacement = np.zeros(count, dtype=bool)
    turn_angles = np.zeros(count, dtype=np.float64)
    valid_turn = np.zeros(count, dtype=bool)

    displacements = np.zeros_like(array)
    if count > 1:
        displacements[1:] = array[1:] - array[:-1]
    lengths = np.linalg.norm(displacements, axis=1)
    for index in range(1, count):
        if mask[index - 1] and mask[index] and lengths[index] > epsilon:
            valid_displacement[index] = True
            headings[index] = math.atan2(
                float(displacements[index, 1]),
                float(displacements[index, 0]),
            )
    for index in range(2, count):
        if valid_displacement[index - 1] and valid_displacement[index]:
            valid_turn[index] = True
            turn_angles[index] = float(
                wrap_pi(headings[index] - headings[index - 1])
            )
    return TurnGeometry(
        headings=headings,
        valid_displacement=valid_displacement,
        turn_angles=turn_angles,
        valid_turn=valid_turn,
        abs_curvature=np.abs(turn_angles),
    )


def compute_topology(
    points: np.ndarray,
    point_mask: np.ndarray,
    *,
    step_epsilon: float = 1e-8,
    reversal_direction_span: int = 2,
    reversal_angle_deg: float = 120.0,
    reversal_merge_gap: int = 2,
    revisit_index_gap: int = 8,
    revisit_radius: float = 0.10,
    max_revisit_links: int = 5,
) -> dict[str, Any]:
    """Return the exact fixed step-4 topology diagnostics on shape points."""

    array, mask = _validated_points_and_mask(points, point_mask)
    _validate_parameters(
        reversal_direction_span=reversal_direction_span,
        reversal_angle_deg=reversal_angle_deg,
        reversal_merge_gap=reversal_merge_gap,
        revisit_index_gap=revisit_index_gap,
        revisit_radius=revisit_radius,
        max_revisit_links=max_revisit_links,
    )
    geometry = compute_turn_geometry(array, mask, step_epsilon=step_epsilon)
    candidates: list[tuple[int, float]] = []
    span = reversal_direction_span
    epsilon = float(step_epsilon)
    for index in range(span, len(array) - span):
        involved = (index - span, index, index + span)
        if not all(mask[item] for item in involved):
            continue
        incoming = array[index] - array[index - span]
        outgoing = array[index + span] - array[index]
        incoming_norm = float(np.linalg.norm(incoming))
        outgoing_norm = float(np.linalg.norm(outgoing))
        if incoming_norm <= epsilon or outgoing_norm <= epsilon:
            continue
        cosine = float(np.dot(incoming, outgoing) / (incoming_norm * outgoing_norm))
        angle_deg = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
        if angle_deg >= float(reversal_angle_deg):
            candidates.append((index, angle_deg))

    reversal_events: list[dict[str, float | int]] = []
    if candidates:
        clusters: list[list[tuple[int, float]]] = [[candidates[0]]]
        for candidate in candidates[1:]:
            if candidate[0] - clusters[-1][-1][0] <= reversal_merge_gap:
                clusters[-1].append(candidate)
            else:
                clusters.append([candidate])
        for cluster in clusters:
            representative = min(cluster, key=lambda item: (-item[1], item[0]))
            reversal_events.append(
                {"index": representative[0], "angle_deg": representative[1]}
            )

    revisit_pairs: list[tuple[float, int, int]] = []
    for left in range(len(array)):
        if not mask[left]:
            continue
        for right in range(left + revisit_index_gap, len(array)):
            if not mask[right]:
                continue
            distance = float(np.linalg.norm(array[right] - array[left]))
            if distance < float(revisit_radius):
                revisit_pairs.append((distance, left, right))
    revisit_pairs.sort(key=lambda item: (item[0], item[1], item[2]))
    selected_links: list[dict[str, float | int]] = []
    used_endpoints: set[int] = set()
    for distance, left, right in revisit_pairs:
        if left in used_endpoints or right in used_endpoints:
            continue
        selected_links.append({"distance": distance, "i": left, "j": right})
        used_endpoints.update((left, right))
        if len(selected_links) >= max_revisit_links:
            break

    absolute_winding = abs(float(geometry.turn_angles.sum())) / (2.0 * math.pi)
    return {
        "abs_curvature": geometry.abs_curvature.tolist(),
        "max_abs_curvature": float(geometry.abs_curvature.max(initial=0.0)),
        "reversal_events": reversal_events,
        "absolute_winding": absolute_winding,
        "revisit_pair_count": len(revisit_pairs),
        "selected_revisit_links": selected_links,
    }


def _validated_points_and_mask(
    points: np.ndarray,
    point_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        array = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TopologyError("points must be finite numeric XY pairs") from exc
    mask = np.asarray(point_mask)
    if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] != 2:
        raise TopologyError("points must have shape [N,2] with N >= 2")
    if not np.isfinite(array).all():
        raise TopologyError("points must be finite")
    if mask.ndim != 1 or len(mask) != len(array):
        raise TopologyError("point_mask must have shape [N]")
    if not np.all(np.isin(mask, [0, 1])):
        raise TopologyError("point_mask values must be 0 or 1")
    return array, mask.astype(bool, copy=False)


def _validate_parameters(
    *,
    reversal_direction_span: int,
    reversal_angle_deg: float,
    reversal_merge_gap: int,
    revisit_index_gap: int,
    revisit_radius: float,
    max_revisit_links: int,
) -> None:
    integer_values = {
        "reversal_direction_span": reversal_direction_span,
        "reversal_merge_gap": reversal_merge_gap,
        "revisit_index_gap": revisit_index_gap,
        "max_revisit_links": max_revisit_links,
    }
    for name, value in integer_values.items():
        minimum = 0 if name == "reversal_merge_gap" else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise TopologyError(f"{name} must be an integer >= {minimum}")
    if (
        isinstance(reversal_angle_deg, bool)
        or not 0.0 <= float(reversal_angle_deg) <= 180.0
    ):
        raise TopologyError("reversal_angle_deg must be within [0,180]")
    if isinstance(revisit_radius, bool) or float(revisit_radius) <= 0.0:
        raise TopologyError("revisit_radius must be positive")
