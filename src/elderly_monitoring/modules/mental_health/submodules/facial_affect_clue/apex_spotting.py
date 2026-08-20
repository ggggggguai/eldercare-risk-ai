from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Sequence

import numpy as np


APEX_SCHEMA_VERSION = "dc_rois_reimplementation_v1"
ROI_NAMES = ("left_eye_brow", "right_eye_brow", "mouth")
ROI_LANDMARKS = {
    "left_eye_brow": tuple(range(17, 22)) + tuple(range(36, 42)),
    "right_eye_brow": tuple(range(22, 27)) + tuple(range(42, 48)),
    "mouth": tuple(range(48, 68)),
}


@dataclass(frozen=True)
class DCRoIsConfig:
    lbp_points: int = 8
    lbp_radius: int = 1
    histogram_bins: int = 256
    roi_margin_ratio: float = 0.18
    minimum_roi_size: int = 8
    combine_method: str = "max"

    def __post_init__(self) -> None:
        if self.lbp_points != 8 or self.lbp_radius != 1:
            raise ValueError("The v1 implementation supports LBP(8, 1) only")
        if self.histogram_bins != 256:
            raise ValueError("LBP(8, 1) requires 256 histogram bins")
        if self.combine_method not in {"max", "mean"}:
            raise ValueError("combine_method must be 'max' or 'mean'")
        if self.minimum_roi_size < 3:
            raise ValueError("minimum_roi_size must be at least 3")

    def fingerprint(self) -> str:
        payload = {"schema_version": APEX_SCHEMA_VERSION, **asdict(self)}
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ApexSpottingResult:
    apex_index: int
    frame_scores: np.ndarray
    roi_scores: np.ndarray
    roi_boxes: np.ndarray
    search_trace: tuple[tuple[int, int, int], ...]


def basic_lbp_8_1(gray: np.ndarray) -> np.ndarray:
    """Compute the original 8-neighbour, radius-1 LBP code image."""
    if gray.ndim != 2 or gray.size == 0:
        raise ValueError("Expected a non-empty grayscale image")
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        raise ValueError("LBP requires an image of at least 3x3")
    center = gray[1:-1, 1:-1]
    neighbours = (
        gray[:-2, :-2],
        gray[:-2, 1:-1],
        gray[:-2, 2:],
        gray[1:-1, 2:],
        gray[2:, 2:],
        gray[2:, 1:-1],
        gray[2:, :-2],
        gray[1:-1, :-2],
    )
    codes = np.zeros(center.shape, dtype=np.uint8)
    for bit, neighbour in enumerate(neighbours):
        codes |= ((neighbour >= center).astype(np.uint8) << bit)
    return codes


def lbp_histogram(gray: np.ndarray, bins: int = 256) -> np.ndarray:
    codes = basic_lbp_8_1(gray)
    histogram = np.bincount(codes.ravel(), minlength=bins).astype(np.float32)
    norm = float(np.linalg.norm(histogram))
    if norm > np.finfo(np.float32).eps:
        histogram /= norm
    return histogram


def cosine_histogram_distance(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float32).reshape(-1)
    second = np.asarray(second, dtype=np.float32).reshape(-1)
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= np.finfo(np.float32).eps:
        return 0.0 if np.array_equal(first, second) else 1.0
    similarity = float(np.dot(first, second) / denominator)
    return float(np.clip(1.0 - similarity, 0.0, 2.0))


def build_roi_boxes(
    landmarks: np.ndarray,
    image_shape: Sequence[int],
    config: DCRoIsConfig,
) -> np.ndarray:
    landmarks = np.asarray(landmarks, dtype=np.float32)
    if landmarks.shape != (68, 2):
        raise ValueError(f"Expected 68x2 landmarks, got {landmarks.shape}")
    height, width = int(image_shape[0]), int(image_shape[1])
    boxes: list[tuple[int, int, int, int]] = []
    for name in ROI_NAMES:
        points = landmarks[list(ROI_LANDMARKS[name])]
        x0, y0 = points.min(axis=0)
        x1, y1 = points.max(axis=0)
        margin_x = max(2, int(round((x1 - x0 + 1) * config.roi_margin_ratio)))
        margin_y = max(2, int(round((y1 - y0 + 1) * config.roi_margin_ratio)))
        left = max(0, int(np.floor(x0)) - margin_x)
        top = max(0, int(np.floor(y0)) - margin_y)
        right = min(width, int(np.ceil(x1)) + margin_x + 1)
        bottom = min(height, int(np.ceil(y1)) + margin_y + 1)
        if right - left < config.minimum_roi_size or bottom - top < config.minimum_roi_size:
            raise ValueError(f"ROI {name} is too small: {(left, top, right, bottom)}")
        boxes.append((left, top, right, bottom))
    return np.asarray(boxes, dtype=np.int32)


def divide_and_conquer_local_peak(
    scores: Sequence[float],
    *,
    minimum_index: int = 1,
    maximum_index: int | None = None,
) -> tuple[int, tuple[tuple[int, int, int], ...]]:
    """Select the highest peak by comparing candidates from divided segments."""
    values = np.asarray(scores, dtype=np.float32).reshape(-1)
    if values.size < 2:
        raise ValueError("At least two scores are required")
    if not np.all(np.isfinite(values)):
        raise ValueError("Scores must be finite")
    low = int(np.clip(minimum_index, 0, values.size - 1))
    high = int(
        np.clip(
            values.size - 1 if maximum_index is None else maximum_index,
            low,
            values.size - 1,
        )
    )
    trace: list[tuple[int, int, int]] = []

    def search(start: int, stop: int) -> int:
        if start == stop:
            return start
        middle = (start + stop) // 2
        trace.append((start, stop, middle))
        left_peak = search(start, middle)
        right_peak = search(middle + 1, stop)
        return left_peak if values[left_peak] >= values[right_peak] else right_peak

    return search(low, high), tuple(trace)


def spot_apex_dc_rois(
    gray_frames: Sequence[np.ndarray],
    onset_landmarks: np.ndarray,
    config: DCRoIsConfig | None = None,
) -> ApexSpottingResult:
    """Reimplement D&C-RoIs for an isolated onset-to-offset face sequence.

    The public method description specifies LBP features from left eye/brow,
    right eye/brow and mouth RoIs, onset-to-frame histogram correlation
    differences, and a divide-and-conquer local-maximum search. The original
    author code was not available, so this implementation versions every
    otherwise underspecified choice.
    """
    config = config or DCRoIsConfig()
    if len(gray_frames) < 3:
        raise ValueError("D&C-RoIs requires onset, apex candidate and offset frames")
    first_shape = gray_frames[0].shape
    if any(frame.ndim != 2 or frame.shape != first_shape for frame in gray_frames):
        raise ValueError("All D&C-RoIs frames must be same-size grayscale images")

    boxes = build_roi_boxes(onset_landmarks, first_shape, config)
    onset_histograms: list[np.ndarray] = []
    for left, top, right, bottom in boxes:
        onset_histograms.append(
            lbp_histogram(gray_frames[0][top:bottom, left:right], config.histogram_bins)
        )

    roi_scores = np.zeros((len(gray_frames), len(ROI_NAMES)), dtype=np.float32)
    for frame_index, frame in enumerate(gray_frames[1:], start=1):
        for roi_index, (left, top, right, bottom) in enumerate(boxes):
            histogram = lbp_histogram(
                frame[top:bottom, left:right], config.histogram_bins
            )
            roi_scores[frame_index, roi_index] = cosine_histogram_distance(
                onset_histograms[roi_index], histogram
            )
    if config.combine_method == "max":
        frame_scores = roi_scores.max(axis=1)
    else:
        frame_scores = roi_scores.mean(axis=1)
    apex_index, trace = divide_and_conquer_local_peak(
        frame_scores,
        minimum_index=1,
        maximum_index=len(gray_frames) - 2,
    )
    return ApexSpottingResult(
        apex_index=apex_index,
        frame_scores=frame_scores.astype(np.float32),
        roi_scores=roi_scores,
        roi_boxes=boxes,
        search_trace=trace,
    )
