from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path

import cv2
import numpy as np


FLOW_SCHEMA_VERSION = "farneback_uvo_tanh_v1"


@dataclass(frozen=True)
class OpticalFlowConfig:
    aligned_width: int = 128
    aligned_height: int = 128
    flow_width: int = 32
    flow_height: int = 32
    normalization_scale: float = 2.0
    pyr_scale: float = 0.5
    levels: int = 3
    winsize: int = 15
    iterations: int = 3
    poly_n: int = 5
    poly_sigma: float = 1.2
    flags: int = 0
    ecc_iterations: int = 50
    ecc_epsilon: float = 1e-5

    def fingerprint(self) -> str:
        payload = {
            "schema_version": FLOW_SCHEMA_VERSION,
            **asdict(self),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return sha256(encoded).hexdigest()


@dataclass(frozen=True)
class AlignmentResult:
    image: np.ndarray
    transform: np.ndarray
    converged: bool
    correlation: float | None


def read_image(path: Path) -> np.ndarray:
    """Read an image through bytes so non-ASCII Windows paths remain reliable."""
    if not path.is_file():
        raise FileNotFoundError(path)
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to decode image: {path}")
    return image


def resize_bgr(image: np.ndarray, config: OpticalFlowConfig) -> np.ndarray:
    return cv2.resize(
        image,
        (config.aligned_width, config.aligned_height),
        interpolation=cv2.INTER_AREA,
    )


def to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def align_to_reference(
    reference_gray: np.ndarray,
    image: np.ndarray,
    config: OpticalFlowConfig,
) -> AlignmentResult:
    gray = to_gray(image)
    transform = np.eye(2, 3, dtype=np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        config.ecc_iterations,
        config.ecc_epsilon,
    )
    try:
        correlation, transform = cv2.findTransformECC(
            reference_gray.astype(np.float32) / 255.0,
            gray.astype(np.float32) / 255.0,
            transform,
            cv2.MOTION_EUCLIDEAN,
            criteria,
            None,
            1,
        )
        aligned = cv2.warpAffine(
            image,
            transform,
            (reference_gray.shape[1], reference_gray.shape[0]),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        return AlignmentResult(aligned, transform, True, float(correlation))
    except cv2.error:
        return AlignmentResult(image.copy(), transform, False, None)


def dense_farneback(
    reference_gray: np.ndarray,
    target_gray: np.ndarray,
    config: OpticalFlowConfig,
) -> np.ndarray:
    return cv2.calcOpticalFlowFarneback(
        reference_gray,
        target_gray,
        None,
        config.pyr_scale,
        config.levels,
        config.winsize,
        config.iterations,
        config.poly_n,
        config.poly_sigma,
        config.flags,
    ).astype(np.float32)


def motion_energy(flow: np.ndarray) -> float:
    magnitude = np.linalg.norm(flow, axis=2)
    height, width = magnitude.shape
    y0, y1 = int(height * 0.10), max(int(height * 0.90), 1)
    x0, x1 = int(width * 0.10), max(int(width * 0.90), 1)
    return float(np.mean(magnitude[y0:y1, x0:x1]))


def normalize_three_channel_flow(
    flow: np.ndarray,
    config: OpticalFlowConfig,
) -> np.ndarray:
    original_height, original_width = flow.shape[:2]
    resized = cv2.resize(
        flow,
        (config.flow_width, config.flow_height),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32)
    resized[..., 0] *= config.flow_width / original_width
    resized[..., 1] *= config.flow_height / original_height
    magnitude = np.linalg.norm(resized, axis=2)
    scale = max(float(config.normalization_scale), np.finfo(np.float32).eps)
    channels = np.stack(
        (
            np.tanh(resized[..., 0] / scale),
            np.tanh(resized[..., 1] / scale),
            np.tanh(magnitude / scale),
        ),
        axis=2,
    )
    return channels.astype(np.float32)
