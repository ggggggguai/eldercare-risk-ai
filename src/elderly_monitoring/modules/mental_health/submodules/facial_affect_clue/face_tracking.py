from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import importlib.util
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np


LANDMARK_SCHEMA_VERSION = "dlib_68_v1"


@dataclass(frozen=True)
class LandmarkResult:
    points: np.ndarray
    source: str
    face_box: tuple[int, int, int, int]
    detection_source: str


class FaceLandmarker(Protocol):
    version: str

    def detect(self, image: np.ndarray) -> LandmarkResult:
        ...


def default_dlib_predictor_path(project_root: Path | None = None) -> Path:
    candidates: list[Path] = []
    if project_root is not None:
        candidates.append(
            project_root
            / "models/mental_health/facial_affect/assets/shape_predictor_68_face_landmarks.dat"
        )
    package = importlib.util.find_spec("face_recognition_models")
    if package and package.submodule_search_locations:
        candidates.append(
            Path(next(iter(package.submodule_search_locations)))
            / "models/shape_predictor_68_face_landmarks.dat"
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    searched = ", ".join(str(path) for path in candidates) or "no candidates"
    raise FileNotFoundError(f"No dlib 68-point predictor found; searched: {searched}")


def _ellipse_points(
    center: tuple[float, float],
    radii: tuple[float, float],
    angles: np.ndarray,
) -> np.ndarray:
    x = center[0] + radii[0] * np.cos(angles)
    y = center[1] + radii[1] * np.sin(angles)
    return np.column_stack((x, y)).astype(np.float32)


def canonical_landmarks_68(width: int, height: int) -> np.ndarray:
    """Return a deterministic 68-point template for an already cropped face."""
    jaw = _ellipse_points(
        (0.50, 0.46), (0.42, 0.50), np.linspace(np.pi, 0.0, 17)
    )
    left_brow = np.array(
        [[0.20, 0.30], [0.27, 0.27], [0.34, 0.26], [0.40, 0.28], [0.45, 0.31]],
        dtype=np.float32,
    )
    right_brow = left_brow.copy()
    right_brow[:, 0] = 1.0 - left_brow[::-1, 0]
    nose_bridge = np.array(
        [[0.50, 0.34], [0.50, 0.42], [0.50, 0.50], [0.50, 0.58]],
        dtype=np.float32,
    )
    nose_base = np.array(
        [[0.39, 0.61], [0.44, 0.64], [0.50, 0.65], [0.56, 0.64], [0.61, 0.61]],
        dtype=np.float32,
    )
    left_eye = _ellipse_points(
        (0.34, 0.39), (0.11, 0.045), np.array([np.pi, 4.0, 5.2, 0.0, 1.0, 2.2])
    )
    right_eye = left_eye.copy()
    right_eye[:, 0] = 1.0 - left_eye[:, 0]
    outer_lip = _ellipse_points(
        (0.50, 0.76),
        (0.20, 0.09),
        np.array([np.pi, 3.65, 4.2, 4.75, 5.3, 5.85, 0.0, 0.55, 1.1, 1.65, 2.2, 2.75]),
    )
    inner_lip = _ellipse_points(
        (0.50, 0.76),
        (0.11, 0.04),
        np.array([np.pi, 3.9, 4.7, 5.5, 0.0, 0.8, 1.6, 2.4]),
    )
    normalized = np.vstack(
        (
            jaw,
            left_brow,
            right_brow,
            nose_bridge,
            nose_base,
            left_eye,
            right_eye,
            outer_lip,
            inner_lip,
        )
    )
    if normalized.shape != (68, 2):
        raise RuntimeError(f"Invalid canonical landmark shape: {normalized.shape}")
    scale = np.array([max(width - 1, 1), max(height - 1, 1)], dtype=np.float32)
    return normalized * scale


class TemplateFaceLandmarker:
    """Engineering fallback that never claims measured facial landmarks."""

    version = "estimated_template_68_v1"

    def detect(self, image: np.ndarray) -> LandmarkResult:
        if image.ndim not in (2, 3) or image.size == 0:
            raise ValueError("Expected a non-empty grayscale or BGR image")
        height, width = image.shape[:2]
        return LandmarkResult(
            points=canonical_landmarks_68(width, height),
            source=self.version,
            face_box=(0, 0, width, height),
            detection_source="cropped_frame_contract",
        )


class DlibFaceLandmarker:
    """Dlib 68-point predictor with an explicit cropped-frame fallback box."""

    version = LANDMARK_SCHEMA_VERSION

    def __init__(
        self,
        predictor_path: Path,
        *,
        allow_cropped_frame_fallback: bool = True,
        detector_upsample: int = 0,
    ) -> None:
        predictor_path = predictor_path.resolve()
        if not predictor_path.is_file():
            raise FileNotFoundError(predictor_path)
        import dlib

        self._dlib = dlib
        self._detector = dlib.get_frontal_face_detector()
        self._predictor = dlib.shape_predictor(str(predictor_path))
        self.predictor_path = predictor_path
        digest = sha256()
        with predictor_path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        self.predictor_sha256 = digest.hexdigest()
        self.allow_cropped_frame_fallback = allow_cropped_frame_fallback
        self.detector_upsample = detector_upsample

    def detect(self, image: np.ndarray) -> LandmarkResult:
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        height, width = gray.shape[:2]
        faces = list(self._detector(gray, self.detector_upsample))
        if faces:
            face = max(faces, key=lambda rect: rect.width() * rect.height())
            detection_source = "dlib_hog"
        elif self.allow_cropped_frame_fallback:
            margin_x = max(1, int(round(width * 0.03)))
            margin_y = max(1, int(round(height * 0.03)))
            face = self._dlib.rectangle(
                margin_x,
                margin_y,
                max(margin_x, width - margin_x - 1),
                max(margin_y, height - margin_y - 1),
            )
            detection_source = "cropped_frame_fallback"
        else:
            raise RuntimeError("No face detected")
        shape = self._predictor(gray, face)
        points = np.array(
            [(shape.part(index).x, shape.part(index).y) for index in range(68)],
            dtype=np.float32,
        )
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        return LandmarkResult(
            points=points,
            source=self.version,
            face_box=(
                max(0, face.left()),
                max(0, face.top()),
                min(width - 1, face.right()),
                min(height - 1, face.bottom()),
            ),
            detection_source=detection_source,
        )
