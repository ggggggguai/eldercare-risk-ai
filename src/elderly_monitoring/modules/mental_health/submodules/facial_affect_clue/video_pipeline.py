from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import ipaddress
import json
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Callable, Mapping, Protocol, Sequence
from urllib.parse import urljoin, urlsplit

import cv2
import httpx
import numpy as np

from .causalnet_preprocess import (
    CausalNetPreprocessConfig,
    build_four_route_input_from_aligned_frames,
)
from .deployment import (
    B0_DEPLOYMENT_MODEL_VERSION,
    B0_ENSEMBLE_RULE,
    B0_EXPECTED_RUN_LOCK_SHA256,
    B0DeploymentError,
    routes_sha256,
)
from .paper_preprocess import optical_strain


FACIAL_AFFECT_VIDEO_REQUEST_SCHEMA = "facial_affect_video_request_v1"
FACIAL_AFFECT_VIDEO_RESPONSE_SCHEMA = "facial_affect_video_response_v1"
FACIAL_AFFECT_VIDEO_INFER_PATH = "/v1/mental-health/facial-affect/video-infer"
SPOTTING_VERSION = "s10_aligned_roi_motion_strain_v1"
MEDICAL_DISCLAIMER = "仅作微表情情绪变化线索，不构成医学诊断。"
TASK_INSUFFICIENT_ERROR_CODES = {"MEDIA_DECODE_FAILED", "NO_VIDEO_TRACK"}


class FacialAffectVideoError(RuntimeError):
    def __init__(self, message: str, *, code: str, status_code: int) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


@dataclass(frozen=True)
class VideoPipelineConfig:
    max_bytes: int = 128 * 1024 * 1024
    download_timeout_seconds: float = 30.0
    max_redirects: int = 2
    allowed_origins: tuple[str, ...] = ()
    temp_root: Path = Path(tempfile.gettempdir()) / "eldercare-facial-affect"
    temp_retention_seconds: int = 3600
    max_duration_seconds: float = 65.0
    max_frames: int = 3600
    max_decode_width: int = 1280
    max_decode_height: int = 720
    max_face_gap_frames: int = 2
    minimum_face_coverage: float = 0.55
    low_fps_warning_threshold: float = 10.0

    def __post_init__(self) -> None:
        if self.max_bytes < 1024 or self.download_timeout_seconds <= 0:
            raise ValueError("invalid facial-affect download limits")
        if self.max_redirects < 0 or self.max_redirects > 5:
            raise ValueError("max_redirects must be in [0,5]")
        if self.max_duration_seconds <= 0 or self.max_frames < 3:
            raise ValueError("invalid facial-affect decode limits")
        if not 0 <= self.minimum_face_coverage <= 1:
            raise ValueError("minimum_face_coverage must be in [0,1]")
        object.__setattr__(self, "temp_root", Path(self.temp_root))
        normalized: list[str] = []
        for value in self.allowed_origins:
            parsed = urlsplit(str(value).strip())
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                raise ValueError(f"invalid allowed origin: {value}")
            normalized.append(_origin(parsed))
        object.__setattr__(self, "allowed_origins", tuple(sorted(set(normalized))))


@dataclass(frozen=True)
class SpottingConfig:
    version: str = SPOTTING_VERSION
    analysis_size: int = 96
    smoothing_frames: int = 3
    minimum_duration_ms: int = 80
    maximum_duration_ms: int = 1000
    minimum_peak_score: float = 0.30
    minimum_peak_z: float = 2.5
    boundary_fraction: float = 0.30
    merge_distance_ms: int = 160
    maximum_head_translation_ratio: float = 0.08
    maximum_illumination_delta: float = 0.18

    def __post_init__(self) -> None:
        if self.version != SPOTTING_VERSION:
            raise ValueError("unsupported spotting version")
        if self.analysis_size < 32 or self.smoothing_frames < 1:
            raise ValueError("invalid spotting image/smoothing configuration")
        if not 0 < self.minimum_duration_ms < self.maximum_duration_ms:
            raise ValueError("invalid spotting duration range")
        if not 0 < self.boundary_fraction < 1:
            raise ValueError("boundary_fraction must be in (0,1)")

    def fingerprint(self) -> str:
        raw = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return sha256(raw.encode("utf-8")).hexdigest()


class FaceDetector(Protocol):
    version: str

    def detect(self, frame: np.ndarray) -> Sequence[tuple[int, int, int, int]]: ...


class OpenCVHaarFaceDetector:
    version = "opencv_haar_frontalface_alt2_v1"

    def __init__(self) -> None:
        path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_alt2.xml"
        self._classifier = cv2.CascadeClassifier(str(path))
        if self._classifier.empty():
            raise FacialAffectVideoError(
                "face detector asset is unavailable",
                code="FACE_DETECTOR_UNAVAILABLE",
                status_code=503,
            )

    def detect(self, frame: np.ndarray) -> Sequence[tuple[int, int, int, int]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        minimum = max(24, min(gray.shape[:2]) // 10)
        boxes = self._classifier.detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=5,
            minSize=(minimum, minimum),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        return [tuple(int(value) for value in box) for box in boxes]


def _origin(parsed) -> str:
    host = (parsed.hostname or "").lower()
    port = parsed.port
    default = 80 if parsed.scheme == "http" else 443
    return f"{parsed.scheme}://{host}" + (f":{port}" if port and port != default else "")


def validate_video_url(url: str, allowed_origins: Sequence[str]) -> str:
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise FacialAffectVideoError(
            "video source URL is invalid", code="INVALID_MEDIA_URL", status_code=422
        ) from exc
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise FacialAffectVideoError(
            "video source must use absolute HTTP or HTTPS",
            code="INVALID_MEDIA_URL",
            status_code=422,
        )
    if parsed.username is not None or parsed.password is not None:
        raise FacialAffectVideoError(
            "video source URL userinfo is forbidden",
            code="INVALID_MEDIA_URL",
            status_code=422,
        )
    if port is not None and not 1 <= port <= 65535:
        raise FacialAffectVideoError(
            "video source URL port is invalid",
            code="INVALID_MEDIA_URL",
            status_code=422,
        )
    origin = _origin(parsed)
    normalized = set(allowed_origins)
    if not normalized or origin not in normalized:
        try:
            address = ipaddress.ip_address(parsed.hostname)
            private_literal = (
                address.is_private
                or address.is_loopback
                or address.is_link_local
                or address.is_reserved
                or address.is_unspecified
            )
        except ValueError:
            private_literal = False
        code = "SSRF_BLOCKED" if private_literal else "ORIGIN_NOT_ALLOWED"
        raise FacialAffectVideoError(
            "video source origin is not allowed", code=code, status_code=422
        )
    return url


class SecureVideoDownloader:
    def __init__(
        self,
        config: VideoPipelineConfig,
        *,
        client_factory: Callable[..., Any] = httpx.Client,
    ) -> None:
        self.config = config
        self.client_factory = client_factory

    def download(self, url: str, destination: Path) -> dict[str, Any]:
        current = validate_video_url(url, self.config.allowed_origins)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.unlink(missing_ok=True)
        downloaded = 0
        content_type = ""
        try:
            with self.client_factory(
                timeout=self.config.download_timeout_seconds,
                follow_redirects=False,
            ) as client:
                for redirect_count in range(self.config.max_redirects + 1):
                    with client.stream(
                        "GET",
                        current,
                        headers={"Accept": "video/mp4,application/octet-stream"},
                    ) as response:
                        if response.status_code in {301, 302, 303, 307, 308}:
                            if redirect_count >= self.config.max_redirects:
                                raise FacialAffectVideoError(
                                    "video source redirected too many times",
                                    code="MEDIA_REDIRECT_LIMIT",
                                    status_code=422,
                                )
                            location = response.headers.get("location")
                            if not location:
                                raise FacialAffectVideoError(
                                    "video source redirect has no location",
                                    code="MEDIA_UNREADABLE",
                                    status_code=422,
                                )
                            current = validate_video_url(
                                urljoin(current, location), self.config.allowed_origins
                            )
                            continue
                        if response.status_code >= 400:
                            raise FacialAffectVideoError(
                                f"video source returned HTTP {response.status_code}",
                                code="MEDIA_SOURCE_HTTP_ERROR",
                                status_code=422,
                            )
                        content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
                        if content_type and content_type not in {
                            "video/mp4",
                            "application/mp4",
                            "application/octet-stream",
                        }:
                            raise FacialAffectVideoError(
                                "video source content type is not MP4",
                                code="UNSUPPORTED_MEDIA_TYPE",
                                status_code=422,
                            )
                        length = response.headers.get("content-length")
                        if length and int(length) > self.config.max_bytes:
                            raise FacialAffectVideoError(
                                "video source exceeds byte limit",
                                code="MEDIA_TOO_LARGE",
                                status_code=413,
                            )
                        with destination.open("wb") as output:
                            for chunk in response.iter_bytes(1024 * 1024):
                                downloaded += len(chunk)
                                if downloaded > self.config.max_bytes:
                                    raise FacialAffectVideoError(
                                        "video source exceeds byte limit",
                                        code="MEDIA_TOO_LARGE",
                                        status_code=413,
                                    )
                                output.write(chunk)
                        break
                else:  # pragma: no cover - loop always exits or raises
                    raise FacialAffectVideoError(
                        "video source could not be downloaded",
                        code="MEDIA_UNREADABLE",
                        status_code=422,
                    )
        except FacialAffectVideoError:
            destination.unlink(missing_ok=True)
            raise
        except httpx.TimeoutException as exc:
            destination.unlink(missing_ok=True)
            raise FacialAffectVideoError(
                "video source download timed out",
                code="MEDIA_DOWNLOAD_TIMEOUT",
                status_code=504,
            ) from exc
        except (httpx.HTTPError, OSError, ValueError) as exc:
            destination.unlink(missing_ok=True)
            raise FacialAffectVideoError(
                "video source could not be downloaded",
                code="MEDIA_UNREADABLE",
                status_code=422,
            ) from exc
        if downloaded < 12:
            destination.unlink(missing_ok=True)
            raise FacialAffectVideoError(
                "video source is empty", code="MEDIA_UNREADABLE", status_code=422
            )
        header = destination.read_bytes()[:64]
        if b"ftyp" not in header:
            destination.unlink(missing_ok=True)
            raise FacialAffectVideoError(
                "downloaded media is not an MP4 container",
                code="UNSUPPORTED_MEDIA_TYPE",
                status_code=422,
            )
        return {"bytes": downloaded, "content_type": content_type}


def _iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    left, top = max(ax, bx), max(ay, by)
    right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0, right - left) * max(0, bottom - top)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union else 0.0


def _select_track_box(
    boxes: Sequence[tuple[int, int, int, int]],
    previous: tuple[int, int, int, int] | None,
) -> tuple[tuple[int, int, int, int] | None, bool]:
    if not boxes:
        return None, False
    if previous is None:
        ordered = sorted(boxes, key=lambda box: box[2] * box[3], reverse=True)
        conflict = len(ordered) > 1 and ordered[1][2] * ordered[1][3] >= ordered[0][2] * ordered[0][3] * 0.75
        return ordered[0], conflict
    px, py, pw, ph = previous
    pcx, pcy = px + pw / 2, py + ph / 2
    scored: list[tuple[float, tuple[int, int, int, int]]] = []
    for box in boxes:
        x, y, w, h = box
        distance = np.hypot(x + w / 2 - pcx, y + h / 2 - pcy) / max(pw, ph, 1)
        area_ratio = min(w * h, pw * ph) / max(w * h, pw * ph, 1)
        score = 2.0 * _iou(previous, box) + 0.5 * area_ratio - 0.35 * distance
        scored.append((float(score), box))
    scored.sort(key=lambda item: item[0], reverse=True)
    conflict = len(scored) > 1 and scored[1][0] >= scored[0][0] - 0.20
    return scored[0][1], conflict


def _smooth_box(
    previous: tuple[int, int, int, int] | None,
    current: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    if previous is None:
        return current
    return tuple(
        int(round(0.65 * old + 0.35 * new))
        for old, new in zip(previous, current, strict=True)
    )


def _align_face(frame: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x, y, width, height = box
    side = int(round(max(width, height) * 1.35))
    center_x = x + width / 2
    center_y = y + height / 2 - 0.04 * side
    left = int(round(center_x - side / 2))
    top = int(round(center_y - side / 2))
    right, bottom = left + side, top + side
    pad_left, pad_top = max(0, -left), max(0, -top)
    pad_right = max(0, right - frame.shape[1])
    pad_bottom = max(0, bottom - frame.shape[0])
    if pad_left or pad_top or pad_right or pad_bottom:
        frame = cv2.copyMakeBorder(
            frame,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_REFLECT_101,
        )
        left += pad_left
        right += pad_left
        top += pad_top
        bottom += pad_top
    crop = frame[top:bottom, left:right]
    if crop.size == 0:
        raise ValueError("face crop is empty")
    return cv2.resize(crop, (256, 256), interpolation=cv2.INTER_AREA)


@dataclass
class DecodedVideo:
    frame_paths: list[Path | None]
    timestamps_ms: list[int]
    decoded_frame_count: int
    effective_fps: float
    container_duration_ms: int
    duration_ms: int
    invalid_frame_count: int
    rotation_degrees: int
    face_coverage: float
    track_continuity: float
    face_scale_mean: float
    face_scale_cv: float
    maximum_track_displacement_ratio: float
    multiple_face_conflicts: int
    interpolated_face_frames: int
    detector_version: str
    warnings: list[str]


class VideoFaceDecoder:
    def __init__(self, config: VideoPipelineConfig, detector: FaceDetector) -> None:
        self.config = config
        self.detector = detector

    def decode(self, video_path: Path, frame_dir: Path) -> DecodedVideo:
        frame_dir.mkdir(parents=True, exist_ok=True)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise FacialAffectVideoError(
                "MP4 cannot be decoded", code="MEDIA_DECODE_FAILED", status_code=422
            )
        nominal_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        nominal_fps = nominal_fps if np.isfinite(nominal_fps) and nominal_fps > 0 else 25.0
        reported_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        container_duration_ms = int(
            round(max(0, reported_frame_count - 1) * 1000.0 / nominal_fps)
        )
        rotation_degrees = 0
        if hasattr(cv2, "CAP_PROP_ORIENTATION_META"):
            rotation_degrees = int(
                round(float(capture.get(cv2.CAP_PROP_ORIENTATION_META) or 0.0))
            )
        if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
            capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
        frame_paths: list[Path | None] = []
        timestamps: list[int] = []
        previous_box: tuple[int, int, int, int] | None = None
        missing_run = 0
        detected_count = 0
        interpolated_count = 0
        conflicts = 0
        invalid_frames = 0
        consecutive_decode_failures = 0
        face_scales: list[float] = []
        track_displacements: list[float] = []
        longest_aligned_run = 0
        aligned_run = 0
        try:
            while len(frame_paths) < self.config.max_frames:
                ok, frame = capture.read()
                if not ok:
                    current_position = int(capture.get(cv2.CAP_PROP_POS_FRAMES) or 0)
                    if (
                        reported_frame_count > 0
                        and current_position < reported_frame_count - 1
                        and consecutive_decode_failures < 3
                    ):
                        invalid_frames += 1
                        consecutive_decode_failures += 1
                        continue
                    break
                consecutive_decode_failures = 0
                if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
                    invalid_frames += 1
                    continue
                height, width = frame.shape[:2]
                scale = min(
                    1.0,
                    self.config.max_decode_width / max(width, 1),
                    self.config.max_decode_height / max(height, 1),
                )
                if scale < 1.0:
                    frame = cv2.resize(
                        frame,
                        (max(1, int(width * scale)), max(1, int(height * scale))),
                        interpolation=cv2.INTER_AREA,
                    )
                raw_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
                fallback_ms = len(frame_paths) * 1000.0 / nominal_fps
                current_ms = int(round(raw_ms if raw_ms > 0 or not timestamps else fallback_ms))
                if timestamps and current_ms <= timestamps[-1]:
                    current_ms = int(round(timestamps[-1] + 1000.0 / nominal_fps))
                if current_ms > self.config.max_duration_seconds * 1000:
                    raise FacialAffectVideoError(
                        "video duration exceeds limit",
                        code="MEDIA_DURATION_LIMIT",
                        status_code=422,
                    )
                boxes = self.detector.detect(frame)
                selected, conflict = _select_track_box(boxes, previous_box)
                conflicts += int(conflict)
                interpolated = False
                if selected is None:
                    missing_run += 1
                    if previous_box is not None and missing_run <= self.config.max_face_gap_frames:
                        selected = previous_box
                        interpolated = True
                        interpolated_count += 1
                else:
                    missing_run = 0
                    detected_count += 1
                    face_scales.append(
                        float(selected[2] * selected[3] / max(frame.shape[0] * frame.shape[1], 1))
                    )
                    if previous_box is not None:
                        old_x, old_y, old_w, old_h = previous_box
                        new_x, new_y, new_w, new_h = selected
                        displacement = np.hypot(
                            new_x + new_w / 2 - (old_x + old_w / 2),
                            new_y + new_h / 2 - (old_y + old_h / 2),
                        ) / max(old_w, old_h, 1)
                        track_displacements.append(float(displacement))
                    selected = _smooth_box(previous_box, selected)
                path: Path | None = None
                if selected is not None:
                    try:
                        aligned = _align_face(frame, selected)
                        path = frame_dir / f"frame_{len(frame_paths):06d}.jpg"
                        if not cv2.imwrite(str(path), aligned, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                            raise OSError("could not write aligned face frame")
                        previous_box = selected
                        aligned_run += 1
                        longest_aligned_run = max(longest_aligned_run, aligned_run)
                    except (ValueError, OSError):
                        path = None
                        aligned_run = 0
                else:
                    aligned_run = 0
                frame_paths.append(path)
                timestamps.append(current_ms)
            if len(frame_paths) >= self.config.max_frames:
                has_more, _ = capture.read()
                if has_more:
                    raise FacialAffectVideoError(
                        "video frame count exceeds limit",
                        code="MEDIA_FRAME_LIMIT",
                        status_code=422,
                    )
        finally:
            capture.release()
        count = len(frame_paths)
        if count < 3:
            raise FacialAffectVideoError(
                "video has fewer than three decodable frames",
                code="NO_VIDEO_TRACK",
                status_code=422,
            )
        duration_ms = max(1, timestamps[-1] - timestamps[0])
        effective_fps = (count - 1) * 1000.0 / duration_ms
        face_coverage = detected_count / count
        continuity = longest_aligned_run / count
        warnings: list[str] = []
        if effective_fps < self.config.low_fps_warning_threshold:
            warnings.append("low_effective_fps")
        if face_coverage < 0.85:
            warnings.append("limited_face_coverage")
        if interpolated_count:
            warnings.append("short_face_gaps_interpolated")
        if conflicts:
            warnings.append("multiple_face_track_conflict")
        if invalid_frames:
            warnings.append("invalid_video_frames_skipped")
        if rotation_degrees:
            warnings.append("rotation_metadata_applied")
        face_scale_mean = float(np.mean(face_scales)) if face_scales else 0.0
        face_scale_cv = (
            float(np.std(face_scales) / max(face_scale_mean, 1e-9))
            if face_scales
            else 0.0
        )
        return DecodedVideo(
            frame_paths=frame_paths,
            timestamps_ms=[value - timestamps[0] for value in timestamps],
            decoded_frame_count=count,
            effective_fps=float(effective_fps),
            container_duration_ms=container_duration_ms,
            duration_ms=duration_ms,
            invalid_frame_count=invalid_frames,
            rotation_degrees=rotation_degrees,
            face_coverage=float(face_coverage),
            track_continuity=float(continuity),
            face_scale_mean=face_scale_mean,
            face_scale_cv=face_scale_cv,
            maximum_track_displacement_ratio=(
                max(track_displacements) if track_displacements else 0.0
            ),
            multiple_face_conflicts=conflicts,
            interpolated_face_frames=interpolated_count,
            detector_version=self.detector.version,
            warnings=warnings,
        )


@dataclass(frozen=True)
class SpotCandidate:
    onset_index: int
    apex_index: int
    offset_index: int
    onset_ms: int
    apex_ms: int
    offset_ms: int
    peak_score: float
    quality_score: float
    head_motion_ratio: float
    illumination_delta: float


class DeterministicMotionStrainSpotter:
    def __init__(self, config: SpottingConfig | None = None) -> None:
        self.config = config or SpottingConfig()

    def _gray(self, path: Path) -> tuple[np.ndarray, float]:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("aligned face frame cannot be read")
        image = cv2.resize(
            image,
            (self.config.analysis_size, self.config.analysis_size),
            interpolation=cv2.INTER_AREA,
        )
        # Farneback's frozen parameters are calibrated for 8-bit intensity.
        return cv2.equalizeHist(image), float(image.mean())

    def spot(
        self, decoded: DecodedVideo
    ) -> tuple[list[SpotCandidate], list[str], int]:
        if decoded.face_coverage < 0.01:
            return [], ["no_stable_face"], 0
        scores = np.full(decoded.decoded_frame_count, np.nan, dtype=np.float64)
        head = np.full_like(scores, np.nan)
        illumination = np.full_like(scores, np.nan)
        previous_gray: np.ndarray | None = None
        previous_luminance: float | None = None
        previous_index: int | None = None
        for index, path in enumerate(decoded.frame_paths):
            if path is None:
                previous_gray = None
                previous_luminance = None
                previous_index = None
                continue
            gray, luminance = self._gray(path)
            if previous_gray is not None and previous_index == index - 1:
                flow = cv2.calcOpticalFlowFarneback(
                    previous_gray,
                    gray,
                    None,
                    0.5,
                    3,
                    15,
                    3,
                    5,
                    1.2,
                    0,
                ).astype(np.float32)
                median_translation = np.median(flow.reshape(-1, 2), axis=0)
                phase_translation, phase_response = cv2.phaseCorrelate(
                    previous_gray.astype(np.float32), gray.astype(np.float32)
                )
                phase_translation = np.asarray(phase_translation, dtype=np.float32)
                translation = (
                    phase_translation
                    if phase_response >= 0.05
                    and np.isfinite(phase_translation).all()
                    and np.linalg.norm(phase_translation)
                    <= self.config.analysis_size / 2
                    else median_translation
                )
                residual = flow - translation
                magnitude = np.linalg.norm(residual, axis=2)
                strain = optical_strain(residual)
                margin = self.config.analysis_size // 10
                roi_mag = magnitude[margin:-margin, margin:-margin]
                roi_strain = strain[margin:-margin, margin:-margin]
                scores[index] = float(
                    0.65 * np.quantile(roi_mag, 0.90)
                    + 0.35 * np.quantile(roi_strain, 0.90)
                )
                head[index] = float(
                    np.linalg.norm(translation) / self.config.analysis_size
                )
                illumination[index] = float(
                    abs(luminance - previous_luminance) / 255.0
                )
            previous_gray = gray
            previous_luminance = luminance
            previous_index = index
        valid = np.isfinite(scores)
        if int(valid.sum()) < 3:
            return [], ["insufficient_consecutive_face_frames"], 0
        smooth = scores.copy()
        radius = self.config.smoothing_frames // 2
        for index in np.flatnonzero(valid):
            start, stop = max(0, index - radius), min(len(scores), index + radius + 1)
            window = scores[start:stop]
            if np.isfinite(window).any():
                smooth[index] = float(np.nanmean(window))
        baseline = float(np.nanmedian(smooth))
        mad = float(np.nanmedian(np.abs(smooth - baseline)))
        threshold = max(
            self.config.minimum_peak_score,
            baseline + self.config.minimum_peak_z * max(1.4826 * mad, 1e-6),
        )
        peak_indices: list[int] = []
        above = np.isfinite(smooth) & (smooth >= threshold)
        index = 1
        while index < len(smooth) - 1:
            if not above[index]:
                index += 1
                continue
            start = index
            while index + 1 < len(smooth) - 1 and above[index + 1]:
                index += 1
            stop = index
            peak_indices.append(
                min(range(start, stop + 1), key=lambda item: (-smooth[item], item))
            )
            index += 1
        peak_indices.sort(key=lambda index: (-smooth[index], index))
        kept: list[int] = []
        for index in peak_indices:
            if all(
                abs(decoded.timestamps_ms[index] - decoded.timestamps_ms[other])
                >= self.config.merge_distance_ms
                for other in kept
            ):
                kept.append(index)
        candidates: list[SpotCandidate] = []
        filtered: list[str] = []
        if float(np.nanmax(head)) > self.config.maximum_head_translation_ratio:
            filtered.append("head_motion_excessive")
        if (
            float(np.nanmax(illumination))
            > self.config.maximum_illumination_delta
        ):
            filtered.append("illumination_change_excessive")
        for apex in kept:
            boundary = baseline + self.config.boundary_fraction * (smooth[apex] - baseline)
            onset = apex - 1
            while onset > 0 and np.isfinite(smooth[onset - 1]) and smooth[onset - 1] > boundary:
                onset -= 1
            offset = apex + 1
            while offset < len(smooth) - 1 and np.isfinite(smooth[offset + 1]) and smooth[offset + 1] > boundary:
                offset += 1
            if not 0 <= onset < apex < offset < decoded.decoded_frame_count:
                filtered.append("invalid_keyframe_order")
                continue
            duration = decoded.timestamps_ms[offset] - decoded.timestamps_ms[onset]
            if not self.config.minimum_duration_ms <= duration <= self.config.maximum_duration_ms:
                filtered.append("duration_out_of_range")
                continue
            local_head = float(np.nanmax(head[onset + 1 : offset + 1]))
            local_illumination = float(
                np.nanmax(illumination[onset + 1 : offset + 1])
            )
            if local_head > self.config.maximum_head_translation_ratio:
                filtered.append("head_motion_excessive")
                continue
            if local_illumination > self.config.maximum_illumination_delta:
                filtered.append("illumination_change_excessive")
                continue
            significance = min(1.0, max(0.0, (smooth[apex] - threshold) / max(threshold, 1e-6)))
            coverage = sum(
                decoded.frame_paths[index] is not None
                for index in range(onset, offset + 1)
            ) / (offset - onset + 1)
            quality = (
                0.45
                + 0.35 * significance
                + 0.20 * coverage
                - 0.35 * min(1.0, local_head / self.config.maximum_head_translation_ratio)
                - 0.20 * min(1.0, local_illumination / self.config.maximum_illumination_delta)
            )
            candidates.append(
                SpotCandidate(
                    onset_index=onset,
                    apex_index=apex,
                    offset_index=offset,
                    onset_ms=decoded.timestamps_ms[onset],
                    apex_ms=decoded.timestamps_ms[apex],
                    offset_ms=decoded.timestamps_ms[offset],
                    peak_score=float(smooth[apex]),
                    quality_score=float(np.clip(quality, 0.0, 1.0)),
                    head_motion_ratio=local_head,
                    illumination_delta=local_illumination,
                )
            )
        candidates.sort(key=lambda item: (-item.quality_score, item.apex_index))
        return candidates, list(dict.fromkeys(filtered)), len(kept)


class B0RuntimeProtocol(Protocol):
    def verify_package(self) -> dict[str, Any]: ...
    def predict(self, routes: Any, *, input_sha256: str | None = None) -> dict[str, Any]: ...


class FacialAffectVideoRuntime:
    def __init__(
        self,
        *,
        b0_runtime: B0RuntimeProtocol,
        config: VideoPipelineConfig,
        detector: FaceDetector | None = None,
        spotter: DeterministicMotionStrainSpotter | None = None,
        downloader: SecureVideoDownloader | None = None,
        preprocess_config: CausalNetPreprocessConfig | None = None,
    ) -> None:
        self.b0_runtime = b0_runtime
        self.config = config
        self.detector = detector or OpenCVHaarFaceDetector()
        self.spotter = spotter or DeterministicMotionStrainSpotter()
        self.downloader = downloader or SecureVideoDownloader(config)
        self.preprocess_config = preprocess_config or CausalNetPreprocessConfig()
        self.decoder = VideoFaceDecoder(config, self.detector)

    def verify_package(self) -> dict[str, Any]:
        manifest = self.b0_runtime.verify_package()
        verify_probe = getattr(self.b0_runtime, "verify_deterministic_probe", None)
        if callable(verify_probe):
            verify_probe()
        return manifest

    def _request_directory(self, request_id: str) -> Path:
        self.config.temp_root.mkdir(parents=True, exist_ok=True)
        if self.config.temp_retention_seconds:
            cutoff = time.time() - self.config.temp_retention_seconds
            for child in self.config.temp_root.iterdir():
                if (
                    child.is_dir()
                    and child.parent.resolve() == self.config.temp_root.resolve()
                    and child.stat().st_mtime < cutoff
                ):
                    shutil.rmtree(child, ignore_errors=True)
        prefix = sha256(request_id.encode("utf-8")).hexdigest()[:24] + "-"
        return Path(tempfile.mkdtemp(prefix=prefix, dir=self.config.temp_root))

    def _insufficient_task(
        self,
        slot: int,
        *,
        warnings: Sequence[str],
        decoded: DecodedVideo | None = None,
        candidate_count: int = 0,
        timings: Mapping[str, int] | None = None,
    ) -> dict[str, Any]:
        return {
            "task_slot": slot,
            "status": "insufficient_input",
            "decoded_frame_count": decoded.decoded_frame_count if decoded else 0,
            "effective_fps": decoded.effective_fps if decoded else 0.0,
            "container_duration_ms": decoded.container_duration_ms if decoded else 0,
            "decoded_duration_ms": decoded.duration_ms if decoded else 0,
            "invalid_frame_count": decoded.invalid_frame_count if decoded else 0,
            "rotation_degrees": decoded.rotation_degrees if decoded else 0,
            "face_coverage": decoded.face_coverage if decoded else 0.0,
            "candidate_count": candidate_count,
            "track_continuity": decoded.track_continuity if decoded else 0.0,
            "face_scale_mean": decoded.face_scale_mean if decoded else 0.0,
            "face_scale_cv": decoded.face_scale_cv if decoded else 0.0,
            "maximum_track_displacement_ratio": (
                decoded.maximum_track_displacement_ratio if decoded else 0.0
            ),
            "multiple_face_conflicts": decoded.multiple_face_conflicts if decoded else 0,
            "interpolated_face_frames": decoded.interpolated_face_frames if decoded else 0,
            "detector_version": decoded.detector_version if decoded else self.detector.version,
            "candidate_duration_ms": None,
            "peak_score": None,
            "head_motion_ratio": None,
            "illumination_delta": None,
            "onset_ms": None,
            "apex_ms": None,
            "offset_ms": None,
            "onset_index": None,
            "apex_index": None,
            "offset_index": None,
            "prediction": None,
            "probabilities": None,
            "confidence": None,
            "quality_score": 0.0,
            "quality_status": "invalid",
            "warnings": list(dict.fromkeys(warnings)),
            "input_sha256": None,
            "preprocessing_config_hash": self.preprocess_config.fingerprint(),
            "spotting_config_hash": self.spotter.config.fingerprint(),
            "timings_ms": dict(timings or {}),
        }

    def _process_task(self, slot: int, source: str, request_dir: Path) -> dict[str, Any]:
        started = time.perf_counter()
        timings: dict[str, int] = {}
        decoded: DecodedVideo | None = None
        try:
            stage = time.perf_counter()
            video_path = request_dir / f"task_{slot}.mp4"
            self.downloader.download(source, video_path)
            timings["download"] = int((time.perf_counter() - stage) * 1000)
            stage = time.perf_counter()
            decoded = self.decoder.decode(video_path, request_dir / f"task_{slot}_frames")
            timings["decode_and_track"] = int((time.perf_counter() - stage) * 1000)
            if (
                decoded.face_coverage < self.config.minimum_face_coverage
                or decoded.track_continuity < self.config.minimum_face_coverage
            ):
                timings["total"] = int((time.perf_counter() - started) * 1000)
                return self._insufficient_task(
                    slot,
                    warnings=[*decoded.warnings, "no_stable_face"],
                    decoded=decoded,
                    timings=timings,
                )
            stage = time.perf_counter()
            candidates, filtered, detected_candidate_count = self.spotter.spot(decoded)
            timings["spotting"] = int((time.perf_counter() - stage) * 1000)
            if not candidates:
                timings["total"] = int((time.perf_counter() - started) * 1000)
                return self._insufficient_task(
                    slot,
                    warnings=[*decoded.warnings, *filtered, "no_valid_microexpression_candidate"],
                    decoded=decoded,
                    candidate_count=detected_candidate_count,
                    timings=timings,
                )
            candidate = candidates[0]
            paths = [
                decoded.frame_paths[candidate.onset_index],
                decoded.frame_paths[candidate.apex_index],
                decoded.frame_paths[candidate.offset_index],
            ]
            if any(path is None for path in paths):
                raise FacialAffectVideoError(
                    "selected key frame is unavailable",
                    code="PREPROCESSING_FAILED",
                    status_code=422,
                )
            key_frames = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in paths]
            if any(frame is None for frame in key_frames):
                raise FacialAffectVideoError(
                    "selected key frame cannot be decoded",
                    code="PREPROCESSING_FAILED",
                    status_code=422,
                )
            stage = time.perf_counter()
            routes, metadata = build_four_route_input_from_aligned_frames(
                key_frames[0], key_frames[1], key_frames[2], self.preprocess_config
            )
            if routes.shape != (4, 3, 28, 28) or routes.dtype != np.float32:
                raise FacialAffectVideoError(
                    "four-route tensor contract drift",
                    code="PREPROCESSING_FAILED",
                    status_code=500,
                )
            if not np.isfinite(routes).all():
                raise FacialAffectVideoError(
                    "four-route tensor contains NaN or Inf",
                    code="PREPROCESSING_FAILED",
                    status_code=500,
                )
            if (
                routes[:2, :2].min() < -1.000001
                or routes[:2, :2].max() > 1.000001
                or routes[:2, 2].min() < -0.000001
                or routes[:2, 2].max() > 1.000001
                or routes[2:].min() < -0.000001
                or routes[2:].max() > 1.000001
            ):
                raise FacialAffectVideoError(
                    "four-route tensor channel range drift",
                    code="PREPROCESSING_FAILED",
                    status_code=500,
                )
            input_hash = routes_sha256(routes)
            timings["preprocessing"] = int((time.perf_counter() - stage) * 1000)
            stage = time.perf_counter()
            prediction = self.b0_runtime.predict(routes, input_sha256=input_hash)
            timings["b0_inference"] = int((time.perf_counter() - stage) * 1000)
            timings["total"] = int((time.perf_counter() - started) * 1000)
            warnings = list(decoded.warnings)
            if filtered:
                warnings.append("additional_candidates_filtered")
            quality_status = "valid" if not warnings else "warning"
            return {
                "task_slot": slot,
                "status": "completed",
                "decoded_frame_count": decoded.decoded_frame_count,
                "effective_fps": decoded.effective_fps,
                "container_duration_ms": decoded.container_duration_ms,
                "decoded_duration_ms": decoded.duration_ms,
                "invalid_frame_count": decoded.invalid_frame_count,
                "rotation_degrees": decoded.rotation_degrees,
                "face_coverage": decoded.face_coverage,
                "candidate_count": detected_candidate_count,
                "track_continuity": decoded.track_continuity,
                "face_scale_mean": decoded.face_scale_mean,
                "face_scale_cv": decoded.face_scale_cv,
                "maximum_track_displacement_ratio": (
                    decoded.maximum_track_displacement_ratio
                ),
                "multiple_face_conflicts": decoded.multiple_face_conflicts,
                "interpolated_face_frames": decoded.interpolated_face_frames,
                "detector_version": decoded.detector_version,
                "candidate_duration_ms": candidate.offset_ms - candidate.onset_ms,
                "peak_score": candidate.peak_score,
                "head_motion_ratio": candidate.head_motion_ratio,
                "illumination_delta": candidate.illumination_delta,
                "onset_ms": candidate.onset_ms,
                "apex_ms": candidate.apex_ms,
                "offset_ms": candidate.offset_ms,
                "onset_index": candidate.onset_index,
                "apex_index": candidate.apex_index,
                "offset_index": candidate.offset_index,
                "prediction": prediction["prediction"],
                "probabilities": prediction["probabilities"],
                "confidence": prediction["confidence"],
                "quality_score": candidate.quality_score,
                "quality_status": quality_status,
                "warnings": warnings,
                "input_sha256": input_hash,
                "preprocessing_config_hash": metadata[
                    "preprocessing_config_hash"
                ],
                "spotting_config_hash": self.spotter.config.fingerprint(),
                "timings_ms": timings,
            }
        except B0DeploymentError:
            raise
        except FacialAffectVideoError as exc:
            if exc.code in TASK_INSUFFICIENT_ERROR_CODES or (
                exc.code == "PREPROCESSING_FAILED" and exc.status_code < 500
            ):
                timings["total"] = int((time.perf_counter() - started) * 1000)
                return self._insufficient_task(
                    slot,
                    warnings=[*(decoded.warnings if decoded else []), exc.code],
                    decoded=decoded,
                    timings=timings,
                )
            raise
        except Exception as exc:
            raise FacialAffectVideoError(
                "facial-affect video processing failed",
                code="INTERNAL_ERROR",
                status_code=500,
            ) from exc

    def infer(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        manifest = self.verify_package()
        request_id = str(payload["request_id"])
        request_dir = self._request_directory(request_id)
        try:
            results = [
                self._process_task(
                    int(item["task_slot"]), str(item["source"]), request_dir
                )
                for item in payload["task_videos"]
            ]
            valid = [item for item in results if item["status"] == "completed"]
            if len(valid) == 3:
                status = "completed"
            elif valid:
                status = "degraded"
            else:
                status = "insufficient_input"
            summary = None
            if valid:
                chosen = sorted(
                    valid,
                    key=lambda item: (-float(item["quality_score"]), item["task_slot"]),
                )[0]
                summary = {
                    "source_task_slot": chosen["task_slot"],
                    "prediction": chosen["prediction"],
                    "probabilities": chosen["probabilities"],
                    "confidence": chosen["confidence"],
                    "quality_score": chosen["quality_score"],
                }
            warnings = [
                f"task_{item['task_slot']}:{warning}"
                for item in results
                for warning in item["warnings"]
            ]
            warnings.append(
                "部署集成未在独立未见数据上验证，不能把开发集指标视为线上指标。"
            )
            return {
                "schema_version": FACIAL_AFFECT_VIDEO_RESPONSE_SCHEMA,
                "request_id": request_id,
                "person_id": str(payload["person_id"]),
                "session_id": str(payload["session_id"]),
                "observed_at": payload["observed_at"],
                "status": status,
                "model_version": B0_DEPLOYMENT_MODEL_VERSION,
                "task_results": results,
                "summary_result": summary,
                "model_lineage": {
                    "candidate_id": "B0",
                    "run_lock_sha256": B0_EXPECTED_RUN_LOCK_SHA256,
                    "package_sha256": manifest["manifest_sha256"],
                    "checkpoint_count": 20,
                    "ensemble_rule": B0_ENSEMBLE_RULE,
                    "spotting_version": self.spotter.config.version,
                    "spotting_config_hash": self.spotter.config.fingerprint(),
                    "evaluation_scope": "CASME II exposed development estimate",
                    "deployment_ensemble_independently_validated": False,
                },
                "diagnosis": False,
                "automatic_risk_level_mapping": False,
                "medical_disclaimer": MEDICAL_DISCLAIMER,
                "warnings": list(dict.fromkeys(warnings)),
            }
        finally:
            shutil.rmtree(request_dir, ignore_errors=True)


__all__ = [
    "FACIAL_AFFECT_VIDEO_INFER_PATH",
    "FACIAL_AFFECT_VIDEO_REQUEST_SCHEMA",
    "FACIAL_AFFECT_VIDEO_RESPONSE_SCHEMA",
    "DeterministicMotionStrainSpotter",
    "FacialAffectVideoError",
    "FacialAffectVideoRuntime",
    "OpenCVHaarFaceDetector",
    "SecureVideoDownloader",
    "SpotCandidate",
    "SpottingConfig",
    "VideoFaceDecoder",
    "VideoPipelineConfig",
    "validate_video_url",
]
