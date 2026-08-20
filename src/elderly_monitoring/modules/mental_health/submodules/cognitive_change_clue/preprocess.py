"""Frozen V3.3 audio, ASR text, and CogPic face preprocessing."""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


TARGET_SAMPLE_RATE = 16_000
AUDIO_WINDOW_MS = 20_000
AUDIO_HOP_MS = 10_000
FACE_SAMPLE_COUNT = 16
FACE_SIZE = 224
FACE_BOX_EXPANSION = 0.20
FACE_MIN_DETECTION_CONFIDENCE = 0.60
IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


@dataclass(frozen=True)
class AudioWindow:
    waveform: np.ndarray
    start_ms: int
    valid_samples: int
    weight_ms: int


@dataclass(frozen=True)
class FacePreprocessResult:
    tensors: tuple[np.ndarray, ...]
    sampled_positions: tuple[int, ...]
    valid_positions: tuple[int, ...]
    source_paths: tuple[str, ...]
    mean_luma: float

    @property
    def valid_count(self) -> int:
        return len(self.tensors)


def normalize_cognitive_text(text: str) -> tuple[str, int]:
    """Apply V3.3 normalization without translating ASCII punctuation."""

    normalized = unicodedata.normalize("NFKC", str(text))
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = re.sub(r"[0-9]+", "<NUM>", normalized)
    kept: list[str] = []
    char_count = 0
    index = 0
    while index < len(normalized):
        if normalized.startswith("<NUM>", index):
            kept.append("<NUM>")
            char_count += 1
            index += 5
            continue
        character = normalized[index]
        if _is_chinese(character) or (character.isascii() and character.isalpha()):
            kept.append(character)
            char_count += 1
        elif character == " " or character in "，。！？；：":
            kept.append(character)
        index += 1
    return re.sub(r" +", " ", "".join(kept)).strip(), char_count


def audio_window_starts(duration_ms: int) -> tuple[int, ...]:
    duration_ms = int(duration_ms)
    if duration_ms <= 0:
        raise ValueError("duration_ms must be positive")
    if duration_ms <= AUDIO_WINDOW_MS:
        return (0,)
    maximum_regular = math.floor(
        (duration_ms - AUDIO_WINDOW_MS) / AUDIO_HOP_MS
    ) * AUDIO_HOP_MS
    starts = list(range(0, maximum_regular + 1, AUDIO_HOP_MS))
    last_start = math.ceil(
        (duration_ms - AUDIO_WINDOW_MS) / AUDIO_HOP_MS
    ) * AUDIO_HOP_MS
    if last_start not in starts:
        starts.append(last_start)
    return tuple(starts)


def make_audio_windows(
    samples: np.ndarray | Sequence[float],
    *,
    sample_rate: int = TARGET_SAMPLE_RATE,
    duration_ms: int | None = None,
) -> tuple[AudioWindow, ...]:
    if sample_rate != TARGET_SAMPLE_RATE:
        raise ValueError(f"sample_rate must be {TARGET_SAMPLE_RATE}")
    waveform = np.asarray(samples, dtype=np.float32).reshape(-1)
    if waveform.size == 0:
        raise ValueError("audio samples must not be empty")
    effective_duration_ms = (
        int(round(waveform.size * 1000.0 / sample_rate))
        if duration_ms is None
        else int(duration_ms)
    )
    starts = audio_window_starts(effective_duration_ms)
    window_samples = AUDIO_WINDOW_MS * sample_rate // 1000
    windows: list[AudioWindow] = []
    for index, start_ms in enumerate(starts):
        start_sample = start_ms * sample_rate // 1000
        available = max(0, min(window_samples, waveform.size - start_sample))
        padded = np.zeros(window_samples, dtype=np.float32)
        if available:
            padded[:available] = waveform[start_sample : start_sample + available]
        next_start = starts[index + 1] if index + 1 < len(starts) else effective_duration_ms
        weight_ms = next_start - start_ms
        if available <= 0 or weight_ms <= 0:
            raise ValueError("audio duration and sample count are inconsistent")
        windows.append(
            AudioWindow(
                waveform=padded,
                start_ms=start_ms,
                valid_samples=available,
                weight_ms=weight_ms,
            )
        )
    return tuple(windows)


def sample_frame_positions(frame_count: int, sample_count: int = FACE_SAMPLE_COUNT) -> tuple[int, ...]:
    frame_count = int(frame_count)
    sample_count = int(sample_count)
    if frame_count <= 0:
        return ()
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if sample_count == 1:
        return (0,)
    return tuple(
        math.floor(index * (frame_count - 1) / (sample_count - 1))
        for index in range(sample_count)
    )


def expand_relative_box(
    *,
    x_min: float,
    y_min: float,
    width: float,
    height: float,
    image_width: int,
    image_height: int,
    expansion: float = FACE_BOX_EXPANSION,
) -> tuple[int, int, int, int] | None:
    if image_width <= 0 or image_height <= 0 or width <= 0 or height <= 0:
        return None
    expand_x = width * expansion
    expand_y = height * expansion
    left = math.floor((x_min - expand_x) * image_width)
    top = math.floor((y_min - expand_y) * image_height)
    right = math.ceil((x_min + width + expand_x) * image_width)
    bottom = math.ceil((y_min + height + expand_y) * image_height)
    left = max(0, min(left, image_width))
    top = max(0, min(top, image_height))
    right = max(0, min(right, image_width))
    bottom = max(0, min(bottom, image_height))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def preprocess_face_frames(
    frame_paths: Iterable[str | Path],
    detector: Any,
) -> FacePreprocessResult:
    """Sample a frame sequence and return normalized CHW crops for single faces."""

    import cv2

    ordered_paths = sort_frame_paths(frame_paths)
    positions = sample_frame_positions(len(ordered_paths))
    tensors: list[np.ndarray] = []
    valid_positions: list[int] = []
    source_paths: list[str] = []
    lumas: list[float] = []
    for position in positions:
        path = ordered_paths[position]
        # cv2.imread uses the Windows ANSI path API; CogPic lives under a
        # Chinese-named directory, so decode the file bytes explicitly.
        encoded = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR) if encoded.size else None
        if image is None or image.size == 0:
            continue
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = detector.process(rgb)
        detections = list(getattr(result, "detections", None) or ())
        if len(detections) != 1:
            continue
        bounds = _detection_bounds(detections[0], image.shape[1], image.shape[0])
        if bounds is None:
            continue
        left, top, right, bottom = bounds
        crop_bgr = image[top:bottom, left:right]
        if crop_bgr.size == 0:
            continue
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        lumas.append(float(np.mean(gray, dtype=np.float64) / 255.0))
        resized_bgr = cv2.resize(crop_bgr, (FACE_SIZE, FACE_SIZE), interpolation=cv2.INTER_LINEAR)
        resized_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalized = (resized_rgb - IMAGENET_MEAN) / IMAGENET_STD
        tensors.append(np.transpose(normalized, (2, 0, 1)).astype(np.float32, copy=False))
        valid_positions.append(position)
        source_paths.append(str(path))
    return FacePreprocessResult(
        tensors=tuple(tensors),
        sampled_positions=positions,
        valid_positions=tuple(valid_positions),
        source_paths=tuple(source_paths),
        mean_luma=sum(lumas) / len(lumas) if lumas else 0.0,
    )


def _frame_sort_key(path: Path) -> tuple[Any, ...]:
    parts = re.split(r"(\d+)", path.name)
    return tuple(int(part) if part.isdigit() else part for part in parts)


def sort_frame_paths(frame_paths: Iterable[str | Path]) -> tuple[Path, ...]:
    return tuple(sorted((Path(path) for path in frame_paths), key=_frame_sort_key))


def _detection_bounds(
    detection: Any,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int] | None:
    location_data = getattr(detection, "location_data", None)
    relative_box = getattr(location_data, "relative_bounding_box", None)
    if relative_box is not None:
        return expand_relative_box(
            x_min=float(relative_box.xmin),
            y_min=float(relative_box.ymin),
            width=float(relative_box.width),
            height=float(relative_box.height),
            image_width=image_width,
            image_height=image_height,
        )
    pixel_box = getattr(detection, "bounding_box", None)
    if pixel_box is None:
        return None
    width = float(pixel_box.width)
    height = float(pixel_box.height)
    if width <= 0 or height <= 0:
        return None
    expand_x = width * FACE_BOX_EXPANSION
    expand_y = height * FACE_BOX_EXPANSION
    left = max(0, min(math.floor(float(pixel_box.origin_x) - expand_x), image_width))
    top = max(0, min(math.floor(float(pixel_box.origin_y) - expand_y), image_height))
    right = max(
        0,
        min(math.ceil(float(pixel_box.origin_x) + width + expand_x), image_width),
    )
    bottom = max(
        0,
        min(math.ceil(float(pixel_box.origin_y) + height + expand_y), image_height),
    )
    return None if right <= left or bottom <= top else (left, top, right, bottom)


def _is_chinese(character: str) -> bool:
    return "\u3400" <= character <= "\u4dbf" or "\u4e00" <= character <= "\u9fff"


__all__ = [
    "AUDIO_HOP_MS",
    "AUDIO_WINDOW_MS",
    "AudioWindow",
    "FACE_MIN_DETECTION_CONFIDENCE",
    "FACE_SAMPLE_COUNT",
    "FacePreprocessResult",
    "TARGET_SAMPLE_RATE",
    "audio_window_starts",
    "expand_relative_box",
    "make_audio_windows",
    "normalize_cognitive_text",
    "preprocess_face_frames",
    "sample_frame_positions",
    "sort_frame_paths",
]
