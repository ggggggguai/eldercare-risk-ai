"""Integrity verification and offline loading for cognitive model packages."""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.encoders import (
    AudioEncoder,
    FaceEncoder,
    TextEncoder,
)
from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.model import (
    CognitiveChangeClueModel,
    model_state_is_finite,
)
from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.preprocess import (
    FACE_MIN_DETECTION_CONFIDENCE,
)
from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.schemas import (
    DEFAULT_MODEL_VERSION,
    SUPPORTED_MODEL_VERSIONS,
    V34_MODEL_VERSION,
)


_CHECKSUM_LINE = re.compile(r"^([0-9a-f]{64})  (.+)$")


class CognitivePackageError(RuntimeError):
    """Raised when the frozen model package is absent, changed, or unreadable."""


@dataclass(frozen=True)
class CognitivePackageMetadata:
    model_version: str
    package_hash: str
    calibration_a: float
    calibration_b: float
    moca_mean: float
    moca_std: float
    primary_modalities: tuple[str, ...]
    research_outputs_available: bool


@dataclass
class CognitiveRuntimeAssets:
    model: Any
    audio_encoder: AudioEncoder
    text_encoder: TextEncoder
    face_encoder: FaceEncoder
    face_detector: Any
    metadata: CognitivePackageMetadata

    def close(self) -> None:
        exit_method = getattr(self.face_detector, "__exit__", None)
        if callable(exit_method):
            exit_method(None, None, None)


class CognitiveModelPackage:
    def __init__(
        self,
        root: str | Path,
        *,
        device: str = "cpu",
        expected_model_version: str = DEFAULT_MODEL_VERSION,
    ) -> None:
        if expected_model_version not in SUPPORTED_MODEL_VERSIONS:
            raise ValueError("expected cognitive model version is unsupported")
        self.root = Path(root).resolve()
        self.device = device
        self.expected_model_version = expected_model_version
        self._lock = threading.Lock()
        self._metadata: CognitivePackageMetadata | None = None
        self._manifest: dict[str, Any] | None = None
        self._assets: CognitiveRuntimeAssets | None = None

    def verify(self, *, force: bool = False) -> CognitivePackageMetadata:
        with self._lock:
            if self._metadata is not None and not force:
                return self._metadata
            if not self.root.is_dir():
                raise CognitivePackageError(f"model package is unavailable: {self.root}")
            checksum_path = self.root / "sha256sums.txt"
            if not checksum_path.is_file():
                raise CognitivePackageError("model package checksum manifest is missing")
            entries = self._read_checksum_entries(checksum_path)
            expected_paths = [relative for relative, _ in entries]
            actual_paths = sorted(
                path.relative_to(self.root).as_posix()
                for path in self.root.rglob("*")
                if path.is_file() and path.name != "sha256sums.txt"
            )
            if expected_paths != actual_paths:
                raise CognitivePackageError("model package file set differs from sha256sums.txt")
            for relative, expected in entries:
                if _sha256_file(self.root / Path(relative)) != expected:
                    raise CognitivePackageError(f"model package checksum mismatch: {relative}")

            manifest = _read_json(self.root / "manifest.json")
            calibration = _read_json(self.root / "calibration.json")
            feature_stats = _read_json(self.root / "feature_stats.json")
            if manifest.get("model_version") != self.expected_model_version:
                raise CognitivePackageError("model package version is unsupported")
            if manifest.get("network_access") != "forbidden":
                raise CognitivePackageError("model package must forbid network access")
            if calibration.get("checkpoint_sha256") != manifest.get("source_checkpoint_sha256"):
                raise CognitivePackageError("calibration checkpoint hash does not match package")
            if feature_stats.get("checkpoint_sha256") != manifest.get("source_checkpoint_sha256"):
                raise CognitivePackageError("feature statistics checkpoint hash does not match package")
            primary_modalities, research_outputs_available = _runtime_policy(
                manifest,
                expected_model_version=self.expected_model_version,
            )
            metadata = CognitivePackageMetadata(
                model_version=self.expected_model_version,
                package_hash=_sha256_file(checksum_path),
                calibration_a=_finite_number(calibration, "a"),
                calibration_b=_finite_number(calibration, "b"),
                moca_mean=_finite_number(feature_stats, "moca_train_subject_mean"),
                moca_std=_positive_number(feature_stats, "moca_train_subject_std"),
                primary_modalities=primary_modalities,
                research_outputs_available=research_outputs_available,
            )
            self._manifest = manifest
            self._metadata = metadata
            return metadata

    def load_assets(self) -> CognitiveRuntimeAssets:
        with self._lock:
            existing = self._assets
        if existing is not None:
            return existing
        # Startup verification may have happened earlier. Re-hash before the first
        # actual load so a package changed between startup and inference is rejected.
        metadata = self.verify(force=True)
        try:
            import torch

            resolved_device = self._resolve_device(torch)
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            manifest = self._manifest or {}
            structure = manifest.get("model_structure") or {}
            model = CognitiveChangeClueModel(
                projection_dim=int(structure.get("projection_dim", 256)),
                fusion_hidden_dims=tuple(structure.get("fusion_hidden_dims", (128, 64))),
                dropout=float(structure.get("dropout", 0.2)),
                input_dims=structure.get("input_dims"),
            )
            checkpoint = torch.load(
                self.root / "fusion.pt",
                map_location="cpu",
                weights_only=False,
            )
            state = checkpoint.get("model_state_dict")
            if not isinstance(state, dict):
                raise CognitivePackageError("fusion checkpoint has no model_state_dict")
            model.load_state_dict(state, strict=True)
            model.eval().to(resolved_device)
            if not model_state_is_finite(model):
                raise CognitivePackageError("fusion checkpoint contains non-finite values")
            audio_encoder = AudioEncoder(
                self.root / "encoders" / "wavlm_base_plus.pth",
                device=resolved_device,
            )
            text_encoder = TextEncoder(
                self.root / "encoders" / "roberta",
                device=resolved_device,
            )
            face_encoder = FaceEncoder(
                self.root / "encoders" / "resnet18_imagenet1k_v1.pth",
                device=resolved_device,
            )
            detector = _create_face_detector(
                self.root / "detectors" / "mediapipe_face_detection_short_range.tflite"
            )
            detector.__enter__()
            assets = CognitiveRuntimeAssets(
                model=model,
                audio_encoder=audio_encoder,
                text_encoder=text_encoder,
                face_encoder=face_encoder,
                face_detector=detector,
                metadata=metadata,
            )
        except CognitivePackageError:
            raise
        except Exception as exc:
            raise CognitivePackageError("model package assets could not be loaded") from exc
        with self._lock:
            if self._assets is None:
                self._assets = assets
                return assets
            assets.close()
            return self._assets

    def close(self) -> None:
        with self._lock:
            assets = self._assets
            self._assets = None
        if assets is not None:
            assets.close()

    def _read_checksum_entries(self, path: Path) -> list[tuple[str, str]]:
        entries: list[tuple[str, str]] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            match = _CHECKSUM_LINE.fullmatch(line)
            if match is None:
                raise CognitivePackageError(f"invalid checksum line {line_number}")
            digest, relative = match.groups()
            relative_path = PurePosixPath(relative)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise CognitivePackageError(f"unsafe checksum path on line {line_number}")
            entries.append((relative, digest))
        paths = [relative for relative, _ in entries]
        if not paths or paths != sorted(paths) or len(paths) != len(set(paths)):
            raise CognitivePackageError("checksum paths must be non-empty, unique, and sorted")
        return entries

    def _resolve_device(self, torch: Any) -> str:
        if self.device == "auto":
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda:0" and not torch.cuda.is_available():
            raise CognitivePackageError("CUDA was requested but is unavailable")
        if self.device not in {"cpu", "cuda:0"}:
            raise CognitivePackageError("cognitive device must be auto, cpu, or cuda:0")
        return self.device


def _create_face_detector(model_path: Path) -> Any:
    import mediapipe as mp
    import numpy as np

    options = mp.tasks.vision.FaceDetectorOptions(
        base_options=mp.tasks.BaseOptions(model_asset_buffer=model_path.read_bytes()),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        min_detection_confidence=FACE_MIN_DETECTION_CONFIDENCE,
    )
    detector = mp.tasks.vision.FaceDetector.create_from_options(options)

    class Adapter:
        def __enter__(self) -> "Adapter":
            detector.__enter__()
            return self

        def __exit__(self, *args: Any) -> Any:
            return detector.__exit__(*args)

        def process(self, rgb: np.ndarray) -> Any:
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            return detector.detect(image)

    return Adapter()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CognitivePackageError(f"model package JSON is unreadable: {path.name}") from exc
    if not isinstance(value, dict):
        raise CognitivePackageError(f"model package JSON must be an object: {path.name}")
    return value


def _runtime_policy(
    manifest: dict[str, Any],
    *,
    expected_model_version: str,
) -> tuple[tuple[str, ...], bool]:
    if expected_model_version == DEFAULT_MODEL_VERSION:
        return ("audio", "text", "face"), True
    if expected_model_version != V34_MODEL_VERSION:
        raise CognitivePackageError("model package version is unsupported")
    candidate = manifest.get("candidate")
    if not isinstance(candidate, dict):
        raise CognitivePackageError("V3.4 model package candidate policy is missing")
    primary_modalities = tuple(candidate.get("primary_modalities") or ())
    if (
        manifest.get("schema_version") != "cognitive_model_manifest_v2"
        or candidate.get("id") != "V34-A2"
        or candidate.get("heads") != "main_head"
        or primary_modalities != ("audio", "text")
        or candidate.get("face_in_primary_score") is not False
    ):
        raise CognitivePackageError("V3.4 model package runtime policy is invalid")
    return primary_modalities, False


def _finite_number(values: dict[str, Any], key: str) -> float:
    import math

    try:
        value = float(values[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise CognitivePackageError(f"model package numeric field is invalid: {key}") from exc
    if not math.isfinite(value):
        raise CognitivePackageError(f"model package numeric field is non-finite: {key}")
    return value


def _positive_number(values: dict[str, Any], key: str) -> float:
    value = _finite_number(values, key)
    if value <= 0:
        raise CognitivePackageError(f"model package numeric field must be positive: {key}")
    return value


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "CognitiveModelPackage",
    "CognitivePackageError",
    "CognitivePackageMetadata",
    "CognitiveRuntimeAssets",
]
