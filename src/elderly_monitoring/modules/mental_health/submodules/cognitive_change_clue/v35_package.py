"""Integrity verification and loading for the frozen V3.5 subject model."""

from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.manifest import (
    CognitiveModelPackage,
    CognitivePackageError,
    CognitiveRuntimeAssets,
)
from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.v35_model import (
    V35SubjectAggregator,
    v35_model_state_is_finite,
)
from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.v35_schemas import (
    V35_MODEL_VERSION,
)


V35_SEEDS = (20260818, 20260828, 20260908)
_CHECKSUM_LINE = re.compile(r"^([0-9a-f]{64})  (.+)$")


@dataclass(frozen=True)
class CognitiveV35PackageMetadata:
    model_version: str
    package_hash: str
    encoder_package_hash: str
    calibration_a: float
    calibration_b: float
    threshold: float
    primary_modalities: tuple[str, ...] = ("audio", "text", "face")
    research_outputs_available: bool = False


@dataclass(frozen=True)
class CognitiveV35RuntimeAssets:
    models: tuple[V35SubjectAggregator, ...]
    encoder_assets: CognitiveRuntimeAssets
    metadata: CognitiveV35PackageMetadata

    @property
    def audio_encoder(self):
        return self.encoder_assets.audio_encoder

    @property
    def text_encoder(self):
        return self.encoder_assets.text_encoder

    @property
    def face_encoder(self):
        return self.encoder_assets.face_encoder

    @property
    def face_detector(self):
        return self.encoder_assets.face_detector


class CognitiveV35ModelPackage:
    """V3.5 classifiers plus the frozen V3.3 encoder asset package."""

    def __init__(
        self,
        root: str | Path,
        *,
        encoder_package: CognitiveModelPackage,
        device: str = "cpu",
    ) -> None:
        self.root = Path(root).resolve()
        self.encoder_package = encoder_package
        self.device = device
        self._lock = threading.Lock()
        self._metadata: CognitiveV35PackageMetadata | None = None
        self._assets: CognitiveV35RuntimeAssets | None = None

    def verify(self, *, force: bool = False) -> CognitiveV35PackageMetadata:
        with self._lock:
            if self._metadata is not None and not force:
                return self._metadata
            if not self.root.is_dir():
                raise CognitivePackageError(f"V3.5 model package is unavailable: {self.root}")
            checksum_path = self.root / "sha256sums.txt"
            if not checksum_path.is_file():
                raise CognitivePackageError("V3.5 checksum manifest is missing")
            entries = self._read_checksum_entries(checksum_path)
            expected_paths = [relative for relative, _ in entries]
            actual_paths = sorted(
                path.relative_to(self.root).as_posix()
                for path in self.root.rglob("*")
                if path.is_file() and path.name != "sha256sums.txt"
            )
            if expected_paths != actual_paths:
                raise CognitivePackageError(
                    "V3.5 package file set differs from sha256sums.txt"
                )
            for relative, expected in entries:
                if _sha256_file(self.root / Path(relative)) != expected:
                    raise CognitivePackageError(
                        f"V3.5 package checksum mismatch: {relative}"
                    )

            manifest = _read_json(self.root / "manifest.json")
            candidate = _read_json(self.root / "candidate_config.json")
            calibration = _read_json(self.root / "calibration.json")
            if (
                manifest.get("model_version") != V35_MODEL_VERSION
                or manifest.get("candidate") != "V35-P=weighted_logit|P0|LOSS-3"
                or manifest.get("status") != "frozen_offline_candidate"
                or manifest.get("official_test_evaluated") is not False
                or manifest.get("official_test_labels_loaded") is not False
                or manifest.get("official_test_media_read") is not False
            ):
                raise CognitivePackageError("V3.5 manifest identity or isolation changed")
            if (
                candidate.get("model_version") != V35_MODEL_VERSION
                or candidate.get("aggregator") != "weighted_logit"
                or candidate.get("task_protocol") != "P0_equal_task_weight"
                or candidate.get("loss") != "LOSS-3"
                or tuple(candidate.get("seeds") or ()) != V35_SEEDS
                or candidate.get("ensemble") != "mean of three raw logits then Platt"
            ):
                raise CognitivePackageError("V3.5 candidate configuration changed")
            if (
                calibration.get("method") != "platt"
                or calibration.get("official_test_evaluated") is not False
            ):
                raise CognitivePackageError("V3.5 calibration policy changed")
            parameters = calibration.get("parameters")
            if not isinstance(parameters, dict):
                raise CognitivePackageError("V3.5 calibration parameters are missing")
            calibration_a = _finite_number(parameters, "a")
            calibration_b = _finite_number(parameters, "b")
            threshold = _probability(calibration, "threshold")
            encoder_metadata = self.encoder_package.verify(force=force)
            metadata = CognitiveV35PackageMetadata(
                model_version=V35_MODEL_VERSION,
                package_hash=_sha256_file(checksum_path),
                encoder_package_hash=encoder_metadata.package_hash,
                calibration_a=calibration_a,
                calibration_b=calibration_b,
                threshold=threshold,
            )
            self._metadata = metadata
            return metadata

    def load_assets(self) -> CognitiveV35RuntimeAssets:
        with self._lock:
            existing = self._assets
        if existing is not None:
            return existing
        metadata = self.verify(force=True)
        try:
            import torch

            resolved_device = self._resolve_device(torch)
            probe = torch.load(
                self.root / "probe_input.pt",
                map_location="cpu",
                weights_only=True,
            )
            expected = _read_json(self.root / "probe_expected.json")
            expected_logits = expected.get("per_model_subject_logits")
            models: list[V35SubjectAggregator] = []
            observed_logits: list[list[float]] = []
            for seed in V35_SEEDS:
                checkpoint = torch.load(
                    self.root / f"model_seed_{seed}.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                if (
                    checkpoint.get("model_version") != V35_MODEL_VERSION
                    or checkpoint.get("candidate") != "weighted_logit|P0|LOSS-3"
                    or checkpoint.get("seed") != seed
                    or checkpoint.get("official_test_evaluated") is not False
                    or checkpoint.get("official_test_labels_loaded") is not False
                    or checkpoint.get("official_test_media_read") is not False
                ):
                    raise CognitivePackageError(
                        f"V3.5 checkpoint identity changed for seed {seed}"
                    )
                state = checkpoint.get("state_dict")
                if not isinstance(state, dict):
                    raise CognitivePackageError(
                        f"V3.5 checkpoint state is missing for seed {seed}"
                    )
                model = V35SubjectAggregator()
                model.load_state_dict(state, strict=True)
                model.eval()
                if not v35_model_state_is_finite(model):
                    raise CognitivePackageError(
                        f"V3.5 checkpoint contains non-finite values for seed {seed}"
                    )
                with torch.inference_mode():
                    observed_logits.append(
                        model(probe)["subject_logit"].detach().cpu().tolist()
                    )
                models.append(model)
            if observed_logits != expected_logits:
                raise CognitivePackageError(
                    "V3.5 deterministic probe differs from frozen expected output"
                )
            for model in models:
                model.to(resolved_device)
            encoder_assets = self.encoder_package.load_assets()
            assets = CognitiveV35RuntimeAssets(
                models=tuple(models),
                encoder_assets=encoder_assets,
                metadata=metadata,
            )
        except CognitivePackageError:
            raise
        except Exception as exc:
            raise CognitivePackageError("V3.5 package assets could not be loaded") from exc
        with self._lock:
            if self._assets is None:
                self._assets = assets
            return self._assets

    def close(self) -> None:
        with self._lock:
            self._assets = None
        self.encoder_package.close()

    def _resolve_device(self, torch: Any) -> str:
        if self.device == "auto":
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda:0" and not torch.cuda.is_available():
            raise CognitivePackageError("CUDA was requested but is unavailable")
        if self.device not in {"cpu", "cuda:0"}:
            raise CognitivePackageError("cognitive device must be auto, cpu, or cuda:0")
        return self.device

    @staticmethod
    def _read_checksum_entries(path: Path) -> list[tuple[str, str]]:
        entries: list[tuple[str, str]] = []
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            1,
        ):
            match = _CHECKSUM_LINE.fullmatch(line)
            if match is None:
                raise CognitivePackageError(
                    f"invalid V3.5 checksum line {line_number}"
                )
            digest, relative = match.groups()
            relative_path = PurePosixPath(relative)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise CognitivePackageError(
                    f"unsafe V3.5 checksum path on line {line_number}"
                )
            entries.append((relative, digest))
        paths = [relative for relative, _ in entries]
        if not paths or paths != sorted(paths) or len(paths) != len(set(paths)):
            raise CognitivePackageError(
                "V3.5 checksum paths must be non-empty, unique, and sorted"
            )
        return entries


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CognitivePackageError(f"invalid V3.5 JSON asset: {path.name}") from exc
    if not isinstance(value, dict):
        raise CognitivePackageError(f"V3.5 JSON asset is not an object: {path.name}")
    return value


def _finite_number(value: dict[str, Any], key: str) -> float:
    import math

    candidate = value.get(key)
    if isinstance(candidate, bool) or not isinstance(candidate, (int, float)):
        raise CognitivePackageError(f"V3.5 {key} must be numeric")
    number = float(candidate)
    if not math.isfinite(number):
        raise CognitivePackageError(f"V3.5 {key} must be finite")
    return number


def _probability(value: dict[str, Any], key: str) -> float:
    number = _finite_number(value, key)
    if not 0.0 <= number <= 1.0:
        raise CognitivePackageError(f"V3.5 {key} must be a probability")
    return number


__all__ = [
    "V35_SEEDS",
    "CognitiveV35ModelPackage",
    "CognitiveV35PackageMetadata",
    "CognitiveV35RuntimeAssets",
]
