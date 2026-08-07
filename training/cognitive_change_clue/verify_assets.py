"""Verify frozen V3.3 assets and offline encoder output dimensions."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path

import numpy as np

from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.encoders import (
    AudioEncoder,
    DEFAULT_ASSET_ROOT,
    FaceEncoder,
    TextEncoder,
    sha256_directory,
    sha256_file as encoder_sha256_file,
)

try:
    from .cache_features import _create_face_detector
    from .common import ALGORITHM_ROOT, atomic_write_json, sha256_file
except ImportError:
    from cache_features import _create_face_detector
    from common import ALGORITHM_ROOT, atomic_write_json, sha256_file


DEFAULT_REPORT = (
    ALGORITHM_ROOT
    / "reports"
    / "cognitive_change_clue"
    / "v3.3.0"
    / "environment_verification.json"
)


def verify_assets(
    *,
    asset_root: str | Path = DEFAULT_ASSET_ROOT,
    report_path: str | Path = DEFAULT_REPORT,
    device: str = "auto",
) -> dict[str, object]:
    import torch

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    asset_root = Path(asset_root)
    manifest = json.loads((asset_root / "asset_manifest.json").read_text(encoding="utf-8"))
    resolved_device = "cuda:0" if device == "auto" and torch.cuda.is_available() else device
    if resolved_device == "auto":
        resolved_device = "cpu"
    if resolved_device == "cuda:0" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    expected_environment = manifest["environment"]
    package_checks = {
        package: {
            "expected": str(expected_environment[key]),
            "actual": importlib.metadata.version(package),
        }
        for package, key in (
            ("torch", "torch"),
            ("torchaudio", "torchaudio"),
            ("torchvision", "torchvision"),
            ("transformers", "transformers"),
            ("mediapipe", "mediapipe"),
        )
    }
    if any(value["expected"] != value["actual"] for value in package_checks.values()):
        raise RuntimeError("installed package matrix does not match the asset manifest")
    requirements_path = ALGORITHM_ROOT / expected_environment["requirements_path"]
    if sha256_file(requirements_path) != expected_environment["requirements_sha256"]:
        raise RuntimeError("requirements-cognitive-v3.3.lock hash mismatch")

    assets = manifest["assets"]
    checks = {
        "audio": encoder_sha256_file(asset_root / assets["audio"]["path"]),
        "text_model": encoder_sha256_file(asset_root / assets["text"]["model_path"]),
        "text_directory": sha256_directory(asset_root / assets["text"]["path"]),
        "face": encoder_sha256_file(asset_root / assets["face"]["path"]),
        "face_detector": encoder_sha256_file(
            asset_root / assets["face_detector"]["path"]
        ),
        "asr": sha256_file(
            ALGORITHM_ROOT / assets["asr"]["path"] / "sha256sums.txt"
        ),
    }
    expected_checks = {
        "audio": assets["audio"]["sha256"],
        "text_model": assets["text"]["model_sha256"],
        "text_directory": assets["text"]["directory_sha256"],
        "face": assets["face"]["sha256"],
        "face_detector": assets["face_detector"]["sha256"],
        "asr": assets["asr"]["sha256sums_sha256"],
    }
    if checks != expected_checks:
        raise RuntimeError("one or more frozen asset hashes do not match")

    audio_encoder = AudioEncoder(device=resolved_device)
    audio = audio_encoder.encode(np.zeros(3 * 16_000, dtype=np.float32), duration_ms=3000)
    del audio_encoder
    if resolved_device.startswith("cuda"):
        torch.cuda.empty_cache()
    text_encoder = TextEncoder(device=resolved_device)
    text = text_encoder.encode("老人正在描述图片内容")
    tokenizer_fast = bool(text_encoder.tokenizer.is_fast)
    del text_encoder
    if resolved_device.startswith("cuda"):
        torch.cuda.empty_cache()
    face_encoder = FaceEncoder(device=resolved_device)
    face = face_encoder.encode((np.zeros((3, 224, 224), dtype=np.float32),))
    del face_encoder
    detector = _create_face_detector(asset_root)
    detector.__enter__()
    try:
        detection_count = len(
            detector.process(np.zeros((224, 224, 3), dtype=np.uint8)).detections
        )
    finally:
        detector.__exit__(None, None, None)

    dimensions = {
        "audio": int(audio.shape[0]),
        "text": int(text.shape[0]),
        "face": int(face.shape[0]),
    }
    if dimensions != {"audio": 768, "text": 768, "face": 512}:
        raise RuntimeError(f"unexpected encoder dimensions: {dimensions}")
    result: dict[str, object] = {
        "schema_version": "cognitive_environment_verification_v1",
        "status": "passed",
        "offline_flags": {
            "HF_HUB_OFFLINE": os.environ["HF_HUB_OFFLINE"],
            "TRANSFORMERS_OFFLINE": os.environ["TRANSFORMERS_OFFLINE"],
        },
        "device": resolved_device,
        "cuda_available": bool(torch.cuda.is_available()),
        "package_checks": package_checks,
        "asset_hashes": checks,
        "encoder_dimensions": dimensions,
        "all_encoder_values_finite": bool(
            np.isfinite(audio).all() and np.isfinite(text).all() and np.isfinite(face).all()
        ),
        "tokenizer_is_fast": tokenizer_fast,
        "face_detector_smoke_detection_count": detection_count,
    }
    atomic_write_json(report_path, result)
    result["report_sha256"] = sha256_file(report_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda:0"), default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = verify_assets(asset_root=args.asset_root, report_path=args.report, device=args.device)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
