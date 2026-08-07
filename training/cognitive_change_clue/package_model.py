"""Build and verify the offline cognitive-mm-v3.3.0 model package."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch

try:
    from .common import ALGORITHM_ROOT, sha256_file, workspace_relative
except ImportError:
    from common import ALGORITHM_ROOT, sha256_file, workspace_relative


MODEL_VERSION = "cognitive-mm-v3.3.0"
FEATURE_VERSION = "cognitive_features_v3.3.0"
TEXT_QUALITY_VERSION = "cognitive_text_quality_v3.3.1"
PACKAGE_ROOT = ALGORITHM_ROOT / "models" / "mental_health" / "cognitive_change_clue" / "v3.3.0"
REPORT_ROOT = ALGORITHM_ROOT / "reports" / "cognitive_change_clue" / "v3.3.0"
SOURCE_ROOT = ALGORITHM_ROOT / "models" / "mental_health" / "cognitive_change_clue" / "v3.3.0"
CHECKPOINT = REPORT_ROOT / "runs" / "COG-20260804-002" / "best_checkpoint.pt"
CALIBRATION = REPORT_ROOT / "calibration.json"
SPLIT = ALGORITHM_ROOT / "data" / "processed" / "cognitive_change_clue" / "v3.3.0" / "cogpic_subject_split_v33.json"
LOCKFILE = ALGORITHM_ROOT / "requirements-cognitive-v3.3.lock"
ASR_ROOT = ALGORITHM_ROOT / "models" / "asr" / "asr-paraformer-zh-v1.0"
MANIFEST_PATH = PACKAGE_ROOT / "manifest.json"
SUMS_PATH = PACKAGE_ROOT / "sha256sums.txt"


class PackageError(RuntimeError):
    pass


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _directory_hash(path: Path) -> str:
    rows: list[str] = []
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        rows.append(f"{sha256_file(child)}  {child.relative_to(path).as_posix()}\n")
    import hashlib

    return hashlib.sha256("".join(rows).encode("utf-8")).hexdigest()


def _copy_frozen(source: Path, target: Path) -> None:
    if not source.is_file():
        raise PackageError(f"required source is missing: {source}")
    if target.exists():
        if sha256_file(source) != sha256_file(target):
            raise PackageError(f"refusing to overwrite divergent package file: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def _relative_file_hash(relative: str) -> str:
    path = PACKAGE_ROOT / relative
    if not path.is_file():
        raise PackageError(f"package file is missing: {path}")
    return sha256_file(path)


def _build_manifest() -> dict[str, Any]:
    checkpoint_sha256 = sha256_file(CHECKPOINT)
    calibration_sha256 = sha256_file(PACKAGE_ROOT / "calibration.json")
    lock_sha256 = sha256_file(PACKAGE_ROOT / "requirements-cognitive-v3.3.lock")
    split_sha256 = sha256_file(PACKAGE_ROOT / "cogpic_subject_split_v33.json")
    asr_sums = ASR_ROOT / "sha256sums.txt"
    if not asr_sums.is_file():
        raise PackageError(f"ASR package checksum file is missing: {asr_sums}")
    assets = {
        "audio": {
            "name": "WAVLM_BASE_PLUS",
            "artifact": "encoders/wavlm_base_plus.pth",
            "sha256": _relative_file_hash("encoders/wavlm_base_plus.pth"),
            "sample_rate": 16000,
            "pooling": "last_layer_valid_frame_mean_window_duration_mean",
        },
        "text": {
            "name": "hfl/chinese-roberta-wwm-ext",
            "hf_revision": "5c58d0b8ec1d9014354d691c538661bf00bfdb44",
            "artifact": "encoders/roberta",
            "sha256": _directory_hash(PACKAGE_ROOT / "encoders" / "roberta"),
            "max_length": 256,
            "pooling": "cls",
            "num_token_mode": "literal_text_no_new_token",
        },
        "face": {
            "name": "resnet18",
            "weights": "IMAGENET1K_V1",
            "artifact": "encoders/resnet18_imagenet1k_v1.pth",
            "sha256": _relative_file_hash("encoders/resnet18_imagenet1k_v1.pth"),
            "pooling": "valid_face_mean",
        },
    }
    artifacts = {
        "fusion": {"path": "fusion.pt", "sha256": _relative_file_hash("fusion.pt")},
        "calibration": {"path": "calibration.json", "sha256": calibration_sha256},
        "feature_stats": {"path": "feature_stats.json", "sha256": _relative_file_hash("feature_stats.json")},
        "split": {"path": "cogpic_subject_split_v33.json", "sha256": split_sha256},
        "config": {"path": "config_snapshot.yaml", "sha256": _relative_file_hash("config_snapshot.yaml")},
        "lockfile": {"path": "requirements-cognitive-v3.3.lock", "sha256": lock_sha256},
        "asr_reference": {"path": "asr_package_ref.json", "sha256": _relative_file_hash("asr_package_ref.json")},
        "detector": {
            "path": "detectors/mediapipe_face_detection_short_range.tflite",
            "sha256": _relative_file_hash("detectors/mediapipe_face_detection_short_range.tflite"),
        },
    }
    return {
        "schema_version": "cognitive_model_manifest_v1",
        "model_version": MODEL_VERSION,
        "feature_version": FEATURE_VERSION,
        "text_quality_version": TEXT_QUALITY_VERSION,
        "lockfile_sha256": lock_sha256,
        "source_checkpoint_sha256": checkpoint_sha256,
        "source_checkpoint": workspace_relative(CHECKPOINT),
        "encoders": assets,
        "asr": {
            "schema_version": "asr_transcript_v1",
            "name": "funasr-paraformer-zh",
            "version": "asr-paraformer-zh-v1.0",
            "package": workspace_relative(ASR_ROOT),
            "package_sha256": sha256_file(asr_sums),
        },
        "detectors": {
            "face_detection": "detectors/mediapipe_face_detection_short_range.tflite",
            "face_detection_sha256": artifacts["detector"]["sha256"],
            "min_detection_confidence": 0.60,
        },
        "training": {
            "dataset": "CogPic",
            "seed": 20260802,
            "split_file": "cogpic_subject_split_v33.json",
            "split_sha256": split_sha256,
            "train_subjects": 367,
            "validation_subjects": 92,
            "test_subjects": 115,
        },
        "model_structure": {"projection_dim": 256, "fusion_hidden_dims": [128, 64], "dropout": 0.2},
        "supported_api_modalities": ["audio+text+face", "audio+text", "audio+face", "audio"],
        "artifacts": artifacts,
        "network_access": "forbidden",
    }


def generate_sha256sums() -> str:
    rows: list[str] = []
    for path in sorted(item for item in PACKAGE_ROOT.rglob("*") if item.is_file() and item.name != "sha256sums.txt"):
        rows.append(f"{sha256_file(path)}  {path.relative_to(PACKAGE_ROOT).as_posix()}\n")
    return "".join(rows)


def verify_package() -> dict[str, Any]:
    if not MANIFEST_PATH.is_file() or not SUMS_PATH.is_file():
        raise PackageError("package manifest or sha256sums.txt is missing")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("model_version") != MODEL_VERSION:
        raise PackageError("model package version mismatch")
    actual_rows = generate_sha256sums()
    expected_rows = SUMS_PATH.read_text(encoding="ascii")
    if actual_rows != expected_rows:
        raise PackageError("sha256sums.txt does not match package files")
    for relative in ("fusion.pt", "calibration.json", "feature_stats.json", "config_snapshot.yaml", "cogpic_subject_split_v33.json", "requirements-cognitive-v3.3.lock", "asr_package_ref.json"):
        if not (PACKAGE_ROOT / relative).is_file():
            raise PackageError(f"required package file is missing: {relative}")
    calibration = json.loads((PACKAGE_ROOT / "calibration.json").read_text(encoding="utf-8"))
    if calibration.get("checkpoint_sha256") != manifest.get("source_checkpoint_sha256"):
        raise PackageError("calibration and source checkpoint hashes disagree")
    return {
        "schema_version": "cognitive_model_package_verification_v1",
        "status": "passed",
        "model_version": MODEL_VERSION,
        "package_root": workspace_relative(PACKAGE_ROOT),
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "sha256sums_sha256": sha256_file(SUMS_PATH),
        "file_count": len([path for path in PACKAGE_ROOT.rglob("*") if path.is_file()]),
        "network_access": "forbidden",
    }


def build_package() -> dict[str, Any]:
    required_sources = [CHECKPOINT, CALIBRATION, SPLIT, LOCKFILE, ASR_ROOT / "sha256sums.txt"]
    for source in required_sources:
        if not source.is_file():
            raise PackageError(f"required source is missing: {source}")
    PACKAGE_ROOT.mkdir(parents=True, exist_ok=True)
    _copy_frozen(CHECKPOINT, PACKAGE_ROOT / "fusion.pt")
    _copy_frozen(CALIBRATION, PACKAGE_ROOT / "calibration.json")
    _copy_frozen(SPLIT, PACKAGE_ROOT / "cogpic_subject_split_v33.json")
    _copy_frozen(LOCKFILE, PACKAGE_ROOT / "requirements-cognitive-v3.3.lock")
    _copy_frozen(ALGORITHM_ROOT / "configs" / "modules" / "cognitive_change_clue_v3_3.yaml", PACKAGE_ROOT / "config_snapshot.yaml")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    stats = checkpoint.get("moca_statistics")
    if not isinstance(stats, Mapping):
        raise PackageError("checkpoint is missing moca_statistics")
    feature_stats = {
        "schema_version": "cognitive_feature_stats_v1",
        "feature_version": FEATURE_VERSION,
        "text_quality_version": TEXT_QUALITY_VERSION,
        "moca_train_subject_mean": float(stats["mean"]),
        "moca_train_subject_std": float(stats["std"]),
        "moca_train_subject_count": int(stats["subject_count"]),
        "moca_std_fallback_used": bool(float(stats["std"]) == 1.0),
        "checkpoint_sha256": sha256_file(CHECKPOINT),
    }
    if (PACKAGE_ROOT / "feature_stats.json").exists():
        existing = json.loads((PACKAGE_ROOT / "feature_stats.json").read_text(encoding="utf-8"))
        if existing != feature_stats:
            raise PackageError("refusing to overwrite divergent feature_stats.json")
    else:
        _atomic_bytes(PACKAGE_ROOT / "feature_stats.json", _json_bytes(feature_stats))
    asr_ref = {
        "schema_version": "asr_package_ref_v1",
        "version": "asr-paraformer-zh-v1.0",
        "path": workspace_relative(ASR_ROOT),
        "sha256": sha256_file(ASR_ROOT / "sha256sums.txt"),
    }
    if (PACKAGE_ROOT / "asr_package_ref.json").exists():
        existing = json.loads((PACKAGE_ROOT / "asr_package_ref.json").read_text(encoding="utf-8"))
        if existing != asr_ref:
            raise PackageError("refusing to overwrite divergent asr_package_ref.json")
    else:
        _atomic_bytes(PACKAGE_ROOT / "asr_package_ref.json", _json_bytes(asr_ref))
    manifest = _build_manifest()
    if MANIFEST_PATH.exists():
        existing = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if existing != manifest:
            raise PackageError("refusing to overwrite divergent manifest.json")
    else:
        _atomic_bytes(MANIFEST_PATH, _json_bytes(manifest))
    _atomic_bytes(SUMS_PATH, generate_sha256sums().encode("ascii"))
    return verify_package()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    result = verify_package() if args.verify else build_package()
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["PackageError", "build_package", "generate_sha256sums", "main", "verify_package"]
