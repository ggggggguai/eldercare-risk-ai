"""Generate resumable per-sample Audio/Text/Face feature caches for V3.3."""

from __future__ import annotations

import argparse
import hashlib
import json
import traceback
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping

import numpy as np

from elderly_monitoring.modules.asr.audio_decode import AudioDecodeError, decode_audio_path
from elderly_monitoring.modules.asr.schemas import ASRTranscript
from elderly_monitoring.modules.asr.settings import ASRSettings
from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.encoders import (
    AudioEncoder,
    DEFAULT_ASSET_ROOT,
    FaceEncoder,
    TextEncoder,
    sha256_directory,
    sha256_file as encoder_sha256_file,
)
from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.preprocess import (
    FACE_MIN_DETECTION_CONFIDENCE,
    normalize_cognitive_text,
    preprocess_face_frames,
    sample_frame_positions,
    sort_frame_paths,
)
from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.quality import (
    TEXT_QUALITY_VERSION,
    compute_audio_quality,
    compute_face_quality,
    compute_text_quality,
)

try:
    from .common import (
        ASR_MODEL_VERSION,
        DEFAULT_MANIFEST_PATH,
        DEFAULT_PROCESSED_ROOT,
        DEFAULT_SPLIT_PATH,
        FEATURE_VERSION,
        atomic_write_json,
        atomic_write_jsonl,
        atomic_write_npz,
        read_manifest,
        resolve_workspace_path,
        sha256_file,
        workspace_relative,
    )
except ImportError:
    from common import (
        ASR_MODEL_VERSION,
        DEFAULT_MANIFEST_PATH,
        DEFAULT_PROCESSED_ROOT,
        DEFAULT_SPLIT_PATH,
        FEATURE_VERSION,
        atomic_write_json,
        atomic_write_jsonl,
        atomic_write_npz,
        read_manifest,
        resolve_workspace_path,
        sha256_file,
        workspace_relative,
    )


DEFAULT_FEATURE_ROOT = DEFAULT_PROCESSED_ROOT / "features"
MODALITIES = ("audio", "text", "face")
DIMENSIONS = {"audio": 768, "text": 768, "face": 512}


class FeatureCacheError(RuntimeError):
    pass


def cache_features(
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    split_path: str | Path = DEFAULT_SPLIT_PATH,
    feature_root: str | Path = DEFAULT_FEATURE_ROOT,
    asset_root: str | Path = DEFAULT_ASSET_ROOT,
    modalities: tuple[str, ...] = MODALITIES,
    device: str = "auto",
    checkpoint_every: int = 10,
    force: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    split_path = Path(split_path)
    feature_root = Path(feature_root)
    asset_root = Path(asset_root)
    unknown = set(modalities).difference(MODALITIES)
    if unknown:
        raise FeatureCacheError(f"unknown modalities: {sorted(unknown)}")
    if not split_path.is_file():
        raise FeatureCacheError(f"fixed split is missing: {split_path}")
    frame = read_manifest(manifest_path).sort_values("sample_id", kind="stable").reset_index(drop=True)
    if frame["derived_split"].isna().any():
        raise FeatureCacheError("manifest contains empty derived_split values")
    process_frame = frame.head(max(0, int(limit))) if limit is not None else frame
    resolved_device = _resolve_device(device)
    assets = _asset_fingerprints(asset_root)
    factories: dict[str, Callable[[], Any]] = {
        "audio": lambda: AudioEncoder(
            asset_root / "encoders" / "wavlm_base_plus.pth", device=resolved_device
        ),
        "text": lambda: TextEncoder(
            asset_root / "encoders" / "roberta", device=resolved_device
        ),
        "face": lambda: FaceEncoder(
            asset_root / "encoders" / "resnet18_imagenet1k_v1.pth",
            device=resolved_device,
        ),
    }
    encoders: dict[str, Any] = {}
    with ExitStack() as stack:
        detector = (
            stack.enter_context(_create_face_detector(asset_root))
            if "face" in modalities
            else None
        )
        reports: dict[str, Any] = {}
        for modality in modalities:
            reports[modality] = _cache_modality(
                modality=modality,
                frame=process_frame,
                full_manifest_rows=len(frame),
                feature_root=feature_root,
                assets=assets,
                encoder_factory=factories[modality],
                encoder_holder=encoders,
                detector=detector,
                checkpoint_every=checkpoint_every,
                force=force,
            )
    cache_manifest = _write_cache_manifest(
        feature_root=feature_root,
        manifest_path=manifest_path,
        split_path=split_path,
        assets=assets,
        reports=reports,
        expected_rows=len(frame),
        device=resolved_device,
    )
    return cache_manifest


def _cache_modality(
    *,
    modality: str,
    frame: Any,
    full_manifest_rows: int,
    feature_root: Path,
    assets: Mapping[str, str],
    encoder_factory: Callable[[], Any],
    encoder_holder: dict[str, Any],
    detector: Any,
    checkpoint_every: int,
    force: bool,
) -> dict[str, Any]:
    index_path = feature_root / f"{modality}_index.jsonl"
    records = _load_index(index_path)
    pending_writes = 0
    for manifest_index, row in zip(frame.index, frame.to_dict(orient="records"), strict=True):
        sample_id = str(row["sample_id"])
        source_sha256 = _source_sha256(modality, row)
        asset_sha256 = assets[modality]
        feature_path = feature_root / modality / f"{sample_id}.npz"
        current = records.get(sample_id)
        if not force and current is not None and _feature_record_valid(
            current,
            source_sha256=source_sha256,
            asset_sha256=asset_sha256,
            feature_path=feature_path,
        ):
            continue
        try:
            if modality not in encoder_holder:
                encoder_holder[modality] = encoder_factory()
            if modality == "audio":
                record, embedding = _audio_feature(row, encoder_holder[modality])
            elif modality == "text":
                record, embedding = _text_feature(row, encoder_holder[modality])
            else:
                record, embedding = _face_feature(row, encoder_holder[modality], detector)
            if embedding is not None:
                expected_shape = (DIMENSIONS[modality],)
                embedding = np.asarray(embedding, dtype=np.float32)
                if embedding.shape != expected_shape or not np.isfinite(embedding).all():
                    raise FeatureCacheError(
                        f"{sample_id} {modality} returned invalid embedding {embedding.shape}"
                    )
                atomic_write_npz(feature_path, embedding=embedding)
                feature_sha256 = sha256_file(feature_path)
                feature_value: str | None = workspace_relative(feature_path)
            else:
                feature_path.unlink(missing_ok=True)
                feature_sha256 = None
                feature_value = None
            records[sample_id] = {
                "schema_version": "cognitive_feature_cache_record_v1",
                "feature_version": FEATURE_VERSION,
                "modality": modality,
                "manifest_index": int(manifest_index),
                "sample_id": sample_id,
                "subject_id": str(row["subject_id"]),
                "derived_split": str(row["derived_split"]),
                "source_sha256": source_sha256,
                "asset_sha256": asset_sha256,
                "feature_path": feature_value,
                "feature_sha256": feature_sha256,
                "output_dim": DIMENSIONS[modality] if embedding is not None else None,
                **record,
            }
        except Exception as exc:
            feature_path.unlink(missing_ok=True)
            records[sample_id] = {
                "schema_version": "cognitive_feature_cache_record_v1",
                "feature_version": FEATURE_VERSION,
                "modality": modality,
                "manifest_index": int(manifest_index),
                "sample_id": sample_id,
                "subject_id": str(row["subject_id"]),
                "derived_split": str(row["derived_split"]),
                "source_sha256": source_sha256,
                "asset_sha256": asset_sha256,
                "feature_path": None,
                "feature_sha256": None,
                "output_dim": None,
                "status": "failed",
                "reason": type(exc).__name__,
                "error": str(exc),
                "quality": 0.0,
                "missing": 1,
            }
        pending_writes += 1
        if pending_writes >= max(1, int(checkpoint_every)):
            _write_index(index_path, records)
            pending_writes = 0
    _write_index(index_path, records)
    selected_ids = set(frame["sample_id"].astype(str))
    selected = [records[sample_id] for sample_id in sorted(selected_ids) if sample_id in records]
    status_counts = dict(sorted(Counter(record["status"] for record in selected).items()))
    return {
        "index_path": str(index_path.resolve()),
        "index_sha256": sha256_file(index_path),
        "indexed_rows": len(selected),
        "expected_rows": full_manifest_rows,
        "status_counts": status_counts,
        "complete": len(selected) == full_manifest_rows,
        "ready_for_training": len(selected) == full_manifest_rows
        and status_counts.get("failed", 0) == 0,
    }


def _audio_feature(row: Mapping[str, Any], encoder: AudioEncoder) -> tuple[dict[str, Any], Any]:
    audio_path = resolve_workspace_path(str(row["audio_path"]))
    transcript = _load_transcript(row)
    try:
        decoded = decode_audio_path(audio_path, ASRSettings(device="auto"))
    except AudioDecodeError as exc:
        return (
            {
                "status": "missing",
                "reason": "unsupported_audio",
                "quality": 0.0,
                "missing": 1,
                "duration_ms": int(row.get("audio_duration_ms") or 0),
                "speech_duration_ms": 0,
                "vad_status": "unavailable",
                "decode_error": str(exc),
            },
            None,
        )
    speech_duration_ms = int(transcript.quality.speech_duration_ms)
    quality = compute_audio_quality(
        duration_ms=decoded.duration_ms,
        speech_duration_ms=speech_duration_ms,
        decode_ok=True,
    )
    embedding = encoder.encode(decoded.samples, duration_ms=decoded.duration_ms)
    return (
        {
            "status": "completed",
            "reason": None,
            "quality": quality.score,
            "missing": quality.missing,
            "duration_ms": decoded.duration_ms,
            "speech_duration_ms": speech_duration_ms,
            "speech_ratio": quality.metrics["speech_ratio"],
            "vad_status": transcript.quality.vad_status,
            "asr_status": transcript.status,
            "asr_model_version": transcript.model.version,
        },
        embedding,
    )


def _text_feature(row: Mapping[str, Any], encoder: TextEncoder) -> tuple[dict[str, Any], Any]:
    transcript = _load_transcript(row)
    normalized, char_count = normalize_cognitive_text(transcript.text)
    quality = compute_text_quality(char_count=char_count, segments=transcript.segments)
    base = {
        "quality": quality.score,
        "missing": quality.missing,
        "char_count": char_count,
        "mean_confidence": quality.metrics["mean_confidence"],
        "confidence_available": quality.metrics["confidence_available"],
        "quality_version": TEXT_QUALITY_VERSION,
        "asr_status": transcript.status,
        "asr_model_version": transcript.model.version,
        "normalized_text_sha256": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
        "num_token_mode": "literal_text_no_new_token",
    }
    if transcript.status != "completed":
        return ({"status": "missing", "reason": f"asr_{transcript.status}", **base}, None)
    if char_count < 5:
        return ({"status": "missing", "reason": "text_too_short", **base}, None)
    return ({"status": "completed", "reason": None, **base}, encoder.encode(normalized))


def _face_feature(
    row: Mapping[str, Any],
    encoder: FaceEncoder,
    detector: Any,
) -> tuple[dict[str, Any], Any]:
    face_path = resolve_workspace_path(str(row["face_path"]))
    frame_paths = tuple(face_path.glob("*.jpg"))
    result = preprocess_face_frames(frame_paths, detector)
    quality = compute_face_quality(
        valid_count=result.valid_count,
        mean_luma=result.mean_luma,
    )
    base = {
        "quality": quality.score,
        "missing": quality.missing,
        "frame_count": len(frame_paths),
        "sampled_positions": list(result.sampled_positions),
        "valid_positions": list(result.valid_positions),
        "valid_face_frame_count": result.valid_count,
        "valid_face_frame_ratio": quality.metrics["valid_face_frame_ratio"],
        "mean_luma": result.mean_luma,
        "sampled_frame_set_sha256": _paths_hash(result.source_paths),
    }
    if result.valid_count == 0:
        return ({"status": "missing", "reason": "no_valid_face", **base}, None)
    return (
        {"status": "completed", "reason": None, **base},
        encoder.encode(result.tensors),
    )


def _load_transcript(row: Mapping[str, Any]) -> ASRTranscript:
    path = resolve_workspace_path(str(row["asr_transcript_path"]))
    if not path.is_file():
        raise FeatureCacheError(f"ASR transcript is missing: {path}")
    return ASRTranscript.model_validate_json(path.read_text(encoding="utf-8"))


def _source_sha256(modality: str, row: Mapping[str, Any]) -> str:
    if modality == "audio":
        path = resolve_workspace_path(str(row["audio_path"]))
        return sha256_file(path)
    if modality == "text":
        path = resolve_workspace_path(str(row["asr_transcript_path"]))
        return sha256_file(path) if path.is_file() else "missing"
    face_path = resolve_workspace_path(str(row["face_path"]))
    frame_paths = sort_frame_paths(face_path.glob("*.jpg"))
    positions = sample_frame_positions(len(frame_paths))
    return _paths_hash(str(frame_paths[position]) for position in positions)


def _paths_hash(paths: Any) -> str:
    digest = hashlib.sha256()
    for raw_path in paths:
        path = Path(raw_path)
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _asset_fingerprints(asset_root: Path) -> dict[str, str]:
    paths = {
        "audio": asset_root / "encoders" / "wavlm_base_plus.pth",
        "text": asset_root / "encoders" / "roberta",
        "face": asset_root / "encoders" / "resnet18_imagenet1k_v1.pth",
        "face_detector": asset_root
        / "detectors"
        / "mediapipe_face_detection_short_range.tflite",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FeatureCacheError("model assets are missing: " + ", ".join(missing))
    return {
        "audio": encoder_sha256_file(paths["audio"]),
        "text": sha256_directory(paths["text"]),
        "face": encoder_sha256_file(paths["face"]),
        "face_detector": encoder_sha256_file(paths["face_detector"]),
    }


def _feature_record_valid(
    record: Mapping[str, Any],
    *,
    source_sha256: str,
    asset_sha256: str,
    feature_path: Path,
) -> bool:
    if record.get("source_sha256") != source_sha256:
        return False
    if record.get("asset_sha256") != asset_sha256:
        return False
    if record.get("feature_path") is None:
        return record.get("status") == "missing" and not feature_path.exists()
    return feature_path.is_file() and record.get("feature_sha256") == sha256_file(feature_path)


def _load_index(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    records: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict) or not value.get("sample_id"):
            raise FeatureCacheError(f"invalid {path.name} row {line_number}")
        sample_id = str(value["sample_id"])
        if sample_id in records:
            raise FeatureCacheError(f"duplicate feature sample_id: {sample_id}")
        records[sample_id] = value
    return records


def _write_index(path: Path, records: Mapping[str, Mapping[str, Any]]) -> None:
    atomic_write_jsonl(path, (records[key] for key in sorted(records)))


def _create_face_detector(asset_root: Path) -> Any:
    import mediapipe as mp

    model_path = asset_root / "detectors" / "mediapipe_face_detection_short_range.tflite"
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


def _resolve_device(device: str) -> str:
    import torch

    if device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cuda:0" and not torch.cuda.is_available():
        raise FeatureCacheError("CUDA was requested but is unavailable")
    return device


def _write_cache_manifest(
    *,
    feature_root: Path,
    manifest_path: Path,
    split_path: Path,
    assets: Mapping[str, str],
    reports: Mapping[str, Any],
    expected_rows: int,
    device: str,
) -> dict[str, Any]:
    existing_path = feature_root / "feature_cache_manifest.json"
    if existing_path.is_file():
        payload = json.loads(existing_path.read_text(encoding="utf-8"))
    else:
        payload = {}
    modality_reports = dict(payload.get("modalities", {}))
    modality_reports.update(reports)
    quality_versions = dict(payload.get("quality_versions", {}))
    if "text" in reports:
        quality_versions["text"] = TEXT_QUALITY_VERSION
    result = {
        "schema_version": "cognitive_feature_cache_manifest_v1",
        "feature_version": FEATURE_VERSION,
        "manifest_sha256": sha256_file(manifest_path),
        "split_sha256": sha256_file(split_path),
        "expected_rows": expected_rows,
        "device": device,
        "assets": dict(assets),
        "dimensions": DIMENSIONS,
        "quality_versions": quality_versions,
        "modalities": modality_reports,
    }
    result["complete"] = all(
        modality_reports.get(modality, {}).get("complete") is True
        for modality in MODALITIES
    )
    result["ready_for_training"] = all(
        modality_reports.get(modality, {}).get("ready_for_training") is True
        for modality in MODALITIES
    )
    atomic_write_json(existing_path, result)
    result["file_sha256"] = sha256_file(existing_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--modality", choices=("all", *MODALITIES), default="all")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda:0"), default="auto")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    modalities = MODALITIES if args.modality == "all" else (args.modality,)
    try:
        result = cache_features(
            manifest_path=args.manifest,
            split_path=args.split,
            feature_root=args.feature_root,
            asset_root=args.asset_root,
            modalities=modalities,
            device=args.device,
            checkpoint_every=args.checkpoint_every,
            force=args.force,
            limit=args.limit,
        )
    except Exception:
        traceback.print_exc()
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
