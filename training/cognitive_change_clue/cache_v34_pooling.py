"""Build official-Train-only WavLM statistics and RoBERTa mean caches."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from elderly_monitoring.modules.asr.audio_decode import decode_audio_path
from elderly_monitoring.modules.asr.schemas import ASRTranscript
from elderly_monitoring.modules.asr.settings import ASRSettings
from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.encoders import (
    AudioEncoder,
    DEFAULT_ASSET_ROOT,
    TextEncoder,
    sha256_directory,
    sha256_file as encoder_sha256_file,
)
from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.preprocess import (
    normalize_cognitive_text,
)

try:
    from .common import (
        ALGORITHM_ROOT,
        DEFAULT_MANIFEST_PATH,
        atomic_write_json,
        atomic_write_jsonl,
        atomic_write_npz,
        resolve_workspace_path,
        sha256_file,
        workspace_relative,
    )
except ImportError:
    from common import (  # type: ignore[no-redef]
        ALGORITHM_ROOT,
        DEFAULT_MANIFEST_PATH,
        atomic_write_json,
        atomic_write_jsonl,
        atomic_write_npz,
        resolve_workspace_path,
        sha256_file,
        workspace_relative,
    )


FEATURE_VERSION = "cognitive_pooling_features_v3.4.0"
SCHEMA_VERSION = "cognitive_v34_pooling_cache_manifest_v1"
RECORD_SCHEMA_VERSION = "cognitive_v34_pooling_cache_record_v1"
DEFAULT_OUTPUT_ROOT = (
    ALGORITHM_ROOT
    / "data"
    / "processed"
    / "cognitive_change_clue"
    / "v3.4.0"
    / "pooling_features"
)
V33_ROOT = (
    ALGORITHM_ROOT / "data" / "processed" / "cognitive_change_clue" / "v3.3.0"
)
MODALITIES = ("audio_stats", "text_mean")
DIMENSIONS = {"audio_stats": 1536, "text_mean": 768}


class V34PoolingCacheError(RuntimeError):
    pass


def _read_index(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sample_id = str(row["sample_id"])
        if sample_id in records:
            raise V34PoolingCacheError(f"duplicate cache record: {sample_id}")
        records[sample_id] = row
    return records


def _write_index(path: Path, records: Mapping[str, Mapping[str, Any]]) -> None:
    atomic_write_jsonl(path, (records[key] for key in sorted(records)))


def _train_frame(manifest_path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(manifest_path, engine="pyarrow")
    train = frame.loc[frame["official_split"].astype(str).eq("train")].copy()
    train = train.sort_values("sample_id", kind="stable").reset_index(drop=True)
    if len(train) != 1377 or train["subject_id"].astype(str).nunique() != 459:
        raise V34PoolingCacheError("official Train must contain 1377 tasks / 459 subjects")
    return train


def _valid_existing(
    record: Mapping[str, Any] | None,
    *,
    source_sha256: str,
    feature_path: Path,
    dimension: int,
) -> bool:
    if not record or record.get("source_sha256") != source_sha256:
        return False
    if int(record.get("missing", 1)) == 1:
        return record.get("feature_path") is None
    return (
        int(record.get("output_dim", -1)) == dimension
        and feature_path.is_file()
        and record.get("feature_sha256") == sha256_file(feature_path)
    )


def _base_record(
    row: Mapping[str, Any],
    *,
    modality: str,
    source_sha256: str,
    asset_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": RECORD_SCHEMA_VERSION,
        "feature_version": FEATURE_VERSION,
        "modality": modality,
        "sample_id": str(row["sample_id"]),
        "subject_id": str(row["subject_id"]),
        "official_split": "train",
        "source_sha256": source_sha256,
        "asset_sha256": asset_sha256,
    }


def _save_embedding(
    path: Path,
    embedding: np.ndarray,
    *,
    dimension: int,
) -> tuple[str, str]:
    value = np.asarray(embedding, dtype=np.float32)
    if value.shape != (dimension,) or not np.isfinite(value).all():
        raise V34PoolingCacheError(f"invalid pooling embedding shape: {value.shape}")
    atomic_write_npz(path, embedding=value)
    return workspace_relative(path), sha256_file(path)


def _cache_audio(
    frame: pd.DataFrame,
    *,
    output_root: Path,
    device: str,
    force: bool,
    checkpoint_every: int,
) -> dict[str, Any]:
    modality = "audio_stats"
    index_path = output_root / f"{modality}_index.jsonl"
    records = _read_index(index_path)
    asset_path = DEFAULT_ASSET_ROOT / "encoders" / "wavlm_base_plus.pth"
    asset_sha256 = encoder_sha256_file(asset_path)
    encoder: AudioEncoder | None = None
    pending = 0
    for row in frame.to_dict(orient="records"):
        sample_id = str(row["sample_id"])
        source_path = resolve_workspace_path(str(row["audio_path"]))
        source_sha256 = sha256_file(source_path)
        output_path = output_root / modality / f"{sample_id}.npz"
        if not force and _valid_existing(
            records.get(sample_id),
            source_sha256=source_sha256,
            feature_path=output_path,
            dimension=DIMENSIONS[modality],
        ):
            continue
        base = _base_record(
            row,
            modality=modality,
            source_sha256=source_sha256,
            asset_sha256=asset_sha256,
        )
        try:
            encoder = encoder or AudioEncoder(asset_path=asset_path, device=device)
            decoded = decode_audio_path(source_path, ASRSettings(device="auto"))
            embedding = encoder.encode_statistics(
                decoded.samples, duration_ms=decoded.duration_ms
            )
            feature_path, feature_sha256 = _save_embedding(
                output_path, embedding, dimension=DIMENSIONS[modality]
            )
            records[sample_id] = {
                **base,
                "status": "completed",
                "reason": None,
                "missing": 0,
                "output_dim": DIMENSIONS[modality],
                "feature_path": feature_path,
                "feature_sha256": feature_sha256,
                "duration_ms": int(decoded.duration_ms),
                "pooling": "weighted_window_mean_and_standard_deviation",
            }
        except Exception as exc:
            output_path.unlink(missing_ok=True)
            records[sample_id] = {
                **base,
                "status": "missing",
                "reason": type(exc).__name__,
                "error": str(exc),
                "missing": 1,
                "output_dim": None,
                "feature_path": None,
                "feature_sha256": None,
            }
        pending += 1
        if pending >= max(1, checkpoint_every):
            _write_index(index_path, records)
            pending = 0
    _write_index(index_path, records)
    return _modality_summary(index_path, frame, records)


def _load_transcript(row: Mapping[str, Any]) -> ASRTranscript | None:
    value = row.get("asr_transcript_path")
    if not value:
        return None
    path = resolve_workspace_path(str(value))
    if not path.is_file():
        return None
    return ASRTranscript.model_validate_json(path.read_text(encoding="utf-8"))


def _cache_text(
    frame: pd.DataFrame,
    *,
    output_root: Path,
    device: str,
    force: bool,
    checkpoint_every: int,
    batch_size: int,
) -> dict[str, Any]:
    modality = "text_mean"
    index_path = output_root / f"{modality}_index.jsonl"
    records = _read_index(index_path)
    asset_path = DEFAULT_ASSET_ROOT / "encoders" / "roberta"
    asset_sha256 = sha256_directory(asset_path)
    encoder: TextEncoder | None = None
    pending_rows: list[tuple[dict[str, Any], Path, str, str]] = []
    writes = 0

    def flush() -> None:
        nonlocal encoder, writes
        if not pending_rows:
            return
        encoder = encoder or TextEncoder(asset_path=asset_path, device=device)
        embeddings = encoder.encode_mean([item[2] for item in pending_rows])
        for (row, output_path, _, source_sha256), embedding in zip(
            pending_rows, embeddings, strict=True
        ):
            sample_id = str(row["sample_id"])
            base = _base_record(
                row,
                modality=modality,
                source_sha256=source_sha256,
                asset_sha256=asset_sha256,
            )
            feature_path, feature_sha256 = _save_embedding(
                output_path, embedding, dimension=DIMENSIONS[modality]
            )
            records[sample_id] = {
                **base,
                "status": "completed",
                "reason": None,
                "missing": 0,
                "output_dim": DIMENSIONS[modality],
                "feature_path": feature_path,
                "feature_sha256": feature_sha256,
                "pooling": "mean_non_padding_non_special_tokens",
            }
            writes += 1
            if writes % max(1, checkpoint_every) == 0:
                _write_index(index_path, records)
        pending_rows.clear()

    for row in frame.to_dict(orient="records"):
        sample_id = str(row["sample_id"])
        transcript = _load_transcript(row)
        source_path = (
            resolve_workspace_path(str(row["asr_transcript_path"]))
            if row.get("asr_transcript_path")
            else None
        )
        source_sha256 = sha256_file(source_path) if source_path and source_path.is_file() else "unavailable"
        output_path = output_root / modality / f"{sample_id}.npz"
        if not force and _valid_existing(
            records.get(sample_id),
            source_sha256=source_sha256,
            feature_path=output_path,
            dimension=DIMENSIONS[modality],
        ):
            continue
        normalized, char_count = normalize_cognitive_text(
            transcript.text if transcript is not None else ""
        )
        if transcript is None or transcript.status != "completed" or char_count < 5:
            output_path.unlink(missing_ok=True)
            records[sample_id] = {
                **_base_record(
                    row,
                    modality=modality,
                    source_sha256=source_sha256,
                    asset_sha256=asset_sha256,
                ),
                "status": "missing",
                "reason": "asr_text_unavailable" if transcript is None else "text_too_short",
                "missing": 1,
                "output_dim": None,
                "feature_path": None,
                "feature_sha256": None,
            }
            writes += 1
            continue
        pending_rows.append((row, output_path, normalized, source_sha256))
        if len(pending_rows) >= max(1, batch_size):
            flush()
    flush()
    _write_index(index_path, records)
    return _modality_summary(index_path, frame, records)


def _modality_summary(
    index_path: Path,
    frame: pd.DataFrame,
    records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    sample_ids = set(frame["sample_id"].astype(str))
    selected = [records[value] for value in sorted(sample_ids)]
    counts = Counter(str(row["status"]) for row in selected)
    return {
        "index_path": workspace_relative(index_path),
        "index_sha256": sha256_file(index_path),
        "indexed_rows": len(selected),
        "expected_rows": 1377,
        "status_counts": dict(sorted(counts.items())),
        "complete": len(selected) == 1377,
    }


def cache_pooling(
    *,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    modalities: Sequence[str] = MODALITIES,
    device: str = "cuda:0",
    force: bool = False,
    checkpoint_every: int = 10,
    text_batch_size: int = 16,
) -> dict[str, Any]:
    unknown = set(modalities).difference(MODALITIES)
    if unknown:
        raise V34PoolingCacheError(f"unsupported pooling modalities: {sorted(unknown)}")
    frame = _train_frame(manifest_path)
    summaries: dict[str, Any] = {}
    if "audio_stats" in modalities:
        summaries["audio_stats"] = _cache_audio(
            frame,
            output_root=output_root,
            device=device,
            force=force,
            checkpoint_every=checkpoint_every,
        )
    if "text_mean" in modalities:
        summaries["text_mean"] = _cache_text(
            frame,
            output_root=output_root,
            device=device,
            force=force,
            checkpoint_every=checkpoint_every,
            batch_size=text_batch_size,
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "feature_version": FEATURE_VERSION,
        "status": "complete" if all(value["complete"] for value in summaries.values()) else "incomplete",
        "source_scope": "cogpic_official_train_only",
        "official_test_media_read": False,
        "subject_count": 459,
        "task_count": 1377,
        "manifest_path": workspace_relative(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "device": device,
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("torch", "torchaudio", "transformers")
        },
        "modalities": summaries,
    }
    atomic_write_json(output_root / "cache_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--modality", choices=("all", *MODALITIES), default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--text-batch-size", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    modalities = MODALITIES if args.modality == "all" else (args.modality,)
    try:
        report = cache_pooling(
            manifest_path=args.manifest,
            output_root=args.output_root,
            modalities=modalities,
            device=args.device,
            force=args.force,
            checkpoint_every=args.checkpoint_every,
            text_batch_size=args.text_batch_size,
        )
    except Exception:
        traceback.print_exc()
        return 1
    print(json.dumps(report, ensure_ascii=False, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_OUTPUT_ROOT",
    "DIMENSIONS",
    "FEATURE_VERSION",
    "MODALITIES",
    "V34PoolingCacheError",
    "cache_pooling",
]
