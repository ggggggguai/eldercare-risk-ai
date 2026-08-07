"""Generate resumable CogPic ASR V1.0 text and transcript caches."""

from __future__ import annotations

import argparse
import json
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from rapidfuzz.distance import Levenshtein

from elderly_monitoring.modules.asr.facade import transcribe_audio
from elderly_monitoring.modules.asr.schemas import ASRModelInfo, ASRQuality, ASRTranscript
from elderly_monitoring.modules.asr.settings import ASRSettings
from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.preprocess import (
    normalize_cognitive_text,
)

try:
    from .common import (
        ALGORITHM_ROOT,
        ASR_MODEL_VERSION,
        DEFAULT_MANIFEST_PATH,
        DEFAULT_PROCESSED_ROOT,
        atomic_write_bytes,
        atomic_write_json,
        atomic_write_jsonl,
        atomic_write_parquet,
        read_manifest,
        resolve_workspace_path,
        sha256_bytes,
        sha256_file,
        workspace_relative,
    )
except ImportError:
    from common import (
        ALGORITHM_ROOT,
        ASR_MODEL_VERSION,
        DEFAULT_MANIFEST_PATH,
        DEFAULT_PROCESSED_ROOT,
        atomic_write_bytes,
        atomic_write_json,
        atomic_write_jsonl,
        atomic_write_parquet,
        read_manifest,
        resolve_workspace_path,
        sha256_bytes,
        sha256_file,
        workspace_relative,
    )


DEFAULT_INDEX_PATH = DEFAULT_PROCESSED_ROOT / "asr_cache_index.jsonl"
DEFAULT_REPORT_PATH = DEFAULT_PROCESSED_ROOT / "asr_difference_audit.json"


def run_asr_cache(
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    index_path: str | Path = DEFAULT_INDEX_PATH,
    report_path: str | Path = DEFAULT_REPORT_PATH,
    device: str = "auto",
    checkpoint_every: int = 10,
    force: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    index_path = Path(index_path)
    report_path = Path(report_path)
    frame = read_manifest(manifest_path).sort_values("sample_id", kind="stable")
    if limit is not None:
        frame_to_process = frame.head(max(0, int(limit)))
    else:
        frame_to_process = frame
    settings = ASRSettings(device=device)
    package_hash = sha256_file(settings.model_root / "sha256sums.txt")
    existing = _load_index(index_path)
    records: dict[str, dict[str, Any]] = dict(existing)
    completed_since_checkpoint = 0

    for row in frame_to_process.to_dict(orient="records"):
        sample_id = str(row["sample_id"])
        audio_path = resolve_workspace_path(str(row["audio_path"]))
        text_path = resolve_workspace_path(str(row["asr_text_path"]))
        transcript_path = resolve_workspace_path(str(row["asr_transcript_path"]))
        audio_sha256 = sha256_file(audio_path)
        current = records.get(sample_id)
        if not force and current is not None and _cache_record_valid(
            current,
            audio_sha256=audio_sha256,
            text_path=text_path,
            transcript_path=transcript_path,
        ):
            continue
        try:
            transcript = transcribe_audio(
                audio_path,
                request_id=f"cogpic:{sample_id}",
                settings=settings,
            )
        except Exception as exc:
            transcript = _failed_transcript(sample_id, type(exc).__name__)
        transcript_payload = (
            json.dumps(
                transcript.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            + "\n"
        ).encode("utf-8")
        text_payload = transcript.text.encode("utf-8")
        atomic_write_bytes(transcript_path, transcript_payload)
        atomic_write_bytes(text_path, text_payload)
        original_text_path = resolve_workspace_path(str(row["original_text_path"]))
        original_text = original_text_path.read_text(encoding="utf-8-sig", errors="replace")
        normalized_asr, asr_char_count = normalize_cognitive_text(transcript.text)
        normalized_original, original_char_count = normalize_cognitive_text(original_text)
        similarity = float(
            Levenshtein.normalized_similarity(normalized_asr, normalized_original)
        )
        records[sample_id] = {
            "schema_version": "cogpic_asr_cache_record_v1",
            "sample_id": sample_id,
            "audio_path": str(row["audio_path"]),
            "audio_sha256": audio_sha256,
            "asr_model_version": transcript.model.version,
            "asr_package_sha256": package_hash,
            "status": transcript.status,
            "quality": transcript.quality.model_dump(mode="json"),
            "text_path": workspace_relative(text_path),
            "text_sha256": sha256_bytes(text_payload),
            "transcript_path": workspace_relative(transcript_path),
            "transcript_sha256": sha256_bytes(transcript_payload),
            "normalized_char_count": asr_char_count,
            "original_text_char_count": original_char_count,
            "normalized_text_similarity": similarity,
            "original_text_sha256": sha256_file(original_text_path),
        }
        completed_since_checkpoint += 1
        if completed_since_checkpoint >= max(1, int(checkpoint_every)):
            _write_index(index_path, records)
            completed_since_checkpoint = 0
    _write_index(index_path, records)
    _update_manifest(frame, records, manifest_path)
    report = _build_report(frame, records, manifest_path, index_path, package_hash)
    atomic_write_json(report_path, report)
    return report


def _failed_transcript(sample_id: str, warning: str) -> ASRTranscript:
    return ASRTranscript(
        request_id=f"cogpic:{sample_id}",
        status="failed",
        quality=ASRQuality(
            audio_duration_ms=0,
            speech_duration_ms=0,
            speech_ratio=0.0,
            mean_confidence=None,
            decode_status="ok",
            vad_status="failed",
        ),
        model=ASRModelInfo(version=ASR_MODEL_VERSION),
        warnings=[f"cache_exception:{warning}"],
    )


def _cache_record_valid(
    record: Mapping[str, Any],
    *,
    audio_sha256: str,
    text_path: Path,
    transcript_path: Path,
) -> bool:
    if record.get("audio_sha256") != audio_sha256:
        return False
    if record.get("asr_model_version") != ASR_MODEL_VERSION:
        return False
    if not text_path.is_file() or not transcript_path.is_file():
        return False
    return (
        record.get("text_sha256") == sha256_file(text_path)
        and record.get("transcript_sha256") == sha256_file(transcript_path)
    )


def _load_index(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    records: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict) or not value.get("sample_id"):
            raise ValueError(f"invalid ASR index row {line_number}")
        sample_id = str(value["sample_id"])
        if sample_id in records:
            raise ValueError(f"duplicate ASR cache sample_id: {sample_id}")
        records[sample_id] = value
    return records


def _write_index(path: Path, records: Mapping[str, Mapping[str, Any]]) -> None:
    atomic_write_jsonl(path, (records[key] for key in sorted(records)))


def _update_manifest(frame: Any, records: Mapping[str, Mapping[str, Any]], path: Path) -> None:
    statuses: list[str | None] = []
    qualities: list[str | None] = []
    for sample_id in frame["sample_id"].astype(str):
        record = records.get(sample_id)
        statuses.append(str(record["status"]) if record else None)
        qualities.append(
            json.dumps(record["quality"], sort_keys=True, separators=(",", ":"))
            if record
            else None
        )
    frame["asr_status"] = statuses
    frame["asr_quality"] = qualities
    atomic_write_parquet(path, frame)


def _build_report(
    frame: Any,
    records: Mapping[str, Mapping[str, Any]],
    manifest_path: Path,
    index_path: Path,
    package_hash: str,
) -> dict[str, Any]:
    selected = [records[sample_id] for sample_id in frame["sample_id"].astype(str) if sample_id in records]
    similarities = [float(record["normalized_text_similarity"]) for record in selected]
    return {
        "schema_version": "cogpic_asr_difference_audit_v1",
        "asr_model_version": ASR_MODEL_VERSION,
        "asr_package_sha256": package_hash,
        "manifest_sha256": sha256_file(manifest_path),
        "index_sha256": sha256_file(index_path),
        "manifest_rows": len(frame),
        "indexed_rows": len(selected),
        "status_counts": dict(sorted(Counter(record["status"] for record in selected).items())),
        "difference_audit": {
            "source_txt_usage": "audit_only_not_model_input",
            "mean_normalized_similarity": (
                sum(similarities) / len(similarities) if similarities else None
            ),
            "minimum_normalized_similarity": min(similarities) if similarities else None,
            "maximum_normalized_similarity": max(similarities) if similarities else None,
        },
        "complete": len(selected) == len(frame),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda:0"), default="auto")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = run_asr_cache(
            manifest_path=args.manifest,
            index_path=args.index,
            report_path=args.report,
            device=args.device,
            checkpoint_every=args.checkpoint_every,
            force=args.force,
            limit=args.limit,
        )
    except Exception:
        traceback.print_exc()
        return 1
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
