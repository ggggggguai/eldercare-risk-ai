"""Recompute V3.3.1 text quality metadata without re-encoding text embeddings."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

from elderly_monitoring.modules.asr.schemas import ASRTranscript
from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.preprocess import (
    normalize_cognitive_text,
)
from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.quality import (
    TEXT_QUALITY_VERSION,
    compute_text_quality,
)

try:
    from .common import (
        DEFAULT_MANIFEST_PATH,
        DEFAULT_PROCESSED_ROOT,
        atomic_write_json,
        atomic_write_jsonl,
        read_manifest,
        resolve_workspace_path,
        sha256_file,
    )
except ImportError:
    from common import (
        DEFAULT_MANIFEST_PATH,
        DEFAULT_PROCESSED_ROOT,
        atomic_write_json,
        atomic_write_jsonl,
        read_manifest,
        resolve_workspace_path,
        sha256_file,
    )


DEFAULT_FEATURE_ROOT = DEFAULT_PROCESSED_ROOT / "features"
DEFAULT_REPORT_PATH = (
    DEFAULT_PROCESSED_ROOT.parents[3]
    / "reports"
    / "cognitive_change_clue"
    / "v3.3.0"
    / "text_quality_migration_v3.3.1.json"
)


class TextQualityMigrationError(RuntimeError):
    pass


def recompute_text_quality(
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    feature_root: str | Path = DEFAULT_FEATURE_ROOT,
    report_path: str | Path = DEFAULT_REPORT_PATH,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    feature_root = Path(feature_root)
    report_path = Path(report_path)
    index_path = feature_root / "text_index.jsonl"
    cache_manifest_path = feature_root / "feature_cache_manifest.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    index_backup_path = report_path.parent / "text_index_v3.3.0.backup.jsonl"
    manifest_backup_path = report_path.parent / "feature_cache_manifest_v3.3.0.backup.json"
    if not index_backup_path.exists():
        shutil.copy2(index_path, index_backup_path)
    if not manifest_backup_path.exists():
        shutil.copy2(cache_manifest_path, manifest_backup_path)

    frame = read_manifest(manifest_path).sort_values("sample_id", kind="stable")
    records = _read_jsonl(index_path)
    sample_ids = tuple(frame["sample_id"].astype(str))
    if set(records) != set(sample_ids):
        raise TextQualityMigrationError("text index sample_ids do not match the manifest")

    embedding_fingerprint_before = _embedding_fingerprint(records.values())
    before_missing = sum(int(record.get("missing", 1)) for record in records.values())
    updated: list[dict[str, Any]] = []
    confidence_counts: Counter[str] = Counter()

    for row in frame.to_dict(orient="records"):
        sample_id = str(row["sample_id"])
        record = dict(records[sample_id])
        transcript_path = resolve_workspace_path(str(row["asr_transcript_path"]))
        transcript = ASRTranscript.model_validate_json(transcript_path.read_text(encoding="utf-8"))
        normalized, char_count = normalize_cognitive_text(transcript.text)
        quality = compute_text_quality(char_count=char_count, segments=transcript.segments)
        confidence_available = bool(quality.metrics["confidence_available"])
        confidence_counts["available" if confidence_available else "unavailable"] += 1

        record.update(
            {
                "quality": quality.score,
                "missing": quality.missing,
                "char_count": char_count,
                "mean_confidence": quality.metrics["mean_confidence"],
                "confidence_available": confidence_available,
                "quality_version": TEXT_QUALITY_VERSION,
                "asr_status": transcript.status,
                "asr_model_version": transcript.model.version,
                "normalized_text_sha256": hashlib.sha256(
                    normalized.encode("utf-8")
                ).hexdigest(),
            }
        )
        if transcript.status != "completed" or char_count < 5:
            if record.get("feature_path") is not None:
                raise TextQualityMigrationError(
                    f"{sample_id} has a text embedding for an unusable transcript"
                )
            record["status"] = "missing"
            record["reason"] = (
                f"asr_{transcript.status}"
                if transcript.status != "completed"
                else "text_too_short"
            )
            record["missing"] = 1
        else:
            if record.get("feature_path") is None:
                raise TextQualityMigrationError(
                    f"{sample_id} is missing an existing text embedding"
                )
            record["status"] = "completed"
            record["reason"] = None
        updated.append(record)

    embedding_fingerprint_after = _embedding_fingerprint(updated)
    if embedding_fingerprint_after != embedding_fingerprint_before:
        raise TextQualityMigrationError("text embedding paths or hashes changed during migration")

    atomic_write_jsonl(index_path, updated)
    index_sha256 = sha256_file(index_path)
    cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    cache_manifest.setdefault("quality_versions", {})["text"] = TEXT_QUALITY_VERSION
    cache_manifest["modalities"]["text"]["index_sha256"] = index_sha256
    atomic_write_json(cache_manifest_path, cache_manifest)

    after_missing = sum(int(record["missing"]) for record in updated)
    result = {
        "schema_version": "cognitive_text_quality_migration_v1",
        "quality_version": TEXT_QUALITY_VERSION,
        "manifest_sha256": sha256_file(manifest_path),
        "text_index_path": str(index_path.resolve()),
        "text_index_sha256": index_sha256,
        "text_index_backup_path": str(index_backup_path.resolve()),
        "text_index_backup_sha256": sha256_file(index_backup_path),
        "feature_cache_manifest_sha256": sha256_file(cache_manifest_path),
        "feature_cache_manifest_backup_path": str(manifest_backup_path.resolve()),
        "feature_cache_manifest_backup_sha256": sha256_file(manifest_backup_path),
        "rows": len(updated),
        "missing_mask_before": before_missing,
        "missing_mask_after": after_missing,
        "recovered_modalities": before_missing - after_missing,
        "confidence_counts": dict(sorted(confidence_counts.items())),
        "embedding_fingerprint_before": embedding_fingerprint_before,
        "embedding_fingerprint_after": embedding_fingerprint_after,
        "embeddings_reencoded": False,
        "asr_rerun": False,
    }
    atomic_write_json(report_path, result)
    result["report_path"] = str(report_path.resolve())
    result["report_sha256"] = sha256_file(report_path)
    return result


def _read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            sample_id = str(record["sample_id"])
            if sample_id in records:
                raise TextQualityMigrationError(f"duplicate text index sample_id: {sample_id}")
            records[sample_id] = record
    return records


def _embedding_fingerprint(records: Any) -> str:
    entries = sorted(
        (str(record.get("sample_id")), record.get("feature_path"), record.get("feature_sha256"))
        for record in records
    )
    payload = json.dumps(entries, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = recompute_text_quality(
            manifest_path=args.manifest,
            feature_root=args.feature_root,
            report_path=args.report,
        )
    except Exception:
        traceback.print_exc()
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
