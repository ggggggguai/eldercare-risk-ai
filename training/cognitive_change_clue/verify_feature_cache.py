"""Verify the complete CogPic ASR and Audio/Text/Face cache set for V3.3."""

from __future__ import annotations

import argparse
import json
import math
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.encoders import (
    DEFAULT_ASSET_ROOT,
    sha256_directory,
    sha256_file as encoder_sha256_file,
)
from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.quality import (
    TEXT_QUALITY_VERSION,
)

try:
    from .common import (
        ASR_MODEL_VERSION,
        DEFAULT_MANIFEST_PATH,
        DEFAULT_PROCESSED_ROOT,
        DEFAULT_SPLIT_PATH,
        FEATURE_VERSION,
        atomic_write_json,
        canonical_json_bytes,
        load_json,
        read_manifest,
        resolve_workspace_path,
        sha256_bytes,
        sha256_file,
    )
except ImportError:
    from common import (
        ASR_MODEL_VERSION,
        DEFAULT_MANIFEST_PATH,
        DEFAULT_PROCESSED_ROOT,
        DEFAULT_SPLIT_PATH,
        FEATURE_VERSION,
        atomic_write_json,
        canonical_json_bytes,
        load_json,
        read_manifest,
        resolve_workspace_path,
        sha256_bytes,
        sha256_file,
    )


DEFAULT_ASR_INDEX_PATH = DEFAULT_PROCESSED_ROOT / "asr_cache_index.jsonl"
DEFAULT_ASR_AUDIT_PATH = DEFAULT_PROCESSED_ROOT / "asr_difference_audit.json"
DEFAULT_FEATURE_ROOT = DEFAULT_PROCESSED_ROOT / "features"
DEFAULT_REPORT_PATH = (
    DEFAULT_PROCESSED_ROOT.parents[3]
    / "reports"
    / "cognitive_change_clue"
    / "v3.3.0"
    / "feature_cache_verification.json"
)
MODALITY_DIMS = {"audio": 768, "text": 768, "face": 512}
ASR_STATUSES = {
    "completed",
    "completed_empty_speech",
    "unsupported_audio",
    "model_unavailable",
    "failed",
}
FEATURE_STATUSES = {"completed", "missing"}


class CacheVerificationError(RuntimeError):
    pass


def verify_feature_cache(
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    split_path: str | Path = DEFAULT_SPLIT_PATH,
    asr_index_path: str | Path = DEFAULT_ASR_INDEX_PATH,
    asr_audit_path: str | Path = DEFAULT_ASR_AUDIT_PATH,
    feature_root: str | Path = DEFAULT_FEATURE_ROOT,
    asset_root: str | Path = DEFAULT_ASSET_ROOT,
    report_path: str | Path = DEFAULT_REPORT_PATH,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    split_path = Path(split_path)
    asr_index_path = Path(asr_index_path)
    asr_audit_path = Path(asr_audit_path)
    feature_root = Path(feature_root)
    asset_root = Path(asset_root)
    report_path = Path(report_path)

    frame = read_manifest(manifest_path).sort_values("sample_id", kind="stable").reset_index(
        drop=True
    )
    _require(len(frame) == 1722, f"manifest row count is {len(frame)}, expected 1722")
    _require(frame["sample_id"].nunique() == len(frame), "manifest sample_id is not unique")
    _require(
        frame["subject_id"].nunique() == 574,
        f"manifest subject count is {frame['subject_id'].nunique()}, expected 574",
    )
    _require(not frame["derived_split"].isna().any(), "manifest has empty derived_split")
    _require(
        set(frame["derived_split"].astype(str)) == {"train", "validation", "test"},
        "manifest has invalid derived_split values",
    )
    rows = {str(row["sample_id"]): row for row in frame.to_dict(orient="records")}
    sample_ids = set(rows)
    manifest_sha256 = sha256_file(manifest_path)
    split_sha256 = sha256_file(split_path)

    split_report = _verify_split(frame, split_path)
    asr_report, asr_records = _verify_asr(
        frame=frame,
        rows=rows,
        sample_ids=sample_ids,
        manifest_sha256=manifest_sha256,
        index_path=asr_index_path,
        audit_path=asr_audit_path,
    )
    asset_hashes = _asset_hashes(asset_root)
    feature_report = _verify_features(
        rows=rows,
        sample_ids=sample_ids,
        manifest_sha256=manifest_sha256,
        split_sha256=split_sha256,
        feature_root=feature_root,
        asset_hashes=asset_hashes,
        asr_records=asr_records,
    )

    result = {
        "schema_version": "cognitive_feature_cache_verification_v1",
        "status": "passed",
        "feature_version": FEATURE_VERSION,
        "manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": manifest_sha256,
            "rows": len(frame),
            "subjects": int(frame["subject_id"].nunique()),
        },
        "split": {"path": str(split_path.resolve()), "sha256": split_sha256, **split_report},
        "asr": asr_report,
        "features": feature_report,
        "assets": asset_hashes,
        "ready_for_model_training": True,
    }
    atomic_write_json(report_path, result)
    result["report_path"] = str(report_path.resolve())
    result["report_sha256"] = sha256_file(report_path)
    return result


def _verify_split(frame: Any, split_path: Path) -> dict[str, Any]:
    payload = load_json(split_path)
    content_sha256 = str(payload.get("content_sha256") or "")
    content = dict(payload)
    content.pop("content_sha256", None)
    _require(
        sha256_bytes(canonical_json_bytes(content)) == content_sha256,
        "split content_sha256 does not match its canonical content",
    )
    _require(payload.get("seed") == 20260802, "split seed is not 20260802")
    _require(
        payload.get("algorithm") == "stratified_largest_remainder_then_pcg64",
        "split algorithm is not frozen V3.3 algorithm",
    )
    splits = payload.get("splits") or {}
    expected_counts = {"train": 367, "validation": 92, "test": 115}
    _require(payload.get("counts") == expected_counts, "split subject counts are incorrect")
    split_sets = {name: set(splits.get(name) or ()) for name in expected_counts}
    _require(
        not (split_sets["train"] & split_sets["validation"]),
        "subject leakage exists between train and validation",
    )
    _require(
        not (split_sets["train"] & split_sets["test"]),
        "subject leakage exists between train and test",
    )
    _require(
        not (split_sets["validation"] & split_sets["test"]),
        "subject leakage exists between validation and test",
    )
    observed_subjects = set().union(*split_sets.values())
    expected_subjects = set(frame["subject_id"].astype(str))
    _require(observed_subjects == expected_subjects, "split does not cover manifest subjects")
    subject_splits = frame.groupby("subject_id")["derived_split"].nunique()
    _require(int(subject_splits.max()) == 1, "a manifest subject crosses derived splits")
    mapping = {
        subject_id: split_name
        for split_name, subjects in split_sets.items()
        for subject_id in subjects
    }
    mismatches = frame[
        frame.apply(
            lambda row: mapping.get(str(row["subject_id"])) != str(row["derived_split"]),
            axis=1,
        )
    ]
    _require(mismatches.empty, "manifest derived_split disagrees with frozen split")
    return {
        "content_sha256": content_sha256,
        "counts": expected_counts,
        "no_subject_leakage": True,
        "manifest_mapping_matches": True,
    }


def _verify_asr(
    *,
    frame: Any,
    rows: Mapping[str, Mapping[str, Any]],
    sample_ids: set[str],
    manifest_sha256: str,
    index_path: Path,
    audit_path: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    records = _read_jsonl(index_path)
    _require(set(records) == sample_ids, "ASR index sample_ids do not match manifest")
    status_counts: Counter[str] = Counter()
    for sample_id in sorted(sample_ids):
        record = records[sample_id]
        manifest_row = rows[sample_id]
        status = str(record.get("status"))
        _require(status in ASR_STATUSES, f"{sample_id} has invalid ASR status {status}")
        _require(
            record.get("asr_model_version") == ASR_MODEL_VERSION,
            f"{sample_id} has wrong ASR model version",
        )
        _require(
            record.get("audio_sha256") == manifest_row.get("audio_sha256"),
            f"{sample_id} ASR audio hash disagrees with manifest",
        )
        transcript_path = resolve_workspace_path(str(record.get("transcript_path") or ""))
        text_path = resolve_workspace_path(str(record.get("text_path") or ""))
        _require(transcript_path.is_file(), f"{sample_id} ASR transcript is missing")
        _require(text_path.is_file(), f"{sample_id} ASR text file is missing")
        _require(
            sha256_file(transcript_path) == record.get("transcript_sha256"),
            f"{sample_id} ASR transcript hash mismatch",
        )
        _require(
            sha256_file(text_path) == record.get("text_sha256"),
            f"{sample_id} ASR text hash mismatch",
        )
        transcript = load_json(transcript_path)
        _require(
            transcript.get("schema_version") == "asr_transcript_v1",
            f"{sample_id} ASR transcript schema mismatch",
        )
        _require(transcript.get("status") == status, f"{sample_id} ASR status mismatch")
        model = transcript.get("model") or {}
        _require(
            model.get("version") == ASR_MODEL_VERSION,
            f"{sample_id} transcript model version mismatch",
        )
        _require(
            str(manifest_row.get("asr_status")) == status,
            f"{sample_id} manifest ASR status mismatch",
        )
        _require(
            str(manifest_row.get("asr_transcript_path")) == str(record.get("transcript_path")),
            f"{sample_id} manifest transcript path mismatch",
        )
        _require(
            str(manifest_row.get("asr_text_path")) == str(record.get("text_path")),
            f"{sample_id} manifest text path mismatch",
        )
        status_counts[status] += 1

    audit = load_json(audit_path)
    _require(audit.get("complete") is True, "ASR difference audit is not complete")
    _require(audit.get("manifest_rows") == len(frame), "ASR audit manifest count mismatch")
    _require(audit.get("indexed_rows") == len(records), "ASR audit index count mismatch")
    _require(audit.get("manifest_sha256") == manifest_sha256, "ASR audit manifest hash mismatch")
    _require(audit.get("index_sha256") == sha256_file(index_path), "ASR audit index hash mismatch")
    _require(
        audit.get("status_counts") == dict(sorted(status_counts.items())),
        "ASR audit status counts mismatch",
    )
    difference = audit.get("difference_audit") or {}
    _require(
        difference.get("source_txt_usage") == "audit_only_not_model_input",
        "ASR audit does not freeze source TXT as audit-only",
    )
    return (
        {
            "index_path": str(index_path.resolve()),
            "index_sha256": sha256_file(index_path),
            "audit_path": str(audit_path.resolve()),
            "audit_sha256": sha256_file(audit_path),
            "rows": len(records),
            "status_counts": dict(sorted(status_counts.items())),
            "all_files_and_hashes_valid": True,
            "source_txt_usage": "audit_only_not_model_input",
        },
        records,
    )


def _verify_features(
    *,
    rows: Mapping[str, Mapping[str, Any]],
    sample_ids: set[str],
    manifest_sha256: str,
    split_sha256: str,
    feature_root: Path,
    asset_hashes: Mapping[str, str],
    asr_records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    cache_manifest_path = feature_root / "feature_cache_manifest.json"
    cache_manifest = load_json(cache_manifest_path)
    _require(cache_manifest.get("complete") is True, "feature cache manifest is incomplete")
    _require(
        cache_manifest.get("ready_for_training") is True,
        "feature cache manifest is not ready for training",
    )
    _require(
        cache_manifest.get("feature_version") == FEATURE_VERSION,
        "feature cache version mismatch",
    )
    _require(
        cache_manifest.get("manifest_sha256") == manifest_sha256,
        "feature cache manifest source hash mismatch",
    )
    _require(
        cache_manifest.get("split_sha256") == split_sha256,
        "feature cache split hash mismatch",
    )
    _require(cache_manifest.get("assets") == dict(asset_hashes), "feature asset hashes mismatch")
    _require(cache_manifest.get("dimensions") == MODALITY_DIMS, "feature dimensions mismatch")
    _require(
        cache_manifest.get("quality_versions", {}).get("text") == TEXT_QUALITY_VERSION,
        "text quality version mismatch",
    )
    modality_reports: dict[str, Any] = {}
    for modality, dimension in MODALITY_DIMS.items():
        index_path = feature_root / f"{modality}_index.jsonl"
        records = _read_jsonl(index_path)
        _require(set(records) == sample_ids, f"{modality} index sample_ids do not match manifest")
        status_counts: Counter[str] = Counter()
        missing_count = 0
        quality_values: list[float] = []
        embedding_count = 0
        for sample_id in sorted(sample_ids):
            record = records[sample_id]
            row = rows[sample_id]
            status = str(record.get("status"))
            _require(
                status in FEATURE_STATUSES,
                f"{sample_id} {modality} has invalid status {status}",
            )
            _require(
                record.get("feature_version") == FEATURE_VERSION,
                f"{sample_id} {modality} feature version mismatch",
            )
            _require(record.get("modality") == modality, f"{sample_id} modality mismatch")
            _require(
                str(record.get("subject_id")) == str(row.get("subject_id")),
                f"{sample_id} {modality} subject mismatch",
            )
            _require(
                str(record.get("derived_split")) == str(row.get("derived_split")),
                f"{sample_id} {modality} split mismatch",
            )
            _require(
                record.get("asset_sha256") == asset_hashes[modality],
                f"{sample_id} {modality} asset hash mismatch",
            )
            if modality == "audio":
                _require(
                    record.get("source_sha256") == row.get("audio_sha256"),
                    f"{sample_id} audio source hash mismatch",
                )
            elif modality == "text":
                _require(
                    record.get("source_sha256")
                    == asr_records[sample_id].get("transcript_sha256"),
                    f"{sample_id} text source hash mismatch",
                )
                _require(
                    record.get("quality_version") == TEXT_QUALITY_VERSION,
                    f"{sample_id} text quality version mismatch",
                )
                _require(
                    isinstance(record.get("confidence_available"), bool),
                    f"{sample_id} text confidence availability is invalid",
                )
            else:
                _require(
                    _is_sha256(record.get("source_sha256")),
                    f"{sample_id} face source hash is invalid",
                )
                _require(
                    _is_sha256(record.get("sampled_frame_set_sha256")),
                    f"{sample_id} valid face frame hash is invalid",
                )
            quality = float(record.get("quality"))
            _require(math.isfinite(quality) and 0.0 <= quality <= 1.0, f"{sample_id} bad quality")
            missing = int(record.get("missing"))
            _require(missing in (0, 1), f"{sample_id} bad missing mask")
            feature_value = record.get("feature_path")
            if feature_value is None:
                _require(status == "missing", f"{sample_id} missing feature has status {status}")
                _require(missing == 1, f"{sample_id} missing feature is marked available")
                _require(record.get("feature_sha256") is None, f"{sample_id} null feature has hash")
                _require(record.get("output_dim") is None, f"{sample_id} null feature has dimension")
            else:
                _require(status == "completed", f"{sample_id} feature file has status {status}")
                feature_path = resolve_workspace_path(str(feature_value))
                _require(feature_path.is_file(), f"{sample_id} {modality} NPZ is missing")
                _require(
                    sha256_file(feature_path) == record.get("feature_sha256"),
                    f"{sample_id} {modality} NPZ hash mismatch",
                )
                _require(
                    record.get("output_dim") == dimension,
                    f"{sample_id} {modality} output dimension metadata mismatch",
                )
                with np.load(feature_path, allow_pickle=False) as payload:
                    _require(payload.files == ["embedding"], f"{sample_id} unexpected NPZ arrays")
                    embedding = np.asarray(payload["embedding"])
                _require(
                    embedding.shape == (dimension,),
                    f"{sample_id} {modality} embedding shape is {embedding.shape}",
                )
                _require(
                    np.issubdtype(embedding.dtype, np.floating),
                    f"{sample_id} {modality} embedding is not floating point",
                )
                _require(
                    bool(np.isfinite(embedding).all()),
                    f"{sample_id} {modality} embedding contains non-finite values",
                )
                _require(
                    bool(np.any(embedding != 0.0)),
                    f"{sample_id} {modality} embedding is an all-zero placeholder",
                )
                embedding_count += 1
            status_counts[status] += 1
            missing_count += missing
            quality_values.append(quality)

        cached = (cache_manifest.get("modalities") or {}).get(modality) or {}
        index_sha256 = sha256_file(index_path)
        _require(cached.get("complete") is True, f"{modality} cache report is incomplete")
        _require(
            cached.get("ready_for_training") is True,
            f"{modality} cache report is not ready for training",
        )
        _require(cached.get("indexed_rows") == len(records), f"{modality} indexed count mismatch")
        _require(cached.get("expected_rows") == len(records), f"{modality} expected count mismatch")
        _require(cached.get("index_sha256") == index_sha256, f"{modality} index hash mismatch")
        _require(
            cached.get("status_counts") == dict(sorted(status_counts.items())),
            f"{modality} status counts mismatch",
        )
        modality_reports[modality] = {
            "index_path": str(index_path.resolve()),
            "index_sha256": index_sha256,
            "rows": len(records),
            "embedding_files": embedding_count,
            "missing_mask_count": missing_count,
            "status_counts": dict(sorted(status_counts.items())),
            "quality": {
                "minimum": min(quality_values),
                "mean": sum(quality_values) / len(quality_values),
                "maximum": max(quality_values),
            },
            "output_dim": dimension,
            "all_hashes_valid": True,
            "all_embeddings_finite_and_nonzero": True,
        }
    return {
        "cache_manifest_path": str(cache_manifest_path.resolve()),
        "cache_manifest_sha256": sha256_file(cache_manifest_path),
        "complete": True,
        "ready_for_training": True,
        "modalities": modality_reports,
    }


def _asset_hashes(asset_root: Path) -> dict[str, str]:
    paths = {
        "audio": asset_root / "encoders" / "wavlm_base_plus.pth",
        "text": asset_root / "encoders" / "roberta",
        "face": asset_root / "encoders" / "resnet18_imagenet1k_v1.pth",
        "face_detector": asset_root
        / "detectors"
        / "mediapipe_face_detection_short_range.tflite",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    _require(not missing, "asset files are missing: " + ", ".join(missing))
    return {
        "audio": encoder_sha256_file(paths["audio"]),
        "text": sha256_directory(paths["text"]),
        "face": encoder_sha256_file(paths["face"]),
        "face_detector": encoder_sha256_file(paths["face_detector"]),
    }


def _read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    _require(path.is_file(), f"JSONL index is missing: {path}")
    records: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        _require(isinstance(value, dict), f"{path.name}:{line_number} is not an object")
        sample_id = str(value.get("sample_id") or "")
        _require(bool(sample_id), f"{path.name}:{line_number} has no sample_id")
        _require(sample_id not in records, f"{path.name} duplicates sample_id {sample_id}")
        records[sample_id] = value
    return records


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise CacheVerificationError(message)


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--asr-index", type=Path, default=DEFAULT_ASR_INDEX_PATH)
    parser.add_argument("--asr-audit", type=Path, default=DEFAULT_ASR_AUDIT_PATH)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = verify_feature_cache(
            manifest_path=args.manifest,
            split_path=args.split,
            asr_index_path=args.asr_index,
            asr_audit_path=args.asr_audit,
            feature_root=args.feature_root,
            asset_root=args.asset_root,
            report_path=args.report,
        )
    except Exception:
        traceback.print_exc()
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
