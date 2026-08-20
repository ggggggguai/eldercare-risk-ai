from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys
from typing import Any, Iterable, Mapping, Sequence

from sklearn import __version__ as sklearn_version
from sklearn.model_selection import StratifiedGroupKFold


MANIFEST_SCHEMA = "casme2_apexfusion_manifest_v1"
CROSSWALK_SCHEMA = "casme2_apexfusion_crosswalk_v1"
SPLIT_SCHEMA = "casme2_nested_loso_splits_v1"
CONFIG_SCHEMA = "casme2_apexfusion_data_config_v1"
AUDIT_SCHEMA = "casme2_apexfusion_data_audit_v1"
HASH_SCHEMA = "casme2_apexfusion_data_hashes_v1"
SPLIT_SEED = 20260808
FINAL_SEEDS = (20260806, 20260817, 20260829)
THREE_CLASS = {"disgust": (0, "negative"), "repression": (0, "negative"), "happiness": (1, "positive"), "surprise": (2, "surprise")}
AUX_CLASS = {"disgust": 0, "repression": 1, "happiness": 2, "surprise": 3}
EXPECTED_THREE_COUNTS = {"negative": 86, "positive": 32, "surprise": 25}
EXPECTED_AUX_COUNTS = {"disgust": 59, "repression": 27, "happiness": 32, "surprise": 25}


class OptMe002DataError(ValueError):
    """Raised when the frozen OPT-ME-002 data protocol is violated."""


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def collection_sha256(values: Iterable[str]) -> str:
    digest = sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _source_warning_codes(source_audit: Mapping[str, Any], sample_id: str) -> list[str]:
    result: list[str] = []
    for issue in source_audit.get("issues", []):
        if issue.get("sample_id") in (None, sample_id):
            result.append(str(issue.get("code")))
    return sorted(set(result))


def _resolved_raw_path(raw_root: Path, source: Mapping[str, Any]) -> Path:
    return (raw_root / str(source["subject_id"]) / str(source["sequence_id"])).resolve()


def _eligible(source: Mapping[str, Any]) -> tuple[bool, str | None]:
    emotion = str(source.get("estimated_emotion") or "").lower()
    if emotion not in THREE_CLASS:
        return False, "emotion_outside_frozen_three_class_space"
    onset, apex, offset = source.get("onset_frame"), source.get("apex_frame"), source.get("offset_frame")
    if apex is None or not (int(onset) < int(apex) < int(offset)):
        return False, "invalid_or_boundary_official_apex"
    if not bool(source.get("raw_required_frames_available")):
        return False, "canonical_raw_required_frame_missing"
    return True, None


def build_formal_manifest(
    *,
    source_rows: Sequence[Mapping[str, Any]],
    source_audit: Mapping[str, Any],
    raw_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    formal: list[dict[str, Any]] = []
    crosswalk_samples: list[dict[str, Any]] = []
    for source in sorted(source_rows, key=lambda row: (str(row["subject_id"]), str(row["sample_id"]))):
        eligible, reason = _eligible(source)
        emotion = str(source.get("estimated_emotion") or "").lower()
        mapped = THREE_CLASS.get(emotion)
        crosswalk_samples.append(
            {
                "sample_id": source["sample_id"],
                "subject_id": source["subject_id"],
                "sequence_id": source["sequence_id"],
                "official_emotion": emotion,
                "official_interior_apex": bool(source.get("apex_frame") is not None and source["onset_frame"] < source["apex_frame"] < source["offset_frame"]),
                "evaluation_eligible": eligible,
                "exclusion_reason": reason,
                "three_class_id": mapped[0] if mapped else None,
                "three_class_name": mapped[1] if mapped else None,
                "aux_emotion_id": AUX_CLASS.get(emotion),
            }
        )
        if not eligible:
            continue
        assert mapped is not None
        raw_path = _resolved_raw_path(raw_root, source)
        formal.append(
            {
                "schema_version": MANIFEST_SCHEMA,
                "sample_id": str(source["sample_id"]),
                "dataset": "CASME II official",
                "subject_id": str(source["subject_id"]),
                "sequence_id": str(source["sequence_id"]),
                "source_manifest_key": str(source["sample_id"]),
                "raw_frame_dir": raw_path.as_posix(),
                "onset_frame": int(source["onset_frame"]),
                "apex_frame": int(source["apex_frame"]),
                "offset_frame": int(source["offset_frame"]),
                "official_emotion": emotion,
                "three_class_id": int(mapped[0]),
                "three_class_name": mapped[1],
                "aux_emotion_id": int(AUX_CLASS[emotion]),
                "aux_emotion_name": emotion,
                "flow_ready_with_official_apex": True,
                "publication_restricted": bool(source.get("publication_restricted")),
                "raw_collection_sha256": source.get("raw", {}).get("collection_sha256"),
                "source_warning_codes": _source_warning_codes(source_audit, str(source["sample_id"])),
                "derived_records": [],
            }
        )
    crosswalk = {
        "schema_version": CROSSWALK_SCHEMA,
        "task_id": "DATA-ME2-001",
        "source_scope": "all_255_official_coding_rows",
        "mapping": {
            emotion: {"three_class_id": value[0], "three_class_name": value[1], "aux_emotion_id": AUX_CLASS[emotion]}
            for emotion, value in THREE_CLASS.items()
        },
        "samples": crosswalk_samples,
        "formal_counts": {
            "samples": len(formal),
            "subjects": len({row["subject_id"] for row in formal}),
            "three_class": dict(sorted(Counter(row["three_class_name"] for row in formal).items())),
            "aux_emotion": dict(sorted(Counter(row["aux_emotion_name"] for row in formal).items())),
        },
        "excluded_counts": dict(sorted(Counter(row["exclusion_reason"] for row in crosswalk_samples if not row["evaluation_eligible"]).items())),
        "test_metrics_read": False,
    }
    return formal, crosswalk


def build_nested_splits(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    stable = sorted(rows, key=lambda row: (str(row["subject_id"]), str(row["sample_id"])))
    subjects = sorted({str(row["subject_id"]) for row in stable})
    folds: list[dict[str, Any]] = []
    for outer_index, test_subject in enumerate(subjects):
        outer_train = [row for row in stable if row["subject_id"] != test_subject]
        y = [int(row["three_class_id"]) for row in outer_train]
        groups = [str(row["subject_id"]) for row in outer_train]
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SPLIT_SEED)
        inner: list[dict[str, Any]] = []
        for inner_index, (train_indices, validation_indices) in enumerate(splitter.split(outer_train, y, groups)):
            train_rows = [outer_train[int(index)] for index in train_indices]
            validation_rows = [outer_train[int(index)] for index in validation_indices]
            inner.append(
                {
                    "inner_fold": inner_index,
                    "train_subjects": sorted({row["subject_id"] for row in train_rows}),
                    "validation_subjects": sorted({row["subject_id"] for row in validation_rows}),
                    "train_sample_ids": [row["sample_id"] for row in train_rows],
                    "validation_sample_ids": [row["sample_id"] for row in validation_rows],
                    "train_class_counts": {str(key): value for key, value in sorted(Counter(row["three_class_id"] for row in train_rows).items())},
                    "validation_class_counts": {str(key): value for key, value in sorted(Counter(row["three_class_id"] for row in validation_rows).items())},
                }
            )
        test_rows = [row for row in stable if row["subject_id"] == test_subject]
        folds.append(
            {
                "outer_fold": outer_index,
                "test_subject": test_subject,
                "outer_train_subjects": sorted({row["subject_id"] for row in outer_train}),
                "outer_train_sample_ids": [row["sample_id"] for row in outer_train],
                "outer_test_sample_ids": [row["sample_id"] for row in test_rows],
                "inner_folds": inner,
            }
        )
    return {
        "schema_version": SPLIT_SCHEMA,
        "task_id": "DATA-ME2-001",
        "protocol": "24-subject strict Nested-LOSO",
        "inner_algorithm": "sklearn.model_selection.StratifiedGroupKFold",
        "inner_n_splits": 5,
        "inner_shuffle": True,
        "split_seed": SPLIT_SEED,
        "stable_sort": ["subject_id", "sample_id"],
        "final_training_seeds": list(FINAL_SEEDS),
        "subjects": subjects,
        "subject_order_sha256": collection_sha256(subjects),
        "outer_folds": folds,
        "outer_test_metrics_generated": False,
    }


def audit_data(
    *,
    rows: Sequence[Mapping[str, Any]],
    crosswalk: Mapping[str, Any],
    splits: Mapping[str, Any],
    source_warning_count: int,
    verify_paths: bool = True,
) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    def require(condition: bool, code: str, detail: Any) -> None:
        if not condition:
            errors.append({"code": code, "detail": detail})

    sample_ids = [str(row["sample_id"]) for row in rows]
    subjects = [str(row["subject_id"]) for row in rows]
    require(len(rows) == 143, "formal_sample_count", len(rows))
    require(len(set(sample_ids)) == len(sample_ids), "duplicate_sample_id", len(sample_ids) - len(set(sample_ids)))
    require(len(set(subjects)) == 24, "formal_subject_count", len(set(subjects)))
    require(dict(Counter(row["three_class_name"] for row in rows)) == EXPECTED_THREE_COUNTS, "three_class_counts", dict(Counter(row["three_class_name"] for row in rows)))
    require(dict(Counter(row["aux_emotion_name"] for row in rows)) == EXPECTED_AUX_COUNTS, "aux_emotion_counts", dict(Counter(row["aux_emotion_name"] for row in rows)))
    for row in rows:
        require(int(row["onset_frame"]) < int(row["apex_frame"]) < int(row["offset_frame"]), "non_interior_apex", row["sample_id"])
        require(row["source_manifest_key"] == row["sample_id"], "source_key_mismatch", row["sample_id"])
        if verify_paths:
            raw = Path(str(row["raw_frame_dir"]))
            require(raw.is_dir(), "raw_path_missing", raw.as_posix())
            for frame in (row["onset_frame"], row["apex_frame"], row["offset_frame"]):
                require((raw / f"img{int(frame)}.jpg").is_file(), "required_frame_missing", {"sample_id": row["sample_id"], "frame": frame})

    outer_folds = list(splits.get("outer_folds", []))
    require(len(outer_folds) == 24, "outer_fold_count", len(outer_folds))
    require(sorted(fold.get("test_subject") for fold in outer_folds) == sorted(set(subjects)), "outer_test_subject_coverage", [fold.get("test_subject") for fold in outer_folds])
    all_ids = set(sample_ids)
    for fold in outer_folds:
        train_ids, test_ids = set(fold["outer_train_sample_ids"]), set(fold["outer_test_sample_ids"])
        require(not train_ids & test_ids and train_ids | test_ids == all_ids, "outer_partition", fold["outer_fold"])
        require(len(fold.get("inner_folds", [])) == 5, "inner_fold_count", fold["outer_fold"])
        pooled: list[str] = []
        for inner in fold.get("inner_folds", []):
            train_subjects, val_subjects = set(inner["train_subjects"]), set(inner["validation_subjects"])
            require(not train_subjects & val_subjects, "inner_subject_leakage", {"outer": fold["outer_fold"], "inner": inner["inner_fold"]})
            require(not ({fold["test_subject"]} & (train_subjects | val_subjects)), "outer_test_in_inner", {"outer": fold["outer_fold"], "inner": inner["inner_fold"]})
            pooled.extend(inner["validation_sample_ids"])
        require(len(pooled) == len(set(pooled)) and set(pooled) == train_ids, "inner_pooled_coverage", fold["outer_fold"])

    require(crosswalk.get("formal_counts", {}).get("samples") == 143, "crosswalk_formal_count", crosswalk.get("formal_counts"))
    require(splits.get("split_seed") == SPLIT_SEED, "split_seed", splits.get("split_seed"))
    require(tuple(splits.get("final_training_seeds", [])) == FINAL_SEEDS, "final_seeds", splits.get("final_training_seeds"))
    if source_warning_count:
        warnings.append({"code": "inherited_source_warnings", "count": source_warning_count, "status": "retained_not_repaired"})
    return {
        "schema_version": AUDIT_SCHEMA,
        "task_id": "DATA-ME2-001",
        "status": "passed" if not errors else "failed",
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
        "inventory": {
            "samples": len(rows),
            "subjects": len(set(subjects)),
            "three_class_counts": dict(sorted(Counter(row["three_class_name"] for row in rows).items())),
            "aux_emotion_counts": dict(sorted(Counter(row["aux_emotion_name"] for row in rows).items())),
            "outer_folds": len(outer_folds),
            "inner_folds_total": sum(len(fold.get("inner_folds", [])) for fold in outer_folds),
        },
        "guards": {
            "outer_test_metrics_read": False,
            "model_trained": False,
            "media_copied": False,
            "derived_sample_count": sum(len(row.get("derived_records", [])) for row in rows),
        },
    }


def build_data_release(
    *,
    source_manifest_path: Path,
    source_audit_path: Path,
    historical_scope_path: Path,
    historical_crosswalk_path: Path,
    raw_root: Path,
    output_dir: Path,
    code_paths: Sequence[Path] = (),
) -> dict[str, Any]:
    source_rows = read_jsonl(source_manifest_path)
    source_audit = json.loads(source_audit_path.read_text(encoding="utf-8"))
    rows, crosswalk = build_formal_manifest(source_rows=source_rows, source_audit=source_audit, raw_root=raw_root)
    historical_ids = {row["sample_id"] for row in read_jsonl(historical_scope_path)}
    current_ids = {row["sample_id"] for row in rows}
    if current_ids != historical_ids:
        raise OptMe002DataError(f"Frozen 143-sample scope mismatch: missing={sorted(historical_ids-current_ids)}, extra={sorted(current_ids-historical_ids)}")
    splits = build_nested_splits(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "manifest": output_dir / "casme2_three_class_manifest_v1.jsonl",
        "crosswalk": output_dir / "casme2_label_crosswalk_v1.json",
        "splits": output_dir / "casme2_nested_loso_splits_v1.json",
        "config": output_dir / "casme2_data_config_v1.json",
        "audit": output_dir / "casme2_data_audit_v1.json",
        "hashes": output_dir / "casme2_data_hashes_v1.json",
    }
    config = {
        "schema_version": CONFIG_SCHEMA,
        "task_id": "DATA-ME2-001",
        "dataset_root": raw_root.parent.parent.as_posix(),
        "canonical_raw_root": raw_root.resolve().as_posix(),
        "formal_samples": 143,
        "formal_subjects": 24,
        "split_seed": SPLIT_SEED,
        "final_training_seeds": list(FINAL_SEEDS),
        "split_algorithm": "StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=20260808)",
        "stable_sort": ["subject_id", "sample_id"],
        "failure_rule": "fail_without_changing_seed_or_protocol",
        "outer_test_access": "partition_only_no_metrics",
        "source_paths": {"official_manifest": source_manifest_path.resolve().as_posix(), "official_audit": source_audit_path.resolve().as_posix(), "historical_scope": historical_scope_path.resolve().as_posix(), "historical_crosswalk": historical_crosswalk_path.resolve().as_posix()},
    }
    write_jsonl(paths["manifest"], rows)
    write_json(paths["crosswalk"], crosswalk)
    write_json(paths["splits"], splits)
    write_json(paths["config"], config)
    audit = audit_data(rows=rows, crosswalk=crosswalk, splits=splits, source_warning_count=int(source_audit.get("warning_count", 0)), verify_paths=True)
    audit["historical_scope_match"] = True
    audit["historical_scope_sha256"] = sha256_file(historical_scope_path)
    audit["historical_crosswalk_sha256"] = sha256_file(historical_crosswalk_path)
    write_json(paths["audit"], audit)
    hash_entries: dict[str, Any] = {}
    for name, path in paths.items():
        if name == "hashes":
            continue
        hash_entries[name] = {"path": path.resolve().as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size}
    for name, path in {"source_manifest": source_manifest_path, "source_audit": source_audit_path, "historical_scope": historical_scope_path, "historical_crosswalk": historical_crosswalk_path}.items():
        hash_entries[name] = {"path": path.resolve().as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size}
    for index, path in enumerate(code_paths):
        hash_entries[f"code_{index}"] = {"path": path.resolve().as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size}
    hashes = {
        "schema_version": HASH_SCHEMA,
        "task_id": "DATA-ME2-001",
        "self_included": False,
        "entries": hash_entries,
        "environment": {"python": sys.version, "platform": platform.platform(), "scikit_learn": sklearn_version},
        "test_metrics_read": False,
    }
    write_json(paths["hashes"], hashes)
    if audit["status"] != "passed":
        raise OptMe002DataError(f"DATA-ME2-001 audit failed: {audit['errors']}")
    return {"paths": {name: path.as_posix() for name, path in paths.items()}, "audit": audit, "hash_manifest_sha256": sha256_file(paths["hashes"])}
