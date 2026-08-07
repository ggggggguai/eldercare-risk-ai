from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


MANIFEST_SCHEMA_VERSION = "microexpression_sequence_manifest_v1"
LABEL_NAMES = {0: "negative", 1: "positive", 2: "surprise"}
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
FRAME_ANNOTATION_SOURCES = {"official", "estimated", "missing"}
SAMPLE_ROLES = {"classification", "spotting_negative"}

MANIFEST_FIELDS = (
    "manifest_schema_version",
    "sample_id",
    "source_dataset",
    "sample_role",
    "subject_id",
    "source_subject_id",
    "local_subject_id",
    "sequence_id",
    "source_sequence_id",
    "frame_dir",
    "frame_count",
    "frame_extension",
    "first_frame_name",
    "last_frame_name",
    "label",
    "label_name",
    "action_units",
    "onset_frame",
    "apex_frame",
    "offset_frame",
    "frame_annotation_source",
    "landmark_status",
    "flow_status",
    "quality_status",
    "path_status",
    "split_id",
    "split_role",
    "exclude_reason",
    "evaluation_scope",
    "paper_reproduction_claim",
    "source_record",
)

PATH_MAPPING_FIELDS = (
    "sample_id",
    "source_dataset",
    "sample_role",
    "subject_id",
    "source_subject_id",
    "local_subject_id",
    "source_sequence_id",
    "sequence_id",
    "frame_dir",
    "path_status",
    "frame_count",
    "frame_extension",
)

_NATURAL_NUMBER = re.compile(r"(\d+)")
_SMIC_SUBJECT = re.compile(r"^s0*(\d+)$", re.IGNORECASE)


class ManifestBuildError(ValueError):
    """Raised when source metadata cannot be mapped without guessing."""


def normalize_smic_subject(source_subject_id: str) -> tuple[str, str]:
    value = str(source_subject_id).strip().lower()
    match = _SMIC_SUBJECT.fullmatch(value)
    if not match:
        raise ManifestBuildError(f"Invalid SMIC subject id: {source_subject_id!r}")
    number = int(match.group(1))
    if number <= 0:
        raise ManifestBuildError(f"Invalid SMIC subject number: {source_subject_id!r}")
    return f"s{number:02d}", f"s{number}"


def _natural_key(path: Path) -> tuple[Any, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in _NATURAL_NUMBER.split(path.name)
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path, required_fields: set[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        actual_fields = set(reader.fieldnames or ())
        missing_fields = sorted(required_fields - actual_fields)
        if missing_fields:
            raise ManifestBuildError(
                f"{path.name} is missing required fields: {missing_fields}"
            )
        return [dict(row) for row in reader]


def _parse_label(value: str, source_record: str) -> int:
    try:
        numeric = float(str(value).strip())
        label = int(numeric)
    except (TypeError, ValueError) as exc:
        raise ManifestBuildError(
            f"Invalid label {value!r} at {source_record}"
        ) from exc
    if numeric != label or label not in LABEL_NAMES:
        raise ManifestBuildError(f"Invalid label {value!r} at {source_record}")
    return label


def _frame_inventory(frame_dir: Path) -> dict[str, Any]:
    if not frame_dir.is_dir():
        return {
            "frame_count": 0,
            "frame_extension": None,
            "first_frame_name": None,
            "last_frame_name": None,
            "path_status": "missing",
            "quality_status": "invalid",
            "exclude_reason": "missing_frame_dir",
        }

    frame_paths = sorted(
        (
            path
            for path in frame_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=_natural_key,
    )
    if not frame_paths:
        return {
            "frame_count": 0,
            "frame_extension": None,
            "first_frame_name": None,
            "last_frame_name": None,
            "path_status": "empty",
            "quality_status": "invalid",
            "exclude_reason": "no_image_frames",
        }

    extensions = sorted({path.suffix.lower() for path in frame_paths})
    if len(extensions) == 1:
        extension = extensions[0]
        quality_status = "pending"
        exclude_reason = None
    else:
        extension = "mixed:" + ",".join(extensions)
        quality_status = "invalid"
        exclude_reason = "mixed_frame_extensions"

    return {
        "frame_count": len(frame_paths),
        "frame_extension": extension,
        "first_frame_name": frame_paths[0].name,
        "last_frame_name": frame_paths[-1].name,
        "path_status": "available",
        "quality_status": quality_status,
        "exclude_reason": exclude_reason,
    }


def _base_record(
    *,
    sample_id: str,
    source_dataset: str,
    sample_role: str,
    subject_id: str,
    source_subject_id: str,
    local_subject_id: str,
    sequence_id: str,
    source_sequence_id: str,
    frame_dir: Path,
    label: int | None,
    label_name: str,
    action_units: str | None,
    evaluation_scope: str,
    source_record: str,
) -> dict[str, Any]:
    record = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "sample_id": sample_id,
        "source_dataset": source_dataset,
        "sample_role": sample_role,
        "subject_id": subject_id,
        "source_subject_id": source_subject_id,
        "local_subject_id": local_subject_id,
        "sequence_id": sequence_id,
        "source_sequence_id": source_sequence_id,
        "frame_dir": frame_dir.resolve().as_posix(),
        "label": label,
        "label_name": label_name,
        "action_units": action_units or None,
        "onset_frame": None,
        "apex_frame": None,
        "offset_frame": None,
        "frame_annotation_source": "missing",
        "landmark_status": "pending",
        "flow_status": "pending",
        "split_id": None,
        "split_role": None,
        "evaluation_scope": evaluation_scope,
        "paper_reproduction_claim": False,
        "source_record": source_record,
    }
    inventory = _frame_inventory(frame_dir)
    for field in (
        "frame_count",
        "frame_extension",
        "first_frame_name",
        "last_frame_name",
        "quality_status",
        "path_status",
        "exclude_reason",
    ):
        record[field] = inventory[field]
    return {field: record[field] for field in MANIFEST_FIELDS}


def _build_smic_classification_records(
    smic_root: Path, smic_csv: Path
) -> list[dict[str, Any]]:
    rows = _read_csv(smic_csv, {"dataset", "sub", "filename_o", "label"})
    records: list[dict[str, Any]] = []
    for source_row_number, row in enumerate(rows, start=2):
        source_record = f"{smic_csv.name}:{source_row_number}"
        if str(row["dataset"]).strip().lower() != "smic":
            raise ManifestBuildError(
                f"Unexpected dataset {row['dataset']!r} at {source_record}"
            )
        source_subject_id = str(row["sub"]).strip().lower()
        subject_id, local_subject_id = normalize_smic_subject(source_subject_id)
        label = _parse_label(row["label"], source_record)
        source_sequence_id = str(row["filename_o"]).strip()
        expected_prefix = f"{source_subject_id}_"
        if not source_sequence_id.lower().startswith(expected_prefix):
            raise ManifestBuildError(
                f"Sequence {source_sequence_id!r} does not match subject "
                f"{source_subject_id!r} at {source_record}"
            )
        sequence_id = local_subject_id + source_sequence_id[len(source_subject_id) :]
        label_name = LABEL_NAMES[label]
        frame_dir = (
            smic_root / local_subject_id / "micro" / label_name / sequence_id
        )

        if not frame_dir.is_dir():
            class_root = smic_root / local_subject_id / "micro"
            conflicting_classes = [
                name
                for name in LABEL_NAMES.values()
                if name != label_name and (class_root / name / sequence_id).is_dir()
            ]
            if conflicting_classes:
                raise ManifestBuildError(
                    f"Label-directory mismatch at {source_record}: label={label_name}, "
                    f"found={conflicting_classes}"
                )

        records.append(
            _base_record(
                sample_id=f"smic_hs__{subject_id}__{sequence_id}",
                source_dataset="smic_hs",
                sample_role="classification",
                subject_id=subject_id,
                source_subject_id=source_subject_id,
                local_subject_id=local_subject_id,
                sequence_id=sequence_id,
                source_sequence_id=source_sequence_id,
                frame_dir=frame_dir,
                label=label,
                label_name=label_name,
                action_units=None,
                evaluation_scope="three_class_candidate",
                source_record=source_record,
            )
        )
    return records


def _build_smic_non_micro_records(smic_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for subject_dir in sorted(
        (path for path in smic_root.iterdir() if path.is_dir()), key=_natural_key
    ):
        subject_id, local_subject_id = normalize_smic_subject(subject_dir.name)
        non_micro_root = subject_dir / "non_micro"
        if not non_micro_root.is_dir():
            continue
        for sequence_dir in sorted(
            (path for path in non_micro_root.iterdir() if path.is_dir()),
            key=_natural_key,
        ):
            sequence_id = sequence_dir.name
            records.append(
                _base_record(
                    sample_id=f"smic_hs__{subject_id}__{sequence_id}",
                    source_dataset="smic_hs",
                    sample_role="spotting_negative",
                    subject_id=subject_id,
                    source_subject_id=subject_dir.name,
                    local_subject_id=local_subject_id,
                    sequence_id=sequence_id,
                    source_sequence_id=sequence_id,
                    frame_dir=sequence_dir,
                    label=None,
                    label_name="non_micro",
                    action_units=None,
                    evaluation_scope="spotting_engineering_only",
                    source_record=f"directory:{sequence_dir.resolve().as_posix()}",
                )
            )
    return records


def _build_casme_records(
    casme_root: Path, casme_csv: Path
) -> list[dict[str, Any]]:
    rows = _read_csv(casme_csv, {"dataset", "sub", "filename_o", "label"})
    records: list[dict[str, Any]] = []
    for source_row_number, row in enumerate(rows, start=2):
        source_record = f"{casme_csv.name}:{source_row_number}"
        if str(row["dataset"]).strip().lower() != "casme2":
            raise ManifestBuildError(
                f"Unexpected dataset {row['dataset']!r} at {source_record}"
            )
        subject_id = str(row["sub"]).strip().lower()
        if not re.fullmatch(r"sub\d+", subject_id):
            raise ManifestBuildError(
                f"Invalid CASME II subject id {subject_id!r} at {source_record}"
            )
        sequence_id = str(row["filename_o"]).strip()
        if not sequence_id:
            raise ManifestBuildError(f"Empty sequence id at {source_record}")
        label = _parse_label(row["label"], source_record)
        label_name = LABEL_NAMES[label]
        frame_dir = casme_root / subject_id / sequence_id
        records.append(
            _base_record(
                sample_id=f"casme2__{subject_id}__{sequence_id}",
                source_dataset="casme2",
                sample_role="classification",
                subject_id=subject_id,
                source_subject_id=subject_id,
                local_subject_id=subject_id,
                sequence_id=sequence_id,
                source_sequence_id=sequence_id,
                frame_dir=frame_dir,
                label=label,
                label_name=label_name,
                action_units=str(row.get("Action Units") or "").strip() or None,
                evaluation_scope="awaiting_estimated_or_official_frames",
                source_record=source_record,
            )
        )
    return records


def validate_manifest(records: Sequence[Mapping[str, Any]]) -> None:
    sample_ids: set[str] = set()
    for index, record in enumerate(records):
        missing_fields = [field for field in MANIFEST_FIELDS if field not in record]
        if missing_fields:
            raise ManifestBuildError(
                f"Manifest row {index} is missing fields: {missing_fields}"
            )

        sample_id = str(record["sample_id"])
        if not sample_id:
            raise ManifestBuildError(f"Manifest row {index} has an empty sample_id")
        if sample_id in sample_ids:
            raise ManifestBuildError(f"Duplicate sample_id: {sample_id}")
        sample_ids.add(sample_id)

        sample_role = str(record["sample_role"])
        if sample_role not in SAMPLE_ROLES:
            raise ManifestBuildError(
                f"Invalid sample_role {sample_role!r} for {sample_id}"
            )
        if record["frame_annotation_source"] not in FRAME_ANNOTATION_SOURCES:
            raise ManifestBuildError(
                f"Invalid frame_annotation_source for {sample_id}"
            )

        if sample_role == "classification":
            label = record["label"]
            if label not in LABEL_NAMES or record["label_name"] != LABEL_NAMES[label]:
                raise ManifestBuildError(
                    f"Invalid classification label for {sample_id}: "
                    f"{label!r}/{record['label_name']!r}"
                )
        elif record["label"] is not None or record["label_name"] != "non_micro":
            raise ManifestBuildError(
                f"spotting_negative must not enter the class label space: {sample_id}"
            )


def build_microexpression_manifest(
    *,
    casme_root: Path,
    smic_root: Path,
    casme_csv: Path,
    smic_csv: Path,
    include_smic_non_micro: bool = True,
) -> list[dict[str, Any]]:
    casme_root = casme_root.resolve()
    smic_root = smic_root.resolve()
    if not casme_root.is_dir():
        raise FileNotFoundError(casme_root)
    if not smic_root.is_dir():
        raise FileNotFoundError(smic_root)

    records = [
        *_build_casme_records(casme_root, casme_csv.resolve()),
        *_build_smic_classification_records(smic_root, smic_csv.resolve()),
    ]
    if include_smic_non_micro:
        records.extend(_build_smic_non_micro_records(smic_root))
    records.sort(
        key=lambda row: (
            str(row["source_dataset"]),
            str(row["subject_id"]),
            str(row["sample_role"]),
            str(row["sequence_id"]),
        )
    )
    validate_manifest(records)
    return records


def _count_raw_casme(casme_root: Path) -> dict[str, int]:
    subject_dirs = [path for path in casme_root.iterdir() if path.is_dir()]
    sequence_dirs = [
        sequence
        for subject in subject_dirs
        for sequence in subject.iterdir()
        if sequence.is_dir()
    ]
    image_count = sum(
        1
        for sequence in sequence_dirs
        for path in sequence.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    return {
        "subject_dirs": len(subject_dirs),
        "sequence_dirs": len(sequence_dirs),
        "image_files": image_count,
    }


def _count_raw_smic(smic_root: Path) -> dict[str, int]:
    subject_dirs = [path for path in smic_root.iterdir() if path.is_dir()]
    micro_dirs: list[Path] = []
    non_micro_dirs: list[Path] = []
    for subject in subject_dirs:
        micro_root = subject / "micro"
        if micro_root.is_dir():
            for class_dir in micro_root.iterdir():
                if class_dir.is_dir():
                    micro_dirs.extend(path for path in class_dir.iterdir() if path.is_dir())
        non_micro_root = subject / "non_micro"
        if non_micro_root.is_dir():
            non_micro_dirs.extend(
                path for path in non_micro_root.iterdir() if path.is_dir()
            )
    image_count = sum(
        1
        for sequence in (*micro_dirs, *non_micro_dirs)
        for path in sequence.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    return {
        "subject_dirs": len(subject_dirs),
        "micro_sequence_dirs": len(micro_dirs),
        "non_micro_sequence_dirs": len(non_micro_dirs),
        "image_files": image_count,
    }


def _nested_counts(
    records: Iterable[Mapping[str, Any]], fields: tuple[str, ...]
) -> dict[str, Any]:
    if not fields:
        return {}
    groups: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    field = fields[0]
    for record in records:
        value = record[field]
        groups["null" if value is None else str(value)].append(record)
    if len(fields) == 1:
        return {key: len(groups[key]) for key in sorted(groups)}
    return {
        key: _nested_counts(groups[key], fields[1:]) for key in sorted(groups)
    }


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    temporary.replace(path)


def _write_mapping_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PATH_MAPPING_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record[field] for field in PATH_MAPPING_FIELDS})
    temporary.replace(path)


def write_microexpression_manifest(
    *,
    records: Sequence[Mapping[str, Any]],
    output_dir: Path,
    casme_root: Path,
    smic_root: Path,
    casme_csv: Path,
    smic_csv: Path,
) -> dict[str, Any]:
    validate_manifest(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "sequence_manifest_v1.jsonl"
    mapping_path = output_dir / "path_mapping_v1.csv"
    summary_path = output_dir / "quality_summary_v1.json"

    _write_jsonl(manifest_path, records)
    _write_mapping_csv(mapping_path, records)

    excluded_records = [record for record in records if record["exclude_reason"]]
    frame_counts = [int(record["frame_count"]) for record in records]
    extension_frames: Counter[str] = Counter()
    for record in records:
        extension = str(record["frame_extension"] or "missing")
        extension_frames[extension] += int(record["frame_count"])

    summary = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "task_id": "DATA-ME-001",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "paper_reproduction_claim": False,
        "source": {
            "casme_root": casme_root.resolve().as_posix(),
            "smic_root": smic_root.resolve().as_posix(),
            "casme_csv": casme_csv.resolve().as_posix(),
            "casme_csv_sha256": _sha256(casme_csv),
            "smic_csv": smic_csv.resolve().as_posix(),
            "smic_csv_sha256": _sha256(smic_csv),
        },
        "raw_inventory": {
            "casme2": _count_raw_casme(casme_root.resolve()),
            "smic_hs": _count_raw_smic(smic_root.resolve()),
        },
        "manifest": {
            "path": manifest_path.resolve().as_posix(),
            "sha256": _sha256(manifest_path),
            "row_count": len(records),
            "subject_count_by_dataset": {
                dataset: len(
                    {
                        str(record["subject_id"])
                        for record in records
                        if record["source_dataset"] == dataset
                    }
                )
                for dataset in ("casme2", "smic_hs")
            },
            "rows_by_dataset_and_role": _nested_counts(
                records, ("source_dataset", "sample_role")
            ),
            "classification_labels_by_dataset": _nested_counts(
                [
                    record
                    for record in records
                    if record["sample_role"] == "classification"
                ],
                ("source_dataset", "label", "label_name"),
            ),
            "path_status": _nested_counts(records, ("path_status",)),
            "quality_status": _nested_counts(records, ("quality_status",)),
            "frame_annotation_source": _nested_counts(
                records, ("frame_annotation_source",)
            ),
            "frame_count": {
                "total": sum(frame_counts),
                "min": min(frame_counts) if frame_counts else 0,
                "max": max(frame_counts) if frame_counts else 0,
                "by_extension": dict(sorted(extension_frames.items())),
            },
            "excluded_count": len(excluded_records),
            "excluded_sample_ids": [
                str(record["sample_id"]) for record in excluded_records
            ],
        },
        "path_mapping": {
            "path": mapping_path.resolve().as_posix(),
            "sha256": _sha256(mapping_path),
            "row_count": len(records),
        },
        "validation": {
            "sample_ids_unique": len({record["sample_id"] for record in records})
            == len(records),
            "all_frame_dirs_available": all(
                record["path_status"] == "available" for record in records
            ),
            "non_micro_outside_class_label_space": all(
                record["label"] is None
                for record in records
                if record["sample_role"] == "spotting_negative"
            ),
        },
    }
    temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(summary_path)
    summary["summary_path"] = summary_path.resolve().as_posix()
    return summary


def audit_smic_mapping(
    *,
    smic_root: Path,
    smic_csv: Path,
    manifest_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Independently compare SMIC CSV labels, directories, and manifest rows."""
    smic_root = smic_root.resolve()
    rows = _read_csv(smic_csv.resolve(), {"dataset", "sub", "filename_o", "label"})
    smic_manifest = [
        record for record in manifest_records if record["source_dataset"] == "smic_hs"
    ]
    classification_manifest = {
        str(record["sample_id"]): record
        for record in smic_manifest
        if record["sample_role"] == "classification"
    }
    spotting_manifest = {
        str(record["sample_id"]): record
        for record in smic_manifest
        if record["sample_role"] == "spotting_negative"
    }

    issues: list[dict[str, Any]] = []
    csv_ids: set[str] = set()
    csv_id_sources: dict[str, list[str]] = defaultdict(list)
    csv_label_counts: Counter[str] = Counter()
    csv_subject_counts: Counter[str] = Counter()
    mapped_records: list[dict[str, Any]] = []

    for source_row_number, row in enumerate(rows, start=2):
        source_record = f"{smic_csv.name}:{source_row_number}"
        if str(row["dataset"]).strip().lower() != "smic":
            issues.append(
                {
                    "code": "unexpected_dataset",
                    "source_record": source_record,
                    "value": row["dataset"],
                }
            )
            continue

        source_subject_id = str(row["sub"]).strip().lower()
        try:
            subject_id, local_subject_id = normalize_smic_subject(source_subject_id)
        except ManifestBuildError as exc:
            issues.append(
                {
                    "code": "invalid_subject_id",
                    "source_record": source_record,
                    "value": source_subject_id,
                    "detail": str(exc),
                }
            )
            continue

        source_sequence_id = str(row["filename_o"]).strip()
        if not source_sequence_id.lower().startswith(source_subject_id):
            issues.append(
                {
                    "code": "sequence_subject_prefix_mismatch",
                    "source_record": source_record,
                    "source_subject_id": source_subject_id,
                    "source_sequence_id": source_sequence_id,
                }
            )
            continue
        sequence_id = local_subject_id + source_sequence_id[len(source_subject_id) :]
        try:
            label = _parse_label(row["label"], source_record)
        except ManifestBuildError as exc:
            issues.append(
                {
                    "code": "invalid_label",
                    "source_record": source_record,
                    "value": row["label"],
                    "detail": str(exc),
                }
            )
            continue

        sample_id = f"smic_hs__{subject_id}__{sequence_id}"
        csv_ids.add(sample_id)
        csv_id_sources[sample_id].append(source_record)
        csv_label_counts[LABEL_NAMES[label]] += 1
        csv_subject_counts[subject_id] += 1

        expected_dir = smic_root / local_subject_id / "micro" / LABEL_NAMES[label] / sequence_id
        present_class_dirs = [
            class_name
            for class_name in LABEL_NAMES.values()
            if (smic_root / local_subject_id / "micro" / class_name / sequence_id).is_dir()
        ]
        manifest_record = classification_manifest.get(sample_id)
        row_result = {
            "source_record": source_record,
            "sample_id": sample_id,
            "subject_id": subject_id,
            "source_subject_id": source_subject_id,
            "local_subject_id": local_subject_id,
            "source_sequence_id": source_sequence_id,
            "sequence_id": sequence_id,
            "csv_label": label,
            "csv_label_name": LABEL_NAMES[label],
            "expected_frame_dir": expected_dir.resolve().as_posix(),
            "present_class_dirs": present_class_dirs,
            "manifest_present": manifest_record is not None,
            "path_available": expected_dir.is_dir(),
            "label_directory_match": present_class_dirs == [LABEL_NAMES[label]],
        }
        mapped_records.append(row_result)

        if source_sequence_id == sequence_id and local_subject_id != source_subject_id:
            issues.append(
                {
                    "code": "subject_normalization_not_applied",
                    "source_record": source_record,
                    "source_subject_id": source_subject_id,
                    "local_subject_id": local_subject_id,
                }
            )
        if not expected_dir.is_dir():
            issues.append(
                {
                    "code": "missing_expected_micro_dir",
                    "source_record": source_record,
                    "path": expected_dir.resolve().as_posix(),
                }
            )
        if present_class_dirs != [LABEL_NAMES[label]]:
            issues.append(
                {
                    "code": "label_directory_mismatch",
                    "source_record": source_record,
                    "expected_label_name": LABEL_NAMES[label],
                    "present_class_dirs": present_class_dirs,
                }
            )
        if manifest_record is None:
            issues.append(
                {
                    "code": "classification_missing_from_manifest",
                    "source_record": source_record,
                    "sample_id": sample_id,
                }
            )
        else:
            if manifest_record["label"] != label:
                issues.append(
                    {
                        "code": "manifest_label_mismatch",
                        "source_record": source_record,
                        "sample_id": sample_id,
                        "csv_label": label,
                        "manifest_label": manifest_record["label"],
                    }
                )
            if Path(str(manifest_record["frame_dir"])).resolve() != expected_dir.resolve():
                issues.append(
                    {
                        "code": "manifest_path_mismatch",
                        "source_record": source_record,
                        "sample_id": sample_id,
                    }
                )

    duplicate_csv_ids = {
        sample_id: source_records
        for sample_id, source_records in csv_id_sources.items()
        if len(source_records) > 1
    }
    if duplicate_csv_ids:
        issues.append(
            {
                "code": "duplicate_csv_sample_id",
                "samples": duplicate_csv_ids,
            }
        )

    micro_directory_ids: set[str] = set()
    micro_directory_labels: dict[str, list[str]] = defaultdict(list)
    for subject_dir in smic_root.iterdir():
        if not subject_dir.is_dir():
            continue
        try:
            subject_id, local_subject_id = normalize_smic_subject(subject_dir.name)
        except ManifestBuildError:
            continue
        micro_root = subject_dir / "micro"
        if not micro_root.is_dir():
            continue
        for label_name in LABEL_NAMES.values():
            class_root = micro_root / label_name
            if not class_root.is_dir():
                continue
            for sequence_dir in class_root.iterdir():
                if not sequence_dir.is_dir():
                    continue
                sequence_id = sequence_dir.name
                sample_id = f"smic_hs__{subject_id}__{sequence_id}"
                micro_directory_ids.add(sample_id)
                micro_directory_labels[sample_id].append(label_name)

    if micro_directory_ids != csv_ids:
        issues.append(
            {
                "code": "micro_directory_csv_set_mismatch",
                "directory_count": len(micro_directory_ids),
                "csv_count": len(csv_ids),
                "missing_in_csv": sorted(micro_directory_ids - csv_ids),
                "missing_in_directory": sorted(csv_ids - micro_directory_ids),
            }
        )
    duplicated_directory_ids = {
        sample_id: labels
        for sample_id, labels in micro_directory_labels.items()
        if len(labels) != 1
    }
    if duplicated_directory_ids:
        issues.append(
            {
                "code": "micro_directory_duplicate_class",
                "samples": duplicated_directory_ids,
            }
        )

    local_subject_dirs = {
        subject_dir.name
        for subject_dir in smic_root.iterdir()
        if subject_dir.is_dir()
    }
    expected_local_subject_dirs = {
        str(record["local_subject_id"])
        for record in mapped_records
    }
    if local_subject_dirs != expected_local_subject_dirs:
        issues.append(
            {
                "code": "subject_directory_set_mismatch",
                "missing_local_subject_dirs": sorted(
                    expected_local_subject_dirs - local_subject_dirs
                ),
                "unreferenced_local_subject_dirs": sorted(
                    local_subject_dirs - expected_local_subject_dirs
                ),
            }
        )

    non_micro_records = [
        record
        for record in smic_manifest
        if record["sample_role"] == "spotting_negative"
    ]
    non_micro_dirs: set[str] = set()
    for subject_dir in smic_root.iterdir():
        if not subject_dir.is_dir():
            continue
        non_micro_root = subject_dir / "non_micro"
        if not non_micro_root.is_dir():
            continue
        non_micro_dirs.update(
            f"{subject_dir.name}__{sequence_dir.name}"
            for sequence_dir in non_micro_root.iterdir()
            if sequence_dir.is_dir()
        )
    non_micro_manifest_dirs = {
        f"{record['local_subject_id']}__{record['sequence_id']}"
        for record in non_micro_records
    }
    if non_micro_dirs != non_micro_manifest_dirs:
        issues.append(
            {
                "code": "non_micro_manifest_coverage_mismatch",
                "directory_count": len(non_micro_dirs),
                "manifest_count": len(non_micro_manifest_dirs),
                "missing_in_manifest": sorted(non_micro_dirs - non_micro_manifest_dirs),
                "unreferenced_in_directory": sorted(
                    non_micro_manifest_dirs - non_micro_dirs
                ),
            }
        )

    manifest_classification_ids = set(classification_manifest)
    if csv_ids != manifest_classification_ids:
        issues.append(
            {
                "code": "csv_manifest_id_set_mismatch",
                "missing_in_manifest": sorted(csv_ids - manifest_classification_ids),
                "unreferenced_in_csv": sorted(manifest_classification_ids - csv_ids),
            }
        )

    subject_summary: dict[str, Any] = {}
    for subject_id in sorted(csv_subject_counts):
        local_subject_id = next(
            row["local_subject_id"]
            for row in mapped_records
            if row["subject_id"] == subject_id
        )
        micro_count = sum(
            row["subject_id"] == subject_id and row["path_available"]
            for row in mapped_records
        )
        non_micro_count = sum(
            record["subject_id"] == subject_id for record in non_micro_records
        )
        if non_micro_count == 0:
            issues.append(
                {
                    "code": "subject_missing_non_micro",
                    "subject_id": subject_id,
                    "local_subject_id": local_subject_id,
                }
            )
        subject_summary[subject_id] = {
            "local_subject_id": local_subject_id,
            "classification_csv_count": csv_subject_counts[subject_id],
            "classification_path_available_count": micro_count,
            "non_micro_manifest_count": non_micro_count,
            "subject_grouping_key": f"smic_hs::{subject_id}",
            "micro_and_non_micro_same_subject": non_micro_count > 0,
        }

    return {
        "schema_version": "smic_mapping_audit_v1",
        "task_id": "DATA-ME-002",
        "status": "pass" if not issues else "fail",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "smic_root": smic_root.as_posix(),
            "smic_csv": smic_csv.resolve().as_posix(),
            "smic_csv_sha256": _sha256(smic_csv.resolve()),
        },
        "mapping_rules": {
            "canonical_subject_id": "sNN",
            "local_subject_id": "sN without zero padding",
            "csv_to_local_example": "s01 -> s1",
            "classification_labels": LABEL_NAMES,
            "non_micro_sample_role": "spotting_negative",
            "non_micro_enters_three_class_label_space": False,
            "subject_grouping_key": "source_dataset::subject_id",
        },
        "csv": {
            "row_count": len(rows),
            "id_count": len(csv_ids),
            "label_counts": dict(sorted(csv_label_counts.items())),
            "subject_counts": dict(sorted(csv_subject_counts.items())),
        },
        "manifest": {
            "smic_row_count": len(smic_manifest),
            "classification_count": len(classification_manifest),
            "spotting_negative_count": len(spotting_manifest),
            "classification_id_count": len(manifest_classification_ids),
        },
        "directories": {
            "local_subject_count": len(local_subject_dirs),
            "local_subject_dirs": sorted(local_subject_dirs),
            "micro_sequence_dir_count": len(micro_directory_ids),
            "non_micro_dir_count": len(non_micro_dirs),
            "non_micro_manifest_count": len(non_micro_manifest_dirs),
        },
        "subject_summary": subject_summary,
        "row_checks": {
            "mapped_rows": len(mapped_records),
            "path_available_rows": sum(row["path_available"] for row in mapped_records),
            "label_directory_match_rows": sum(
                row["label_directory_match"] for row in mapped_records
            ),
            "manifest_present_rows": sum(
                row["manifest_present"] for row in mapped_records
            ),
        },
        "issues": issues,
        "limitations": [
            "本任务只审计 SMIC 目录、CSV 标签和 manifest 映射，不检查人脸关键点或光流质量。",
            "non_micro 用于 spotting 工程误报调试，不作为三分类第四类训练标签。",
            "subject-level split 尚未生成，当前只冻结 source_dataset::subject_id 分组键。",
        ],
    }


def write_smic_mapping_audit(report: Mapping[str, Any], output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(output_path)
    return _sha256(output_path)
