"""Build the deterministic CogPic V3.3 task manifest and issue report."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import soundfile as sf

try:
    from .common import (
        ASR_MODEL_VERSION,
        DEFAULT_EXCLUSION_REPORT_PATH,
        DEFAULT_MANIFEST_PATH,
        DEFAULT_PROCESSED_ROOT,
        DEFAULT_SPLIT_PATH,
        atomic_write_json,
        atomic_write_parquet,
        load_json,
        resolve_cogpic_root,
        sha256_file,
        workspace_relative,
    )
except ImportError:
    from common import (
        ASR_MODEL_VERSION,
        DEFAULT_EXCLUSION_REPORT_PATH,
        DEFAULT_MANIFEST_PATH,
        DEFAULT_PROCESSED_ROOT,
        DEFAULT_SPLIT_PATH,
        atomic_write_json,
        atomic_write_parquet,
        load_json,
        resolve_cogpic_root,
        sha256_file,
        workspace_relative,
    )


SUBJECT_PATTERN = re.compile(
    r"^(?P<diagnosis>AD|MCI|HC)_subj_(?P<number>\d+)_"
    r"(?P<sex>\d+)_(?P<age>\d+)_(?P<education>\d+)_"
    r"(?P<mmse>NA|\d+)_(?P<moca>NA|\d+)$"
)
TASK_PATTERN = re.compile(r"^pic_(?P<number>[1-9]\d*)$")
DIAGNOSES = ("HC", "MCI", "AD")


@dataclass(frozen=True)
class ExpectedCounts:
    subjects: int = 574
    tasks: int = 1722
    official_train_subjects: int = 459
    official_test_subjects: int = 115


class ManifestBuildError(RuntimeError):
    pass


def build_manifest(
    *,
    dataset_root: str | Path | None = None,
    output_path: str | Path = DEFAULT_MANIFEST_PATH,
    report_path: str | Path = DEFAULT_EXCLUSION_REPORT_PATH,
    split_path: str | Path | None = None,
    expected_counts: ExpectedCounts | None = ExpectedCounts(),
) -> dict[str, Any]:
    root = resolve_cogpic_root(dataset_root)
    output = Path(output_path)
    report_target = Path(report_path)
    split = _load_split(Path(split_path) if split_path else DEFAULT_SPLIT_PATH)
    rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    seen_subjects: dict[str, tuple[str, str]] = {}

    for official_dir, official_split in (("Train", "train"), ("Test", "test")):
        for diagnosis in ("AD", "MCI", "HC"):
            diagnosis_root = root / official_dir / diagnosis
            if not diagnosis_root.is_dir():
                _issue(issues, "error", "missing_diagnosis_directory", str(diagnosis_root))
                continue
            for subject_path in sorted(
                (path for path in diagnosis_root.iterdir() if path.is_dir()),
                key=lambda path: path.name,
            ):
                match = SUBJECT_PATTERN.fullmatch(subject_path.name)
                if match is None:
                    _issue(issues, "error", "invalid_subject_directory", str(subject_path))
                    continue
                values = match.groupdict()
                if values["diagnosis"] != diagnosis:
                    _issue(issues, "error", "diagnosis_directory_mismatch", str(subject_path))
                    continue
                subject_id = f"{diagnosis}_subj_{values['number']}"
                prior = seen_subjects.get(subject_id)
                if prior is not None:
                    _issue(
                        issues,
                        "error",
                        "duplicate_subject_id",
                        str(subject_path),
                        subject_id=subject_id,
                        prior_path=prior[1],
                    )
                    continue
                seen_subjects[subject_id] = (diagnosis, str(subject_path))
                task_paths = sorted(
                    (path for path in subject_path.iterdir() if path.is_dir()),
                    key=lambda path: path.name,
                )
                parsed_tasks: list[tuple[int, Path]] = []
                for task_path in task_paths:
                    task_match = TASK_PATTERN.fullmatch(task_path.name)
                    if task_match is None:
                        _issue(issues, "error", "invalid_task_directory", str(task_path))
                        continue
                    parsed_tasks.append((int(task_match.group("number")), task_path))
                if [number for number, _ in parsed_tasks] != [1, 2, 3]:
                    _issue(
                        issues,
                        "error",
                        "unexpected_task_set",
                        str(subject_path),
                        observed=[number for number, _ in parsed_tasks],
                    )
                    continue
                for task_number, task_path in parsed_tasks:
                    row = _task_row(
                        task_path=task_path,
                        subject_id=subject_id,
                        diagnosis=diagnosis,
                        official_split=official_split,
                        task_number=task_number,
                        subject_values=values,
                        split=split,
                        issues=issues,
                    )
                    if row is not None:
                        rows.append(row)

    rows.sort(key=lambda row: row["sample_id"])
    _validate_rows(rows, issues, expected_counts)
    error_count = sum(issue["severity"] == "error" for issue in issues)
    report = _report(root, rows, issues)
    atomic_write_json(report_target, report)
    if error_count:
        raise ManifestBuildError(
            f"CogPic manifest has {error_count} fatal issue(s); see {report_target}"
        )
    frame = pd.DataFrame(rows)
    atomic_write_parquet(output, frame)
    report["artifacts"] = {
        "manifest_path": str(output.resolve()),
        "manifest_sha256": sha256_file(output),
        "report_path": str(report_target.resolve()),
    }
    atomic_write_json(report_target, report)
    return report


def _task_row(
    *,
    task_path: Path,
    subject_id: str,
    diagnosis: str,
    official_split: str,
    task_number: int,
    subject_values: Mapping[str, str],
    split: Mapping[str, str],
    issues: list[dict[str, Any]],
) -> dict[str, Any] | None:
    sample_id = f"{subject_id}__pic_{task_number}"
    audio_files = tuple(task_path.glob("audio.wav"))
    text_files = tuple(task_path.glob("audio.txt"))
    face_directories = tuple(path for path in task_path.glob("frames_face") if path.is_dir())
    if len(audio_files) != 1 or len(text_files) != 1 or len(face_directories) != 1:
        _issue(
            issues,
            "error",
            "media_mapping_not_unique",
            str(task_path),
            sample_id=sample_id,
            audio_count=len(audio_files),
            text_count=len(text_files),
            face_directory_count=len(face_directories),
        )
        return None
    audio_path = audio_files[0]
    text_path = text_files[0]
    face_path = face_directories[0]
    frame_paths = tuple(sorted(face_path.glob("*.jpg"), key=lambda path: path.name))
    if not frame_paths:
        _issue(
            issues,
            "warning",
            "empty_face_directory",
            str(face_path),
            sample_id=sample_id,
        )
    try:
        audio_info = sf.info(str(audio_path))
        duration_ms = int(round(audio_info.duration * 1000))
    except Exception as exc:
        _issue(
            issues,
            "error",
            "audio_probe_failed",
            str(audio_path),
            sample_id=sample_id,
            message=type(exc).__name__,
        )
        return None
    if not 3000 <= duration_ms <= 60000:
        _issue(
            issues,
            "warning",
            "audio_duration_out_of_contract",
            str(audio_path),
            sample_id=sample_id,
            duration_ms=duration_ms,
        )
    derived_split = split.get(subject_id)
    asr_text_path = DEFAULT_PROCESSED_ROOT / "asr_text" / f"{sample_id}.txt"
    asr_transcript_path = DEFAULT_PROCESSED_ROOT / "asr_transcripts" / f"{sample_id}.json"
    return {
        "sample_id": sample_id,
        "subject_id": subject_id,
        "source_dataset": "CogPic",
        "task_number": task_number,
        "audio_path": workspace_relative(audio_path),
        "original_text_path": workspace_relative(text_path),
        "face_path": workspace_relative(face_path),
        "diagnosis_label": diagnosis,
        "moca_label": _nullable_number(subject_values["moca"]),
        "official_split": official_split,
        "derived_split": derived_split,
        "asr_text_path": workspace_relative(asr_text_path),
        "asr_model_version": ASR_MODEL_VERSION,
        "asr_transcript_path": workspace_relative(asr_transcript_path),
        "asr_status": None,
        "asr_quality": None,
        "exclude_reason": None,
        "audio_sha256": sha256_file(audio_path),
        "original_text_sha256": sha256_file(text_path),
        "audio_duration_ms": duration_ms,
        "audio_sample_rate": int(audio_info.samplerate),
        "audio_channels": int(audio_info.channels),
        "face_frame_count": len(frame_paths),
        "sex_code": int(subject_values["sex"]),
        "age": int(subject_values["age"]),
        "education_code": int(subject_values["education"]),
    }


def _validate_rows(
    rows: list[dict[str, Any]],
    issues: list[dict[str, Any]],
    expected: ExpectedCounts | None,
) -> None:
    sample_counts = Counter(row["sample_id"] for row in rows)
    for sample_id, count in sample_counts.items():
        if count != 1:
            _issue(issues, "error", "duplicate_sample_id", sample_id, count=count)
    subject_labels: dict[str, set[str]] = {}
    for row in rows:
        subject_id = str(row["subject_id"])
        if not subject_id:
            _issue(issues, "error", "empty_subject_id", str(row.get("sample_id")))
        if row["diagnosis_label"] not in DIAGNOSES:
            _issue(issues, "error", "invalid_diagnosis", str(row.get("sample_id")))
        subject_labels.setdefault(subject_id, set()).add(str(row["diagnosis_label"]))
    for subject_id, labels in subject_labels.items():
        if len(labels) != 1:
            _issue(
                issues,
                "error",
                "subject_label_conflict",
                subject_id,
                labels=sorted(labels),
            )
    if expected is None:
        return
    subjects = {row["subject_id"] for row in rows}
    train_subjects = {
        row["subject_id"] for row in rows if row["official_split"] == "train"
    }
    test_subjects = {
        row["subject_id"] for row in rows if row["official_split"] == "test"
    }
    expected_values = {
        "subjects": (len(subjects), expected.subjects),
        "tasks": (len(rows), expected.tasks),
        "official_train_subjects": (len(train_subjects), expected.official_train_subjects),
        "official_test_subjects": (len(test_subjects), expected.official_test_subjects),
    }
    for name, (actual, wanted) in expected_values.items():
        if actual != wanted:
            _issue(
                issues,
                "error",
                "count_mismatch",
                name,
                actual=actual,
                expected=wanted,
            )


def _report(root: Path, rows: list[dict[str, Any]], issues: list[dict[str, Any]]) -> dict[str, Any]:
    subjects = {row["subject_id"] for row in rows}
    return {
        "schema_version": "cogpic_manifest_report_v1",
        "source_dataset": "CogPic",
        "dataset_root": str(root),
        "sample_id_rule": "{diagnosis}_subj_{zero_padded_subject_number}__pic_{task_number}",
        "subject_id_rule": "{diagnosis}_subj_{zero_padded_subject_number}",
        "face_path_rule": "task_directory/frames_face",
        "counts": {
            "subjects": len(subjects),
            "tasks": len(rows),
            "official_train_subjects": len(
                {row["subject_id"] for row in rows if row["official_split"] == "train"}
            ),
            "official_test_subjects": len(
                {row["subject_id"] for row in rows if row["official_split"] == "test"}
            ),
            "wav": len(rows),
            "txt": len(rows),
            "jpg": sum(int(row["face_frame_count"]) for row in rows),
            "errors": sum(issue["severity"] == "error" for issue in issues),
            "warnings": sum(issue["severity"] == "warning" for issue in issues),
        },
        "diagnosis_subject_counts": dict(
            sorted(
                Counter(
                    label
                    for label in {
                        row["subject_id"]: row["diagnosis_label"] for row in rows
                    }.values()
                ).items()
            )
        ),
        "audio_format": {
            "sample_rates": dict(sorted(Counter(row["audio_sample_rate"] for row in rows).items())),
            "channels": dict(sorted(Counter(row["audio_channels"] for row in rows).items())),
        },
        "issues": issues,
    }


def _load_split(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    payload = load_json(path)
    mapping: dict[str, str] = {}
    for split_name, subject_ids in payload.get("splits", {}).items():
        for subject_id in subject_ids:
            mapping[str(subject_id)] = str(split_name)
    return mapping


def _nullable_number(value: str) -> float | None:
    return None if value == "NA" else float(value)


def _issue(
    issues: list[dict[str, Any]],
    severity: str,
    code: str,
    path: str,
    **details: Any,
) -> None:
    portable_path = path
    candidate = Path(path)
    if candidate.exists():
        try:
            portable_path = workspace_relative(candidate)
        except ValueError:
            portable_path = str(candidate)
    issues.append(
        {
            "severity": severity,
            "code": code,
            "path": portable_path,
            **details,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_EXCLUSION_REPORT_PATH)
    parser.add_argument("--split", type=Path)
    parser.add_argument("--allow-count-mismatch", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_manifest(
        dataset_root=args.dataset_root,
        output_path=args.output,
        report_path=args.report,
        split_path=args.split,
        expected_counts=None if args.allow_count_mismatch else ExpectedCounts(),
    )
    print(json.dumps(report["counts"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
