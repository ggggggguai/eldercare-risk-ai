"""Deterministic publication of source-specific v2 fall-risk label candidates."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


PUBLISH_SCHEMA_VERSION = "fall-risk-label-publish-v2"
ROOT_ISOLATED_BATCH_IDS: frozenset[str] = frozenset()


def publish_v2_labels(
    source_root: Path | str,
    *,
    action_output: Path | str,
    event_output: Path | str,
    report_output: Path | str,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Merge v2 source candidates into the controlled root label files.

    Official LE2I TXT fall windows take precedence over overlapping CVAT-mapped
    fall events. Other CVAT-mapped events remain available for task-specific
    analysis. All source rows and exclusions are recorded in the publish report.
    """
    source_root = Path(source_root)
    action_output = Path(action_output)
    event_output = Path(event_output)
    report_output = Path(report_output)
    outputs = (action_output, event_output, report_output)
    if not overwrite:
        existing = [str(path) for path in outputs if path.exists()]
        if existing:
            raise FileExistsError(f"refusing to overwrite existing outputs: {existing}")

    all_action_files = sorted(
        path for path in source_root.glob("*/action_labels.jsonl")
    )
    all_event_files = sorted(
        path for path in source_root.glob("*/event_labels.jsonl")
    )
    source_files = [
        path
        for path in all_action_files
        if path.parent.name not in ROOT_ISOLATED_BATCH_IDS
    ]
    event_files = [
        path
        for path in all_event_files
        if path.parent.name not in ROOT_ISOLATED_BATCH_IDS
    ]
    excluded_action_files = [
        path for path in all_action_files if path.parent.name in ROOT_ISOLATED_BATCH_IDS
    ]
    excluded_event_files = [
        path for path in all_event_files if path.parent.name in ROOT_ISOLATED_BATCH_IDS
    ]
    if not source_files:
        raise FileNotFoundError(f"no v2 action label files under {source_root}")
    if not event_files:
        raise FileNotFoundError(f"no v2 event label files under {source_root}")

    action_rows = _read_rows(source_files)
    event_rows = _read_rows(event_files)
    excluded_action_count = len(_read_rows(excluded_action_files))
    excluded_event_count = len(_read_rows(excluded_event_files))
    _ensure_unique_ids(action_rows, "action")
    _ensure_unique_ids(event_rows, "event")

    official_events = [row for row in event_rows if row.get("label_source") == "le2i_txt"]
    cvat_events = [row for row in event_rows if row.get("label_source") != "le2i_txt"]
    excluded: list[dict[str, Any]] = []
    retained_events: list[dict[str, Any]] = []
    for row in cvat_events:
        if row.get("event_type") == "fall" and _overlaps_official(row, official_events):
            excluded.append(
                {
                    "label_id": row.get("label_id"),
                    "video_id": row.get("video_id"),
                    "source_record_id": row.get("source_record_id"),
                    "reason": "official_le2i_window_precedence",
                }
            )
        else:
            retained_events.append(row)

    actions = _sorted_rows(action_rows)
    events = _sorted_rows([*retained_events, *official_events])
    _write_jsonl(action_output, actions)
    _write_jsonl(event_output, events)
    report = {
        "schema_version": PUBLISH_SCHEMA_VERSION,
        "source_root": source_root.as_posix(),
        "source_files": [
            {"path": path.as_posix(), "sha256": _sha256_file(path)}
            for path in [*source_files, *event_files]
        ],
        "excluded_source_batches": [
            {
                "batch_id": batch_id,
                "reason": "source_isolated",
            }
            for batch_id in sorted(
                {
                    path.parent.name
                    for path in [*excluded_action_files, *excluded_event_files]
                }
            )
        ],
        "precedence": "le2i_txt_overlapping_fall_over_cvat_action_mapping",
        "input_counts": {
            "action_labels": len(action_rows),
            "event_labels": len(event_rows),
            "official_le2i_events": len(official_events),
            "cvat_events": len(cvat_events),
            "excluded_source_isolated_action_labels": excluded_action_count,
            "excluded_source_isolated_event_labels": excluded_event_count,
        },
        "output_counts": {
            "action_labels": len(actions),
            "event_labels": len(events),
            "excluded_overlapping_cvat_fall_events": len(excluded),
        },
        "excluded_events": excluded,
        "output_sha256": {
            "action_labels": _sha256_file(action_output),
            "event_labels": _sha256_file(event_output),
        },
    }
    _write_json(report_output, report)
    return report


def _read_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: JSONL row must be an object")
            rows.append(value)
    return rows


def _ensure_unique_ids(rows: Iterable[Mapping[str, Any]], kind: str) -> None:
    counts = Counter(str(row.get("label_id", "")) for row in rows)
    duplicated = sorted(label_id for label_id, count in counts.items() if not label_id or count > 1)
    if duplicated:
        raise ValueError(f"duplicate or missing {kind} label_id values: {duplicated[:5]}")


def _overlaps_official(row: Mapping[str, Any], official_rows: Iterable[Mapping[str, Any]]) -> bool:
    video_id = row.get("video_id")
    start = float(row["start_time"])
    end = float(row["end_time"])
    return any(
        video_id == other.get("video_id")
        and min(end, float(other["end_time"])) >= max(start, float(other["start_time"]))
        for other in official_rows
    )


def _sorted_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            str(row.get("video_id", "")),
            float(row.get("start_time", 0.0)),
            float(row.get("end_time", 0.0)),
            str(row.get("label_id", "")),
        ),
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
