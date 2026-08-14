from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Sequence

from elderly_monitoring.modules.fall_risk.sit_stand_event_publishing import (
    publish_sit_stand_event_labels,
)


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSON object required at {path}:{line_number}")
        rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Publish sit-stand event/background intervals from project-owner-confirmed "
            "human v3 action annotations; never infers unlabelled background or test truth."
        )
    )
    parser.add_argument(
        "--action-labels",
        type=Path,
        default=Path("data/annotations/fall_risk/action_labels_v3.jsonl"),
    )
    parser.add_argument(
        "--assignments",
        type=Path,
        default=Path("data/splits/fall_risk/training_labels_v3/assignments.jsonl"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/fall_risk_video_manifest.jsonl"),
    )
    parser.add_argument(
        "--decision",
        type=Path,
        default=Path(
            "configs/data/fall_risk_sit_stand_training_decision_20260810.json"
        ),
    )
    parser.add_argument(
        "--labels-output",
        type=Path,
        default=Path("data/annotations/fall_risk/sit_stand_event_labels_v1.jsonl"),
    )
    parser.add_argument(
        "--review-log-output",
        type=Path,
        default=Path("data/annotations/fall_risk/sit_stand_event_review_log_v1.jsonl"),
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("reports/fall_risk/sit-stand-event-label-publication-v1.json"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outputs = (args.labels_output, args.review_log_output, args.report_output)
    try:
        existing = [path for path in outputs if path.exists()]
        if existing and not args.overwrite:
            raise FileExistsError(
                "sit-stand publication output already exists: "
                + ", ".join(str(path) for path in existing)
            )
        result = publish_sit_stand_event_labels(
            _read_jsonl(args.action_labels),
            _read_jsonl(args.assignments),
            _read_jsonl(args.manifest),
            decision=_read_json(args.decision),
            source_action_labels_sha256=_sha256(args.action_labels),
        )
        _write_jsonl(args.labels_output, result["labels"])
        _write_jsonl(args.review_log_output, result["review_log"])
        args.report_output.parent.mkdir(parents=True, exist_ok=True)
        args.report_output.write_text(
            json.dumps(result["report"], ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result["report"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
