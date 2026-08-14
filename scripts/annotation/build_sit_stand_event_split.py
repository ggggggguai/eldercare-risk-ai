from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from elderly_monitoring.modules.fall_risk.sit_stand_event_labels import (
    validate_sit_stand_event_labels,
)
from elderly_monitoring.modules.fall_risk.sit_stand_event_split import (
    build_sit_stand_event_split,
)


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the provisional leakage-protected sit-stand event split."
    )
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--review-log", type=Path, required=True)
    parser.add_argument(
        "--source-assignments",
        type=Path,
        default=Path("data/splits/fall_risk/training_labels_v3/assignments.jsonl"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/fall_risk_video_manifest.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/splits/fall_risk/sit_stand_event_v1"),
    )
    parser.add_argument(
        "--materialized-samples",
        type=Path,
        help="Optional development samples.jsonl used only for pose-availability balancing.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.output_dir.exists():
            raise FileExistsError(f"split output already exists: {args.output_dir}")
        labels = _read_jsonl(args.labels)
        validate_sit_stand_event_labels(labels, _read_jsonl(args.review_log))
        materializable_label_ids = None
        if args.materialized_samples is not None:
            materializable_label_ids = {
                str(row["label_id"])
                for row in _read_jsonl(args.materialized_samples)
                if isinstance(row.get("label_id"), str)
            }
        result = build_sit_stand_event_split(
            labels,
            _read_jsonl(args.source_assignments),
            _read_jsonl(args.manifest),
            materializable_label_ids=materializable_label_ids,
        )
        args.output_dir.mkdir(parents=True)
        (args.output_dir / "assignments.jsonl").write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in result["assignments"]
            ),
            encoding="utf-8",
        )
        (args.output_dir / "split.json").write_text(
            json.dumps(result["split"], ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result["split"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
