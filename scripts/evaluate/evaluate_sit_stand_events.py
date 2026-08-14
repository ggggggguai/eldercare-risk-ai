from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import yaml

from elderly_monitoring.modules.fall_risk.sit_stand_event_evaluation import (
    evaluate_sit_stand_events,
    write_sit_stand_evaluation_bundle,
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
        description="Evaluate continuous sit-stand events under one-to-one matching."
    )
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument(
        "--partition", choices=("train", "validation"), default="validation"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/evaluation/sit_stand_event_v1.provisional.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("evaluation config must be a YAML mapping")
        assignment_rows = _read_jsonl(args.assignments)
        allowed_label_ids = {
            str(row["label_id"])
            for row in assignment_rows
            if row.get("partition") == args.partition
        }
        truth_rows = [
            row
            for row in _read_jsonl(args.ground_truth)
            if str(row.get("label_id")) in allowed_label_ids
        ]
        if not truth_rows:
            raise ValueError(
                f"no ground truth is assigned to development partition {args.partition}"
            )
        if any(row.get("partition") == "test" for row in assignment_rows):
            raise ValueError("development evaluation assignments must not expose test rows")
        result = evaluate_sit_stand_events(
            truth_rows,
            _read_jsonl(args.predictions),
            config=config,
        )
        write_sit_stand_evaluation_bundle(result, args.output_dir)
    except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
