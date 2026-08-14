from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import yaml

from elderly_monitoring.modules.fall_risk.baseline_longitudinal import (
    LongitudinalConfigError,
    LongitudinalDataError,
    build_longitudinal_split,
    write_longitudinal_split,
)


def _read_jsonl(path: Path, *, missing_as_empty: bool = False) -> list[dict[str, Any]]:
    if missing_as_empty and not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSON object required at {path}:{line_number}")
        rows.append(value)
    return rows


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _file_sha256(path: Path, *, missing_as_empty: bool = False) -> str:
    if missing_as_empty and not path.is_file():
        return hashlib.sha256(b"").hexdigest()
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the leakage-safe forward split for longitudinal personal baselines."
    )
    parser.add_argument(
        "--observations",
        type=Path,
        default=Path("data/annotations/fall_risk/longitudinal_observations_v1.jsonl"),
    )
    parser.add_argument(
        "--risk-labels",
        type=Path,
        default=Path("data/annotations/fall_risk/risk_labels.jsonl"),
    )
    parser.add_argument(
        "--subject-profiles",
        type=Path,
        default=Path("data/annotations/fall_risk/subject_profiles.json"),
    )
    parser.add_argument(
        "--review-log",
        type=Path,
        default=Path("data/annotations/fall_risk/longitudinal_risk_review_log_v1.jsonl"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/evaluation/fall_baseline_longitudinal_v1.provisional.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/splits/fall_risk/longitudinal_baseline_v1_phase2"),
    )
    parser.add_argument(
        "--formal-validation-report",
        type=Path,
        help="Required formal v2 validation report when protocol_status is frozen.",
    )
    parser.add_argument("--overwrite-development", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        protocol = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        if not isinstance(protocol, dict):
            raise ValueError("evaluation config must be a YAML mapping")
        formal_report = (
            _read_object(args.formal_validation_report)
            if args.formal_validation_report is not None
            else None
        )
        formal_input_sha256 = {
            "risk_labels": _file_sha256(args.risk_labels, missing_as_empty=True),
            "subject_profiles": _file_sha256(args.subject_profiles),
        }
        artifact = build_longitudinal_split(
            _read_jsonl(args.observations, missing_as_empty=True),
            _read_jsonl(args.risk_labels, missing_as_empty=True),
            _read_object(args.subject_profiles),
            _read_jsonl(args.review_log, missing_as_empty=True),
            protocol,
            formal_validation_report=formal_report,
            formal_input_sha256=formal_input_sha256,
        )
        write_longitudinal_split(
            artifact,
            args.output_dir,
            overwrite=args.overwrite_development,
        )
    except (
        FileNotFoundError,
        OSError,
        ValueError,
        json.JSONDecodeError,
        yaml.YAMLError,
        LongitudinalConfigError,
        LongitudinalDataError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(artifact["metadata"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if artifact["metadata"]["status"] != "blocked" else 3


if __name__ == "__main__":
    raise SystemExit(main())
