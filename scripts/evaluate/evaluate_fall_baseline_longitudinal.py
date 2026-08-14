from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

import yaml

from elderly_monitoring.modules.fall_risk.baseline_longitudinal import (
    LongitudinalConfigError,
    LongitudinalDataError,
    evaluate_longitudinal_ablation,
    write_longitudinal_evaluation_bundle,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSON object required at {path}:{line_number}")
        rows.append(value)
    return rows


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the fixed four-variant longitudinal baseline ablation."
    )
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--risk-labels", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/evaluation/fall_baseline_longitudinal_v1.provisional.yaml"),
    )
    parser.add_argument("--partition", choices=("validation", "test"), default="validation")
    parser.add_argument(
        "--test-release-ack",
        type=Path,
        help="Custodian acknowledgement required for a frozen test run.",
    )
    parser.add_argument("--data-status", choices=("synthetic", "development", "formal"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _validate_data_status(
    *,
    data_status: str,
    partition: str,
    protocol: dict[str, Any],
    split_metadata: dict[str, Any],
) -> None:
    protocol_status = protocol.get("protocol_status")
    split_status = split_metadata.get("status")
    if partition == "test" and data_status != "formal":
        raise ValueError("test partition requires --data-status formal")
    if data_status == "formal" and (
        protocol_status != "frozen" or split_status != "frozen"
    ):
        raise ValueError("formal data status requires a frozen protocol and frozen split")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        protocol = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        if not isinstance(protocol, dict):
            raise ValueError("evaluation config must be a YAML mapping")
        split_metadata = json.loads(args.split.read_text(encoding="utf-8"))
        if not isinstance(split_metadata, dict):
            raise ValueError("split metadata must be a JSON object")
        _validate_data_status(
            data_status=args.data_status,
            partition=args.partition,
            protocol=protocol,
            split_metadata=split_metadata,
        )
        test_release_ack = (
            json.loads(args.test_release_ack.read_text(encoding="utf-8"))
            if args.test_release_ack is not None
            else None
        )
        if test_release_ack is not None and not isinstance(test_release_ack, dict):
            raise ValueError("test release acknowledgement must be a JSON object")
        code_commit = None
        if args.data_status == "formal":
            code_commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            dirty = subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            if dirty:
                raise ValueError("formal evaluation requires a clean Git worktree")
        result = evaluate_longitudinal_ablation(
            _read_jsonl(args.observations),
            _read_jsonl(args.risk_labels),
            _read_jsonl(args.assignments),
            _read_jsonl(args.predictions),
            protocol,
            partition=args.partition,
            split_metadata=split_metadata,
            test_release_ack=test_release_ack,
            code_commit=code_commit,
        )
        write_longitudinal_evaluation_bundle(
            result,
            args.output_dir,
            metadata={
                "data_status": args.data_status,
                "config_path": str(args.config),
                "config_file_sha256": _file_sha256(args.config),
                "observations_file_sha256": _file_sha256(args.observations),
                "risk_labels_file_sha256": _file_sha256(args.risk_labels),
                "assignments_file_sha256": _file_sha256(args.assignments),
                "split_file_sha256": _file_sha256(args.split),
                "predictions_path": str(args.predictions),
                "predictions_file_sha256": _file_sha256(args.predictions),
                "code_commit": code_commit,
            },
            overwrite=args.overwrite,
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

    print(json.dumps(result.metrics_by_variant, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
