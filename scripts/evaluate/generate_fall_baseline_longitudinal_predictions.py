from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import yaml

from elderly_monitoring.modules.fall_risk.baseline import load_baseline_config
from elderly_monitoring.modules.fall_risk.baseline_longitudinal import (
    LongitudinalConfigError,
    LongitudinalDataError,
    generate_longitudinal_ablation_predictions,
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


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the four outcome-blind longitudinal baseline ablation predictions."
    )
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument(
        "--evaluation-config",
        type=Path,
        default=Path("configs/evaluation/fall_baseline_longitudinal_v1.provisional.yaml"),
    )
    parser.add_argument(
        "--baseline-config",
        type=Path,
        default=Path("configs/modules/fall_risk_baseline.yaml"),
    )
    parser.add_argument("--partition", choices=("train", "validation"), default="validation")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        protocol = yaml.safe_load(args.evaluation_config.read_text(encoding="utf-8"))
        if not isinstance(protocol, dict):
            raise ValueError("evaluation config must be a YAML mapping")
        predictions = generate_longitudinal_ablation_predictions(
            _read_jsonl(args.observations),
            _read_jsonl(args.assignments),
            _read_object(args.split),
            protocol,
            partition=args.partition,
            baseline_config=load_baseline_config(args.baseline_config),
        )
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in predictions
        )
        args.output.write_text(payload, encoding="utf-8")
    except (
        FileExistsError,
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
    print(json.dumps({"prediction_count": len(predictions), "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
