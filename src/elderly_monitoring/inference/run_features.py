from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

from elderly_monitoring.modules.fall_risk import FallRiskPipeline
from elderly_monitoring.modules.mental_health import MentalHealthRiskPipeline
from elderly_monitoring.modules.mental_health.offline import (
    load_json_or_jsonl_records,
    load_jsonl_records,
    render_jsonl,
    run_daily_mental_health,
    write_jsonl_atomic,
)


def load_sample(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError("Input feature file must contain a JSON object.")
    return data


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run risk inference from a feature JSON file.")
    parser.add_argument("--module", choices=["fall_risk", "mental_health"], required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--history-behavior", type=Path)
    parser.add_argument("--current-behavior", type=Path)
    parser.add_argument("--sleep", type=Path)
    parser.add_argument("--self-report", type=Path)
    parser.add_argument("--evaluation-time")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    daily_requested = args.history_behavior is not None or args.current_behavior is not None
    try:
        if args.module == "mental_health" and daily_requested:
            if args.input is not None:
                raise ValueError("--input cannot be combined with daily mental-health inputs")
            if args.history_behavior is None or args.current_behavior is None:
                raise ValueError(
                    "daily mental-health inference requires --history-behavior and --current-behavior"
                )
            outputs = run_daily_mental_health(
                history_behavior=load_jsonl_records(args.history_behavior),
                current_behavior=load_jsonl_records(args.current_behavior),
                sleep=(
                    load_json_or_jsonl_records(args.sleep)
                    if args.sleep is not None
                    else None
                ),
                self_report=(
                    load_json_or_jsonl_records(args.self_report)
                    if args.self_report is not None
                    else None
                ),
                evaluation_time=args.evaluation_time,
            )
            payload = render_jsonl(outputs)
            if args.output is None:
                sys.stdout.write(payload)
            else:
                write_jsonl_atomic(args.output, payload)
            return 0

        if daily_requested or any(
            value is not None
            for value in (args.sleep, args.self_report, args.evaluation_time, args.output)
        ):
            raise ValueError("daily input options are only valid for daily mental-health inference")
        if args.input is None:
            raise ValueError("--input is required for single-feature inference")

        sample = load_sample(args.input)
        if args.module == "fall_risk":
            event = FallRiskPipeline().predict_from_features(sample)
        else:
            event = MentalHealthRiskPipeline().predict_from_features(sample)

        print(json.dumps(event.to_dict(), ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
