from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import yaml

from elderly_monitoring.modules.fall_risk.baseline_supervised import (
    SupervisedCandidateConfig,
    train_longitudinal_logistic_candidate,
)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the provisional Phase 3 longitudinal Logistic candidate.")
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--risk-labels", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--regularization-c", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    split = json.loads(args.split.read_text(encoding="utf-8"))
    report = train_longitudinal_logistic_candidate(
        _jsonl(args.observations),
        _jsonl(args.risk_labels),
        _jsonl(args.assignments),
        split,
        protocol,
        output_dir=args.output_dir,
        candidate_config=SupervisedCandidateConfig(
            regularization_c=args.regularization_c,
            max_iter=args.max_iter,
        ),
        overwrite=args.overwrite,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 3 if report.get("status") == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
