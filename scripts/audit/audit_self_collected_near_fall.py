from __future__ import annotations

import argparse
from pathlib import Path

from elderly_monitoring.modules.fall_risk.self_collected_scf import (
    evaluate_scf_near_fall_gate,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit whether SCF_MVP_V1 near-fall E1-E3 may enter loss."
    )
    parser.add_argument("--import-report", type=Path, required=True)
    parser.add_argument("--near-fall-candidates", type=Path, required=True)
    parser.add_argument("--baseline-metrics", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_scf_near_fall_gate(
        import_report_path=args.import_report,
        near_fall_candidates_path=args.near_fall_candidates,
        output_path=args.output,
        baseline_runs=args.baseline_metrics,
    )
    print(f"status={report['status']} output={args.output}")


if __name__ == "__main__":
    main()
