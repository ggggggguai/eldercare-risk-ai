from __future__ import annotations

import argparse
from pathlib import Path

from elderly_monitoring.modules.mental_health.wandering.rf_baseline import (
    build_frozen_wp_test_artifacts,
)

from train_rf_baseline import _write_supplementary_reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate manifest-bound train-only RF models on the frozen 240-row WP public shape benchmark. "
            "This command cannot read SmartCare official validation and cannot retrain."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/modules/wandering_rf_v1.yaml"),
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--development-dir",
        type=Path,
        default=Path("reports/mental_health/wandering_step5/development/v1"),
    )
    parser.add_argument(
        "--expected-development-manifest-sha256",
        required=True,
        help="Externally supplied SHA-256 trust root printed by train_rf_baseline.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/mental_health/wandering_step5/public_shape_benchmark/v1"),
    )
    parser.add_argument(
        "--report-root",
        type=Path,
        default=Path("reports/mental_health/wandering_step5"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_frozen_wp_test_artifacts(
        rf_config_path=args.config,
        project_root=args.project_root,
        development_dir=args.development_dir,
        expected_development_manifest_sha256=args.expected_development_manifest_sha256,
        output_dir=args.output,
    )
    _write_supplementary_reports(
        report_root=args.report_root,
        runtime=None,
        metrics=result.test_metrics,
        failures=result.failure_cases,
        phase="public_shape_benchmark",
    )
    print(f"public_shape_benchmark_manifest_sha256={result.manifest_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
