from __future__ import annotations

import argparse
from pathlib import Path

from elderly_monitoring.modules.mental_health.wandering.tcn_baseline import (
    build_tcn_frozen_wp_test_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate trusted train-only TCN checkpoints on the fixed 240-row WP shape benchmark. "
            "This command cannot retrain and has no SmartCare official input."
        )
    )
    parser.add_argument("--config", type=Path, default=Path("configs/modules/wandering_tcn_v1.yaml"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--development-dir",
        type=Path,
        default=Path("reports/mental_health/wandering_step6/development/v1"),
    )
    parser.add_argument("--expected-development-manifest-sha256", required=True)
    parser.add_argument("--expected-rf-public-manifest-sha256", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/mental_health/wandering_step6/public_shape_benchmark/v1"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_tcn_frozen_wp_test_artifacts(
        tcn_config_path=args.config,
        project_root=args.project_root,
        development_dir=args.development_dir,
        expected_development_manifest_sha256=args.expected_development_manifest_sha256,
        expected_rf_public_manifest_sha256=args.expected_rf_public_manifest_sha256,
        output_dir=args.output,
    )
    print(f"public_shape_benchmark_manifest_sha256={result.manifest_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
