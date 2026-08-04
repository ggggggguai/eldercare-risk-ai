from __future__ import annotations

import argparse
from pathlib import Path

from elderly_monitoring.modules.mental_health.wandering.tcn_baseline import (
    benchmark_tcn_runtime,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark trusted primary-seed pure-TCN models at CPU batch=1 without changing the deterministic bundle."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/modules/wandering_tcn_v1.yaml"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--development-dir",
        type=Path,
        default=Path("reports/mental_health/wandering_step6/development/v1"),
    )
    parser.add_argument("--expected-development-manifest-sha256", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/mental_health/wandering_step6/runtime_benchmark.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    benchmark_tcn_runtime(
        tcn_config_path=args.config,
        project_root=args.project_root,
        development_dir=args.development_dir,
        expected_development_manifest_sha256=args.expected_development_manifest_sha256,
        output_path=args.output,
    )
    print(f"runtime_benchmark={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
