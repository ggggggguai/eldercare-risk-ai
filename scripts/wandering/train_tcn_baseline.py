from __future__ import annotations

import argparse
from pathlib import Path

from elderly_monitoring.modules.mental_health.wandering.tcn_baseline import (
    build_tcn_development_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the comparison_only two-task pure-TCN development bundle on CPU. "
            "Only train is optimized; validation is checkpoint-selection/reporting only."
        )
    )
    parser.add_argument("--config", type=Path, default=Path("configs/modules/wandering_tcn_v1.yaml"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/mental_health/wandering_step6/development/v1"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_tcn_development_artifacts(
        tcn_config_path=args.config,
        project_root=args.project_root,
        output_dir=args.output,
    )
    print(f"development_manifest_sha256={result.manifest_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
