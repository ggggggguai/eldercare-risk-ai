from __future__ import annotations

import argparse
from pathlib import Path

from elderly_monitoring.modules.mental_health.wandering.camera_baseline_deviation_preview import (
    build_wandering_baseline_deviation_preview,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a synthetic WanderingBaselineDeviationPreview from one fully "
            "validated MVP-3S v2 baseline bundle and one or more fully validated "
            "MVP-2S observation daily bundles."
        )
    )
    parser.add_argument(
        "--baseline-bundle",
        type=Path,
        required=True,
        help="Fully validated MVP-3S v2 baseline bundle.",
    )
    parser.add_argument(
        "--observation-daily-bundle",
        type=Path,
        action="append",
        required=True,
        help="Repeat once for each fully validated MVP-2S observation daily bundle.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = build_wandering_baseline_deviation_preview(
        args.baseline_bundle,
        args.observation_daily_bundle,
        args.output,
    )
    print(
        f"Built WanderingBaselineDeviationPreview at {result.output_dir} "
        f"(observation_bundles={result.observation_bundle_count}, "
        f"observation_rows={result.observation_row_count}, people={result.person_count}, "
        f"previews={result.preview_count}, manifest_sha256={result.manifest_sha256})"
    )


if __name__ == "__main__":
    main()
