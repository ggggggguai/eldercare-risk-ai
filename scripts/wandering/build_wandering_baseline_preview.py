from __future__ import annotations

import argparse
from pathlib import Path

from elderly_monitoring.modules.mental_health.wandering.camera_baseline_preview import (
    build_wandering_baseline_preview,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a synthetic WanderingBaselineProfilePreview from one or more "
            "fully validated MVP-2S daily bundles."
        )
    )
    parser.add_argument(
        "--daily-bundle",
        type=Path,
        action="append",
        required=True,
        help="Repeat once for each fully validated MVP-2S daily bundle.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = build_wandering_baseline_preview(args.daily_bundle, args.output)
    print(
        f"Built WanderingBaselineProfilePreview at {result.output_dir} "
        f"(daily_bundles={result.input_bundle_count}, "
        f"daily_rows={result.input_daily_row_count}, people={result.person_count}, "
        f"profiles={result.profile_count}, manifest_sha256={result.manifest_sha256})"
    )


if __name__ == "__main__":
    main()
