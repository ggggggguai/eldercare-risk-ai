from __future__ import annotations

import argparse
from pathlib import Path

from elderly_monitoring.modules.mental_health.wandering.camera_daily_summary import (
    build_wandering_daily_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a synthetic WanderingDailySummary from fully validated MVP-1 "
            "products and an explicit person/session/time/presence binding."
        )
    )
    parser.add_argument(
        "--product-bundle",
        type=Path,
        action="append",
        required=True,
        help="Repeat once for each fully validated MVP-1 product bundle.",
    )
    parser.add_argument("--binding-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = build_wandering_daily_summary(
        args.product_bundle,
        args.binding_manifest,
        args.output,
    )
    print(
        f"Built WanderingDailySummary at {result.output_dir} "
        f"(products={result.input_product_count}, sessions={result.session_count}, "
        f"people={result.person_count}, dates={result.local_date_count}, "
        f"rows={result.summary_count}, manifest_sha256={result.manifest_sha256})"
    )


if __name__ == "__main__":
    main()
