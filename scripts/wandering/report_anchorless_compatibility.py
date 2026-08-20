"""Report clean-to-corrupted drift for the four frozen primary-seed models."""

from __future__ import annotations

import argparse
import json

from elderly_monitoring.modules.mental_health.wandering.compatibility_report import (
    build_compatibility_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--augmentation-bundle", required=True)
    parser.add_argument("--expected-augmentation-manifest-sha256", required=True)
    parser.add_argument("--visual-review-dir", required=True)
    parser.add_argument("--expected-visual-manifest-sha256", required=True)
    parser.add_argument("--expected-human-review-sha256", required=True)
    parser.add_argument("--rf-development-dir", required=True)
    parser.add_argument("--expected-rf-development-manifest-sha256", required=True)
    parser.add_argument("--tcn-development-dir", required=True)
    parser.add_argument("--expected-tcn-development-manifest-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = build_compatibility_report(
        config_path=args.config,
        project_root=args.project_root,
        augmentation_bundle=args.augmentation_bundle,
        expected_augmentation_manifest_sha256=args.expected_augmentation_manifest_sha256,
        visual_review_dir=args.visual_review_dir,
        expected_visual_manifest_sha256=args.expected_visual_manifest_sha256,
        expected_human_review_sha256=args.expected_human_review_sha256,
        rf_development_dir=args.rf_development_dir,
        expected_rf_development_manifest_sha256=args.expected_rf_development_manifest_sha256,
        tcn_development_dir=args.tcn_development_dir,
        expected_tcn_development_manifest_sha256=args.expected_tcn_development_manifest_sha256,
        output_dir=args.output,
    )
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "manifest_sha256": result.manifest_sha256,
                "prediction_count": result.prediction_count,
                "metric_group_count": result.metric_group_count,
                "warning_count": result.warning_count,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
