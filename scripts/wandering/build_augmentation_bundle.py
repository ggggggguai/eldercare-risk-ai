"""Build the deterministic wandering step-8 augmentation bundle."""

from __future__ import annotations

import argparse
import json

from elderly_monitoring.modules.mental_health.wandering.augmentation import (
    build_augmentation_bundle,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = build_augmentation_bundle(
        config_path=args.config,
        project_root=args.project_root,
        output_dir=args.output,
    )
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "manifest_sha256": result.manifest_sha256,
                "attempt_count": result.attempt_count,
                "train_pair_count": result.train_pair_count,
                "validation_pressure_count": result.validation_pressure_count,
                "gated_fault_count": result.gated_fault_count,
                "limitation_control_count": result.limitation_control_count,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
