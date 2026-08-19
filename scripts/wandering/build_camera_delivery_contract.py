from __future__ import annotations

import argparse
import json
from pathlib import Path

from elderly_monitoring.modules.mental_health.wandering.camera_delivery_contract import (
    build_wandering_camera_delivery_contract,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/modules/wandering_camera_5d_contract_v1.yaml"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the verified W5D-00 B01+B02 development index, fixed runtime "
            "identity, and versioned handoff schemas in a new output directory."
        )
    )
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="Override the external B01/B02 root, for example /mnt/d/徘徊数据集/自采数据.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = build_wandering_camera_delivery_contract(
        project_root=args.project_root,
        config_path=args.config,
        source_root=args.source_root,
        output_dir=args.output,
    )
    print(
        json.dumps(
            {
                "output_dir": result.output_dir.as_posix(),
                "development_record_count": result.development_record_count,
                "batch_counts": result.batch_counts,
                "manifest_sha256": result.manifest_sha256,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
