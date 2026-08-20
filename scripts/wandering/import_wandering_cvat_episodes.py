from __future__ import annotations

import argparse
import json
from pathlib import Path

from elderly_monitoring.modules.mental_health.wandering.camera_adapter import (
    validate_media_sidecar,
)
from elderly_monitoring.modules.mental_health.wandering.camera_episode_import import (
    build_cvat_episode_import_bundle,
)
from elderly_monitoring.modules.mental_health.wandering.camera_inference import (
    load_camera_config,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAMERA_CONFIG = ROOT / "configs/modules/wandering_camera_v1.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Import CVAT for video 1.1 episode tracks into separate truth-free "
            "boundary and truth JSONL files."
        )
    )
    parser.add_argument("--camera-config", type=Path, default=DEFAULT_CAMERA_CONFIG)
    parser.add_argument("--cvat-xml", type=Path, required=True)
    parser.add_argument("--media-sidecar", type=Path, required=True)
    parser.add_argument("--target-track-id", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    camera_config = load_camera_config(args.camera_config)
    media = validate_media_sidecar(
        json.loads(args.media_sidecar.read_text(encoding="utf-8")),
        camera_config,
    )
    result = build_cvat_episode_import_bundle(
        cvat_xml_path=args.cvat_xml,
        media=media,
        target_track_id=args.target_track_id,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "episode_count": len(result.boundaries),
                "boundary_truth_views_separated": True,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
