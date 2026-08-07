from __future__ import annotations

import argparse
import json
from pathlib import Path

from elderly_monitoring.modules.fall_risk.fall_tiktok import (
    prepare_fall_tiktok_cvat_export,
    write_prepared_fall_tiktok_export,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Redact and normalize the fall_tiktok CVAT project export."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--source-map",
        type=Path,
        default=Path("configs/data/fall_tiktok_source_map_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/annotations/fall_risk/cvat_exports/raw/fall_tiktok/"
            "fall_tiktok_cvat_redacted.zip"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    try:
        result = prepare_fall_tiktok_cvat_export(args.input, args.source_map)
        write_prepared_fall_tiktok_export(
            result, args.output, overwrite=args.overwrite
        )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    print(
        json.dumps(
            {
                "output": args.output.as_posix(),
                "source_sha256": result.source_sha256,
                "prepared_sha256": result.prepared_sha256,
                "task_count": result.task_count,
                "track_count": result.track_count,
                "removed_identity_elements": result.removed_identity_elements,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
