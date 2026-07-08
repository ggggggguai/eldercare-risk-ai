from __future__ import annotations

import argparse
from pathlib import Path

from elderly_monitoring.modules.fall_risk.ltmm import write_ltmm_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a local JSONL manifest for the LTMM PhysioNet dataset."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/external/ltmm/raw"),
        help="LTMM raw directory containing RECORDS, tables, .hea and .dat files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/manifests/ltmm_manifest.jsonl"),
        help="Output JSONL manifest path.",
    )
    args = parser.parse_args()

    count = write_ltmm_manifest(args.raw_dir, args.output)
    print(f"已写入 {count} 条 LTMM manifest 记录：{args.output}")


if __name__ == "__main__":
    main()
