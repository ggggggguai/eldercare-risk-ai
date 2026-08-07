from __future__ import annotations

import argparse
import json
from pathlib import Path

from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.manifest import (
    build_microexpression_manifest,
    write_microexpression_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the audited CASME II and SMIC sequence manifest."
    )
    parser.add_argument("--casme-root", type=Path, required=True)
    parser.add_argument("--smic-root", type=Path, required=True)
    parser.add_argument("--casme-csv", type=Path, required=True)
    parser.add_argument("--smic-csv", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "manifests" / "microexpression",
    )
    parser.add_argument(
        "--exclude-smic-non-micro",
        action="store_true",
        help="Exclude SMIC non_micro sequences from the spotting audit rows.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = build_microexpression_manifest(
        casme_root=args.casme_root,
        smic_root=args.smic_root,
        casme_csv=args.casme_csv,
        smic_csv=args.smic_csv,
        include_smic_non_micro=not args.exclude_smic_non_micro,
    )
    summary = write_microexpression_manifest(
        records=records,
        output_dir=args.output_dir,
        casme_root=args.casme_root,
        smic_root=args.smic_root,
        casme_csv=args.casme_csv,
        smic_csv=args.smic_csv,
    )
    result = {
        "status": "pass" if summary["manifest"]["excluded_count"] == 0 else "warning",
        "row_count": summary["manifest"]["row_count"],
        "excluded_count": summary["manifest"]["excluded_count"],
        "manifest_sha256": summary["manifest"]["sha256"],
        "manifest_path": summary["manifest"]["path"],
        "mapping_path": summary["path_mapping"]["path"],
        "summary_path": summary["summary_path"],
    }
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
