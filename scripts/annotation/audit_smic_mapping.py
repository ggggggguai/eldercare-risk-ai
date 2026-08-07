from __future__ import annotations

import argparse
import json
from pathlib import Path

from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.manifest import (
    audit_smic_mapping,
    write_smic_mapping_audit,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit SMIC labels, subject ids, directories, and manifest mapping."
    )
    parser.add_argument("--smic-root", type=Path, required=True)
    parser.add_argument("--smic-csv", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data/manifests/microexpression/sequence_manifest_v1.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "data/manifests/microexpression/smic_mapping_audit_v1.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report = audit_smic_mapping(
        smic_root=args.smic_root,
        smic_csv=args.smic_csv,
        manifest_records=records,
    )
    output_sha256 = write_smic_mapping_audit(report, args.output)
    result = {
        "status": report["status"],
        "issue_count": len(report["issues"]),
        "csv_row_count": report["csv"]["row_count"],
        "classification_count": report["manifest"]["classification_count"],
        "spotting_negative_count": report["manifest"]["spotting_negative_count"],
        "output": args.output.resolve().as_posix(),
        "output_sha256": output_sha256,
    }
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
