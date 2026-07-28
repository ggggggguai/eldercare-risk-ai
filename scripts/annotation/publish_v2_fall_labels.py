from __future__ import annotations

import argparse
import json
from pathlib import Path

from elderly_monitoring.modules.fall_risk.label_publish import publish_v2_labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish source-specific fall-risk v2 labels to root files.")
    parser.add_argument("--source-root", type=Path, default=Path("data/annotations/fall_risk/generated/v2"))
    parser.add_argument("--action-output", type=Path, default=Path("data/annotations/fall_risk/action_labels.jsonl"))
    parser.add_argument("--event-output", type=Path, default=Path("data/annotations/fall_risk/event_labels.jsonl"))
    parser.add_argument("--report-output", type=Path, default=Path("reports/fall_risk/fall-risk-data-v2-root-publish.json"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    report = publish_v2_labels(
        args.source_root,
        action_output=args.action_output,
        event_output=args.event_output,
        report_output=args.report_output,
        overwrite=args.overwrite,
    )
    print(json.dumps(report["output_counts"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
