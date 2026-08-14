from __future__ import annotations

import argparse
import json
from pathlib import Path

from elderly_monitoring.modules.fall_risk.self_collected_scf import (
    replay_near_fall_video,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay near-fall rule and frozen TCN checkpoints on cleaned pose JSONL."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    report = replay_near_fall_video(
        cleaned_pose_path=args.input,
        checkpoint_paths=args.checkpoint,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"output={args.output} pose_records={report['pose_record_count']}")


if __name__ == "__main__":
    main()
