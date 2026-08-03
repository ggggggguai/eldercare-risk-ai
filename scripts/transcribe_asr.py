"""Transcribe one audio file through the stable ASRTranscript v1 facade."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from elderly_monitoring.modules.asr import transcribe_audio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe audio through the ASR module facade")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--request-id", default="asr-local-001")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = transcribe_audio(args.audio, request_id=args.request_id)
    payload = result.model_dump(mode="json")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"output={args.output.resolve()}")
    else:
        print(json.dumps(payload, ensure_ascii=True, indent=2))
    return 0 if result.status in {"completed", "completed_empty_speech"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
