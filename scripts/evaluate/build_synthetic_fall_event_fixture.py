from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def build_fixture(output_dir: Path, evaluation_config: Path) -> dict[str, str]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if not evaluation_config.is_file():
        raise FileNotFoundError(evaluation_config)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".fall-event-synthetic-", dir=output_dir.parent)
    )
    try:
        truth = {
            "eligibility": True,
            "end_time": 12.0,
            "event_type": "fall",
            "label_id": "synthetic-ground-truth-1",
            "onset_time": 10.0,
            "review_status": "final",
            "source_group_id": "synthetic-group-1",
            "start_time": 10.0,
            "subject_id": "synthetic-subject-1",
            "task_type": "fall_event",
            "video_id": "synthetic-video-1",
        }
        manifest = {
            "asset_id": "synthetic-video-1",
            "continuous_monitoring_eligible": False,
            "duration_sec": 20.0,
            "eligibility": True,
            "source_group_id": "synthetic-group-1",
            "subject_id": "synthetic-subject-1",
            "video_id": "synthetic-video-1",
        }
        assignment = {
            "asset_id": "synthetic-video-1",
            "partition": "validation",
            "sample_id": "synthetic-video-1",
            "task_type": "fall_event",
            "video_id": "synthetic-video-1",
        }
        truth_payload = _json_bytes(truth)
        manifest_payload = _json_bytes(manifest)
        config_sha256 = _sha256(evaluation_config.read_bytes())
        split_payload = {
            "assignments": [assignment],
            "config_sha256": config_sha256,
            "labels_sha256": _sha256(truth_payload),
            "manifest_canonical_sha256": _sha256(manifest_payload),
            "manifest_sha256": _sha256(manifest_payload),
            "protocol_status": "provisional",
            "schema_version": "1.0",
            "split_name": "fall_event_v1_synthetic",
            "status": "ready",
            "task_type": "fall_event",
            "validation_report_sha256": None,
        }
        split_sha256 = _sha256(
            json.dumps(
                split_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        split_id = f"fall_event_v1_synthetic:sha256:{split_sha256}"
        assignment_payload = _json_bytes({**assignment, "split_id": split_id})
        prediction = {
            "config_hash": config_sha256,
            "end_time": 12.1,
            "event_type": "fall",
            "model_version": "synthetic-perfect-match-v1",
            "onset_time": 10.1,
            "prediction_id": "synthetic-prediction-1",
            "quality_state": "usable",
            "score": 0.9,
            "split_id": split_id,
            "start_time": 10.1,
            "status": "emitted",
            "task_type": "fall_event",
            "video_id": "synthetic-video-1",
        }
        split = {
            **{key: value for key, value in split_payload.items() if key != "assignments"},
            "assignments_sha256": _sha256(assignment_payload),
            "split_id": split_id,
            "split_sha256": split_sha256,
        }
        _write(staging / "ground_truth.jsonl", truth_payload)
        _write(staging / "manifest.jsonl", manifest_payload)
        _write(staging / "assignments.jsonl", assignment_payload)
        _write(staging / "predictions.jsonl", _json_bytes(prediction))
        _write(staging / "split.json", _json_bytes(split))
        os.replace(staging, output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "split_id": split_id,
        "split_sha256": split_sha256,
        "labels_sha256": split_payload["labels_sha256"],
        "manifest_sha256": split_payload["manifest_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a deterministic synthetic fall-event evaluation fixture."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--evaluation-config",
        type=Path,
        default=Path("configs/evaluation/fall_event_v1.provisional.yaml"),
    )
    args = parser.parse_args()
    try:
        result = build_fixture(args.output_dir, args.evaluation_config)
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
