from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from elderly_monitoring.modules.fall_risk.pose import run_yolov8_pose, write_jsonl
from elderly_monitoring.modules.fall_risk.pose_quality import (
    PoseQualityConfig,
    process_pose_records,
)
from elderly_monitoring.modules.fall_risk.self_collected_scf import (
    replay_near_fall_video,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run resumable SCF_MVP_V1 near-fall baseline replay."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--action-labels", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--model", type=Path, default=Path("models/yolov8n-pose.pt"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-videos", type=int, default=None)
    args = parser.parse_args()

    try:
        if not args.model.is_file():
            raise FileNotFoundError(args.model)
        for checkpoint in args.checkpoint:
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
        manifest = _read_jsonl(args.manifest)
        actions = _read_jsonl(args.action_labels)
        planned_actions = _planned_actions(actions)
        if args.max_videos is not None:
            if args.max_videos < 1:
                raise ValueError("max-videos must be positive")
            manifest = manifest[: args.max_videos]
        contract = {
            "schema_version": "self-collected-near-fall-batch-replay-v1",
            "manifest_sha256": _sha256(args.manifest),
            "action_labels_sha256": _sha256(args.action_labels),
            "model_path": args.model.as_posix(),
            "model_sha256": _sha256(args.model),
            "checkpoint_hashes": [
                {"path": path.as_posix(), "sha256": _sha256(path)}
                for path in args.checkpoint
            ],
            "device": args.device,
            "short_clip_replay": True,
            "fp_hour_reported": False,
            "test_pose_read": False,
            "test_evaluated": False,
        }
        _ensure_contract(args.output_dir, contract)
        from ultralytics import YOLO

        model = YOLO(args.model.as_posix())
        results = []
        for index, row in enumerate(manifest, 1):
            result = _run_video(
                row,
                planned_action=planned_actions.get(str(row["video_id"])),
                output_dir=args.output_dir,
                model=model,
                model_path=args.model,
                device=args.device,
                checkpoints=args.checkpoint,
            )
            results.append(result)
            print(
                f"[{index}/{len(manifest)}] {result['status']} {row['video_id']}",
                flush=True,
            )
            _write_json(
                args.output_dir / "progress.json",
                _summary(results, contract=contract, final=False),
            )
        summary = _summary(results, contract=contract, final=True)
        _write_json(args.output_dir / "summary.json", summary)
        print(json.dumps({key: summary[key] for key in ("video_count", "status_counts", "video_positive_counts")}, sort_keys=True))
        return 2 if summary["status_counts"].get("failed", 0) else 0
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}")
        return 2


def _run_video(
    manifest: Mapping[str, Any],
    *,
    planned_action: str | None,
    output_dir: Path,
    model: Any,
    model_path: Path,
    device: str,
    checkpoints: list[Path],
) -> dict[str, Any]:
    video_id = str(manifest["video_id"])
    raw_path = output_dir / "raw_pose" / f"{video_id}.jsonl"
    cleaned_path = output_dir / "cleaned_pose" / f"{video_id}.jsonl"
    replay_path = output_dir / "videos" / f"{video_id}.json"
    if replay_path.is_file():
        replay = _read_json(replay_path)
        if replay.get("source_sha256") == manifest.get("content_sha256"):
            return _result_row(manifest, planned_action, replay, "skipped_existing")

    started = time.monotonic()
    raw_partial = raw_path.with_suffix(".jsonl.part")
    cleaned_partial = cleaned_path.with_suffix(".jsonl.part")
    try:
        source_path = Path(str(manifest["path"]))
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        if _sha256(source_path) != manifest.get("content_sha256"):
            raise ValueError(f"source SHA-256 mismatch: {video_id}")
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        cleaned_path.parent.mkdir(parents=True, exist_ok=True)
        count = run_yolov8_pose(
            video_path=source_path,
            output_path=raw_partial,
            model_name=model_path.as_posix(),
            scene_region="self_collected_home",
            person_id_prefix=video_id,
            confidence_threshold=0.25,
            iou_threshold=0.5,
            tracker_config="bytetrack.yaml",
            max_frames=None,
            normalize_coordinates=True,
            device=device,
            model=model,
            persist_tracker=False,
        )
        cleaned = process_pose_records(_read_jsonl(raw_partial), config=PoseQualityConfig())
        cleaned_count = write_jsonl(cleaned, cleaned_partial)
        if cleaned_count != count:
            raise ValueError(f"raw/cleaned pose count mismatch: {video_id}")
        os.replace(raw_partial, raw_path)
        os.replace(cleaned_partial, cleaned_path)
        replay = replay_near_fall_video(
            cleaned_pose_path=cleaned_path,
            checkpoint_paths=checkpoints,
        )
        replay.update(
            {
                "video_id": video_id,
                "file_name": manifest["file_name"],
                "provisional_subject": manifest["provisional_subject"],
                "planned_action": planned_action,
                "source_sha256": manifest["content_sha256"],
                "elapsed_sec": round(time.monotonic() - started, 3),
            }
        )
        _write_json(replay_path, replay)
        return _result_row(manifest, planned_action, replay, "completed")
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        for partial in (raw_partial, cleaned_partial):
            partial.unlink(missing_ok=True)
        return {
            "video_id": video_id,
            "file_name": manifest["file_name"],
            "provisional_subject": manifest["provisional_subject"],
            "planned_action": planned_action,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "elapsed_sec": round(time.monotonic() - started, 3),
        }


def _result_row(
    manifest: Mapping[str, Any],
    planned_action: str | None,
    replay: Mapping[str, Any],
    status: str,
) -> dict[str, Any]:
    return {
        "video_id": manifest["video_id"],
        "file_name": manifest["file_name"],
        "provisional_subject": manifest["provisional_subject"],
        "planned_action": planned_action,
        "status": status,
        "pose_record_count": replay.get("pose_record_count", 0),
        "track_count": replay.get("track_count", 0),
        "causal_window_count": replay.get("causal_window_count", 0),
        "rule_max_score": replay.get("rule", {}).get("max_near_fall_event_score"),
        "rule_video_positive": (
            replay.get("rule", {}).get("max_near_fall_event_score") is not None
            and float(replay["rule"]["max_near_fall_event_score"]) >= 0.5
        ),
        "tcn_video_positive": [bool(row.get("video_positive")) for row in replay.get("tcn", [])],
        "tcn_max_scores": [row.get("max_near_fall_event_score") for row in replay.get("tcn", [])],
        "elapsed_sec": replay.get("elapsed_sec", 0.0),
    }


def _summary(
    rows: list[Mapping[str, Any]], *, contract: Mapping[str, Any], final: bool
) -> dict[str, Any]:
    status = Counter(str(row["status"]) for row in rows)
    completed = [row for row in rows if row["status"] != "failed"]
    evaluable = [row for row in completed if int(row.get("causal_window_count", 0)) > 0]
    tcn_count = len(contract["checkpoint_hashes"])
    video_positive_counts = {
        "rule": sum(bool(row.get("rule_video_positive")) for row in completed),
        **{
            f"tcn_seed_{index}": sum(
                len(row.get("tcn_video_positive", [])) > index
                and bool(row["tcn_video_positive"][index])
                for row in completed
            )
            for index in range(tcn_count)
        },
    }
    tcn_evaluable_counts = {
        f"tcn_seed_{index}": sum(
            len(row.get("tcn_max_scores", [])) > index
            and row["tcn_max_scores"][index] is not None
            for row in completed
        )
        for index in range(tcn_count)
    }
    strata: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {"videos": 0, "rule_positive": 0, "tcn_positive": [0] * tcn_count}
    )
    for row in completed:
        key = (str(row.get("provisional_subject")), str(row.get("planned_action")))
        item = strata[key]
        item["videos"] += 1
        item["rule_positive"] += int(bool(row.get("rule_video_positive")))
        for index, value in enumerate(row.get("tcn_video_positive", [])):
            item["tcn_positive"][index] += int(bool(value))
    return {
        **dict(contract),
        "final": final,
        "video_count": len(rows),
        "status_counts": dict(sorted(status.items())),
        "pose_record_count": sum(int(row.get("pose_record_count", 0)) for row in completed),
        "causal_window_count": sum(int(row.get("causal_window_count", 0)) for row in completed),
        "tcn_unavailable_video_count": len(completed) - len(evaluable),
        "tcn_evaluable_video_counts": tcn_evaluable_counts,
        "video_positive_counts": video_positive_counts,
        "strata": [
            {"provisional_subject": key[0], "planned_action": key[1], **value}
            for key, value in sorted(strata.items())
        ],
        "failures": [dict(row) for row in rows if row["status"] == "failed"],
        "results": [dict(row) for row in rows] if final else [],
        "main_path_unchanged": True,
        "fp_hour_reported": False,
        "test_pose_read": False,
        "test_evaluated": False,
    }


def _planned_actions(rows: list[Mapping[str, Any]]) -> dict[str, str]:
    values: dict[str, str] = {}
    for row in rows:
        video_id = str(row["video_id"])
        action_id = str(row["action_id"])
        filename_action = str(row["file_name"]).rsplit("_", 1)[-1].removesuffix(".mp4")
        if action_id == filename_action:
            values[video_id] = action_id
    return values


def _ensure_contract(output_dir: Path, contract: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "contract.json"
    if path.is_file():
        if _read_json(path) != dict(contract):
            raise ValueError("batch replay contract mismatch")
        return
    if any(output_dir.iterdir()):
        raise ValueError("batch replay output has files but no contract")
    _write_json(path, contract)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row must be an object: {path}")
            rows.append(value)
    return rows


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    partial.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(partial, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
