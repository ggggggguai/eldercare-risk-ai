from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from elderly_monitoring.modules.fall_risk.pose import run_yolov8_pose, write_jsonl
from elderly_monitoring.modules.fall_risk.pose_quality import (
    PoseQualityConfig,
    process_pose_records,
)


CACHE_SCHEMA_VERSION = "fall-pose-cache-v1"
BATCH_SCHEMA_VERSION = "fall-pose-cache-batch-v1"
COMPLETED_STATUSES = {"completed", "completed_no_pose"}
REVISION_TOLERANT_CACHE_FIELDS = {"manifest_sha256"}


def build_pose_cache_jobs(
    manifest_rows: Sequence[Mapping[str, Any]],
    *,
    datasets: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    selected_datasets = {str(dataset) for dataset in datasets or () if str(dataset)}
    manifests: dict[str, dict[str, Any]] = {}
    for source in manifest_rows:
        manifest = dict(source)
        raw_video_id = manifest.get("video_id")
        if raw_video_id is None or raw_video_id == "":
            continue
        video_id = str(raw_video_id)
        if video_id in manifests:
            raise ValueError(f"duplicate pose-cache manifest video_id: {video_id}")
        manifests[video_id] = manifest

    jobs: list[dict[str, Any]] = []
    for video_id, manifest in manifests.items():
        dataset = str(manifest.get("dataset", ""))
        if selected_datasets and dataset not in selected_datasets:
            continue
        if manifest.get("eligibility") is not True:
            continue
        media_type = str(manifest.get("media_type", "video"))
        if media_type != "video":
            continue
        _validate_path_component(video_id, field="video_id")
        video_path = str(manifest.get("path", ""))
        if not video_path:
            raise ValueError(f"pose-cache manifest is missing video path: {video_id}")
        if not dataset:
            raise ValueError(f"pose-cache manifest is missing dataset: {video_id}")
        if not _is_positive_number(manifest.get("fps")):
            raise ValueError(f"pose-cache manifest has invalid fps: {video_id}")
        if not _is_positive_number(manifest.get("frame_count")):
            raise ValueError(f"pose-cache manifest has invalid frame_count: {video_id}")
        if not _is_positive_number(manifest.get("width")):
            raise ValueError(f"pose-cache manifest has invalid width: {video_id}")
        if not _is_positive_number(manifest.get("height")):
            raise ValueError(f"pose-cache manifest has invalid height: {video_id}")
        source_sha256 = str(manifest.get("sha256", ""))
        if not source_sha256:
            raise ValueError(f"pose-cache manifest is missing sha256: {video_id}")
        jobs.append(
            {
                "video_id": video_id,
                "asset_id": str(manifest.get("asset_id", "")),
                "video_path": video_path,
                "scene": str(manifest.get("scene_region", "unknown")),
                "dataset": dataset,
                "source_group_id": str(manifest.get("source_group_id", "")),
                "source_sha256": source_sha256,
                "fps": float(manifest["fps"]),
                "frame_count": int(manifest["frame_count"]),
                "duration_sec": float(manifest.get("duration_sec") or 0.0),
                "width": int(manifest["width"]),
                "height": int(manifest["height"]),
            }
        )
    return sorted(jobs, key=lambda job: (job["dataset"], job["video_id"]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract a reusable full-video pose cache for fall-risk datasets."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/fall_risk_video_manifest.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/fall_risk/pose_quality_y8n_v1"),
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="只处理指定 dataset；可重复传入。省略时处理全部合格视频。",
    )
    parser.add_argument("--batch-id", default=None)
    parser.add_argument("--model", default="yolov8n-pose.pt")
    parser.add_argument("--device", default=None)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="记录单个视频失败并继续处理剩余任务。",
    )
    return parser


def ensure_cache_contract(output_dir: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    contract_path = output_dir / "cache_manifest.json"
    expected = dict(contract)
    if contract_path.is_file():
        existing = _read_json(contract_path)
        if existing != expected:
            differences = sorted(
                key
                for key in set(existing) | set(expected)
                if existing.get(key) != expected.get(key)
            )
            incompatible = [
                key for key in differences if key not in REVISION_TOLERANT_CACHE_FIELDS
            ]
            if incompatible:
                raise ValueError(
                    "cache contract mismatch for "
                    f"{contract_path}; differing fields: {', '.join(incompatible)}"
                )
        return existing

    existing_files = [path for path in output_dir.rglob("*") if path.is_file()]
    if existing_files:
        raise ValueError(
            f"cache output contains files but has no contract: {output_dir}"
        )
    _atomic_write_json(contract_path, expected)
    return expected


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _validate_args(args)
        manifest_rows = _read_jsonl(args.manifest)
        jobs = build_pose_cache_jobs(manifest_rows, datasets=args.dataset)
        if args.max_videos is not None:
            jobs = jobs[: args.max_videos]
        if not jobs:
            selected = ", ".join(args.dataset) if args.dataset else "all datasets"
            raise ValueError(f"no eligible pose-cache videos selected for {selected}")

        model_path = Path(args.model)
        if not model_path.is_file():
            raise FileNotFoundError(
                f"pose model file not found; automatic downloads are disabled: {model_path}"
            )
        input_hashes = {
            "manifest_sha256": _sha256_file(args.manifest),
            "model_sha256": _sha256_file(model_path),
        }
        quality_config = PoseQualityConfig()
        contract = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "manifest_path": args.manifest.as_posix(),
            **input_hashes,
            "model": args.model,
            "device": args.device,
            "confidence": args.confidence,
            "iou": args.iou,
            "tracker": args.tracker,
            "normalize_coordinates": True,
            "pose_quality_config": asdict(quality_config),
            "layout": {
                "raw": "raw/<video_id>.jsonl",
                "cleaned": "cleaned/<video_id>.jsonl",
                "state": "state/<video_id>.json",
            },
        }
        contract = ensure_cache_contract(args.output_dir, contract)
        contract_sha256 = _sha256_json(contract)

        batch_id = args.batch_id or _default_batch_id(args.dataset)
        _validate_path_component(batch_id, field="batch_id")
        batch_dir = args.output_dir / "batches" / batch_id
        batch_contract = {
            "schema_version": BATCH_SCHEMA_VERSION,
            "batch_id": batch_id,
            "datasets": sorted(set(args.dataset)),
            "job_count": len(jobs),
            "job_ids_sha256": _sha256_text(
                "\n".join(str(job["video_id"]) for job in jobs) + "\n"
            ),
            "cache_contract_sha256": contract_sha256,
        }
        _ensure_batch_contract(batch_dir, batch_contract)

        from ultralytics import YOLO

        model = YOLO(args.model)
        args.output_dir.joinpath("raw").mkdir(parents=True, exist_ok=True)
        args.output_dir.joinpath("cleaned").mkdir(parents=True, exist_ok=True)
        args.output_dir.joinpath("state").mkdir(parents=True, exist_ok=True)

        results: list[dict[str, Any]] = []
        for index, job in enumerate(jobs, start=1):
            result = _run_job(
                args=args,
                job=job,
                model=model,
                quality_config=quality_config,
                contract_sha256=contract_sha256,
                index=index,
                total=len(jobs),
            )
            results.append(result)
            progress = _build_summary(
                args=args,
                batch_contract=batch_contract,
                results=results,
                final=False,
            )
            _atomic_write_json(batch_dir / "preparation_progress.json", progress)

            if result["status"] == "failed" and not args.continue_on_error:
                break
            if index % 25 == 0:
                _assert_inputs_unchanged(args, input_hashes)

        _assert_inputs_unchanged(args, input_hashes)
        summary = _build_summary(
            args=args,
            batch_contract=batch_contract,
            results=results,
            final=True,
        )
        _atomic_write_json(batch_dir / "preparation_summary.json", summary)
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    console_summary = {
        key: summary[key]
        for key in (
            "batch_id",
            "job_count",
            "processed_job_count",
            "remaining_job_count",
            "status_counts",
            "source_frame_count",
            "raw_pose_count",
            "cleaned_pose_count",
            "elapsed_sec",
        )
    }
    console_summary["summary_path"] = (
        batch_dir / "preparation_summary.json"
    ).as_posix()
    print(json.dumps(console_summary, ensure_ascii=False, sort_keys=True))
    return 2 if summary["status_counts"].get("failed", 0) else 0


def _run_job(
    *,
    args: argparse.Namespace,
    job: Mapping[str, Any],
    model: Any,
    quality_config: PoseQualityConfig,
    contract_sha256: str,
    index: int,
    total: int,
) -> dict[str, Any]:
    video_id = str(job["video_id"])
    raw_path = args.output_dir / "raw" / f"{video_id}.jsonl"
    cleaned_path = args.output_dir / "cleaned" / f"{video_id}.jsonl"
    state_path = args.output_dir / "state" / f"{video_id}.json"
    state = _completed_state(
        state_path,
        raw_path=raw_path,
        cleaned_path=cleaned_path,
        contract_sha256=contract_sha256,
        source_sha256=str(job["source_sha256"]),
    )
    if state is not None:
        print(f"[{index}/{total}] skip {video_id}", flush=True)
        return {
            **dict(job),
            "status": "skipped_existing",
            "raw_pose_count": int(state["raw_pose_count"]),
            "cleaned_pose_count": int(state["cleaned_pose_count"]),
        }

    started = time.monotonic()
    raw_partial = raw_path.with_suffix(f"{raw_path.suffix}.part")
    cleaned_partial = cleaned_path.with_suffix(f"{cleaned_path.suffix}.part")
    try:
        video_path = Path(str(job["video_path"]))
        if not video_path.is_file():
            raise FileNotFoundError(f"source pose-cache video not found: {video_path}")
        print(f"[{index}/{total}] pose {video_id}", flush=True)
        raw_count = run_yolov8_pose(
            video_path=video_path,
            output_path=raw_partial,
            model_name=args.model,
            scene_region=str(job["scene"]),
            person_id_prefix=video_id,
            confidence_threshold=args.confidence,
            iou_threshold=args.iou,
            tracker_config=args.tracker,
            max_frames=None,
            normalize_coordinates=True,
            device=args.device,
            model=model,
            persist_tracker=False,
        )
        cleaned = process_pose_records(_read_jsonl(raw_partial), config=quality_config)
        cleaned_count = write_jsonl(cleaned, cleaned_partial)
        if cleaned_count != raw_count:
            raise ValueError(
                f"pose count mismatch for {video_id}: raw={raw_count}, cleaned={cleaned_count}"
            )
        os.replace(raw_partial, raw_path)
        os.replace(cleaned_partial, cleaned_path)
        status = "completed" if cleaned_count else "completed_no_pose"
        elapsed_sec = round(time.monotonic() - started, 3)
        state = {
            "schema_version": "fall-pose-cache-job-v1",
            "video_id": video_id,
            "dataset": str(job["dataset"]),
            "status": status,
            "source_sha256": str(job["source_sha256"]),
            "cache_contract_sha256": contract_sha256,
            "raw_path": raw_path.as_posix(),
            "cleaned_path": cleaned_path.as_posix(),
            "raw_pose_count": raw_count,
            "cleaned_pose_count": cleaned_count,
            "raw_size_bytes": raw_path.stat().st_size,
            "cleaned_size_bytes": cleaned_path.stat().st_size,
            "elapsed_sec": elapsed_sec,
            "completed_at": datetime.now(UTC).isoformat(),
        }
        _atomic_write_json(state_path, state)
        return {
            **dict(job),
            "status": status,
            "raw_pose_count": raw_count,
            "cleaned_pose_count": cleaned_count,
            "elapsed_sec": elapsed_sec,
        }
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        elapsed_sec = round(time.monotonic() - started, 3)
        failed_state = {
            "schema_version": "fall-pose-cache-job-v1",
            "video_id": video_id,
            "dataset": str(job["dataset"]),
            "status": "failed",
            "source_sha256": str(job["source_sha256"]),
            "cache_contract_sha256": contract_sha256,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "elapsed_sec": elapsed_sec,
            "failed_at": datetime.now(UTC).isoformat(),
        }
        _atomic_write_json(state_path, failed_state)
        print(f"[{index}/{total}] failed {video_id}: {exc}", file=sys.stderr, flush=True)
        return {**dict(job), **failed_state}


def _completed_state(
    state_path: Path,
    *,
    raw_path: Path,
    cleaned_path: Path,
    contract_sha256: str,
    source_sha256: str,
) -> dict[str, Any] | None:
    if not state_path.is_file():
        return None
    try:
        state = _read_json(state_path)
    except (OSError, ValueError):
        return None
    if state.get("status") not in COMPLETED_STATUSES:
        return None
    if state.get("cache_contract_sha256") != contract_sha256:
        return None
    if state.get("source_sha256") != source_sha256:
        return None
    if not raw_path.is_file() or not cleaned_path.is_file():
        return None
    if raw_path.stat().st_size != int(state.get("raw_size_bytes", -1)):
        return None
    if cleaned_path.stat().st_size != int(state.get("cleaned_size_bytes", -1)):
        return None
    return state


def _build_summary(
    *,
    args: argparse.Namespace,
    batch_contract: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    final: bool,
) -> dict[str, Any]:
    status_counts: dict[str, int] = defaultdict(int)
    raw_pose_count = 0
    cleaned_pose_count = 0
    frame_count = 0
    elapsed_sec = 0.0
    failures: list[dict[str, Any]] = []
    for result in results:
        status = str(result.get("status", "unknown"))
        status_counts[status] += 1
        raw_pose_count += int(result.get("raw_pose_count", 0))
        cleaned_pose_count += int(result.get("cleaned_pose_count", 0))
        frame_count += int(result.get("frame_count", 0))
        elapsed_sec += float(result.get("elapsed_sec", 0.0))
        if status == "failed":
            failures.append(
                {
                    "video_id": result.get("video_id"),
                    "error_type": result.get("error_type"),
                    "error": result.get("error"),
                }
            )
    summary: dict[str, Any] = {
        **dict(batch_contract),
        "final": final,
        "processed_job_count": len(results),
        "remaining_job_count": int(batch_contract["job_count"]) - len(results),
        "status_counts": dict(sorted(status_counts.items())),
        "source_frame_count": frame_count,
        "raw_pose_count": raw_pose_count,
        "cleaned_pose_count": cleaned_pose_count,
        "elapsed_sec": round(elapsed_sec, 3),
        "failures": failures,
        "updated_at": datetime.now(UTC).isoformat(),
        "parameters": {
            "model": args.model,
            "device": args.device,
            "confidence": args.confidence,
            "iou": args.iou,
            "tracker": args.tracker,
            "normalize_coordinates": True,
        },
    }
    if final:
        summary["jobs"] = list(results)
    elif results:
        summary["last_job"] = dict(results[-1])
    return summary


def _ensure_batch_contract(batch_dir: Path, contract: Mapping[str, Any]) -> None:
    batch_dir.mkdir(parents=True, exist_ok=True)
    path = batch_dir / "batch_manifest.json"
    expected = dict(contract)
    if path.is_file():
        existing = _read_json(path)
        if existing != expected:
            differences = sorted(
                key
                for key in set(existing) | set(expected)
                if existing.get(key) != expected.get(key)
            )
            raise ValueError(
                f"batch contract mismatch for {path}; differing fields: {', '.join(differences)}"
            )
        return
    _atomic_write_json(path, expected)


def _validate_args(args: argparse.Namespace) -> None:
    if not 0.0 < args.confidence <= 1.0:
        raise ValueError("confidence must be in (0, 1]")
    if not 0.0 < args.iou <= 1.0:
        raise ValueError("iou must be in (0, 1]")
    if args.max_videos is not None and args.max_videos < 1:
        raise ValueError("max-videos must be positive")
    for dataset in args.dataset:
        _validate_path_component(str(dataset), field="dataset")


def _assert_inputs_unchanged(
    args: argparse.Namespace,
    expected_hashes: Mapping[str, str],
) -> None:
    current = {
        "manifest_sha256": _sha256_file(args.manifest),
        "model_sha256": _sha256_file(Path(args.model)),
    }
    if current != dict(expected_hashes):
        changed = sorted(key for key in current if current[key] != expected_hashes.get(key))
        raise ValueError(f"pose-cache inputs changed during extraction: {', '.join(changed)}")


def _default_batch_id(datasets: Sequence[str]) -> str:
    unique = sorted(set(datasets))
    return "all" if not unique else "__".join(unique)


def _validate_path_component(value: str, *, field: str) -> None:
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"invalid {field}: {value!r}")


def _is_positive_number(value: Any) -> bool:
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return False


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(f"{path.suffix}.part")
    partial.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            rows.append(value)
    return rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: Mapping[str, Any]) -> str:
    return _sha256_text(
        json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
