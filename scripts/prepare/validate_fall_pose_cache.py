from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


VALIDATION_SCHEMA_VERSION = "fall-pose-cache-validation-v1"
COMPLETED_JOB_STATUSES = {"completed", "completed_no_pose", "skipped_existing"}
LOWER_BODY_NAMES = {
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
}
EXPECTED_KEYPOINT_NAMES = {
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
}
COUNT_FIELDS = (
    "videos",
    "source_frames",
    "detected_frames",
    "pose_records",
    "multi_person_frames",
    "keypoints",
    "valid_keypoints",
    "lower_body_points",
    "valid_lower_body_points",
    "interpolated_points",
    "jump_outliers",
    "low_quality_records",
    "usable_gait_records",
    "usable_sit_stand_records",
    "usable_near_fall_records",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fully parse and validate a completed fall-risk pose-cache batch."
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/processed/fall_risk/pose_quality_y8n_v1"),
    )
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def validate_pose_cache(*, cache_dir: Path, summary_path: Path) -> dict[str, Any]:
    errors: list[str] = []
    part_files: list[str] = []

    try:
        cache_contract = _read_json(cache_dir / "cache_manifest.json")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        cache_contract = {}
        errors.append(f"invalid cache contract: {exc}")
    try:
        summary = _read_json(summary_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return _failed_report(errors + [f"invalid batch summary: {exc}"], part_files)
    try:
        batch_contract = _read_json(summary_path.parent / "batch_manifest.json")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        batch_contract = {}
        errors.append(f"invalid batch contract: {exc}")
    for field in (
        "schema_version",
        "batch_id",
        "datasets",
        "job_count",
        "job_ids_sha256",
        "cache_contract_sha256",
    ):
        if batch_contract and summary.get(field) != batch_contract.get(field):
            errors.append(f"batch contract mismatch for {field}")

    jobs = summary.get("jobs")
    if not isinstance(jobs, list):
        return _failed_report(errors + ["batch summary has no jobs array"], part_files)
    batch_video_ids = {
        str(job.get("video_id"))
        for job in jobs
        if isinstance(job, dict) and job.get("video_id")
    }
    part_files = sorted(
        path.as_posix()
        for directory in ("raw", "cleaned", "state")
        for path in (cache_dir / directory).glob("*.part")
        if path.name.removesuffix(".jsonl.part").removesuffix(".json.part")
        in batch_video_ids
    )
    if part_files:
        errors.append(f"batch contains {len(part_files)} partial files")
    if summary.get("final") is not True:
        errors.append("batch summary is not final")
    if _non_negative_int(summary.get("processed_job_count"), default=-1) != len(jobs):
        errors.append("batch summary processed job count mismatch")
    if _non_negative_int(summary.get("remaining_job_count"), default=-1) != 0:
        errors.append("batch summary has remaining jobs")
    if _non_negative_int(summary.get("job_count"), default=-1) != len(jobs):
        errors.append("batch summary job count mismatch")
    if summary.get("failures"):
        errors.append("batch summary contains failed jobs")

    contract_sha256 = _sha256_json(cache_contract) if cache_contract else None
    expected_contract_sha256 = summary.get("cache_contract_sha256")
    if contract_sha256 and expected_contract_sha256 != contract_sha256:
        errors.append("cache contract hash does not match batch summary")

    overall = _new_counts()
    by_dataset: dict[str, dict[str, int]] = defaultdict(_new_counts)
    by_scene: dict[str, dict[str, int]] = defaultdict(_new_counts)
    by_action: dict[str, dict[str, int]] = defaultdict(_new_counts)
    low_quality_videos: list[str] = []
    multiple_track_videos: list[str] = []
    missing_last_frame_videos: list[str] = []
    seen_video_ids: set[str] = set()
    derived_status_counts: Counter[str] = Counter()
    raw_pose_total = 0

    for job_index, raw_job in enumerate(jobs, start=1):
        if not isinstance(raw_job, dict):
            errors.append(f"job {job_index} is not an object")
            continue
        job = dict(raw_job)
        video_id = str(job.get("video_id", ""))
        if not video_id:
            errors.append(f"job {job_index} has no video_id")
            continue
        if video_id in seen_video_ids:
            errors.append(f"duplicate batch video_id: {video_id}")
            continue
        seen_video_ids.add(video_id)
        if job.get("status") not in COMPLETED_JOB_STATUSES:
            errors.append(f"job is not complete: {video_id}")
        derived_status_counts[str(job.get("status"))] += 1

        dataset = str(job.get("dataset") or "unknown")
        scene = str(job.get("scene") or "unknown")
        action = _action_code(video_id)
        groups = (overall, by_dataset[dataset], by_scene[scene], by_action[action])
        source_frames = _non_negative_int(job.get("frame_count"), default=0)
        for counts in groups:
            counts["videos"] += 1
            counts["source_frames"] += source_frames

        raw_path = cache_dir / "raw" / f"{video_id}.jsonl"
        cleaned_path = cache_dir / "cleaned" / f"{video_id}.jsonl"
        state_path = cache_dir / "state" / f"{video_id}.json"
        try:
            state = _read_json(state_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"invalid state for {video_id}: {exc}")
            continue
        if state.get("status") not in {"completed", "completed_no_pose"}:
            errors.append(f"state is not complete for {video_id}")
        if state.get("video_id") != video_id:
            errors.append(f"state video_id mismatch for {video_id}")

        if not raw_path.is_file() or not cleaned_path.is_file():
            errors.append(f"pose files are missing for {video_id}")
            continue
        if raw_path.stat().st_size != _non_negative_int(state.get("raw_size_bytes"), default=-1):
            errors.append(f"raw size mismatch for {video_id}")
        if cleaned_path.stat().st_size != _non_negative_int(
            state.get("cleaned_size_bytes"), default=-1
        ):
            errors.append(f"cleaned size mismatch for {video_id}")

        raw_count = _count_jsonl(raw_path, errors=errors, label=f"raw {video_id}")
        frame_counts: Counter[int] = Counter()
        frame_people: set[tuple[int, str]] = set()
        track_ids: set[str] = set()
        video_low_quality = 0
        cleaned_count = 0
        try:
            for line_number, record in _iter_jsonl(cleaned_path):
                cleaned_count += 1
                prefix = f"{cleaned_path}:{line_number}"
                frame_id = record.get("frame_id")
                if not isinstance(frame_id, int) or frame_id < 0:
                    errors.append(f"{prefix} has invalid frame_id")
                    continue
                frame_counts[frame_id] += 1
                person_id = str(record.get("person_id", ""))
                frame_person = (frame_id, person_id)
                if not person_id or frame_person in frame_people:
                    errors.append(f"{prefix} has empty or duplicate person_id in frame")
                frame_people.add(frame_person)
                track_ids.add(str(record.get("track_id")))
                _validate_record(record, prefix=prefix, errors=errors)
                for counts in groups:
                    _update_record_counts(counts, record)
                if record.get("quality_state") != "usable":
                    video_low_quality += 1
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"invalid cleaned {video_id}: {exc}")

        detected_frames = len(frame_counts)
        multi_person_frames = sum(1 for count in frame_counts.values() if count > 1)
        for counts in groups:
            counts["detected_frames"] += detected_frames
            counts["multi_person_frames"] += multi_person_frames
        if video_low_quality:
            low_quality_videos.append(video_id)
        if len(track_ids) > 1:
            multiple_track_videos.append(video_id)
        if source_frames and source_frames - 1 not in frame_counts:
            missing_last_frame_videos.append(video_id)

        expected_raw = _non_negative_int(state.get("raw_pose_count"), default=-1)
        expected_cleaned = _non_negative_int(state.get("cleaned_pose_count"), default=-1)
        if raw_count != expected_raw or raw_count != _non_negative_int(
            job.get("raw_pose_count"), default=-1
        ):
            errors.append(f"raw count mismatch for {video_id}")
        if cleaned_count != expected_cleaned or cleaned_count != _non_negative_int(
            job.get("cleaned_pose_count"), default=-1
        ):
            errors.append(f"cleaned count mismatch for {video_id}")
        if raw_count != cleaned_count:
            errors.append(f"raw/cleaned count mismatch for {video_id}")
        raw_pose_total += raw_count

    if dict(sorted(derived_status_counts.items())) != summary.get("status_counts"):
        errors.append("batch summary status counts mismatch")
    for field, actual in (
        ("source_frame_count", overall["source_frames"]),
        ("raw_pose_count", raw_pose_total),
        ("cleaned_pose_count", overall["pose_records"]),
    ):
        if _non_negative_int(summary.get(field), default=-1) != actual:
            errors.append(f"batch summary {field} mismatch")

    report = {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "status": "failed" if errors else "passed",
        "updated_at": datetime.now(UTC).isoformat(),
        "cache_contract_sha256": contract_sha256,
        "manifest_sha256": cache_contract.get("manifest_sha256"),
        "model_sha256": cache_contract.get("model_sha256"),
        "parameters": {
            key: cache_contract.get(key)
            for key in (
                "model",
                "device",
                "confidence",
                "iou",
                "tracker",
                "normalize_coordinates",
            )
        },
        "videos": len(seen_video_ids),
        "overall": _finalize_counts(overall),
        "by_dataset": _finalize_groups(by_dataset),
        "by_scene": _finalize_groups(by_scene),
        "by_action": _finalize_groups(by_action),
        "videos_with_low_quality_records": sorted(low_quality_videos),
        "videos_with_multiple_tracks": sorted(multiple_track_videos),
        "videos_without_last_frame_detection": sorted(missing_last_frame_videos),
        "part_files": part_files,
        "errors": errors,
    }
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary_path = args.summary or (
        args.cache_dir / "batches" / args.batch_id / "preparation_summary.json"
    )
    output_path = args.output or (
        args.cache_dir / "batches" / args.batch_id / "validation_summary.json"
    )
    report = validate_pose_cache(cache_dir=args.cache_dir, summary_path=summary_path)
    _atomic_write_json(output_path, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "videos": report.get("videos", 0),
                "errors": len(report["errors"]),
                "overall": report.get("overall", {}),
                "output": output_path.as_posix(),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "passed" else 2


def _new_counts() -> dict[str, int]:
    return {field: 0 for field in COUNT_FIELDS}


def _update_record_counts(counts: dict[str, int], record: Mapping[str, Any]) -> None:
    counts["pose_records"] += 1
    if record.get("quality_state") != "usable":
        counts["low_quality_records"] += 1
    window = record.get("window_quality")
    if isinstance(window, dict):
        for key, target in (
            ("usable_for_gait", "usable_gait_records"),
            ("usable_for_sit_stand", "usable_sit_stand_records"),
            ("usable_for_near_fall", "usable_near_fall_records"),
        ):
            if window.get(key) is True:
                counts[target] += 1
    keypoints = record.get("keypoints")
    if not isinstance(keypoints, list):
        return
    for point in keypoints:
        if not isinstance(point, dict):
            continue
        counts["keypoints"] += 1
        is_valid = point.get("valid") is True
        if is_valid:
            counts["valid_keypoints"] += 1
        if point.get("name") in LOWER_BODY_NAMES:
            counts["lower_body_points"] += 1
            if is_valid:
                counts["valid_lower_body_points"] += 1
        if point.get("source") == "interpolated":
            counts["interpolated_points"] += 1
        if point.get("is_jump_outlier") is True:
            counts["jump_outliers"] += 1


def _validate_record(record: Mapping[str, Any], *, prefix: str, errors: list[str]) -> None:
    if record.get("coordinate_system") != "image_normalized_0_1":
        errors.append(f"{prefix} has invalid coordinate system")
    timestamp = record.get("timestamp_sec")
    if not _finite_number(timestamp) or float(timestamp) < 0:
        errors.append(f"{prefix} has invalid timestamp")
    keypoints = record.get("keypoints")
    if not isinstance(keypoints, list) or len(keypoints) != 17:
        errors.append(f"{prefix} must contain 17 keypoints")
        return
    names: set[str] = set()
    for point_index, raw_point in enumerate(keypoints):
        if not isinstance(raw_point, dict):
            errors.append(f"{prefix} keypoint {point_index} is not an object")
            continue
        name = str(raw_point.get("name", ""))
        if not name or name in names:
            errors.append(f"{prefix} has duplicate or empty keypoint name")
        names.add(name)
        for field in ("x", "y", "score"):
            value = raw_point.get(field)
            if not _unit_number(value):
                errors.append(f"{prefix} keypoint {name} has invalid {field}")
        if raw_point.get("valid") is True:
            for field in ("x_smooth", "y_smooth", "quality_weight"):
                if not _unit_number(raw_point.get(field)):
                    errors.append(f"{prefix} keypoint {name} has invalid {field}")
    if names != EXPECTED_KEYPOINT_NAMES:
        errors.append(f"{prefix} has an invalid keypoint name set")


def _count_jsonl(path: Path, *, errors: list[str], label: str) -> int:
    count = 0
    try:
        for _, _ in _iter_jsonl(path):
            count += 1
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"invalid {label}: {exc}")
    return count


def _iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            yield line_number, value


def _finalize_groups(groups: Mapping[str, Mapping[str, int]]) -> dict[str, dict[str, Any]]:
    return {key: _finalize_counts(value) for key, value in sorted(groups.items())}


def _finalize_counts(raw_counts: Mapping[str, int]) -> dict[str, Any]:
    counts: dict[str, Any] = {field: int(raw_counts.get(field, 0)) for field in COUNT_FIELDS}
    counts["detected_frame_coverage"] = _ratio(
        counts["detected_frames"], counts["source_frames"]
    )
    counts["valid_keypoint_ratio"] = _ratio(counts["valid_keypoints"], counts["keypoints"])
    counts["valid_lower_body_ratio"] = _ratio(
        counts["valid_lower_body_points"], counts["lower_body_points"]
    )
    counts["interpolated_point_ratio"] = _ratio(
        counts["interpolated_points"], counts["keypoints"]
    )
    counts["usable_gait_record_ratio"] = _ratio(
        counts["usable_gait_records"], counts["pose_records"]
    )
    counts["usable_sit_stand_record_ratio"] = _ratio(
        counts["usable_sit_stand_records"], counts["pose_records"]
    )
    counts["usable_near_fall_record_ratio"] = _ratio(
        counts["usable_near_fall_records"], counts["pose_records"]
    )
    return counts


def _failed_report(errors: list[str], part_files: list[str]) -> dict[str, Any]:
    return {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "status": "failed",
        "updated_at": datetime.now(UTC).isoformat(),
        "videos": 0,
        "overall": _finalize_counts(_new_counts()),
        "by_dataset": {},
        "by_scene": {},
        "by_action": {},
        "videos_with_low_quality_records": [],
        "videos_with_multiple_tracks": [],
        "videos_without_last_frame_detection": [],
        "part_files": part_files,
        "errors": errors,
    }


def _action_code(video_id: str) -> str:
    match = re.search(r"_a(\d{3})_", video_id, flags=re.IGNORECASE)
    return f"A{match.group(1)}" if match else "unknown"


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _unit_number(value: Any) -> bool:
    return _finite_number(value) and 0.0 <= float(value) <= 1.0


def _non_negative_int(value: Any, *, default: int) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else default


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def _sha256_json(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(f"{path.suffix}.part")
    partial.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


if __name__ == "__main__":
    raise SystemExit(main())
