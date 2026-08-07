from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.prepare.validate_fall_pose_cache import validate_pose_cache


KEYPOINT_NAMES = (
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
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _record(*, frame_id: int = 0, person_id: str = "person_001") -> dict[str, object]:
    return {
        "frame_id": frame_id,
        "timestamp_sec": frame_id / 30.0,
        "person_id": person_id,
        "track_id": 1,
        "coordinate_system": "image_normalized_0_1",
        "quality_state": "usable",
        "keypoints": [
            {
                "name": name,
                "x": 0.5,
                "y": 0.5,
                "score": 0.9,
                "x_smooth": 0.5,
                "y_smooth": 0.5,
                "valid": True,
                "source": "observed",
                "quality_weight": 0.9,
                "is_jump_outlier": False,
            }
            for name in KEYPOINT_NAMES
        ],
        "window_quality": {
            "usable_for_gait": True,
            "usable_for_sit_stand": True,
            "usable_for_near_fall": True,
        },
    }


def _build_cache(root: Path) -> tuple[Path, Path]:
    cache_dir = root / "cache"
    video_id = "ntu_rgbd_s001_p001_r001_a008_c001"
    raw_path = cache_dir / "raw" / f"{video_id}.jsonl"
    cleaned_path = cache_dir / "cleaned" / f"{video_id}.jsonl"
    _write_json(raw_path, _record())
    _write_json(cleaned_path, _record())
    cache_contract = {
        "schema_version": "fall-pose-cache-v1",
        "manifest_sha256": "manifest-hash",
        "model_sha256": "model-hash",
        "model": "yolov8n-pose.pt",
        "device": "mps",
        "confidence": 0.25,
        "iou": 0.5,
        "tracker": "bytetrack.yaml",
        "normalize_coordinates": True,
    }
    _write_json(cache_dir / "cache_manifest.json", cache_contract)
    contract_sha256 = hashlib.sha256(
        json.dumps(cache_contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    state = {
        "schema_version": "fall-pose-cache-job-v1",
        "video_id": video_id,
        "dataset": "ntu_rgbd",
        "status": "completed",
        "raw_path": raw_path.as_posix(),
        "cleaned_path": cleaned_path.as_posix(),
        "raw_pose_count": 1,
        "cleaned_pose_count": 1,
        "raw_size_bytes": raw_path.stat().st_size,
        "cleaned_size_bytes": cleaned_path.stat().st_size,
    }
    _write_json(cache_dir / "state" / f"{video_id}.json", state)
    summary_path = cache_dir / "batches" / "ntu_rgbd" / "preparation_summary.json"
    batch_contract = {
        "schema_version": "fall-pose-cache-batch-v1",
        "batch_id": "ntu_rgbd",
        "datasets": ["ntu_rgbd"],
        "job_count": 1,
        "job_ids_sha256": "job-ids-hash",
        "cache_contract_sha256": contract_sha256,
    }
    _write_json(summary_path.parent / "batch_manifest.json", batch_contract)
    _write_json(
        summary_path,
        {
            **batch_contract,
            "final": True,
            "processed_job_count": 1,
            "remaining_job_count": 0,
            "status_counts": {"completed": 1},
            "failures": [],
            "source_frame_count": 1,
            "raw_pose_count": 1,
            "cleaned_pose_count": 1,
            "jobs": [
                {
                    "video_id": video_id,
                    "dataset": "ntu_rgbd",
                    "scene": "lab",
                    "frame_count": 1,
                    "status": "completed",
                    "raw_pose_count": 1,
                    "cleaned_pose_count": 1,
                }
            ],
        },
    )
    return cache_dir, summary_path


def test_validate_pose_cache_parses_complete_batch_and_reports_quality() -> None:
    with TemporaryDirectory() as temp_dir:
        cache_dir, summary_path = _build_cache(Path(temp_dir))

        report = validate_pose_cache(cache_dir=cache_dir, summary_path=summary_path)

        assert report["status"] == "passed"
        assert report["errors"] == []
        assert report["overall"]["videos"] == 1
        assert report["overall"]["pose_records"] == 1
        assert report["overall"]["detected_frame_coverage"] == 1.0
        assert report["overall"]["valid_keypoint_ratio"] == 1.0
        assert report["by_action"]["A008"]["videos"] == 1


def test_validate_pose_cache_fails_on_count_mismatch_and_partial_file() -> None:
    with TemporaryDirectory() as temp_dir:
        cache_dir, summary_path = _build_cache(Path(temp_dir))
        video_id = "ntu_rgbd_s001_p001_r001_a008_c001"
        cleaned_path = cache_dir / "cleaned" / f"{video_id}.jsonl"
        with cleaned_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_record(frame_id=1)) + "\n")
        (cache_dir / "raw" / f"{video_id}.jsonl.part").write_text("partial", encoding="utf-8")

        report = validate_pose_cache(cache_dir=cache_dir, summary_path=summary_path)

        assert report["status"] == "failed"
        assert len(report["part_files"]) == 1
        assert any("cleaned size mismatch" in error for error in report["errors"])
