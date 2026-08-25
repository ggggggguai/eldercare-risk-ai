"""Fresh non-overwriting batch runner for the home-development repair profile."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from elderly_monitoring.modules.mental_health.wandering.camera_adapter import (
    canonical_json_bytes,
    canonical_jsonl_bytes,
)
from elderly_monitoring.modules.mental_health.wandering.camera_episode_boundary import (
    build_camera_episode_boundary_development_bundle,
)
from elderly_monitoring.modules.mental_health.wandering.camera_episode_proposal_inference import (
    build_camera_episode_proposal_inference_bundle,
)


HOME_REPAIR_SUMMARY_SCHEMA_VERSION = "wandering-camera-home-repair-summary-v1"
HOME_REPAIR_VIDEO_SUMMARY_SCHEMA_VERSION = (
    "wandering-camera-home-repair-video-summary-v1"
)


class CameraHomeRepairError(ValueError):
    """A cached tracking input or home-repair batch binding is invalid."""


@dataclass(frozen=True)
class CameraHomeRepairBuildResult:
    output_dir: Path
    video_count: int
    proposal_count: int
    ready_count: int
    unavailable_count: int


def build_camera_home_repair_batch(
    *,
    project_root: str | Path,
    tracking_root: str | Path,
    output_dir: str | Path,
    source_video_ids: Sequence[str] | None = None,
) -> CameraHomeRepairBuildResult:
    """Re-run proposal and shape stages from cached tracking in a fresh directory."""

    root = Path(project_root).resolve(strict=True)
    inputs_root = Path(tracking_root).resolve(strict=True)
    output = Path(output_dir).resolve(strict=False)
    if output.exists():
        raise FileExistsError(f"home repair output already exists: {output}")
    if source_video_ids:
        video_ids = sorted({_safe_video_id(value) for value in source_video_ids})
    else:
        video_ids = sorted(
            item.name
            for item in inputs_root.iterdir()
            if item.is_dir()
            and (item / "inputs/tracking.jsonl").is_file()
            and (item / "inputs/media_sidecar.json").is_file()
        )
    if not video_ids:
        raise CameraHomeRepairError("home repair found no cached tracking bundles")

    profile_path = (
        root / "configs/modules/wandering_camera_episode_boundary_home_v1.yaml"
    ).resolve(strict=True)
    shape_config_path = (
        root / "configs/modules/wandering_camera_episode_proposal_shape_v1.yaml"
    ).resolve(strict=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        batch_rows: list[dict[str, str]] = []
        proposal_summaries: dict[str, dict[str, Any]] = {}
        for video_id in video_ids:
            source_dir = (inputs_root / video_id / "inputs").resolve(strict=True)
            tracking_path = (source_dir / "tracking.jsonl").resolve(strict=True)
            sidecar_path = (source_dir / "media_sidecar.json").resolve(strict=True)
            media = _load_json(sidecar_path)
            if media.get("source_video_id") != video_id:
                raise CameraHomeRepairError(
                    f"cached sidecar identity differs for {video_id}"
                )
            proposal_dir = stage / "tasks" / video_id / "proposals"
            build_camera_episode_boundary_development_bundle(
                project_root=root,
                development_profile_path=profile_path,
                tracking_jsonl_path=tracking_path,
                media_sidecar_path=sidecar_path,
                output_dir=proposal_dir,
            )
            proposal_summaries[video_id] = _load_json(
                proposal_dir / "summary.json"
            )
            batch_rows.append(
                {
                    "bundle_id": f"home-repair-{video_id}",
                    "proposal_bundle_dir": proposal_dir.as_posix(),
                    "tracking_jsonl": tracking_path.as_posix(),
                    "media_sidecar": sidecar_path.as_posix(),
                    "participant_id": "home-development-participant",
                    "session_id": "home-repair-20260822",
                    "camera_setup_id": str(media["setup_id"]),
                    "clock_domain_id": "home-video-relative-time",
                }
            )

        batch_index = stage / "shape_batch_index.jsonl"
        batch_index.write_bytes(canonical_jsonl_bytes(batch_rows))
        shape_result = build_camera_episode_proposal_inference_bundle(
            project_root=root,
            config_path=shape_config_path,
            batch_index_path=batch_index,
            output_dir=stage / "classification",
        )
        predictions = _load_jsonl(
            stage / "classification/proposal_shape_predictions.jsonl"
        )
        predictions_by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in predictions:
            predictions_by_video[str(row["source_video_id"])].append(row)

        video_summaries: list[dict[str, Any]] = []
        for video_id in video_ids:
            proposal_summary = proposal_summaries[video_id]
            video_predictions = predictions_by_video[video_id]
            prediction_counts = Counter(
                str(row["prediction_status"]) for row in video_predictions
            )
            pattern_counts = Counter(
                str(row["predicted_pattern"])
                for row in video_predictions
                if row["predicted_pattern"] is not None
            )
            video_summaries.append(
                {
                    "schema_version": HOME_REPAIR_VIDEO_SUMMARY_SCHEMA_VERSION,
                    "source_video_id": video_id,
                    "segmenter_profile_id": proposal_summary["producer_config_id"],
                    "trusted_minimum_track_confidence": proposal_summary[
                        "minimum_track_confidence"
                    ],
                    "source_observation_count": proposal_summary[
                        "source_observation_count"
                    ],
                    "accepted_observation_count": proposal_summary[
                        "accepted_observation_count"
                    ],
                    "proposal_count": proposal_summary["proposal_count"],
                    "locomotion_proposal_count": proposal_summary[
                        "locomotion_proposal_count"
                    ],
                    "proposal_status_counts": proposal_summary[
                        "proposal_status_counts"
                    ],
                    "proposal_duration_seconds": proposal_summary[
                        "proposal_duration_seconds"
                    ],
                    "prediction_status_counts": dict(sorted(prediction_counts.items())),
                    "predicted_pattern_counts": dict(sorted(pattern_counts.items())),
                    "model_forward_invocation_count": sum(
                        not bool(row["model_invocation_skipped"])
                        for row in video_predictions
                    ),
                }
            )
        (stage / "video_summary.jsonl").write_bytes(
            canonical_jsonl_bytes(video_summaries)
        )
        summary = {
            "schema_version": HOME_REPAIR_SUMMARY_SCHEMA_VERSION,
            "status": "home_development_repair_batch_completed",
            "validation_scope": "same_participant_home_development",
            "video_count": len(video_ids),
            "proposal_count": shape_result.proposal_count,
            "ready_count": shape_result.ready_count,
            "unavailable_count": shape_result.unavailable_count,
            "boundary_uncertain_count": shape_result.boundary_uncertain_count,
            "model_forward_invocation_count": (
                shape_result.model_forward_invocation_count
            ),
            "segmenter_profile_id": "home-recall-confidence070-v1",
            "trusted_minimum_track_confidence": 0.70,
            "tracking_reused": True,
            "tracking_or_detection_truth_consumed": False,
            "behavior_truth_consumed": False,
            "automatic_boundaries_human_accepted": False,
            "source_tracking_root": inputs_root.as_posix(),
        }
        (stage / "summary.json").write_bytes(canonical_json_bytes(summary))
        batch_index.unlink()
        _fsync_tree(stage)
        stage.replace(output)
        return CameraHomeRepairBuildResult(
            output_dir=output,
            video_count=len(video_ids),
            proposal_count=shape_result.proposal_count,
            ready_count=shape_result.ready_count,
            unavailable_count=shape_result.unavailable_count,
        )
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def _safe_video_id(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or Path(value).name != value
        or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for character in value)
    ):
        raise CameraHomeRepairError("source video ID is invalid")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CameraHomeRepairError(f"cannot read JSON input: {path}") from exc
    if not isinstance(value, dict):
        raise CameraHomeRepairError(f"JSON input is not an object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        values = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CameraHomeRepairError(f"cannot read JSONL input: {path}") from exc
    if not all(isinstance(value, dict) for value in values):
        raise CameraHomeRepairError(f"JSONL input contains a non-object: {path}")
    return values


def _fsync_tree(root: Path) -> None:
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        with path.open("rb") as handle:
            os.fsync(handle.fileno())


__all__ = [
    "CameraHomeRepairBuildResult",
    "CameraHomeRepairError",
    "build_camera_home_repair_batch",
]
