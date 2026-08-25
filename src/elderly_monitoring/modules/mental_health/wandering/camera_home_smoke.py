"""Fresh W5D-03B home-video tracking, episode, shape, and context smoke."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import cv2
import ultralytics

from elderly_monitoring.modules.fall_risk.tracking import run_yolov8_bytetrack
from elderly_monitoring.modules.mental_health.wandering.camera_adapter import (
    canonical_json_bytes,
    canonical_jsonl_bytes,
    load_camera_inputs,
)
from elderly_monitoring.modules.mental_health.wandering.camera_context_review import (
    DeterministicFakeContextProvider,
    DisabledContextProvider,
    _build_context_review_row,
    _validate_context_review,
    extract_episode_frames,
    select_eligible_episode_results,
)
from elderly_monitoring.modules.mental_health.wandering.camera_episode_boundary import (
    build_camera_episode_boundary_development_bundle,
)
from elderly_monitoring.modules.mental_health.wandering.camera_episode_pipeline import (
    _build_episode_result,
    _validate_episode_result,
)
from elderly_monitoring.modules.mental_health.wandering.camera_episode_proposal_inference import (
    build_camera_episode_proposal_inference_bundle,
)
from elderly_monitoring.modules.mental_health.wandering.camera_inference import (
    load_camera_config,
)


HOME_SMOKE_SUMMARY_SCHEMA_VERSION = "wandering-camera-home-smoke-summary-v1"
HOME_SMOKE_MANIFEST_SCHEMA_VERSION = "wandering-camera-home-smoke-manifest-v1"


class CameraHomeSmokeError(ValueError):
    """Home smoke input, identity, pipeline, or output failed closed."""


@dataclass(frozen=True)
class CameraHomeSmokeBuildResult:
    output_dir: Path
    proposal_count: int
    episode_result_count: int
    context_review_count: int
    home_smoke_status: str


def build_camera_home_smoke_bundle(
    *,
    project_root: str | Path,
    input_video_path: str | Path,
    output_dir: str | Path,
    run_id: str,
    source_video_id: str,
    source_group_id: str,
    device_id: str,
    setup_id: str,
    stream_epoch: str,
    media_ref: str,
    participant_id: str,
    session_id: str,
    clock_domain_id: str,
    timezone_name: str | None,
    home_annotation_status: str = "not_provided",
    capture_started_at: str | None = None,
    provider_mode: str = "fake",
    detector_model: str = "yolov8n.pt",
    detector_confidence: float = 0.25,
    detector_iou: float = 0.5,
    tracker_config: str = "bytetrack.yaml",
) -> CameraHomeSmokeBuildResult:
    """Run one raw MP4 through fresh tracking, episode, shape, and context."""

    output = Path(output_dir).resolve(strict=False)
    if output.exists():
        raise FileExistsError(f"home smoke output already exists: {output}")
    root = Path(project_root).resolve(strict=True)
    video = Path(input_video_path).resolve(strict=True)
    identifiers = {
        "run_id": _token(run_id, "run_id"),
        "source_video_id": _token(source_video_id, "source_video_id"),
        "source_group_id": _token(source_group_id, "source_group_id"),
        "device_id": _token(device_id, "device_id"),
        "setup_id": _token(setup_id, "setup_id"),
        "stream_epoch": _token(stream_epoch, "stream_epoch"),
        "participant_id": _token(participant_id, "participant_id"),
        "session_id": _token(session_id, "session_id"),
        "clock_domain_id": _token(clock_domain_id, "clock_domain_id"),
    }
    if provider_mode not in {"fake", "disabled"}:
        raise CameraHomeSmokeError("home smoke provider_mode must be fake or disabled")
    if home_annotation_status not in {"not_provided", "unlabeled", "available"}:
        raise CameraHomeSmokeError("home_annotation_status is invalid")
    if (
        not isinstance(media_ref, str)
        or not media_ref
        or "\\" in media_ref
        or PurePosixPath(media_ref).is_absolute()
        or ".." in PurePosixPath(media_ref).parts
    ):
        raise CameraHomeSmokeError("media_ref must be a safe relative POSIX reference")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    started_at = _utc_now()
    try:
        input_dir = stage / "inputs"
        input_dir.mkdir()
        tracking_path = input_dir / "tracking.jsonl"
        sidecar_path = input_dir / "media_sidecar.json"
        media = _track_raw_video(
            video=video,
            tracking_path=tracking_path,
            source_video_id=identifiers["source_video_id"],
            source_group_id=identifiers["source_group_id"],
            device_id=identifiers["device_id"],
            setup_id=identifiers["setup_id"],
            stream_epoch=identifiers["stream_epoch"],
            media_ref=media_ref,
            timezone_name=timezone_name,
            capture_started_at=capture_started_at,
            detector_model=detector_model,
            detector_confidence=detector_confidence,
            detector_iou=detector_iou,
            tracker_config=tracker_config,
        )
        sidecar_path.write_bytes(canonical_json_bytes(media))
        camera_config = load_camera_config(
            root / "configs/modules/wandering_camera_v1.yaml"
        )
        adapter = load_camera_inputs(tracking_path, sidecar_path, camera_config)
        if adapter.media_sidecar["source_sha256"] != _sha256_file(video):
            raise CameraHomeSmokeError("fresh sidecar video SHA-256 mismatch")

        profile_path = (
            root
            / "configs/modules/wandering_camera_episode_boundary_home_v1.yaml"
        ).resolve(strict=True)
        proposals_dir = stage / "proposals"
        proposal_result = build_camera_episode_boundary_development_bundle(
            project_root=root,
            development_profile_path=profile_path,
            tracking_jsonl_path=tracking_path,
            media_sidecar_path=sidecar_path,
            output_dir=proposals_dir,
        )
        shape_index_path = stage / "shape_batch_index.jsonl"
        shape_index_path.write_bytes(
            canonical_jsonl_bytes(
                [
                    {
                        "bundle_id": f"home-{identifiers['source_video_id']}",
                        "proposal_bundle_dir": proposals_dir.as_posix(),
                        "tracking_jsonl": tracking_path.as_posix(),
                        "media_sidecar": sidecar_path.as_posix(),
                        "participant_id": identifiers["participant_id"],
                        "session_id": identifiers["session_id"],
                        "camera_setup_id": identifiers["setup_id"],
                        "clock_domain_id": identifiers["clock_domain_id"],
                    }
                ]
            )
        )
        classification_dir = stage / "classification"
        shape_result = build_camera_episode_proposal_inference_bundle(
            project_root=root,
            config_path=(
                root / "configs/modules/wandering_camera_episode_proposal_shape_v1.yaml"
            ),
            batch_index_path=shape_index_path,
            output_dir=classification_dir,
        )
        shape_rows = _read_jsonl(
            classification_dir / "proposal_shape_predictions.jsonl"
        )
        shape_index_path.unlink()
        proposal_summary = json.loads(
            (proposals_dir / "summary.json").read_text(encoding="utf-8")
        )
        trusted_minimum_track_confidence = float(
            proposal_summary["minimum_track_confidence"]
        )
        profile_sha = _sha256_file(profile_path)
        policy = {
            "schema_version": HOME_SMOKE_SUMMARY_SCHEMA_VERSION,
            "validation_scope": "home_truth_free_smoke",
            "provider_mode": provider_mode,
            "frame_positions": ["start", "middle", "end"],
            "bbox_max_gap_seconds": 0.75,
            "crop_padding_ratio": 0.12,
        }
        config_sha = hashlib.sha256(canonical_json_bytes(policy)).hexdigest()
        artifact_row = {
            "participant_id": identifiers["participant_id"],
            "session_id": identifiers["session_id"],
            "source_video_id": identifiers["source_video_id"],
            "artifacts": {
                "tracking": {
                    "path": (output / "inputs/tracking.jsonl").as_posix(),
                    "sha256": adapter.source_tracking_sha256,
                },
                "media_sidecar": {
                    "path": (output / "inputs/media_sidecar.json").as_posix(),
                    "sha256": _sha256_file(sidecar_path),
                },
                "truth": None,
            },
        }
        context = {"row": artifact_row, "sidecar": media}
        episodes: list[dict[str, Any]] = []
        for shape_row in shape_rows:
            episode = _build_episode_result(
                prediction=shape_row,
                index_row=artifact_row,
                config_id="w5d03b-home-truth-free-smoke-v1",
                config_sha256=config_sha,
                profile_id=str(proposal_summary["producer_config_id"]),
                profile_sha256=profile_sha,
                context=context,
                index_descriptor=None,
            )
            episode["quality_flags"] = list(
                dict.fromkeys(
                    [
                        *episode["quality_flags"],
                        "home_truth_free_smoke",
                        "human_truth_not_consumed",
                    ]
                )
            )
            episode["source_refs"] = [
                ref
                for ref in episode["source_refs"]
                if ref["ref_type"] != "development_index"
            ]
            episode["source_refs"].append(
                {
                    "ref_type": "source_video",
                    "ref_id": identifiers["source_video_id"],
                    "artifact_path": video.as_posix(),
                    "sha256": media["source_sha256"],
                }
            )
            _validate_episode_result(episode)
            episodes.append(episode)
        episodes.sort(
            key=lambda row: (
                float(row["start_sec"]),
                int(row["technical_segment_index"]),
                str(row["episode_id"]),
            )
        )
        eligible, skipped = select_eligible_episode_results(episodes)
        if not eligible:
            raise CameraHomeSmokeError("home smoke produced no context-eligible episode")
        provider = (
            DeterministicFakeContextProvider()
            if provider_mode == "fake"
            else DisabledContextProvider()
        )
        context_rows: list[dict[str, Any]] = []
        tracking_rows = [
            dict(row)
            for row in adapter.normalized_rows
            if float(row["track_confidence"])
            >= trusted_minimum_track_confidence
        ]
        context_policy_id = "w5d03b-home-wandering-or-uncertain-v1"
        context_policy_sha = hashlib.sha256(
            context_policy_id.encode("utf-8")
        ).hexdigest()
        final_shape_path = output / "classification/proposal_shape_predictions.jsonl"
        source_refs = [
            {
                "ref_type": "source_video",
                "ref_id": identifiers["source_video_id"],
                "artifact_path": video.as_posix(),
                "sha256": media["source_sha256"],
            },
            {
                "ref_type": "tracking",
                "ref_id": identifiers["source_video_id"],
                "artifact_path": (output / "inputs/tracking.jsonl").as_posix(),
                "sha256": adapter.source_tracking_sha256,
            },
            {
                "ref_type": "proposal_shape_predictions",
                "ref_id": identifiers["source_video_id"],
                "artifact_path": final_shape_path.as_posix(),
                "sha256": _sha256_file(
                    classification_dir / "proposal_shape_predictions.jsonl"
                ),
            },
        ]
        for episode in eligible:
            frames = extract_episode_frames(
                episode=episode,
                video_path=video,
                tracking_rows=tracking_rows,
                stage_output_root=stage,
                final_output_root=output,
                jpeg_quality=85,
                bbox_max_gap_seconds=0.75,
                crop_padding_ratio=0.12,
            )
            row = _build_context_review_row(
                episode=episode,
                frames=frames,
                provider=provider,
                config_id="w5d03b-home-context-smoke-v1",
                config_sha256=config_sha,
                policy_id=context_policy_id,
                policy_sha256=context_policy_sha,
                source_refs=source_refs,
            )
            row["quality_flags"] = list(
                dict.fromkeys(
                    [
                        *row["quality_flags"],
                        "home_truth_free_smoke",
                        "fake_context_engineering_only"
                        if provider_mode == "fake"
                        else "context_provider_disabled",
                    ]
                )
            )
            _validate_context_review(row)
            context_rows.append(row)
        context_rows.sort(
            key=lambda row: (float(row["start_sec"]), str(row["episode_id"]))
        )
        episode_payload = canonical_jsonl_bytes(episodes)
        context_payload = canonical_jsonl_bytes(context_rows)
        (stage / "episode_results.jsonl").write_bytes(episode_payload)
        (stage / "context_reviews.jsonl").write_bytes(context_payload)
        finished_at = _utc_now()
        episode_status_counts = Counter(str(row["status"]) for row in episodes)
        context_status_counts = Counter(str(row["status"]) for row in context_rows)
        smoke_status = (
            "error"
            if episode_status_counts["error"] or context_status_counts["error"]
            else "uncertain"
            if not context_rows or context_status_counts["unavailable"]
            else "ready"
        )
        summary = {
            "schema_version": HOME_SMOKE_SUMMARY_SCHEMA_VERSION,
            "module": "mental_health",
            "run_id": identifiers["run_id"],
            "started_at": started_at,
            "finished_at": finished_at,
            "status": smoke_status,
            "delivery_status": "completed_with_home_smoke",
            "validation_scope": "home_truth_free_smoke",
            "home_input_status": "available",
            "home_annotation_status": home_annotation_status,
            "home_smoke_status": smoke_status,
            "source_video_id": identifiers["source_video_id"],
            "person_id": identifiers["participant_id"],
            "session_id": identifiers["session_id"],
            "camera_setup_id": identifiers["setup_id"],
            "clock_domain_id": identifiers["clock_domain_id"],
            "timezone": timezone_name,
            "capture_started_at": capture_started_at,
            "tracking_mode": "fresh_full_mp4_yolov8_bytetrack",
            "segmenter_profile_id": proposal_summary["producer_config_id"],
            "trusted_minimum_track_confidence": (
                trusted_minimum_track_confidence
            ),
            "input_counts": {"source_videos": 1, "human_truth_rows": 0},
            "output_counts": {
                "tracking_observations": len(adapter.observations),
                "proposals": proposal_result.proposal_count,
                "episode_results": len(episodes),
                "context_eligible": len(eligible),
                "context_reviews": len(context_rows),
            },
            "episode_status_counts": dict(sorted(episode_status_counts.items())),
            "context_status_counts": dict(sorted(context_status_counts.items())),
            "context_skipped_counts": skipped,
            "model_forward_invocation_count": shape_result.model_forward_invocation_count,
            "human_truth_consumed": False,
            "algorithm_event_emitted": False,
            "medical_diagnosis_emitted": False,
            "backend_or_frontend_implemented": False,
            "quality_flags": [
                "home_truth_free_smoke",
                "automatic_boundary_not_human_accepted",
                "human_truth_not_consumed",
                "context_fake_engineering_only"
                if provider_mode == "fake"
                else "context_provider_disabled",
                "capture_clock_unavailable"
                if capture_started_at is None
                else "capture_clock_available",
            ],
        }
        summary_payload = canonical_json_bytes(summary)
        (stage / "run_summary.json").write_bytes(summary_payload)
        readme = _readme(summary)
        verification = _verification(summary, media)
        (stage / "README.md").write_text(readme, encoding="utf-8")
        (stage / "VERIFICATION.md").write_text(verification, encoding="utf-8")
        artifacts = {}
        for name, schema_version in (
            ("episode_results.jsonl", "wandering-handoff-episode-result-v1"),
            ("context_reviews.jsonl", "wandering-handoff-context-review-v2"),
            ("run_summary.json", HOME_SMOKE_SUMMARY_SCHEMA_VERSION),
            ("README.md", None),
            ("VERIFICATION.md", None),
        ):
            payload = (stage / name).read_bytes()
            artifacts[name] = {
                "schema_version": schema_version,
                "byte_count": len(payload),
                "record_count": len(payload.splitlines())
                if name.endswith(".jsonl")
                else 1,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        manifest = {
            "schema_version": HOME_SMOKE_MANIFEST_SCHEMA_VERSION,
            "module": "mental_health",
            "run_id": identifiers["run_id"],
            "validation_scope": "home_truth_free_smoke",
            "home_input_status": "available",
            "home_annotation_status": home_annotation_status,
            "home_smoke_status": smoke_status,
            "artifacts": artifacts,
            "source_video": {
                "artifact_path": video.as_posix(),
                "sha256": media["source_sha256"],
            },
            "algorithm_event_emitted": False,
            "medical_diagnosis_emitted": False,
        }
        (stage / "manifest.json").write_bytes(canonical_json_bytes(manifest))
        _fsync_tree(stage)
        stage.replace(output)
        return CameraHomeSmokeBuildResult(
            output_dir=output,
            proposal_count=proposal_result.proposal_count,
            episode_result_count=len(episodes),
            context_review_count=len(context_rows),
            home_smoke_status=smoke_status,
        )
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def _track_raw_video(
    *,
    video: Path,
    tracking_path: Path,
    source_video_id: str,
    source_group_id: str,
    device_id: str,
    setup_id: str,
    stream_epoch: str,
    media_ref: str,
    timezone_name: str | None,
    capture_started_at: str | None,
    detector_model: str,
    detector_confidence: float,
    detector_iou: float,
    tracker_config: str,
) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        capture.release()
        raise CameraHomeSmokeError("cannot open home MP4")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    capture.release()
    if fps <= 0.0 or width <= 0 or height <= 0 or frame_count <= 0:
        raise CameraHomeSmokeError("home MP4 metadata is incomplete")
    run_yolov8_bytetrack(
        video_path=video,
        output_path=tracking_path,
        model_name=detector_model,
        scene_region="unknown",
        person_id_prefix="temporary_track",
        confidence_threshold=detector_confidence,
        iou_threshold=detector_iou,
        tracker_config=tracker_config,
        max_frames=None,
    )
    return {
        "schema_version": "wandering-media-v1",
        "source_video_id": source_video_id,
        "source_group_id": source_group_id,
        "device_id": device_id,
        "setup_id": setup_id,
        "stream_epoch": stream_epoch,
        "media_ref": media_ref,
        "source_sha256": _sha256_file(video),
        "tracking_jsonl_sha256": _sha256_file(tracking_path),
        "video_width": width,
        "video_height": height,
        "nominal_fps": fps,
        "duration_sec": frame_count / fps,
        "capture_started_at": capture_started_at,
        "timezone": timezone_name,
        "coordinate_system": "pixel_xyxy_top_left",
        "detector": {
            "backend": "ultralytics_yolo",
            "model": Path(detector_model).name,
            "version": str(ultralytics.__version__),
        },
        "tracker": {
            "backend": "bytetrack",
            "config": Path(tracker_config).name,
            "version": str(ultralytics.__version__),
        },
        "fixed_camera_assumed": True,
        "camera_motion_state": "not_checked",
        "authorization_status": "authorized_camera_engineering_smoke",
        "deidentification_status": "anonymous_internal_development",
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise CameraHomeSmokeError(
                f"shape output line {line_number} is not an object"
            )
        rows.append(dict(value))
    if not rows:
        raise CameraHomeSmokeError("shape output is empty")
    return rows


def _readme(summary: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# W5D-03B home truth-free smoke",
            "",
            f"Run ID: `{summary['run_id']}`",
            f"Home smoke status: `{summary['home_smoke_status']}`",
            f"Source video: `{summary['source_video_id']}`",
            f"Episode results: `{summary['output_counts']['episode_results']}`",
            f"Context reviews: `{summary['output_counts']['context_reviews']}`",
            "",
            "This is one raw MP4 -> fresh YOLOv8/ByteTrack -> automatic episode -> ",
            "frozen TopoWander shape -> three-frame context engineering smoke. Human ",
            "truth was not consumed. Fake context verifies the adapter contract and is ",
            "not context-label accuracy evidence or a medical conclusion.",
            "",
        ]
    )


def _verification(summary: Mapping[str, Any], media: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Verification",
            "",
            f"- validation_scope: `{summary['validation_scope']}`",
            f"- tracking_mode: `{summary['tracking_mode']}`",
            f"- source_sha256: `{media['source_sha256']}`",
            f"- tracking_jsonl_sha256: `{media['tracking_jsonl_sha256']}`",
            f"- human_truth_consumed: `{str(summary['human_truth_consumed']).lower()}`",
            f"- algorithm_event_emitted: `{str(summary['algorithm_event_emitted']).lower()}`",
            f"- capture_started_at: `{summary['capture_started_at']}`",
            "- Automatic boundaries remain proposals and fake context is engineering-only.",
            "",
        ]
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _token(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 160
        or any(char in value for char in "\\/\r\n\0")
    ):
        raise CameraHomeSmokeError(f"{field} must be a safe non-empty token")
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _fsync_tree(root: Path) -> None:
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        with path.open("rb") as handle:
            os.fsync(handle.fileno())


__all__ = [
    "HOME_SMOKE_MANIFEST_SCHEMA_VERSION",
    "HOME_SMOKE_SUMMARY_SCHEMA_VERSION",
    "CameraHomeSmokeBuildResult",
    "CameraHomeSmokeError",
    "build_camera_home_smoke_bundle",
]
