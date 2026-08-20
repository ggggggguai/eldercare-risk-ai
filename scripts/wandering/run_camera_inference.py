from __future__ import annotations

import argparse
import hashlib
import tempfile
from pathlib import Path

from elderly_monitoring.modules.fall_risk.tracking import run_yolov8_bytetrack
from elderly_monitoring.modules.mental_health.wandering.camera_adapter import (
    AUTHORIZATION_STATUSES,
    canonical_json_bytes,
)
from elderly_monitoring.modules.mental_health.wandering.camera_inference import (
    build_camera_inference_bundle,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mp4_inputs(args: argparse.Namespace, temporary: Path) -> tuple[Path, Path]:
    required = {
        "source_video_id": args.source_video_id,
        "source_group_id": args.source_group_id,
        "device_id": args.device_id,
        "setup_id": args.setup_id,
        "stream_epoch": args.stream_epoch,
        "media_ref": args.media_ref,
        "authorization_status": args.authorization_status,
        "deidentification_status": args.deidentification_status,
    }
    missing = [name.replace("_", "-") for name, value in required.items() if not value]
    if missing:
        raise ValueError(f"MP4 mode requires explicit fields: {', '.join(missing)}")
    video = args.input_video.resolve(strict=True)
    try:
        import cv2
        import ultralytics
    except ImportError as exc:
        raise RuntimeError("MP4 mode requires the project vision dependencies") from exc
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"cannot open input video: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    capture.release()
    if fps <= 0.0 or width <= 0 or height <= 0 or frame_count <= 0:
        raise ValueError("input video metadata must provide positive width/height/fps/frame count")

    tracking = temporary / "tracking.jsonl"
    run_yolov8_bytetrack(
        video_path=video,
        output_path=tracking,
        model_name=args.detector_model,
        scene_region="unknown",
        person_id_prefix="temporary_track",
        confidence_threshold=args.detector_confidence,
        iou_threshold=args.detector_iou,
        tracker_config=args.tracker_config,
        max_frames=args.max_frames,
    )
    sidecar = {
        "schema_version": "wandering-media-v1",
        "source_video_id": args.source_video_id,
        "source_group_id": args.source_group_id,
        "device_id": args.device_id,
        "setup_id": args.setup_id,
        "stream_epoch": args.stream_epoch,
        "media_ref": args.media_ref,
        "source_sha256": _sha256_file(video),
        "tracking_jsonl_sha256": _sha256_file(tracking),
        "video_width": width,
        "video_height": height,
        "nominal_fps": fps,
        "duration_sec": frame_count / fps,
        "capture_started_at": args.capture_started_at,
        "timezone": args.timezone,
        "coordinate_system": "pixel_xyxy_top_left",
        "detector": {
            "backend": "ultralytics_yolo",
            "model": Path(args.detector_model).name,
            "version": str(ultralytics.__version__),
        },
        "tracker": {
            "backend": "bytetrack",
            "config": Path(args.tracker_config).name,
            "version": str(ultralytics.__version__),
        },
        "fixed_camera_assumed": True,
        "camera_motion_state": args.camera_motion_state,
        "authorization_status": args.authorization_status,
        "deidentification_status": args.deidentification_status,
    }
    sidecar_path = temporary / "media_sidecar.json"
    sidecar_path.write_bytes(canonical_json_bytes(sidecar))
    return tracking, sidecar_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the deterministic wandering step-7 bbox-only offline comparison chain."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--tracking-jsonl", type=Path)
    source.add_argument("--input-video", type=Path)
    parser.add_argument("--media-sidecar", type=Path)
    parser.add_argument("--rf-development-dir", type=Path, required=True)
    parser.add_argument("--expected-rf-development-manifest-sha256", required=True)
    parser.add_argument("--tcn-development-dir", type=Path, required=True)
    parser.add_argument("--expected-tcn-development-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)

    parser.add_argument("--source-video-id")
    parser.add_argument("--source-group-id")
    parser.add_argument("--device-id")
    parser.add_argument("--setup-id")
    parser.add_argument("--stream-epoch")
    parser.add_argument("--media-ref")
    parser.add_argument("--authorization-status", choices=AUTHORIZATION_STATUSES)
    parser.add_argument("--deidentification-status")
    parser.add_argument("--camera-motion-state", choices=("stable", "moved", "not_checked"), default="not_checked")
    parser.add_argument("--capture-started-at", default=None)
    parser.add_argument("--timezone", default=None)
    parser.add_argument("--detector-model", default="yolov8n.pt")
    parser.add_argument("--detector-confidence", type=float, default=0.25)
    parser.add_argument("--detector-iou", type=float, default=0.5)
    parser.add_argument("--tracker-config", default="bytetrack.yaml")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    if args.tracking_jsonl is not None and args.media_sidecar is None:
        parser.error("--tracking-jsonl requires --media-sidecar")
    if args.input_video is not None and args.media_sidecar is not None:
        parser.error("--media-sidecar is only valid with --tracking-jsonl")
    if args.output.exists():
        raise FileExistsError(f"camera output already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="wandering-camera-input-", dir=args.output.parent) as tmpdir:
        if args.input_video is not None:
            tracking, sidecar = _mp4_inputs(args, Path(tmpdir))
        else:
            tracking = args.tracking_jsonl
            sidecar = args.media_sidecar
        result = build_camera_inference_bundle(
            camera_config_path=args.config,
            project_root=args.project_root,
            tracking_jsonl_path=tracking,
            media_sidecar_path=sidecar,
            rf_development_dir=args.rf_development_dir,
            expected_rf_development_manifest_sha256=args.expected_rf_development_manifest_sha256,
            tcn_development_dir=args.tcn_development_dir,
            expected_tcn_development_manifest_sha256=args.expected_tcn_development_manifest_sha256,
            output_dir=args.output,
        )
    print(
        f"Built {result.window_count} camera windows at {result.output_dir} "
        f"(ready={result.ready_window_count}, unavailable={result.unavailable_window_count}, "
        f"inference_error={result.inference_error_count}, manifest_sha256={result.manifest_sha256})"
    )


if __name__ == "__main__":
    main()
