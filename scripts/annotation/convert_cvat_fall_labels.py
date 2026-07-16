from __future__ import annotations

import argparse
from pathlib import Path

from elderly_monitoring.modules.fall_risk.annotations import (
    convert_cvat_xml,
    write_converted_fall_labels,
)


DEFAULT_MANIFEST = Path("data/manifests/fall_risk_video_manifest.jsonl")
DEFAULT_OUTPUT_DIR = Path("data/annotations/fall_risk/generated/v1")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert CVAT for video 1.1 XML/ZIP into traceable fall-risk "
            "action and mapped-event candidates."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Video manifest providing per-video path, FPS, frames and duration.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Candidate output directory; existing outputs are never overwritten.",
    )
    parser.add_argument("--action-output", type=Path, default=None)
    parser.add_argument("--event-output", type=Path, default=None)
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Development-only FPS override; must agree with manifest metadata.",
    )
    parser.add_argument(
        "--development-override",
        action="store_true",
        help="Allow legacy development inputs such as --fps or --file-root.",
    )
    parser.add_argument("--file-root", type=Path, default=None)
    parser.add_argument("--labeler", default="unknown")
    parser.add_argument(
        "--review-status",
        default="pending",
        help="Compatibility option. Only pending is accepted by this converter.",
    )
    parser.add_argument("--subject-id", default="unknown")
    parser.add_argument("--scene", default="home")
    parser.add_argument("--view", default="fixed_camera")
    parser.add_argument("--time-tolerance-sec", type=float, default=0.001)
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if args.review_status != "pending":
        parser.error(
            "cannot promote converted labels to reviewed/final; use a validated review log"
        )
    if (args.fps is not None or args.file_root is not None) and not args.development_override:
        parser.error("--fps and --file-root require --development-override")
    if (args.action_output is None) != (args.event_output is None):
        parser.error("--action-output and --event-output must be provided together")
    if not args.manifest.is_file():
        if not args.development_override or args.fps is None:
            parser.error(f"manifest does not exist: {args.manifest}")
        manifest_path = None
    else:
        manifest_path = args.manifest

    action_output = args.action_output or args.output_dir / "action_labels.jsonl"
    event_output = args.event_output or args.output_dir / "event_labels.jsonl"
    try:
        converted = convert_cvat_xml(
            args.input,
            manifest_path=manifest_path,
            fps=args.fps if manifest_path is not None else args.fps,
            file_root=args.file_root,
            labeler=args.labeler,
            review_status="pending",
            default_subject_id=args.subject_id,
            default_scene=args.scene,
            default_view=args.view,
            time_tolerance_sec=args.time_tolerance_sec,
        )
        counts = write_converted_fall_labels(
            converted,
            action_output_path=action_output,
            event_output_path=event_output,
            overwrite=False,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    print(f"action_labels={counts['action_labels']} output={action_output}")
    print(f"event_labels={counts['event_labels']} output={event_output}")
    if converted.identity_metadata_present:
        print(
            "warning: CVAT identity metadata is present in the source export; "
            "values were not copied to generated labels."
        )


if __name__ == "__main__":
    main()
