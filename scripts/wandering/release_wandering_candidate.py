#!/usr/bin/env python3
"""Freeze, build, load, or evaluate the manifest-bound M0-R candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from elderly_monitoring.modules.mental_health.wandering.release import (
    CandidateManifestError,
    RELEASE_IMPLEMENTATION_SOURCE_FILES,
    ReleaseArtifactError,
    WPReleaseDataError,
    build_candidate_bundle,
    freeze_release_implementation_identity,
    run_authorized_frozen_wp_score,
    run_primary_validation_parity,
    run_wp_records_evaluation,
)


RELEASE_SOURCE_FILES = RELEASE_IMPLEMENTATION_SOURCE_FILES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    freeze = subparsers.add_parser(
        "freeze-implementation",
        help="Persist exact release.py/CLI/test bytes and write release implementation identity.",
    )
    freeze.add_argument("--project-root", type=Path, default=Path.cwd())
    freeze.add_argument("--archive-path", type=Path, required=True)
    freeze.add_argument("--identity-path", type=Path, required=True)
    freeze.add_argument("--git-head", required=True)

    build = subparsers.add_parser(
        "build-bundle", help="Create a compact model/config/manifest candidate directory."
    )
    build.add_argument("--project-root", type=Path, default=Path.cwd())
    build.add_argument("--training-identity", type=Path, required=True)
    build.add_argument("--expected-training-identity-sha256", required=True)
    build.add_argument("--release-identity", type=Path, required=True)
    build.add_argument("--expected-release-identity-sha256", required=True)
    build.add_argument("--output-dir", type=Path, required=True)

    parity = subparsers.add_parser(
        "validation-parity",
        help="Fresh-load the candidate and reproduce the frozen WP validation cohort only.",
    )
    parity.add_argument("--project-root", type=Path, default=Path.cwd())
    parity.add_argument("--manifest", type=Path, required=True)
    parity.add_argument("--expected-manifest-sha256", required=True)
    parity.add_argument("--output-dir", type=Path, required=True)
    parity.add_argument("--tolerance", type=float, default=1.0e-7)

    score = subparsers.add_parser(
        "score-frozen-wp",
        help="Run the sole authorized formal score entry using the internal frozen-WP accessor.",
    )
    score.add_argument("--project-root", type=Path, required=True)
    score.add_argument("--manifest", type=Path, required=True)
    score.add_argument("--expected-manifest-sha256", required=True)
    score.add_argument("--output-dir", type=Path, required=True)

    evaluate = subparsers.add_parser(
        "evaluate-records",
        help="Evaluate one externally supplied fixed WP validation JSONL cohort.",
    )
    evaluate.add_argument("--manifest", type=Path, required=True)
    evaluate.add_argument("--expected-manifest-sha256", required=True)
    evaluate.add_argument("--records", type=Path, required=True)
    evaluate.add_argument("--expected-split", choices=("validation",), required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.mode == "freeze-implementation":
            result = freeze_release_implementation_identity(
                project_root=args.project_root,
                source_files=RELEASE_SOURCE_FILES,
                archive_path=args.archive_path,
                identity_path=args.identity_path,
                git_head=args.git_head,
            )
            payload = {
                "mode": args.mode,
                "identity_path": str(result.identity_path),
                "identity_sha256": result.identity_sha256,
                "source_archive_path": str(result.archive_path),
                "source_archive_sha256": result.identity["source_archive"]["sha256"],
                "availability": "current_machine_local_only",
            }
        elif args.mode == "build-bundle":
            result = build_candidate_bundle(
                project_root=args.project_root,
                training_identity_path=args.training_identity,
                expected_training_identity_sha256=args.expected_training_identity_sha256,
                release_identity_path=args.release_identity,
                expected_release_identity_sha256=args.expected_release_identity_sha256,
                output_dir=args.output_dir,
            )
            payload = {
                "mode": args.mode,
                "bundle_dir": str(result.bundle_dir),
                "candidate_manifest": str(result.manifest_path),
                "candidate_manifest_sha256": result.manifest_sha256,
                "availability": "current_machine_local_only",
            }
        elif args.mode == "validation-parity":
            output = run_primary_validation_parity(
                project_root=args.project_root,
                manifest_path=args.manifest,
                expected_manifest_sha256=args.expected_manifest_sha256,
                output_dir=args.output_dir,
                tolerance=args.tolerance,
            )
            parity_path = output / "parity.json"
            payload = {
                "mode": args.mode,
                "output_dir": str(output),
                "parity_sha256": hashlib.sha256(parity_path.read_bytes()).hexdigest(),
                "wp_test_accessed_by_release_path": False,
            }
        elif args.mode == "score-frozen-wp":
            output = run_authorized_frozen_wp_score(
                project_root=args.project_root,
                manifest_path=args.manifest,
                expected_manifest_sha256=args.expected_manifest_sha256,
                output_dir=args.output_dir,
                active_cli_path=Path(__file__).resolve(),
            )
            payload = {
                "mode": args.mode,
                "output_dir": str(output),
                "candidate_manifest_sha256": args.expected_manifest_sha256,
                "metrics_not_printed": True,
                "formal_score_committed": True,
            }
        else:
            output = run_wp_records_evaluation(
                manifest_path=args.manifest,
                expected_manifest_sha256=args.expected_manifest_sha256,
                records_path=args.records,
                expected_split=args.expected_split,
                output_dir=args.output_dir,
                batch_size=args.batch_size,
            )
            payload = {
                "mode": args.mode,
                "phase": args.expected_split,
                "output_dir": str(output),
                "metrics_not_printed": True,
            }
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)
        return 0
    except (
        CandidateManifestError,
        ReleaseArtifactError,
        WPReleaseDataError,
        FileExistsError,
        OSError,
        ValueError,
    ) as exc:
        print(f"M0-RH {args.mode} failed: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
