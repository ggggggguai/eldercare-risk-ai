from __future__ import annotations

import argparse
import json
from pathlib import Path

from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.face_tracking import (
    DlibFaceLandmarker,
    TemplateFaceLandmarker,
    default_dlib_predictor_path,
)
from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.optical_flow import (
    OpticalFlowConfig,
)
from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.preprocess import (
    SequencePreprocessor,
    audit_preprocess_artifacts,
    write_preprocess_artifact,
    write_preprocess_audit,
    write_preprocess_reports,
    write_visualization,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build fixed MHSSA-TGCN u/v/magnitude flow and patch artifacts."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT
        / "data/manifests/microexpression/sequence_manifest_v1.jsonl",
    )
    parser.add_argument("--dataset", default="smic_hs")
    parser.add_argument("--role", default="classification")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-id")
    parser.add_argument(
        "--landmark-mode", choices=("template", "dlib"), default="dlib"
    )
    parser.add_argument(
        "--predictor",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data/processed/microexpression/flow_v1",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=PROJECT_ROOT / "data/manifests/microexpression",
    )
    parser.add_argument("--visualize-count", type=int, default=1)
    return parser.parse_args()


def load_records(args: argparse.Namespace) -> list[dict[str, object]]:
    records = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = [
        record
        for record in records
        if record["source_dataset"] == args.dataset
        and record["sample_role"] == args.role
        and (args.sample_id is None or record["sample_id"] == args.sample_id)
    ]
    if args.limit is not None:
        selected = selected[: args.limit]
    return selected


def main() -> int:
    args = parse_args()
    if args.landmark_mode == "dlib":
        predictor = args.predictor or default_dlib_predictor_path(PROJECT_ROOT)
        landmarker = DlibFaceLandmarker(predictor)
    else:
        landmarker = TemplateFaceLandmarker()
    config = OpticalFlowConfig()
    preprocessor = SequencePreprocessor(landmarker, config)
    selected = load_records(args)
    if not selected:
        raise SystemExit("No manifest records matched the requested filters")

    output_records: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    for index, record in enumerate(selected):
        try:
            result = preprocessor.process(record)
            artifact_path = (
                args.output_dir
                / str(record["source_dataset"])
                / str(record["subject_id"])
                / f"{record['sample_id']}.npz"
            )
            output_records.append(write_preprocess_artifact(result, artifact_path))
            if index < args.visualize_count:
                write_visualization(
                    result,
                    args.output_dir
                    / "visualizations"
                    / f"{record['sample_id']}.png",
                )
        except Exception as exc:
            failures.append(
                {"sample_id": str(record["sample_id"]), "error": str(exc)}
            )

    suffix = f"_{args.dataset}_{args.role}_v1"
    summary = write_preprocess_reports(
        output_records,
        manifest_path=args.report_dir / f"flow_artifact_manifest{suffix}.jsonl",
        summary_path=args.report_dir / f"flow_preprocess_summary{suffix}.json",
        config=config,
        landmark_mode=landmarker.version,
        landmark_asset_sha256=getattr(landmarker, "predictor_sha256", None),
    )
    artifact_audit = audit_preprocess_artifacts(output_records)
    audit_path = args.report_dir / f"flow_artifact_audit{suffix}.json"
    artifact_audit_sha256 = write_preprocess_audit(artifact_audit, audit_path)
    result_payload = {
        "status": "pass" if not failures and output_records else "partial_or_fail",
        "requested": len(selected),
        "completed": len(output_records),
        "failed": len(failures),
        "failures": failures,
        "summary": summary,
        "artifact_audit": {
            **artifact_audit,
            "path": audit_path.resolve().as_posix(),
            "sha256": artifact_audit_sha256,
        },
    }
    print(json.dumps(result_payload, ensure_ascii=True, indent=2))
    return (
        0
        if not failures
        and output_records
        and summary["status"] == "pass"
        and artifact_audit["status"] == "pass"
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
