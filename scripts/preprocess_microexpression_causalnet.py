from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.causalnet_preprocess import (
    CausalNetPreprocessConfig,
    SPATIAL_VARIANTS,
    audit_artifacts,
    build_four_route_input,
    directory_fingerprint,
    sha256_file,
    source_code_hash,
    write_artifact,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_ROOT = PROJECT_ROOT / "src/elderly_monitoring/modules/mental_health/submodules/facial_affect_clue"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CausalNet four-route SMIC artifacts")
    parser.add_argument("--sequence-manifest", type=Path, default=PROJECT_ROOT / "data/manifests/microexpression/sequence_manifest_v1.jsonl")
    parser.add_argument("--paper-manifest", type=Path, default=PROJECT_ROOT / "data/manifests/microexpression/paper_v2/flow_artifact_manifest_smic_hs_classification_combined_v2.jsonl")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/preprocessing/microexpression_causalnet_v1.yaml")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/processed/microexpression/causalnet_v1")
    parser.add_argument("--report-dir", type=Path, default=PROJECT_ROOT / "data/manifests/microexpression/causalnet_v1")
    parser.add_argument("--dataset", default="smic_hs")
    parser.add_argument("--role", default="classification")
    parser.add_argument("--variant", choices=SPATIAL_VARIANTS)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-id")
    parser.add_argument("--visualize-count", type=int, default=6)
    return parser.parse_args()


def load_config(path: Path, variant: str | None) -> CausalNetPreprocessConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    allowed = {field.name for field in fields(CausalNetPreprocessConfig)}
    values = {key: value for key, value in payload.items() if key in allowed}
    if variant:
        values["spatial_variant"] = variant
    return CausalNetPreprocessConfig(**values)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _visualization(inputs, path: Path) -> None:
    import cv2
    import numpy as np

    tiles = []
    for route in inputs:
        rgb = np.transpose(route, (1, 2, 0))
        tile = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        tiles.append(cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))
    canvas = cv2.vconcat([cv2.hconcat(tiles[:2]), cv2.hconcat(tiles[2:])])
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = cv2.imencode(path.suffix or ".png", canvas)[1]
    if encoded is None:
        raise RuntimeError(f"Unable to encode visualization: {path}")
    encoded.tofile(path)


def main() -> int:
    args = parse_args()
    config = load_config(args.config, args.variant)
    sequence_rows = load_jsonl(args.sequence_manifest)
    paper_rows = load_jsonl(args.paper_manifest)
    sequence_by_id = {str(row["sample_id"]): row for row in sequence_rows}
    paper_by_id = {str(row["sample_id"]): row for row in paper_rows}
    selected_ids = [
        sample_id for sample_id, row in sequence_by_id.items()
        if row.get("source_dataset") == args.dataset
        and row.get("sample_role") == args.role
        and sample_id in paper_by_id
        and (args.sample_id is None or sample_id == args.sample_id)
    ]
    selected_ids.sort()
    if args.limit is not None:
        selected_ids = selected_ids[:args.limit]
    if not selected_ids:
        raise SystemExit("No matching SMIC records")

    v1_path = PROJECT_ROOT / "data/processed/microexpression/flow_v1"
    paper_v2_path = PROJECT_ROOT / "data/processed/microexpression/paper_v2/combined"
    history_before = {"flow_v1": directory_fingerprint(v1_path), "paper_v2_combined": directory_fingerprint(paper_v2_path)}
    code_paths = [MODULE_ROOT / "causalnet_preprocess.py", Path(__file__)]
    code_hash = source_code_hash(code_paths)
    source_manifest_hash = hashlib.sha256(args.sequence_manifest.read_bytes() + args.paper_manifest.read_bytes()).hexdigest()
    output_records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    visualized = 0
    for sample_id in selected_ids:
        record = dict(sequence_by_id[sample_id])
        paper = paper_by_id[sample_id]
        for key in ("onset_index", "apex_index", "offset_index", "onset_frame", "apex_frame", "offset_frame", "alignment_transform", "apex_method", "frame_annotation_source"):
            if key in paper:
                record[key] = paper[key]
        try:
            inputs, metadata = build_four_route_input(record, config)
            metadata["code_hash"] = code_hash
            metadata["source_manifest_hash"] = source_manifest_hash
            artifact_path = args.output_dir / config.spatial_variant / args.dataset / str(record["subject_id"]) / f"{sample_id}.npz"
            output_records.append(write_artifact(inputs, metadata, artifact_path))
            if visualized < args.visualize_count:
                _visualization(inputs, args.output_dir / config.spatial_variant / "visualizations" / f"{sample_id}.png")
                visualized += 1
        except Exception as exc:
            failures.append({"sample_id": sample_id, "error": repr(exc)})

    args.report_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.report_dir / f"causalnet_artifact_manifest_{args.dataset}_{config.spatial_variant}_v1.jsonl"
    manifest_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in output_records), encoding="utf-8")
    history_after = {"flow_v1": directory_fingerprint(v1_path), "paper_v2_combined": directory_fingerprint(paper_v2_path)}
    audit = audit_artifacts(output_records, expected_variant=config.spatial_variant)
    history_status = "pass" if history_before == history_after else "fail"
    if history_status != "pass":
        audit["status"] = "fail"
        audit["issue_count"] = int(audit["issue_count"]) + 1
        audit["issues"].append({"code": "historical_directory_modified"})
    audit_payload = {
        "schema_version": "causalnet_artifact_audit_v1",
        "task_id": "FLOW-ME-004",
        "status": audit["status"],
        "audit": audit,
        "history_directory_fingerprint": {"status": history_status, "before": history_before, "after": history_after},
        "selected": len(selected_ids),
        "completed": len(output_records),
        "failed": len(failures),
        "failures": failures,
        "variant": config.spatial_variant,
        "config_path": args.config.resolve().as_posix(),
        "config_sha256": sha256_file(args.config),
        "source_manifest_hash": source_manifest_hash,
        "code_hash": code_hash,
    }
    audit_path = args.report_dir / f"causalnet_artifact_audit_{args.dataset}_{config.spatial_variant}_v1.json"
    audit_path.write_text(json.dumps(audit_payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    failures_path = args.report_dir / f"causalnet_preprocess_failures_{args.dataset}_{config.spatial_variant}_v1.json"
    failures_path.write_text(
        json.dumps(
            {
                "schema_version": "causalnet_preprocess_failures_v1",
                "task_id": "FLOW-ME-004",
                "failure_count": len(failures),
                "failures": failures,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    history_path = args.report_dir / f"causalnet_history_fingerprint_{args.dataset}_{config.spatial_variant}_v1.json"
    history_path.write_text(
        json.dumps(
            {
                "schema_version": "causalnet_history_fingerprint_v1",
                "task_id": "FLOW-ME-004",
                "status": history_status,
                "before": history_before,
                "after": history_after,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    summary = {
        "schema_version": "causalnet_preprocess_summary_v1",
        "task_id": "FLOW-ME-004",
        "status": "pass" if not failures and len(output_records) == len(selected_ids) and audit["status"] == "pass" else "partial_or_fail",
        "requested": len(selected_ids),
        "completed": len(output_records),
        "failed": len(failures),
        "manifest_path": manifest_path.resolve().as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "audit_path": audit_path.resolve().as_posix(),
        "audit_sha256": sha256_file(audit_path),
        "failures_path": failures_path.resolve().as_posix(),
        "failures_sha256": sha256_file(failures_path),
        "history_fingerprint_path": history_path.resolve().as_posix(),
        "history_fingerprint_sha256": sha256_file(history_path),
        "variant": config.spatial_variant,
        "full_face_generated": config.spatial_variant == "full_face",
        "full_face_code_path": (MODULE_ROOT / "causalnet_preprocess.py").resolve().as_posix(),
    }
    summary_path = args.report_dir / f"causalnet_preprocess_summary_{args.dataset}_{config.spatial_variant}_v1.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"summary": summary, "audit": audit_payload}, ensure_ascii=True, indent=2))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
