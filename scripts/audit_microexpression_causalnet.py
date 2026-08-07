from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.causalnet import (  # noqa: E402
    ARCHITECTURE_CORRECTION,
    MODEL_SCHEMA_VERSION,
    code_hash as model_code_hash,
)
from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.causalnet_dataset import (  # noqa: E402
    EXPECTED_INPUT_SHAPE,
    EXPECTED_ROUTE_ORDER,
)
from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.causalnet_preprocess import (  # noqa: E402
    CAUSALNET_UPSTREAM_COMMIT,
    CAUSALNET_UPSTREAM_LICENSE,
    CAUSALNET_UPSTREAM_REPOSITORY,
    CausalNetPreprocessConfig,
    sha256_file,
    source_code_hash,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Independently audit FLOW-ME-004 and MODEL-ME-008")
    parser.add_argument("--sequence-manifest", type=Path, default=PROJECT_ROOT / "data/manifests/microexpression/sequence_manifest_v1.jsonl")
    parser.add_argument("--paper-manifest", type=Path, default=PROJECT_ROOT / "data/manifests/microexpression/paper_v2/flow_artifact_manifest_smic_hs_classification_combined_v2.jsonl")
    parser.add_argument("--artifact-manifest", type=Path, default=PROJECT_ROOT / "data/manifests/microexpression/causalnet_v1/causalnet_artifact_manifest_smic_hs_source_compatible_roi_v1.jsonl")
    parser.add_argument("--artifact-audit", type=Path, default=PROJECT_ROOT / "data/manifests/microexpression/causalnet_v1/causalnet_artifact_audit_smic_hs_source_compatible_roi_v1.json")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/preprocessing/microexpression_causalnet_v1.yaml")
    parser.add_argument("--visualization-dir", type=Path, default=PROJECT_ROOT / "data/processed/microexpression/causalnet_v1/source_compatible_roi/visualizations")
    parser.add_argument("--model-smoke-report", type=Path, default=PROJECT_ROOT / "reports/microexpression/causalnet_v1/MODEL-ME-008-CausalNet-GPU-SMOKE/model_me_008_smoke_report.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/manifests/microexpression/causalnet_v1/causalnet_flow_model_completion_audit_v1.json")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def add_issue(issues: list[dict[str, Any]], code: str, **details: Any) -> None:
    issues.append({"code": code, **details})


def load_config(path: Path) -> CausalNetPreprocessConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    allowed = {field.name for field in fields(CausalNetPreprocessConfig)}
    return CausalNetPreprocessConfig(**{key: value for key, value in payload.items() if key in allowed})


def audit_artifacts(args: argparse.Namespace, issues: list[dict[str, Any]]) -> dict[str, Any]:
    sequence = {
        str(row["sample_id"]): row
        for row in load_jsonl(args.sequence_manifest)
        if row.get("source_dataset") == "smic_hs" and row.get("sample_role") == "classification"
    }
    paper = {str(row["sample_id"]): row for row in load_jsonl(args.paper_manifest)}
    artifacts = load_jsonl(args.artifact_manifest)
    artifact_by_id = {str(row["sample_id"]): row for row in artifacts}
    if len(artifacts) != len(artifact_by_id):
        add_issue(issues, "duplicate_artifact_sample_id")
    expected_ids = set(sequence) & set(paper)
    if len(expected_ids) != 164:
        add_issue(issues, "unexpected_frozen_source_count", count=len(expected_ids))
    if set(artifact_by_id) != expected_ids:
        add_issue(
            issues,
            "artifact_coverage_mismatch",
            missing=sorted(expected_ids - set(artifact_by_id)),
            unexpected=sorted(set(artifact_by_id) - expected_ids),
        )

    preprocess_paths = [
        SRC_ROOT / "elderly_monitoring/modules/mental_health/submodules/facial_affect_clue/causalnet_preprocess.py",
        PROJECT_ROOT / "scripts/preprocess_microexpression_causalnet.py",
    ]
    expected_code_hash = source_code_hash(preprocess_paths)
    expected_source_hash = hashlib.sha256(args.sequence_manifest.read_bytes() + args.paper_manifest.read_bytes()).hexdigest()
    expected_config_fingerprint = load_config(args.config).fingerprint()
    for sample_id, record in artifact_by_id.items():
        source = sequence[sample_id]
        frozen = paper[sample_id]
        path = Path(str(record["artifact_path"]))
        if not path.is_file():
            add_issue(issues, "missing_artifact", sample_id=sample_id)
            continue
        if sha256_file(path) != record.get("artifact_sha256"):
            add_issue(issues, "artifact_hash_mismatch", sample_id=sample_id)
        try:
            with np.load(path, allow_pickle=False) as artifact:
                inputs = np.asarray(artifact["inputs"])
                label = int(np.asarray(artifact["label"]).item())
                metadata = json.loads(str(np.asarray(artifact["metadata_json"]).item()))
        except Exception as exc:
            add_issue(issues, "artifact_read_error", sample_id=sample_id, error=repr(exc))
            continue
        if inputs.shape != EXPECTED_INPUT_SHAPE or not np.isfinite(inputs).all():
            add_issue(issues, "artifact_tensor_contract", sample_id=sample_id, shape=list(inputs.shape))
        if label != int(source["label"]) or label != int(frozen["label"]):
            add_issue(issues, "artifact_label_mismatch", sample_id=sample_id)
        for key in ("onset_index", "apex_index", "offset_index", "onset_frame", "apex_frame", "offset_frame", "frame_annotation_source", "apex_method"):
            if metadata.get(key) != frozen.get(key):
                add_issue(issues, "frozen_metadata_mismatch", sample_id=sample_id, field=key)
        if not np.array_equal(np.asarray(metadata.get("alignment_transform")), np.asarray(frozen.get("alignment_transform"))):
            add_issue(issues, "alignment_transform_mismatch", sample_id=sample_id)
        if metadata.get("frame_dir") != source.get("frame_dir"):
            add_issue(issues, "frame_dir_mismatch", sample_id=sample_id)
        if tuple(metadata.get("route_order", ())) != EXPECTED_ROUTE_ORDER:
            add_issue(issues, "route_order_mismatch", sample_id=sample_id)
        if metadata.get("spatial_variant") != "source_compatible_roi":
            add_issue(issues, "spatial_variant_mismatch", sample_id=sample_id)
        if metadata.get("code_hash") != expected_code_hash:
            add_issue(issues, "preprocess_code_hash_mismatch", sample_id=sample_id)
        if metadata.get("source_manifest_hash") != expected_source_hash:
            add_issue(issues, "source_manifest_hash_mismatch", sample_id=sample_id)
        if metadata.get("preprocess_config_sha256") != expected_config_fingerprint:
            add_issue(issues, "preprocess_config_hash_mismatch", sample_id=sample_id)
        if metadata.get("preprocessing_config_hash") != expected_config_fingerprint:
            add_issue(issues, "canonical_preprocessing_config_hash_mismatch", sample_id=sample_id)
        if metadata.get("source_commit") != CAUSALNET_UPSTREAM_COMMIT:
            add_issue(issues, "source_commit_mismatch", sample_id=sample_id)
        if set(metadata.get("flow_channel_definition", {})) != {
            "channel_0",
            "channel_1",
            "channel_2",
        }:
            add_issue(issues, "flow_channel_definition_mismatch", sample_id=sample_id)

    visualizations = sorted(args.visualization_dir.glob("*.png"))
    decoded = [
        cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        for path in visualizations
    ]
    readable = sum(image is not None for image in decoded)
    nonblank = sum(image is not None and float(image.std()) > 1.0 for image in decoded)
    if len(visualizations) < 6 or readable < 6 or nonblank < 6:
        add_issue(
            issues,
            "insufficient_visualizations",
            count=len(visualizations),
            readable=readable,
            nonblank=nonblank,
        )
    history_audit = json.loads(args.artifact_audit.read_text(encoding="utf-8"))
    guard = history_audit.get("history_directory_fingerprint", {})
    if history_audit.get("status") != "pass" or guard.get("status") != "pass" or guard.get("before") != guard.get("after"):
        add_issue(issues, "historical_directory_guard_failed")
    return {
        "expected_count": len(expected_ids),
        "artifact_count": len(artifacts),
        "manifest_sha256": sha256_file(args.artifact_manifest),
        "current_preprocess_code_hash": expected_code_hash,
        "current_source_manifest_hash": expected_source_hash,
        "current_config_fingerprint": expected_config_fingerprint,
        "visualization_count": len(visualizations),
        "readable_visualization_count": readable,
        "nonblank_visualization_count": nonblank,
        "history_guard": guard,
    }


def audit_model(args: argparse.Namespace, issues: list[dict[str, Any]]) -> dict[str, Any]:
    report = json.loads(args.model_smoke_report.read_text(encoding="utf-8"))
    if report.get("status") != "pass":
        add_issue(issues, "model_smoke_failed")
    if report.get("architecture_correction") != ARCHITECTURE_CORRECTION:
        add_issue(issues, "model_architecture_correction_mismatch")
    gpu = report.get("production_real_artifact_gpu_smoke", {})
    if (
        gpu.get("status") != "pass"
        or gpu.get("device") != "cuda"
        or gpu.get("input_shape") != [1, 4, 3, 28, 28]
        or gpu.get("logits_shape") != [1, 3]
        or not np.isfinite(gpu.get("loss", np.nan))
        or not np.isfinite(gpu.get("gradient_max", np.nan))
    ):
        add_issue(issues, "production_gpu_smoke_contract")
    checkpoint_info = report.get("checkpoint_smoke", {})
    checkpoint_path = Path(str(checkpoint_info.get("path", "")))
    if not checkpoint_path.is_file() or sha256_file(checkpoint_path) != checkpoint_info.get("sha256"):
        add_issue(issues, "checkpoint_hash_mismatch")
        payload: dict[str, Any] = {}
    else:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected_checkpoint = {
        "model_schema_version": MODEL_SCHEMA_VERSION,
        "upstream_repository": CAUSALNET_UPSTREAM_REPOSITORY,
        "upstream_commit": CAUSALNET_UPSTREAM_COMMIT,
        "upstream_license": CAUSALNET_UPSTREAM_LICENSE,
        "architecture_correction": ARCHITECTURE_CORRECTION,
        "upstream_bit_exact": False,
        "paper_reproduction_claim": False,
        "code_hash": model_code_hash(),
        "source_manifest_hash": sha256_file(args.artifact_manifest),
        "preprocessing_config_hash": sha256_file(args.config),
    }
    for key, expected in expected_checkpoint.items():
        if payload.get(key) != expected:
            add_issue(issues, "checkpoint_metadata_mismatch", field=key)
    if checkpoint_info.get("max_logit_difference") != 0.0:
        add_issue(issues, "checkpoint_round_trip_difference")
    model_source = (SRC_ROOT / "elderly_monitoring/modules/mental_health/submodules/facial_affect_clue/causalnet.py").read_text(encoding="utf-8")
    if "third_party" in model_source or "test_fold" in model_source:
        add_issue(issues, "forbidden_runtime_dependency_or_test_fold")
    return {
        "smoke_report_sha256": sha256_file(args.model_smoke_report),
        "gpu": gpu,
        "checkpoint_path": checkpoint_path.resolve().as_posix() if checkpoint_path else None,
        "checkpoint_sha256": checkpoint_info.get("sha256"),
        "checkpoint_metadata_expected": expected_checkpoint,
    }


def main() -> int:
    args = parse_args()
    issues: list[dict[str, Any]] = []
    artifact_result = audit_artifacts(args, issues)
    model_result = audit_model(args, issues)
    result = {
        "schema_version": "causalnet_flow_model_completion_audit_v1",
        "tasks": ["FLOW-ME-004", "MODEL-ME-008"],
        "status": "pass" if not issues else "fail",
        "issue_count": len(issues),
        "issues": issues,
        "artifacts": artifact_result,
        "model": model_result,
        "training_started": False,
        "final_metrics_claim": False,
        "system_integration_started": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
