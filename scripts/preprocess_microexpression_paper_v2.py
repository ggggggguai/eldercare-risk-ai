from __future__ import annotations

import argparse
from dataclasses import fields
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import yaml

from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.apex_spotting import (
    DCRoIsConfig,
)
from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.face_tracking import (
    DlibFaceLandmarker,
    TemplateFaceLandmarker,
    default_dlib_predictor_path,
)
from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.paper_preprocess import (
    PREPROCESS_VARIANTS,
    PaperPreprocessConfig,
    PaperSequencePreprocessor,
    audit_paper_artifacts,
    sha256_file,
    write_paper_artifact,
    write_paper_audit,
    write_paper_reports,
    write_paper_visualization,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_ROOT = (
    PROJECT_ROOT
    / "src/elderly_monitoring/modules/mental_health/submodules/facial_affect_clue"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build thesis-priority SMIC D&C-RoIs, optical-strain and ordered "
            "5x5 patch artifacts in the isolated paper_v2 path."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT
        / "data/manifests/microexpression/sequence_manifest_v1.jsonl",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT
        / "configs/preprocessing/microexpression_smic_paper_v2.yaml",
    )
    parser.add_argument("--dataset", default="smic_hs")
    parser.add_argument("--role", default="classification")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-id")
    parser.add_argument("--variant", choices=PREPROCESS_VARIANTS)
    parser.add_argument("--flow-estimator", choices=("farneback", "tvl1"))
    parser.add_argument(
        "--landmark-mode", choices=("template", "dlib"), default="dlib"
    )
    parser.add_argument("--predictor", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data/processed/microexpression/paper_v2",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=PROJECT_ROOT / "data/manifests/microexpression/paper_v2",
    )
    parser.add_argument("--visualize-count", type=int, default=6)
    return parser.parse_args()


def _dataclass_kwargs(cls: type, values: dict[str, Any]) -> dict[str, Any]:
    allowed = {field.name for field in fields(cls)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {unknown}")
    return values


def load_configs(
    path: Path,
    *,
    variant: str | None,
    flow_estimator: str | None,
) -> tuple[PaperPreprocessConfig, DCRoIsConfig]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    preprocess_values = dict(payload.get("preprocess", {}))
    apex_values = dict(payload.get("apex", {}))
    if variant:
        preprocess_values["preprocess_variant"] = variant
    if flow_estimator:
        preprocess_values["flow_estimator"] = flow_estimator
    preprocess = PaperPreprocessConfig(
        **_dataclass_kwargs(PaperPreprocessConfig, preprocess_values)
    )
    apex = DCRoIsConfig(**_dataclass_kwargs(DCRoIsConfig, apex_values))
    return preprocess, apex


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


def directory_fingerprint(path: Path) -> dict[str, object]:
    files = sorted(item for item in path.rglob("*") if item.is_file()) if path.exists() else []
    digest = sha256()
    for item in files:
        relative = item.relative_to(path).as_posix().encode()
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(item)))
    return {
        "path": path.resolve().as_posix(),
        "file_count": len(files),
        "sha256": digest.hexdigest(),
    }


def main() -> int:
    args = parse_args()
    config, apex_config = load_configs(
        args.config,
        variant=args.variant,
        flow_estimator=args.flow_estimator,
    )
    if args.landmark_mode == "dlib":
        predictor = args.predictor or default_dlib_predictor_path(PROJECT_ROOT)
        landmarker = DlibFaceLandmarker(predictor)
    else:
        landmarker = TemplateFaceLandmarker()
    preprocessor = PaperSequencePreprocessor(landmarker, config, apex_config)
    selected = load_records(args)
    if not selected:
        raise SystemExit("No manifest records matched the requested filters")

    v1_path = PROJECT_ROOT / "data/processed/microexpression/flow_v1"
    v1_guard_before = directory_fingerprint(v1_path)
    output_records: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    visualized_labels: set[str] = set()
    visualization_count = 0
    for record in selected:
        try:
            result = preprocessor.process(record)
            artifact_path = (
                args.output_dir
                / config.preprocess_variant
                / str(record["source_dataset"])
                / str(record["subject_id"])
                / f"{record['sample_id']}.npz"
            )
            output_records.append(write_paper_artifact(result, artifact_path))
            label = str(record.get("label_name"))
            should_visualize = (
                visualization_count < args.visualize_count
                and (label not in visualized_labels or len(visualized_labels) >= 3)
            )
            if should_visualize:
                write_paper_visualization(
                    result,
                    args.output_dir
                    / config.preprocess_variant
                    / "visualizations"
                    / f"{record['sample_id']}.png",
                )
                visualization_count += 1
                visualized_labels.add(label)
        except Exception as exc:
            failures.append(
                {"sample_id": str(record["sample_id"]), "error": repr(exc)}
            )

    suffix = f"_{args.dataset}_{args.role}_{config.preprocess_variant}_v2"
    manifest_path = args.report_dir / f"flow_artifact_manifest{suffix}.jsonl"
    summary_path = args.report_dir / f"flow_preprocess_summary{suffix}.json"
    summary = write_paper_reports(
        output_records,
        manifest_path=manifest_path,
        summary_path=summary_path,
        config=config,
        apex_config=apex_config,
        landmark_mode=landmarker.version,
        landmark_asset_sha256=getattr(landmarker, "predictor_sha256", None),
        input_manifest_path=args.manifest,
        source_code_paths=(
            MODULE_ROOT / "apex_spotting.py",
            MODULE_ROOT / "paper_preprocess.py",
            Path(__file__),
        ),
        config_path=args.config,
    )
    artifact_audit = audit_paper_artifacts(output_records)
    v1_guard_after = directory_fingerprint(v1_path)
    artifact_audit["historical_v1_guard"] = {
        "status": "pass" if v1_guard_before == v1_guard_after else "fail",
        "before": v1_guard_before,
        "after": v1_guard_after,
    }
    if v1_guard_before != v1_guard_after:
        artifact_audit["status"] = "fail"
        artifact_audit["issue_count"] = int(artifact_audit["issue_count"]) + 1
        artifact_audit["issues"].append({"code": "historical_v1_modified"})
    audit_path = args.report_dir / f"flow_artifact_audit{suffix}.json"
    audit_sha256 = write_paper_audit(artifact_audit, audit_path)
    success = (
        not failures
        and len(output_records) == len(selected)
        and summary["status"] == "pass"
        and artifact_audit["status"] == "pass"
    )
    result_payload = {
        "status": "pass" if success else "partial_or_fail",
        "task_id": "FLOW-ME-002",
        "requested": len(selected),
        "completed": len(output_records),
        "failed": len(failures),
        "failures": failures,
        "variant": config.preprocess_variant,
        "flow_estimator": config.flow_estimator,
        "summary": summary,
        "artifact_audit": {
            **artifact_audit,
            "path": audit_path.resolve().as_posix(),
            "sha256": audit_sha256,
        },
    }
    print(json.dumps(result_payload, ensure_ascii=True, indent=2))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
