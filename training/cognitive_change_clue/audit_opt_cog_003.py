"""Independent cache, prediction, and selection audit for OPT-COG-003."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:
    from .cache_v34_lightweight import DEFAULT_OUTPUT_ROOT as LIGHTWEIGHT_ROOT
    from .cache_v34_pooling import DEFAULT_OUTPUT_ROOT as POOLING_ROOT
    from .common import WORKSPACE_ROOT, resolve_workspace_path, sha256_file, workspace_relative
    from .train import write_json
    from .train_v34 import REPORT_ROOT, V34TrainingError, verify_prediction_bundle
    from .train_v34_features import VARIANT_CANDIDATES
except ImportError:
    from cache_v34_lightweight import DEFAULT_OUTPUT_ROOT as LIGHTWEIGHT_ROOT  # type: ignore[no-redef]
    from cache_v34_pooling import DEFAULT_OUTPUT_ROOT as POOLING_ROOT  # type: ignore[no-redef]
    from common import (  # type: ignore[no-redef]
        WORKSPACE_ROOT,
        resolve_workspace_path,
        sha256_file,
        workspace_relative,
    )
    from train import write_json  # type: ignore[no-redef]
    from train_v34 import REPORT_ROOT, V34TrainingError, verify_prediction_bundle  # type: ignore[no-redef]
    from train_v34_features import VARIANT_CANDIDATES  # type: ignore[no-redef]


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _audit_lightweight() -> dict[str, Any]:
    manifest_path = LIGHTWEIGHT_ROOT / "cache_manifest.json"
    index_path = LIGHTWEIGHT_ROOT / "lightweight_index.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = _rows(index_path)
    if (
        manifest.get("status") != "complete"
        or manifest.get("official_test_media_read") is not False
        or int(manifest.get("indexed_rows", -1)) != 1377
        or len(rows) != 1377
        or len({row["sample_id"] for row in rows}) != 1377
        or len({row["subject_id"] for row in rows}) != 459
        or {row["official_split"] for row in rows} != {"train"}
        or sha256_file(index_path) != manifest.get("index_sha256")
    ):
        raise V34TrainingError("lightweight feature cache boundary audit failed")
    for row in rows:
        path = resolve_workspace_path(str(row["feature_path"]))
        if sha256_file(path) != row["feature_sha256"]:
            raise V34TrainingError(f"lightweight feature hash mismatch: {row['sample_id']}")
        with np.load(path, allow_pickle=False) as payload:
            for group, dimension in row["dimensions"].items():
                value = np.asarray(payload[group])
                if value.shape != (int(dimension),) or not np.isfinite(value).all():
                    raise V34TrainingError(
                        f"invalid lightweight vector {group}: {row['sample_id']}"
                    )
    return {
        "manifest_path": workspace_relative(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "index_sha256": sha256_file(index_path),
        "rows": len(rows),
        "subjects": 459,
        "missing_counts": manifest["missing_counts"],
    }


def _audit_pooling() -> dict[str, Any]:
    manifest_path = POOLING_ROOT / "cache_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("official_test_media_read") is not False:
        raise V34TrainingError("pooling feature cache boundary audit failed")
    result: dict[str, Any] = {
        "manifest_path": workspace_relative(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "modalities": {},
    }
    for modality, dimension in (("audio_stats", 1536), ("text_mean", 768)):
        index_path = POOLING_ROOT / f"{modality}_index.jsonl"
        rows = _rows(index_path)
        if (
            len(rows) != 1377
            or len({row["sample_id"] for row in rows}) != 1377
            or len({row["subject_id"] for row in rows}) != 459
            or {row["official_split"] for row in rows} != {"train"}
            or sha256_file(index_path) != manifest["modalities"][modality]["index_sha256"]
        ):
            raise V34TrainingError(f"pooling index boundary mismatch: {modality}")
        completed = 0
        for row in rows:
            if int(row["missing"]) == 1:
                if row.get("feature_path") is not None:
                    raise V34TrainingError(f"missing pooling row has a feature: {row['sample_id']}")
                continue
            path = resolve_workspace_path(str(row["feature_path"]))
            if sha256_file(path) != row["feature_sha256"]:
                raise V34TrainingError(f"pooling feature hash mismatch: {row['sample_id']}")
            with np.load(path, allow_pickle=False) as payload:
                value = np.asarray(payload["embedding"])
            if value.shape != (dimension,) or not np.isfinite(value).all():
                raise V34TrainingError(f"invalid pooling vector: {row['sample_id']}")
            completed += 1
        result["modalities"][modality] = {
            "index_sha256": sha256_file(index_path),
            "rows": len(rows),
            "completed": completed,
            "missing": len(rows) - completed,
        }
    return result


def audit(
    *,
    selection_path: Path,
    variant_runs: Mapping[str, Path],
    output_path: Path,
) -> dict[str, Any]:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("status") != "passed" or selection.get("official_test_evaluated") is not False:
        raise V34TrainingError("OPT-COG-003 selection did not pass")
    split_sha = str(selection["split_sha256"])
    summaries = []
    for variant, run_dir in variant_runs.items():
        candidate_id = VARIANT_CANDIDATES[variant]
        summary_path = run_dir / f"{candidate_id}_cv_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("official_test_evaluated") is not False or int(summary.get("fold_count", -1)) != 5:
            raise V34TrainingError(f"invalid feature summary: {candidate_id}")
        for outer_fold in range(5):
            candidate_dir = run_dir / f"fold_{outer_fold}" / candidate_id
            verify_prediction_bundle(candidate_dir, expected_split_sha256=split_sha)
            audit_path = candidate_dir / "feature_view_audit.json"
            feature_audit = json.loads(audit_path.read_text(encoding="utf-8"))
            if (
                feature_audit.get("variant") != variant
                or int(feature_audit.get("outer_fold", -1)) != outer_fold
                or feature_audit.get("official_test_media_read") is not False
            ):
                raise V34TrainingError(f"feature fold audit mismatch: {candidate_id}/{outer_fold}")
        summaries.append(
            {
                "variant": variant,
                "candidate_id": candidate_id,
                "run_path": workspace_relative(run_dir),
                "summary_sha256": sha256_file(summary_path),
            }
        )
    selected_run = WORKSPACE_ROOT / str(selection["selected_run_path"])
    selected_summary = selected_run / f"{selection['selected_candidate_id']}_cv_summary.json"
    if sha256_file(selected_summary) != selection["selected_summary_sha256"]:
        raise V34TrainingError("selected feature summary hash mismatch")
    report = {
        "schema_version": "cognitive_v34_opt_cog_003_audit_v1",
        "task_id": "OPT-COG-003",
        "status": "passed",
        "official_test_evaluated": False,
        "selection_path": workspace_relative(selection_path),
        "selection_sha256": sha256_file(selection_path),
        "split_sha256": split_sha,
        "lightweight_cache": _audit_lightweight(),
        "pooling_cache": _audit_pooling(),
        "candidate_runs": summaries,
        "selected_variant": selection["selected_variant"],
        "selected_candidate_id": selection["selected_candidate_id"],
    }
    write_json(output_path, report)
    return report


def _resolve_run(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPORT_ROOT / "runs" / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection",
        type=Path,
        default=REPORT_ROOT / "OPT-COG-003_selection_report.json",
    )
    for variant in VARIANT_CANDIDATES:
        parser.add_argument(f"--{variant.replace('_', '-')}-run", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORT_ROOT / "OPT-COG-003_run_audit.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runs = {
        variant: _resolve_run(getattr(args, f"{variant}_run"))
        for variant in VARIANT_CANDIDATES
    }
    report = audit(selection_path=args.selection, variant_runs=runs, output_path=args.output)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["audit"]
