"""Run one OPT-COG-003 feature variant on the frozen V3.4 folds."""

from __future__ import annotations

import argparse
import copy
import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .build_subject_cv_v34 import DEFAULT_AUDIT_PATH, DEFAULT_OUTPUT_PATH
    from .common import sha256_file
    from .train import write_json
    from .train_v34 import (
        CANDIDATES,
        CONFIG_PATH,
        REPORT_ROOT,
        V34TrainingError,
        _device,
        aggregate_candidate,
        allocate_run_id,
        load_config,
        train_fold,
    )
    from .v34_data import load_fold, load_official_train_pool
    from .v34_features import VARIANTS, build_fold_feature_view
except ImportError:
    from build_subject_cv_v34 import DEFAULT_AUDIT_PATH, DEFAULT_OUTPUT_PATH  # type: ignore[no-redef]
    from common import sha256_file  # type: ignore[no-redef]
    from train import write_json  # type: ignore[no-redef]
    from train_v34 import (  # type: ignore[no-redef]
        CANDIDATES,
        CONFIG_PATH,
        REPORT_ROOT,
        V34TrainingError,
        _device,
        aggregate_candidate,
        allocate_run_id,
        load_config,
        train_fold,
    )
    from v34_data import load_fold, load_official_train_pool  # type: ignore[no-redef]
    from v34_features import VARIANTS, build_fold_feature_view  # type: ignore[no-redef]


VARIANT_CANDIDATES = {
    "audio_basic": "V34-F1-AudioBasic",
    "egemaps": "V34-F2-eGeMAPS",
    "text_stats": "V34-F3-TextStats",
    "text_continuous_quality": "V34-F4-TextContinuous",
    "wavlm_mean_std": "V34-P1-WavLMStats",
    "roberta_mean": "V34-P2-RoBERTaMean",
}
SUPPORTED_BASES = ("V34-A1", "V34-A2")


def _selected_backbone(selection_path: Path) -> str:
    payload = json.loads(selection_path.read_text(encoding="utf-8"))
    if (
        payload.get("status") != "passed"
        or payload.get("official_test_evaluated") is not False
        or payload.get("split_sha256") != sha256_file(DEFAULT_OUTPUT_PATH)
    ):
        raise V34TrainingError("OPT-COG-002 selection is not frozen and isolated")
    candidate = str(payload["selected_audio_text_candidate"])
    if candidate not in SUPPORTED_BASES:
        raise V34TrainingError(
            "the selected late-fusion A3 backbone requires its component feature runner"
        )
    return candidate


def run_feature_variant(
    *,
    variant: str,
    run_id: str | None,
    selection_path: Path,
    config_path: Path,
    requested_folds: list[int],
    device_name: str,
) -> dict[str, Any]:
    if variant not in VARIANT_CANDIDATES:
        raise V34TrainingError(f"unsupported OPT-COG-003 variant: {variant}")
    audit = json.loads(DEFAULT_AUDIT_PATH.read_text(encoding="utf-8"))
    if audit.get("status") != "passed" or int(audit.get("official_test_overlap_count", -1)) != 0:
        raise V34TrainingError("V3.4 split audit did not pass")
    base_candidate = _selected_backbone(selection_path)
    candidate_id = VARIANT_CANDIDATES[variant]
    candidate = copy.deepcopy(CANDIDATES[base_candidate])
    candidate["feature_variant"] = variant
    candidate["base_candidate"] = base_candidate
    CANDIDATES[candidate_id] = candidate

    config = load_config(config_path)
    records, frozen_hashes = load_official_train_pool(verify_hashes=True)
    device = _device(device_name)
    run_id = run_id or allocate_run_id()
    run_dir = REPORT_ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "run_manifest.json"
    write_json(
        manifest_path,
        {
            "schema_version": "cognitive_v34_feature_candidate_run_v1",
            "status": "running",
            "task_id": "OPT-COG-003",
            "run_id": run_id,
            "candidate_id": candidate_id,
            "base_candidate": base_candidate,
            "feature_variant": variant,
            "requested_folds": requested_folds,
            "pid": os.getpid(),
            "device": str(device),
            "started_or_resumed_at": datetime.now(timezone.utc).astimezone().isoformat(),
            "split_sha256": sha256_file(DEFAULT_OUTPUT_PATH),
            "config_sha256": sha256_file(config_path),
            "selection_report_sha256": sha256_file(selection_path),
            "official_test_evaluated": False,
        },
    )
    completed = []
    for outer_fold in requested_folds:
        fold = load_fold(outer_fold)
        view = build_fold_feature_view(
            records,
            inner_train_subjects=fold["inner_train_subjects"],
            variant=variant,
        )
        fold_audit = {
            **view.audit,
            "task_id": "OPT-COG-003",
            "candidate_id": candidate_id,
            "base_candidate": base_candidate,
            "outer_fold": outer_fold,
            "split_sha256": str(fold["split_sha256"]),
            "official_test_media_read": False,
        }
        train_fold(
            candidate_id=candidate_id,
            outer_fold=outer_fold,
            run_id=run_id,
            run_dir=run_dir,
            base_config=config,
            records=view.records,
            frozen_input_hashes=frozen_hashes,
            device=device,
            input_dims=view.input_dims,
            feature_audit=fold_audit,
        )
        completed.append(outer_fold)
    summary = aggregate_candidate(run_dir, candidate_id=candidate_id) if len(completed) == 5 else None
    final_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    final_manifest.update(
        {
            "status": "completed" if summary is not None else "partial",
            "completed_folds": completed,
            "completed_at": datetime.now(timezone.utc).astimezone().isoformat(),
            "summary_sha256": (
                sha256_file(run_dir / f"{candidate_id}_cv_summary.json")
                if summary is not None
                else None
            ),
        }
    )
    write_json(manifest_path, final_manifest)
    return summary or final_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=sorted(VARIANT_CANDIDATES))
    parser.add_argument("--run-id")
    parser.add_argument(
        "--selection",
        type=Path,
        default=REPORT_ROOT / "OPT-COG-002_selection_report.json",
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--folds", default="all")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    folds = list(range(5)) if args.folds == "all" else [int(value) for value in args.folds.split(",")]
    if any(value not in range(5) for value in folds):
        raise SystemExit("folds must be all or a comma-separated subset of 0..4")
    try:
        result = run_feature_variant(
            variant=args.variant,
            run_id=args.run_id,
            selection_path=args.selection,
            config_path=args.config,
            requested_folds=folds,
            device_name=args.device,
        )
    except Exception:
        traceback.print_exc()
        return 1
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["VARIANT_CANDIDATES", "run_feature_variant"]
