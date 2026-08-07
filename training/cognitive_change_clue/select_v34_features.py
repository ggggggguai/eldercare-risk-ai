"""Select the final Audio/Text feature view for OPT-COG-003."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

try:
    from .common import WORKSPACE_ROOT, sha256_file, workspace_relative
    from .train import write_json
    from .train_v34 import PREDICTION_FILES, REPORT_ROOT, V34TrainingError, verify_prediction_bundle
    from .train_v34_features import VARIANT_CANDIDATES
except ImportError:
    from common import WORKSPACE_ROOT, sha256_file, workspace_relative  # type: ignore[no-redef]
    from train import write_json  # type: ignore[no-redef]
    from train_v34 import (  # type: ignore[no-redef]
        PREDICTION_FILES,
        REPORT_ROOT,
        V34TrainingError,
        verify_prediction_bundle,
    )
    from train_v34_features import VARIANT_CANDIDATES  # type: ignore[no-redef]


VARIANCE_INCREASE_LIMIT = 0.01
COMPLEXITY = {
    "baseline": 0,
    "audio_basic": 1,
    "text_continuous_quality": 1,
    "text_stats": 2,
    "egemaps": 3,
    "wavlm_mean_std": 4,
    "roberta_mean": 4,
}


def _resolve_run(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPORT_ROOT / "runs" / path


def _summary(run_dir: Path, candidate_id: str) -> dict[str, Any]:
    path = run_dir / f"{candidate_id}_cv_summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("candidate_id") != candidate_id
        or payload.get("official_test_evaluated") is not False
        or int(payload.get("fold_count", -1)) != 5
    ):
        raise V34TrainingError(f"invalid feature candidate summary: {path}")
    return payload


def _sample_ids(candidate_dir: Path, role: str) -> set[str]:
    path = candidate_dir / PREDICTION_FILES[role]
    return {
        str(json.loads(line)["sample_id"])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def select_features(
    *,
    backbone_selection_path: Path,
    variant_runs: Mapping[str, Path],
    output_path: Path,
) -> dict[str, Any]:
    backbone = json.loads(backbone_selection_path.read_text(encoding="utf-8"))
    if backbone.get("status") != "passed" or backbone.get("official_test_evaluated") is not False:
        raise V34TrainingError("OPT-COG-002 selection did not pass")
    split_sha = str(backbone["split_sha256"])
    base_candidate = str(backbone["selected_audio_text_candidate"])
    base_run_record = backbone["runs"][base_candidate]
    base_run = WORKSPACE_ROOT / str(base_run_record["path"])
    base_summary = _summary(base_run, base_candidate)
    base_auc = float(base_summary["subject_raw_auc"]["mean"])
    base_std = float(base_summary["subject_raw_auc"]["population_std"])
    rows = [
        {
            "variant": "baseline",
            "candidate_id": base_candidate,
            "run_path": workspace_relative(base_run),
            "summary_sha256": sha256_file(base_run / f"{base_candidate}_cv_summary.json"),
            "subject_auc_mean": base_auc,
            "subject_auc_population_std": base_std,
            "auc_gain_vs_backbone": 0.0,
            "std_increase_vs_backbone": 0.0,
            "temporary_specificity": float(base_summary["pooled_temporary_workpoint"]["specificity"]),
            "task_auc_mean": float(base_summary["task_raw_auc"]["mean"]),
            "complexity_rank": COMPLEXITY["baseline"],
            "eligible": True,
            "eligibility_reason": "frozen_backbone_reference",
        }
    ]
    alignment: list[dict[str, Any]] = []
    for outer_fold in range(5):
        base_dir = base_run / f"fold_{outer_fold}" / base_candidate
        verify_prediction_bundle(base_dir, expected_split_sha256=split_sha)
        role_counts: dict[str, int] = {}
        for role in PREDICTION_FILES:
            expected = _sample_ids(base_dir, role)
            for variant, run_dir in variant_runs.items():
                candidate_id = VARIANT_CANDIDATES[variant]
                candidate_dir = run_dir / f"fold_{outer_fold}" / candidate_id
                verify_prediction_bundle(candidate_dir, expected_split_sha256=split_sha)
                if _sample_ids(candidate_dir, role) != expected:
                    raise V34TrainingError(
                        f"feature population mismatch variant={variant} fold={outer_fold} role={role}"
                    )
            role_counts[role] = len(expected)
        alignment.append({"outer_fold": outer_fold, "task_counts": role_counts})
    for variant, run_dir in variant_runs.items():
        candidate_id = VARIANT_CANDIDATES[variant]
        summary = _summary(run_dir, candidate_id)
        auc = float(summary["subject_raw_auc"]["mean"])
        std = float(summary["subject_raw_auc"]["population_std"])
        gain = auc - base_auc
        std_increase = std - base_std
        eligible = gain > 0.0 and std_increase <= VARIANCE_INCREASE_LIMIT
        reason = (
            "positive_auc_gain_and_variance_within_limit"
            if eligible
            else "non_positive_auc_gain"
            if gain <= 0.0
            else "variance_increase_exceeds_limit"
        )
        rows.append(
            {
                "variant": variant,
                "candidate_id": candidate_id,
                "run_path": workspace_relative(run_dir),
                "summary_sha256": sha256_file(run_dir / f"{candidate_id}_cv_summary.json"),
                "subject_auc_mean": auc,
                "subject_auc_population_std": std,
                "auc_gain_vs_backbone": gain,
                "std_increase_vs_backbone": std_increase,
                "temporary_specificity": float(summary["pooled_temporary_workpoint"]["specificity"]),
                "task_auc_mean": float(summary["task_raw_auc"]["mean"]),
                "complexity_rank": COMPLEXITY[variant],
                "eligible": eligible,
                "eligibility_reason": reason,
            }
        )
    eligible = [row for row in rows if row["eligible"]]
    best_auc = max(float(row["subject_auc_mean"]) for row in eligible)
    contenders = [
        row for row in eligible if best_auc - float(row["subject_auc_mean"]) < 0.005
    ]
    contenders.sort(
        key=lambda row: (
            float(row["subject_auc_population_std"]),
            -float(row["temporary_specificity"]),
            -float(row["task_auc_mean"]),
            int(row["complexity_rank"]),
            str(row["candidate_id"]),
        )
    )
    selected = contenders[0]
    report = {
        "schema_version": "cognitive_v34_opt_cog_003_selection_v1",
        "task_id": "OPT-COG-003",
        "status": "passed",
        "official_test_evaluated": False,
        "split_sha256": split_sha,
        "backbone_selection_path": workspace_relative(backbone_selection_path),
        "backbone_selection_sha256": sha256_file(backbone_selection_path),
        "backbone_candidate": base_candidate,
        "compared_candidate_count_this_stage": len(rows),
        "cumulative_structure_and_feature_candidate_count": int(backbone["candidate_count"]) + len(variant_runs),
        "selection_rule": {
            "positive_auc_gain_required": True,
            "maximum_population_std_increase": VARIANCE_INCREASE_LIMIT,
            "auc_tie_tolerance": 0.005,
            "tie_breaks": [
                "subject_auc_population_std_ascending",
                "temporary_specificity_descending",
                "task_auc_mean_descending",
                "complexity_ascending",
            ],
        },
        "candidates": rows,
        "selected_variant": selected["variant"],
        "selected_candidate_id": selected["candidate_id"],
        "selected_run_path": selected["run_path"],
        "selected_summary_sha256": selected["summary_sha256"],
        "final_z_at_prediction_scope": "selected_run_frozen_three_partition_raw_predictions",
        "population_alignment": alignment,
        "pooling_follow_up": {
            variant: {
                "positive_gain": next(row for row in rows if row["variant"] == variant)["auc_gain_vs_backbone"] > 0.0,
                "variance_within_limit": next(row for row in rows if row["variant"] == variant)["std_increase_vs_backbone"] <= VARIANCE_INCREASE_LIMIT,
                "attention_pooling_required_for_opt_cog_003": False,
            }
            for variant in ("wavlm_mean_std", "roberta_mean")
        },
        "limitations": [
            "All candidates are CogPic official-Train five-fold development results.",
            "No official Test media or prediction was read or generated.",
        ],
    }
    write_json(output_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backbone-selection",
        type=Path,
        default=REPORT_ROOT / "OPT-COG-002_selection_report.json",
    )
    for variant in VARIANT_CANDIDATES:
        parser.add_argument(f"--{variant.replace('_', '-')}-run", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORT_ROOT / "OPT-COG-003_selection_report.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runs = {
        variant: _resolve_run(getattr(args, f"{variant}_run"))
        for variant in VARIANT_CANDIDATES
    }
    report = select_features(
        backbone_selection_path=args.backbone_selection,
        variant_runs=runs,
        output_path=args.output,
    )
    print(json.dumps(report, ensure_ascii=False, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["select_features"]
