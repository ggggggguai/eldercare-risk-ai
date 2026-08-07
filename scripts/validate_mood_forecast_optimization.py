"""Independently validate FORECAST-OPT-001B/C artifacts and isolation."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ALGORITHM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ALGORITHM_ROOT / "src"))

from elderly_monitoring.modules.mental_health.mood_social.forecast_experiment import (  # noqa: E402
    sha256_file,
)


BASELINE_ROOT = (
    ALGORITHM_ROOT / "artifacts" / "mental_health" / "mood_social" / "forecast_v3.4"
)
CANDIDATE_ROOT = BASELINE_ROOT / "optimization_candidates"
CONFIG_PATH = (
    ALGORITHM_ROOT / "configs" / "experiments" / "mood_social_forecast_v3_4_opt.yaml"
)
EXPECTED_COUNTS = {"forecast_1m": 9393, "forecast_2m": 9280}
EXPECTED_PARTICIPANTS = {"forecast_1m": 3635, "forecast_2m": 3593}
EXPECTED_FEATURE_COUNTS = {
    "anchor_only": 54,
    "anchor_delta": 162,
    "anchor_delta_rolling": 432,
}


@dataclass
class Checks:
    count: int = 0

    def require(self, condition: bool, message: str) -> None:
        if not condition:
            raise RuntimeError(message)
        self.count += 1


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_sums(root: Path) -> None:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            rows.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    (root / "SHA256SUMS").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _verify_sums(checks: Checks, root: Path) -> None:
    sums_path = root / "SHA256SUMS"
    listed: set[str] = set()
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        checks.require(relative not in listed, f"duplicate SHA entry: {relative}")
        listed.add(relative)
        path = root / relative
        checks.require(path.is_file(), f"SHA entry is missing: {relative}")
        checks.require(sha256_file(path) == digest, f"SHA mismatch: {relative}")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    checks.require(listed == actual, "SHA256SUMS does not cover the full candidate")


def _baseline_snapshot() -> dict[str, str]:
    return {
        path.relative_to(BASELINE_ROOT).as_posix(): sha256_file(path)
        for path in sorted(BASELINE_ROOT.rglob("*"))
        if path.is_file() and CANDIDATE_ROOT not in path.parents
    }


def _inner_assignments(config: dict[str, Any]) -> dict[int, dict[str, int]]:
    payload = _read_json(ALGORITHM_ROOT / config["inner_fold_manifest"])
    result = {outer_fold: {} for outer_fold in range(5)}
    for row in payload["rows"]:
        result[int(row["outer_fold_id"])][str(row["global_participant_id"])] = int(
            row["inner_fold_id"]
        )
    return result


def _validate_context(
    checks: Checks, root: Path, config: dict[str, Any]
) -> tuple[dict[str, tuple[str, ...]], dict[str, pd.DataFrame]]:
    manifest = _read_json(root / "manifest.json")
    checks.require(manifest["status"] == "context_built", "context status changed")
    checks.require(
        manifest["task_counts"] == EXPECTED_COUNTS, "context sample counts changed"
    )
    checks.require(
        manifest["feature_group_counts"] == EXPECTED_FEATURE_COUNTS,
        "context feature counts changed",
    )
    for key, expected in (
        ("release_status", "experimental"),
        ("execution_mode", "offline_only"),
        ("decision_authority", "shadow_only"),
    ):
        checks.require(manifest[key] == expected, f"context permission changed: {key}")
    checks.require(manifest["product_visible"] is False, "context became visible")
    feature_manifest = _read_json(root / "context" / "feature_manifest.json")
    source_names = tuple(feature_manifest["source_feature_names"])
    checks.require(len(source_names) == 27, "source feature count is not 27")
    groups = {
        name: tuple(feature_manifest["feature_groups"][name])
        for name in EXPECTED_FEATURE_COUNTS
    }
    for name, expected in EXPECTED_FEATURE_COUNTS.items():
        checks.require(len(groups[name]) == expected, f"feature count changed: {name}")
    all_features = groups["anchor_delta_rolling"]
    checks.require(len(set(all_features)) == 432, "derived features are duplicated")
    checks.require(
        not any("phq9" in name.lower() for name in all_features),
        "PHQ-9 entered the feature manifest",
    )
    checks.require(
        all(
            any(name.startswith(f"{source}__") for source in source_names)
            for name in all_features
        ),
        "a derived feature is not based on a frozen source field",
    )
    samples: dict[str, pd.DataFrame] = {}
    for task_id, expected_count in EXPECTED_COUNTS.items():
        frame = pd.read_parquet(
            root / "context" / task_id / "samples_all_features.parquet"
        )
        samples[task_id] = frame
        checks.require(len(frame) == expected_count, f"sample count changed: {task_id}")
        checks.require(
            frame["global_participant_id"].nunique() == EXPECTED_PARTICIPANTS[task_id],
            f"participant count changed: {task_id}",
        )
        checks.require(
            frame["target_window_id"].is_unique,
            f"target windows are duplicated: {task_id}",
        )
        checks.require(
            bool(
                (
                    frame["feature_month_slot"]
                    == frame["label_month_slot"] - frame["nominal_gap_months"]
                ).all()
            ),
            f"cutoff relation changed: {task_id}",
        )
        for group_name, feature_names in groups.items():
            ablation = pd.read_parquet(
                root / "context" / task_id / f"samples_{group_name}.parquet"
            )
            checks.require(
                len(ablation) == expected_count,
                f"ablation row count changed: {task_id}/{group_name}",
            )
            checks.require(
                tuple(ablation.columns[-len(feature_names) :]) == feature_names,
                f"ablation feature order changed: {task_id}/{group_name}",
            )
    return groups, samples


def _validate_candidate_matrix(
    checks: Checks,
    search: pd.DataFrame,
    task_id: str,
    outer_fold: int,
) -> None:
    checks.require(
        len(search) == 36, f"candidate count changed: {task_id}/{outer_fold}"
    )
    checks.require(
        search["candidate_id"].nunique() == 36,
        f"candidate IDs are duplicated: {task_id}/{outer_fold}",
    )
    checks.require(
        set(search["feature_group"]) == set(EXPECTED_FEATURE_COUNTS),
        f"feature groups are incomplete: {task_id}/{outer_fold}",
    )
    checks.require(
        set(search["model_family"]) == {"elasticnet_logistic", "lightgbm", "catboost"},
        f"model families are incomplete: {task_id}/{outer_fold}",
    )
    checks.require(
        set(search["auxiliary_mode"]) == {"none", "phq9_score", "phq9_category"},
        f"auxiliary modes are incomplete: {task_id}/{outer_fold}",
    )


def _validate_selection(
    checks: Checks,
    root: Path,
    config: dict[str, Any],
    samples: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    manifest = _read_json(root / "selection_manifest.json")
    checks.require(
        manifest["status"] == "selection_completed", "selection status changed"
    )
    checks.require(
        manifest["candidate_count_per_outer_fold"] == 36,
        "selection candidate count changed",
    )
    checks.require(
        manifest["outer_test_used_for_selection"] is False,
        "outer test was marked as used",
    )
    for field in (
        "final_fusion_performed",
        "final_calibration_performed",
        "outer_oof_generated",
    ):
        checks.require(manifest[field] is False, f"Goal boundary changed: {field}")
    checks.require(manifest["failed_candidate_count"] == 0, "a candidate failed")
    for key, expected in (
        ("release_status", "experimental"),
        ("execution_mode", "offline_only"),
        ("decision_authority", "shadow_only"),
    ):
        checks.require(
            manifest[key] == expected, f"selection permission changed: {key}"
        )
    checks.require(manifest["product_visible"] is False, "selection became visible")
    assignments = _inner_assignments(config)
    selections: list[dict[str, Any]] = []
    for task_id, frame in samples.items():
        for outer_fold in range(5):
            fold_root = root / "selection" / task_id / f"outer_fold_{outer_fold}"
            scope = _read_json(fold_root / "selection.json")
            search = pd.read_parquet(fold_root / "candidate_search.parquet")
            all_oof = pd.read_parquet(fold_root / "all_candidate_inner_oof.parquet")
            selected_oof = pd.read_parquet(fold_root / "selected_inner_oof.parquet")
            _validate_candidate_matrix(checks, search, task_id, outer_fold)
            outer_train = frame.loc[frame["outer_fold_id"].ne(outer_fold)]
            outer_test = frame.loc[frame["outer_fold_id"].eq(outer_fold)]
            checks.require(
                scope["outer_test_labels_read_for_selection"] is False,
                f"outer-test label policy changed: {task_id}/{outer_fold}",
            )
            checks.require(
                set(scope["selection_participant_ids"])
                == set(outer_train["global_participant_id"]),
                f"selection participant scope changed: {task_id}/{outer_fold}",
            )
            checks.require(
                set(scope["untouched_outer_test_participant_ids"])
                == set(outer_test["global_participant_id"]),
                f"outer-test participant scope changed: {task_id}/{outer_fold}",
            )
            checks.require(
                set(scope["selection_participant_ids"]).isdisjoint(
                    scope["untouched_outer_test_participant_ids"]
                ),
                f"outer participants leaked: {task_id}/{outer_fold}",
            )
            passed = search.loc[search["status"].eq("passed")]
            checks.require(
                len(passed) == 36, f"a candidate failed: {task_id}/{outer_fold}"
            )
            checks.require(
                set(all_oof["candidate_id"]) == set(passed["candidate_id"]),
                f"OOF candidates changed: {task_id}/{outer_fold}",
            )
            counts = all_oof.groupby("candidate_id")["target_window_id"].size()
            checks.require(
                bool(counts.eq(len(outer_train)).all()),
                f"OOF coverage changed: {task_id}/{outer_fold}",
            )
            checks.require(
                not bool(all_oof["outer_fold_id"].eq(outer_fold).any()),
                f"outer-test rows entered inner OOF: {task_id}/{outer_fold}",
            )
            expected_inner = all_oof["global_participant_id"].map(
                assignments[outer_fold]
            )
            checks.require(
                bool(expected_inner.eq(all_oof["inner_fold_id"]).all()),
                f"inner fold assignments changed: {task_id}/{outer_fold}",
            )
            no_aux = all_oof["auxiliary_mode"].eq("none")
            checks.require(
                bool(all_oof.loc[no_aux, "auxiliary_prediction"].isna().all()),
                f"no-aux candidate has auxiliary values: {task_id}/{outer_fold}",
            )
            checks.require(
                bool(all_oof.loc[~no_aux, "auxiliary_prediction"].notna().all()),
                f"auxiliary OOF is incomplete: {task_id}/{outer_fold}",
            )
            selected = scope["selected"]
            checks.require(
                len(selected_oof) == len(outer_train),
                f"selected OOF coverage changed: {task_id}/{outer_fold}",
            )
            checks.require(
                selected["candidate_id"] in set(passed["candidate_id"]),
                f"selected candidate is not valid: {task_id}/{outer_fold}",
            )
            checks.require(
                selected["feature_group"] == "anchor_delta_rolling",
                f"unexpected selected feature group: {task_id}/{outer_fold}",
            )
            selections.append(
                {
                    "task_id": task_id,
                    "outer_fold_id": outer_fold,
                    **selected,
                }
            )
    return {"manifest": manifest, "selections": selections}


def validate(run_id: str) -> dict[str, Any]:
    root = CANDIDATE_ROOT / run_id
    if root.parent != CANDIDATE_ROOT or not root.is_dir():
        raise RuntimeError("candidate run path is invalid")
    checks = Checks()
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    checks.require(config["backend_change"] is False, "backend change was enabled")
    checks.require(config["api_change"] is False, "API change was enabled")
    checks.require(config["v3_3_package_change"] is False, "V3.3 change was enabled")
    _verify_sums(checks, root)
    groups, samples = _validate_context(checks, root, config)
    checks.require(bool(groups), "feature groups are empty")
    selection = _validate_selection(checks, root, config, samples)
    expected_baseline = _read_json(root / "context" / "baseline_snapshot.json")
    checks.require(len(expected_baseline) == 189, "baseline snapshot count changed")
    checks.require(
        _baseline_snapshot() == expected_baseline,
        "immutable V3.4 baseline files changed",
    )
    return {
        "status": "pass",
        "run_id": run_id,
        "checks_passed": checks.count,
        "task_counts": EXPECTED_COUNTS,
        "participant_counts": EXPECTED_PARTICIPANTS,
        "feature_group_counts": EXPECTED_FEATURE_COUNTS,
        "selections": selection["selections"],
        "release_status": "experimental",
        "execution_mode": "offline_only",
        "decision_authority": "shadow_only",
        "product_visible": False,
        "validation_code_sha256": sha256_file(Path(__file__).resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    root = CANDIDATE_ROOT / args.run_id
    if args.write_report:
        _write_json(root / "validation_report.json", {})
        _write_sums(root)
        result = validate(args.run_id)
        _write_json(root / "validation_report.json", result)
        _write_sums(root)
        result = validate(args.run_id)
    else:
        result = validate(args.run_id)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
