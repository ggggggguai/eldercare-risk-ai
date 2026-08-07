"""Independently validate FORECAST-OPT-001E external auxiliary artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml


ALGORITHM_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ALGORITHM_ROOT.parents[1]
sys.path.insert(0, str(ALGORITHM_ROOT / "src"))

from elderly_monitoring.modules.mental_health.mood_social.forecast_experiment import (  # noqa: E402
    sha256_file,
)
from elderly_monitoring.modules.mental_health.mood_social.forecast_external_auxiliary import (  # noqa: E402
    ALPHAS,
    apply_conscious_proxy_to_psyche,
)
from elderly_monitoring.datasets.adapters.conscious_wearable import (  # noqa: E402
    COMMON_FEATURES,
    source_manifest as conscious_source_manifest,
)
from elderly_monitoring.datasets.adapters.obf_psychiatric import (  # noqa: E402
    MODEL_FEATURES as OBF_MODEL_FEATURES,
    source_manifest as obf_source_manifest,
)


CONFIG_PATH = (
    ALGORITHM_ROOT
    / "configs"
    / "experiments"
    / "mood_social_forecast_v3_4_external_auxiliary.yaml"
)
FORECAST_ROOT = (
    ALGORITHM_ROOT / "artifacts" / "mental_health" / "mood_social" / "forecast_v3.4"
)
CANDIDATE_ROOT = FORECAST_ROOT / "optimization_candidates"
TASK_COUNTS = {"forecast_1m": 9393, "forecast_2m": 9280}


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
        newline="\n",
    )


def _write_sums(root: Path) -> None:
    rows = [
        f"{sha256_file(path)}  {path.relative_to(root).as_posix()}"
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "SHA256SUMS"
    ]
    (root / "SHA256SUMS").write_text(
        "\n".join(rows) + "\n", encoding="utf-8", newline="\n"
    )


def _verify_sums(checks: Checks, root: Path) -> None:
    sums_path = root / "SHA256SUMS"
    listed: set[str] = set()
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        checks.require(relative not in listed, f"duplicate SHA entry: {relative}")
        listed.add(relative)
        path = root / relative
        checks.require(path.is_file(), f"missing SHA entry: {relative}")
        checks.require(sha256_file(path) == digest, f"SHA mismatch: {relative}")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    checks.require(actual == listed, "SHA256SUMS does not cover the complete run")


def _resolve_source(relative: str) -> Path:
    path = (WORKSPACE_ROOT / relative).resolve()
    if not path.is_dir():
        raise RuntimeError(f"external source is missing: {relative}")
    return path


def _validate_manifest(
    checks: Checks,
    config: dict[str, Any],
    upstream: Path,
    output: Path,
) -> dict[str, Any]:
    manifest = _read_json(output / "manifest.json")
    checks.require(manifest["task_id"] == "FORECAST-OPT-001E", "task changed")
    checks.require(
        manifest["status"] == "completed_with_obf_transfer_stop",
        "001E completion status changed",
    )
    checks.require(
        manifest["upstream_run_id"] == config["upstream_run_id"],
        "upstream run changed",
    )
    checks.require(
        manifest["config_sha256"] == sha256_file(CONFIG_PATH),
        "001E config hash changed",
    )
    checks.require(
        manifest["runner_sha256"]
        == sha256_file(
            ALGORITHM_ROOT / "scripts" / "run_mood_forecast_external_auxiliary.py"
        ),
        "001E runner hash changed",
    )
    for key, expected in (
        ("release_status", "experimental"),
        ("execution_mode", "offline_only"),
        ("decision_authority", "shadow_only"),
        ("product_visible", False),
        ("product_integration_performed", False),
        ("promotion_decision_performed", False),
        ("bootstrap_performed", False),
        ("psyche_labels_read_for_external_proxy", False),
        ("obf_psyche_transfer_performed", False),
        ("corona_health_status", "deferred_p2_nonblocking"),
        ("failed_experiment_count", 0),
    ):
        checks.require(manifest[key] == expected, f"manifest boundary changed: {key}")
    for key in (
        "backend_change",
        "api_change",
        "v3_3_package_change",
        "physiology_in_scope",
    ):
        checks.require(config[key] is False, f"isolation boundary changed: {key}")
    checks.require(
        sha256_file(upstream / "selection_manifest.json")
        == config["upstream_selection_manifest_sha256"],
        "001B/001C selection manifest changed",
    )
    checks.require(
        sha256_file(upstream / "fusion_manifest.json")
        == config["upstream_fusion_manifest_sha256"],
        "001D fusion manifest changed",
    )
    selection_validation = _read_json(upstream / "validation_report.json")
    fusion_validation = _read_json(upstream / "fusion_validation_report.json")
    checks.require(
        selection_validation["status"] == "pass"
        and selection_validation["checks_passed"] == 422,
        "001B/001C validation evidence changed",
    )
    checks.require(
        fusion_validation["status"] == "pass"
        and fusion_validation["checks_passed"] == 920,
        "001D validation evidence changed",
    )
    return manifest


def _validate_sources(
    checks: Checks, config: dict[str, Any], output: Path
) -> dict[str, Any]:
    saved = _read_json(output / "source_manifest.json")
    conscious_root = _resolve_source(config["sources"]["conscious_root"])
    obf_root = _resolve_source(config["sources"]["obf_root"])
    current = {
        "conscious": conscious_source_manifest(conscious_root),
        "obf": obf_source_manifest(obf_root),
    }
    checks.require(saved == current, "external source hashes or byte counts changed")
    checks.require(
        current["conscious"]["files"]["survey.csv"]["bytes"] == 3562,
        "Conscious survey byte count changed",
    )
    checks.require(current["obf"]["file_count"] == 167, "OBF file count changed")
    checks.require(
        current["obf"]["total_bytes"] == 101472815,
        "OBF total byte count changed",
    )
    return current


def _validate_conscious(
    checks: Checks,
    config: dict[str, Any],
    upstream: Path,
    output: Path,
) -> dict[str, Any]:
    windows = pd.read_parquet(output / "conscious_windows.parquet")
    expected_windows = int(config["conscious"]["expected_windows"])
    expected_participants = int(config["conscious"]["expected_participants"])
    checks.require(len(windows) == expected_windows, "Conscious window count changed")
    checks.require(
        windows["global_participant_id"].nunique() == expected_participants,
        "Conscious participant count changed",
    )
    checks.require(windows["window_id"].is_unique, "Conscious windows duplicated")
    checks.require(
        windows.groupby("global_participant_id").size().eq(2).all(),
        "Conscious participant window coverage changed",
    )
    checks.require(
        windows["diary_day_count"].eq(14).all(),
        "Conscious strict 14-day window changed",
    )

    tasks = tuple(config["conscious"]["tasks"])
    variants = tuple(config["conscious"]["variants"])
    oof = pd.read_parquet(output / "conscious_oof.parquet")
    checks.require(
        len(oof) == expected_windows * len(tasks) * len(variants),
        "Conscious OOF row count changed",
    )
    checks.require(
        not oof.duplicated(["task", "variant", "window_id"]).any(),
        "Conscious OOF keys duplicated",
    )
    fold_reference: pd.Series | None = None
    for task in tasks:
        for variant in variants:
            part = oof.loc[oof["task"].eq(task) & oof["variant"].eq(variant)]
            checks.require(
                len(part) == expected_windows, f"OOF incomplete: {task}/{variant}"
            )
            checks.require(
                set(part["window_id"]) == set(windows["window_id"]),
                f"OOF window coverage changed: {task}/{variant}",
            )
            checks.require(
                part[["prediction", "null_prediction"]].notna().all().all(),
                f"OOF prediction missing: {task}/{variant}",
            )
            participant_folds = part.groupby("global_participant_id")["outer_fold_id"]
            checks.require(
                participant_folds.nunique().eq(1).all(),
                f"participant split across folds: {task}/{variant}",
            )
            current = participant_folds.first().sort_index()
            if fold_reference is None:
                fold_reference = current
            else:
                checks.require(
                    current.equals(fold_reference),
                    f"fold assignment changed between experiments: {task}/{variant}",
                )

    metrics = _read_json(output / "conscious_metrics.json")["experiments"]
    metric_map = {(row["task"], row["variant"]): row for row in metrics}
    checks.require(
        set(metric_map) == {(task, variant) for task in tasks for variant in variants},
        "Conscious metric experiment matrix changed",
    )
    for key, row in metric_map.items():
        checks.require(row["row_count"] == 98, f"metric rows changed: {key}")
        checks.require(
            row["participant_count"] == 49, f"metric participants changed: {key}"
        )
        checks.require(len(row["fold_audits"]) == 5, f"fold audit incomplete: {key}")
        checks.require(
            sum(item["test_participant_count"] for item in row["fold_audits"]) == 49,
            f"fold participant coverage changed: {key}",
        )

    decision = _read_json(output / "conscious_transfer_decision.json")
    checks.require(
        decision["status"] == "eligible_as_fixed_external_auxiliary_candidate_only",
        "Conscious transfer status changed",
    )
    checks.require(decision["psyche_labels_read"] is False, "PSYCHE labels marked read")
    checks.require(decision["entered_001d_fusion"] is False, "proxy entered 001D")
    checks.require(
        tuple(decision["semantic_features"]) == COMMON_FEATURES,
        "Conscious semantic bridge changed",
    )

    proxy = pd.read_parquet(output / "conscious_psyche_proxy.parquet")
    expected_proxy_columns = {
        "task_id",
        "global_participant_id",
        "target_window_id",
        "outer_fold_id",
        "conscious_phq9_proxy",
        "proxy_source",
    }
    checks.require(set(proxy.columns) == expected_proxy_columns, "proxy schema changed")
    checks.require(len(proxy) == sum(TASK_COUNTS.values()), "proxy row count changed")
    checks.require(
        not proxy.duplicated(["task_id", "target_window_id"]).any(),
        "proxy keys duplicated",
    )
    checks.require(
        np.isfinite(proxy["conscious_phq9_proxy"]).all(),
        "proxy contains non-finite values",
    )

    model = joblib.load(output / "conscious_proxy_model.joblib")
    checks.require(
        tuple(model.feature_names) == COMMON_FEATURES, "proxy model features changed"
    )
    safe_samples: dict[str, pd.DataFrame] = {}
    for task_id, expected_count in TASK_COUNTS.items():
        source = pd.read_parquet(
            upstream / "context" / task_id / "samples_all_features.parquet"
        )
        safe_columns = [
            "global_participant_id",
            "target_window_id",
            "outer_fold_id",
            *(f"{name}__anchor" for name in COMMON_FEATURES),
        ]
        safe_samples[task_id] = source.loc[:, safe_columns].copy()
        saved = proxy.loc[proxy["task_id"].eq(task_id)]
        checks.require(
            len(saved) == expected_count, f"proxy task count changed: {task_id}"
        )
        checks.require(
            set(saved["target_window_id"]) == set(source["target_window_id"]),
            f"proxy task coverage changed: {task_id}",
        )
    reapplied = apply_conscious_proxy_to_psyche(model, safe_samples)
    sort_key = ["task_id", "target_window_id"]
    expected = proxy.sort_values(sort_key, kind="mergesort").reset_index(drop=True)
    actual = reapplied.sort_values(sort_key, kind="mergesort").reset_index(drop=True)
    checks.require(
        expected.drop(columns="conscious_phq9_proxy").equals(
            actual.drop(columns="conscious_phq9_proxy")
        ),
        "label-free proxy identity columns changed",
    )
    checks.require(
        np.allclose(
            expected["conscious_phq9_proxy"],
            actual["conscious_phq9_proxy"],
            rtol=0,
            atol=1e-12,
        ),
        "saved proxy cannot be reproduced from the 19 safe fields alone",
    )
    return {
        "participant_count": expected_participants,
        "window_count": expected_windows,
        "oof_row_count": len(oof),
        "proxy_row_count": len(proxy),
        "semantic_feature_count": len(COMMON_FEATURES),
        "psyche_labels_read": False,
    }


def _validate_obf(
    checks: Checks, config: dict[str, Any], output: Path
) -> dict[str, Any]:
    features = pd.read_parquet(output / "obf_participant_features.parquet")
    paired = features.loc[
        features["madrs1"].notna() & features["madrs2"].notna()
    ].copy()
    checks.require(
        len(features) == int(config["obf"]["expected_participants"]),
        "OBF participant count changed",
    )
    checks.require(
        features["global_participant_id"].is_unique, "OBF participants duplicated"
    )
    checks.require(
        len(paired) == int(config["obf"]["expected_paired_madrs"]),
        "OBF paired MADRS count changed",
    )

    loocv = pd.read_parquet(output / "obf_madrs_loocv.parquet")
    year_oof = pd.read_parquet(output / "obf_madrs_leave_year_out.parquet")
    paired_ids = set(paired["global_participant_id"])
    for name, frame in (("nested LOOCV", loocv), ("leave-year-out", year_oof)):
        checks.require(len(frame) == len(paired), f"OBF {name} OOF count changed")
        checks.require(
            frame["global_participant_id"].is_unique, f"OBF {name} duplicated"
        )
        checks.require(
            set(frame["global_participant_id"]) == paired_ids,
            f"OBF {name} coverage changed",
        )
        checks.require(
            frame[["prediction", "null_prediction"]].notna().all().all(),
            f"OBF {name} prediction missing",
        )
    checks.require(
        set(loocv["selected_alpha"].astype(float)).issubset(set(ALPHAS)),
        "OBF LOOCV alpha outside frozen grid",
    )
    total = float(loocv["madrs_change"].sum())
    expected_null = (total - loocv["madrs_change"]) / (len(loocv) - 1)
    checks.require(
        np.allclose(loocv["null_prediction"], expected_null, rtol=0, atol=1e-12),
        "OBF LOOCV null prediction used the held-out label",
    )
    for year, held_out in year_oof.groupby("collection_start_year"):
        expected_mean = year_oof.loc[
            year_oof["collection_start_year"].ne(year), "madrs_change"
        ].mean()
        checks.require(
            np.allclose(held_out["null_prediction"], expected_mean, rtol=0, atol=1e-12),
            f"OBF leave-year null prediction used held-out year {year}",
        )

    metrics = _read_json(output / "obf_metrics.json")
    nested = metrics["nested_loocv"]
    leave_year = metrics["leave_collection_start_year_out"]
    checks.require(
        tuple(nested["features"]) == OBF_MODEL_FEATURES, "OBF feature set changed"
    )
    checks.require(nested["row_count"] == 23, "OBF nested metric rows changed")
    checks.require(len(leave_year["audits"]) == 4, "OBF year audit count changed")
    checks.require(
        nested["mae_delta_vs_null"] > 0, "OBF nested negative result changed"
    )
    checks.require(
        leave_year["mae_delta_vs_null"] > 0, "OBF year negative result changed"
    )

    stop = _read_json(output / "obf_transfer_stop.json")
    checks.require(
        stop["status"] == "stopped_before_psyche_transfer",
        "OBF transfer stop changed",
    )
    checks.require(
        stop["batch_sensitivity_completed"] is True, "OBF batch check missing"
    )
    checks.require(stop["entered_001d_fusion"] is False, "OBF entered 001D fusion")
    checks.require(
        stop["madrs_used_as_phq9_equivalent"] is False, "MADRS treated as PHQ-9"
    )
    checks.require(
        stop["diagnosis_group_used_as_product_target"] is False,
        "OBF diagnosis group became a product target",
    )
    return {
        "participant_count": len(features),
        "paired_madrs_count": len(paired),
        "nested_loocv_mae_delta_vs_null": nested["mae_delta_vs_null"],
        "leave_year_mae_delta_vs_null": leave_year["mae_delta_vs_null"],
        "transfer_status": stop["status"],
    }


def validate(run_id: str) -> dict[str, Any]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["task_id"] != "FORECAST-OPT-001E" or config["run_id"] != run_id:
        raise RuntimeError("001E config identity changed")
    upstream = CANDIDATE_ROOT / config["upstream_run_id"]
    output = upstream / "external_auxiliary" / run_id
    if output.parent.parent != upstream or not output.is_dir():
        raise RuntimeError("001E output path is invalid")
    checks = Checks()
    _verify_sums(checks, output)
    manifest = _validate_manifest(checks, config, upstream, output)
    sources = _validate_sources(checks, config, output)
    conscious = _validate_conscious(checks, config, upstream, output)
    obf = _validate_obf(checks, config, output)
    failures = pd.read_parquet(output / "failures.parquet")
    checks.require(failures.empty, "001E failure table is not empty")
    checks.require(
        tuple(failures.columns)
        == ("dataset", "stage", "error_type", "error", "traceback"),
        "001E failure schema changed",
    )
    return {
        "status": "pass",
        "task_id": "FORECAST-OPT-001E",
        "run_id": run_id,
        "upstream_run_id": config["upstream_run_id"],
        "checks_passed": checks.count,
        "conscious": conscious,
        "obf": obf,
        "source_manifest": sources,
        "failed_experiment_count": 0,
        "release_status": manifest["release_status"],
        "execution_mode": manifest["execution_mode"],
        "decision_authority": manifest["decision_authority"],
        "product_visible": manifest["product_visible"],
        "promotion_decision_performed": False,
        "validation_code_sha256": sha256_file(Path(__file__).resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="MH-20260807-FOPT-AUX-001")
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    upstream = CANDIDATE_ROOT / config["upstream_run_id"]
    output = upstream / "external_auxiliary" / args.run_id
    result = validate(args.run_id)
    if args.write_report:
        _write_json(output / "validation_report.json", result)
        _write_sums(output)
        _write_sums(upstream)
        result = validate(args.run_id)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
