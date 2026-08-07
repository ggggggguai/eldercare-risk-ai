"""Run FORECAST-OPT-001E external auxiliary and transfer sensitivity."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml


ALGORITHM_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ALGORITHM_ROOT.parents[1]
sys.path.insert(0, str(ALGORITHM_ROOT / "src"))

from elderly_monitoring.modules.mental_health.mood_social.forecast_external_auxiliary import (  # noqa: E402
    apply_conscious_proxy_to_psyche,
    fit_final_ridge,
    leave_year_out_ridge,
    nested_group_ridge_oof,
    nested_leave_one_out_ridge,
)
from elderly_monitoring.datasets.adapters.conscious_wearable import (  # noqa: E402
    ACTIVITY_FEATURES,
    COMMON_FEATURES,
    SLEEP_FEATURES,
    build_windows,
    source_manifest as conscious_source_manifest,
)
from elderly_monitoring.datasets.adapters.obf_psychiatric import (  # noqa: E402
    MODEL_FEATURES as OBF_MODEL_FEATURES,
    build_participant_features,
    source_manifest as obf_source_manifest,
)
from elderly_monitoring.modules.mental_health.mood_social.forecast_experiment import (  # noqa: E402
    sha256_file,
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


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _resolve_source(relative: str) -> Path:
    path = (WORKSPACE_ROOT / relative).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"external dataset root is missing: {relative}")
    return path


def _load_config(run_id: str) -> tuple[dict[str, Any], Path, Path]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["task_id"] != "FORECAST-OPT-001E" or config["run_id"] != run_id:
        raise RuntimeError("001E config identity changed")
    if any(
        config[key]
        for key in (
            "backend_change",
            "api_change",
            "v3_3_package_change",
            "physiology_in_scope",
        )
    ):
        raise RuntimeError("001E isolation boundary was relaxed")
    upstream = CANDIDATE_ROOT / config["upstream_run_id"]
    if (
        sha256_file(upstream / "selection_manifest.json")
        != config["upstream_selection_manifest_sha256"]
    ):
        raise RuntimeError("001C selection manifest changed")
    if (
        sha256_file(upstream / "fusion_manifest.json")
        != config["upstream_fusion_manifest_sha256"]
    ):
        raise RuntimeError("001D fusion manifest changed")
    output = upstream / "external_auxiliary" / run_id
    return config, upstream, output


def _conscious_experiments(
    windows: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    variants = {
        "sleep_only": SLEEP_FEATURES,
        "activity_only": ACTIVITY_FEATURES,
        "shared_combined": COMMON_FEATURES,
    }
    oof_parts = []
    metrics = []
    for target in ("phq9_score", "phq9_change"):
        for variant, features in variants.items():
            oof, audit = nested_group_ridge_oof(
                windows,
                features,
                target,
                "global_participant_id",
            )
            oof.insert(0, "variant", variant)
            oof.insert(0, "task", target)
            oof["window_id"] = windows["window_id"].to_numpy()
            oof_parts.append(oof)
            metrics.append({"task": target, "variant": variant, **audit})
    return pd.concat(oof_parts, ignore_index=True), metrics


def run(run_id: str) -> dict[str, Any]:
    config, upstream, output = _load_config(run_id)
    if output.exists():
        raise FileExistsError(f"001E output already exists: {output}")
    output.mkdir(parents=True)
    failures: list[dict[str, Any]] = []
    try:
        conscious_root = _resolve_source(config["sources"]["conscious_root"])
        obf_root = _resolve_source(config["sources"]["obf_root"])
        source_audit = {
            "conscious": conscious_source_manifest(conscious_root),
            "obf": obf_source_manifest(obf_root),
        }
        _write_json(output / "source_manifest.json", source_audit)

        conscious = build_windows(conscious_root)
        conscious.to_parquet(output / "conscious_windows.parquet", index=False)
        conscious_oof, conscious_metrics = _conscious_experiments(conscious)
        conscious_oof.to_parquet(output / "conscious_oof.parquet", index=False)
        _write_json(
            output / "conscious_metrics.json", {"experiments": conscious_metrics}
        )

        proxy_model = fit_final_ridge(
            conscious,
            COMMON_FEATURES,
            "phq9_score",
            "global_participant_id",
        )
        joblib.dump(proxy_model, output / "conscious_proxy_model.joblib")
        samples = {
            task_id: pd.read_parquet(
                upstream / "context" / task_id / "samples_all_features.parquet"
            )
            for task_id in ("forecast_1m", "forecast_2m")
        }
        proxy = apply_conscious_proxy_to_psyche(proxy_model, samples)
        proxy.to_parquet(output / "conscious_psyche_proxy.parquet", index=False)
        _write_json(
            output / "conscious_transfer_decision.json",
            {
                "status": "eligible_as_fixed_external_auxiliary_candidate_only",
                "model_alpha": proxy_model.alpha,
                "semantic_feature_count": len(COMMON_FEATURES),
                "semantic_features": list(COMMON_FEATURES),
                "psyche_labels_read": False,
                "entered_001d_fusion": False,
                "promotion_decision_deferred_to": "FORECAST-OPT-001F",
                "limitations": [
                    "49 young adults and 98 design-aligned windows",
                    "self-reported sleep versus Fitbit-derived PSYCHE-D sleep",
                    "fixed external proxy requires paired 001F evaluation before any use",
                ],
            },
        )

        obf = build_participant_features(obf_root)
        obf.to_parquet(output / "obf_participant_features.parquet", index=False)
        paired = obf.loc[obf["madrs1"].notna() & obf["madrs2"].notna()].copy()
        obf_oof, obf_metrics = nested_leave_one_out_ridge(
            paired,
            OBF_MODEL_FEATURES,
            "madrs_change",
        )
        year_oof, year_metrics = leave_year_out_ridge(
            paired,
            OBF_MODEL_FEATURES,
            "madrs_change",
        )
        obf_oof.to_parquet(output / "obf_madrs_loocv.parquet", index=False)
        year_oof.to_parquet(output / "obf_madrs_leave_year_out.parquet", index=False)
        _write_json(
            output / "obf_metrics.json",
            {
                "nested_loocv": obf_metrics,
                "leave_collection_start_year_out": year_metrics,
            },
        )
        _write_json(
            output / "obf_transfer_stop.json",
            {
                "status": "stopped_before_psyche_transfer",
                "reason": "OBF minute activity units and rhythm features do not have strict same-semantic mappings to the frozen PSYCHE-D 27 fields",
                "batch_sensitivity_completed": True,
                "diagnosis_group_used_as_product_target": False,
                "madrs_used_as_phq9_equivalent": False,
                "entered_001d_fusion": False,
                "required_next_evidence": "project-compatible activity rhythm features with audited window semantics",
            },
        )

        pd.DataFrame(
            failures,
            columns=["dataset", "stage", "error_type", "error", "traceback"],
        ).to_parquet(output / "failures.parquet", index=False)
        manifest = {
            "schema_version": "mood-social-forecast-v3.4-external-auxiliary-manifest-v1",
            "task_id": "FORECAST-OPT-001E",
            "run_id": run_id,
            "upstream_run_id": config["upstream_run_id"],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "completed_with_obf_transfer_stop",
            "config_sha256": sha256_file(CONFIG_PATH),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "conscious_participant_count": conscious["global_participant_id"].nunique(),
            "conscious_window_count": len(conscious),
            "conscious_proxy_row_count": len(proxy),
            "obf_participant_count": len(obf),
            "obf_paired_madrs_count": len(paired),
            "obf_psyche_transfer_performed": False,
            "psyche_labels_read_for_external_proxy": False,
            "corona_health_status": "deferred_p2_nonblocking",
            "bootstrap_performed": False,
            "promotion_decision_performed": False,
            "product_integration_performed": False,
            "failed_experiment_count": len(failures),
            "release_status": config["release_status"],
            "execution_mode": config["execution_mode"],
            "decision_authority": config["decision_authority"],
            "product_visible": config["product_visible"],
        }
        _write_json(output / "manifest.json", manifest)
        _write_sums(output)
        _write_sums(upstream)
        return manifest
    except Exception as exc:
        failures.append(
            {
                "dataset": "unknown",
                "stage": "FORECAST-OPT-001E",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        pd.DataFrame(failures).to_parquet(output / "failures.parquet", index=False)
        _write_json(
            output / "failure.json",
            {
                "task_id": "FORECAST-OPT-001E",
                "run_id": run_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        _write_sums(output)
        _write_sums(upstream)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="MH-20260807-FOPT-AUX-001")
    args = parser.parse_args()
    result = run(args.run_id)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
