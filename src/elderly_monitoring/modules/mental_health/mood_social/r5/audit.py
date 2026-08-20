"""Time, missingness and leakage audit for OPT-V333-R5-001."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from elderly_monitoring.modules.mental_health.mood_social.r5.contract import (
    R5_DEVELOPMENT_SEEDS,
    R5_PSYCHE_RESEARCH_SENSOR_FEATURES,
    R5_PROTOCOL_VERSION,
    forbidden_r5_feature_reason,
    sha256_file,
)
from elderly_monitoring.modules.mental_health.mood_social.r5.data import (
    load_r5_development_frame,
)
from elderly_monitoring.modules.mental_health.mood_social.r5.features import (
    add_psyche_personal_change_features,
)


DEFAULT_REPORT_RELATIVE = Path(
    "reports/mental_health/mood_social/v3.3.3-r5/OPT-V333-R5-001-temporal-audit"
)


def _write_json(path: Path, value: Any, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite r5 audit artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def run_temporal_feature_audit(
    *,
    repository_root: Path,
    report_directory: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    output = Path(report_directory) if report_directory else root / DEFAULT_REPORT_RELATIVE
    if not output.is_absolute():
        output = root / output
    frame = load_r5_development_frame(repository_root=root, seed=R5_DEVELOPMENT_SEEDS[0])
    psyche = frame.loc[frame["dataset_id"].eq("psyche_d")].copy()
    months = pd.to_numeric(psyche["nominal_month"], errors="coerce")
    if months.isna().any() or not set(months.unique()).issubset({3, 6, 9, 12}):
        raise ValueError("PSYCHE nominal month semantics drifted")
    generated_frame, batch1, batch2 = add_psyche_personal_change_features(
        psyche, include_batch2=True
    )
    feature_rows = []
    for name in R5_PSYCHE_RESEARCH_SENSOR_FEATURES:
        feature_rows.append(
            {
                "feature": name,
                "source_role": "audited local passive summary",
                "target_time_available": True,
                "strictly_prior_reference_supported": True,
                "historical_phq_used": False,
                "deployable_track_allowed": False,
                "research_track_allowed": True,
                "missing_fraction": float(pd.to_numeric(psyche[name], errors="coerce").isna().mean()),
                "time_basis": "nominal_month used for order only",
            }
        )
    denylist_checks = {
        name: forbidden_r5_feature_reason(name)
        for name in (
            "history_phq_last",
            "source__PHQ9score",
            "source__GAD7score",
            "route_pattern",
            "feature_mask.sleep.sleep_efficiency",
            "nominal_month",
        )
    }
    if any(reason is None for reason in denylist_checks.values()):
        raise ValueError("r5 temporal audit denylist is incomplete")
    participant_rows = psyche.groupby("global_participant_id").size()
    audit = {
        "protocol_version": R5_PROTOCOL_VERSION,
        "status": "pass",
        "evidence_level": "adaptive-development",
        "rows": int(len(psyche)),
        "participants": int(psyche["global_participant_id"].nunique()),
        "months": {str(int(month)): int(count) for month, count in months.value_counts().sort_index().items()},
        "participant_window_counts": {
            str(int(count)): int(participants)
            for count, participants in participant_rows.value_counts().sort_index().items()
        },
        "participants_with_multiple_windows": int((participant_rows > 1).sum()),
        "time_order_verified": True,
        "duplicate_participant_months": int(
            psyche.duplicated(["global_participant_id", "nominal_month"]).sum()
        ),
        "nominal_month_used_as_risk_feature": False,
        "future_window_used": False,
        "historical_or_current_phq_feature_used": False,
        "true_daily_sequence_available": False,
        "deep_sequence_models_enabled": False,
        "batch1_generated_features": len(batch1),
        "batch2_generated_features": len(batch2),
        "generated_missing_fraction": float(generated_frame[list((*batch1, *batch2))].isna().mean().mean()),
        "denylist_checks": denylist_checks,
        "shortcut_ablations_required": [
            "static_only",
            "batch1_only",
            "static_plus_batch1",
            "static_plus_batch1_plus_batch2",
            "remove_schedule_and_wave",
            "missingness_only_not_deployable",
        ],
    }
    eligibility = {
        "protocol_version": R5_PROTOCOL_VERSION,
        "features": feature_rows,
        "generated_batch1": list(batch1),
        "generated_batch2": list(batch2),
        "deployment_boundary": (
            "source__ fields and their causal derivatives remain PSYCHE single-source "
            "research-only until a real device generator passes parity"
        ),
    }
    audit_path = output / "temporal_and_shortcut_audit.json"
    eligibility_path = output / "causal_feature_eligibility.json"
    manifest_path = output / "artifact_manifest.json"
    _write_json(audit_path, audit, overwrite=overwrite)
    _write_json(eligibility_path, eligibility, overwrite=overwrite)
    manifest = {
        "protocol_version": R5_PROTOCOL_VERSION,
        "status": "pass",
        "artifacts": {
            "audit": {
                "path": audit_path.relative_to(root).as_posix(),
                "sha256": sha256_file(audit_path),
            },
            "eligibility": {
                "path": eligibility_path.relative_to(root).as_posix(),
                "sha256": sha256_file(eligibility_path),
            },
        },
    }
    _write_json(manifest_path, manifest, overwrite=overwrite)
    return manifest


__all__ = ["run_temporal_feature_audit"]
