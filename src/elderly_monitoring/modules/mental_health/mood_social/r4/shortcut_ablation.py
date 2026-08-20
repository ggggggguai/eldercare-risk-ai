"""Postlocked coverage-only and profile-only shortcut diagnostics for R4-001."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from elderly_monitoring.modules.mental_health.mood_social.r4.baseline import ap_context_metrics
from elderly_monitoring.modules.mental_health.mood_social.r4.data import (
    load_r4_development_frame,
    select_label_task,
)
from elderly_monitoring.modules.mental_health.mood_social.r4.features import (
    DEPLOYABLE_PROFILE_FEATURES,
)
from elderly_monitoring.modules.mental_health.mood_social.r4.selection import (
    CandidateSpec,
    fit_model,
    participant_equal_weights,
    predict_model,
)


DEFAULT_OUTPUT = Path(
    "reports/mental_health/mood_social/v3.3.3-r4/"
    "OPT-V333-R4-001-postlocked-shortcut-ablation"
)
COVERAGE_FEATURES = (
    "activity.valid_days",
    "activity.feature_coverage",
    "sleep.valid_nights",
    "sleep.feature_coverage",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def diagnostic_metric_table(frame: pd.DataFrame, probability: np.ndarray) -> dict[str, Any]:
    scored = frame.assign(probability=np.asarray(probability, dtype=float))

    def block(part: pd.DataFrame) -> dict[str, Any]:
        natural = ap_context_metrics(part["binary_target"], part["probability"])
        participant = ap_context_metrics(
            part["binary_target"],
            part["probability"],
            sample_weight=participant_equal_weights(part),
        )
        return {
            "rows": int(len(part)),
            "participants": int(part["global_participant_id"].nunique()),
            "positive_rows": int(part["binary_target"].sum()),
            "natural": natural,
            "participant_equal": participant,
        }

    return {
        "overall": block(scored),
        "by_source": {
            str(key): block(part) for key, part in scored.groupby("dataset_id", sort=True)
        },
        "by_route": {
            str(key): block(part) for key, part in scored.groupby("route_pattern", sort=True)
        },
    }


def _oof(frame: pd.DataFrame, features: tuple[str, ...], spec: CandidateSpec) -> np.ndarray:
    probability = pd.Series(np.nan, index=frame.index, dtype=float)
    for fold in sorted(frame["outer_fold"].astype(int).unique()):
        train = frame.loc[frame["outer_fold"].ne(fold)]
        test = frame.loc[frame["outer_fold"].eq(fold)]
        model = fit_model(train, features, spec, participant_equal=True)
        probability.loc[test.index] = predict_model(model, test, features)
    if probability.isna().any():
        raise ValueError("shortcut diagnostic OOF is incomplete")
    return probability.loc[frame.index].to_numpy(float)


def run_shortcut_ablation(repository_root: Path, output: Path | None = None) -> dict[str, Any]:
    root = repository_root.resolve()
    destination = (output or root / DEFAULT_OUTPUT).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise FileExistsError(f"refusing to overwrite shortcut ablation: {destination}")
    base = load_r4_development_frame(repository_root=root)
    prediction_parts: list[pd.DataFrame] = []
    reports: dict[str, Any] = {}
    specs = {
        "coverage_only": CandidateSpec(
            "coverage_only_hist", "hist_gradient",
            {"max_leaf_nodes": 7, "l2_regularization": 5.0, "max_iter": 150, "learning_rate": 0.04},
            20260812,
        ),
        "profile_only": CandidateSpec(
            "profile_only_elasticnet", "elasticnet", {"C": 0.1, "l1_ratio": 0.2}, 20260812
        ),
    }
    for task in ("phq_ge5_current", "phq_ge10_current"):
        frame = select_label_task(base, task)
        for diagnostic, features in (
            ("coverage_only", COVERAGE_FEATURES),
            ("profile_only", DEPLOYABLE_PROFILE_FEATURES),
        ):
            probability = _oof(frame, tuple(features), specs[diagnostic])
            report_key = f"{task}:{diagnostic}"
            reports[report_key] = diagnostic_metric_table(frame, probability)
            part = frame[
                ["r4_row_id", "global_participant_id", "dataset_id", "route_pattern", "outer_fold", "binary_target"]
            ].copy()
            part["r4_task_id"] = task
            part["diagnostic"] = diagnostic
            part["probability"] = probability
            prediction_parts.append(part)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    prediction_path = destination / "coverage_and_profile_shortcut_oof.parquet"
    report_path = destination / "coverage_and_profile_shortcut_metrics.json"
    predictions.to_parquet(prediction_path, index=False)
    _dump(
        report_path,
        {
            "protocol_version": "mood-social-v3.3.3-r4",
            "evidence_level": "postlocked_shortcut_diagnostic_not_deployable_not_model_selection",
            "coverage_fields_enter_formal_risk_model": False,
            "formal_locked_candidate_changed": False,
            "results": reports,
        },
    )
    manifest = {
        "status": "pass",
        "prediction_path": prediction_path.relative_to(root).as_posix(),
        "prediction_sha256": _sha256(prediction_path),
        "report_path": report_path.relative_to(root).as_posix(),
        "report_sha256": _sha256(report_path),
    }
    manifest_path = destination / "artifact_manifest.json"
    _dump(manifest_path, manifest)
    return manifest


__all__ = ["diagnostic_metric_table", "run_shortcut_ablation"]
