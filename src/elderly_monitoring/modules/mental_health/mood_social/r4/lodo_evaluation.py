"""Complete source-held-out retraining for the r4 deployable track."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from elderly_monitoring.modules.mental_health.mood_social.r3.baseline import (
    legacy_train_and_predict,
)
from elderly_monitoring.modules.mental_health.mood_social.r4.baseline import (
    ap_context_metrics,
)
from elderly_monitoring.modules.mental_health.mood_social.r4.calibration import (
    calibrate_outer_prediction,
)
from elderly_monitoring.modules.mental_health.mood_social.r4.contract import (
    R4_LABEL_TASKS,
    R4_PROTOCOL_VERSION,
)
from elderly_monitoring.modules.mental_health.mood_social.r4.data import (
    load_r4_development_frame,
    select_label_task,
)
from elderly_monitoring.modules.mental_health.mood_social.r4.features import (
    deployable_features_for_signature,
    route_signature,
)
from elderly_monitoring.modules.mental_health.mood_social.r4.selection import (
    participant_equal_weights,
    select_and_fit_predict,
)


MAJOR_SOURCES = (
    "nhanes",
    "nhanes_ssq_2005_2008",
    "psyche_d",
    "shenzhen_elderly",
)
DEFAULT_OUTPUT_RELATIVE = Path(
    "reports/mental_health/mood_social/v3.3.3-r4/OPT-V333-R4-008-evaluation/lodo"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def compatible_training_mask(frame: pd.DataFrame, signature: str) -> pd.Series:
    route = frame["route_pattern"].astype(str).str.pad(3, side="left", fillchar="0")
    if signature == "profile":
        return route.str[2].eq("1")
    if signature == "sleep":
        return route.str[1].eq("1")
    if signature == "activity":
        return route.str[0].eq("1")
    if signature == "joint":
        return route.eq("111")
    raise ValueError(f"unknown LODO signature: {signature}")


def _metrics(frame: pd.DataFrame) -> dict[str, Any]:
    selected = frame["candidate_probability"].notna() & frame["baseline_probability"].notna()
    scored = frame.loc[selected]
    if scored.empty:
        return {
            "rows": int(len(frame)),
            "participants": int(frame["global_participant_id"].nunique()),
            "coverage": 0.0,
            "status": "no_common_predictions",
        }
    target = scored["binary_target"].to_numpy(int)
    candidate = scored["candidate_probability"].to_numpy(float)
    baseline = scored["baseline_probability"].to_numpy(float)
    candidate_metric = ap_context_metrics(target, candidate)
    baseline_metric = ap_context_metrics(target, baseline)
    weight = participant_equal_weights(scored)
    candidate_participant = ap_context_metrics(target, candidate, sample_weight=weight)
    baseline_participant = ap_context_metrics(target, baseline, sample_weight=weight)
    return {
        "status": "pass",
        "rows": int(len(frame)),
        "scored_rows": int(len(scored)),
        "participants": int(frame["global_participant_id"].nunique()),
        "positive_rows": int(frame["binary_target"].sum()),
        "prevalence": float(frame["binary_target"].mean()),
        "coverage": float(selected.mean()),
        "candidate": candidate_metric,
        "baseline": baseline_metric,
        "delta": {
            key: float(candidate_metric[key] - baseline_metric[key])
            for key in ("auprc", "auroc", "brier", "ece", "normalized_ap")
        },
        "participant_equal_delta_auprc": float(
            candidate_participant["auprc"] - baseline_participant["auprc"]
        ),
    }


def _run_one(
    frame: pd.DataFrame,
    *,
    source: str,
    task_id: str,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    task_frame = select_label_task(frame, task_id)
    train_all = task_frame.loc[task_frame["dataset_id"].ne(source)].copy()
    test = task_frame.loc[task_frame["dataset_id"].eq(source)].copy()
    if test.empty:
        raise ValueError(f"LODO source absent: {source}")
    signatures = sorted({route_signature(value) for value in test["route_pattern"]})
    if "unsupported" in signatures or len(signatures) != 1:
        raise ValueError(f"major LODO source has ambiguous signature: {source} {signatures}")
    signature = signatures[0]
    train = train_all.loc[compatible_training_mask(train_all, signature)].copy()
    features = deployable_features_for_signature(signature)
    audit: dict[str, Any] = {
        "held_out_source": source,
        "task_id": task_id,
        "feature_signature": signature,
        "train_rows_all_sources": int(len(train_all)),
        "compatible_train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_sources": sorted(train["dataset_id"].astype(str).unique()),
        "participant_overlap": int(
            len(
                set(train_all["global_participant_id"].astype(str))
                & set(test["global_participant_id"].astype(str))
            )
        ),
        "held_source_labels_used_in_fit": False,
    }
    output = test[
        [
            "r4_row_id",
            "r3_row_id",
            "dataset_id",
            "global_participant_id",
            "route_pattern",
            "binary_target",
        ]
    ].copy()
    output["r4_task_id"] = task_id
    output["held_out_source"] = source
    output["feature_signature"] = signature

    try:
        raw_test, raw_train_oof, search_rows, selection = select_and_fit_predict(
            train,
            test,
            features,
            train["outer_fold"].astype(int),
            participant_equal=True,
            seed=seed,
        )
        calibration_frame = train[["global_participant_id", "binary_target"]].copy()
        calibration_frame["inner_fold"] = train["outer_fold"].astype(int).to_numpy()
        candidate, calibration = calibrate_outer_prediction(
            calibration_frame, raw_train_oof, raw_test
        )
        output["candidate_probability"] = candidate
        audit["candidate_status"] = "pass"
        audit["candidate_selection"] = selection
        audit["candidate_calibration"] = calibration
        audit["candidate_search"] = search_rows
    except Exception as exc:  # fail closed and retain source-local diagnosis
        output["candidate_probability"] = np.nan
        audit["candidate_status"] = "failed_abstain"
        audit["candidate_failure"] = f"{type(exc).__name__}: {exc}"

    try:
        baseline_train, _raw, baseline_probability = legacy_train_and_predict(
            train_all,
            train_all["outer_fold"].astype(int),
            test,
        )
        output["baseline_probability"] = baseline_probability
        audit["baseline_status"] = "pass"
        audit["baseline_second_level_oof_rows"] = int(len(baseline_train))
    except Exception as exc:
        output["baseline_probability"] = np.nan
        audit["baseline_status"] = "failed_abstain"
        audit["baseline_failure"] = f"{type(exc).__name__}: {exc}"

    audit["metrics"] = _metrics(output)
    audit["route_warning"] = (
        "block_related_route"
        if audit["metrics"].get("coverage", 0.0) < 0.93
        or audit["metrics"].get("delta", {}).get("auprc", 0.0) < -0.03
        or audit["metrics"].get("delta", {}).get("auroc", 0.0) < -0.05
        else "none"
    )
    return output, audit


def run_lodo_evaluation(
    *,
    repository_root: Path,
    output_directory: Path | None = None,
    overwrite: bool = False,
    seed: int = 20260812,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    output = Path(output_directory) if output_directory else root / DEFAULT_OUTPUT_RELATIVE
    if not output.is_absolute():
        output = root / output
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "_checkpoints"
    checkpoint.mkdir(parents=True, exist_ok=True)
    frame = load_r4_development_frame(repository_root=root)
    predictions: list[pd.DataFrame] = []
    audits: dict[str, Any] = {}
    for task_id in R4_LABEL_TASKS:
        for source in MAJOR_SOURCES:
            safe = source.replace("_", "-")
            prediction_path = checkpoint / f"{task_id}__{safe}.parquet"
            audit_path = checkpoint / f"{task_id}__{safe}.json"
            if prediction_path.exists() and audit_path.exists() and not overwrite:
                prediction = pd.read_parquet(prediction_path)
                audit = json.loads(audit_path.read_text(encoding="utf-8"))
            else:
                prediction, audit = _run_one(
                    frame, source=source, task_id=task_id, seed=seed
                )
                prediction.to_parquet(prediction_path, index=False)
                _write_json(audit_path, audit)
            predictions.append(prediction)
            audits[f"{task_id}:{source}"] = audit
    combined = pd.concat(predictions, ignore_index=True)
    prediction_path = output / "lodo_predictions.parquet"
    report_path = output / "lodo_report.json"
    manifest_path = output / "artifact_manifest.json"
    for path in (prediction_path, report_path, manifest_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite LODO artifact: {path}")
    combined.to_parquet(prediction_path, index=False)
    report = {
        "protocol_version": R4_PROTOCOL_VERSION,
        "status": "complete_with_route_local_warnings"
        if any(row["route_warning"] != "none" for row in audits.values())
        else "pass",
        "policy": "diagnostic_and_route_local_warning_not_global_veto",
        "historical_phq_used": False,
        "sources": audits,
    }
    _write_json(report_path, report)
    manifest = {
        "protocol_version": R4_PROTOCOL_VERSION,
        "status": report["status"],
        "artifacts": {
            "predictions": {
                "path": prediction_path.relative_to(root).as_posix(),
                "sha256": _sha256(prediction_path),
            },
            "report": {
                "path": report_path.relative_to(root).as_posix(),
                "sha256": _sha256(report_path),
            },
        },
    }
    _write_json(manifest_path, manifest)
    return manifest


__all__ = ["compatible_training_mask", "run_lodo_evaluation"]
