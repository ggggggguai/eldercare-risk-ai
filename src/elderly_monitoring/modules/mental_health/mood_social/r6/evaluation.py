"""One-time sealed confirmation and route-aware LODO evaluation for r6.

This module is intentionally outside the implementation tree that was hashed in
R6-005.  It may execute the locked recipe, but it cannot alter candidate
selection after confirmation is opened.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from elderly_monitoring.modules.mental_health.mood_social.r4.calibration import fit_calibration
from elderly_monitoring.modules.mental_health.mood_social.r4.features import route_signature
from elderly_monitoring.modules.mental_health.mood_social.r4.selection import participant_equal_weights
from elderly_monitoring.modules.mental_health.mood_social.r5.competition import _predict_multisource_split
from elderly_monitoring.modules.mental_health.mood_social.r5.evaluation import (
    MAJOR_SOURCES,
    binary_metrics,
    compatible_training_mask,
    paired_table,
)
from elderly_monitoring.modules.mental_health.mood_social.r5.fusion import (
    nested_fusion_predictions,
    select_train_only_workpoints,
)
from elderly_monitoring.modules.mental_health.mood_social.r5.objectives import project_independent_heads
from elderly_monitoring.modules.mental_health.mood_social.r6.baseline import (
    DEFAULT_REPORT_RELATIVE as BASELINE_RELATIVE,
    r5_locked_recipe,
    r5_locked_structure,
    run_r5_recipe_baseline,
)
from elderly_monitoring.modules.mental_health.mood_social.r6.competition import (
    _run_spec,
    _track_frame,
    predict_candidate,
)
from elderly_monitoring.modules.mental_health.mood_social.r6.confirmation import (
    DEFAULT_CONFIRMATION_RELATIVE,
    LOCKED_IMPLEMENTATION_FILES,
    confirmation_state,
    locked_confirmation_recipe,
    open_confirmation_once,
    seal_confirmation,
)
from elderly_monitoring.modules.mental_health.mood_social.r6.contract import (
    R6_CONFIRMATION_SEED,
    R6_PROTOCOL_VERSION,
    sha256_file,
    tree_hash,
    write_json,
)
from elderly_monitoring.modules.mental_health.mood_social.r6.data import r6_inner_fold_series
from elderly_monitoring.modules.mental_health.mood_social.r6.finalization import _select_calibration
from elderly_monitoring.modules.mental_health.mood_social.r6.selection import R6CandidateSpec


HEADS = {"ge5": "phq9_ge5_r3_target", "ge10": "phq9_ge10_r3_target"}
TRACKS = (
    "psyche_d_single_source_research",
    "multisource_deployable",
    "joint_route_recovery",
)


def _spec_from_lock(payload: dict[str, Any], track: str, seed: int) -> R6CandidateSpec:
    value = dict(payload["recipe"]["candidates"][track])
    value["seed"] = int(seed)
    return R6CandidateSpec(**value)


def _strict_confirmation_track(
    root: Path,
    *,
    track: str,
    seed: int,
    output: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    prediction_path = output / "candidate" / track / f"seed-{seed}.parquet"
    audit_path = output / "candidate" / track / f"seed-{seed}.audit.json"
    if prediction_path.is_file() and audit_path.is_file():
        return pd.read_parquet(prediction_path), json.loads(audit_path.read_text(encoding="utf-8"))

    lock = locked_confirmation_recipe(root)
    spec = _spec_from_lock(lock, track, seed)
    frame, features = _track_frame(root, seed, track)
    parts: list[pd.DataFrame] = []
    audits: dict[str, Any] = {}
    for outer in range(5):
        checkpoint = output / "_checkpoints" / "candidate" / track / f"outer-{outer}"
        checkpoint_prediction = checkpoint.with_suffix(".parquet")
        checkpoint_audit = checkpoint.with_suffix(".audit.json")
        if checkpoint_prediction.is_file() and checkpoint_audit.is_file():
            parts.append(pd.read_parquet(checkpoint_prediction))
            audits[f"outer-{outer}"] = json.loads(checkpoint_audit.read_text(encoding="utf-8"))
            continue
        train = frame.loc[frame["outer_fold"].ne(outer)].copy()
        test = frame.loc[frame["outer_fold"].eq(outer)].copy()
        inner = r6_inner_fold_series(train, outer).astype(int)
        inner_prediction, _ = _run_spec(train, features, spec, track, "full_inner", inner)
        inner_prediction = inner_prediction.merge(
            train[["r6_row_id"]].assign(inner_fold=inner.to_numpy(int)),
            on="r6_row_id",
            validate="one_to_one",
        )
        raw5, raw10 = predict_candidate(train, test, features, spec, track)
        calibrated_inner: dict[str, np.ndarray] = {}
        calibrated_outer: dict[str, np.ndarray] = {}
        calibration_audit: dict[str, Any] = {}
        for head, raw_outer in (("ge5", raw5), ("ge10", raw10)):
            method, crossfit, audit = _select_calibration(
                inner_prediction,
                inner_prediction[f"probability_{head}"].to_numpy(float),
                head,
            )
            fit = inner_prediction[["global_participant_id", HEADS[head]]].copy()
            fit["binary_target"] = fit[HEADS[head]].astype(int)
            calibrator = fit_calibration(
                fit,
                inner_prediction[f"probability_{head}"].to_numpy(float),
                method,  # type: ignore[arg-type]
            )
            calibrated_inner[head] = crossfit
            calibrated_outer[head] = calibrator.predict(raw_outer)
            calibration_audit[head] = audit
        inner5, inner10 = project_independent_heads(
            calibrated_inner["ge5"], calibrated_inner["ge10"]
        )
        outer5, outer10 = project_independent_heads(
            calibrated_outer["ge5"], calibrated_outer["ge10"]
        )
        workpoints = {
            "ge5": select_train_only_workpoints(train[HEADS["ge5"]].to_numpy(int), inner5),
            "ge10": select_train_only_workpoints(train[HEADS["ge10"]].to_numpy(int), inner10),
        }
        part = test[[
            "r6_row_id", "global_participant_id", "dataset_id", "route_pattern",
            "outer_fold", HEADS["ge5"], HEADS["ge10"],
        ]].copy()
        part["seed"] = int(seed)
        part["track"] = track
        part["candidate_id"] = spec.candidate_id
        part["probability_ge5"] = outer5
        part["probability_ge10"] = outer10
        for head, probability in (("ge5", outer5), ("ge10", outer10)):
            for point in ("competition", "safety"):
                threshold = float(workpoints[head][point]["threshold"])
                part[f"{head}_{point}_threshold"] = threshold
                part[f"{head}_{point}_positive"] = probability >= threshold
        audit = {
            "outer_fold": outer,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "participant_overlap_count": int(
                len(set(train["global_participant_id"]) & set(test["global_participant_id"]))
            ),
            "calibration": calibration_audit,
            "workpoints": workpoints,
            "outer_test_labels_used_for_selection": False,
            "monotonic_violation_count": int(np.sum(outer10 > outer5 + 1.0e-12)),
        }
        checkpoint_prediction.parent.mkdir(parents=True, exist_ok=True)
        part.to_parquet(checkpoint_prediction, index=False)
        write_json(checkpoint_audit, audit)
        parts.append(part)
        audits[f"outer-{outer}"] = audit
    result = pd.concat(parts, ignore_index=True)
    if len(result) != len(frame) or result.duplicated("r6_row_id").any():
        raise ValueError(f"r6 confirmation coverage drifted for {track}")
    if int(np.sum(result["probability_ge10"] > result["probability_ge5"] + 1.0e-12)):
        raise ValueError(f"r6 confirmation monotonicity failed for {track}")
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(prediction_path, index=False)
    write_json(audit_path, audits)
    return result, audits


def run_confirmation_predictions(*, repository_root: Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    state = confirmation_state(root)
    if not state["opened"] or state["sealed"]:
        raise PermissionError("r6 confirmation predictions require opened and unsealed state")
    output = root / DEFAULT_CONFIRMATION_RELATIVE
    baseline = run_r5_recipe_baseline(
        repository_root=root,
        seed=R6_CONFIRMATION_SEED,
        allow_confirmation=True,
    )
    candidate: dict[str, Any] = {}
    for track in TRACKS:
        prediction, audit = _strict_confirmation_track(
            root, track=track, seed=R6_CONFIRMATION_SEED, output=output
        )
        candidate[track] = {
            "rows": int(len(prediction)),
            "participants": int(prediction["global_participant_id"].nunique()),
            "outer_audits": len(audit),
        }
    return {"baseline": baseline, "candidate": candidate}


def _align(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    head: str,
) -> pd.DataFrame:
    selected = [
        "r6_row_id", f"baseline_probability_{head}", "outer_fold",
    ]
    result = candidate.merge(
        baseline[selected],
        on="r6_row_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_baseline"),
    )
    if result[f"baseline_probability_{head}"].isna().any():
        raise ValueError(f"r6 confirmation baseline alignment failed for {head}")
    if not result["outer_fold"].astype(int).eq(result["outer_fold_baseline"].astype(int)).all():
        raise ValueError(f"r6 confirmation fold alignment failed for {head}")
    result["binary_target"] = result[HEADS[head]].astype(int)
    result["candidate_probability"] = result[f"probability_{head}"].astype(float)
    result["baseline_probability"] = result[f"baseline_probability_{head}"].astype(float)
    return result


def _participant_aggregate(frame: pd.DataFrame) -> dict[str, Any]:
    grouped = frame.groupby(["global_participant_id", "dataset_id"], sort=True).agg(
        binary_target=("binary_target", "max"),
        candidate_probability=("candidate_probability", "mean"),
        baseline_probability=("baseline_probability", "mean"),
    ).reset_index()
    candidate = binary_metrics(grouped["binary_target"], grouped["candidate_probability"])
    baseline = binary_metrics(grouped["binary_target"], grouped["baseline_probability"])
    return {
        "semantics": "participant max target with mean window probability; descriptive",
        "participants": int(len(grouped)),
        "candidate": candidate,
        "baseline": baseline,
        "delta": {key: float(candidate[key] - baseline[key]) for key in ("auprc", "normalized_ap", "auroc", "brier", "ece")},
    }


def _macro_source(frame: pd.DataFrame) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for source, part in frame.groupby("dataset_id", sort=True):
        if part["binary_target"].nunique() != 2:
            rows[str(source)] = {"status": "single_class"}
            continue
        candidate = binary_metrics(part["binary_target"], part["candidate_probability"])
        baseline = binary_metrics(part["binary_target"], part["baseline_probability"])
        rows[str(source)] = {
            "status": "pass",
            "candidate": candidate,
            "baseline": baseline,
            "delta_auprc": float(candidate["auprc"] - baseline["auprc"]),
        }
    usable = [value for value in rows.values() if value["status"] == "pass"]
    return {
        "sources": rows,
        "macro_candidate_auprc": float(np.mean([value["candidate"]["auprc"] for value in usable])),
        "macro_baseline_auprc": float(np.mean([value["baseline"]["auprc"] for value in usable])),
    }


def evaluate_confirmation(
    *, repository_root: Path, repetitions: int = 2000
) -> dict[str, Any]:
    if repetitions != 2000:
        raise ValueError("formal r6 confirmation requires exactly 2000 bootstrap repetitions")
    root = Path(repository_root).resolve()
    state = confirmation_state(root)
    if not state["opened"] or state["sealed"]:
        raise PermissionError("r6 confirmation evaluation requires opened and unsealed state")
    output = root / DEFAULT_CONFIRMATION_RELATIVE
    report_path = output / "confirmation_evaluation.json"
    if report_path.is_file():
        return json.loads(report_path.read_text(encoding="utf-8"))
    baseline_path = root / BASELINE_RELATIVE / f"seed-{R6_CONFIRMATION_SEED}" / "r5_recipe_paired_baseline_oof.parquet"
    baseline = pd.read_parquet(baseline_path)
    baseline_multi = baseline.loc[baseline["track"].eq("multisource_deployable")].copy()
    baseline_psyche = baseline.loc[baseline["track"].eq("psyche_d_single_source_research")].copy()
    candidates = {
        track: pd.read_parquet(output / "candidate" / track / f"seed-{R6_CONFIRMATION_SEED}.parquet")
        for track in TRACKS
    }
    frames = {
        "multisource_phq_ge10": _align(candidates["multisource_deployable"], baseline_multi, head="ge10"),
        "multisource_phq_ge5_noninferiority": _align(candidates["multisource_deployable"], baseline_multi, head="ge5"),
        "psyche_d_phq_ge5": _align(candidates["psyche_d_single_source_research"], baseline_psyche, head="ge5"),
        "psyche_d_phq_ge10": _align(candidates["psyche_d_single_source_research"], baseline_psyche, head="ge10"),
        "joint_101_111_phq_ge10": _align(candidates["joint_route_recovery"], baseline_multi.loc[baseline_multi["route_pattern"].isin(["101", "111"])], head="ge10"),
    }
    table_heads = {
        "multisource_phq_ge10": "ge10",
        "multisource_phq_ge5_noninferiority": "ge5",
        "psyche_d_phq_ge5": "ge5",
        "psyche_d_phq_ge10": "ge10",
        "joint_101_111_phq_ge10": "ge10",
    }
    tables: dict[str, Any] = {}
    for index, (name, frame) in enumerate(frames.items()):
        table = paired_table(
            frame,
            head=table_heads[name],
            repetitions=repetitions,
            bootstrap_seed=R6_CONFIRMATION_SEED + index * 101,
        )
        table["participant_aggregate"] = _participant_aggregate(frame)
        table["macro_source"] = _macro_source(frame)
        tables[name] = table
    by_source: dict[str, Any] = {}
    by_route: dict[str, Any] = {}
    for name in ("multisource_phq_ge10", "multisource_phq_ge5_noninferiority"):
        frame = frames[name]
        head = table_heads[name]
        for source, part in frame.groupby("dataset_id", sort=True):
            by_source[f"{head}:{source}"] = paired_table(part, head=head, repetitions=0)
        for route, part in frame.groupby("route_pattern", sort=True):
            by_route[f"{head}:{route}"] = paired_table(part, head=head, repetitions=0)
    lock = locked_confirmation_recipe(root)
    development_gate = lock["recipe"]["development_gate_pass"]
    monotonic = {
        track: int(np.sum(part["probability_ge10"] > part["probability_ge5"] + 1.0e-12))
        for track, part in candidates.items()
    }

    def stable(table: dict[str, Any]) -> bool:
        return bool(
            table["nonnegative_auprc_folds"] >= 3
            or table["paired_participant_bootstrap"]["natural_delta"]["auprc"]["lower_95"] >= -0.012
        )

    multi10 = tables["multisource_phq_ge10"]
    multi5 = tables["multisource_phq_ge5_noninferiority"]
    psyche5 = tables["psyche_d_phq_ge5"]
    psyche10 = tables["psyche_d_phq_ge10"]
    paths = {
        "multisource_primary_delta_ge_0_004": multi10["natural"]["delta"]["auprc"] >= 0.004,
        "psyche_primary_delta_ge_0_004": psyche5["natural"]["delta"]["auprc"] >= 0.004,
        "compressed_path": False,
    }
    safeguards = {
        "multisource_other_head_floor": multi5["natural"]["delta"]["auprc"] >= -0.005,
        "psyche_other_head_floor": psyche10["natural"]["delta"]["auprc"] >= -0.005,
        "stability": stable(multi10) and stable(multi5) and stable(psyche5) and stable(psyche10),
        "auroc_floor": min(tables[name]["natural"]["delta"]["auroc"] for name in tables) >= -0.012,
        "brier_ceiling": max(tables[name]["natural"]["delta"]["brier"] for name in tables) <= 0.012,
        "ece_ceiling": max(tables[name]["natural"]["delta"]["ece"] for name in tables) <= 0.025,
        "coverage": min(table["candidate_coverage"] for table in tables.values()) >= 0.97,
        "dual_head_monotonicity": not any(monotonic.values()),
    }
    confirmation_metric_pass = bool(any(paths.values()) and all(safeguards.values()))
    development_pass = bool(any(development_gate.values()))
    offline_gate = {
        "paths": paths,
        "safeguards": safeguards,
        "confirmation_metric_pass": confirmation_metric_pass,
        "development_gate_precondition": development_pass,
        "pass": bool(confirmation_metric_pass and development_pass),
    }
    best_delta = float(max(table["natural"]["delta"]["auprc"] for table in tables.values()))
    report = {
        "protocol_version": R6_PROTOCOL_VERSION,
        "status": "offline_candidate_pass" if offline_gate["pass"] else "research_candidate_not_promoted",
        "evidence_level": "sealed reused-cohort confirmation; not a new-subject blind test",
        "confirmation_seed": R6_CONFIRMATION_SEED,
        "bootstrap_repetitions_per_table": repetitions,
        "tables": tables,
        "by_source": by_source,
        "by_route": by_route,
        "monotonic_violation_count": monotonic,
        "development_gate_pass": development_gate,
        "offline_gate": offline_gate,
        "best_confirmation_delta_auprc": best_delta,
        "same_information_model_stacking_stop": best_delta < 0.002,
        "confirmation_used_for_retuning": False,
        "production_default": "MH-20260802-013",
    }
    write_json(report_path, report)
    manifest = {
        "protocol_version": R6_PROTOCOL_VERSION,
        "status": report["status"],
        "evidence_level": report["evidence_level"],
        "artifacts": {
            "baseline_oof": {"path": baseline_path.relative_to(root).as_posix(), "sha256": sha256_file(baseline_path)},
            "candidate_oof": {
                track: {
                    "path": (output / "candidate" / track / f"seed-{R6_CONFIRMATION_SEED}.parquet").relative_to(root).as_posix(),
                    "sha256": sha256_file(output / "candidate" / track / f"seed-{R6_CONFIRMATION_SEED}.parquet"),
                }
                for track in TRACKS
            },
            "evaluation": {"path": report_path.relative_to(root).as_posix(), "sha256": sha256_file(report_path)},
        },
    }
    write_json(output / "confirmation_artifact_manifest.json", manifest)
    return report


def _r5_spec(root: Path, seed: int) -> Any:
    value = r5_locked_recipe(root)["tracks"]["multisource_deployable"]["stable_single_spec"]
    from elderly_monitoring.modules.mental_health.mood_social.r5.selection import R5CandidateSpec
    return R5CandidateSpec(str(value["candidate_id"]), str(value["family"]), dict(value["params"]), int(seed))


def _lodo_metric(frame: pd.DataFrame, head: str) -> dict[str, Any]:
    candidate_column = f"probability_{head}"
    baseline_column = f"baseline_probability_{head}"
    common = np.isfinite(frame[candidate_column]) & np.isfinite(frame[baseline_column])
    if not common.any():
        return {"status": "explicit_failure", "rows": int(len(frame)), "coverage": 0.0}
    scored = frame.loc[common].copy()
    if scored[HEADS[head]].nunique() != 2:
        return {"status": "single_class", "rows": int(len(frame)), "coverage": float(common.mean())}
    candidate = binary_metrics(scored[HEADS[head]], scored[candidate_column])
    baseline = binary_metrics(scored[HEADS[head]], scored[baseline_column])
    weight = participant_equal_weights(scored)
    pc = binary_metrics(scored[HEADS[head]], scored[candidate_column], sample_weight=weight)
    pb = binary_metrics(scored[HEADS[head]], scored[baseline_column], sample_weight=weight)
    return {
        "status": "pass",
        "rows": int(len(frame)),
        "scored_rows": int(len(scored)),
        "participants": int(scored["global_participant_id"].nunique()),
        "coverage": float(common.mean()),
        "candidate": candidate,
        "baseline": baseline,
        "delta": {key: float(candidate[key] - baseline[key]) for key in ("auprc", "normalized_ap", "auroc", "brier", "ece")},
        "participant_equal_delta": {key: float(pc[key] - pb[key]) for key in ("auprc", "normalized_ap", "auroc", "brier", "ece")},
    }


def _run_lodo_source(root: Path, frame: pd.DataFrame, source: str, output: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    prediction_path = output / "_checkpoints" / f"{source}.parquet"
    audit_path = output / "_checkpoints" / f"{source}.json"
    if prediction_path.is_file() and audit_path.is_file():
        return pd.read_parquet(prediction_path), json.loads(audit_path.read_text(encoding="utf-8"))
    test = frame.loc[frame["dataset_id"].eq(source)].copy()
    signatures = sorted({route_signature(value) for value in test["route_pattern"]})
    if not len(test) or len(signatures) != 1 or signatures[0] == "unsupported":
        raise ValueError(f"invalid held source runtime signature: {source} {signatures}")
    signature = signatures[0]
    train_all = frame.loc[frame["dataset_id"].ne(source)].copy()
    train = train_all.loc[compatible_training_mask(train_all, signature)].copy()
    train["feature_signature"] = signature
    test["feature_signature"] = signature
    candidate_track = "joint_route_recovery" if signature == "joint" else "multisource_deployable"
    lock = locked_confirmation_recipe(root)
    candidate_spec = _spec_from_lock(lock, candidate_track, R6_CONFIRMATION_SEED)
    output_frame = test[[
        "r6_row_id", "global_participant_id", "dataset_id", "route_pattern",
        "outer_fold", HEADS["ge5"], HEADS["ge10"],
    ]].copy()
    audit: dict[str, Any] = {
        "held_out_source": source,
        "feature_signature": signature,
        "candidate_track": candidate_track,
        "compatible_train_rows": int(len(train)),
        "compatible_train_participants": int(train["global_participant_id"].nunique()),
        "train_sources": sorted(train["dataset_id"].astype(str).unique()),
        "test_rows": int(len(test)),
        "participant_overlap_count": int(len(set(train["global_participant_id"]) & set(test["global_participant_id"]))),
        "held_source_labels_used_in_fit_calibration_or_workpoints": False,
    }
    try:
        if len(train) < 20:
            raise ValueError("fewer than 20 compatible training rows")
        inner = train["outer_fold"].astype(int)
        inner_prediction, _ = _run_spec(train, None, candidate_spec, candidate_track, "full_inner", inner)
        raw5, raw10 = predict_candidate(train, test, None, candidate_spec, candidate_track)
        calibrated: dict[str, np.ndarray] = {}
        for head, raw in (("ge5", raw5), ("ge10", raw10)):
            method, _, calibration_audit = _select_calibration(
                inner_prediction.assign(inner_fold=inner.to_numpy(int)),
                inner_prediction[f"probability_{head}"].to_numpy(float),
                head,
            )
            fit = inner_prediction[["global_participant_id", HEADS[head]]].copy()
            fit["binary_target"] = fit[HEADS[head]].astype(int)
            calibrator = fit_calibration(fit, inner_prediction[f"probability_{head}"].to_numpy(float), method)  # type: ignore[arg-type]
            calibrated[head] = calibrator.predict(raw)
            audit[f"candidate_{head}_calibration"] = calibration_audit
        p5, p10 = project_independent_heads(calibrated["ge5"], calibrated["ge10"])
        output_frame["probability_ge5"] = p5
        output_frame["probability_ge10"] = p10
        audit["candidate_status"] = "pass"
    except Exception as error:
        output_frame["probability_ge5"] = np.nan
        output_frame["probability_ge10"] = np.nan
        audit["candidate_status"] = "explicit_failure_abstain"
        audit["candidate_failure"] = f"{type(error).__name__}: {error}"
    try:
        r5_spec = _r5_spec(root, R6_CONFIRMATION_SEED)
        inner = train["outer_fold"].astype(int)
        baseline_prediction, baseline_audit = nested_fusion_predictions(
            train,
            test,
            None,
            inner,
            [r5_spec],
            "multisource_deployable",
            r5_locked_structure(root),
        )
        output_frame = output_frame.merge(
            baseline_prediction[["r5_row_id", "probability_ge5", "probability_ge10"]].rename(
                columns={
                    "r5_row_id": "r6_row_id",
                    "probability_ge5": "baseline_probability_ge5",
                    "probability_ge10": "baseline_probability_ge10",
                }
            ),
            on="r6_row_id",
            how="left",
            validate="one_to_one",
        )
        audit["baseline_status"] = "pass"
        audit["baseline_nested_audit"] = baseline_audit
    except Exception as error:
        output_frame["baseline_probability_ge5"] = np.nan
        output_frame["baseline_probability_ge10"] = np.nan
        audit["baseline_status"] = "explicit_failure_abstain"
        audit["baseline_failure"] = f"{type(error).__name__}: {error}"
    audit["heads"] = {head: _lodo_metric(output_frame, head) for head in HEADS}
    audit["route_decision"] = "none"
    for metric in audit["heads"].values():
        if metric.get("status") != "pass" or metric.get("coverage", 0.0) < 0.97:
            audit["route_decision"] = "block_related_source_route"
        elif metric["delta"]["auprc"] < -0.04 or metric["delta"]["auroc"] < -0.08:
            audit["route_decision"] = "block_related_source_route"
        elif metric["delta"]["auprc"] < -0.025 or metric["delta"]["auroc"] < -0.06:
            audit["route_decision"] = "warning"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    output_frame.to_parquet(prediction_path, index=False)
    write_json(audit_path, audit)
    return output_frame, audit


def run_lodo_evaluation(*, repository_root: Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    state = confirmation_state(root)
    if not state["opened"] or state["sealed"]:
        raise PermissionError("r6 LODO requires opened and unsealed confirmation state")
    output = root / DEFAULT_CONFIRMATION_RELATIVE / "lodo"
    report_path = output / "lodo_report.json"
    if report_path.is_file():
        return json.loads(report_path.read_text(encoding="utf-8"))
    frame, _ = _track_frame(root, R6_CONFIRMATION_SEED, "multisource_deployable")
    predictions: list[pd.DataFrame] = []
    audits: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    for source in MAJOR_SOURCES:
        try:
            prediction, audit = _run_lodo_source(root, frame, source, output)
        except Exception as error:
            test = frame.loc[frame["dataset_id"].eq(source)]
            prediction = test[["r6_row_id", "global_participant_id", "dataset_id", "route_pattern", "outer_fold", HEADS["ge5"], HEADS["ge10"]]].copy()
            for head in HEADS:
                prediction[f"probability_{head}"] = np.nan
                prediction[f"baseline_probability_{head}"] = np.nan
            audit = {
                "held_out_source": source,
                "candidate_status": "explicit_failure_abstain",
                "failure": f"{type(error).__name__}: {error}",
                "heads": {head: {"status": "explicit_failure", "rows": int(len(test)), "coverage": 0.0} for head in HEADS},
                "route_decision": "block_related_source_route",
            }
        predictions.append(prediction)
        audits[source] = audit
        for head, metric in audit["heads"].items():
            if metric.get("status") != "pass":
                failures.append({
                    "source": source,
                    "head": head,
                    "status": metric.get("status"),
                    "candidate_failure": audit.get("candidate_failure", audit.get("failure")),
                    "baseline_failure": audit.get("baseline_failure"),
                })
    combined = pd.concat(predictions, ignore_index=True)
    output.mkdir(parents=True, exist_ok=True)
    prediction_path = output / "lodo_predictions.parquet"
    failure_path = output / "failed_runs.json"
    combined.to_parquet(prediction_path, index=False)
    write_json(failure_path, failures)
    nhanes_ge10 = audits["nhanes"]["heads"]["ge10"]
    joint_unblocked = bool(
        nhanes_ge10.get("status") == "pass"
        and nhanes_ge10["delta"]["auprc"] >= -0.02
        and nhanes_ge10["delta"]["auroc"] >= -0.05
    )
    pass_heads = sum(metric.get("status") == "pass" for audit in audits.values() for metric in audit["heads"].values())
    report = {
        "protocol_version": R6_PROTOCOL_VERSION,
        "status": "pass" if pass_heads == 8 else "complete_with_explicit_failures",
        "evidence_level": "sealed reused-cohort confirmation LODO; not independent external validation",
        "seed": R6_CONFIRMATION_SEED,
        "policy": "diagnostic and source-route-local blocking; not a global veto",
        "sources": audits,
        "head_accounting": {"expected": 8, "passed": int(pass_heads), "explicit_failures": int(8 - pass_heads)},
        "joint_nhanes_unblock_gate": {
            "delta_auprc_floor": -0.02,
            "delta_auroc_floor": -0.05,
            "pass": joint_unblocked,
            "decision": "eligible_to_request_unblock" if joint_unblocked else "keep_MH-20260802-013_fallback",
        },
    }
    write_json(report_path, report)
    manifest = {
        "protocol_version": R6_PROTOCOL_VERSION,
        "status": report["status"],
        "artifacts": {
            "predictions": {"path": prediction_path.relative_to(root).as_posix(), "sha256": sha256_file(prediction_path)},
            "report": {"path": report_path.relative_to(root).as_posix(), "sha256": sha256_file(report_path)},
            "failures": {"path": failure_path.relative_to(root).as_posix(), "sha256": sha256_file(failure_path)},
        },
    }
    write_json(output / "artifact_manifest.json", manifest)
    return report


def preflight_confirmation(*, repository_root: Path) -> dict[str, Any]:
    """Verify all immutable prerequisites without opening the confirmation seed."""

    root = Path(repository_root).resolve()
    state = confirmation_state(root)
    if state["opened"] or state["sealed"]:
        raise PermissionError("r6 confirmation preflight must run before opening")
    lock = locked_confirmation_recipe(root)
    observed, files = tree_hash(root, LOCKED_IMPLEMENTATION_FILES)
    expected = lock["recipe"]["implementation_code_tree_sha256"]
    if observed != expected:
        raise PermissionError("r6 locked implementation tree drifted")
    split = root / "data/processed/mental_health/mood_social/v3.3.3-r6/protocol/splits" / f"seed-{R6_CONFIRMATION_SEED}.parquet"
    if not split.is_file():
        raise FileNotFoundError("r6 confirmation split is missing")
    synthetic = pd.DataFrame({
        "global_participant_id": [f"p{i}" for i in range(12)],
        "dataset_id": ["a"] * 6 + ["b"] * 6,
        "route_pattern": ["001"] * 12,
        "outer_fold": [i % 5 for i in range(12)],
        "binary_target": [0, 1] * 6,
        "candidate_probability": np.linspace(0.1, 0.9, 12),
        "baseline_probability": np.linspace(0.15, 0.85, 12),
    })
    synthetic_result = paired_table(synthetic, head="ge10", repetitions=10, bootstrap_seed=7)
    if synthetic_result.get("status") != "pass":
        raise RuntimeError("r6 synthetic metric preflight failed")
    return {
        "status": "pass",
        "confirmation_opened": False,
        "lock_sha256": sha256_file(root / DEFAULT_CONFIRMATION_RELATIVE / "candidate_recipe_lock.json"),
        "implementation_code_tree_sha256": observed,
        "implementation_files": files,
        "confirmation_split_sha256": sha256_file(split),
        "synthetic_metric_preflight": "pass",
    }


def run_sealed_confirmation(*, repository_root: Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    state = confirmation_state(root)
    if state["sealed"]:
        raise PermissionError("r6 confirmation is sealed and cannot be rerun")
    if not state["opened"]:
        open_confirmation_once(repository_root=root)
    run_confirmation_predictions(repository_root=root)
    evaluation = evaluate_confirmation(repository_root=root, repetitions=2000)
    lodo = run_lodo_evaluation(repository_root=root)
    output = root / DEFAULT_CONFIRMATION_RELATIVE
    artifacts = [
        root / BASELINE_RELATIVE / f"seed-{R6_CONFIRMATION_SEED}" / "r5_recipe_paired_baseline_oof.parquet",
        root / BASELINE_RELATIVE / f"seed-{R6_CONFIRMATION_SEED}" / "artifact_manifest.json",
        output / "confirmation_evaluation.json",
        output / "confirmation_artifact_manifest.json",
        output / "lodo" / "lodo_predictions.parquet",
        output / "lodo" / "lodo_report.json",
        output / "lodo" / "failed_runs.json",
        output / "lodo" / "artifact_manifest.json",
    ]
    for track in TRACKS:
        artifacts.extend([
            output / "candidate" / track / f"seed-{R6_CONFIRMATION_SEED}.parquet",
            output / "candidate" / track / f"seed-{R6_CONFIRMATION_SEED}.audit.json",
        ])
    seal = seal_confirmation(repository_root=root, artifact_paths=artifacts)
    return {"status": evaluation["status"], "evaluation": evaluation, "lodo": lodo, "seal": seal}


__all__ = [
    "HEADS", "TRACKS", "evaluate_confirmation", "preflight_confirmation",
    "run_confirmation_predictions", "run_lodo_evaluation", "run_sealed_confirmation",
]
