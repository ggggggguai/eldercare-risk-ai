"""R6-005 strict development selection, calibration and recipe lock."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from elderly_monitoring.modules.mental_health.mood_social.r4.baseline import ap_context_metrics
from elderly_monitoring.modules.mental_health.mood_social.r4.calibration import _crossfit, fit_calibration
from elderly_monitoring.modules.mental_health.mood_social.r4.selection import participant_equal_weights
from elderly_monitoring.modules.mental_health.mood_social.r5.fusion import select_train_only_workpoints
from elderly_monitoring.modules.mental_health.mood_social.r5.objectives import project_independent_heads
from elderly_monitoring.modules.mental_health.mood_social.r6.baseline import (
    DEFAULT_REPORT_RELATIVE as BASELINE_RELATIVE,
    load_r6_track_frame,
    r5_locked_recipe,
)
from elderly_monitoring.modules.mental_health.mood_social.r6.competition import (
    _run_spec,
    _save_run,
    _track_frame,
)
from elderly_monitoring.modules.mental_health.mood_social.r6.confirmation import lock_confirmation_recipe
from elderly_monitoring.modules.mental_health.mood_social.r6.contract import (
    R6_DEVELOPMENT_SEEDS,
    R6_PROTOCOL_VERSION,
    sha256_file,
    tree_hash,
    write_json,
)
from elderly_monitoring.modules.mental_health.mood_social.r6.data import r6_inner_fold_series
from elderly_monitoring.modules.mental_health.mood_social.r6.selection import R6CandidateSpec


DEFAULT_REPORT_RELATIVE = Path("reports/mental_health/mood_social/v3.3.3-r6/OPT-V333-R6-005-lock")
COMPETITION_RELATIVE = Path("reports/mental_health/mood_social/v3.3.3-r6/OPT-V333-R6-004-competition")
CALIBRATION_METHODS = ("none", "platt", "beta")
PRIMARY_HEAD = {
    "psyche_d_single_source_research": "ge5",
    "multisource_deployable": "ge10",
    "joint_route_recovery": "ge10",
}


def _metric(frame: pd.DataFrame, probability: np.ndarray, head: str) -> dict[str, Any]:
    target = frame[f"phq9_{head}_r3_target"].to_numpy(int)
    return {
        "natural": ap_context_metrics(target, probability),
        "participant_equal": ap_context_metrics(target, probability, sample_weight=participant_equal_weights(frame)),
    }


def _baseline_oof(root: Path, seed: int, track: str) -> pd.DataFrame:
    path = root / BASELINE_RELATIVE / f"seed-{seed}" / "r5_recipe_paired_baseline_oof.parquet"
    frame = pd.read_parquet(path)
    baseline_track = "multisource_deployable" if track == "joint_route_recovery" else track
    result = frame.loc[frame["track"].eq(baseline_track)].copy()
    if track == "joint_route_recovery":
        result = result.loc[result["route_pattern"].isin(["101", "111"])].copy()
    return result


def _weighted_reference_spec(root: Path, track: str, seed: int) -> R6CandidateSpec:
    recipe_track = "psyche_d_single_source_research" if track == "psyche_d_single_source_research" else "multisource_deployable"
    value = r5_locked_recipe(root)["tracks"][recipe_track]["stable_single_spec"]
    return R6CandidateSpec(
        f"r6_capped_weight_r5_topology_{'psyche' if track == 'psyche_d_single_source_research' else 'multi'}",
        track, value["family"], dict(value["params"]), int(seed),  # type: ignore[arg-type]
    )


def _run_weighted_references(root: Path, output: Path, overwrite: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in R6_DEVELOPMENT_SEEDS:
        for track in ("psyche_d_single_source_research", "multisource_deployable"):
            spec = _weighted_reference_spec(root, track, seed)
            directory = output / "weighted_reference" / f"seed-{seed}" / track / spec.candidate_id
            frame, features = _track_frame(root, seed, track)
            if (directory / "manifest.json").is_file() and not overwrite:
                metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
            else:
                prediction, metrics = _run_spec(frame, features, spec, track, "repeated_outer")
                _save_run(root, directory, spec, prediction, metrics, overwrite=overwrite)
            rows.append({"seed": seed, "track": track, "spec": asdict(spec), "metrics": metrics})
    return rows


def _candidate_path(root: Path, output: Path, row: dict[str, Any]) -> Path:
    seed = int(row["seed"]); track = str(row["track"]); candidate = str(row["spec"]["candidate_id"])
    if candidate.startswith("r6_capped_weight_r5_topology_"):
        return output / "weighted_reference" / f"seed-{seed}" / track / candidate / "oof.parquet"
    return root / COMPETITION_RELATIVE / "repeated_outer" / f"seed-{seed}" / track / candidate / "oof.parquet"


def _paired_delta(root: Path, output: Path, row: dict[str, Any]) -> dict[str, Any]:
    prediction = pd.read_parquet(_candidate_path(root, output, row))
    baseline = _baseline_oof(root, int(row["seed"]), str(row["track"]))
    merged = prediction.merge(
        baseline[["r6_row_id", "baseline_probability_ge5", "baseline_probability_ge10"]],
        on="r6_row_id", validate="one_to_one",
    )
    result: dict[str, Any] = {}
    for head in ("ge5", "ge10"):
        candidate = _metric(merged, merged[f"probability_{head}"].to_numpy(float), head)
        reference = _metric(merged, merged[f"baseline_probability_{head}"].to_numpy(float), head)
        result[head] = {
            "candidate": candidate, "baseline": reference,
            "delta_auprc": candidate["natural"]["auprc"] - reference["natural"]["auprc"],
            "delta_participant_auprc": candidate["participant_equal"]["auprc"] - reference["participant_equal"]["auprc"],
            "delta_auroc": candidate["natural"]["auroc"] - reference["natural"]["auroc"],
            "delta_brier": candidate["natural"]["brier"] - reference["natural"]["brier"],
            "delta_ece": candidate["natural"]["ece"] - reference["natural"]["ece"],
        }
    return result


def _select_candidates(root: Path, output: Path, competition_rows: list[dict[str, Any]], weighted_rows: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    rows = [*competition_rows, *weighted_rows]
    detailed: list[dict[str, Any]] = []
    for row in rows:
        value = dict(row)
        value["paired"] = _paired_delta(root, output, row)
        detailed.append(value)
    selected: dict[str, dict[str, Any]] = {}
    for track in PRIMARY_HEAD:
        candidates: list[dict[str, Any]] = []
        for candidate_id in sorted({row["spec"]["candidate_id"] for row in detailed if row["track"] == track}):
            values = [row for row in detailed if row["track"] == track and row["spec"]["candidate_id"] == candidate_id]
            if len(values) != len(R6_DEVELOPMENT_SEEDS):
                continue
            aggregate = {
                "track": track,
                "spec": values[0]["spec"],
                "by_seed": {str(row["seed"]): row["paired"] for row in values},
                "mean": {
                    head: {
                        metric: float(np.mean([row["paired"][head][metric] for row in values]))
                        for metric in ("delta_auprc", "delta_participant_auprc", "delta_auroc", "delta_brier", "delta_ece")
                    }
                    for head in ("ge5", "ge10")
                },
            }
            primary = PRIMARY_HEAD[track]
            aggregate["development_gate"] = bool(
                aggregate["mean"][primary]["delta_auprc"] >= 0.002
                and aggregate["mean"]["ge10" if primary == "ge5" else "ge5"]["delta_auprc"] >= -0.005
                and sum(row["paired"][primary]["delta_auprc"] >= 0 for row in values) >= 2
            )
            candidates.append(aggregate)
        if not candidates:
            continue
        primary = PRIMARY_HEAD[track]
        selected[track] = max(
            candidates,
            key=lambda row: (
                row["mean"][primary]["delta_auprc"],
                row["mean"][primary]["delta_participant_auprc"],
                row["mean"]["ge10" if primary == "ge5" else "ge5"]["delta_auprc"],
                row["spec"]["candidate_id"],
            ),
        )
    return selected, detailed


def _fusion_audit(root: Path, output: Path, detailed: list[dict[str, Any]]) -> dict[str, Any]:
    audit: dict[str, Any] = {}
    for track in PRIMARY_HEAD:
        by_candidate: dict[str, list[dict[str, Any]]] = {}
        for row in detailed:
            if row["track"] == track:
                by_candidate.setdefault(row["spec"]["candidate_id"], []).append(row)
        correlations: dict[str, list[float]] = {}
        ids = sorted(by_candidate)
        for left_index, left in enumerate(ids):
            for right in ids[left_index + 1 :]:
                common = sorted(set(row["seed"] for row in by_candidate[left]) & set(row["seed"] for row in by_candidate[right]))
                for seed in common:
                    left_row = next(row for row in by_candidate[left] if row["seed"] == seed)
                    right_row = next(row for row in by_candidate[right] if row["seed"] == seed)
                    a = pd.read_parquet(_candidate_path(root, output, left_row))
                    b = pd.read_parquet(_candidate_path(root, output, right_row))
                    merged = a.merge(b[["r6_row_id", "probability_ge5", "probability_ge10"]], on="r6_row_id", suffixes=("_a", "_b"), validate="one_to_one")
                    residual_a = np.concatenate([merged["probability_ge5_a"] - merged["phq9_ge5_r3_target"], merged["probability_ge10_a"] - merged["phq9_ge10_r3_target"]])
                    residual_b = np.concatenate([merged["probability_ge5_b"] - merged["phq9_ge5_r3_target"], merged["probability_ge10_b"] - merged["phq9_ge10_r3_target"]])
                    correlations.setdefault(f"{left}__{right}", []).append(float(np.corrcoef(residual_a, residual_b)[0, 1]))
        noninferior = {
            candidate: bool(
                len(rows) == 3
                and np.mean([row["paired"][PRIMARY_HEAD[track]]["delta_auprc"] for row in rows]) >= -0.002
                and np.mean([row["paired"]["ge10" if PRIMARY_HEAD[track] == "ge5" else "ge5"]["delta_auprc"] for row in rows]) >= -0.005
            )
            for candidate, rows in by_candidate.items()
        }
        eligible_pairs = [
            pair for pair, values in correlations.items()
            if all(noninferior.get(member, False) for member in pair.split("__"))
            and max(abs(value) for value in values) < 0.98
        ]
        audit[track] = {
            "noninferior_members": noninferior,
            "pairwise_residual_correlation_by_seed": correlations,
            "maximum_allowed_absolute_correlation": 0.98,
            "eligible_pairs": eligible_pairs,
            "fusion_methods_considered": ["convex_probability", "rank_average", "restricted_stack"],
            "fusion_run": False,
            "reason": "no pair satisfied both noninferiority and residual-correlation gates" if not eligible_pairs else "eligible pair exists; fusion implementation required",
            "fusion_gain_minimum": 0.0015,
        }
    return audit


def _select_calibration(frame: pd.DataFrame, raw: np.ndarray, head: str) -> tuple[str, np.ndarray, dict[str, Any]]:
    fit = frame[["r6_row_id", "global_participant_id", "inner_fold", f"phq9_{head}_r3_target"]].copy()
    fit["binary_target"] = fit[f"phq9_{head}_r3_target"].astype(int)
    candidates: dict[str, Any] = {}
    crossfitted: dict[str, np.ndarray] = {}
    raw_metric = ap_context_metrics(fit["binary_target"].to_numpy(int), raw)
    for method in CALIBRATION_METHODS:
        values = _crossfit(fit, raw, method)  # type: ignore[arg-type]
        crossfitted[method] = values
        candidates[method] = ap_context_metrics(fit["binary_target"].to_numpy(int), values)
    rank_safe = [method for method in CALIBRATION_METHODS if candidates[method]["auprc"] - raw_metric["auprc"] >= -0.001]
    selected = min(rank_safe or ["none"], key=lambda method: (candidates[method]["brier"] + 0.25 * candidates[method]["ece"], CALIBRATION_METHODS.index(method)))
    return selected, crossfitted[selected], {"raw": raw_metric, "candidates": candidates, "selected": selected, "selection_used_outer_labels": False}


def _strict_nested_candidate(root: Path, output: Path, selected: dict[str, Any], overwrite: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
    track = str(selected["track"]); base_spec = R6CandidateSpec(**selected["spec"])
    parts: list[pd.DataFrame] = []; audits: dict[str, Any] = {}
    for seed in R6_DEVELOPMENT_SEEDS:
        spec = R6CandidateSpec(base_spec.candidate_id, base_spec.track, base_spec.family, base_spec.params, int(seed))
        raw_path = None
        row = {"seed": seed, "track": track, "spec": asdict(spec)}
        if spec.candidate_id.startswith("r6_capped_weight_r5_topology_"):
            raw_path = output / "weighted_reference" / f"seed-{seed}" / track / spec.candidate_id / "oof.parquet"
        else:
            raw_path = root / COMPETITION_RELATIVE / "repeated_outer" / f"seed-{seed}" / track / spec.candidate_id / "oof.parquet"
        raw_outer_all = pd.read_parquet(raw_path)
        frame, features = _track_frame(root, seed, track)
        for outer in range(5):
            train = frame.loc[frame["outer_fold"].ne(outer)].copy()
            test = frame.loc[frame["outer_fold"].eq(outer)].copy()
            inner = r6_inner_fold_series(train, outer).astype(int)
            inner_prediction, _inner_metrics = _run_spec(train, features, spec, track, "full_inner", inner)
            inner_prediction = inner_prediction.merge(
                train[["r6_row_id"]].assign(inner_fold=inner.to_numpy(int)),
                on="r6_row_id", validate="one_to_one",
            )
            raw_outer = test[["r6_row_id"]].merge(
                raw_outer_all[["r6_row_id", "probability_ge5", "probability_ge10"]],
                on="r6_row_id", validate="one_to_one",
            )
            calibrated_inner: dict[str, np.ndarray] = {}; calibrated_outer: dict[str, np.ndarray] = {}; audit: dict[str, Any] = {}
            for head in ("ge5", "ge10"):
                method, crossfit, calibration_audit = _select_calibration(inner_prediction, inner_prediction[f"probability_{head}"].to_numpy(float), head)
                fit = inner_prediction[["global_participant_id", f"phq9_{head}_r3_target"]].copy()
                fit["binary_target"] = fit[f"phq9_{head}_r3_target"].astype(int)
                calibrator = fit_calibration(fit, inner_prediction[f"probability_{head}"].to_numpy(float), method)  # type: ignore[arg-type]
                calibrated_inner[head] = crossfit
                calibrated_outer[head] = calibrator.predict(raw_outer[f"probability_{head}"].to_numpy(float))
                audit[head] = calibration_audit
            inner5, inner10 = project_independent_heads(calibrated_inner["ge5"], calibrated_inner["ge10"])
            outer5, outer10 = project_independent_heads(calibrated_outer["ge5"], calibrated_outer["ge10"])
            workpoints = {
                "ge5": select_train_only_workpoints(inner_prediction["phq9_ge5_r3_target"].to_numpy(int), inner5),
                "ge10": select_train_only_workpoints(inner_prediction["phq9_ge10_r3_target"].to_numpy(int), inner10),
            }
            part = test[["r6_row_id", "global_participant_id", "dataset_id", "route_pattern", "outer_fold", "phq9_ge5_r3_target", "phq9_ge10_r3_target"]].copy()
            part["seed"] = seed; part["track"] = track; part["candidate_id"] = spec.candidate_id
            part["probability_ge5"] = outer5; part["probability_ge10"] = outer10
            for head, probability in (("ge5", outer5), ("ge10", outer10)):
                for point in ("competition", "safety"):
                    threshold = float(workpoints[head][point]["threshold"])
                    part[f"{head}_{point}_threshold"] = threshold
                    part[f"{head}_{point}_positive"] = probability >= threshold
            parts.append(part)
            audits[f"seed-{seed}:outer-{outer}"] = {"calibration": audit, "workpoints": workpoints, "selection_used_outer_labels": False}
    result = pd.concat(parts, ignore_index=True)
    expected = sum(len(_track_frame(root, seed, track)[0]) for seed in R6_DEVELOPMENT_SEEDS)
    if len(result) != expected or result.duplicated(["seed", "r6_row_id"]).any():
        raise ValueError(f"strict nested candidate coverage drifted for {track}")
    return result, audits


def run_r6_finalization(*, repository_root: Path, output_directory: Path | None = None, overwrite: bool = False, lock_confirmation: bool = True) -> dict[str, Any]:
    root = Path(repository_root).resolve(); output = Path(output_directory) if output_directory else root / DEFAULT_REPORT_RELATIVE
    if not output.is_absolute(): output = root / output
    output.mkdir(parents=True, exist_ok=True)
    weighted_rows = _run_weighted_references(root, output, overwrite)
    competition = json.loads((root / COMPETITION_RELATIVE / "repeated_outer_results.json").read_text(encoding="utf-8"))["rows"]
    selected, detailed = _select_candidates(root, output, competition, weighted_rows)
    fusion = _fusion_audit(root, output, detailed)
    if any(value["eligible_pairs"] for value in fusion.values()):
        raise RuntimeError("R6-005 found an eligible fusion pair; explicit fusion is required before lock")
    strict_parts: list[pd.DataFrame] = []; strict_audits: dict[str, Any] = {}
    for track, candidate in selected.items():
        checkpoint = output / "strict_nested" / track
        prediction_path = checkpoint / "oof.parquet"
        audit_checkpoint = checkpoint / "audit.json"
        if prediction_path.is_file() and audit_checkpoint.is_file() and not overwrite:
            prediction = pd.read_parquet(prediction_path)
            audit = json.loads(audit_checkpoint.read_text(encoding="utf-8"))
        else:
            prediction, audit = _strict_nested_candidate(root, output, candidate, overwrite)
            checkpoint.mkdir(parents=True, exist_ok=True)
            prediction.to_parquet(prediction_path, index=False)
            write_json(audit_checkpoint, audit, overwrite=overwrite)
        strict_parts.append(prediction); strict_audits[track] = audit
    strict = pd.concat(strict_parts, ignore_index=True)
    strict_path = output / "strict_nested_development_oof.parquet"; audit_path = output / "strict_nested_calibration_workpoint_audit.json"
    selection_path = output / "development_selection.json"; fusion_path = output / "fusion_audit.json"; weighted_path = output / "weighted_reference_results.json"
    strict.to_parquet(strict_path, index=False)
    write_json(audit_path, strict_audits, overwrite=overwrite)
    write_json(selection_path, {"protocol_version": R6_PROTOCOL_VERSION, "status": "development-selected", "selected": selected, "confirmation_opened": False}, overwrite=overwrite)
    write_json(fusion_path, fusion, overwrite=overwrite)
    write_json(weighted_path, {"protocol_version": R6_PROTOCOL_VERSION, "rows": weighted_rows}, overwrite=overwrite)
    code_files = [
        Path("src/elderly_monitoring/modules/mental_health/mood_social/r6") / name
        for name in ("selection.py", "competition.py", "objectives.py", "finalization.py", "confirmation.py")
    ]
    code_sha, _ = tree_hash(root, code_files)
    development_sha, _ = tree_hash(root, [selection_path.relative_to(root), fusion_path.relative_to(root), strict_path.relative_to(root), audit_path.relative_to(root)])
    recipe = {
        "features": {"psyche_d_single_source_research": "r5_full_486", "multisource_deployable": "runtime_signature_allowlist", "joint_route_recovery": "runtime_joint_signature"},
        "candidates": {track: value["spec"] for track, value in selected.items()},
        "objective_structure": "conditional_ordered",
        "training_weights": "participant_equal_times_capped_class",
        "fusion": {"selected": "single_model", "reason": "no eligible low-correlation noninferior pair", "minimum_gain": 0.0015},
        "calibration": {"methods": list(CALIBRATION_METHODS), "selection": "per-outer secondary crossfit; rank loss >=-0.001 then Brier+0.25*ECE"},
        "workpoints": "per-outer calibrated inner OOF competition and safety",
        "development_evidence_sha256": development_sha,
        "implementation_code_tree_sha256": code_sha,
        "development_gate_pass": {track: bool(value["development_gate"]) for track, value in selected.items()},
        "release_if_gate_fails": "research_candidate_only; keep MH-20260802-013 production fallback",
    }
    lock = lock_confirmation_recipe(repository_root=root, recipe=recipe) if lock_confirmation else None
    recipe_path = output / "locked_recipe.json"
    write_json(recipe_path, {"protocol_version": R6_PROTOCOL_VERSION, "status": "locked_before_confirmation", "recipe": recipe, "confirmation_lock": lock}, overwrite=overwrite)
    manifest = {
        "protocol_version": R6_PROTOCOL_VERSION, "status": "pass", "stage": "R6-005",
        "selected_tracks": sorted(selected), "development_gate_pass": recipe["development_gate_pass"],
        "fusion_selected": "single_model", "confirmation_opened": False,
        "artifacts": {path.name: sha256_file(path) for path in (strict_path, audit_path, selection_path, fusion_path, weighted_path, recipe_path)},
    }
    write_json(output / "artifact_manifest.json", manifest, overwrite=overwrite)
    return manifest


__all__ = ["CALIBRATION_METHODS", "run_r6_finalization"]
