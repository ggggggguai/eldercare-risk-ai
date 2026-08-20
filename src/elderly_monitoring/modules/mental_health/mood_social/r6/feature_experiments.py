"""R6-002 fold-local stable-core, causal context and group ablations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from elderly_monitoring.modules.mental_health.mood_social.r4.baseline import ap_context_metrics
from elderly_monitoring.modules.mental_health.mood_social.r4.selection import participant_equal_weights
from elderly_monitoring.modules.mental_health.mood_social.r5.competition import _predict_psyche_split
from elderly_monitoring.modules.mental_health.mood_social.r5.features import add_psyche_personal_change_features
from elderly_monitoring.modules.mental_health.mood_social.r5.selection import R5CandidateSpec
from elderly_monitoring.modules.mental_health.mood_social.r6.baseline import load_r6_track_frame, r5_locked_recipe, r5_locked_structure
from elderly_monitoring.modules.mental_health.mood_social.r6.contract import R6_DEVELOPMENT_SEEDS, R6_PROTOCOL_VERSION, sha256_file, write_json


DEFAULT_REPORT_RELATIVE = Path("reports/mental_health/mood_social/v3.3.3-r6/OPT-V333-R6-002-features")
DIAGNOSTIC_RELATIVE = Path("reports/mental_health/mood_social/v3.3.3-r6/OPT-V333-R6-001-diagnostics")


def _spec(value: dict[str, Any]) -> R5CandidateSpec:
    return R5CandidateSpec(str(value["candidate_id"]), str(value["family"]), dict(value["params"]), int(value["seed"]))  # type: ignore[arg-type]


def _metrics(frame: pd.DataFrame, probability: np.ndarray, target: str) -> dict[str, Any]:
    y = frame[target].to_numpy(int)
    return {
        "natural": ap_context_metrics(y, probability),
        "participant_equal": ap_context_metrics(y, probability, sample_weight=participant_equal_weights(frame)),
    }


def _feature_group(name: str) -> str:
    if not name.startswith("r5_causal."):
        return "current_passive_summary"
    if "interaction_" in name or "weekend_minus" in name or "abnormal_sleep" in name:
        return "activity_sleep_interaction"
    if any(token in name for token in ("slope", "std", "mad", "iqr", "volatility")):
        return "prior_slope_volatility"
    if any(token in name for token in ("delta_", "robust_deviation", "log_ratio", "recent_minus")):
        return "personal_deviation"
    if any(token in name for token in ("lag1", "prior_", "recent2_")):
        return "strict_prior_context"
    return "other_causal"


def _run_scheme(
    frame: pd.DataFrame,
    features_by_fold: dict[int, tuple[str, ...]],
    spec: R5CandidateSpec,
    weight_scheme: str,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for outer in range(5):
        train = frame.loc[frame["outer_fold"].ne(outer)].copy()
        test = frame.loc[frame["outer_fold"].eq(outer)].copy()
        features = features_by_fold[outer]
        p5, p10 = _predict_psyche_split(train, test, features, spec, weight_scheme)
        part = test[["r6_row_id", "global_participant_id", "dataset_id", "route_pattern", "outer_fold", "phq9_ge5_r3_target", "phq9_ge10_r3_target"]].copy()
        part["probability_ge5"] = p5
        part["probability_ge10"] = p10
        parts.append(part)
    result = pd.concat(parts, ignore_index=True)
    if len(result) != len(frame) or result["r6_row_id"].duplicated().any():
        raise ValueError("R6-002 OOF coverage drifted")
    return result


def run_r6_feature_experiments(*, repository_root: Path, output_directory: Path | None = None, overwrite: bool = False) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    output = Path(output_directory) if output_directory else root / DEFAULT_REPORT_RELATIVE
    if not output.is_absolute():
        output = root / output
    output.mkdir(parents=True, exist_ok=True)
    stability = json.loads((root / DIAGNOSTIC_RELATIVE / "fold_local_feature_stability.json").read_text(encoding="utf-8"))
    stable_core = json.loads((root / DIAGNOSTIC_RELATIVE / "stable_core_feature_set.json").read_text(encoding="utf-8"))
    recipe = r5_locked_recipe(root)
    structure = r5_locked_structure(root)
    spec = _spec(recipe["tracks"]["psyche_d_single_source_research"]["stable_single_spec"])
    summary: dict[str, Any] = {}
    all_oof: list[pd.DataFrame] = []
    for seed in R6_DEVELOPMENT_SEEDS:
        base, full_features = load_r6_track_frame(root, seed=seed, track="psyche_d_single_source_research")
        assert full_features is not None
        enriched, _batch1, interactions = add_psyche_personal_change_features(
            base[[name for name in base.columns if not name.startswith("r5_causal.")]], include_batch2=True
        )
        fold_core = {
            outer: tuple(name for name in stability[str(seed)][str(outer)]["selected"] if name in enriched.columns)
            for outer in range(5)
        }
        schemes = {
            "full_486": {outer: tuple(full_features) for outer in range(5)},
            "fold_local_stable_core": fold_core,
            "fold_local_core_plus_interactions": {outer: tuple((*fold_core[outer], *interactions)) for outer in range(5)},
        }
        summary[str(seed)] = {"schemes": {}, "ablations": {}}
        scheme_predictions: dict[str, pd.DataFrame] = {}
        for scheme, fold_features in schemes.items():
            prediction = _run_scheme(enriched, fold_features, spec, structure["weight_scheme"])
            prediction["seed"] = seed
            prediction["scheme"] = scheme
            scheme_predictions[scheme] = prediction
            all_oof.append(prediction)
            summary[str(seed)]["schemes"][scheme] = {
                head: _metrics(prediction, prediction[f"probability_{head}"].to_numpy(float), f"phq9_{head}_r3_target")
                for head in ("ge5", "ge10")
            }
            summary[str(seed)]["schemes"][scheme]["feature_count_by_fold"] = {str(outer): len(fold_features[outer]) for outer in range(5)}
        # Group ablations are evaluated on the first development seed; the
        # selected feature scheme itself is evaluated on all three seeds.
        if seed == R6_DEVELOPMENT_SEEDS[0]:
            groups = sorted({_feature_group(name) for values in schemes["fold_local_core_plus_interactions"].values() for name in values})
            for group in groups:
                ablated = {
                    outer: tuple(name for name in schemes["fold_local_core_plus_interactions"][outer] if _feature_group(name) != group)
                    for outer in range(5)
                }
                prediction = _run_scheme(enriched, ablated, spec, structure["weight_scheme"])
                summary[str(seed)]["ablations"][group] = {
                    head: _metrics(prediction, prediction[f"probability_{head}"].to_numpy(float), f"phq9_{head}_r3_target")
                    for head in ("ge5", "ge10")
                }
    scheme_means: dict[str, Any] = {}
    for scheme in ("full_486", "fold_local_stable_core", "fold_local_core_plus_interactions"):
        scheme_means[scheme] = {}
        for head in ("ge5", "ge10"):
            values = [summary[str(seed)]["schemes"][scheme][head] for seed in R6_DEVELOPMENT_SEEDS]
            scheme_means[scheme][head] = {
                "mean_auprc": float(np.mean([value["natural"]["auprc"] for value in values])),
                "mean_participant_auprc": float(np.mean([value["participant_equal"]["auprc"] for value in values])),
                "by_seed_auprc": {str(seed): value["natural"]["auprc"] for seed, value in zip(R6_DEVELOPMENT_SEEDS, values)},
            }
    chosen = max(scheme_means, key=lambda scheme: 0.6 * scheme_means[scheme]["ge5"]["mean_auprc"] + 0.4 * scheme_means[scheme]["ge10"]["mean_auprc"])
    result_path = output / "feature_experiment_results.json"
    oof_path = output / "feature_experiment_oof.parquet"
    selection_path = output / "selected_feature_recipe.json"
    pd.concat(all_oof, ignore_index=True).to_parquet(oof_path, index=False)
    write_json(result_path, {"protocol_version": R6_PROTOCOL_VERSION, "status": "pass", "confirmation_opened": False, "by_seed": summary, "scheme_means": scheme_means}, overwrite=overwrite)
    write_json(selection_path, {
        "protocol_version": R6_PROTOCOL_VERSION, "status": "development-selected", "selected_scheme": chosen,
        "stable_core_feature_count": stable_core["stable_core_feature_count"], "feature_reduction_fraction": stable_core["reduction_fraction"],
        "causal_ordering": "nominal_month orders strictly-prior passive windows only",
        "single_window_fallback": "current passive summaries remain; unavailable causal context is imputed fold-locally",
        "participant_window_count_used_as_risk_feature": False, "confirmation_opened": False,
    }, overwrite=overwrite)
    manifest = {
        "protocol_version": R6_PROTOCOL_VERSION, "status": "pass", "stage": "R6-002", "selected_scheme": chosen,
        "confirmation_opened": False, "historical_or_current_phq_feature_used": False,
        "artifacts": {path.name: sha256_file(path) for path in (result_path, oof_path, selection_path)},
    }
    write_json(output / "artifact_manifest.json", manifest, overwrite=overwrite)
    return manifest


__all__ = ["DEFAULT_REPORT_RELATIVE", "run_r6_feature_experiments"]
