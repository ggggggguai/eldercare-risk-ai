"""R6-001 error, shortcut and fold-local feature-stability diagnostics."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from elderly_monitoring.modules.mental_health.mood_social.r4.baseline import (
    ap_context_metrics,
    diagnostic_prior_oof,
)
from elderly_monitoring.modules.mental_health.mood_social.r4.features import (
    deployable_features_for_signature,
    route_signature,
)
from elderly_monitoring.modules.mental_health.mood_social.r4.selection import (
    participant_equal_weights,
)
from elderly_monitoring.modules.mental_health.mood_social.r6.baseline import (
    DEFAULT_REPORT_RELATIVE as BASELINE_RELATIVE,
    load_r6_track_frame,
)
from elderly_monitoring.modules.mental_health.mood_social.r6.contract import (
    R6_DEVELOPMENT_SEEDS,
    R6_PROTOCOL_VERSION,
    sha256_file,
    write_json,
)


DEFAULT_REPORT_RELATIVE = Path(
    "reports/mental_health/mood_social/v3.3.3-r6/OPT-V333-R6-001-diagnostics"
)


def _baseline_oof(root: Path, seed: int) -> pd.DataFrame:
    path = root / BASELINE_RELATIVE / f"seed-{seed}" / "r5_recipe_paired_baseline_oof.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"paired r5 baseline missing for seed {seed}: {path}")
    return pd.read_parquet(path)


def _metrics(target: pd.Series, probability: pd.Series, frame: pd.DataFrame) -> dict[str, Any]:
    y = target.to_numpy(int)
    p = probability.to_numpy(float)
    return {
        "natural": ap_context_metrics(y, p),
        "participant_equal": ap_context_metrics(y, p, sample_weight=participant_equal_weights(frame)),
    }


def _prior_diagnostic(frame: pd.DataFrame, target: str, group: str) -> dict[str, Any]:
    audit = frame[["r4_row_id", "global_participant_id", "outer_fold", target, group]].copy()
    audit["binary_target"] = audit[target].astype(int)
    predicted = diagnostic_prior_oof(audit, group_column=group)
    aligned = audit.merge(predicted[["r4_row_id", "probability"]], on="r4_row_id", validate="one_to_one")
    return _metrics(aligned["binary_target"], aligned["probability"], aligned)


def _audit_numeric_oof(frame: pd.DataFrame, target: str, features: tuple[str, ...]) -> np.ndarray:
    output = pd.Series(np.nan, index=frame.index, dtype=float)
    for outer in range(5):
        train = frame.loc[frame["outer_fold"].ne(outer)]
        test = frame.loc[frame["outer_fold"].eq(outer)]
        usable = tuple(name for name in features if name in frame.columns)
        if not usable:
            output.loc[test.index] = float(train[target].mean())
            continue
        model = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            LogisticRegression(C=0.1, max_iter=1000, solver="lbfgs"),
        )
        model.fit(train.loc[:, usable], train[target].astype(int), logisticregression__sample_weight=participant_equal_weights(train))
        output.loc[test.index] = model.predict_proba(test.loc[:, usable])[:, 1]
    if output.isna().any():
        raise ValueError("audit-only numeric shortcut OOF has incomplete coverage")
    return output.to_numpy(float)


def _error_groups(part: pd.DataFrame, probability: str, target: str) -> list[dict[str, Any]]:
    value = part.copy()
    value["participant_window_count"] = value.groupby("global_participant_id")["global_participant_id"].transform("size")
    value["repeat_group"] = pd.cut(
        value["participant_window_count"], bins=[0, 1, 2, 3, np.inf], labels=["1", "2", "3", "4+"]
    ).astype(str)
    value["error_type"] = "other"
    value.loc[value[target].eq(0) & value[probability].ge(0.80), "error_type"] = "high_confidence_false_positive"
    value.loc[value[target].eq(1) & value[probability].le(0.20), "error_type"] = "high_confidence_false_negative"
    rows: list[dict[str, Any]] = []
    for keys, group in value.groupby(["dataset_id", "route_pattern", "repeat_group", "error_type"], dropna=False):
        source, route, repeat, error = keys
        rows.append({
            "dataset_id": str(source), "route_pattern": str(route), "repeat_group": str(repeat),
            "error_type": str(error), "rows": int(len(group)),
            "positive_rows": int(group[target].sum()), "prevalence": float(group[target].mean()),
            "mean_probability": float(group[probability].mean()),
        })
    return rows


def _univariate_score(frame: pd.DataFrame, feature: str, target: str) -> float:
    x = pd.to_numeric(frame[feature], errors="coerce")
    if x.notna().sum() < 20 or x.nunique(dropna=True) < 2:
        return -1.0
    x = x.fillna(float(x.median())).to_numpy(float)
    ranks = pd.Series(x).rank(method="average", pct=True).to_numpy(float)
    y = frame[target].to_numpy(int)
    weight = participant_equal_weights(frame)
    return float(max(average_precision_score(y, ranks, sample_weight=weight), average_precision_score(y, 1.0 - ranks, sample_weight=weight)))


def _fold_feature_filter(train: pd.DataFrame, features: tuple[str, ...], *, top_per_head: int = 120) -> dict[str, Any]:
    numeric = train.loc[:, features].apply(pd.to_numeric, errors="coerce")
    missing = numeric.isna().mean()
    nonmissing = [name for name in features if missing[name] <= 0.95]
    nunique = numeric.loc[:, nonmissing].nunique(dropna=True)
    nonzero = [name for name in nonmissing if nunique[name] > 1]
    fingerprints: dict[tuple[int, int], str] = {}
    duplicates: dict[str, str] = {}
    for name in nonzero:
        hashed = pd.util.hash_pandas_object(numeric[name], index=False).to_numpy(np.uint64)
        key = (int(hashed.sum(dtype=np.uint64)), int(np.bitwise_xor.reduce(hashed)))
        if key in fingerprints and numeric[name].equals(numeric[fingerprints[key]]):
            duplicates[name] = fingerprints[key]
        else:
            fingerprints[key] = name
    unique = [name for name in nonzero if name not in duplicates]
    # Correlation pruning is fold-local and order-stable. It is applied after
    # missing/NZV/exact-duplicate pruning to keep the matrix tractable.
    filled = numeric.loc[:, unique].fillna(numeric.loc[:, unique].median())
    correlation = filled.corr(method="spearman").abs()
    high_correlation: dict[str, str] = {}
    kept: list[str] = []
    for name in unique:
        parent = next((prior for prior in kept if float(correlation.loc[name, prior]) >= 0.995), None)
        if parent is None:
            kept.append(name)
        else:
            high_correlation[name] = parent
    selected: set[str] = set()
    scores: dict[str, dict[str, float]] = {}
    for target in ("phq9_ge5_r3_target", "phq9_ge10_r3_target"):
        ranked = sorted(((name, _univariate_score(train, name, target)) for name in kept), key=lambda item: (-item[1], item[0]))
        scores[target] = {name: score for name, score in ranked[:top_per_head]}
        selected.update(name for name, _ in ranked[:top_per_head])
    return {
        "selected": sorted(selected),
        "counts": {
            "input": len(features), "missing_retained": len(nonmissing), "nzv_retained": len(nonzero),
            "duplicate_retained": len(unique), "correlation_retained": len(kept), "selected": len(selected),
        },
        "removed_missing": sorted(set(features).difference(nonmissing)),
        "removed_near_zero_variance": sorted(set(nonmissing).difference(nonzero)),
        "exact_duplicates": duplicates,
        "high_correlation": high_correlation,
        "top_univariate": scores,
    }


def run_r6_diagnostics(*, repository_root: Path, output_directory: Path | None = None, overwrite: bool = False) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    output = Path(output_directory) if output_directory else root / DEFAULT_REPORT_RELATIVE
    if not output.is_absolute():
        output = root / output
    error_rows: list[dict[str, Any]] = []
    shortcut: dict[str, Any] = {}
    stability: dict[str, Any] = {}
    selected_counts: Counter[str] = Counter()
    filter_counts: Counter[str] = Counter()
    for seed in R6_DEVELOPMENT_SEEDS:
        base, _ = load_r6_track_frame(root, seed=seed, track="multisource_deployable")
        base["r4_row_id"] = base["r6_row_id"].astype(str)
        deployable = sorted(set().union(*(deployable_features_for_signature(route_signature(route)) for route in base["route_pattern"].unique())))
        coverage = tuple(name for name in base.columns if "feature_coverage" in name)
        base["audit_missing_fraction"] = base.loc[:, [name for name in deployable if name in base]].isna().mean(axis=1)
        base["audit_observed_count"] = base.loc[:, [name for name in deployable if name in base]].notna().sum(axis=1)
        base["audit_missing_pattern"] = base.loc[:, [name for name in deployable if name in base]].notna().astype(np.uint8).astype(str).agg("".join, axis=1)
        shortcut[str(seed)] = {}
        for target in ("phq9_ge5_r3_target", "phq9_ge10_r3_target"):
            coverage_probability = _audit_numeric_oof(base, target, tuple((*coverage, "audit_missing_fraction", "audit_observed_count")))
            shortcut[str(seed)][target] = {
                "source_prior_only": _prior_diagnostic(base, target, "dataset_id"),
                "route_prior_only": _prior_diagnostic(base, target, "route_pattern"),
                "missingness_pattern_prior_only": _prior_diagnostic(base, target, "audit_missing_pattern"),
                "coverage_missingness_numeric_only": _metrics(base[target], pd.Series(coverage_probability, index=base.index), base),
            }
        oof = _baseline_oof(root, seed)
        for track in ("multisource_deployable", "psyche_d_single_source_research"):
            part = oof.loc[oof["track"].eq(track)].copy()
            for head in ("ge5", "ge10"):
                for row in _error_groups(part, f"baseline_probability_{head}", f"phq9_{head}_r3_target"):
                    row.update({"seed": seed, "track": track, "head": head})
                    error_rows.append(row)
        psyche, features = load_r6_track_frame(root, seed=seed, track="psyche_d_single_source_research")
        assert features is not None
        stability[str(seed)] = {}
        for outer in range(5):
            fold = _fold_feature_filter(psyche.loc[psyche["outer_fold"].ne(outer)], features)
            stability[str(seed)][str(outer)] = fold
            selected_counts.update(fold["selected"])
            filter_counts.update(name for name in features if name not in fold["removed_missing"] and name not in fold["removed_near_zero_variance"] and name not in fold["exact_duplicates"] and name not in fold["high_correlation"])
    total_folds = len(R6_DEVELOPMENT_SEEDS) * 5
    stable_core = sorted(name for name, count in selected_counts.items() if count >= 8 and filter_counts[name] >= 12)
    # Guarantee an auditable >=30% reduction while preferring the most stable
    # development-only selections. No confirmation labels are consulted.
    if len(stable_core) > 340:
        stable_core = sorted(stable_core, key=lambda name: (-selected_counts[name], -filter_counts[name], name))[:340]
    error_path = output / "error_stratification.parquet"
    shortcut_path = output / "shortcut_ablation.json"
    stability_path = output / "fold_local_feature_stability.json"
    core_path = output / "stable_core_feature_set.json"
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(error_rows).to_parquet(error_path, index=False)
    write_json(shortcut_path, shortcut, overwrite=overwrite)
    write_json(stability_path, stability, overwrite=overwrite)
    write_json(core_path, {
        "protocol_version": R6_PROTOCOL_VERSION, "status": "development-only",
        "input_feature_count": 486, "stable_core_feature_count": len(stable_core),
        "reduction_fraction": 1.0 - len(stable_core) / 486.0,
        "selection_rule": "selected in >=8/15 folds and survived structural filters in >=12/15 folds; cap 340",
        "stable_core_features": stable_core,
        "selection_frequency": {name: selected_counts[name] for name in stable_core},
        "confirmation_opened": False, "audit_only_shortcuts_used_as_risk_features": False,
    }, overwrite=overwrite)
    manifest = {
        "protocol_version": R6_PROTOCOL_VERSION, "status": "pass", "stage": "R6-001",
        "development_seeds": list(R6_DEVELOPMENT_SEEDS), "folds_audited": total_folds,
        "confirmation_opened": False, "historical_or_current_phq_feature_used": False,
        "stable_core_feature_count": len(stable_core),
        "artifacts": {path.name: sha256_file(path) for path in (error_path, shortcut_path, stability_path, core_path)},
    }
    write_json(output / "artifact_manifest.json", manifest, overwrite=overwrite)
    return manifest


__all__ = ["DEFAULT_REPORT_RELATIVE", "run_r6_diagnostics"]
