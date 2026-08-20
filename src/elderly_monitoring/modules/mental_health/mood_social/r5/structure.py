"""Inner-only ordered dual-head structure comparison for OPT-V333-R5-003."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from elderly_monitoring.modules.mental_health.mood_social.r4.baseline import (
    ap_context_metrics,
)
from elderly_monitoring.modules.mental_health.mood_social.r4.selection import (
    participant_equal_weights,
)
from elderly_monitoring.modules.mental_health.mood_social.r5.contract import (
    R5_DEVELOPMENT_SEEDS,
    R5_PROTOCOL_VERSION,
    R5_PSYCHE_RESEARCH_SENSOR_FEATURES,
    sha256_file,
)
from elderly_monitoring.modules.mental_health.mood_social.r5.data import (
    load_r5_development_frame,
    r5_inner_fold_series,
    select_r5_label_task,
)
from elderly_monitoring.modules.mental_health.mood_social.r5.features import (
    add_psyche_personal_change_features,
)
from elderly_monitoring.modules.mental_health.mood_social.r5.objectives import (
    conditional_ordered_heads,
    ordinal_heads,
    phq_severity_class,
    project_independent_heads,
)
from elderly_monitoring.modules.mental_health.mood_social.r5.selection import (
    R5CandidateSpec,
    build_r5_pipeline,
    fit_r5_model,
    predict_r5_model,
)
from elderly_monitoring.modules.mental_health.mood_social.r5.weights import (
    R5_WEIGHT_SCHEMES,
    WeightScheme,
    r5_training_weights,
)


DEFAULT_REPORT_RELATIVE = Path(
    "reports/mental_health/mood_social/v3.3.3-r5/OPT-V333-R5-003-structure"
)
RETAINED_FEATURE_RELATIVE = Path(
    "reports/mental_health/mood_social/v3.3.3-r5/"
    "OPT-V333-R5-002-feature-ablation/retained_feature_set.json"
)
SCREENING_SEED = R5_DEVELOPMENT_SEEDS[0]
SCREENING_OUTER_FOLD = 0
STRUCTURES = ("independent_projected", "conditional_ordered", "ordinal_five_class")
FIXED_SPEC = R5CandidateSpec(
    candidate_id="r5_structure_lgbm_fixed",
    family="lightgbm",
    params={
        "n_estimators": 300,
        "learning_rate": 0.03,
        "num_leaves": 15,
        "min_child_samples": 40,
        "reg_lambda": 7.0,
        "reg_alpha": 0.7,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
    },
    seed=SCREENING_SEED,
)


def _write_json(path: Path, value: Any, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite r5 structure artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _retained_features(root: Path) -> tuple[str, ...]:
    path = root / RETAINED_FEATURE_RELATIVE
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "pass" or payload.get("batch2_retained") is not False:
        raise ValueError("r5 retained feature decision is not sealed as expected")
    features = tuple(str(name) for name in payload["retained_features"])
    if len(features) != int(payload["retained_feature_count"]):
        raise ValueError("r5 retained feature count drifted")
    return features


def _binary_oof(
    frame: pd.DataFrame,
    features: tuple[str, ...],
    inner_fold: pd.Series,
    scheme: WeightScheme,
) -> np.ndarray:
    probability = pd.Series(np.nan, index=frame.index, dtype=float)
    for fold in sorted(inner_fold.astype(int).unique()):
        train = frame.loc[inner_fold.ne(fold)]
        validation = frame.loc[inner_fold.eq(fold)]
        model = fit_r5_model(
            train,
            features,
            FIXED_SPEC,
            sample_weight=r5_training_weights(train, scheme),
        )
        probability.loc[validation.index] = predict_r5_model(model, validation, features)
    if probability.isna().any():
        raise ValueError("r5 structure binary OOF is incomplete")
    return probability.loc[frame.index].to_numpy(float)


def _independent_oof(
    frame: pd.DataFrame,
    features: tuple[str, ...],
    inner_fold: pd.Series,
    scheme: WeightScheme,
) -> tuple[np.ndarray, np.ndarray]:
    ge5 = select_r5_label_task(frame, "phq_ge5_current")
    ge10 = select_r5_label_task(frame, "phq_ge10_current")
    p5 = _binary_oof(ge5, features, inner_fold, scheme)
    p10 = _binary_oof(ge10, features, inner_fold, scheme)
    return project_independent_heads(p5, p10)


def _conditional_oof(
    frame: pd.DataFrame,
    features: tuple[str, ...],
    inner_fold: pd.Series,
    scheme: WeightScheme,
) -> tuple[np.ndarray, np.ndarray]:
    ge5 = select_r5_label_task(frame, "phq_ge5_current")
    p5 = _binary_oof(ge5, features, inner_fold, scheme)
    conditional_probability = pd.Series(np.nan, index=frame.index, dtype=float)
    for fold in sorted(inner_fold.astype(int).unique()):
        train_mask = inner_fold.ne(fold) & frame["phq9_ge5_r3_target"].eq(1)
        validation_mask = inner_fold.eq(fold)
        train = frame.loc[train_mask].copy()
        validation = frame.loc[validation_mask].copy()
        train["binary_target"] = train["phq9_ge10_r3_target"].astype(int)
        if train["binary_target"].nunique() != 2:
            raise ValueError("r5 conditional ge10|ge5 train fold lacks both classes")
        model = fit_r5_model(
            train,
            features,
            FIXED_SPEC,
            sample_weight=r5_training_weights(train, scheme),
        )
        conditional_probability.loc[validation.index] = predict_r5_model(
            model, validation, features
        )
    if conditional_probability.isna().any():
        raise ValueError("r5 conditional OOF is incomplete")
    return conditional_ordered_heads(
        p5, conditional_probability.loc[frame.index].to_numpy(float)
    )


def _ordinal_oof(
    frame: pd.DataFrame,
    features: tuple[str, ...],
    inner_fold: pd.Series,
    scheme: WeightScheme,
) -> tuple[np.ndarray, np.ndarray]:
    class_probability = np.full((len(frame), 5), np.nan, dtype=float)
    score = frame["phq9_score_r3_target"].to_numpy(float)
    severity = phq_severity_class(score)
    for fold in sorted(inner_fold.astype(int).unique()):
        train_mask = inner_fold.ne(fold).to_numpy(bool)
        validation_mask = inner_fold.eq(fold).to_numpy(bool)
        train = frame.loc[train_mask]
        validation = frame.loc[validation_mask]
        model = build_r5_pipeline(train, features, FIXED_SPEC)
        model.set_params(
            model=LGBMClassifier(
                objective="multiclass",
                num_class=5,
                n_estimators=300,
                learning_rate=0.03,
                num_leaves=15,
                min_child_samples=40,
                reg_lambda=7.0,
                reg_alpha=0.7,
                subsample=0.85,
                colsample_bytree=0.85,
                verbosity=-1,
                n_jobs=1,
                random_state=SCREENING_SEED,
            )
        )
        model.fit(
            train[list(features)],
            severity[train_mask],
            model__sample_weight=r5_training_weights(train, scheme),
        )
        predicted = np.asarray(model.predict_proba(validation[list(features)]), float)
        classes = np.asarray(model.named_steps["model"].classes_, int)
        if not set(classes).issubset(set(range(5))):
            raise ValueError("r5 ordinal model produced invalid severity classes")
        aligned = np.zeros((len(validation), 5), dtype=float)
        aligned[:, classes] = predicted
        class_probability[validation_mask] = aligned
    if not np.isfinite(class_probability).all():
        raise ValueError("r5 ordinal OOF is incomplete")
    return ordinal_heads(class_probability)


def _head_metric(frame: pd.DataFrame, target_column: str, probability: np.ndarray) -> dict[str, float]:
    target = frame[target_column].to_numpy(int)
    natural = ap_context_metrics(target, probability)
    participant = ap_context_metrics(
        target, probability, sample_weight=participant_equal_weights(frame)
    )
    return {
        **natural,
        "participant_auprc": participant["auprc"],
        "selection_score": 0.65 * natural["auprc"] + 0.35 * participant["auprc"],
    }


def _select(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reference = next(
        row
        for row in rows
        if row["structure"] == "independent_projected"
        and row["weight_scheme"] == "participant_equal"
    )
    eligible: list[dict[str, Any]] = []
    for row in rows:
        deltas = {
            head: row["metrics"][head]["auprc"] - reference["metrics"][head]["auprc"]
            for head in ("ge5", "ge10")
        }
        row["delta_vs_reference"] = deltas
        row["screening_gate"] = bool(
            row is reference
            or (max(deltas.values()) >= 0.001 and min(deltas.values()) >= -0.002)
        )
        row["joint_selection_score"] = float(
            np.mean([row["metrics"][head]["selection_score"] for head in ("ge5", "ge10")])
        )
        if row["screening_gate"]:
            eligible.append(row)
    selected = max(
        eligible,
        key=lambda row: (
            row["joint_selection_score"],
            min(row["metrics"][head]["normalized_ap"] for head in ("ge5", "ge10")),
            -STRUCTURES.index(row["structure"]),
            -R5_WEIGHT_SCHEMES.index(row["weight_scheme"]),
        ),
    )
    return {
        "structure": selected["structure"],
        "weight_scheme": selected["weight_scheme"],
        "joint_selection_score": selected["joint_selection_score"],
        "delta_vs_reference": selected["delta_vs_reference"],
        "monotonic_violation_count": selected["monotonic_violation_count"],
        "selection_rule": "highest joint dual-head score among screening-gate candidates",
    }


def run_structure_comparison(
    *, repository_root: Path, report_directory: Path | None = None, overwrite: bool = False
) -> dict[str, Any]:
    """Compare all frozen structures and weights without reading outer-test labels."""

    root = Path(repository_root).resolve()
    output = Path(report_directory) if report_directory else root / DEFAULT_REPORT_RELATIVE
    if not output.is_absolute():
        output = root / output
    base = load_r5_development_frame(repository_root=root, seed=SCREENING_SEED)
    psyche = base.loc[
        base["dataset_id"].eq("psyche_d")
        & base["outer_fold"].ne(SCREENING_OUTER_FOLD)
    ].copy()
    enriched, _batch1, _batch2 = add_psyche_personal_change_features(
        psyche, include_batch2=False
    )
    features = _retained_features(root)
    if not set(features).issubset(enriched.columns):
        raise ValueError("r5 retained structure features are not reproducible")
    inner = r5_inner_fold_series(enriched, SCREENING_OUTER_FOLD).astype(int)
    rows: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    failures: list[dict[str, Any]] = []
    functions = {
        "independent_projected": _independent_oof,
        "conditional_ordered": _conditional_oof,
        "ordinal_five_class": _ordinal_oof,
    }
    for structure in STRUCTURES:
        for scheme in R5_WEIGHT_SCHEMES:
            try:
                p5, p10 = functions[structure](enriched, features, inner, scheme)
            except Exception as error:
                failures.append(
                    {
                        "structure": structure,
                        "weight_scheme": scheme,
                        "error_type": type(error).__name__,
                        "message": str(error),
                        "predictions_opened": False,
                    }
                )
                continue
            violations = int(np.sum(p10 > p5 + 1.0e-12))
            if violations:
                raise ValueError(f"r5 structure produced {violations} monotonic violations")
            metrics = {
                "ge5": _head_metric(enriched, "phq9_ge5_r3_target", p5),
                "ge10": _head_metric(enriched, "phq9_ge10_r3_target", p10),
            }
            rows.append(
                {
                    "structure": structure,
                    "weight_scheme": scheme,
                    "feature_count": len(features),
                    "metrics": metrics,
                    "monotonic_violation_count": violations,
                }
            )
            part = enriched[
                ["r5_row_id", "global_participant_id", "nominal_month"]
            ].copy()
            part["inner_fold"] = inner.to_numpy(int)
            part["structure"] = structure
            part["weight_scheme"] = scheme
            part["target_ge5"] = enriched["phq9_ge5_r3_target"].to_numpy(int)
            part["target_ge10"] = enriched["phq9_ge10_r3_target"].to_numpy(int)
            part["probability_ge5"] = p5
            part["probability_ge10"] = p10
            predictions.append(part)
    expected = len(STRUCTURES) * len(R5_WEIGHT_SCHEMES)
    if len(rows) != expected:
        _write_json(output / "failed_runs.json", failures, overwrite=overwrite)
        raise RuntimeError(f"r5 structure comparison completed {len(rows)}/{expected} runs")
    selected = _select(rows)
    prediction_path = output / "structure_inner_oof.parquet"
    comparison_path = output / "structure_comparison.json"
    selection_path = output / "selected_structure.json"
    manifest_path = output / "artifact_manifest.json"
    if prediction_path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite r5 structure artifact: {prediction_path}")
    output.mkdir(parents=True, exist_ok=True)
    pd.concat(predictions, ignore_index=True).to_parquet(prediction_path, index=False)
    _write_json(comparison_path, {"runs": rows}, overwrite=overwrite)
    _write_json(selection_path, selected, overwrite=overwrite)
    _write_json(output / "failed_runs.json", failures, overwrite=overwrite)
    manifest = {
        "protocol_version": R5_PROTOCOL_VERSION,
        "status": "pass",
        "evidence_level": "adaptive-development inner-only",
        "seed": SCREENING_SEED,
        "screening_outer_fold_excluded": SCREENING_OUTER_FOLD,
        "outer_test_labels_read": False,
        "confirmation_opened": False,
        "feature_count": len(features),
        "structure_count": len(STRUCTURES),
        "weight_scheme_count": len(R5_WEIGHT_SCHEMES),
        "selected": selected,
        "artifacts": {
            path.name: {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
            }
            for path in (prediction_path, comparison_path, selection_path)
        },
    }
    _write_json(manifest_path, manifest, overwrite=overwrite)
    return manifest


__all__ = ["run_structure_comparison"]
