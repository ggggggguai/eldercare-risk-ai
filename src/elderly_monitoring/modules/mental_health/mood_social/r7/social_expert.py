"""Development, one-time confirmation and full-fit recipe for SocialContactExpert."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from elderly_monitoring.modules.mental_health.mood_social.r4.calibration import fit_calibration
from elderly_monitoring.modules.mental_health.mood_social.r4.selection import CandidateSpec, fit_model, predict_model
from elderly_monitoring.modules.mental_health.mood_social.r7.contract import (
    DEFAULT_DATA_RELATIVE,
    DEFAULT_REPORT_RELATIVE,
    R7_DEPREST_NESTED_SEED,
    sha256_file,
    write_json,
)
from elderly_monitoring.modules.mental_health.mood_social.r7.deprest import SOCIAL_PRODUCTION_FEATURES, assert_social_feature_contract
from elderly_monitoring.modules.mental_health.mood_social.r7.modeling import (
    binary_metrics,
    fit_outer_head,
    inner_fold_series,
    participant_equal_weights,
    project_heads,
    select_specificity_threshold,
)


SOCIAL_REPORT_RELATIVE = Path("reports/mental_health/mood_social/v3.3.3-r7/OPT-V333-R7-003-social")


def _load_partition(root: Path, name: str) -> pd.DataFrame:
    path = root / DEFAULT_DATA_RELATIVE / "social" / f"{name}.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def _outer_raw_oof(frame: pd.DataFrame, features: tuple[str, ...], spec: CandidateSpec, target: str) -> np.ndarray:
    probability = pd.Series(np.nan, index=frame.index, dtype=float)
    for fold in range(5):
        train = frame.loc[frame["outer_fold"].ne(fold)].copy()
        validation = frame.loc[frame["outer_fold"].eq(fold)].copy()
        train["binary_target"] = train[target].astype(int)
        model = fit_model(train, features, spec, participant_equal=True)
        probability.loc[validation.index] = predict_model(model, validation, features)
    if probability.isna().any():
        raise ValueError("social selected-spec OOF is incomplete")
    return probability.loc[frame.index].to_numpy(float)


def _spec_from_votes(audits: dict[str, Any], head: str) -> CandidateSpec:
    rows = [audits[head][str(fold)]["selected"] for fold in range(5)]
    winner = Counter(row["candidate_id"] for row in rows).most_common(1)[0][0]
    selected = next(row for row in rows if row["candidate_id"] == winner)
    return CandidateSpec(
        candidate_id=str(selected["candidate_id"]),
        family=str(selected["family"]),  # type: ignore[arg-type]
        params=dict(selected["params"]),
        seed=R7_DEPREST_NESTED_SEED,
    )


def _fold_superiority(frame: pd.DataFrame, head: str) -> int:
    count = 0
    for _fold, part in frame.groupby("outer_fold", sort=True):
        metric = binary_metrics(part[f"phq9_{head}_target"], part[f"probability_{head}"])
        count += int(metric["auprc"] > metric["prevalence"])
    return count


def run_social_expert(*, repository_root: Path, overwrite: bool = False) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    report = root / SOCIAL_REPORT_RELATIVE
    features = assert_social_feature_contract(SOCIAL_PRODUCTION_FEATURES)
    development = _load_partition(root, "development")
    confirmation_path = root / DEFAULT_DATA_RELATIVE / "social/model_unseen_confirmation.parquet"
    seal_path = root / DEFAULT_REPORT_RELATIVE / "social_confirmation_seal.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if seal.get("model_metrics_opened") is not False or seal.get("candidate_recipe_locked") is not False:
        raise PermissionError("social confirmation is already opened or recipe already locked")

    predictions = development[["participant_id", "age_group", "outer_fold", "phq9_ge5_target", "phq9_ge10_target"]].copy()
    audits: dict[str, Any] = {}
    for head in ("ge5", "ge10"):
        candidate = pd.Series(np.nan, index=development.index, dtype=float)
        baseline = pd.Series(np.nan, index=development.index, dtype=float)
        audits[head] = {}
        for outer_fold in range(5):
            train = development.loc[development["outer_fold"].ne(outer_fold)].copy()
            test = development.loc[development["outer_fold"].eq(outer_fold)].copy()
            train["binary_target"] = train[f"phq9_{head}_target"].astype(int)
            test["binary_target"] = test[f"phq9_{head}_target"].astype(int)
            folds = inner_fold_series(development, outer_fold).loc[train.index].astype(int)
            selected, base, audit = fit_outer_head(
                train, test, features, folds, seed=R7_DEPREST_NESTED_SEED + outer_fold
            )
            candidate.loc[test.index] = selected
            baseline.loc[test.index] = base
            audits[head][str(outer_fold)] = audit
        predictions[f"probability_{head}"] = candidate.loc[development.index].to_numpy(float)
        predictions[f"baseline_probability_{head}"] = baseline.loc[development.index].to_numpy(float)
    predictions["probability_ge5"], predictions["probability_ge10"] = project_heads(
        predictions["probability_ge5"], predictions["probability_ge10"]
    )
    selected_specs = {head: _spec_from_votes(audits, head) for head in ("ge5", "ge10")}
    development_metrics = {
        head: {
            "candidate": binary_metrics(predictions[f"phq9_{head}_target"], predictions[f"probability_{head}"]),
            "baseline": binary_metrics(predictions[f"phq9_{head}_target"], predictions[f"baseline_probability_{head}"]),
            "folds_above_prevalence": _fold_superiority(predictions, head),
        }
        for head in ("ge5", "ge10")
    }
    for head in ("ge5", "ge10"):
        development_metrics[head]["delta_auprc"] = (
            development_metrics[head]["candidate"]["auprc"] - development_metrics[head]["baseline"]["auprc"]
        )
    recipe = {
        "protocol": "mood-social-v3.3.3-r7",
        "domain": "social",
        "production_track": "calls-only direction-independent D-14..D-1 and D-28..D-1",
        "features": list(features),
        "features_sha256": __import__("hashlib").sha256("\n".join(features).encode("utf-8")).hexdigest(),
        "selected_specs": {head: {"candidate_id": spec.candidate_id, "family": spec.family, "params": spec.params, "seed": spec.seed} for head, spec in selected_specs.items()},
        "development_metrics": development_metrics,
        "development_sha256": sha256_file(root / DEFAULT_DATA_RELATIVE / "social/development.parquet"),
        "confirmation_sha256": sha256_file(confirmation_path),
        "confirmation_metrics_opened": False,
        "no_recall_after_confirmation": True,
    }
    recipe_path = report / "candidate_recipe_lock.json"
    write_json(recipe_path, recipe, overwrite=overwrite)
    updated_seal = dict(seal)
    updated_seal["candidate_recipe_locked"] = True
    updated_seal["candidate_recipe_sha256"] = sha256_file(recipe_path)
    write_json(seal_path, updated_seal, overwrite=True)

    # Only after the complete recipe is written and hashed may labels/predictions
    # on the 74-person model-unseen partition be evaluated.
    confirmation = pd.read_parquet(confirmation_path)
    output = confirmation[["participant_id", "age_group", "phq9_ge5_target", "phq9_ge10_target"]].copy()
    calibration_audit: dict[str, Any] = {}
    for head in ("ge5", "ge10"):
        target = f"phq9_{head}_target"
        work = development.copy()
        work["binary_target"] = work[target].astype(int)
        raw_oof = _outer_raw_oof(work, features, selected_specs[head], target)
        calibrator = fit_calibration(work, raw_oof, "platt")
        model = fit_model(work, features, selected_specs[head], participant_equal=True)
        raw_confirmation = predict_model(model, confirmation, features)
        output[f"probability_{head}"] = calibrator.predict(raw_confirmation)
        calibration_audit[head] = {
            "method": calibrator.method,
            "threshold_spec080": select_specificity_threshold(work["binary_target"], calibrator.predict(raw_oof), 0.80),
        }
    output["probability_ge5"], output["probability_ge10"] = project_heads(output["probability_ge5"], output["probability_ge10"])
    confirmation_metrics = {
        head: binary_metrics(output[f"phq9_{head}_target"], output[f"probability_{head}"])
        for head in ("ge5", "ge10")
    }
    main = confirmation_metrics["ge10"]
    gate = {
        "ap_gain_ge_0_02_or_normalized_ap_ge_0_03": bool(main["ap_gain"] >= 0.02 or main["normalized_ap"] >= 0.03),
        "auroc_ge_0_58": bool(main["auroc"] >= 0.58),
        "development_at_least_3_of_5_folds_above_prevalence": bool(development_metrics["ge10"]["folds_above_prevalence"] >= 3),
        "dual_head_monotonicity": bool((output["probability_ge10"] <= output["probability_ge5"] + 1e-12).all()),
    }
    gate["pass"] = all(gate.values())
    result = {
        "status": "offline_validated_integration_ready" if gate["pass"] else "anomaly_only_no_phq_probability",
        "evidence": "DepreST-CAT model-unseen confirmation; non-elderly smartphone-to-S10 proxy",
        "development": development_metrics,
        "confirmation": {"rows": int(len(output)), "metrics": confirmation_metrics},
        "calibration": calibration_audit,
        "gate": gate,
        "limitations": ["56+ subgroup has only eight people in the full dataset", "COVID-era smartphone proxy", "no real S10 device parity"],
    }
    report.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(report / "development_outer_oof.parquet", index=False)
    output.to_parquet(report / "confirmation_predictions.parquet", index=False)
    write_json(report / "development_selection_audit.json", audits, overwrite=overwrite)
    write_json(report / "confirmation_report.json", result, overwrite=overwrite)
    write_json(report / "OPENED.json", {"opened_once": True, "recipe_sha256": sha256_file(recipe_path), "confirmation_sha256": sha256_file(confirmation_path), "reopen_allowed": False}, overwrite=overwrite)
    updated_seal["model_metrics_opened"] = True
    updated_seal["opened_record_sha256"] = sha256_file(report / "OPENED.json")
    write_json(seal_path, updated_seal, overwrite=True)
    return result


__all__ = ["SOCIAL_REPORT_RELATIVE", "run_social_expert"]
