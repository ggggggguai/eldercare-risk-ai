"""Participant-isolated PSYCHE-D evaluation of the R7 history expert."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from elderly_monitoring.modules.mental_health.mood_social.r4.data import load_r4_development_frame
from elderly_monitoring.modules.mental_health.mood_social.r4.selection import CandidateSpec, fit_model, predict_model
from elderly_monitoring.modules.mental_health.mood_social.r7.contract import DEFAULT_DATA_RELATIVE, R7_EXISTING_EXPERT_SEEDS, repository_root
from elderly_monitoring.modules.mental_health.mood_social.r7.modeling import binary_metrics, participant_equal_weights, project_heads, write_json
from elderly_monitoring.modules.mental_health.mood_social.r7.phq_history_expert import HISTORY_FEATURES, build_psyche_history_frame


REPORT_RELATIVE = Path("reports/mental_health/mood_social/v3.3.3-r7/OPT-V333-R7-004-history")


def run_history_evaluation(*, repository_root_value: Path | None = None, overwrite: bool = False) -> dict[str, Any]:
    root = repository_root(repository_root_value)
    base = load_r4_development_frame(repository_root=root)
    history = build_psyche_history_frame(base)
    seed_predictions: list[pd.DataFrame] = []
    spec = CandidateSpec("history_logistic_c01", "elasticnet", {"C": 0.1, "l1_ratio": 0.2}, R7_EXISTING_EXPERT_SEEDS[0])
    for seed in R7_EXISTING_EXPERT_SEEDS:
        split = pd.read_parquet(root / DEFAULT_DATA_RELATIVE / "protocol/existing_splits" / f"seed-{seed}.parquet")
        frame = history.merge(split[["r4_row_id", "outer_fold"]], on="r4_row_id", how="left", validate="one_to_one")
        prediction = frame[["r4_row_id", "global_participant_id", "nominal_month", "outer_fold", "phq9_ge5_target", "phq9_ge10_target", "history.age_days"]].copy()
        for head in ("ge5", "ge10"):
            probability = pd.Series(np.nan, index=frame.index, dtype=float)
            baseline = pd.Series(np.nan, index=frame.index, dtype=float)
            for fold in range(5):
                train = frame.loc[frame["outer_fold"].ne(fold)].copy()
                test = frame.loc[frame["outer_fold"].eq(fold)].copy()
                train["binary_target"] = train[f"phq9_{head}_target"].astype(int)
                current = CandidateSpec(spec.candidate_id, spec.family, spec.params, seed + fold)
                model = fit_model(train, HISTORY_FEATURES, current, participant_equal=True)
                probability.loc[test.index] = predict_model(model, test, HISTORY_FEATURES)
                baseline.loc[test.index] = test[f"history.last_{head}"].to_numpy(float)
            prediction[f"probability_{head}"] = probability
            prediction[f"baseline_probability_{head}"] = np.clip(baseline, 1e-6, 1 - 1e-6)
        prediction["probability_ge5"], prediction["probability_ge10"] = project_heads(prediction["probability_ge5"], prediction["probability_ge10"])
        seed_predictions.append(prediction)
        output = root / REPORT_RELATIVE
        output.mkdir(parents=True, exist_ok=True)
        prediction.to_parquet(output / f"outer_oof_seed-{seed}.parquet", index=False)
    mean = seed_predictions[0].copy()
    for head in ("ge5", "ge10"):
        mean[f"probability_{head}"] = np.mean([frame[f"probability_{head}"].to_numpy(float) for frame in seed_predictions], axis=0)
        mean[f"baseline_probability_{head}"] = np.mean([frame[f"baseline_probability_{head}"].to_numpy(float) for frame in seed_predictions], axis=0)
    metrics: dict[str, Any] = {}
    for head in ("ge5", "ge10"):
        weight = participant_equal_weights(mean)
        candidate = binary_metrics(mean[f"phq9_{head}_target"], mean[f"probability_{head}"], weight=weight)
        baseline = binary_metrics(mean[f"phq9_{head}_target"], mean[f"baseline_probability_{head}"], weight=weight)
        metrics[head] = {"candidate": candidate, "last_score_baseline": baseline, "delta_auprc": candidate["auprc"] - baseline["auprc"]}
    result = {
        "status": "pass",
        "rows": int(len(mean)),
        "participants": int(mean["global_participant_id"].nunique()),
        "features": list(HISTORY_FEATURES),
        "metrics": metrics,
        "time_policy": "strictly prior PSYCHE nominal wave; first wave excluded",
        "limitations": ["PSYCHE nominal months, not natural assessment timestamps", "same dataset reused adaptive evidence"],
    }
    mean.to_parquet(root / REPORT_RELATIVE / "three_seed_mean_oof.parquet", index=False)
    write_json(root / REPORT_RELATIVE / "metrics.json", result, overwrite=overwrite)
    return result


__all__ = ["REPORT_RELATIVE", "run_history_evaluation"]
