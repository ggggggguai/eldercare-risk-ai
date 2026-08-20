"""Strict cross-horizon stacking helpers for FORECAST-OPT-004E."""

from __future__ import annotations

import hashlib
import json
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


STRUCTURES = ("base_only", "probability_prior", "residual", "ordinal_plus_residual")


def participant_equal_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = (
        frame.groupby("global_participant_id", sort=False)["global_participant_id"]
        .transform("size")
        .to_numpy(dtype="float64")
    )
    return 1.0 / (float(frame["global_participant_id"].nunique()) * counts)


def participant_set_sha256(participants: Sequence[str]) -> str:
    payload = "\n".join(sorted(map(str, participants))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def strict_join(one_month: pd.DataFrame, two_month: pd.DataFrame) -> pd.DataFrame:
    """Left join the same future window and enforce the earlier 2m cutoff."""

    key = ["global_participant_id", "target_window_id", "evaluation_outer_fold"]
    if one_month.duplicated(key).any() or two_month.duplicated(key).any():
        raise ValueError("cross-horizon input keys must be unique")
    source_columns = key + [
        "inner_fold_id",
        "feature_month_slot",
        "label_month_slot",
        "outer_fold_id",
        "selected_fused_score",
        "score_pointwise",
        "score_ordinal",
        "rank_pointwise",
        "rank_ordinal",
        "rank_expert",
        "selected_weight_pointwise",
        "selected_weight_ordinal",
        "selected_weight_expert",
        "selected_weight_ranking",
    ]
    source = two_month[source_columns].rename(
        columns={
            "inner_fold_id": "source_inner_fold",
            "feature_month_slot": "source_feature_month_slot",
            "label_month_slot": "source_label_month_slot",
            "outer_fold_id": "source_row_outer_fold",
            "selected_fused_score": "source_fused_score",
            "score_pointwise": "source_pointwise_probability",
            "score_ordinal": "source_ordinal_probability",
            "rank_pointwise": "source_rank_pointwise",
            "rank_ordinal": "source_rank_ordinal",
            "rank_expert": "source_rank_expert",
        }
    )
    result = one_month.merge(source, on=key, how="left", validate="one_to_one")
    result["stack_available"] = result["source_inner_fold"].notna().astype("int8")
    available = result["stack_available"] == 1
    if not (
        result.loc[available, "source_feature_month_slot"]
        < result.loc[available, "feature_month_slot"]
    ).all():
        raise ValueError("2m source cutoff is not strictly earlier than 1m cutoff")
    if not (
        result.loc[available, "source_label_month_slot"]
        == result.loc[available, "label_month_slot"]
    ).all():
        raise ValueError("cross-horizon rows do not share a label window")
    if not (
        result.loc[available, "source_inner_fold"].astype(int)
        == result.loc[available, "inner_fold_id"].astype(int)
    ).all():
        raise ValueError("cross-horizon inner fold lineage mismatch")
    return result


def percentile_rank_by_fold(values: np.ndarray, inner_fold: np.ndarray) -> np.ndarray:
    result = np.empty(len(values), dtype="float64")
    for fold in sorted(np.unique(inner_fold)):
        mask = inner_fold == fold
        result[mask] = (
            pd.Series(values[mask]).rank(method="average", pct=True).to_numpy()
        )
    return result


def candidate_specs() -> list[dict[str, float | str]]:
    specs: list[dict[str, float | str]] = [
        {
            "structure": "base_only",
            "prior_weight": 0.0,
            "ordinal_weight": 0.0,
            "residual_weight": 0.0,
        }
    ]
    for prior_weight in (0.25, 0.5, 0.75):
        specs.append(
            {
                "structure": "probability_prior",
                "prior_weight": prior_weight,
                "ordinal_weight": 0.0,
                "residual_weight": 0.0,
            }
        )
    for residual_weight in (0.25, 0.5, 0.75, 1.0):
        specs.append(
            {
                "structure": "residual",
                "prior_weight": 0.0,
                "ordinal_weight": 0.0,
                "residual_weight": residual_weight,
            }
        )
    for ordinal_weight in (0.25, 0.5, 0.75):
        for residual_weight in (0.25, 0.5, 0.75):
            specs.append(
                {
                    "structure": "ordinal_plus_residual",
                    "prior_weight": 0.0,
                    "ordinal_weight": ordinal_weight,
                    "residual_weight": residual_weight,
                }
            )
    return specs


def score_candidate(frame: pd.DataFrame, spec: dict[str, float | str]) -> np.ndarray:
    base = frame["selected_fused_score"].to_numpy(dtype="float64")
    score = base.copy()
    available = frame["stack_available"].to_numpy(dtype="int8") == 1
    if not available.any() or spec["structure"] == "base_only":
        return score
    source_ordinal = frame["source_ordinal_probability"].to_numpy(dtype="float64")
    source_fused = frame["source_fused_score"].to_numpy(dtype="float64")
    source_pointwise_rank = frame["source_rank_pointwise"].to_numpy(dtype="float64")
    one_month_ordinal_rank = frame["rank_ordinal"].to_numpy(dtype="float64")
    residual = source_fused - source_pointwise_rank
    if spec["structure"] == "probability_prior":
        weight = float(spec["prior_weight"])
        score[available] = (1.0 - weight) * base[available] + weight * source_ordinal[
            available
        ]
    elif spec["structure"] == "residual":
        score[available] = (
            base[available] + float(spec["residual_weight"]) * residual[available]
        )
    elif spec["structure"] == "ordinal_plus_residual":
        ordinal_weight = float(spec["ordinal_weight"])
        score[available] = (
            (1.0 - ordinal_weight) * base[available]
            + ordinal_weight * one_month_ordinal_rank[available]
            + float(spec["residual_weight"]) * residual[available]
        )
    else:
        raise ValueError(f"unsupported structure: {spec['structure']}")
    # Preserve exact base fallback while normalizing only available rows by inner fold.
    normalized = score.copy()
    for fold in sorted(frame["inner_fold_id"].unique()):
        mask = available & (frame["inner_fold_id"].to_numpy() == fold)
        if mask.any():
            normalized[mask] = (
                pd.Series(score[mask]).rank(method="average", pct=True).to_numpy()
            )
    return normalized


def ranking_metrics(frame: pd.DataFrame, score: np.ndarray) -> dict[str, float]:
    y = frame["future_binary_target"].to_numpy(dtype="int8")
    weights = participant_equal_weights(frame)
    weights /= weights.sum()
    return {
        "auprc": float(average_precision_score(y, score, sample_weight=weights)),
        "auroc": float(roc_auc_score(y, score, sample_weight=weights)),
    }


def model_members_json(row: pd.Series) -> str:
    members = {
        "source_run_id": "MH-20260808-FOPT-018",
        "source_horizon": "forecast_2m",
        "weights": {
            "pointwise": float(row["selected_weight_pointwise_y"]),
            "ordinal": float(row["selected_weight_ordinal_y"]),
            "expert": float(row["selected_weight_expert_y"]),
            "ranking": float(row["selected_weight_ranking_y"]),
        },
    }
    return json.dumps(members, ensure_ascii=False, sort_keys=True)
