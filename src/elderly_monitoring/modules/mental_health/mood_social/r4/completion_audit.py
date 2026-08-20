"""Locked-posthoc completion audits for R4-005/R4-006.

These experiments are deliberately diagnostic-only.  The formal three-seed
evaluation was already opened before this file existed, so none of these
results may be used to modify or re-select the sealed r4 candidate.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from elderly_monitoring.modules.mental_health.mood_social.r4.baseline import ap_context_metrics
from elderly_monitoring.modules.mental_health.mood_social.r4.data import (
    load_r4_development_frame,
    select_label_task,
)
from elderly_monitoring.modules.mental_health.mood_social.r4.evaluation import ap_at_prevalence
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
    "OPT-V333-R4-005-006-postlocked-audit"
)
PROFILE_SPEC = CandidateSpec(
    "r4_domain_audit_elasticnet",
    "elasticnet",
    {"C": 0.1, "l1_ratio": 0.2},
    20260812,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dump(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def add_causal_subject_bag_features(
    frame: pd.DataFrame, source_features: tuple[str, ...]
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Add permutation-invariant statistics over each D-day causal prefix."""

    result = frame.copy()
    ordered = result.sort_values(
        ["global_participant_id", "nominal_month"], kind="stable"
    )
    generated: list[str] = []
    group = ordered.groupby("global_participant_id", sort=False)
    for feature in source_features:
        numeric = pd.to_numeric(ordered[feature], errors="coerce")
        by_person = numeric.groupby(ordered["global_participant_id"], sort=False)
        stats = {
            "count": numeric.notna().groupby(ordered["global_participant_id"], sort=False).cumsum(),
            "mean": by_person.expanding().mean().reset_index(level=0, drop=True),
            "std": by_person.expanding().std(ddof=0).reset_index(level=0, drop=True),
            "min": by_person.expanding().min().reset_index(level=0, drop=True),
            "max": by_person.expanding().max().reset_index(level=0, drop=True),
        }
        for suffix, values in stats.items():
            name = f"psyche_bag.{feature}_{suffix}"
            result.loc[ordered.index, name] = values.reindex(ordered.index).to_numpy(float)
            generated.append(name)
    return result, tuple(generated)


def source_balanced_weights(frame: pd.DataFrame, alpha: float = 0.25) -> np.ndarray:
    participant = participant_equal_weights(frame)
    counts = frame.groupby("dataset_id")["dataset_id"].transform("size").to_numpy(float)
    source = np.power(len(frame) / counts, alpha)
    weight = participant * source
    return weight * len(weight) / weight.sum()


def _source_metrics(frame: pd.DataFrame, probability: np.ndarray) -> dict[str, Any]:
    scored = frame.assign(probability=np.asarray(probability, dtype=float))
    sources: dict[str, Any] = {}
    for source, part in scored.groupby("dataset_id", sort=True):
        natural = ap_context_metrics(part["binary_target"], part["probability"])
        natural["ap_at_10pct"] = ap_at_prevalence(
            part["binary_target"].to_numpy(int), part["probability"].to_numpy(float)
        )
        sources[str(source)] = natural
    return {
        "sources": sources,
        "macro_ap_at_10pct": float(np.mean([row["ap_at_10pct"] for row in sources.values()])),
        "worst_source_auprc": float(min(row["auprc"] for row in sources.values())),
    }


def _outer_predictions(
    frame: pd.DataFrame,
    weight_builder: Callable[[pd.DataFrame], np.ndarray],
) -> np.ndarray:
    prediction = pd.Series(np.nan, index=frame.index, dtype=float)
    for fold in sorted(frame["outer_fold"].astype(int).unique()):
        train = frame.loc[frame["outer_fold"].ne(fold)]
        test = frame.loc[frame["outer_fold"].eq(fold)]
        model = fit_model(
            train,
            DEPLOYABLE_PROFILE_FEATURES,
            PROFILE_SPEC,
            participant_equal=False,
            sample_weight=weight_builder(train),
        )
        prediction.loc[test.index] = predict_model(model, test, DEPLOYABLE_PROFILE_FEATURES)
    if prediction.isna().any():
        raise ValueError("domain robustness OOF is incomplete")
    return prediction.loc[frame.index].to_numpy(float)


def _groupdro_predictions(frame: pd.DataFrame) -> np.ndarray:
    """Two-step train-only GroupDRO approximation on the shared profile route."""

    prediction = pd.Series(np.nan, index=frame.index, dtype=float)
    for fold in sorted(frame["outer_fold"].astype(int).unique()):
        train = frame.loc[frame["outer_fold"].ne(fold)]
        test = frame.loc[frame["outer_fold"].eq(fold)]
        initial_weight = participant_equal_weights(train)
        initial = fit_model(
            train,
            DEPLOYABLE_PROFILE_FEATURES,
            PROFILE_SPEC,
            participant_equal=False,
            sample_weight=initial_weight,
        )
        initial_probability = predict_model(initial, train, DEPLOYABLE_PROFILE_FEATURES)
        loss = -(train["binary_target"].to_numpy(int) * np.log(initial_probability) +
                 (1 - train["binary_target"].to_numpy(int)) * np.log(1 - initial_probability))
        loss_frame = train[["dataset_id"]].copy()
        loss_frame["loss"] = loss
        source_loss = loss_frame.groupby("dataset_id")["loss"].mean()
        centered = source_loss - source_loss.mean()
        multiplier = np.exp(0.5 * centered).clip(0.75, 1.5)
        final_weight = initial_weight * train["dataset_id"].map(multiplier).to_numpy(float)
        final = fit_model(
            train,
            DEPLOYABLE_PROFILE_FEATURES,
            PROFILE_SPEC,
            participant_equal=False,
            sample_weight=final_weight,
        )
        prediction.loc[test.index] = predict_model(final, test, DEPLOYABLE_PROFILE_FEATURES)
    if prediction.isna().any():
        raise ValueError("GroupDRO diagnostic OOF is incomplete")
    return prediction.loc[frame.index].to_numpy(float)


def run_completion_audit(repository_root: Path, output: Path | None = None) -> dict[str, Any]:
    root = repository_root.resolve()
    destination = (output or root / DEFAULT_OUTPUT).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise FileExistsError(f"refusing to overwrite locked-posthoc audit: {destination}")

    base = load_r4_development_frame(repository_root=root)
    psyche = base.loc[base["dataset_id"].eq("psyche_d")].copy()
    psyche_bag, bag_features = add_causal_subject_bag_features(
        psyche,
        (
            "source__steps_awake_mean",
            "source__steps_mvpa_sum_recent",
            "source__steps_lpa_sum_recent",
            "source__sleep_asleep_mean_recent",
            "source__sleep_in_bed_mean_recent",
            "source__sleep_ratio_asleep_in_bed_mean_recent",
        ),
    )
    mixed = {}
    for task, target_column in (
        ("phq_ge5_current", "phq9_ge5_r3_target"),
        ("phq_ge10_current", "phq9_ge10_r3_target"),
    ):
        by_person = psyche.groupby("global_participant_id")[target_column].agg(["size", "nunique"])
        mixed[task] = {
            "participants": int(len(by_person)),
            "multirow_participants": int(by_person["size"].gt(1).sum()),
            "participants_with_valid_within_person_ranking_pairs": int(by_person["nunique"].gt(1).sum()),
        }

    domain_results: dict[str, Any] = {}
    for task in ("phq_ge5_current", "phq_ge10_current"):
        task_frame = select_label_task(base, task)
        profile = task_frame.loc[task_frame["route_pattern"].eq("001")].copy()
        methods = {
            "natural_erm": lambda part: np.ones(len(part)),
            "participant_equal_erm": participant_equal_weights,
            "mild_source_balanced_erm_alpha_0_25": source_balanced_weights,
        }
        task_result: dict[str, Any] = {}
        for method, builder in methods.items():
            probability = _outer_predictions(profile, builder)
            task_result[method] = {
                "overall": ap_context_metrics(profile["binary_target"], probability),
                **_source_metrics(profile, probability),
            }
        groupdro = _groupdro_predictions(profile)
        task_result["groupdro_two_step_train_only"] = {
            "overall": ap_context_metrics(profile["binary_target"], groupdro),
            **_source_metrics(profile, groupdro),
        }
        domain_results[task] = task_result

    payload = {
        "protocol_version": "mood-social-v3.3.3-r4",
        "evidence_level": "postlocked_diagnostic_only_not_model_selection",
        "formal_locked_candidate_changed": False,
        "subject_bag": {
            "status": "implemented_as_causal_permutation_invariant_prefix_statistics",
            "feature_count": len(bag_features),
            "row_count": int(len(psyche_bag)),
            "maximum_windows": int(psyche.groupby("global_participant_id").size().max()),
            "future_rows_used": False,
        },
        "ranking_pairwise": {
            "status": "audited_not_promoted",
            "reason": "only within-participant ordering is causally defensible; most participants have no mixed-label pair and the formal candidate was already locked",
            "counts": mixed,
        },
        "deep_temporal": {
            "status": "skipped_by_frozen_condition",
            "reason": "available rows are month-level aggregates at nominal months 3/6/9/12, not verified 14/28-day raw ordered sequences",
            "verified_raw_sequence_coverage": 0.0,
            "required_coverage": 0.6,
        },
        "dual_head": {
            "independent_heads": "completed_in_formal_locked_evaluation",
            "ordered_constraint": "midpoint monotonic projection completed with zero violations",
            "shared_neural_head": "not_promoted_after_lock; the explicit monotone postprocessing satisfies the frozen alternative",
        },
        "domain_robustness": {
            "scope": "shared profile signature only; source ID used for train weights/diagnostics and never enters inference vector",
            "results": domain_results,
            "source_wise_robust_scaling": {
                "status": "structurally_rejected",
                "reason": "requires source identity or source-specific transform at inference and would violate the deployment feature contract",
            },
            "coral": {
                "status": "structurally_rejected",
                "reason": "source identity is nearly collinear with missingness/route and target-domain alignment is unavailable without held-source information",
            },
        },
    }
    report = destination / "r4_005_006_completion_audit.json"
    _dump(report, payload)
    manifest = {
        "status": "pass",
        "report": report.relative_to(root).as_posix(),
        "report_sha256": _sha256(report),
    }
    manifest_path = destination / "artifact_manifest.json"
    _dump(manifest_path, manifest)
    return manifest


__all__ = [
    "add_causal_subject_bag_features",
    "run_completion_audit",
    "source_balanced_weights",
]
