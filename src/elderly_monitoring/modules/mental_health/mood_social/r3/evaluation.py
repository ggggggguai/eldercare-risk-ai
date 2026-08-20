"""Once-only formal outer evaluation for OPT-V333-007."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from elderly_monitoring.modules.mental_health.mood_social.r3.baseline import (
    legacy_train_and_predict,
)
from elderly_monitoring.modules.mental_health.mood_social.r3.calibration import (
    _calibration_metrics,
    _ece,
    _threshold_metrics,
    crossfit_calibrator,
    fit_calibrator,
    select_workpoints,
)
from elderly_monitoring.modules.mental_health.mood_social.r3.contract import (
    DEFAULT_OUTPUT_RELATIVE,
    R3_PROTOCOL_VERSION,
)
from elderly_monitoring.modules.mental_health.mood_social.r3.data import (
    eligible_rows,
    inner_fold_series,
    load_r3_training_frame,
)
from elderly_monitoring.modules.mental_health.mood_social.r3.fusion import (
    FusionSpec,
    _expert_choices,
    _fit_predict_fusion,
    _merge_meta_features,
    candidate_expert_train_and_predict,
)
from elderly_monitoring.modules.mental_health.mood_social.r3.modeling import (
    WeightSpec,
    binary_metrics,
    sample_weights,
    weighted_binary_metrics,
)


EVALUATION_VERSION = "mood-social-v3.3.3-r3-formal-outer-v4"
DEFAULT_EXPERT_SELECTION = Path(
    "reports/mental_health/mood_social/v3.3.3-r3/OPT-V333-003-004-selection/selected_experts.json"
)
DEFAULT_FUSION_SELECTION = Path(
    "reports/mental_health/mood_social/v3.3.3-r3/OPT-V333-005-fusion/selected_fusions.json"
)
DEFAULT_FUSION_INNER_OOF = Path(
    "reports/mental_health/mood_social/v3.3.3-r3/OPT-V333-005-fusion/selected_fusion_inner_oof.parquet"
)
DEFAULT_CALIBRATION_SELECTION = Path(
    "reports/mental_health/mood_social/v3.3.3-r3/OPT-V333-006-calibration/calibration_and_workpoints.json"
)
DEFAULT_BASELINE_OOF = Path(
    "reports/mental_health/mood_social/v3.3.3-r3/OPT-V333-002-baseline-replay/strict_baseline_oof.parquet"
)
DEFAULT_CONFIG = Path("configs/training/mood_social_v3_3_3_r3.yaml")
DEFAULT_REPORT_RELATIVE = Path(
    "reports/mental_health/mood_social/v3.3.3-r3/OPT-V333-007-formal-outer"
)
PARTICIPANT_WEIGHT = WeightSpec("participant_eval", participant_equal=True)
DEPLOYMENT_ROUTES = frozenset({"110", "111"})
NONINFERIORITY_MARGIN = -0.005


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_default(value: Any) -> Any:
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def _metric_pair(frame: pd.DataFrame, candidate_column: str = "candidate_probability") -> dict[str, Any]:
    target = frame["binary_target"].to_numpy(int)
    candidate = frame[candidate_column].to_numpy(float)
    baseline = frame["baseline_probability"].to_numpy(float)
    candidate_metrics = _calibration_metrics(
        frame.assign(probability=candidate), candidate
    )
    baseline_metrics = _calibration_metrics(frame.assign(probability=baseline), baseline)
    return {
        "row_count": int(len(frame)),
        "participant_count": int(frame["global_participant_id"].nunique()),
        "positive_row_count": int(target.sum()),
        "prevalence": float(target.mean()),
        "prediction_coverage": float(np.mean(np.isfinite(candidate))),
        "candidate": candidate_metrics,
        "baseline": baseline_metrics,
        "delta": {
            key: float(candidate_metrics[key] - baseline_metrics[key])
            for key in ("auprc", "auroc", "brier", "log_loss", "ece")
        },
    }


def _group_metrics(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for value, group in frame.groupby(column, sort=True):
        if group["binary_target"].nunique() < 2:
            continue
        result[str(value)] = _metric_pair(group)
    return result


def _participant_bootstrap(
    frame: pd.DataFrame,
    *,
    replicates: int = 2000,
    seed: int = 20260728,
) -> dict[str, Any]:
    participant_rows = (
        frame.reset_index(drop=True)
        .groupby(["dataset_id", "global_participant_id"], sort=True)
        .indices
    )
    source_groups: dict[str, list[np.ndarray]] = {}
    for (source, _participant), indices in participant_rows.items():
        source_groups.setdefault(str(source), []).append(np.asarray(indices, dtype=int))
    participant_count = sum(len(values) for values in source_groups.values())
    target = frame["binary_target"].to_numpy(int)
    candidate = frame["candidate_probability"].to_numpy(float)
    baseline = frame["baseline_probability"].to_numpy(float)
    rng = np.random.default_rng(seed)
    candidate_values = np.empty(replicates, dtype=float)
    baseline_values = np.empty(replicates, dtype=float)
    candidate_participant_values = np.empty(replicates, dtype=float)
    baseline_participant_values = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        sampled_rows: list[np.ndarray] = []
        sampled_weights: list[np.ndarray] = []
        for source in sorted(source_groups):
            arrays = source_groups[source]
            sampled = rng.integers(0, len(arrays), size=len(arrays))
            for index in sampled:
                rows = arrays[int(index)]
                sampled_rows.append(rows)
                sampled_weights.append(
                    np.full(len(rows), 1.0 / float(len(rows)), dtype=float)
                )
        indices = np.concatenate(sampled_rows)
        participant_weight = np.concatenate(sampled_weights)
        y = target[indices]
        candidate_values[replicate] = average_precision_score(y, candidate[indices])
        baseline_values[replicate] = average_precision_score(y, baseline[indices])
        candidate_participant_values[replicate] = average_precision_score(
            y, candidate[indices], sample_weight=participant_weight
        )
        baseline_participant_values[replicate] = average_precision_score(
            y, baseline[indices], sample_weight=participant_weight
        )
    delta = candidate_values - baseline_values
    participant_delta = candidate_participant_values - baseline_participant_values

    def summary(values: np.ndarray) -> dict[str, float]:
        lower, upper = np.quantile(values, [0.025, 0.975])
        return {
            "mean": float(values.mean()),
            "ci95_lower": float(lower),
            "ci95_upper": float(upper),
        }

    return {
        "unit": "global_participant_id with all windows carried together",
        "stratification": "dataset_id",
        "participant_count": int(participant_count),
        "replicates": int(replicates),
        "seed": int(seed),
        "candidate_auprc": summary(candidate_values),
        "baseline_auprc": summary(baseline_values),
        "delta_auprc": {
            **summary(delta),
            "probability_gt_zero": float(np.mean(delta > 0.0)),
            "probability_ge_0_005": float(np.mean(delta >= 0.005)),
            "probability_ge_0_010": float(np.mean(delta >= 0.010)),
        },
        "participant_equal_candidate_auprc": summary(candidate_participant_values),
        "participant_equal_baseline_auprc": summary(baseline_participant_values),
        "participant_equal_delta_auprc": {
            **summary(participant_delta),
            "probability_gt_zero": float(np.mean(participant_delta > 0.0)),
        },
    }


def _weighted_metric_pair(frame: pd.DataFrame, weight: np.ndarray) -> dict[str, Any]:
    candidate = weighted_binary_metrics(
        frame["binary_target"], frame["candidate_probability"], weight
    )
    baseline = weighted_binary_metrics(
        frame["binary_target"], frame["baseline_probability"], weight
    )
    return {
        "candidate": candidate,
        "baseline": baseline,
        "delta": {
            key: float(candidate[key] - baseline[key])
            for key in ("auprc", "auroc", "brier")
        },
    }


def _participant_aggregate_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    """One screening record per participant for the required participant table.

    The r3 dataset has no common natural timestamp across sources.  The frozen
    aggregation therefore uses an ever-positive screening target and the mean
    OOF probability across all windows for that participant.
    """

    participant = (
        frame.groupby("global_participant_id", sort=True, as_index=False)
        .agg(
            binary_target=("binary_target", "max"),
            candidate_probability=("candidate_probability", "mean"),
            baseline_probability=("baseline_probability", "mean"),
            window_count=("r3_row_id", "size"),
        )
    )
    return {
        "aggregation": {
            "target": "max binary target across all participant windows",
            "probability": "mean strict-OOF probability across all participant windows",
            "reason": "no common natural timestamp exists across the six-source canonical table",
        },
        **_metric_pair(participant),
        "window_count": {
            "min": int(participant["window_count"].min()),
            "median": float(participant["window_count"].median()),
            "max": int(participant["window_count"].max()),
        },
    }


def _modal_expert_choices(selection: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Freeze production/LODO hyperparameters without consulting outer results."""

    choices: dict[str, dict[str, Any]] = {}
    for expert_id in sorted(next(iter(selection["outer_choices"].values()))):
        records = [
            dict(outer[expert_id]) for outer in selection["outer_choices"].values()
        ]
        ranked: list[tuple[int, float, str, dict[str, Any]]] = []
        for candidate_id in sorted({str(item["candidate_id"]) for item in records}):
            matching = [
                item for item in records if str(item["candidate_id"]) == candidate_id
            ]
            mean_score = float(
                np.mean(
                    [
                        float(item["inner_oof_metrics"]["selection_score"])
                        for item in matching
                    ]
                )
            )
            ranked.append((len(matching), mean_score, candidate_id, matching[0]))
        choices[expert_id] = max(
            ranked, key=lambda item: (item[0], item[1], item[2])
        )[3]
    return choices


def _modal_fusion_spec(selection: Mapping[str, Any]) -> FusionSpec:
    records = [
        dict(item["selected_spec"]) for item in selection["outer_choices"].values()
    ]
    counts: dict[str, int] = {}
    for item in records:
        candidate_id = str(item["candidate_id"])
        counts[candidate_id] = counts.get(candidate_id, 0) + 1
    selected_id = max(sorted(counts), key=lambda item: (counts[item], item))
    payload = next(item for item in records if str(item["candidate_id"]) == selected_id)
    return FusionSpec(
        candidate_id=selected_id,
        kind=str(payload["kind"]),
        params=dict(payload["params"]),
    )


def _modal_calibration_method(selection: Mapping[str, Any]) -> str:
    methods = [
        str(item["selected_method"]) for item in selection["outer_choices"].values()
    ]
    return max(sorted(set(methods)), key=lambda item: (methods.count(item), item))


def _lodo_source_eligibility(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for source, group in frame.groupby("dataset_id", sort=True):
        participant_target = group.groupby("global_participant_id", sort=False)[
            "binary_target"
        ].max()
        rows = int(len(group))
        positives = int(participant_target.sum())
        result[str(source)] = {
            "row_count": rows,
            "participant_count": int(len(participant_target)),
            "positive_participant_count": positives,
            "hard_gate_eligible": rows >= 500 and positives >= 100,
        }
    return result


def _crossfit_fusion_probability(
    meta: pd.DataFrame, spec: FusionSpec
) -> np.ndarray:
    probability = np.full(len(meta), np.nan, dtype=float)
    fold = meta["fusion_validation_fold"].to_numpy(int)
    for validation_fold in sorted(np.unique(fold)):
        fit = meta.loc[fold != validation_fold]
        validation = meta.loc[fold == validation_fold]
        probability[fold == validation_fold] = _fit_predict_fusion(
            spec, fit, validation
        )
    if not np.isfinite(probability).all():
        raise ValueError("LODO fusion OOF is incomplete")
    return probability


def _formal_lodo_source(
    frame: pd.DataFrame,
    *,
    source: str,
    expert_choices: Mapping[str, Mapping[str, Any]],
    fusion_spec: FusionSpec,
    calibration_method: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Retrain every learned stage after excluding one complete source."""

    train = frame.loc[frame["dataset_id"].astype(str).ne(source)].copy()
    test = frame.loc[frame["dataset_id"].astype(str).eq(source)].copy()
    train_fold = train["outer_fold"].astype(int)
    new_train, new_test = candidate_expert_train_and_predict(
        train, train_fold, test, expert_choices
    )
    old_train, _, old_test_probability = legacy_train_and_predict(
        train, train_fold, test
    )
    train_meta = _merge_meta_features(new_train, old_train)
    test_old = test[["r3_row_id"]].copy()
    test_old["baseline_probability"] = old_test_probability
    test_meta = _merge_meta_features(new_test, test_old)

    fusion_oof = _crossfit_fusion_probability(train_meta, fusion_spec)
    calibration_frame = train_meta.copy()
    calibration_frame["probability"] = fusion_oof
    calibrated_oof, calibration_audit = crossfit_calibrator(
        calibration_frame, calibration_method
    )
    workpoints = select_workpoints(
        calibration_frame["binary_target"], calibrated_oof
    )
    calibrator = fit_calibrator(calibration_frame, calibration_method)
    raw_test = _fit_predict_fusion(fusion_spec, train_meta, test_meta)
    calibration_input = test_meta.copy()
    calibration_input["probability"] = raw_test
    candidate = calibrator.predict(calibration_input)

    output = test_meta[
        [
            "r3_row_id",
            "dataset_id",
            "global_participant_id",
            "binary_target",
            "route_pattern",
            "baseline_probability",
        ]
    ].copy()
    output["candidate_probability"] = candidate
    output["uncalibrated_candidate_probability"] = raw_test
    output["held_out_source"] = source
    audit = {
        "held_out_source": source,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_sources": sorted(train["dataset_id"].astype(str).unique()),
        "participant_overlap": int(
            len(
                set(train["global_participant_id"].astype(str))
                & set(test["global_participant_id"].astype(str))
            )
        ),
        "expert_candidate_ids": {
            key: str(value["candidate_id"]) for key, value in expert_choices.items()
        },
        "fusion_candidate_id": fusion_spec.candidate_id,
        "calibration_method": calibration_method,
        "calibration_crossfit_audit": calibration_audit,
        "workpoints": workpoints,
    }
    return output, audit


def _seal_inputs(root: Path) -> dict[str, Any]:
    paths = {
        "training_config": root / DEFAULT_CONFIG,
        "r2_split_manifest": root
        / "data/processed/mental_health/mood_social/v3.3.3-r2/splits/split_manifest.json",
        "population": root / DEFAULT_OUTPUT_RELATIVE / "population_assignments.parquet",
        "population_manifest": root / DEFAULT_OUTPUT_RELATIVE / "artifact_manifest.json",
        "expert_selection": root / DEFAULT_EXPERT_SELECTION,
        "fusion_selection": root / DEFAULT_FUSION_SELECTION,
        "fusion_inner_oof": root / DEFAULT_FUSION_INNER_OOF,
        "calibration_selection": root / DEFAULT_CALIBRATION_SELECTION,
        "baseline_oof": root / DEFAULT_BASELINE_OOF,
        "environment_manifest": root
        / "reports/mental_health/mood_social/v3.3.3-r3/environment_manifest.json",
    }
    code_root = root / "src/elderly_monitoring/modules/mental_health/mood_social/r3"
    for path in sorted(code_root.glob("*.py"), key=lambda item: item.name):
        paths[f"code_{path.stem}"] = path
    paths["formal_entrypoint"] = (
        root / "scripts/evaluate_mood_social_v3_3_3_r3_formal.py"
    )
    failed_precompute_root = root / DEFAULT_REPORT_RELATIVE
    failed_seal = failed_precompute_root / "FORMAL_OUTER_PRECOMPUTE_FAILED_v2.json"
    failed_log = failed_precompute_root / "PRECOMPUTE_FAILED_v2.stderr.log"
    if failed_seal.exists():
        paths["precompute_failed_v2_seal"] = failed_seal
    if failed_log.exists():
        paths["precompute_failed_v2_log"] = failed_log
    serialization_failure_names = (
        "FORMAL_OUTER_SERIALIZATION_FAILED_v3.json",
        "SERIALIZATION_FAILED_v3.stderr.log",
        "SERIALIZATION_FAILED_v3.formal_outer_oof.parquet",
        "SERIALIZATION_FAILED_v3.lodo_nhanes.parquet",
    )
    for name in serialization_failure_names:
        path = failed_precompute_root / name
        if path.exists():
            paths[f"serialization_failed_v3_{path.stem}"] = path
    return {
        "protocol_version": R3_PROTOCOL_VERSION,
        "evaluation_version": EVALUATION_VERSION,
        "outer_open_count": 1,
        "inputs": {
            name: {
                "path": path.relative_to(root).as_posix(),
                "sha256": _sha256_file(path),
            }
            for name, path in paths.items()
        },
    }


def _formal_outer_fold(
    frame: pd.DataFrame,
    *,
    outer_fold: int,
    expert_selection: Mapping[str, Any],
    fusion_selection: Mapping[str, Any],
    fusion_inner_oof: pd.DataFrame,
    calibration_selection: Mapping[str, Any],
    frozen_baseline: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    outer_train = frame.loc[frame["outer_fold"].ne(outer_fold)].copy()
    outer_test = frame.loc[frame["outer_fold"].eq(outer_fold)].copy()
    inner = inner_fold_series(frame, outer_fold).loc[outer_train.index].astype(int)
    new_train, new_test = candidate_expert_train_and_predict(
        outer_train,
        inner,
        outer_test,
        _expert_choices(expert_selection, outer_fold),
    )
    old_train, _, recomputed_old_test = legacy_train_and_predict(
        outer_train, inner, outer_test
    )
    frozen_fold = frozen_baseline.loc[frozen_baseline["outer_fold"].eq(outer_fold)][
        ["r3_row_id", "baseline_probability"]
    ]
    parity = outer_test[["r3_row_id"]].copy()
    parity["recomputed"] = recomputed_old_test
    parity = parity.merge(frozen_fold, on="r3_row_id", validate="one_to_one")
    max_baseline_parity_error = float(
        np.max(np.abs(parity["recomputed"] - parity["baseline_probability"]))
    )
    if max_baseline_parity_error > 1.0e-12:
        raise ValueError(
            f"frozen baseline replay parity failed for outer {outer_fold}: {max_baseline_parity_error}"
        )
    train_meta = _merge_meta_features(new_train, old_train)
    test_old = outer_test[["r3_row_id"]].merge(
        frozen_fold, on="r3_row_id", validate="one_to_one"
    )
    test_meta = _merge_meta_features(new_test, test_old)
    selected_spec_payload = fusion_selection["outer_choices"][str(outer_fold)]["selected_spec"]
    selected_spec = FusionSpec(
        candidate_id=str(selected_spec_payload["candidate_id"]),
        kind=str(selected_spec_payload["kind"]),
        params=dict(selected_spec_payload["params"]),
    )
    uncalibrated = _fit_predict_fusion(selected_spec, train_meta, test_meta)
    inner_calibration_frame = fusion_inner_oof.loc[
        fusion_inner_oof["outer_context"].eq(outer_fold)
    ].copy()
    method = str(
        calibration_selection["outer_choices"][str(outer_fold)]["selected_method"]
    )
    fitted_calibrator = fit_calibrator(inner_calibration_frame, method)
    calibration_input = test_meta.copy()
    calibration_input["probability"] = uncalibrated
    calibrated = fitted_calibrator.predict(calibration_input)
    output = test_meta[
        [
            "r3_row_id",
            "dataset_id",
            "global_participant_id",
            "binary_target",
            "route_pattern",
            "baseline_probability",
        ]
    ].copy()
    output["outer_fold"] = outer_fold
    output["fusion_candidate_id"] = selected_spec.candidate_id
    output["calibration_method"] = method
    output["uncalibrated_candidate_probability"] = uncalibrated
    output["candidate_probability"] = calibrated
    workpoints = calibration_selection["outer_choices"][str(outer_fold)]["workpoints"]
    output["competition_prediction"] = (
        calibrated >= float(workpoints["competition"]["threshold"])
    ).astype(int)
    output["safety_prediction"] = (
        calibrated >= float(workpoints["safety"]["threshold"])
    ).astype(int)
    audit = {
        "outer_fold": outer_fold,
        "outer_train_rows": int(len(outer_train)),
        "outer_test_rows": int(len(outer_test)),
        "participant_overlap": int(
            len(
                set(outer_train["global_participant_id"].astype(str))
                & set(outer_test["global_participant_id"].astype(str))
            )
        ),
        "baseline_parity_max_abs_error": max_baseline_parity_error,
        "fusion_candidate_id": selected_spec.candidate_id,
        "calibration_method": method,
        "calibration_eligible_routes": list(fitted_calibrator.eligible_routes),
    }
    return output, audit


def run_formal_outer_evaluation(
    *,
    repository_root: Path,
    report_directory: Path | None = None,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    output = Path(report_directory) if report_directory else root / DEFAULT_REPORT_RELATIVE
    if not output.is_absolute():
        output = root / output
    output.mkdir(parents=True, exist_ok=True)
    seal_path = output / "FORMAL_OUTER_OPENED.json"
    expected_seal = _seal_inputs(root)
    if seal_path.exists():
        actual_seal = _read_json(seal_path)
        if actual_seal != expected_seal:
            raise ValueError("formal outer seal exists with different frozen inputs")
    else:
        seal_path.write_text(
            json.dumps(
                expected_seal,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
                default=_json_default,
            )
            + "\n",
            encoding="utf-8",
        )

    all_frame = load_r3_training_frame(repository_root=root)
    frame = eligible_rows(all_frame)
    no_evidence = all_frame.loc[all_frame["no_evidence"].astype(bool)].copy()
    coverage_report = {
        "total_rows": int(len(all_frame)),
        "r3_eligible_rows": int(len(frame)),
        "no_evidence_rows": int(len(no_evidence)),
        "prediction_coverage": float(len(frame) / len(all_frame)),
        "no_evidence_positive_rows": int(no_evidence["binary_target"].sum()),
        "no_evidence_prevalence": float(no_evidence["binary_target"].mean()),
        "no_evidence_by_source": {
            str(source): int(len(group))
            for source, group in no_evidence.groupby("dataset_id", sort=True)
        },
        "no_evidence_policy": "abstain; do not emit a learned risk probability",
    }
    expert_selection = _read_json(root / DEFAULT_EXPERT_SELECTION)
    fusion_selection = _read_json(root / DEFAULT_FUSION_SELECTION)
    fusion_inner_oof = pd.read_parquet(root / DEFAULT_FUSION_INNER_OOF)
    calibration_selection = _read_json(root / DEFAULT_CALIBRATION_SELECTION)
    baseline = pd.read_parquet(root / DEFAULT_BASELINE_OOF)
    checkpoint_root = output / "_checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    predictions: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    for outer_fold in range(5):
        prediction_path = checkpoint_root / f"outer{outer_fold}.parquet"
        audit_path = checkpoint_root / f"outer{outer_fold}.json"
        if prediction_path.exists() and audit_path.exists():
            prediction = pd.read_parquet(prediction_path)
            audit = _read_json(audit_path)
        else:
            prediction, audit = _formal_outer_fold(
                frame,
                outer_fold=outer_fold,
                expert_selection=expert_selection,
                fusion_selection=fusion_selection,
                fusion_inner_oof=fusion_inner_oof,
                calibration_selection=calibration_selection,
                frozen_baseline=baseline,
            )
            prediction.to_parquet(prediction_path, index=False)
            audit_path.write_text(
                    json.dumps(
                        audit,
                        ensure_ascii=False,
                        sort_keys=True,
                        indent=2,
                        allow_nan=False,
                        default=_json_default,
                    )
                + "\n",
                encoding="utf-8",
            )
        predictions.append(prediction)
        audits.append(audit)
    oof = pd.concat(predictions, ignore_index=True).sort_values(
        ["dataset_id", "r3_row_id"], kind="stable"
    ).reset_index(drop=True)
    if len(oof) != len(frame) or oof["r3_row_id"].duplicated().any():
        raise ValueError("formal outer candidate OOF coverage is invalid")

    lodo_eligibility = _lodo_source_eligibility(frame)
    lodo_expert_choices = _modal_expert_choices(expert_selection)
    lodo_fusion_spec = _modal_fusion_spec(fusion_selection)
    lodo_calibration_method = _modal_calibration_method(calibration_selection)
    lodo_predictions: dict[str, pd.DataFrame] = {}
    lodo_audits: dict[str, Any] = {}
    lodo_metrics: dict[str, Any] = {}
    for source, eligibility in lodo_eligibility.items():
        if not eligibility["hard_gate_eligible"]:
            lodo_metrics[source] = {
                **eligibility,
                "status": "descriptive_only_not_retrained",
            }
            continue
        safe_source = "".join(
            character if character.isalnum() or character in {"-", "_"} else "_"
            for character in source
        )
        prediction_path = checkpoint_root / f"lodo_{safe_source}.parquet"
        audit_path = checkpoint_root / f"lodo_{safe_source}.json"
        try:
            if prediction_path.exists() and audit_path.exists():
                prediction = pd.read_parquet(prediction_path)
                audit = _read_json(audit_path)
            else:
                prediction, audit = _formal_lodo_source(
                    frame,
                    source=source,
                    expert_choices=lodo_expert_choices,
                    fusion_spec=lodo_fusion_spec,
                    calibration_method=lodo_calibration_method,
                )
                prediction.to_parquet(prediction_path, index=False)
                audit_path.write_text(
                    json.dumps(
                        audit,
                        ensure_ascii=False,
                        sort_keys=True,
                        indent=2,
                        allow_nan=False,
                        default=_json_default,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            lodo_predictions[source] = prediction
            lodo_audits[source] = audit
            lodo_metrics[source] = {
                **eligibility,
                "status": "pass",
                "metrics": _metric_pair(prediction),
            }
        except Exception as exc:  # preserve the once-open formal result and fail closed
            lodo_metrics[source] = {
                **eligibility,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
    overall = _metric_pair(oof)
    participant_weight = sample_weights(oof, PARTICIPANT_WEIGHT)
    participant_metrics = _weighted_metric_pair(oof, participant_weight)
    natural_weight = np.ones(len(oof), dtype=float)
    mixed_weight = 0.6 * (natural_weight / natural_weight.mean()) + 0.4 * (
        participant_weight / participant_weight.mean()
    )
    mixed_metrics = _weighted_metric_pair(oof, mixed_weight)
    participant_aggregate = _participant_aggregate_metrics(oof)
    deployment = oof.loc[oof["route_pattern"].isin(DEPLOYMENT_ROUTES)].copy()
    if deployment.empty:
        raise ValueError("C+S/C+S+P deployment main table is empty")
    deployment_metrics = _metric_pair(deployment)
    route_metrics = _group_metrics(oof, "route_pattern")
    source_metrics = _group_metrics(oof, "dataset_id")
    outer_metrics = _group_metrics(oof, "outer_fold")
    excluded_source_aggregate_sensitivity = {
        source: _metric_pair(oof.loc[oof["dataset_id"].ne(source)])
        for source in sorted(oof["dataset_id"].astype(str).unique())
    }
    bootstrap = _participant_bootstrap(oof)
    competition_threshold_metrics = {
        str(fold): _threshold_metrics(
            group["binary_target"].to_numpy(int),
            group["candidate_probability"].to_numpy(float),
            float(
                calibration_selection["outer_choices"][str(fold)]["workpoints"]["competition"]["threshold"]
            ),
        )
        for fold, group in oof.groupby("outer_fold", sort=True)
    }
    safety_threshold_metrics = {
        str(fold): _threshold_metrics(
            group["binary_target"].to_numpy(int),
            group["candidate_probability"].to_numpy(float),
            float(
                calibration_selection["outer_choices"][str(fold)]["workpoints"]["safety"]["threshold"]
            ),
        )
        for fold, group in oof.groupby("outer_fold", sort=True)
    }
    route_regressions = {
        route: detail["delta"]["auprc"]
        for route, detail in route_metrics.items()
        if detail["row_count"] >= 500
    }
    fold_deltas = {
        fold: detail["delta"]["auprc"] for fold, detail in outer_metrics.items()
    }
    fold_nonnegative_count = sum(value >= 0.0 for value in fold_deltas.values())
    worst_fold_delta = min(fold_deltas.values())
    gates = {
        "outer_isolation": all(audit["participant_overlap"] == 0 for audit in audits),
        "baseline_replay_parity": all(audit["baseline_parity_max_abs_error"] <= 1.0e-12 for audit in audits),
        "prediction_coverage": len(oof) == 22070 and oof["candidate_probability"].notna().all(),
        "deployment_cs_gain_ge_0_005": deployment_metrics["delta"]["auprc"] >= 0.005,
        "common_support_not_below_baseline": overall["delta"]["auprc"] >= 0.0,
        "effective_gain_ge_0_005": overall["delta"]["auprc"] >= 0.005,
        "priority_gain_ge_0_010": overall["delta"]["auprc"] >= 0.010,
        "auroc_noninferior_ge_minus_0_005": overall["delta"]["auroc"] >= NONINFERIORITY_MARGIN,
        "participant_equal_auprc_noninferior_ge_minus_0_005": (
            participant_metrics["delta"]["auprc"] >= NONINFERIORITY_MARGIN
        ),
        "participant_bootstrap_delta_ci_ge_minus_0_005": (
            bootstrap["delta_auprc"]["ci95_lower"] >= NONINFERIORITY_MARGIN
        ),
        "outer_fold_stability": fold_nonnegative_count >= 3 or worst_fold_delta >= -0.02,
        "no_large_route_regression": all(value >= -0.02 for value in route_regressions.values()),
        "brier_not_materially_worse": overall["delta"]["brier"] <= 0.005,
        "lodo_major_sources_noninferior_ge_minus_0_005": all(
            detail.get("status") == "pass"
            and detail["metrics"]["delta"]["auprc"] >= NONINFERIORITY_MARGIN
            for detail in lodo_metrics.values()
            if detail["hard_gate_eligible"]
        ),
    }
    metric_gate_pass = bool(
        gates["outer_isolation"]
        and gates["baseline_replay_parity"]
        and gates["prediction_coverage"]
        and gates["deployment_cs_gain_ge_0_005"]
        and gates["effective_gain_ge_0_005"]
        and gates["auroc_noninferior_ge_minus_0_005"]
        and gates["participant_equal_auprc_noninferior_ge_minus_0_005"]
        and gates["participant_bootstrap_delta_ci_ge_minus_0_005"]
        and gates["outer_fold_stability"]
        and gates["no_large_route_regression"]
        and gates["brier_not_materially_worse"]
        and gates["lodo_major_sources_noninferior_ge_minus_0_005"]
    )
    report = {
        "protocol_version": R3_PROTOCOL_VERSION,
        "evaluation_version": EVALUATION_VERSION,
        "task_id": "OPT-V333-007",
        "formal_outer_open_count": 1,
        "precompute_retry_audit": {
            "attempt_count": 3,
            "first_attempt_reached_outer_predictions": False,
            "first_attempt_failure": "coverage report referenced absent r3_population column",
            "second_attempt_reached_outer_predictions": True,
            "second_attempt_result_inspected": False,
            "second_attempt_failure": "numpy.bool_ JSON serialization at final report write",
            "fixes": [
                "use frozen boolean no_evidence assignment column",
                "serialize numpy scalar values with explicit lossless Python conversions",
            ],
            "checkpoint_policy": "reuse frozen v3 prediction checkpoints; do not refit or retune completed folds",
            "outer_result_driven_change": False,
        },
        "population": "common-support equals r3-eligible under frozen r3 contract",
        "coverage_and_no_evidence": coverage_report,
        "promotion_main_table_common_support": overall,
        "deployment_main_table_cs_and_csp": deployment_metrics,
        "overall": overall,
        "participant_equal": participant_metrics,
        "preregistered_mixed_weight": {
            "definition": "0.6 natural-row + 0.4 participant-equal normalized sample weights",
            **mixed_metrics,
        },
        "participant_aggregate": participant_aggregate,
        "by_route": route_metrics,
        "by_source": source_metrics,
        "by_outer_fold": outer_metrics,
        "excluded_source_aggregate_sensitivity_not_lodo": excluded_source_aggregate_sensitivity,
        "leave_one_dataset_out_retraining": {
            "protocol": (
                "freeze modal inner-selected hyperparameters, exclude the complete source, "
                "then refit preprocessing, experts, legacy baseline, fusion, calibration, and workpoints"
            ),
            "eligibility": "at least 500 eligible rows and 100 positive participants",
            "modal_expert_candidate_ids": {
                key: str(value["candidate_id"])
                for key, value in lodo_expert_choices.items()
            },
            "modal_fusion_candidate_id": lodo_fusion_spec.candidate_id,
            "modal_calibration_method": lodo_calibration_method,
            "sources": lodo_metrics,
            "audits": lodo_audits,
        },
        "participant_bootstrap": bootstrap,
        "competition_workpoint_outer_metrics": competition_threshold_metrics,
        "safety_workpoint_outer_metrics": safety_threshold_metrics,
        "outer_isolation_audit": audits,
        "outer_fold_stability": {
            "auprc_deltas": fold_deltas,
            "nonnegative_fold_count": int(fold_nonnegative_count),
            "worst_fold_delta": float(worst_fold_delta),
            "gate": "at least 3/5 nonnegative OR worst fold no lower than -0.02",
        },
        "gates": gates,
        "metric_gate_pass": metric_gate_pass,
        "deployment_pass": False,
        "release_status": (
            "formal_metrics_pass_pending_OPT-V333-008_hard_gates"
            if metric_gate_pass
            else "research_only_keep_MH-20260802-013_fallback"
        ),
    }
    oof_path = output / "formal_outer_oof.parquet"
    report_path = output / "formal_evaluation.json"
    if oof_path.exists() or report_path.exists():
        raise FileExistsError("formal outer final artifacts already exist")
    oof.to_parquet(oof_path, index=False)
    lodo_artifacts: dict[str, dict[str, Any]] = {}
    for source, prediction in lodo_predictions.items():
        safe_source = "".join(
            character if character.isalnum() or character in {"-", "_"} else "_"
            for character in source
        )
        path = output / f"lodo_{safe_source}.parquet"
        prediction.to_parquet(path, index=False)
        lodo_artifacts[source] = {
            "path": path.relative_to(root).as_posix(),
            "sha256": _sha256_file(path),
        }
    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "protocol_version": R3_PROTOCOL_VERSION,
        "evaluation_version": EVALUATION_VERSION,
        "task_id": "OPT-V333-007",
        "status": "pass",
        "formal_outer_open_count": 1,
        "metric_gate_pass": metric_gate_pass,
        "deployment_pass": False,
        "seal_path": seal_path.relative_to(root).as_posix(),
        "seal_sha256": _sha256_file(seal_path),
        "oof_path": oof_path.relative_to(root).as_posix(),
        "oof_sha256": _sha256_file(oof_path),
        "report_path": report_path.relative_to(root).as_posix(),
        "report_sha256": _sha256_file(report_path),
        "lodo_artifacts": lodo_artifacts,
    }
    manifest_path = output / "artifact_manifest.json"
    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


__all__ = ["EVALUATION_VERSION", "run_formal_outer_evaluation"]
