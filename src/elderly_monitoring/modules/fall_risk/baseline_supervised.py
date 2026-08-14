"""Failure-closed supervised candidate for longitudinal fall baselines.

Phase 3 is deliberately separate from the frozen Phase 2 statistical evaluator.
The candidate is trained on train scoring periods only, while reference periods
are replayed causally to construct features. Validation is the only place where
calibration and threshold selection are allowed. Test data is never accepted by
the training or prediction APIs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from elderly_monitoring.modules.fall_risk.baseline import BaselineModelConfig, load_baseline_config
from elderly_monitoring.modules.fall_risk.baseline_longitudinal import (
    LongitudinalDataError,
    _value_hash,
    _rows_hash,
    generate_longitudinal_ablation_predictions,
)


MODEL_SCHEMA = "fall-baseline-supervised-candidate-v1"
MODEL_VERSION = "fall-personal-baseline-logistic-v1-provisional"
FEATURE_NAMES = (
    "no_personal_baseline_score",
    "mean_std_fusion_score",
    "median_mad_fusion_score",
    "robust_ewma_cusum_guarded_fusion_score",
    "baseline_deviation_score",
    "baseline_confidence",
    "valid_monitoring_hours",
    "quality_mean",
    "baseline_state_cold",
    "baseline_state_initial",
    "baseline_state_stable",
    "baseline_state_drift_suspected",
    "baseline_state_recovery",
    "baseline_deviation_available",
    "slow_reference_frozen",
    "cusum_triggered",
    "mean_gait_speed",
    "mean_gait_speed_available",
    "mean_sit_stand_duration",
    "mean_sit_stand_duration_available",
    "near_fall_rate_per_hour",
    "near_fall_rate_per_hour_available",
    "nighttime_activity_rate_per_hour",
    "nighttime_activity_rate_per_hour_available",
    "activity_volume",
    "activity_volume_available",
    "mean_gait_speed_deviation",
    "mean_gait_speed_deviation_available",
    "mean_sit_stand_duration_deviation",
    "mean_sit_stand_duration_deviation_available",
    "near_fall_rate_per_hour_deviation",
    "near_fall_rate_per_hour_deviation_available",
    "nighttime_activity_rate_per_hour_deviation",
    "nighttime_activity_rate_per_hour_deviation_available",
    "activity_volume_deviation",
    "activity_volume_deviation_available",
    "mean_gait_speed_cusum",
    "mean_sit_stand_duration_cusum",
    "near_fall_rate_per_hour_cusum",
    "nighttime_activity_rate_per_hour_cusum",
    "activity_volume_cusum",
)


@dataclass(frozen=True)
class SupervisedCandidateConfig:
    regularization_c: float = 1.0
    max_iter: int = 1000
    seed: int = 42
    calibration: str = "platt_validation"
    threshold_metric: str = "f1"

    def __post_init__(self) -> None:
        if self.regularization_c <= 0 or self.max_iter < 1:
            raise ValueError("regularization_c and max_iter must be positive")
        if self.calibration != "platt_validation":
            raise ValueError("only platt_validation calibration is supported")
        if self.threshold_metric != "f1":
            raise ValueError("only validation F1 threshold selection is supported")


def build_supervised_feature_rows(
    observations: Iterable[Mapping[str, Any]],
    assignments: Iterable[Mapping[str, Any]],
    split_metadata: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    partition: str,
    baseline_config: BaselineModelConfig | None = None,
) -> list[dict[str, Any]]:
    """Build causal scoring features without reading outcomes."""
    if partition not in {"train", "validation"}:
        raise LongitudinalDataError("supervised candidate only accepts train or validation")
    rows = [dict(row) for row in observations]
    assignments_rows = [dict(row) for row in assignments]
    if not any(row.get("partition") == partition for row in assignments_rows):
        return []
    predictions = generate_longitudinal_ablation_predictions(
        rows,
        assignments_rows,
        split_metadata,
        protocol,
        partition=partition,
        baseline_config=baseline_config,
    )
    by_observation: dict[str, dict[str, Any]] = {}
    for prediction in predictions:
        by_observation.setdefault(str(prediction["observation_id"]), {})[
            str(prediction["variant"])
        ] = prediction
    observation_index = {str(row["observation_id"]): row for row in rows}
    selected = [
        row for row in assignments_rows if row.get("partition") == partition
    ]
    scoring_ids = {
        str(row["observation_id"])
        for row in selected
        if row.get("role") == "scoring"
    }
    output: list[dict[str, Any]] = []
    for observation_id in sorted(scoring_ids):
        current = observation_index[observation_id]
        variants = by_observation.get(observation_id, {})
        robust = variants.get("robust_ewma_cusum_guarded")
        median = variants.get("median_mad_personal_baseline")
        legacy = variants.get("mean_std_personal_baseline")
        state = str((robust or median or legacy or {}).get("baseline_state", "unavailable"))
        baseline_output = _replay_baseline_output(current, rows, baseline_config)
        values = {
            "no_personal_baseline_score": _number(current.get("no_personal_baseline_score")),
            "mean_std_fusion_score": _number(legacy and legacy.get("score")),
            "median_mad_fusion_score": _number(median and median.get("score")),
            "robust_ewma_cusum_guarded_fusion_score": _number(robust and robust.get("score")),
            "baseline_deviation_score": _number(baseline_output.get("baseline_deviation_score")),
            "baseline_confidence": _number(baseline_output.get("baseline_confidence")),
            "valid_monitoring_hours": _number(current.get("valid_monitoring_hours")),
            "quality_mean": _number(baseline_output.get("baseline_quality", {}).get("current_quality_mean")),
            "baseline_deviation_available": baseline_output.get("baseline_deviation_score") is not None,
            "slow_reference_frozen": bool(baseline_output.get("slow_reference_frozen")),
            "cusum_triggered": bool(baseline_output.get("fast_state", {}).get("cusum_triggered")),
        }
        for name in ("cold", "initial", "stable", "drift_suspected", "recovery"):
            values[f"baseline_state_{name}"] = state == name
        for metric in ("mean_gait_speed", "mean_sit_stand_duration", "near_fall_rate_per_hour", "nighttime_activity_rate_per_hour", "activity_volume"):
            raw = _metric_value(current, metric)
            values[metric] = raw
            values[f"{metric}_available"] = raw is not None
            deviation = _number(
                baseline_output.get("metric_deviation_scores", {}).get(metric)
            )
            values[f"{metric}_deviation"] = deviation
            values[f"{metric}_deviation_available"] = deviation is not None
            values[f"{metric}_cusum"] = _number(
                baseline_output.get("fast_state", {}).get("cusum", {}).get(metric)
            )
        features = [0.0 if values[name] is None else float(values[name]) for name in FEATURE_NAMES]
        mask = [values[name] is not None if not name.endswith("_available") else True for name in FEATURE_NAMES]
        output.append({
            "observation_id": observation_id,
            "person_id": current.get("person_id"),
            "partition": partition,
            "baseline_state": state,
            "quality_state": current.get("quality_state"),
            "features": features,
            "feature_mask": mask,
            "feature_values": values,
        })
    return output


def train_longitudinal_logistic_candidate(
    observations: Iterable[Mapping[str, Any]],
    risk_labels: Iterable[Mapping[str, Any]],
    assignments: Iterable[Mapping[str, Any]],
    split_metadata: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    output_dir: str | Path,
    baseline_config: BaselineModelConfig | None = None,
    candidate_config: SupervisedCandidateConfig | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Train and validate the logistic candidate, or write a blocked report."""
    cfg = candidate_config or SupervisedCandidateConfig()
    destination = Path(output_dir)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"output exists: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    base = baseline_config or load_baseline_config()
    observations_rows = [dict(row) for row in observations]
    labels_rows = [dict(row) for row in risk_labels]
    assignments_rows = [dict(row) for row in assignments]
    observation_ids = {str(row.get("observation_id")) for row in observations_rows}
    sealed_test_ids = {
        str(row.get("observation_id"))
        for row in assignments_rows
        if row.get("partition") == "test"
    }
    if observation_ids & sealed_test_ids:
        raise LongitudinalDataError(
            "training input contains sealed test observations"
        )
    observation_index = {
        str(row.get("observation_id")): row for row in observations_rows
    }
    allowed_label_ids = {
        str(row.get("outcome_label_id"))
        for observation_id, row in observation_index.items()
        if observation_id not in sealed_test_ids and row.get("outcome_label_id")
    }
    supplied_label_ids = {str(row.get("label_id")) for row in labels_rows}
    if not supplied_label_ids <= allowed_label_ids:
        raise LongitudinalDataError("training input contains sealed or unrelated labels")
    blockers: list[str] = []
    if split_metadata.get("status") not in {"ready", "frozen"} or not split_metadata.get("split_id"):
        blockers.append("split_not_ready")
    if split_metadata.get("protocol_status") != protocol.get("protocol_status"):
        blockers.append("protocol_split_mismatch")
    if not observations_rows:
        blockers.append("no_observations")
    if not labels_rows:
        blockers.append("no_risk_labels")
    if not blockers:
        partition_ids = {
            partition: {
                str(row["observation_id"])
                for row in assignments_rows
                if row.get("partition") == partition
            }
            for partition in ("train", "validation")
        }
        partition_observations = {
            partition: [
                row
                for row in observations_rows
                if str(row.get("observation_id")) in partition_ids[partition]
            ]
            for partition in ("train", "validation")
        }
        train_features = build_supervised_feature_rows(
            partition_observations["train"], assignments_rows, split_metadata,
            protocol, partition="train", baseline_config=base,
        )
        validation_features = build_supervised_feature_rows(
            partition_observations["validation"], assignments_rows, split_metadata,
            protocol, partition="validation", baseline_config=base,
        )
        label_index = {str(row.get("label_id")): row for row in labels_rows}
        train_rows = _attach_targets(train_features, observations_rows, label_index)
        validation_rows = _attach_targets(validation_features, observations_rows, label_index)
        if len(train_rows) < 2 or {row["target"] for row in train_rows} != {0, 1}:
            blockers.append("train_requires_two_classes")
        if len(validation_rows) < 2 or {row["target"] for row in validation_rows} != {0, 1}:
            blockers.append("validation_requires_two_classes")
    if blockers:
        report = {
            "schema_version": MODEL_SCHEMA,
            "status": "blocked",
            "candidate_status": "blocked",
            "model_version": MODEL_VERSION,
            "blockers": sorted(set(blockers)),
            "fallback_policy": "retain_robust_ewma_cusum_guarded",
            "test_access": False,
            "split_id": split_metadata.get("split_id"),
            "protocol_sha256": _value_hash(protocol),
        }
        (destination / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return report
    try:
        import joblib
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RuntimeError("Phase 3 Logistic candidate requires the optional ml dependencies") from exc
    model = Pipeline([("scaler", StandardScaler()), ("logistic", LogisticRegression(C=cfg.regularization_c, max_iter=cfg.max_iter, random_state=cfg.seed, class_weight="balanced"))])
    model.fit([row["features"] for row in train_rows], [row["target"] for row in train_rows])
    validation_probabilities = model.predict_proba([row["features"] for row in validation_rows])[:, 1].tolist()
    calibration = _fit_validation_calibrator(validation_probabilities, [row["target"] for row in validation_rows], cfg)
    calibrated = [_apply_calibrator(value, calibration) for value in validation_probabilities]
    threshold = _select_f1_threshold(calibrated, [row["target"] for row in validation_rows])
    feature_contract_sha256 = _value_hash(
        {"feature_names": FEATURE_NAMES, "mask_separate": True}
    )
    candidate_id = f"fall-baseline-logistic:sha256:{_value_hash({'split_id': split_metadata['split_id'], 'protocol_sha256': _value_hash(protocol), 'baseline_config_sha256': _value_hash(asdict(base)), 'feature_contract_sha256': feature_contract_sha256, 'training_config': asdict(cfg)})[:16]}"
    model_payload = {
        "schema_version": MODEL_SCHEMA,
        "model_version": MODEL_VERSION,
        "candidate_id": candidate_id,
        "model": model,
        "feature_names": list(FEATURE_NAMES),
        "feature_mask_separate": True,
        "calibration": calibration,
        "threshold": threshold,
        "split_id": split_metadata["split_id"],
        "protocol_sha256": _value_hash(protocol),
        "baseline_config_sha256": _value_hash(asdict(base)),
        "feature_contract_sha256": feature_contract_sha256,
        "training_config": asdict(cfg),
        "test_access": False,
    }
    joblib.dump(model_payload, destination / "model.joblib")
    predictions = _prediction_rows(validation_rows, calibrated, threshold)
    metrics = _metrics(predictions)
    report = {
        "schema_version": MODEL_SCHEMA,
        "status": "provisional",
        "candidate_status": "validation_only",
        "model_version": MODEL_VERSION,
        "candidate_id": candidate_id,
        "split_id": split_metadata["split_id"],
        "protocol_sha256": _value_hash(protocol),
        "baseline_config_sha256": _value_hash(asdict(base)),
        "feature_contract_sha256": feature_contract_sha256,
        "feature_names": list(FEATURE_NAMES),
        "calibration": calibration,
        "threshold": threshold,
        "metrics": metrics,
        "statistical_reference_metrics": _guarded_reference_metrics(
            validation_rows, threshold
        ),
        "fallback_policy": "retain_robust_ewma_cusum_guarded_on_failure_or_low_quality",
        "test_access": False,
        "train_observations_sha256": _rows_hash(train_rows),
        "validation_observations_sha256": _rows_hash(validation_rows),
    }
    (destination / "validation_predictions.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions), encoding="utf-8")
    (destination / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def predict_longitudinal_logistic_candidate(model_path: str | Path, feature_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Predict validation-style rows; test partition is intentionally refused."""
    import joblib
    try:
        payload = joblib.load(Path(model_path))
    except Exception:
        return [_fallback_prediction(row, reason="model_load_failed") for row in feature_rows]
    if payload.get("test_access") is not False:
        raise LongitudinalDataError("candidate checkpoint has invalid test access metadata")
    model = payload["model"]
    calibration = payload["calibration"]
    threshold = float(payload["threshold"])
    output = []
    for row in feature_rows:
        if row.get("partition") == "test":
            raise LongitudinalDataError("Phase 3 candidate cannot predict sealed test features")
        if row.get("quality_state") == "unavailable":
            output.append(_fallback_prediction(row, reason="quality_unavailable"))
            continue
        try:
            probability = float(model.predict_proba([row["features"]])[0, 1])
            score = _apply_calibrator(probability, calibration)
        except Exception:
            output.append(_fallback_prediction(row, reason="prediction_failed"))
            continue
        output.append({"observation_id": row["observation_id"], "score": round(score, 6), "threshold": threshold, "model_version": payload["model_version"], "baseline_state": row["baseline_state"], "quality_state": row["quality_state"], "used_fallback": False, "fallback_reason": None})
    return output


def _replay_baseline_output(current: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], config: BaselineModelConfig | None) -> dict[str, Any]:
    from elderly_monitoring.modules.fall_risk.baseline import PersonalBaselineTracker
    tracker = PersonalBaselineTracker(config=config)
    timeline = sorted((row for row in rows if row.get("person_id") == current.get("person_id") and row.get("camera_profile_id") == current.get("camera_profile_id")), key=lambda row: str(row.get("period_start")))
    for row in timeline:
        result = tracker.update(row)
        if row.get("observation_id") == current.get("observation_id"):
            return result
    return {"baseline_deviation_score": None, "baseline_confidence": 0.0, "baseline_quality": {"current_quality_mean": 0.0}}


def _fallback_prediction(row: Mapping[str, Any], *, reason: str) -> dict[str, Any]:
    values = row.get("feature_values")
    guarded = None
    if isinstance(values, Mapping):
        guarded = _number(values.get("robust_ewma_cusum_guarded_fusion_score"))
        if guarded is None:
            guarded = _number(values.get("robust_ewma_cusum_guarded_score"))
    if guarded is None:
        guarded = 0.0
    return {
        "observation_id": row.get("observation_id"),
        "score": round(guarded, 6),
        "threshold": None,
        "model_version": "robust_ewma_cusum_guarded-fallback-v1",
        "baseline_state": row.get("baseline_state", "unavailable"),
        "quality_state": row.get("quality_state", "unavailable"),
        "used_fallback": True,
        "fallback_reason": reason,
    }


def _attach_targets(features: Sequence[Mapping[str, Any]], observations: Sequence[Mapping[str, Any]], labels: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    observation_index = {str(row["observation_id"]): row for row in observations}
    output = []
    for row in features:
        observation = observation_index[row["observation_id"]]
        label = labels.get(str(observation.get("outcome_label_id")))
        if label is None:
            continue
        output.append({**dict(row), "target": int(int(label.get("risk_level", 0)) >= 3)})
    return output


def _fit_validation_calibrator(probabilities: Sequence[float], targets: Sequence[int], config: SupervisedCandidateConfig) -> dict[str, Any]:
    if len(set(targets)) < 2:
        return {"method": "identity", "status": "unavailable"}
    from sklearn.linear_model import LogisticRegression
    eps = 1e-6
    logits = [[math.log(min(1 - eps, max(eps, p)) / (1 - min(1 - eps, max(eps, p))))] for p in probabilities]
    calibrator = LogisticRegression(C=1.0, max_iter=config.max_iter, random_state=config.seed)
    calibrator.fit(logits, targets)
    return {"method": "platt_validation", "status": "fit", "intercept": float(calibrator.intercept_[0]), "coefficient": float(calibrator.coef_[0][0])}


def _apply_calibrator(probability: float, calibration: Mapping[str, Any]) -> float:
    if calibration.get("method") != "platt_validation" or calibration.get("status") != "fit":
        return max(0.0, min(1.0, float(probability)))
    eps = 1e-6
    p = min(1 - eps, max(eps, float(probability)))
    logit = math.log(p / (1 - p))
    value = float(calibration["intercept"]) + float(calibration["coefficient"]) * logit
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, value))))


def _select_f1_threshold(probabilities: Sequence[float], targets: Sequence[int]) -> float:
    candidates = sorted({0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95})
    best = (0.0, 0.5)
    for threshold in candidates:
        tp = sum(int(p >= threshold and y == 1) for p, y in zip(probabilities, targets))
        fp = sum(int(p >= threshold and y == 0) for p, y in zip(probabilities, targets))
        fn = sum(int(p < threshold and y == 1) for p, y in zip(probabilities, targets))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if (f1, -threshold) > (best[0], -best[1]):
            best = (f1, threshold)
    return float(best[1])


def _prediction_rows(rows: Sequence[Mapping[str, Any]], probabilities: Sequence[float], threshold: float) -> list[dict[str, Any]]:
    return [{"observation_id": row["observation_id"], "target": int(row["target"]), "score": round(float(probability), 6), "predicted": int(float(probability) >= threshold), "baseline_state": row["baseline_state"], "quality_state": row["quality_state"]} for row, probability in zip(rows, probabilities)]


def _metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    tp = sum(int(row["predicted"] == 1 and row["target"] == 1) for row in rows)
    fp = sum(int(row["predicted"] == 1 and row["target"] == 0) for row in rows)
    fn = sum(int(row["predicted"] == 0 and row["target"] == 1) for row in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"sample_count": len(rows), "positive_count": sum(int(row["target"]) for row in rows), "precision": round(precision, 6), "recall": round(recall, 6), "f1": round(2 * precision * recall / (precision + recall), 6) if precision + recall else 0.0}


def _guarded_reference_metrics(
    rows: Sequence[Mapping[str, Any]], threshold: float
) -> dict[str, Any]:
    predictions = []
    for row in rows:
        score = _number(
            row.get("feature_values", {}).get(
                "robust_ewma_cusum_guarded_fusion_score"
            )
        )
        if score is None:
            score = 0.0
        predictions.append(
            {
                "target": row["target"],
                "predicted": int(score >= threshold),
            }
        )
    return _metrics(predictions)


def _metric_value(row: Mapping[str, Any], metric: str) -> float | None:
    candidates = [metric]
    if metric == "mean_gait_speed":
        candidates.append("gait_stability_features")
    if metric == "mean_sit_stand_duration":
        candidates.append("sit_stand_features")
    for candidate in candidates:
        value = row.get(candidate)
        if candidate in {"gait_stability_features", "sit_stand_features"} and isinstance(value, Mapping):
            value = value.get("mean_center_speed_norm_per_sec" if metric == "mean_gait_speed" else "mean_duration_sec")
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
            return float(value)
    return None


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) else None
