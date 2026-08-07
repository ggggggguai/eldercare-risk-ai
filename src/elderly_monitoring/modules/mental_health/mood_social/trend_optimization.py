"""Robust PersonalTrend candidates for OPT-TREND-001.

Runtime feature extraction implements 7/28-day median/MAD baselines, EWMA,
Theil-Sen slope, change-point contrast, weekday/weekend rhythm and abnormal-day
exclusion.  Public-data evaluation uses the frozen TREND-001 component OOF and
participant-level nested calibration; it is an engineering proxy, not household
deployment validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
import yaml


TASK_ID = "OPT-TREND-001"
RUN_ID = "MH-20260804-016"
MODEL_VERSION = "personal-trend-optimization-candidate-v3.3.4-v1"
RANDOM_SEED = 20260728
OUTER_FOLDS = 5
INNER_FOLDS = 5
EPSILON = 1.0e-7
BRANCHES = ("activity", "sleep", "social")
CANDIDATES = (
    "baseline_probability",
    "robust_dual_scale",
    "ewma_theilsen_change_point",
    "concordance_guarded",
)
TREND_OOF_RELATIVE = (
    "reports/mental_health/mood_social/MH-20260801-008/oof_predictions.parquet"
)
TREND_OOF_SHA256 = "f23d79b7be21a6a863414e139f537d61c8a9cb06e9a64e061dc60627893d408a"
COMPONENTS = (
    "robust_deviation",
    "risk_slope",
    "isolation_forest",
    "change_point",
    "persistence",
    "history_coverage",
)


class TrendOptimizationError(RuntimeError):
    """Raised when the frozen OPT-TREND-001 protocol is violated."""


@dataclass(frozen=True)
class TrendOptimizationConfig:
    repository_root: Path
    payload: Mapping[str, Any]
    config_path: Path

    @property
    def report_directory(self) -> Path:
        return _resolve(
            self.repository_root, self.payload["output"]["report_directory"]
        )

    @property
    def model_path(self) -> Path:
        return _resolve(self.repository_root, self.payload["output"]["model_path"])

    @property
    def manifest_path(self) -> Path:
        return _resolve(self.repository_root, self.payload["output"]["manifest_path"])


@dataclass(frozen=True)
class RobustTrendFeatures:
    short_deviation: float
    long_deviation: float
    ewma_deviation: float
    theil_sen_slope: float
    change_point: float
    rhythm_deviation: float
    valid_short_days: int
    valid_long_days: int
    excluded_abnormal_days: int
    reliability: float


@dataclass(frozen=True)
class TrendCandidateModel:
    branch: str
    candidate_id: str
    calibrator: LogisticRegression
    social_reliability_cap: float = 0.25

    def predict_frame(self, frame: pd.DataFrame) -> np.ndarray:
        raw = _candidate_score(frame, self.branch, self.candidate_id)
        calibrated = self.calibrator.predict_proba(_logit(raw)[:, None])[:, 1]
        return np.clip(calibrated, EPSILON, 1.0 - EPSILON)


def compute_robust_trend_features(
    history: pd.DataFrame,
    *,
    current_date: date,
    current_value: float,
    value_column: str = "value",
    date_column: str = "date",
    abnormal_column: str = "abnormal",
) -> RobustTrendFeatures:
    """Compute the runtime 7/28-day trend candidate without label access."""
    required = {date_column, value_column, abnormal_column}
    if required - set(history.columns) or not np.isfinite(float(current_value)):
        raise TrendOptimizationError("robust trend history contract is invalid")
    frame = history.loc[:, [date_column, value_column, abnormal_column]].copy()
    frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce")
    frame[value_column] = pd.to_numeric(frame[value_column], errors="coerce")
    frame[abnormal_column] = frame[abnormal_column].astype(bool)
    before = frame[date_column].dt.date < current_date
    within = frame[date_column].dt.date >= current_date - pd.Timedelta(days=28)
    excluded = int((before & within & frame[abnormal_column]).sum())
    valid = frame.loc[before & within & ~frame[abnormal_column]].dropna(
        subset=[date_column, value_column]
    )
    valid = valid.sort_values(date_column, kind="stable")
    if valid.empty:
        return RobustTrendFeatures(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, excluded, 0.0)
    short_cutoff = current_date - pd.Timedelta(days=7)
    short = valid.loc[valid[date_column].dt.date >= short_cutoff]
    short_values = short[value_column].to_numpy(dtype=float)
    long_values = valid[value_column].to_numpy(dtype=float)
    short_deviation = _robust_deviation(float(current_value), short_values)
    long_deviation = _robust_deviation(float(current_value), long_values)
    ewma = _ewma(long_values, alpha=0.35)
    ewma_scale = max(_mad(long_values), 0.05)
    ewma_deviation = float(
        np.clip(abs(float(current_value) - ewma) / (2.0 * ewma_scale), 0.0, 1.0)
    )
    ordinals = np.array(
        [
            (value.date() - valid.iloc[0][date_column].date()).days
            for value in valid[date_column]
        ],
        dtype=float,
    )
    slope = _theil_sen(ordinals, long_values)
    slope_value = float(
        np.clip(abs(slope) * 7.0 / max(_mad(long_values), 0.05), 0.0, 1.0)
    )
    change = _change_point(long_values)
    same_rhythm = valid.loc[
        valid[date_column].dt.weekday.lt(5) == (current_date.weekday() < 5),
        value_column,
    ].to_numpy(dtype=float)
    rhythm = (
        _robust_deviation(float(current_value), same_rhythm)
        if len(same_rhythm) >= 3
        else 0.0
    )
    coverage = min(1.0, len(long_values) / 14.0)
    reliability = float(
        np.clip(
            coverage * (1.0 - min(1.0, abs(short_deviation - long_deviation))), 0.0, 1.0
        )
    )
    return RobustTrendFeatures(
        short_deviation,
        long_deviation,
        ewma_deviation,
        slope_value,
        change,
        rhythm,
        len(short_values),
        len(long_values),
        excluded,
        reliability,
    )


def load_trend_optimization_config(
    path: str | Path, *, repository_root: str | Path | None = None
) -> TrendOptimizationConfig:
    config_path = Path(path).resolve()
    root = Path(repository_root or config_path.parents[2]).resolve()
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise TrendOptimizationError("OPT-TREND-001 config is unreadable") from exc
    if not isinstance(payload, Mapping):
        raise TrendOptimizationError("OPT-TREND-001 config must be a mapping")
    if payload.get("task_id") != TASK_ID or payload.get("run_id") != RUN_ID:
        raise TrendOptimizationError("OPT-TREND-001 identity changed")
    if (
        tuple(payload.get("branches", ())) != BRANCHES
        or tuple(payload.get("candidates", {}).get("ids", ())) != CANDIDATES
    ):
        raise TrendOptimizationError("trend candidate grid changed")
    runtime = payload.get("runtime_features", {})
    if (
        runtime.get("windows_days") != [7, 28]
        or runtime.get("exclude_abnormal_days") is not True
    ):
        raise TrendOptimizationError("robust trend runtime policy changed")
    boundary = payload.get("production_boundary", {})
    required_false = (
        "use_dataset_id_as_feature",
        "use_synthetic_calls_as_supervision",
        "include_model006_predictions",
        "select_threshold",
        "change_http_behavior",
    )
    if (
        any(boundary.get(key) is not False for key in required_false)
        or boundary.get("offline_only") is not True
    ):
        raise TrendOptimizationError("trend production boundary changed")
    return TrendOptimizationConfig(root, payload, config_path)


def load_trend_optimization_inputs(
    config: TrendOptimizationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    oof_path = _resolve(config.repository_root, config.payload["upstream"]["oof_path"])
    split_path = _resolve(config.repository_root, config.payload["split"]["path"])
    if (
        _sha256_file(oof_path) != str(config.payload["upstream"]["oof_sha256"])
        or _sha256_file(oof_path) != TREND_OOF_SHA256
    ):
        raise TrendOptimizationError("TREND-001 OOF drifted")
    if _sha256_file(split_path) != str(config.payload["split"]["sha256"]):
        raise TrendOptimizationError("DATA-007 split drifted")
    frame = pd.read_parquet(oof_path).reset_index(drop=True)
    required = {
        "branch",
        "dataset_id",
        "prediction_id",
        "canonical_row_index",
        "global_participant_id",
        "outer_fold",
        "target",
        "available",
        "reliability",
        "probability",
        *COMPONENTS,
    }
    if required - set(frame.columns) or set(frame["branch"]) != set(BRANCHES):
        raise TrendOptimizationError("TREND-001 OOF schema changed")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    assignments = pd.DataFrame(split["participant_assignments"])
    if assignments["global_participant_id"].duplicated().any():
        raise TrendOptimizationError("DATA-007 participant identity changed")
    frame = _attach_concordance(frame)
    protection = {
        "trend_oof_sha256": TREND_OOF_SHA256,
        "split_sha256": str(config.payload["split"]["sha256"]),
        "dataset_id_as_model_input": False,
        "synthetic_calls_as_supervision": False,
        "social_direct_s10_phq9_validation": False,
        "model006_online": False,
    }
    return frame, assignments.set_index("global_participant_id"), protection


def optimize_personal_trend(
    config: TrendOptimizationConfig,
    *,
    overwrite: bool = False,
    command: Sequence[str] = (),
) -> dict[str, Any]:
    frame, assignments, protection = load_trend_optimization_inputs(config)
    _check_outputs(config, overwrite=overwrite)
    oof_parts: list[pd.DataFrame] = []
    search_rows: list[dict[str, Any]] = []
    outer_rows: list[dict[str, Any]] = []
    branch_rows: list[dict[str, Any]] = []
    final_models: dict[str, TrendCandidateModel] = {}
    selected_global: dict[str, str] = {}
    for branch in BRANCHES:
        part = frame[frame["branch"].eq(branch)].copy().reset_index(drop=True)
        available = part["available"].astype(bool).to_numpy()
        target = part["target"].to_numpy(dtype=int)
        fold = part["outer_fold"].to_numpy(dtype=int)
        probabilities = {
            candidate: np.full(len(part), np.nan) for candidate in CANDIDATES
        }
        selections = np.full(len(part), "", dtype=object)
        for outer_fold in range(OUTER_FOLDS):
            outer_test = available & (fold == outer_fold)
            outer_train = available & ~outer_test
            positions = np.flatnonzero(outer_train)
            candidates: list[dict[str, Any]] = []
            for candidate in CANDIDATES:
                inner_prediction = np.full(len(positions), np.nan)
                for inner_fold in range(INNER_FOLDS):
                    validation = np.array(
                        [
                            _inner_fold(assignments, participant, outer_fold)
                            == inner_fold
                            for participant in part.loc[
                                positions, "global_participant_id"
                            ].astype(str)
                        ],
                        dtype=bool,
                    )
                    raw_train = _candidate_score(
                        part.iloc[positions[~validation]], branch, candidate
                    )
                    calibrator = _fit_calibrator(
                        raw_train, target[positions[~validation]]
                    )
                    raw_validation = _candidate_score(
                        part.iloc[positions[validation]], branch, candidate
                    )
                    inner_prediction[validation] = calibrator.predict_proba(
                        _logit(raw_validation)[:, None]
                    )[:, 1]
                metrics = _metrics(target[positions], inner_prediction)
                row = {
                    "branch": branch,
                    "outer_fold": outer_fold,
                    "candidate_id": candidate,
                    **metrics,
                }
                candidates.append(row)
            chosen = sorted(
                candidates,
                key=lambda row: (-row["auprc"], row["brier"], row["candidate_id"]),
            )[0]["candidate_id"]
            selections[outer_test] = chosen
            for row in candidates:
                row["selected"] = row["candidate_id"] == chosen
                search_rows.append(row)
            for candidate in CANDIDATES:
                raw_train = _candidate_score(part.iloc[positions], branch, candidate)
                calibrator = _fit_calibrator(raw_train, target[positions])
                raw_test = _candidate_score(part.loc[outer_test], branch, candidate)
                probabilities[candidate][outer_test] = calibrator.predict_proba(
                    _logit(raw_test)[:, None]
                )[:, 1]
            chosen_probability = probabilities[chosen][outer_test]
            outer_rows.append(
                {
                    "branch": branch,
                    "outer_fold": outer_fold,
                    "selected_candidate": chosen,
                    "row_count": int(outer_test.sum()),
                    "positive_rows": int(target[outer_test].sum()),
                    **_metrics(target[outer_test], chosen_probability),
                    "mean_reliability": float(
                        part.loc[outer_test, "optimized_reliability"].mean()
                    ),
                }
            )
        selected_probability = np.full(len(part), np.nan)
        for candidate in CANDIDATES:
            mask = selections == candidate
            selected_probability[mask] = probabilities[candidate][mask]
            part[f"probability__{candidate}"] = probabilities[candidate]
        part["selected_candidate"] = selections
        part["selected_probability"] = selected_probability
        oof_parts.append(part)
        passed = pd.DataFrame([row for row in search_rows if row["branch"] == branch])
        counts = passed.groupby("candidate_id")["selected"].sum().to_dict()
        selected = sorted(counts, key=lambda value: (-int(counts[value]), value))[0]
        selected_global[branch] = selected
        available_part = part[available]
        branch_rows.append(
            {
                "branch": branch,
                "candidate_id": selected,
                "available_rows": int(available.sum()),
                "unavailable_rows": int((~available).sum()),
                "mean_reliability": float(
                    available_part["optimized_reliability"].mean()
                ),
                **_metrics(
                    available_part["target"], available_part["selected_probability"]
                ),
            }
        )
        raw_all = _candidate_score(available_part, branch, selected)
        final_models[branch] = TrendCandidateModel(
            branch,
            selected,
            _fit_calibrator(raw_all, available_part["target"].to_numpy(dtype=int)),
        )
    oof = pd.concat(oof_parts, ignore_index=True)
    playback = _playback_stability(oof)
    reliability = _reliability_report(oof)
    overall = pd.DataFrame(branch_rows)
    summary = {
        "task_id": TASK_ID,
        "run_id": RUN_ID,
        "status": "pass",
        "strict_oof": True,
        "selected_candidates": selected_global,
        "promotion_status": "report_only_candidate_not_production",
        "social_evidence_scope": "engineering_proxy_no_direct_s10_phq9_validation",
        "synthetic_calls_as_supervision": False,
        "runtime_features": [
            "median_mad_7d",
            "median_mad_28d",
            "ewma",
            "theil_sen",
            "change_point",
            "weekday_weekend_rhythm",
            "abnormal_day_exclusion",
        ],
    }
    _publish(
        config,
        final_models,
        pd.DataFrame(search_rows),
        oof,
        overall,
        pd.DataFrame(outer_rows),
        playback,
        reliability,
        summary,
        protection,
        command,
    )
    return {
        "status": "pass",
        "task_id": TASK_ID,
        "run_id": RUN_ID,
        "selected_candidates": selected_global,
        "model_path": str(config.model_path),
        "report_directory": str(config.report_directory),
    }


def _attach_concordance(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["optimized_reliability"] = output["reliability"].clip(0.0, 1.0)
    output.loc[output["branch"].eq("social"), "optimized_reliability"] = output.loc[
        output["branch"].eq("social"), "optimized_reliability"
    ].clip(upper=0.25)
    keys = ["dataset_id", "global_participant_id", "canonical_row_index", "outer_fold"]
    strong = output[output["branch"].isin(["activity", "sleep"])].pivot_table(
        index=keys, columns="branch", values="robust_deviation", aggfunc="first"
    )
    strong["activity_sleep_concordance"] = (
        (strong.get("activity", pd.Series(index=strong.index, dtype=float)) >= 0.5)
        & (strong.get("sleep", pd.Series(index=strong.index, dtype=float)) >= 0.5)
    ).astype(float)
    output = output.merge(
        strong[["activity_sleep_concordance"]].reset_index(), on=keys, how="left"
    )
    output["activity_sleep_concordance"] = output["activity_sleep_concordance"].fillna(
        0.0
    )
    strong_mask = output["branch"].isin(["activity", "sleep"])
    output.loc[strong_mask, "optimized_reliability"] = np.clip(
        output.loc[strong_mask, "optimized_reliability"]
        + 0.15 * output.loc[strong_mask, "activity_sleep_concordance"],
        0.0,
        1.0,
    )
    return output


def _candidate_score(frame: pd.DataFrame, branch: str, candidate: str) -> np.ndarray:
    if candidate not in CANDIDATES:
        raise TrendOptimizationError(f"unknown trend candidate: {candidate}")
    values = {
        name: frame[name].fillna(0.0).to_numpy(dtype=float) for name in COMPONENTS
    }
    reliability = frame["optimized_reliability"].fillna(0.0).to_numpy(dtype=float)
    if candidate == "baseline_probability":
        raw = frame["probability"].fillna(0.5).to_numpy(dtype=float)
    elif candidate == "robust_dual_scale":
        evidence = (
            0.35 * values["robust_deviation"]
            + 0.20 * values["risk_slope"]
            + 0.15 * values["change_point"]
            + 0.15 * values["persistence"]
            + 0.10 * values["isolation_forest"]
            + 0.05 * values["history_coverage"]
        )
        raw = 0.5 + reliability * (evidence - 0.5)
    elif candidate == "ewma_theilsen_change_point":
        evidence = (
            0.25 * values["robust_deviation"]
            + 0.25 * values["risk_slope"]
            + 0.20 * values["change_point"]
            + 0.20 * values["persistence"]
            + 0.10 * values["history_coverage"]
        )
        raw = 0.5 + reliability * (evidence - 0.5)
    else:
        concordance = (
            frame["activity_sleep_concordance"].fillna(0.0).to_numpy(dtype=float)
        )
        evidence = (
            0.30 * values["robust_deviation"]
            + 0.20 * values["risk_slope"]
            + 0.20 * values["change_point"]
            + 0.15 * values["persistence"]
            + 0.10 * values["isolation_forest"]
            + 0.05 * concordance
        )
        if branch == "social":
            evidence = 0.5 + 0.25 * (evidence - 0.5)
        raw = 0.5 + reliability * (evidence - 0.5)
    return np.clip(raw, EPSILON, 1.0 - EPSILON)


def _fit_calibrator(raw: np.ndarray, target: np.ndarray) -> LogisticRegression:
    y = np.asarray(target, dtype=int)
    if set(np.unique(y)) != {0, 1}:
        raise TrendOptimizationError("trend calibration requires both classes")
    model = LogisticRegression(
        C=1.0, solver="lbfgs", max_iter=1000, random_state=RANDOM_SEED
    )
    model.fit(_logit(raw)[:, None], y)
    return model


def _metrics(target: Sequence[int], probability: Sequence[float]) -> dict[str, float]:
    y = np.asarray(target, dtype=int)
    p = np.asarray(probability, dtype=float)
    if (
        len(y) == 0
        or len(y) != len(p)
        or not np.isfinite(p).all()
        or set(np.unique(y)) != {0, 1}
    ):
        raise TrendOptimizationError("trend metric input is invalid")
    return {
        "auprc": float(average_precision_score(y, p)),
        "auroc": float(roc_auc_score(y, p)),
        "brier": float(np.mean((p - y) ** 2)),
    }


def _playback_stability(oof: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (branch, participant), group in oof[oof["available"]].groupby(
        ["branch", "global_participant_id"], sort=True
    ):
        ordered = group.sort_values("canonical_row_index", kind="stable")
        values = ordered["selected_probability"].to_numpy(dtype=float)
        smoothed = pd.Series(values).rolling(3, min_periods=1).mean().to_numpy()
        rows.append(
            {
                "branch": branch,
                "global_participant_id": participant,
                "day_count": len(values),
                "mean_absolute_raw_step": float(np.abs(np.diff(values)).mean())
                if len(values) > 1
                else 0.0,
                "mean_absolute_smoothed_step": float(np.abs(np.diff(smoothed)).mean())
                if len(smoothed) > 1
                else 0.0,
                "smoothing_non_increasing_jitter": bool(
                    (np.abs(np.diff(smoothed)).mean() if len(smoothed) > 1 else 0.0)
                    <= (np.abs(np.diff(values)).mean() if len(values) > 1 else 0.0)
                    + 1e-12
                ),
                "evidence_interruptions": int(ordered["available"].eq(False).sum()),
            }
        )
    return pd.DataFrame(rows)


def _reliability_report(oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for branch, group in oof.groupby("branch", sort=True):
        available = group[group["available"]]
        rows.append(
            {
                "branch": branch,
                "row_count": len(group),
                "available_rows": len(available),
                "availability_rate": float(len(available) / len(group)),
                "mean_original_reliability": float(available["reliability"].mean()),
                "mean_optimized_reliability": float(
                    available["optimized_reliability"].mean()
                ),
                "maximum_optimized_reliability": float(
                    available["optimized_reliability"].max()
                ),
                "direct_s10_phq9_validation": False if branch == "social" else None,
            }
        )
    return pd.DataFrame(rows)


def _inner_fold(assignments: pd.DataFrame, participant: str, outer_fold: int) -> int:
    mapping = assignments.loc[str(participant), "inner_validation_fold_by_outer_fold"]
    value = mapping.get(str(outer_fold))
    if value is None:
        raise TrendOptimizationError("outer-test participant entered trend inner fold")
    return int(value)


def _robust_deviation(current: float, values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    center = float(np.median(values))
    scale = max(_mad(values), 0.05)
    return float(np.clip(abs(current - center) / (2.0 * scale), 0.0, 1.0))


def _mad(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    center = np.median(values)
    return float(1.4826 * np.median(np.abs(values - center)))


def _ewma(values: np.ndarray, *, alpha: float) -> float:
    current = float(values[0])
    for value in values[1:]:
        current = alpha * float(value) + (1.0 - alpha) * current
    return current


def _theil_sen(x: np.ndarray, y: np.ndarray) -> float:
    slopes = [
        (y[j] - y[i]) / (x[j] - x[i])
        for i in range(len(x))
        for j in range(i + 1, len(x))
        if x[j] != x[i]
    ]
    return float(np.median(slopes)) if slopes else 0.0


def _change_point(values: np.ndarray) -> float:
    if len(values) < 6:
        return 0.0
    earlier = values[:-3]
    recent = values[-3:]
    scale = max(_mad(values), 0.05)
    return float(
        np.clip(
            abs(float(np.median(recent)) - float(np.median(earlier))) / (2.0 * scale),
            0.0,
            1.0,
        )
    )


def _publish(
    config: TrendOptimizationConfig,
    models: Mapping[str, TrendCandidateModel],
    search: pd.DataFrame,
    oof: pd.DataFrame,
    overall: pd.DataFrame,
    outer: pd.DataFrame,
    playback: pd.DataFrame,
    reliability: pd.DataFrame,
    summary: Mapping[str, Any],
    protection: Mapping[str, Any],
    command: Sequence[str],
) -> None:
    report = config.report_directory
    report.mkdir(parents=True, exist_ok=True)
    config.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(models), config.model_path, compress=3)
    manifest = {
        "task_id": TASK_ID,
        "run_id": RUN_ID,
        "model_version": MODEL_VERSION,
        "strict_oof": True,
        "production": False,
        "selected_candidates": dict(summary["selected_candidates"]),
        "runtime_features": list(summary["runtime_features"]),
        "dataset_id_as_model_input": False,
        "synthetic_calls_as_supervision": False,
        "social_direct_s10_phq9_validation": False,
        "model006_online": False,
        "model_sha256": _sha256_file(config.model_path),
    }
    _write_json(config.manifest_path, manifest)
    for name, value in (
        ("candidate_search.parquet", search),
        ("oof_predictions.parquet", oof),
        ("overall_metrics.parquet", overall),
        ("outer_fold_stability.parquet", outer),
        ("playback_stability.parquet", playback),
        ("branch_reliability.parquet", reliability),
    ):
        value.to_parquet(report / name, index=False)
    _write_json(report / "summary.json", summary)
    _write_json(report / "upstream_protection.json", protection)
    _write_json(report / "config.json", config.payload)
    _write_json(
        report / "run.json",
        {
            "task_id": TASK_ID,
            "run_id": RUN_ID,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "command": list(command),
        },
    )
    _write_json(
        report / "model_card.json",
        {
            "task_id": TASK_ID,
            "run_id": RUN_ID,
            "production_status": "offline_candidate_not_production",
            "social_scope": summary["social_evidence_scope"],
            "limitations": [
                "public OOF proxy",
                "no real S10-PHQ-9 paired longitudinal labels",
            ],
        },
    )
    (report / "model_card.md").write_text(
        "# OPT-TREND-001 candidate\n\nRobust PersonalTrend candidate. Social trend remains a low-reliability engineering proxy without direct S10-PHQ-9 validation.\n",
        encoding="utf-8",
    )
    artifacts = [
        {"path": path.name, "bytes": path.stat().st_size, "sha256": _sha256_file(path)}
        for path in sorted(report.iterdir(), key=lambda item: item.name.encode("utf-8"))
        if path.is_file() and path.name != "artifacts.json"
    ]
    _write_json(
        report / "artifacts.json",
        {"version": "mood-social-opt-trend-artifacts-v1", "artifacts": artifacts},
    )
    sums = [
        f"{_sha256_file(path)}  {path.name}"
        for path in sorted(
            config.model_path.parent.iterdir(),
            key=lambda item: item.name.encode("utf-8"),
        )
        if path.is_file() and path.name != "SHA256SUMS"
    ]
    (config.model_path.parent / "SHA256SUMS").write_text(
        "\n".join(sums) + "\n", encoding="utf-8"
    )


def _check_outputs(config: TrendOptimizationConfig, *, overwrite: bool) -> None:
    if not overwrite and any(
        path.exists()
        for path in (config.report_directory, config.model_path, config.manifest_path)
    ):
        raise TrendOptimizationError("OPT-TREND-001 output exists; use --overwrite")


def _logit(value: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(value, dtype=float), EPSILON, 1.0 - EPSILON)
    return np.log(p / (1.0 - p))


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, indent=2, default=_json_default
        )
        + "\n",
        encoding="utf-8",
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


__all__ = [
    "BRANCHES",
    "CANDIDATES",
    "RUN_ID",
    "TASK_ID",
    "RobustTrendFeatures",
    "TrendCandidateModel",
    "TrendOptimizationConfig",
    "TrendOptimizationError",
    "compute_robust_trend_features",
    "load_trend_optimization_config",
    "load_trend_optimization_inputs",
    "optimize_personal_trend",
]
