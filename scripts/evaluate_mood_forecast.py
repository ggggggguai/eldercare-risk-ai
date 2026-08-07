"""Evaluate and document the strict V3.4 forecast outer OOF results."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd

ALGORITHM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ALGORITHM_ROOT / "src"))

from elderly_monitoring.modules.mental_health.mood_social.forecast_experiment import (  # noqa: E402
    sha256_file,
)
from elderly_monitoring.modules.mental_health.mood_social.forecast_modeling import (  # noqa: E402
    MODEL_FAMILIES,
    SEED,
    calibration_bins,
    participant_equal_weights,
    weighted_metrics,
)


OUTPUT_ROOT = (
    ALGORITHM_ROOT / "artifacts" / "mental_health" / "mood_social" / "forecast_v3.4"
)
REPORT_ROOT = OUTPUT_ROOT / "reports"
FEATURE_NAMES = tuple(
    json.loads((OUTPUT_ROOT / "raw_feature_names.json").read_text(encoding="utf-8"))[
        "raw_feature_names"
    ]
)
METRIC_NAMES = (
    "auprc",
    "auroc",
    "macro_f1",
    "sensitivity",
    "specificity",
    "brier",
    "ece",
)
EXPECTED_COUNTS = {"forecast_1m": 9393, "forecast_2m": 9280}
BOOTSTRAP_REPETITIONS = 2000


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    return value


def _validate_oof(oof: pd.DataFrame) -> None:
    required = {
        "participant_id",
        "global_participant_id",
        "target_window_id",
        "task_id",
        "model_family",
        "outer_fold_id",
        "y_true",
        "p_raw",
        "p_calibrated",
        "fold_threshold",
        "train_sample_weight_policy",
        "evaluation_weight",
    }
    if not required.issubset(oof.columns):
        raise RuntimeError(
            f"OOF table is missing columns: {sorted(required - set(oof.columns))}"
        )
    for task, expected in EXPECTED_COUNTS.items():
        for family in MODEL_FAMILIES:
            subset = oof[oof.task_id.eq(task) & oof.model_family.eq(family)]
            if len(subset) != expected or subset.target_window_id.nunique() != expected:
                raise RuntimeError(f"{task}/{family} OOF coverage does not match")
            if set(subset.outer_fold_id.unique()) != set(range(5)):
                raise RuntimeError(f"{task}/{family} does not cover five outer folds")
            if not np.isclose(subset.evaluation_weight.sum(), 1.0):
                raise RuntimeError(
                    f"{task}/{family} evaluation weights do not sum to one"
                )
            if (
                not subset.p_raw.between(0.0, 1.0).all()
                or not subset.p_calibrated.between(0.0, 1.0).all()
            ):
                raise RuntimeError(f"{task}/{family} contains invalid probabilities")


def _metrics(frame: pd.DataFrame, *, participant_equal: bool) -> dict[str, float]:
    weights = (
        participant_equal_weights(frame)
        if participant_equal
        else np.full(len(frame), 1.0 / len(frame))
    )
    result = weighted_metrics(
        frame.y_true,
        frame.p_raw,
        frame.p_calibrated,
        frame.fold_threshold.to_numpy(),
        weights,
    )
    calibrated_discrimination = weighted_metrics(
        frame.y_true,
        frame.p_calibrated,
        frame.p_calibrated,
        frame.fold_threshold.to_numpy(),
        weights,
    )
    result["auprc_calibrated"] = calibrated_discrimination["auprc"]
    result["auroc_calibrated"] = calibrated_discrimination["auroc"]
    result.pop("threshold", None)
    return result


def _bootstrap(
    frame: pd.DataFrame, repetitions: int = BOOTSTRAP_REPETITIONS
) -> dict[str, Any]:
    participants = sorted(frame.global_participant_id.unique().tolist())
    groups = {
        participant: np.flatnonzero(
            frame.global_participant_id.to_numpy() == participant
        )
        for participant in participants
    }
    y = frame.y_true.to_numpy(dtype="int8")
    raw = frame.p_raw.to_numpy(dtype="float64")
    calibrated = frame.p_calibrated.to_numpy(dtype="float64")
    thresholds = frame.fold_threshold.to_numpy(dtype="float64")
    rng = np.random.Generator(np.random.PCG64(SEED))
    values = {name: [] for name in METRIC_NAMES}
    degenerate = 0
    while len(values["auprc"]) < repetitions:
        selected = rng.integers(0, len(participants), size=len(participants))
        index_parts: list[np.ndarray] = []
        weight_parts: list[np.ndarray] = []
        for position in selected:
            indices = groups[participants[int(position)]]
            index_parts.append(indices)
            weight_parts.append(
                np.full(len(indices), 1.0 / (len(participants) * len(indices)))
            )
        indices = np.concatenate(index_parts)
        if np.unique(y[indices]).size < 2:
            degenerate += 1
            continue
        weights = np.concatenate(weight_parts)
        metrics = weighted_metrics(
            y[indices], raw[indices], calibrated[indices], thresholds[indices], weights
        )
        for name in METRIC_NAMES:
            values[name].append(metrics[name])
    return {
        "method": "participant_cluster_percentile",
        "rng": "numpy_pcg64",
        "seed": SEED,
        "valid_repetitions": repetitions,
        "degenerate_repetitions": degenerate,
        "confidence_level": 0.95,
        "intervals": {
            name: {
                "lower": float(np.percentile(series, 2.5)),
                "upper": float(np.percentile(series, 97.5)),
            }
            for name, series in values.items()
        },
    }


def _paired_frame(oof: pd.DataFrame, family: str) -> pd.DataFrame:
    common = pd.read_parquet(OUTPUT_ROOT / "common_target_windows.parquet")[
        ["global_participant_id", "target_window_id"]
    ]
    parts = []
    for task in ("forecast_1m", "forecast_2m"):
        frame = oof[oof.task_id.eq(task) & oof.model_family.eq(family)].merge(
            common,
            on=["global_participant_id", "target_window_id"],
            how="inner",
            validate="one_to_one",
        )
        parts.append(frame)
    paired = parts[0].merge(
        parts[1],
        on=["global_participant_id", "target_window_id"],
        how="inner",
        validate="one_to_one",
        suffixes=("_1m", "_2m"),
    )
    if len(paired) != 8494 or not paired.y_true_1m.equals(paired.y_true_2m):
        raise RuntimeError(f"{family} paired common-window contract failed")
    return paired


def _paired_bootstrap(
    paired: pd.DataFrame, repetitions: int = BOOTSTRAP_REPETITIONS
) -> dict[str, Any]:
    participants = sorted(paired.global_participant_id.unique().tolist())
    participant_values = paired.global_participant_id.to_numpy()
    groups = {
        participant: np.flatnonzero(participant_values == participant)
        for participant in participants
    }
    rng = np.random.Generator(np.random.PCG64(SEED))
    differences = {name: [] for name in METRIC_NAMES}
    degenerate = 0
    while len(differences["auprc"]) < repetitions:
        selected = rng.integers(0, len(participants), size=len(participants))
        index_parts: list[np.ndarray] = []
        weight_parts: list[np.ndarray] = []
        for position in selected:
            indices = groups[participants[int(position)]]
            index_parts.append(indices)
            weight_parts.append(
                np.full(len(indices), 1.0 / (len(participants) * len(indices)))
            )
        indices = np.concatenate(index_parts)
        y = paired.y_true_1m.to_numpy(dtype="int8")[indices]
        if np.unique(y).size < 2:
            degenerate += 1
            continue
        weights = np.concatenate(weight_parts)
        horizon_metrics = []
        for suffix in ("1m", "2m"):
            horizon_metrics.append(
                weighted_metrics(
                    y,
                    paired[f"p_raw_{suffix}"].to_numpy()[indices],
                    paired[f"p_calibrated_{suffix}"].to_numpy()[indices],
                    paired[f"fold_threshold_{suffix}"].to_numpy()[indices],
                    weights,
                )
            )
        for name in METRIC_NAMES:
            differences[name].append(
                horizon_metrics[0][name] - horizon_metrics[1][name]
            )
    point_weights = participant_equal_weights(
        paired.rename(columns={"y_true_1m": "y_true"})
    )
    point = []
    for suffix in ("1m", "2m"):
        point.append(
            weighted_metrics(
                paired.y_true_1m,
                paired[f"p_raw_{suffix}"],
                paired[f"p_calibrated_{suffix}"],
                paired[f"fold_threshold_{suffix}"],
                point_weights,
            )
        )
    return {
        "common_window_count": len(paired),
        "participant_count": len(participants),
        "direction": "forecast_1m_minus_forecast_2m",
        "valid_repetitions": repetitions,
        "degenerate_repetitions": degenerate,
        "point_differences": {
            name: point[0][name] - point[1][name] for name in METRIC_NAMES
        },
        "intervals": {
            name: {
                "lower": float(np.percentile(series, 2.5)),
                "upper": float(np.percentile(series, 97.5)),
            }
            for name, series in differences.items()
        },
    }


def _raw_matrix(frame: pd.DataFrame) -> np.ndarray:
    values = (
        frame.loc[:, list(FEATURE_NAMES)]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype="float64", copy=True)
    )
    values[~np.isfinite(values)] = np.nan
    return values


def _explanations(task: str, family: str) -> pd.DataFrame:
    samples = pd.read_parquet(OUTPUT_ROOT / task / "samples.parquet")
    sums: np.ndarray | None = None
    row_count = 0
    names: Sequence[str] = FEATURE_NAMES
    for outer_fold in range(5):
        bundle = joblib.load(
            OUTPUT_ROOT / task / family / "folds" / f"outer_fold_{outer_fold}.joblib"
        )
        test = samples[samples.outer_fold_id.eq(outer_fold)]
        if family == "lightgbm":
            contributions = bundle["estimator"].booster_.predict(
                _raw_matrix(test), pred_contrib=True
            )[:, :-1]
        elif family == "elasticnet_logistic":
            matrix = bundle["preprocessor"].transform(test)
            contributions = matrix * bundle["estimator"].coef_[0]
            names = bundle["preprocessor"].derived_feature_names
        else:
            return pd.DataFrame(
                columns=[
                    "task_id",
                    "model_family",
                    "feature_name",
                    "mean_absolute_contribution",
                    "mean_signed_contribution",
                ]
            )
        absolute = np.abs(contributions).sum(axis=0)
        signed = contributions.sum(axis=0)
        current = np.stack([absolute, signed], axis=1)
        sums = current if sums is None else sums + current
        row_count += len(test)
    if sums is None:
        raise RuntimeError("no explanation rows were generated")
    return pd.DataFrame(
        {
            "task_id": task,
            "model_family": family,
            "feature_name": list(names),
            "mean_absolute_contribution": sums[:, 0] / row_count,
            "mean_signed_contribution": sums[:, 1] / row_count,
        }
    ).sort_values("mean_absolute_contribution", ascending=False, kind="mergesort")


def _v33_reference(oof: pd.DataFrame) -> dict[str, Any]:
    reference_path = (
        ALGORITHM_ROOT
        / "reports"
        / "mental_health"
        / "mood_social"
        / "MH-20260804-021"
        / "oof_predictions.parquet"
    )
    canonical_path = (
        ALGORITHM_ROOT
        / "data"
        / "processed"
        / "mental_health"
        / "mood_social"
        / "v3.3.3"
        / "psyche_d"
        / "canonical_psyche_d.parquet"
    )
    if not reference_path.exists() or not canonical_path.exists():
        return {
            "status": "unavailable",
            "reason": "frozen V3.3 reference artifacts are absent",
        }
    reference = pd.read_parquet(reference_path)
    reference = reference[reference.dataset_id.eq("psyche_d")].copy()
    canonical = pd.read_parquet(
        canonical_path,
        columns=["nominal_month"],
    )
    canonical = canonical.reset_index(names="canonical_row_index")
    reference = reference.merge(
        canonical, on="canonical_row_index", how="inner", validate="one_to_one"
    )
    reference["target_window_id"] = (
        reference.global_participant_id + "::m" + reference.nominal_month.astype(str)
    )
    result: dict[str, Any] = {
        "status": "available",
        "source": str(reference_path.relative_to(ALGORITHM_ROOT)).replace("\\", "/"),
        "tasks": {},
    }
    for task in EXPECTED_COUNTS:
        target_ids = set(oof[oof.task_id.eq(task)].target_window_id)
        selected = reference[reference.target_window_id.isin(target_ids)].copy()
        selected = selected.rename(columns={"binary_target": "y_true"})
        weights = participant_equal_weights(selected)
        columns = {
            "current_online_baseline": "baseline_probability",
            "corrected_calibration_reference": "offline_reference_probability",
        }
        metrics: dict[str, Any] = {}
        for name, column in columns.items():
            values = weighted_metrics(
                selected.y_true, selected[column], selected[column], 0.5, weights
            )
            metrics[name] = {
                key: values[key] for key in ("auprc", "auroc", "brier", "ece")
            }
        result["tasks"][task] = {
            "matched_target_window_count": len(selected),
            "participant_count": int(selected.global_participant_id.nunique()),
            "metrics": metrics,
            "comparison_scope": "same target windows; V3.3 current-state context only",
        }
    return result


def _model_card(task: str, family: str, payload: dict[str, Any]) -> str:
    metric = payload["primary_metrics"]
    return f"""# {task} / {family} model card

## Purpose

Offline PSYCHE-D nominal-month conditional experiment for estimating a later higher PHQ-9 symptom-pattern label. This is not a diagnosis and has no product decision authority.

## Data and method

- Frozen raw inputs: 27 activity and sleep fields from the earlier nominal month.
- Prediction unit: participant-isolated target window.
- Evaluation: strict five-fold outer OOF with participant-equal primary weighting.
- Calibration: {"none; raw prior retained" if family == "dummy" else "fold-local Isotonic Regression fitted on inner OOF only"}.
- Status: experimental, offline_only, shadow_only, product_visible=false.

## Primary OOF result

- AUPRC: {metric["auprc"]:.6f}
- AUROC: {metric["auroc"]:.6f}
- Macro-F1: {metric["macro_f1"]:.6f}
- Sensitivity: {metric["sensitivity"]:.6f}
- Specificity: {metric["specificity"]:.6f}
- Brier: {metric["brier"]:.6f}
- ECE: {metric["ece"]:.6f}

## Limitations

The public matrix has design-level nominal-month evidence rather than row-level natural timestamps. PSYCHE-D is not an elderly-China validation cohort, and its Fitbit-derived fields do not establish transfer to the project's camera or sleep-monitor device domain. Feature contributions describe model association, not causation.
"""


def run_evaluation(repetitions: int = BOOTSTRAP_REPETITIONS) -> dict[str, Any]:
    oof_path = OUTPUT_ROOT / "oof_predictions.parquet"
    if not oof_path.exists():
        raise RuntimeError("training OOF artifact is missing")
    oof = pd.read_parquet(oof_path)
    _validate_oof(oof)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {
        "schema_version": "mood-social-forecast-v3.4-metrics-v1",
        "experiment_version": "mood-social-forecast-v3.4",
        "bootstrap_repetitions": repetitions,
        "tasks": {},
        "paired_comparisons": {},
        "release_status": "experimental",
        "execution_mode": "offline_only",
        "decision_authority": "shadow_only",
        "product_visible": False,
    }
    calibration_rows: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    explanation_rows: list[pd.DataFrame] = []
    failure_rows: list[pd.DataFrame] = []
    extreme_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    for task in EXPECTED_COUNTS:
        samples = pd.read_parquet(OUTPUT_ROOT / task / "samples.parquet")
        for feature in FEATURE_NAMES:
            missing_rows.append(
                {
                    "task_id": task,
                    "feature_name": feature,
                    "missing_rate": float(samples[feature].isna().mean()),
                    "missing_count": int(samples[feature].isna().sum()),
                    "sample_count": len(samples),
                }
            )
        task_result: dict[str, Any] = {
            "sample_count": EXPECTED_COUNTS[task],
            "participant_count": int(samples.global_participant_id.nunique()),
            "models": {},
        }
        for family in MODEL_FAMILIES:
            print(
                json.dumps(
                    {
                        "event": "evaluation_model_start",
                        "task_id": task,
                        "model_family": family,
                        "bootstrap_repetitions": repetitions,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )
            frame = oof[oof.task_id.eq(task) & oof.model_family.eq(family)].copy()
            primary = _metrics(frame, participant_equal=True)
            sensitivity = _metrics(frame, participant_equal=False)
            bootstrap = _bootstrap(frame, repetitions)
            model_result = {
                "primary_metrics": primary,
                "window_unweighted_sensitivity_metrics": sensitivity,
                "bootstrap": bootstrap,
            }
            task_result["models"][family] = model_result
            weights = participant_equal_weights(frame)
            curve = pd.DataFrame(
                calibration_bins(frame.y_true, frame.p_calibrated, weights)
            )
            curve.insert(0, "model_family", family)
            curve.insert(0, "task_id", task)
            calibration_rows.append(curve)
            for fold in range(5):
                fold_frame = frame[frame.outer_fold_id.eq(fold)]
                fold_rows.append(
                    {
                        "task_id": task,
                        "model_family": family,
                        "outer_fold_id": fold,
                        "sample_count": len(fold_frame),
                        "participant_count": int(
                            fold_frame.global_participant_id.nunique()
                        ),
                        "positive_window_count": int(fold_frame.y_true.sum()),
                        "positive_participant_count": int(
                            fold_frame.loc[
                                fold_frame.y_true.eq(1), "global_participant_id"
                            ].nunique()
                        ),
                        **_metrics(fold_frame, participant_equal=True),
                    }
                )
            predicted = (frame.p_calibrated >= frame.fold_threshold).astype("int8")
            failed = frame[predicted.ne(frame.y_true)].copy()
            failed["failure_type"] = np.where(
                failed.y_true.eq(1), "false_negative", "false_positive"
            )
            failed["absolute_probability_error"] = np.abs(
                failed.p_calibrated - failed.y_true
            )
            failure_rows.append(failed)
            for region, mask in {
                "low_le_0_05": frame.p_calibrated.le(0.05),
                "high_ge_0_95": frame.p_calibrated.ge(0.95),
            }.items():
                subset = frame[mask]
                extreme_rows.append(
                    {
                        "task_id": task,
                        "model_family": family,
                        "region": region,
                        "sample_count": len(subset),
                        "participant_count": int(
                            subset.global_participant_id.nunique()
                        ),
                        "mean_probability": float(subset.p_calibrated.mean())
                        if len(subset)
                        else None,
                        "observed_positive_rate": float(
                            np.average(
                                subset.y_true, weights=participant_equal_weights(subset)
                            )
                        )
                        if len(subset)
                        else None,
                    }
                )
            if family != "dummy":
                explanation_rows.append(_explanations(task, family))
            card_payload = {
                "task_id": task,
                "model_family": family,
                **model_result,
                "release_status": "experimental",
                "execution_mode": "offline_only",
                "decision_authority": "shadow_only",
                "product_visible": False,
            }
            _write_json(
                OUTPUT_ROOT / "model_cards" / f"{task}_{family}.json",
                _json_safe(card_payload),
            )
            (OUTPUT_ROOT / "model_cards" / f"{task}_{family}.md").write_text(
                _model_card(task, family, model_result), encoding="utf-8"
            )
            print(
                json.dumps(
                    {
                        "event": "evaluation_model_complete",
                        "task_id": task,
                        "model_family": family,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )
        results["tasks"][task] = task_result
    for family in MODEL_FAMILIES:
        results["paired_comparisons"][family] = _paired_bootstrap(
            _paired_frame(oof, family), repetitions
        )
        print(
            json.dumps(
                {
                    "event": "paired_comparison_complete",
                    "model_family": family,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
    results["v3_3_current_state_reference"] = _v33_reference(oof)
    pd.concat(calibration_rows, ignore_index=True).to_parquet(
        OUTPUT_ROOT / "calibration_curves.parquet", index=False
    )
    pd.DataFrame(fold_rows).to_parquet(
        OUTPUT_ROOT / "fold_metrics.parquet", index=False
    )
    pd.DataFrame(missing_rows).to_parquet(
        OUTPUT_ROOT / "feature_missingness.parquet", index=False
    )
    pd.DataFrame(extreme_rows).to_parquet(
        OUTPUT_ROOT / "extreme_calibration_analysis.parquet", index=False
    )
    pd.concat(failure_rows, ignore_index=True).sort_values(
        "absolute_probability_error", ascending=False
    ).to_parquet(OUTPUT_ROOT / "failure_samples.parquet", index=False)
    pd.concat(explanation_rows, ignore_index=True).to_parquet(
        OUTPUT_ROOT / "feature_contributions.parquet", index=False
    )
    _write_json(OUTPUT_ROOT / "metrics.json", _json_safe(results))
    _write_reports(results)
    _write_experiment_manifest(results)
    return results


def _write_reports(results: dict[str, Any]) -> None:
    lines = [
        "# V3.4 Forecast 离线实验结果",
        "",
        "本报告只描述 PSYCHE-D 名义月份条件实验，不构成医学诊断，也不代表项目目标老人或设备域已经验证。结果保持 `experimental / offline_only / shadow_only`，不进入产品。",
        "",
        "## 严格折外结果",
        "",
        "| 任务 | 模型 | 样本/参与者 | AUPRC（95%区间） | AUROC | Macro-F1 | Sensitivity | Specificity | Brier | ECE |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for task, task_result in results["tasks"].items():
        for family, model in task_result["models"].items():
            metric = model["primary_metrics"]
            interval = model["bootstrap"]["intervals"]["auprc"]
            lines.append(
                f"| {task} | {family} | {task_result['sample_count']}/{task_result['participant_count']} | {metric['auprc']:.4f} [{interval['lower']:.4f}, {interval['upper']:.4f}] | {metric['auroc']:.4f} | {metric['macro_f1']:.4f} | {metric['sensitivity']:.4f} | {metric['specificity']:.4f} | {metric['brier']:.4f} | {metric['ece']:.4f} |"
            )
    lines += [
        "",
        "主结果按参与者等权，AUPRC/AUROC 使用未校准概率，Brier/ECE 使用折内校准后的概率；分类指标使用各外层折在内层 OOF 上冻结的阈值。窗口未加权结果、每折结果和置信区间保存在机器可读产物中。",
        "",
        "## 两个 horizon 的配对比较",
        "",
    ]
    for family, comparison in results["paired_comparisons"].items():
        diff = comparison["point_differences"]
        auprc_interval = comparison["intervals"]["auprc"]
        lines.append(
            f"- {family}: 共同窗口 {comparison['common_window_count']}；AUPRC 差值 {diff['auprc']:.4f}（95%区间 [{auprc_interval['lower']:.4f}, {auprc_interval['upper']:.4f}]），AUROC 差值 {diff['auroc']:.4f}，Brier 差值 {diff['brier']:.4f}。差值方向均为 forecast_1m - forecast_2m。"
        )
    lines += [
        "",
        "## V3.3 当前状态对照",
        "",
    ]
    reference = results.get("v3_3_current_state_reference", {})
    if reference.get("status") == "available":
        lines.append(
            "V3.3 对照仅在相同目标窗口上按可匹配行计算，因 V3.3 旧 OOF 有少量缺行，匹配数略少于 V3.4 全量；它是当前状态上下文，不是 V3.4 的调参集。"
        )
        for task, item in reference["tasks"].items():
            for model_name, metric in item["metrics"].items():
                lines.append(
                    f"- {task} / {model_name}：匹配窗口 {item['matched_target_window_count']}，AUPRC {metric['auprc']:.4f}，AUROC {metric['auroc']:.4f}，Brier {metric['brier']:.4f}，ECE {metric['ece']:.4f}。"
                )
    else:
        lines.append("V3.3 对照产物不可用，未进行跨版本点估计比较。")
    lines += [
        "",
        "## 限制",
        "",
        "- 公开矩阵只提供设计级名义月份关系，没有逐行真实日期。",
        "- PSYCHE-D 不是中国老年人专门验证队列。",
        "- Fitbit 派生字段不能直接证明摄像头与睡眠仪设备域迁移效果。",
        "- PHQ-9 阈值仅作为公开数据监督标签，不是医学诊断。",
        "- 特征贡献只说明模型关联，不表示因果关系。",
        "",
    ]
    (OUTPUT_ROOT / "V3.4结果报告.md").write_text("\n".join(lines), encoding="utf-8")


def _write_experiment_manifest(results: dict[str, Any]) -> None:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ALGORITHM_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
    inventory = {}
    for path in sorted(
        p
        for p in OUTPUT_ROOT.rglob("*")
        if p.is_file() and p.name not in {"SHA256SUMS", "experiment_manifest.json"}
    ):
        inventory[path.relative_to(OUTPUT_ROOT).as_posix()] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    _write_json(
        OUTPUT_ROOT / "experiment_manifest.json",
        {
            "schema_version": "mood-social-forecast-v3.4-experiment-manifest-v1",
            "experiment_version": "mood-social-forecast-v3.4",
            "context_manifest_sha256": sha256_file(OUTPUT_ROOT / "manifest.json"),
            "training_manifest_sha256": sha256_file(
                OUTPUT_ROOT / "training_manifest.json"
            ),
            "metrics_sha256": sha256_file(OUTPUT_ROOT / "metrics.json"),
            "code_commit": commit,
            "code_file_sha256": {
                "scripts/train_mood_forecast.py": sha256_file(
                    ALGORITHM_ROOT / "scripts" / "train_mood_forecast.py"
                ),
                "scripts/evaluate_mood_forecast.py": sha256_file(
                    Path(__file__).resolve()
                ),
                "src/elderly_monitoring/modules/mental_health/mood_social/forecast_modeling.py": sha256_file(
                    ALGORITHM_ROOT
                    / "src"
                    / "elderly_monitoring"
                    / "modules"
                    / "mental_health"
                    / "mood_social"
                    / "forecast_modeling.py"
                ),
            },
            "artifact_inventory": inventory,
            "release_status": "experimental",
            "execution_mode": "offline_only",
            "decision_authority": "shadow_only",
            "product_visible": False,
            "limitations": [
                "no_row_level_timestamps",
                "not_elderly_specific",
                "fitbit_to_project_device_domain_gap",
                "non_diagnostic_screening_label",
            ],
        },
    )
    lines = []
    for path in sorted(
        p for p in OUTPUT_ROOT.rglob("*") if p.is_file() and p.name != "SHA256SUMS"
    ):
        lines.append(f"{sha256_file(path)}  {path.relative_to(OUTPUT_ROOT).as_posix()}")
    (OUTPUT_ROOT / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bootstrap-repetitions", type=int, default=BOOTSTRAP_REPETITIONS
    )
    args = parser.parse_args()
    if args.bootstrap_repetitions != BOOTSTRAP_REPETITIONS:
        raise SystemExit(
            "formal V3.4 evaluation requires exactly 2000 bootstrap repetitions"
        )
    results = run_evaluation(args.bootstrap_repetitions)
    print(
        json.dumps(
            {
                "status": "pass",
                "metrics_path": str(OUTPUT_ROOT / "metrics.json"),
                "tasks": list(results["tasks"]),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
