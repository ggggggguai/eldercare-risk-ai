"""Evaluate FORECAST-OPT-001F against the frozen V3.4 baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ALGORITHM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ALGORITHM_ROOT / "src"))

from elderly_monitoring.modules.mental_health.mood_social.forecast_experiment import (  # noqa: E402
    sha256_file,
)
from elderly_monitoring.modules.mental_health.mood_social.forecast_modeling import (  # noqa: E402
    weighted_metrics,
    participant_equal_weights,
)


CONFIG_PATH = (
    ALGORITHM_ROOT
    / "configs"
    / "experiments"
    / "mood_social_forecast_v3_4_evaluation.yaml"
)
FORECAST_ROOT = (
    ALGORITHM_ROOT / "artifacts" / "mental_health" / "mood_social" / "forecast_v3.4"
)
BASELINE_ROOT = FORECAST_ROOT
CANDIDATE_ROOT = FORECAST_ROOT / "optimization_candidates"
UPSTREAM_ROOT = CANDIDATE_ROOT / "MH-20260807-FOPT-002"
EVALUATION_ROOT = UPSTREAM_ROOT / "evaluation"
AUXILIARY_ROOT = UPSTREAM_ROOT / "external_auxiliary" / "MH-20260807-FOPT-AUX-001"
TASK_COUNTS = {"forecast_1m": 9393, "forecast_2m": 9280}
METRIC_NAMES = (
    "auprc",
    "auroc",
    "macro_f1",
    "sensitivity",
    "specificity",
    "brier",
    "ece",
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    return value


def _write_sums(root: Path) -> None:
    rows = [
        f"{sha256_file(path)}  {path.relative_to(root).as_posix()}"
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "SHA256SUMS"
    ]
    (root / "SHA256SUMS").write_text(
        "\n".join(rows) + "\n", encoding="utf-8", newline="\n"
    )


def _load_config() -> dict[str, Any]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["task_id"] != "FORECAST-OPT-001F":
        raise RuntimeError("001F task identity changed")
    for key in (
        "backend_change",
        "api_change",
        "v3_3_package_change",
        "physiology_in_scope",
    ):
        if config[key] is not False:
            raise RuntimeError(f"001F isolation boundary changed: {key}")
    if config["bootstrap"]["repetitions"] != 2000:
        raise RuntimeError(
            "formal 001F evaluation requires 2,000 bootstrap repetitions"
        )
    return config


def _load_frames(
    config: dict[str, Any],
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    fusion_manifest = json.loads(
        (UPSTREAM_ROOT / "fusion_manifest.json").read_text(encoding="utf-8")
    )
    if (
        fusion_manifest["status"] != "fusion_completed"
        or fusion_manifest["failed_fold_count"] != 0
    ):
        raise RuntimeError("001D fusion manifest is not a successful frozen run")
    if (
        sha256_file(UPSTREAM_ROOT / "fusion_manifest.json")
        != config["upstream_fusion_manifest_sha256"]
    ):
        raise RuntimeError("001D fusion manifest hash changed")
    baseline = pd.read_parquet(BASELINE_ROOT / "oof_predictions.parquet")
    candidate = pd.read_parquet(UPSTREAM_ROOT / "fusion" / "outer_oof.parquet")
    baseline_by_task: dict[str, pd.DataFrame] = {}
    candidate_by_task: dict[str, pd.DataFrame] = {}
    for task, expected in TASK_COUNTS.items():
        family = config["baseline_models"][task]
        base = baseline.loc[
            baseline["task_id"].eq(task) & baseline["model_family"].eq(family)
        ].copy()
        opt = candidate.loc[candidate["task_id"].eq(task)].copy()
        if len(base) != expected or len(opt) != expected:
            raise RuntimeError(f"OOF count changed: {task}")
        key = ["global_participant_id", "target_window_id"]
        if base.duplicated(key).any() or opt.duplicated(key).any():
            raise RuntimeError(f"OOF key duplicated: {task}")
        if set(base[key].itertuples(index=False, name=None)) != set(
            opt[key].itertuples(index=False, name=None)
        ):
            raise RuntimeError(f"baseline/candidate windows differ: {task}")
        if base["outer_fold_id"].isna().any() or opt["outer_fold_id"].isna().any():
            raise RuntimeError(f"outer fold missing: {task}")
        if not base["global_participant_id"].isin(opt["global_participant_id"]).all():
            raise RuntimeError(f"participant coverage changed: {task}")
        if not np.array_equal(
            base.sort_values(key)["y_true"].to_numpy(dtype="int8"),
            opt.sort_values(key)["future_binary_target"].to_numpy(dtype="int8"),
        ):
            raise RuntimeError(f"baseline/candidate labels differ: {task}")
        baseline_by_task[task] = base
        candidate_by_task[task] = opt
    return baseline_by_task, candidate_by_task


def _metric_row(
    frame: pd.DataFrame,
    *,
    y_column: str,
    raw_column: str,
    calibrated_column: str,
    threshold_column: str | None = None,
    threshold: float | None = None,
) -> dict[str, float]:
    if threshold_column is not None:
        thresholds = frame[threshold_column].to_numpy(dtype="float64")
    elif threshold is not None:
        thresholds = np.full(len(frame), threshold, dtype="float64")
    else:
        raise RuntimeError("a threshold column or scalar is required")
    weights = participant_equal_weights(frame)
    result = weighted_metrics(
        frame[y_column].to_numpy(dtype="int8"),
        frame[raw_column].to_numpy(dtype="float64"),
        frame[calibrated_column].to_numpy(dtype="float64"),
        thresholds,
        weights,
    )
    return {name: float(result[name]) for name in METRIC_NAMES}


def _common_pair(
    baseline: pd.DataFrame, candidate: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    common = pd.read_parquet(FORECAST_ROOT / "common_target_windows.parquet")
    keys = ["global_participant_id", "target_window_id"]
    common_keys = common[keys]
    base = baseline.merge(common_keys, on=keys, how="inner", validate="one_to_one")
    opt = candidate.merge(common_keys, on=keys, how="inner", validate="one_to_one")
    if len(base) != 8494 or len(opt) != 8494:
        raise RuntimeError("common-window count changed")
    return base, opt


def _bootstrap_pair(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    scope: str,
    repetitions: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    keys = ["global_participant_id", "target_window_id"]
    if (
        not baseline[keys]
        .reset_index(drop=True)
        .equals(candidate[keys].reset_index(drop=True))
    ):
        candidate = (
            candidate.set_index(keys).loc[baseline.set_index(keys).index].reset_index()
        )
    participants = sorted(baseline["global_participant_id"].unique().tolist())
    base_groups = {
        participant: np.flatnonzero(
            baseline["global_participant_id"].to_numpy() == participant
        )
        for participant in participants
    }
    candidate_groups = {
        participant: np.flatnonzero(
            candidate["global_participant_id"].to_numpy() == participant
        )
        for participant in participants
    }
    rng = np.random.Generator(np.random.PCG64(seed))
    rows: list[dict[str, Any]] = []
    degenerate = 0
    while len(rows) < repetitions:
        selected = rng.integers(0, len(participants), size=len(participants))
        base_indices: list[np.ndarray] = []
        candidate_indices: list[np.ndarray] = []
        weight_parts: list[np.ndarray] = []
        for position in selected:
            participant = participants[int(position)]
            base_indices.append(base_groups[participant])
            candidate_indices.append(candidate_groups[participant])
            weight_parts.append(
                np.full(
                    len(base_groups[participant]),
                    1.0 / (len(participants) * len(base_groups[participant])),
                )
            )
        base_index = np.concatenate(base_indices)
        candidate_index = np.concatenate(candidate_indices)
        weights = np.concatenate(weight_parts)
        y = baseline["y_true"].to_numpy(dtype="int8")[base_index]
        if np.unique(y).size < 2:
            degenerate += 1
            continue
        base_metrics = weighted_metrics(
            y,
            baseline["p_raw"].to_numpy(dtype="float64")[base_index],
            baseline["p_calibrated"].to_numpy(dtype="float64")[base_index],
            baseline["fold_threshold"].to_numpy(dtype="float64")[base_index],
            weights,
        )
        candidate_metrics = weighted_metrics(
            y,
            candidate["p_blended"].to_numpy(dtype="float64")[candidate_index],
            candidate["p_calibrated"].to_numpy(dtype="float64")[candidate_index],
            candidate["threshold"].to_numpy(dtype="float64")[candidate_index],
            weights,
        )
        row: dict[str, Any] = {"scope": scope, "replicate": len(rows)}
        for name in METRIC_NAMES:
            row[f"baseline_{name}"] = float(base_metrics[name])
            row[f"candidate_{name}"] = float(candidate_metrics[name])
            row[f"delta_{name}"] = float(candidate_metrics[name] - base_metrics[name])
        rows.append(row)
    table = pd.DataFrame(rows)
    summary = {
        "scope": scope,
        "participant_count": len(participants),
        "valid_repetitions": len(rows),
        "degenerate_repetitions": degenerate,
        "seed": seed,
        "confidence_level": 0.95,
        "intervals": {
            name: {
                "lower": float(table[f"delta_{name}"].quantile(0.025)),
                "upper": float(table[f"delta_{name}"].quantile(0.975)),
            }
            for name in METRIC_NAMES
        },
        "medians": {
            name: float(table[f"delta_{name}"].median()) for name in METRIC_NAMES
        },
    }
    return table, summary


def _fold_stability(
    baseline_by_task: dict[str, pd.DataFrame],
    candidate_by_task: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for task in TASK_COUNTS:
        for fold in range(5):
            base = baseline_by_task[task].loc[
                baseline_by_task[task]["outer_fold_id"].eq(fold)
            ]
            candidate = candidate_by_task[task].loc[
                candidate_by_task[task]["outer_fold_id"].eq(fold)
            ]
            base_metric = _metric_row(
                base,
                y_column="y_true",
                raw_column="p_raw",
                calibrated_column="p_calibrated",
                threshold_column="fold_threshold",
            )
            candidate_metric = _metric_row(
                candidate,
                y_column="future_binary_target",
                raw_column="p_blended",
                calibrated_column="p_calibrated",
                threshold_column="threshold",
            )
            rows.append(
                {
                    "task_id": task,
                    "outer_fold_id": fold,
                    "sample_count": len(candidate),
                    "participant_count": candidate["global_participant_id"].nunique(),
                    **{f"baseline_{k}": v for k, v in base_metric.items()},
                    **{f"candidate_{k}": v for k, v in candidate_metric.items()},
                    **{
                        f"delta_{k}": candidate_metric[k] - base_metric[k]
                        for k in METRIC_NAMES
                    },
                }
            )
    return pd.DataFrame(rows)


def _ablation_metrics(candidate_by_task: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    columns = (
        ("elasticnet_logistic", "p_elasticnet_logistic"),
        ("lightgbm", "p_lightgbm"),
        ("catboost", "p_catboost"),
        ("un calibrated_blend", "p_blended"),
        ("calibrated_fusion", "p_calibrated"),
    )
    for task, frame in candidate_by_task.items():
        for name, column in columns:
            if name == "calibrated_fusion":
                raw = "p_calibrated"
                calibrated = "p_calibrated"
                threshold_column = "threshold"
            else:
                raw = column
                calibrated = column
                threshold_column = None
            metric = _metric_row(
                frame,
                y_column="future_binary_target",
                raw_column=raw,
                calibrated_column=calibrated,
                threshold_column=threshold_column,
                threshold=0.5 if threshold_column is None else None,
            )
            rows.append(
                {
                    "task_id": task,
                    "ablation": name,
                    "eligible_for_promotion": False,
                    **metric,
                }
            )
    return pd.DataFrame(rows)


def _proxy_metrics(candidate_by_task: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    proxy = pd.read_parquet(AUXILIARY_ROOT / "conscious_psyche_proxy.parquet")
    rows: list[dict[str, Any]] = []
    for task, frame in candidate_by_task.items():
        joined = frame.merge(
            proxy.loc[proxy["task_id"].eq(task)],
            left_on=["global_participant_id", "target_window_id", "outer_fold_id"],
            right_on=["global_participant_id", "target_window_id", "outer_fold_id"],
            how="inner",
            validate="one_to_one",
        )
        if (
            len(joined) != TASK_COUNTS[task]
            or joined["conscious_phq9_proxy"].isna().any()
        ):
            raise RuntimeError(f"Conscious proxy coverage changed: {task}")
        score = (
            joined["conscious_phq9_proxy"].rank(method="average", pct=True).to_numpy()
        )
        proxy_frame = joined.assign(proxy_score=score)
        metric = _metric_row(
            proxy_frame,
            y_column="future_binary_target",
            raw_column="proxy_score",
            calibrated_column="proxy_score",
            threshold=0.5,
        )
        rows.append(
            {
                "task_id": task,
                "scope": "full",
                "proxy_row_count": len(joined),
                "proxy_participant_count": joined["global_participant_id"].nunique(),
                "mapping": "within-task percentile rank; no label fitting",
                "eligible_for_promotion": False,
                "reason": "external proxy lacks project-domain calibrated probability and was not in 001D",
                **metric,
            }
        )
    return rows


def _promotion(
    config: dict[str, Any],
    full_metrics: dict[str, dict[str, dict[str, float]]],
    fold_stability: pd.DataFrame,
    bootstrap_summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    gate = config["promotion_gate"]
    per_task: dict[str, Any] = {}
    for task in TASK_COUNTS:
        baseline = full_metrics[task]["baseline"]
        candidate = full_metrics[task]["candidate"]
        diffs = {name: candidate[name] - baseline[name] for name in METRIC_NAMES}
        folds = fold_stability.loc[fold_stability["task_id"].eq(task)]
        auprc_fold_pass = int((folds["delta_auprc"] >= 0).sum())
        criteria = {
            "auprc_delta_at_least_0_005": diffs["auprc"] >= gate["auprc_min_delta"],
            "auroc_delta_not_below_minus_0_005": diffs["auroc"]
            >= gate["auroc_min_delta"],
            "brier_delta_at_most_0_002": diffs["brier"] <= gate["brier_max_delta"],
            "ece_within_frozen_limit": candidate["ece"]
            <= max(baseline["ece"] + gate["ece_max_delta"], gate["ece_max_absolute"]),
            "at_least_three_non_decreasing_folds": auprc_fold_pass
            >= gate["min_non_decreasing_folds"],
            "no_fold_auprc_drop_over_0_02": float(folds["delta_auprc"].min())
            >= -gate["max_fold_auprc_drop"],
            "paired_bootstrap_auprc_median_positive": bootstrap_summaries[task][
                "medians"
            ]["auprc"]
            > 0,
        }
        per_task[task] = {
            "status": "promote_candidate"
            if all(criteria.values())
            else "retain_baseline",
            "criteria": criteria,
            "point_deltas": diffs,
            "auprc_non_decreasing_fold_count": auprc_fold_pass,
            "min_fold_auprc_delta": float(folds["delta_auprc"].min()),
            "bootstrap_auprc_median": bootstrap_summaries[task]["medians"]["auprc"],
        }
    overall = all(item["status"] == "promote_candidate" for item in per_task.values())
    return {
        "status": "promote_candidate" if overall else "retain_v3_4_baseline",
        "reason": "both horizons must pass every frozen gate; failed gates retain the original offline baseline",
        "per_task": per_task,
        "product_integration_performed": False,
        "product_visible": False,
    }


def _write_report(
    result: dict[str, Any],
    full_metrics: dict[str, dict[str, dict[str, float]]],
    promotion: dict[str, Any],
    proxy_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# FORECAST-OPT-001F 评价报告",
        "",
        "本报告比较冻结 V3.4 基线与 001D 非负融合候选。所有预测来自严格 outer OOF；本阶段只做离线评价，不接入后端或替换 V3.3。",
        "",
        f"- 运行：`{result['run_id']}`；bootstrap：{result['bootstrap_repetitions']} 次参与者聚类重采样。",
        f"- 晋级结论：`{promotion['status']}`。若任一 horizon 未通过全部门槛，继续保留原 V3.4 离线基线。",
        "",
        "## 全量结果",
        "",
        "| Horizon | 方案 | AUPRC | AUROC | Brier | ECE |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for task in TASK_COUNTS:
        for name, label in (("baseline", "冻结基线"), ("candidate", "001D融合候选")):
            metric = full_metrics[task][name]
            lines.append(
                f"| {task} | {label} | {metric['auprc']:.4f} | {metric['auroc']:.4f} | {metric['brier']:.4f} | {metric['ece']:.4f} |"
            )
    lines += ["", "## Conscious 影子结果", ""]
    lines.append(
        "Conscious 仅以任务内百分位排名作为无标签拟合的影子分数，不参与晋级门。"
    )
    for row in proxy_rows:
        lines.append(
            f"- {row['task_id']}：AUPRC `{row['auprc']:.4f}`，AUROC `{row['auroc']:.4f}`；资格：`{row['eligible_for_promotion']}`。"
        )
    lines += [
        "",
        "## 边界",
        "",
        "OBF 已在 001E 停止迁移，CoronaHealth 延后；本报告不把外部代理或负结果写成产品效果。",
    ]
    (EVALUATION_ROOT / "evaluation_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def evaluate() -> dict[str, Any]:
    config = _load_config()
    if EVALUATION_ROOT.exists():
        raise FileExistsError(f"001F output already exists: {EVALUATION_ROOT}")
    EVALUATION_ROOT.mkdir(parents=True)
    baseline_by_task, candidate_by_task = _load_frames(config)
    full_metrics: dict[str, dict[str, dict[str, float]]] = {}
    common_metrics: dict[str, dict[str, dict[str, float]]] = {}
    bootstrap_tables: list[pd.DataFrame] = []
    bootstrap_summaries: dict[str, dict[str, Any]] = {}
    for task in TASK_COUNTS:
        baseline = baseline_by_task[task]
        candidate = candidate_by_task[task]
        full_metrics[task] = {
            "baseline": _metric_row(
                baseline,
                y_column="y_true",
                raw_column="p_raw",
                calibrated_column="p_calibrated",
                threshold_column="fold_threshold",
            ),
            "candidate": _metric_row(
                candidate,
                y_column="future_binary_target",
                raw_column="p_blended",
                calibrated_column="p_calibrated",
                threshold_column="threshold",
            ),
        }
        base_common, candidate_common = _common_pair(baseline, candidate)
        common_metrics[task] = {
            "baseline": _metric_row(
                base_common,
                y_column="y_true",
                raw_column="p_raw",
                calibrated_column="p_calibrated",
                threshold_column="fold_threshold",
            ),
            "candidate": _metric_row(
                candidate_common,
                y_column="future_binary_target",
                raw_column="p_blended",
                calibrated_column="p_calibrated",
                threshold_column="threshold",
            ),
        }
        table, summary = _bootstrap_pair(
            baseline,
            candidate,
            scope="full",
            repetitions=config["bootstrap"]["repetitions"],
            seed=config["bootstrap"]["seed"],
        )
        common_table, common_summary = _bootstrap_pair(
            base_common,
            candidate_common,
            scope="common",
            repetitions=config["bootstrap"]["repetitions"],
            seed=config["bootstrap"]["seed"],
        )
        table.insert(0, "task_id", task)
        common_table.insert(0, "task_id", task)
        bootstrap_tables.extend([table, common_table])
        bootstrap_summaries[task] = summary
        bootstrap_summaries[f"{task}::common"] = common_summary
    fold_stability = _fold_stability(baseline_by_task, candidate_by_task)
    ablation = _ablation_metrics(candidate_by_task)
    proxy_rows = _proxy_metrics(candidate_by_task)
    promotion = _promotion(config, full_metrics, fold_stability, bootstrap_summaries)
    metrics_rows: list[dict[str, Any]] = []
    for task in TASK_COUNTS:
        for scope, payload in (
            ("full", full_metrics[task]),
            ("common", common_metrics[task]),
        ):
            for variant, metric in payload.items():
                metrics_rows.append(
                    {"task_id": task, "scope": scope, "variant": variant, **metric}
                )
    pd.DataFrame(metrics_rows).to_parquet(
        EVALUATION_ROOT / "task_metrics.parquet", index=False
    )
    pd.concat(bootstrap_tables, ignore_index=True).to_parquet(
        EVALUATION_ROOT / "bootstrap_deltas.parquet", index=False
    )
    fold_stability.to_parquet(EVALUATION_ROOT / "fold_stability.parquet", index=False)
    ablation.to_parquet(EVALUATION_ROOT / "ablation_metrics.parquet", index=False)
    _write_json(EVALUATION_ROOT / "bootstrap_summary.json", bootstrap_summaries)
    _write_json(EVALUATION_ROOT / "promotion_decision.json", promotion)
    _write_json(EVALUATION_ROOT / "conscious_proxy_metrics.json", {"rows": proxy_rows})
    manifest = {
        "schema_version": "mood-social-forecast-v3.4-optimization-evaluation-manifest-v1",
        "task_id": "FORECAST-OPT-001F",
        "run_id": config["run_id"],
        "baseline_run_id": config["baseline_run_id"],
        "status": "evaluation_completed",
        "bootstrap_performed": True,
        "bootstrap_repetitions": config["bootstrap"]["repetitions"],
        "common_window_count": config["common_window_count"],
        "outer_test_used_for_evaluation": True,
        "outer_test_used_for_selection": False,
        "promotion_decision_performed": True,
        "product_integration_performed": False,
        "release_status": config["release_status"],
        "execution_mode": config["execution_mode"],
        "decision_authority": config["decision_authority"],
        "product_visible": config["product_visible"],
        "config_sha256": sha256_file(CONFIG_PATH),
        "evaluation_code_sha256": sha256_file(Path(__file__).resolve()),
        "baseline_oof_sha256": sha256_file(BASELINE_ROOT / "oof_predictions.parquet"),
        "fusion_oof_sha256": sha256_file(
            UPSTREAM_ROOT / "fusion" / "outer_oof.parquet"
        ),
        "conscious_proxy_sha256": sha256_file(
            AUXILIARY_ROOT / "conscious_psyche_proxy.parquet"
        ),
        "promotion_status": promotion["status"],
        "failure_count": 0,
    }
    _write_json(EVALUATION_ROOT / "evaluation_manifest.json", manifest)
    result = {
        **manifest,
        "full_metrics": full_metrics,
        "common_metrics": common_metrics,
        "promotion": promotion,
        "bootstrap_summaries": bootstrap_summaries,
    }
    _write_report(result, full_metrics, promotion, proxy_rows)
    _write_sums(EVALUATION_ROOT)
    _write_sums(UPSTREAM_ROOT)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="MH-20260807-FOPT-002")
    args = parser.parse_args()
    if args.run_id != "MH-20260807-FOPT-002":
        raise RuntimeError("only the frozen 001D run is eligible for 001F")
    result = evaluate()
    print(json.dumps(_json_safe(result), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
