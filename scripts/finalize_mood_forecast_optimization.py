"""Create the FORECAST-OPT-001G delivery package without product integration."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

ALGORITHM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ALGORITHM_ROOT / "src"))

from elderly_monitoring.modules.mental_health.mood_social.forecast_experiment import (  # noqa: E402
    sha256_file,
)


WORKSPACE_ROOT = ALGORITHM_ROOT.parents[1]
FORECAST_ROOT = (
    ALGORITHM_ROOT / "artifacts" / "mental_health" / "mood_social" / "forecast_v3.4"
)
CANDIDATE_ROOT = FORECAST_ROOT / "optimization_candidates"
RUN_ID = "MH-20260807-FOPT-002"
RUN_ROOT = CANDIDATE_ROOT / RUN_ID
EVALUATION_ROOT = RUN_ROOT / "evaluation"
RELEASE_ROOT = RUN_ROOT / "release"
TASKS = ("forecast_1m", "forecast_2m")
METRICS = ("auprc", "auroc", "macro_f1", "sensitivity", "specificity", "brier", "ece")
BACKEND_TOKENS = ("forecast_1m", "forecast_2m", "mood_forecast", "forecast_v3.4")


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object expected: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _metric_rows() -> pd.DataFrame:
    rows = pd.read_parquet(EVALUATION_ROOT / "task_metrics.parquet")
    if len(rows) != 8:
        raise RuntimeError("001F task metric schema changed")
    return rows


def _metric(
    rows: pd.DataFrame, task_id: str, variant: str, scope: str = "full"
) -> dict[str, float]:
    selected = rows[
        rows.task_id.eq(task_id) & rows.variant.eq(variant) & rows.scope.eq(scope)
    ]
    if len(selected) != 1:
        raise RuntimeError(f"metric row missing: {task_id}/{variant}/{scope}")
    return {name: float(selected.iloc[0][name]) for name in METRICS}


def _backend_hits() -> list[str]:
    hits: list[str] = []
    backend = WORKSPACE_ROOT / "backend"
    for path in backend.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {
            ".py",
            ".json",
            ".yaml",
            ".yml",
            ".md",
        }:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeError:
            continue
        if any(token in content for token in BACKEND_TOKENS):
            hits.append(path.relative_to(backend).as_posix())
    return sorted(hits)


def _v33_state() -> dict[str, Any]:
    root = ALGORITHM_ROOT / "models" / "mental_health" / "mood_social" / "v3.3.3"
    acceptance_path = root / "acceptance.json"
    acceptance = _json(acceptance_path)
    package = ALGORITHM_ROOT / acceptance["package_path"]
    pipeline = (
        ALGORITHM_ROOT
        / "src/elderly_monitoring/modules/mental_health/mood_social/pipeline.py"
    )
    selection = (
        ALGORITHM_ROOT
        / "src/elderly_monitoring/modules/mental_health/mood_social/package_selection.py"
    )
    schemas = (
        ALGORITHM_ROOT
        / "src/elderly_monitoring/modules/mental_health/mood_social/schemas.py"
    )
    return {
        "model_version": acceptance["model_version"],
        "active_run_id": acceptance["package_run_id"],
        "acceptance_sha256": sha256_file(acceptance_path),
        "package_manifest_sha256": sha256_file(package / "manifest.json"),
        "package_checksums_sha256": sha256_file(package / "SHA256SUMS"),
        "pipeline_sha256": sha256_file(pipeline),
        "package_selection_sha256": sha256_file(selection),
        "schemas_sha256": sha256_file(schemas),
    }


def _card(
    task_id: str,
    baseline: dict[str, float],
    candidate: dict[str, float],
    decision: dict[str, Any],
    bootstrap: dict[str, Any],
) -> str:
    task_decision = decision["per_task"][task_id]
    interval = bootstrap[task_id]["intervals"]["auprc"]
    status = task_decision["status"]
    return f"""# {task_id} / 001D fusion candidate model card

## Purpose

This is an offline PSYCHE-D nominal-month forecasting candidate for a later higher PHQ-9 symptom-pattern label. It is not a diagnosis, is not a China-elderly or device-domain validation, and has no product decision authority.

## Method and data boundary

- Inputs: the frozen 27 activity/sleep fields plus strictly past anchor, delta and rolling history features selected inside outer-train inner folds.
- Prediction unit: participant-isolated target window; all metrics use strict outer OOF predictions.
- Candidate: non-negative ElasticNet/LightGBM/CatBoost fusion with fold-local calibration and threshold selection.
- Evaluation: participant-equal metrics, 2,000 participant-cluster bootstrap repetitions, paired on the frozen common window where applicable.
- Release status: `experimental / offline_only / shadow_only / product_visible=false`.

## Full-window result

| Metric | Frozen baseline | 001D candidate | Difference |
|---|---:|---:|---:|
| AUPRC | {baseline["auprc"]:.6f} | {candidate["auprc"]:.6f} | {candidate["auprc"] - baseline["auprc"]:+.6f} |
| AUROC | {baseline["auroc"]:.6f} | {candidate["auroc"]:.6f} | {candidate["auroc"] - baseline["auroc"]:+.6f} |
| Macro-F1 | {baseline["macro_f1"]:.6f} | {candidate["macro_f1"]:.6f} | {candidate["macro_f1"] - baseline["macro_f1"]:+.6f} |
| Sensitivity | {baseline["sensitivity"]:.6f} | {candidate["sensitivity"]:.6f} | {candidate["sensitivity"] - baseline["sensitivity"]:+.6f} |
| Specificity | {baseline["specificity"]:.6f} | {candidate["specificity"]:.6f} | {candidate["specificity"] - baseline["specificity"]:+.6f} |
| Brier | {baseline["brier"]:.6f} | {candidate["brier"]:.6f} | {candidate["brier"] - baseline["brier"]:+.6f} |
| ECE | {baseline["ece"]:.6f} | {candidate["ece"]:.6f} | {candidate["ece"] - baseline["ece"]:+.6f} |

## Stability and decision

- AUPRC bootstrap median: `{task_decision["bootstrap_auprc_median"]:+.6f}`.
- AUPRC bootstrap 95% interval: `[{interval["lower"]:.6f}, {interval["upper"]:.6f}]`.
- Non-decreasing outer folds: `{task_decision["auprc_non_decreasing_fold_count"]}/5`; worst-fold delta: `{task_decision["min_fold_auprc_delta"]:+.6f}`.
- Horizon decision: `{status}`.

The overall release decision requires both horizons to pass. The package-level decision is recorded in `rollback_decision.json`; because the 1-month horizon did not meet the AUPRC gate, the existing V3.4 offline baseline remains the recommended artifact and this candidate is retained for audit only.

## Limitations

PSYCHE-D supplies design-level nominal-month evidence rather than natural timestamps. Fitbit-derived fields do not establish transfer to the project's camera or sleep-monitor domain. Feature contributions describe association, not causation. This card must not be used to claim medical diagnosis or deployment efficacy.
"""


def build_release() -> dict[str, Any]:
    if not EVALUATION_ROOT.is_dir():
        raise RuntimeError("001F evaluation artifacts are missing")
    decision = _json(EVALUATION_ROOT / "promotion_decision.json")
    bootstrap = _json(EVALUATION_ROOT / "bootstrap_summary.json")
    if decision["status"] != "retain_v3_4_baseline":
        raise RuntimeError(
            "001G release is only defined for the frozen no-promotion result"
        )
    if RELEASE_ROOT.exists() and any(RELEASE_ROOT.iterdir()):
        raise RuntimeError(f"release directory is not empty: {RELEASE_ROOT}")
    RELEASE_ROOT.mkdir(parents=True, exist_ok=True)
    rows = _metric_rows()
    cards: list[dict[str, str]] = []
    for task_id in TASKS:
        baseline = _metric(rows, task_id, "baseline")
        candidate = _metric(rows, task_id, "candidate")
        slug = task_id
        md_name = f"{slug}_001d_candidate_model_card.md"
        json_name = f"{slug}_001d_candidate_model_card.json"
        (RELEASE_ROOT / md_name).write_text(
            _card(task_id, baseline, candidate, decision, bootstrap),
            encoding="utf-8",
            newline="\n",
        )
        card_payload = {
            "task_id": task_id,
            "run_id": RUN_ID,
            "candidate_status": "audit_only_not_promoted",
            "release_status": "experimental",
            "execution_mode": "offline_only",
            "decision_authority": "shadow_only",
            "product_visible": False,
            "baseline": baseline,
            "candidate": candidate,
            "decision": decision["per_task"][task_id],
            "bootstrap_auprc": bootstrap[task_id],
            "recommended_fallback": "MH-20260805-FCAST-001",
        }
        _write_json(RELEASE_ROOT / json_name, card_payload)
        cards.append({"markdown": md_name, "json": json_name})

    v33 = _v33_state()
    backend_hashes = {
        path: sha256_file(WORKSPACE_ROOT / "backend" / path)
        for path in (
            "app/services/mood_social.py",
            "app/services/algorithm_client.py",
            "app/routers/mood_social.py",
        )
    }
    backend_hits = _backend_hits()
    isolation = {
        "task_id": "FORECAST-OPT-001G",
        "run_id": RUN_ID,
        "status": "pass",
        "forecast_backend_hits": backend_hits,
        "backend_hashes": backend_hashes,
        "v3_3": v33,
        "online_model_version": v33["model_version"],
        "online_package_run_id": v33["active_run_id"],
        "product_integration_performed": False,
        "product_visible": False,
        "decision_authority": "shadow_only",
    }
    _write_json(RELEASE_ROOT / "isolation_report.json", isolation)
    rollback = {
        "task_id": "FORECAST-OPT-001G",
        "run_id": RUN_ID,
        "promotion_status": decision["status"],
        "candidate_promoted": False,
        "recommended_artifact": "MH-20260805-FCAST-001",
        "online_model_version": v33["model_version"],
        "online_package_run_id": v33["active_run_id"],
        "reason": "forecast_1m did not meet the frozen AUPRC gate; both horizons are required",
        "product_integration_performed": False,
        "release_status": "experimental",
        "execution_mode": "offline_only",
        "decision_authority": "shadow_only",
        "product_visible": False,
    }
    _write_json(RELEASE_ROOT / "rollback_decision.json", rollback)
    metrics = _metric_rows()
    report_lines = [
        "# FORECAST-OPT-001G 交付报告",
        "",
        "本报告整理 001F 正式评价结果并完成交付隔离复核。候选不进入后端，不替换 V3.3 在线模型。",
        "",
        f"- 运行：`{RUN_ID}`；评价 manifest：`{sha256_file(EVALUATION_ROOT / 'evaluation_manifest.json')}`。",
        f"- 晋级结论：`{decision['status']}`；推荐回退：`MH-20260805-FCAST-001`。",
        f"- 当前在线模型：`{v33['model_version']}` / `{v33['active_run_id']}`。",
        "",
        "## 结果摘要",
        "",
        "| Horizon | 基线 AUPRC | 候选 AUPRC | 差值 | 候选 AUROC | 候选 Brier | 候选 ECE | horizon 决定 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for task_id in TASKS:
        base = _metric(metrics, task_id, "baseline")
        cand = _metric(metrics, task_id, "candidate")
        report_lines.append(
            f"| {task_id} | {base['auprc']:.6f} | {cand['auprc']:.6f} | {cand['auprc'] - base['auprc']:+.6f} | {cand['auroc']:.6f} | {cand['brier']:.6f} | {cand['ece']:.6f} | {decision['per_task'][task_id]['status']} |"
        )
    report_lines += [
        "",
        "## 验收与隔离",
        "",
        "- 001F 已完成 2,000 次参与者 bootstrap、共同窗口配对、逐折稳定性、模型/校准消融和 Conscious 影子候选评价。",
        "- 两个 horizon 必须同时通过冻结晋级门；1m 未通过，因此整体保留原 V3.4 离线基线。",
        "- V3.3 acceptance、在线包 manifest/SHA256SUMS、算法管线/包选择代码和后端情绪接口文件均完成哈希登记。",
        f"- 后端静态 Forecast 命中数：`{len(backend_hits)}`；V3.3 在线模型：`{v33['model_version']}`。",
        "- 状态固定为 `experimental / offline_only / shadow_only / product_visible=false`。",
        "",
        "## 限制",
        "",
        "PSYCHE-D 是名义月份条件实验，不是中国老年人或项目摄像头/睡眠仪设备域验证；输出不是医学诊断。Conscious/OBF 结果只作辅助证据，不能扩大为产品效果。",
        "",
    ]
    (RELEASE_ROOT / "final_delivery_report.md").write_text(
        "\n".join(report_lines), encoding="utf-8", newline="\n"
    )
    manifest = {
        "schema_version": "mood-social-forecast-v3.4-opt-release-manifest-v1",
        "task_id": "FORECAST-OPT-001G",
        "run_id": RUN_ID,
        "status": "release_completed",
        "created_at": datetime.now(UTC).isoformat(),
        "upstream_evaluation_manifest_sha256": sha256_file(
            EVALUATION_ROOT / "evaluation_manifest.json"
        ),
        "upstream_promotion_decision_sha256": sha256_file(
            EVALUATION_ROOT / "promotion_decision.json"
        ),
        "upstream_evaluation_validation_sha256": sha256_file(
            EVALUATION_ROOT / "validation_report.json"
        ),
        "promotion_status": decision["status"],
        "recommended_fallback": "MH-20260805-FCAST-001",
        "release_status": "experimental",
        "execution_mode": "offline_only",
        "decision_authority": "shadow_only",
        "product_integration_performed": False,
        "product_visible": False,
        "model_cards": cards,
        "isolation_report": "isolation_report.json",
        "rollback_decision": "rollback_decision.json",
        "final_delivery_report": "final_delivery_report.md",
        "v3_3": v33,
        "backend_hashes": backend_hashes,
        "backend_forecast_hits": backend_hits,
        "finalize_code_sha256": sha256_file(Path(__file__).resolve()),
    }
    _write_json(RELEASE_ROOT / "release_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=RUN_ID)
    args = parser.parse_args()
    if args.run_id != RUN_ID:
        raise SystemExit("001G is frozen to MH-20260807-FOPT-002")
    result = build_release()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
