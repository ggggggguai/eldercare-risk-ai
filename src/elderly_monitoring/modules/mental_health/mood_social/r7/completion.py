"""Final evidence audit and release decision for the frozen R7 development run."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import quantiles
from time import perf_counter
from typing import Any

from elderly_monitoring.modules.mental_health.mood_social.r7.contract import (
    repository_root,
    sha256_file,
)
from elderly_monitoring.modules.mental_health.mood_social.r7.release import (
    R7_PACKAGE_RELATIVE,
    R7_PACKAGE_RUN_ID,
    _load_verified_package,
    infer_mood_social_r7_candidate,
)


COMPLETION_RELATIVE = Path(
    "reports/mental_health/mood_social/v3.3.3-r7/OPT-V333-R7-008-completion"
)
FREEZE_RELATIVE = Path(
    "项目文档/开发协作/情绪与社交关注模块V3.3.3优化开发/"
    "00-V3.3.3-r7优化方案冻结说明.md"
)
FREEZE_SHA256 = "E80C3AC90F021AE4AEEDEA14CFEE26B141FC689BF71CEA29A86232C9DE565274"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _minimal_payload() -> dict[str, Any]:
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    target = (datetime.now(ZoneInfo("Asia/Shanghai")).date() - timedelta(days=1)).isoformat()
    return {
        "schema_version": "mood_social_infer_request_v3",
        "request_id": "r7-completion-benchmark",
        "person_id": "synthetic-local-replay",
        "target_date": target,
        "timezone": "Asia/Shanghai",
        "available_sources": ["camera"],
        "profile": {"age_group": "70_79", "sex": "female"},
        "current_daily_features": {
            "date": target,
            "activity": None,
            "sleep": None,
            "physiology": None,
            "social": None,
        },
        "history_daily_features": [],
        "history_attention_indices": [],
        "historical_phq9_assessments": [],
    }


def _benchmark(package: Path, repetitions: int = 500) -> dict[str, Any]:
    _load_verified_package.cache_clear()
    payload = _minimal_payload()
    start = perf_counter()
    first = infer_mood_social_r7_candidate(payload, package_directory=package)
    cold_ms = (perf_counter() - start) * 1000.0
    samples: list[float] = []
    for _ in range(repetitions):
        start = perf_counter()
        result = infer_mood_social_r7_candidate(payload, package_directory=package)
        samples.append((perf_counter() - start) * 1000.0)
        if result != first:
            raise RuntimeError("R7 inference is not deterministic")
    return {
        "repetitions": repetitions,
        "synthetic_no_evidence_only": True,
        "cold_start_ms": cold_ms,
        "mean_ms": sum(samples) / len(samples),
        "p95_ms": quantiles(samples, n=100)[94],
        "package_bytes": sum(path.stat().st_size for path in package.rglob("*") if path.is_file()),
    }


def _artifact_index(root: Path) -> dict[str, Any]:
    candidates: list[Path] = []
    candidates.extend((root / "src/elderly_monitoring/modules/mental_health/mood_social/r7").glob("*.py"))
    candidates.extend(root.glob("scripts/*mood_social*7*.py"))
    candidates.extend(root.glob("tests/test_mood_social_v3_3_3_r7*.py"))
    candidates.append(root / "configs/training/mood_social_v3_3_3_r7.yaml")
    for relative in (
        "data/processed/mental_health/mood_social/v3.3.3-r7",
        "reports/mental_health/mood_social/v3.3.3-r7",
        "models/mental_health/mood_social/v3.3.3-r7",
    ):
        candidates.extend((root / relative).rglob("*"))
    index_path = root / COMPLETION_RELATIVE / "r7_complete_artifact_index.json"
    rows = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(set(candidates))
        if path.is_file() and path != index_path
    ]
    return {"artifact_count": len(rows), "self_excluded": True, "artifacts": rows}


def run_completion(*, repository_root_value: Path | None = None) -> dict[str, Any]:
    root = repository_root(repository_root_value)
    workspace = root.parents[1]
    output = root / COMPLETION_RELATIVE
    package = root / R7_PACKAGE_RELATIVE
    experts = _read(root / "reports/mental_health/mood_social/v3.3.3-r7/OPT-V333-R7-002-independent-experts/summary.json")
    social = _read(root / "reports/mental_health/mood_social/v3.3.3-r7/OPT-V333-R7-003-social/confirmation_report.json")
    history = _read(root / "reports/mental_health/mood_social/v3.3.3-r7/OPT-V333-R7-004-history/metrics.json")
    evaluation = _read(root / "reports/mental_health/mood_social/v3.3.3-r7/OPT-V333-R7-006-evaluation/evaluation_report.json")
    package_report = _read(root / "reports/mental_health/mood_social/v3.3.3-r7/OPT-V333-R7-007-package/package_build_report.json")
    package_manifest = _read(package / "manifest.json")
    seal = _read(root / "reports/mental_health/mood_social/v3.3.3-r7/OPT-V333-R7-000/social_confirmation_seal.json")
    opened = _read(root / "reports/mental_health/mood_social/v3.3.3-r7/OPT-V333-R7-003-social/OPENED.json")
    runtime = _read(root / "configs/runtime/mood_social_current_state_production.json")

    checks = {
        "freeze_sha_matches": sha256_file(workspace / FREEZE_RELATIVE).upper() == FREEZE_SHA256,
        "package_checksums_valid": bool(_load_verified_package(str(package.resolve()))),
        "social_confirmation_opened_once": opened == {
            "confirmation_sha256": seal["confirmation_parquet_sha256"],
            "opened_once": True,
            "recipe_sha256": seal["candidate_recipe_sha256"],
            "reopen_allowed": False,
        },
        "no_full_five_domain_metric_claim": evaluation["complete_five_domain_auprc_reported"] is False,
        "production_default_unchanged": runtime["primary_package_run_id"] == "MH-20260812-R5-001",
        "forecast_unchanged": package_manifest["forecast_changed"] is False,
        "candidate_not_promoted": package_manifest["promotion_authorized"] is False,
        "social_probability_disabled": social["status"] == "anomaly_only_no_phq_probability",
        "package_identity_matches": package_report["package_run_id"] == R7_PACKAGE_RUN_ID,
    }
    if not all(checks.values()):
        raise RuntimeError(f"R7 completion checks failed: {checks}")

    expert_decisions = {
        "activity": "retain baseline logistic fallback; experimental shallow tree rejected",
        "sleep": "promote inside R7 candidate package",
        "social": "anomaly-only; PHQ probability disabled",
        "profile": "background vulnerability only; never a dynamic vote",
        "physiology": "support-only; never independently escalates",
        "phq_history": "promote as history-informed current-state expert; not passive-only",
    }
    benchmark = _benchmark(package)
    release = {
        "decision": "offline-validated/integration-ready/device-validation-pending",
        "production_switch_authorized": False,
        "online_default_package": runtime["primary_package_run_id"],
        "r7_candidate_package": R7_PACKAGE_RUN_ID,
        "v3_4_forecast_changed": False,
        "expert_decisions": expert_decisions,
        "blocking_external_items": [
            "real-device paired data are not yet available",
            "DepreST-CAT commercial/default-release license review is pending",
            "DepreST-CAT has only eight participants aged 56+",
        ],
    }
    completion = {
        "status": "pass",
        "tasks_completed": [f"OPT-V333-R7-{index:03d}" for index in range(9)],
        "checks": checks,
        "release": release,
        "benchmark": benchmark,
        "key_metrics": {
            "sleep_ge10_auprc": experts["sleep"]["metrics"]["ge10"]["candidate"]["auprc"],
            "sleep_ge10_delta_auprc": experts["sleep"]["metrics"]["ge10"]["delta"]["auprc"],
            "sleep_ge5_auprc": experts["sleep"]["metrics"]["ge5"]["candidate"]["auprc"],
            "sleep_ge5_delta_auprc": experts["sleep"]["metrics"]["ge5"]["delta"]["auprc"],
            "history_ge10_auprc": history["metrics"]["ge10"]["candidate"]["auprc"],
            "history_ge10_delta_auprc": history["metrics"]["ge10"]["delta_auprc"],
            "history_ge5_auprc": history["metrics"]["ge5"]["candidate"]["auprc"],
            "history_ge5_delta_auprc": history["metrics"]["ge5"]["delta_auprc"],
            "social_confirmation_ge10_auprc": social["confirmation"]["metrics"]["ge10"]["auprc"],
            "social_confirmation_ge5_auprc": social["confirmation"]["metrics"]["ge5"]["auprc"],
            "partial_rule_ge10_auprc": evaluation["partial_scenario_rule_fusion"]["ge10"]["auprc"],
            "partial_rule_ge5_auprc": evaluation["partial_scenario_rule_fusion"]["ge5"]["auprc"],
        },
    }
    _write_json(output / "completion_audit.json", completion)
    _write_json(output / "release_decision.json", release)

    failed_runs = [
        {"stage": "environment", "failure": "default Python lacked numpy", "resolution": "used locked eldercare-ai environment"},
        {"stage": "data-audit", "failure": "shell inline path encoding could not resolve Chinese path", "resolution": "used repository-root path resolution in Python module"},
        {"stage": "independent-experts", "failure": "initial broad candidate run exceeded short command timeout", "resolution": "re-ran frozen bounded families; no outer-label tuning"},
        {"stage": "social", "failure": "first run omitted inner-fold column", "resolution": "joined sealed split explicitly and restarted before confirmation opening"},
        {"stage": "social", "failure": "second run omitted global participant id", "resolution": "used pseudonymous participant id as grouped identity before confirmation opening"},
        {"stage": "compile-check", "failure": "PowerShell wildcard was passed literally to compileall", "resolution": "re-ran compileall with explicit directories"},
        {"stage": "release-regression", "failure": "request validation ran before package checksum after introducing the runtime cache", "resolution": "restored checksum-first fail-closed ordering and re-ran the full regression"},
    ]
    (output / "failed_runs.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in failed_runs),
        encoding="utf-8",
        newline="\n",
    )

    model_card = f"""# R7 候选模型卡（{R7_PACKAGE_RUN_ID}）

状态：`offline-validated / integration-ready / device-validation-pending`。当前线上仍为 R5 `MH-20260812-R5-001`；本包未获默认切换授权，V3.4 未来 1/2 月预测未改变。

R7 判断目标日 D 的当前关注状态，不作医学诊断。它由活动、睡眠、社交、画像、生理和历史 PHQ 独立证据组成，最终使用冻结的确定性规则融合；规则综合关注指数不是 PHQ 概率。历史 PHQ 仅进入独立 history-informed 专家，且必须 `assessment_date < D`、`known_at <= D`。缺失 `known_at` 时 R7 不使用该记录。

## 专家决策

- 活动：保留逻辑回归兜底；浅树候选未晋级。
- 睡眠：晋级。PHQ≥10 AUPRC `{experts['sleep']['metrics']['ge10']['candidate']['auprc']:.6f}`（Δ `{experts['sleep']['metrics']['ge10']['delta']['auprc']:+.6f}`）；PHQ≥5 AUPRC `{experts['sleep']['metrics']['ge5']['candidate']['auprc']:.6f}`（Δ `{experts['sleep']['metrics']['ge5']['delta']['auprc']:+.6f}`）。
- 社交：DepreST-CAT 确认集未通过完整门，仅 anomaly-only，不输出 PHQ 概率。
- 历史 PHQ：晋级 history-informed 当前状态专家。PHQ≥10 AUPRC `{history['metrics']['ge10']['candidate']['auprc']:.6f}`；PHQ≥5 AUPRC `{history['metrics']['ge5']['candidate']['auprc']:.6f}`。
- 画像：只输出背景脆弱性；生理：只输出个人异常辅助；二者均不作为动态票。

## 证据边界

既有五源结果属于 adaptive/reused evidence。DepreST-CAT 369 人确认边界只对冻结社交配方模型未见，不是老年设备外部验证：56 岁以上仅 8 人，且为 COVID 时期智能手机样本。DepreST-CAT 许可为 CC BY-NC-SA/学术用途，商业/default 发布前必须另做许可审查。当前无真实设备配对数据。

完整五域没有同人同期 PHQ 数据，因此不报告全五域 AUPRC。A+S 同人部分场景的规则分仅作描述性 QA，不能代表完整 R7 效果。
"""
    (output / "MODEL_CARD.md").write_text(model_card, encoding="utf-8", newline="\n")

    report = f"""# V3.3.3 R7 优化开发完成报告

`OPT-V333-R7-000—008` 已完成，完成审计为 `pass`。R7 已形成候选包 `{R7_PACKAGE_RUN_ID}` 和显式算法接口 `/v1/mental-health/mood-social/r7-candidate/infer`，默认生产 `/infer` 仍使用 R5，V3.4 保持独立。

## 本轮有效提升

| 专家 | PHQ≥10 AUPRC | 同折 ΔAUPRC | PHQ≥5 AUPRC | 同折 ΔAUPRC | 决策 |
|---|---:|---:|---:|---:|---|
| 睡眠 | {experts['sleep']['metrics']['ge10']['candidate']['auprc']:.6f} | {experts['sleep']['metrics']['ge10']['delta']['auprc']:+.6f} | {experts['sleep']['metrics']['ge5']['candidate']['auprc']:.6f} | {experts['sleep']['metrics']['ge5']['delta']['auprc']:+.6f} | 晋级 |
| 活动浅树候选 | {experts['activity']['metrics']['ge10']['candidate']['auprc']:.6f} | {experts['activity']['metrics']['ge10']['delta']['auprc']:+.6f} | {experts['activity']['metrics']['ge5']['candidate']['auprc']:.6f} | {experts['activity']['metrics']['ge5']['delta']['auprc']:+.6f} | 不晋级，保留逻辑回归 |
| 历史 PHQ | {history['metrics']['ge10']['candidate']['auprc']:.6f} | {history['metrics']['ge10']['delta_auprc']:+.6f} | {history['metrics']['ge5']['candidate']['auprc']:.6f} | {history['metrics']['ge5']['delta_auprc']:+.6f} | history-informed 晋级 |

社交确认集 AUPRC 为 PHQ≥10 `{social['confirmation']['metrics']['ge10']['auprc']:.6f}`（prevalence `{social['confirmation']['metrics']['ge10']['prevalence']:.6f}`）和 PHQ≥5 `{social['confirmation']['metrics']['ge5']['auprc']:.6f}`（prevalence `{social['confirmation']['metrics']['ge5']['prevalence']:.6f}`），但 AUROC/开发稳定性门未通过，因此已降级为异常证据，不包装成成功概率模型。

## 规则融合边界

规则 SHA 为 `{evaluation['rule_contract_sha256']}`，已通过 125 种状态组合真值表。真实同人的 A+S 部分场景共 {evaluation['partial_scenario_rule_fusion']['participants']} 人，其规则分 AUPRC 为 PHQ≥10 `{evaluation['partial_scenario_rule_fusion']['ge10']['auprc']:.6f}`、PHQ≥5 `{evaluation['partial_scenario_rule_fusion']['ge5']['auprc']:.6f}`；这只是 partial-scenario 描述结果，不是完整五域模型指标。

## 发布结论

R7 可用于离线演示和显式候选接口联调，不能替代默认 R5。真实设备联调数据、DepreST-CAT 许可审查和老年域验证完成后，才可另行作 shadow/default 决策。候选包大小 `{benchmark['package_bytes']}` bytes；合成无证据回放 p95 `{benchmark['p95_ms']:.3f} ms`，不代表真实设备端到端延迟。
"""
    (output / "V3.3.3-r7优化结果报告.md").write_text(report, encoding="utf-8", newline="\n")
    _write_json(output / "r7_complete_artifact_index.json", _artifact_index(root))
    return completion


__all__ = ["COMPLETION_RELATIVE", "run_completion"]
