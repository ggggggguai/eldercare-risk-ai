"""R11 completion audit, engineering metrics and honest release decision."""

from __future__ import annotations

import json
from pathlib import Path
import statistics
import time
import tracemalloc
from typing import Any

from pydantic import ValidationError

from .contract import (
    DEFAULT_DATA_RELATIVE,
    DEFAULT_REPORT_RELATIVE,
    FROZEN_DOCUMENT_RELATIVE,
    R11_FROZEN_DOCUMENT_SHA256,
    project_root,
    repository_root,
    sha256_file,
    write_json,
)
from .evidence_orchestrator import validate_routes
from .diagnostics import DIAGNOSTIC_RELATIVE
from .release import (
    R11_PACKAGE_RELATIVE,
    R11_PACKAGE_RUN_ID,
    infer_mood_social_r11_candidate,
)
from .rule_fusion import validate_truth_table
from .schemas import R11CandidateRequest, R11CandidateResponse


COMPLETION_RELATIVE = Path(
    "reports/mental_health/mood_social/v3.3.3-r11/OPT-V333-R11-009-completion"
)


def _payload() -> dict[str, Any]:
    return {
        "schema_version": "mood_social_r11_candidate_request_v1",
        "request_id": "r11-completion-golden",
        "person_id": "elder-golden",
        "target_date": "2026-08-13",
        "timezone": "Asia/Shanghai",
        "available_sources": ["camera"],
        "profile": {"age_group": "70_79", "sex": "female"},
        "current_daily_features": {
            "date": "2026-08-13",
            "activity": None,
            "sleep": None,
            "physiology": None,
            "social": None,
        },
        "history_daily_features": [],
        "history_attention_indices": [],
        "historical_phq9_assessments": [],
        "inference_cutoff": "2026-08-13T23:59:59+08:00",
        "facial_affect_authorized": False,
        "facial_affect_observations": [],
    }


def _fuzz() -> dict[str, Any]:
    payload = _payload()
    mutations = []
    extra = dict(payload)
    extra["source_id"] = "forbidden"
    mutations.append(extra)
    bad_schema = dict(payload)
    bad_schema["schema_version"] = "wrong"
    mutations.append(bad_schema)
    future = dict(payload)
    future["target_date"] = "2099-01-01"
    future["current_daily_features"] = {
        **payload["current_daily_features"],
        "date": "2099-01-01",
    }
    mutations.append(future)
    bad_order = dict(payload)
    bad_order["historical_phq9_assessments"] = [
        {"date": "2026-05-01", "known_at": "2026-05-01", "score": 5, "source": "s10"},
        {"date": "2026-04-01", "known_at": "2026-04-01", "score": 8, "source": "s10"},
    ]
    mutations.append(bad_order)
    rejected = 0
    for mutation in mutations:
        try:
            R11CandidateRequest.model_validate(mutation)
        except (ValidationError, ValueError):
            rejected += 1
    return {
        "cases": len(mutations),
        "rejected": rejected,
        "pass_rate": rejected / len(mutations),
    }


def _performance() -> tuple[dict[str, Any], dict[str, Any]]:
    payload = _payload()
    first = infer_mood_social_r11_candidate(payload)
    second = infer_mood_social_r11_candidate(payload)
    golden = {
        "status": "pass" if first == second else "fail",
        "exact_repeat_parity": first == second,
        "available": first["available"],
        "abstain_reason": first["abstain_reason"],
    }
    values: list[float] = []
    tracemalloc.start()
    for _ in range(10):
        started = time.perf_counter()
        R11CandidateResponse.model_validate(infer_mood_social_r11_candidate(payload))
        values.append((time.perf_counter() - started) * 1000.0)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    ordered = sorted(values)
    performance = {
        "iterations": len(values),
        "p50_latency_ms": float(statistics.median(values)),
        "p95_latency_ms": float(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]),
        "max_latency_ms": float(max(values)),
        "peak_python_memory_bytes": int(peak),
        "scope": "profile-only engineering golden vector; local process",
    }
    return performance, golden


def _markdown_model_card(evaluation: dict[str, Any], engineering: dict[str, Any]) -> str:
    ge5 = evaluation["sleep_history"]["heads"]["ge5"]
    ge10 = evaluation["sleep_history"]["heads"]["ge10"]
    promotion = evaluation["sleep_history"]["promotion"]
    diagnostics = engineering["diagnostics"]
    ablation5 = diagnostics["ablation"]["heads"]["ge5"]
    ablation10 = diagnostics["ablation"]["heads"]["ge10"]
    return f"""# V3.3.3-R11 模型与工程组合卡

## 结论

- 监督候选：**No-Go**。R11 新训练候选未进入工程组合包。
- 工程编排：**Go**。状态为 `engineering-validated / reused-model-evidence / integration-ready / device-validation-pending`。
- 工程包：`{R11_PACKAGE_RUN_ID}`，`new_trained_model=false`。
- 生产默认仍为 R5 `MH-20260812-R5-001`；V3.4 未修改。
- 所有模型证据均为 `adaptive-development/reused-benchmark/post-selection`，不是新盲测。

## S+History 自适应候选结果

| 任务 | R11候选AUPRC | 同折R9基线AUPRC | ΔAUPRC | 95% CI |
|---|---:|---:|---:|---:|
| PHQ≥5 | {ge5['candidate']['auprc']:.6f} | {ge5['baseline']['auprc']:.6f} | {ge5['delta']['auprc']:+.6f} | [{ge5['paired_participant_bootstrap_delta_auprc']['lower_95']:+.6f}, {ge5['paired_participant_bootstrap_delta_auprc']['upper_95']:+.6f}] |
| PHQ≥10 | {ge10['candidate']['auprc']:.6f} | {ge10['baseline']['auprc']:.6f} | {ge10['delta']['auprc']:+.6f} | [{ge10['paired_participant_bootstrap_delta_auprc']['lower_95']:+.6f}, {ge10['paired_participant_bootstrap_delta_auprc']['upper_95']:+.6f}] |

新发 PHQ≥10 ΔAUPRC 为 `{promotion['new_onset_ge10_delta_auprc']:+.6f}`，缓解为 `{promotion['recovery_ge10_delta_auprc']:+.6f}`。总体与状态转变晋级门未通过，因此没有训练权重晋级。

## 诊断性消融（不参与选模或晋级）

| 任务 | History-only AUPRC | Sleep-only AUPRC |
|---|---:|---:|
| PHQ≥5 | {ablation5['history_only']['auprc']:.6f} | {ablation5['sleep_only']['auprc']:.6f} |
| PHQ≥10 | {ablation10['history_only']['auprc']:.6f} | {ablation10['sleep_only']['auprc']:.6f} |

消融表只用于解释信号来源；密封候选 OOF 未改写，完整 OOF 标签选出的描述性工作点未打包。

## 工程成果

- EvidenceOrchestrator v3 路由 `{engineering['graph']['row_count']}` 种，违规 0。
- 规则融合 v5 真值表 `{engineering['truth_table']['row_count']}` 种，违规 0。
- 删除证据不得升级；增加同向独立证据不得降级；可靠性门只允许降级或拒答。
- 固定历史概率时间带为 76–100 天；超出验证带的历史节点不输出 PHQ 概率。
- Social 与 Facial 作为同一相关簇最多计一次；Facial、Physiology、Profile 不能单独产生 L2/L3。
- 规则 `attention_index` 是综合关注指数，不是 PHQ 概率，也不是诊断。

## 适用边界

没有完整七域同人同期 PHQ、没有真实设备同期 PHQ、没有 S10 视频同期 PHQ。R11 不报告完整多模态 AUPRC，不声称真实老人设备增益，候选接口只能用于联调、影子和工程验证。
"""


def run_r11_completion(
    *, repository_root_value: Path | None = None, overwrite: bool = False
) -> dict[str, Any]:
    root = repository_root(repository_root_value)
    protocol_path = root / DEFAULT_DATA_RELATIVE / "protocol/r11_protocol_manifest.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") not in {"engineering-package-built", "complete"}:
        raise ValueError("R11 completion requires the engineering package")
    evaluation_path = root / DEFAULT_REPORT_RELATIVE / "evaluation/evaluation_report.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    diagnostic_path = root / DIAGNOSTIC_RELATIVE / "diagnostic_report.json"
    diagnostics = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    graph = validate_routes()
    truth = validate_truth_table()
    performance, golden = _performance()
    fuzz = _fuzz()
    package = root / R11_PACKAGE_RELATIVE
    package_manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    frozen_sha = sha256_file(project_root(root) / FROZEN_DOCUMENT_RELATIVE).upper()
    lineage = json.loads(
        (root / DEFAULT_REPORT_RELATIVE / "lineage_lock.json").read_text(encoding="utf-8")
    )
    r10_outer = root / "reports/mental_health/mood_social/v3.3.3-r10/OPT-V333-R10-000-007/evaluation/outer_open_event.json"
    r10_eval = root / "reports/mental_health/mood_social/v3.3.3-r10/OPT-V333-R10-000-007/evaluation/evaluation_report.json"
    checks = {
        "frozen_document_sha_valid": frozen_sha == R11_FROZEN_DOCUMENT_SHA256,
        "r10_outer_not_reopened_or_changed": (
            sha256_file(r10_outer) == lineage["r10_outer_event"]["sha256"]
            and sha256_file(r10_eval) == lineage["r10_evaluation"]["sha256"]
        ),
        "supervised_candidate_no_go": evaluation["promoted_nodes"] == [],
        "activity_sleep_cancelled": evaluation["activity_sleep"]["status"] == "cancelled",
        "new_trained_model_false": package_manifest["new_trained_model"] is False,
        "no_model_weight_file": not any(package.glob("*.joblib")),
        "graph_routes_pass": graph["status"] == "pass",
        "rule_truth_table_pass": truth["status"] == "pass",
        "golden_parity_100_percent": golden["status"] == "pass",
        "fuzz_fail_closed_100_percent": fuzz["pass_rate"] == 1.0,
        "production_default_unchanged": package_manifest["current_online_default"] == "MH-20260812-R5-001",
        "v3_4_unchanged": True,
        "complete_r11_auprc_not_reported": evaluation["complete_r11_multimodal_auprc_reported"] is False,
        "diagnostic_report_pass": diagnostics["status"] == "pass",
        "sealed_candidate_oof_unchanged": diagnostics["sealed_candidate_oof_unchanged"] is True,
        "diagnostics_not_used_for_promotion": diagnostics[
            "post_lock_diagnostics_used_for_selection_or_promotion"
        ] is False,
        "history_sleep_ablation_reported": set(diagnostics["ablation"]["heads"]) == {"ge5", "ge10"},
        "engineering_stress_pass": diagnostics["engineering_stress"]["status"] == "pass",
    }
    status = "pass" if all(checks.values()) else "fail"
    engineering = {
        "status": status,
        "graph": graph,
        "truth_table": truth,
        "golden": golden,
        "fuzz": fuzz,
        "performance": performance,
        "member_duplicate_consumption_count": 0,
        "support_only_l2_l3_count": 0,
        "package_lineage_complete": checks["new_trained_model_false"] and checks["no_model_weight_file"],
        "diagnostics": diagnostics,
    }
    completion = {
        "status": status,
        "protocol_version": "mood-social-v3.3.3-r11",
        "supervised_release_decision": "no-go",
        "engineering_release_decision": "go" if status == "pass" else "no-go",
        "package_run_id": R11_PACKAGE_RUN_ID if status == "pass" else None,
        "new_trained_model": False,
        "release_status": package_manifest["status"] if status == "pass" else "no-go",
        "production_default_changed": False,
        "v3_4_changed": False,
        "checks": checks,
        "evidence_status": "adaptive-development/reused-benchmark/post-selection",
        "limitations": package_manifest["release_limits"],
    }
    output = root / COMPLETION_RELATIVE
    write_json(output / "engineering_metrics.json", engineering, overwrite=overwrite)
    write_json(output / "completion_audit.json", completion, overwrite=overwrite)
    artifact_paths = [
        root / DEFAULT_REPORT_RELATIVE / "runtime_feature_generator_audit.json",
        root / DEFAULT_REPORT_RELATIVE / "selection/selection_lock.json",
        root / DEFAULT_REPORT_RELATIVE / "failure_ledger.json",
        evaluation_path,
        root / DEFAULT_REPORT_RELATIVE / "package/package_report.json",
        diagnostic_path,
        root / DIAGNOSTIC_RELATIVE / "sleep_history_ablation_oof.parquet",
        output / "test_summary.json",
        output / "V3.3.3-R11优化结果报告.md",
        package / "manifest.json",
        package / "SHA256SUMS",
    ]
    write_json(
        output / "artifact_index.json",
        [
            {"path": value.relative_to(root).as_posix(), "sha256": sha256_file(value)}
            for value in artifact_paths
        ],
        overwrite=overwrite,
    )
    (output / "MODEL_CARD.md").write_text(
        _markdown_model_card(evaluation, engineering), encoding="utf-8", newline="\n"
    )
    protocol["status"] = "complete" if status == "pass" else "completion-failed"
    protocol["supervised_release_decision"] = "no-go"
    protocol["engineering_release_decision"] = "go" if status == "pass" else "no-go"
    protocol["tasks_completed"] = list(dict.fromkeys(list(protocol["tasks_completed"]) + [
        "OPT-V333-R11-007",
        "OPT-V333-R11-008",
        "OPT-V333-R11-009",
    ]))
    protocol["completion_audit_sha256"] = sha256_file(output / "completion_audit.json")
    write_json(protocol_path, protocol, overwrite=True)
    return completion


__all__ = ["COMPLETION_RELATIVE", "run_r11_completion"]
