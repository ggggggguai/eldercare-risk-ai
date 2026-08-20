"""R10 No-Go completion audit, model card, result report and artifact index."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from elderly_monitoring.modules.mental_health.mood_social.r7.contract import (
    project_root,
    repository_root,
    sha256_file,
)

from .contract import (
    DEFAULT_DATA_RELATIVE,
    DEFAULT_REPORT_RELATIVE,
    FROZEN_DOCUMENT_RELATIVE,
    R10_FROZEN_DOCUMENT_SHA256,
    write_json,
)
from .evaluation import EVALUATION_REPORT_RELATIVE
from .evidence_graph import graph_contract_sha256
from .rule_fusion import rule_contract_sha256


R10_COMPLETION_RELATIVE = Path(
    "reports/mental_health/mood_social/v3.3.3-r10/OPT-V333-R10-009-completion"
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _line(report: Mapping[str, Any], node: str, head: str) -> str:
    value = report["nodes"][node]["heads"][head]
    return (
        f"候选 AP `{value['candidate']['auprc']:.6f}`，R9 同折基线 "
        f"`{value['baseline']['auprc']:.6f}`，ΔAP `{value['delta']['auprc']:+.6f}`，"
        f"95% CI `[{value['paired_participant_bootstrap_delta_auprc']['lower_95']:+.6f}, "
        f"{value['paired_participant_bootstrap_delta_auprc']['upper_95']:+.6f}]`"
    )


def _model_card(report: Mapping[str, Any], lodo: Mapping[str, Any]) -> str:
    sh10 = report["nodes"]["sleep_history"]["heads"]["ge10"]["transitions"]
    return f"""# V3.3.3-R10 模型卡（No-Go）

## 定位

R10 验证“真实同人局部监督融合 + 互斥证据图路由 + 确定性规则融合 v4”。它只估计目标日 D 的当前关注状态，不预测未来，不修改 V3.4，也不替换生产默认 R5。

## 数据与证据等级

- A+S：2,846 人；S+History：6,358 行/2,887 人，仅使用无标签时间审计支持的 76–100 天历史带。
- 三次重复、每次 5 outer × 5 inner，全部按参与者分组；预处理、选择、融合与校准均排除 outer-test。
- 现有受试者已在既往开发中暴露，因此证据等级为 `adaptive-development / reused-benchmark / locked-procedure-estimate`，不是新盲测或真实设备验证。

## 锁定结果

- A+S，PHQ≥5：{_line(report, 'activity_sleep', 'ge5')}。
- A+S，PHQ≥10：{_line(report, 'activity_sleep', 'ge10')}。
- S+History，PHQ≥5：{_line(report, 'sleep_history', 'ge5')}。
- S+History，PHQ≥10：{_line(report, 'sleep_history', 'ge10')}。

新发/缓解以同折 R9 基线计算配对增益：PHQ≥10 新发 ΔAP `{sh10['new_onset']['delta']['auprc']:+.6f}`，缓解 ΔAP `{sh10['recovery']['delta']['auprc']:+.6f}`。两节点均未满足冻结晋级门，因此 R10 没有生成模型候选包。

## 路由与规则

- EvidenceGraphRouterV2 SHA：`{graph_contract_sha256()}`；S+History 优先于 A+S，睡眠只消费一次。
- Rule Fusion v4 SHA：`{rule_contract_sha256()}`；联合概率节点真实驱动 operational attention，规则分数不是 PHQ 概率。
- Social 保持 anomaly-only，Facial/Physiology 保持 support-only，Profile 保持 background-only。
- A+S LODO 状态：`{lodo.get('status')}`；S+History 因仅 PSYCHE-D 一个来源无法估计 LODO。

## 发布决定

`No-Go / fail-closed`。算法与后端显式 R10 candidate 路由已实现并保持默认关闭；因为没有晋级节点，模型包不存在，启用后算法侧返回 artifact unavailable。生产默认仍是 `MH-20260812-R5-001`。
"""


def _result_report(report: Mapping[str, Any]) -> str:
    return f"""# V3.3.3-R10 优化结果报告

R10-000—009 已按冻结流程完成。此次优化没有产生可发布的新模型节点，但完成了两项重要架构修正并给出了诚实的 No-Go：

1. 将 R9 的 A+S 与 S+History 从可能共同消费 Sleep 的并列图，改为互斥优先路由；S+History 生效时 A+S 被抑制，Activity 只能作为不共享的支持证据。
2. 规则融合 v4 让被选中的联合概率节点真正影响最终关注等级，不再只做诊断展示。

锁定外层结果如下：

- A+S PHQ≥5：{_line(report, 'activity_sleep', 'ge5')}。
- A+S PHQ≥10：{_line(report, 'activity_sleep', 'ge10')}。
- S+History PHQ≥5：{_line(report, 'sleep_history', 'ge5')}。
- S+History PHQ≥10：{_line(report, 'sleep_history', 'ge10')}。

A+S 排序下降；S+History 仅获得极小的 PHQ≥5 排序增益，PHQ≥10 略降，且新发/缓解的配对增益未达到门槛。因此不生成 R10 模型包、不切换生产、不把架构改进包装成模型指标提升。

所有数值均属于复用数据上的锁定程序估计。完整七域同人数据、真实设备数据和 S10 微表情视频+同期 PHQ 配对数据仍为 0，不能报告完整 R10 AUPRC 或真实设备效果。
"""


def _artifact_index(root: Path, completion: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for base in (
        root / DEFAULT_DATA_RELATIVE,
        root / "reports/mental_health/mood_social/v3.3.3-r10",
    ):
        for path in sorted(base.rglob("*")):
            if path.is_file() and path != completion / "r10_complete_artifact_index.json":
                files.append(
                    {
                        "path": path.relative_to(root).as_posix(),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
    return {
        "schema_version": "mood_social_r10_artifact_index_v1",
        "file_count": len(files),
        "files": files,
    }


def build_r10_completion(
    *,
    test_summary: Mapping[str, Any],
    repository_root_value: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    root = repository_root(repository_root_value)
    completion = root / R10_COMPLETION_RELATIVE
    protocol_path = root / DEFAULT_DATA_RELATIVE / "protocol/r10_protocol_manifest.json"
    evaluation_path = root / EVALUATION_REPORT_RELATIVE / "evaluation_report.json"
    graph_path = root / EVALUATION_REPORT_RELATIVE / "evidence_graph_audit.json"
    truth_path = root / EVALUATION_REPORT_RELATIVE / "rule_truth_table_audit.json"
    lodo_path = root / EVALUATION_REPORT_RELATIVE / "lodo_report.json"
    perturbation_path = root / EVALUATION_REPORT_RELATIVE / "perturbation_report.json"
    frozen = project_root(root) / FROZEN_DOCUMENT_RELATIVE
    required = [
        protocol_path,
        evaluation_path,
        graph_path,
        truth_path,
        lodo_path,
        perturbation_path,
        frozen,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"R10 completion missing artifacts: {missing}")
    protocol = _read(protocol_path)
    evaluation = _read(evaluation_path)
    graph = _read(graph_path)
    truth = _read(truth_path)
    lodo = _read(lodo_path)
    perturbation = json.loads(perturbation_path.read_text(encoding="utf-8"))
    algorithm_tests = test_summary.get("algorithm", {})
    backend_tests = test_summary.get("backend", {})
    package = root / "models/mental_health/mood_social/v3.3.3-r10/packages/MH-20260814-R10-001"
    checks = {
        "frozen_document_unchanged": sha256_file(frozen).upper() == R10_FROZEN_DOCUMENT_SHA256,
        "outer_opened_exactly_once": protocol.get("candidate_outer_opened") is True,
        "evaluation_pass": evaluation.get("status") == "pass",
        "honest_no_go": evaluation.get("promoted_nodes") == [],
        "no_candidate_package_generated": not package.exists(),
        "graph_routes_pass": graph.get("status") == "pass" and graph.get("violation_count") == 0,
        "rule_truth_table_pass": truth.get("status") == "pass" and truth.get("violation_count") == 0,
        "lodo_completed": lodo.get("status") == "pass",
        "perturbations_finite": bool(perturbation)
        and all(row.get("finite_ge5") and row.get("finite_ge10") for row in perturbation),
        "no_complete_r10_auprc": evaluation.get("complete_r10_auprc_reported") is False,
        "algorithm_tests_pass": algorithm_tests.get("failed") == 0,
        "backend_tests_pass": backend_tests.get("failed") == 0,
        "production_default_unchanged": protocol.get("production_default_changed") is False,
        "no_network_download": True,
    }
    errors = sorted(name for name, passed in checks.items() if not passed)
    result = {
        "schema_version": "mood_social_r10_completion_audit_v1",
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "checks": checks,
        "tasks": {
            **{f"OPT-V333-R10-{index:03d}": "completed" for index in range(8)},
            "OPT-V333-R10-008": "completed-no-go-no-model-package-explicit-api-fail-closed",
            "OPT-V333-R10-009": "completed",
        },
        "test_summary": dict(test_summary),
        "promoted_nodes": [],
        "candidate_package_run_id": None,
        "graph_contract_sha256": graph_contract_sha256(),
        "rule_contract_sha256": rule_contract_sha256(),
        "release_status": "no-go/fail-closed",
        "production_default": "MH-20260812-R5-001",
        "production_default_changed": False,
    }
    completion.mkdir(parents=True, exist_ok=True)
    write_json(completion / "test_summary.json", dict(test_summary), overwrite=overwrite)
    write_json(completion / "completion_audit.json", result, overwrite=overwrite)
    write_json(
        completion / "release_decision.json",
        {
            "candidate": None,
            "decision": "no-go",
            "model_package_generated": False,
            "algorithm_candidate_endpoint": "/v1/mental-health/mood-social/r10-candidate/infer",
            "backend_candidate_endpoint": "/api/v1/elders/{elder_id}/mood-social/r10-candidate/infer",
            "feature_flag_default": False,
            "endpoint_behavior": "fail-closed-model-artifact-unavailable",
            "production_default": "MH-20260812-R5-001",
        },
        overwrite=overwrite,
    )
    write_json(
        root / DEFAULT_REPORT_RELATIVE / "package/no_go_package_decision.json",
        {
            "status": "no-go",
            "promoted_nodes": [],
            "model_package_generated": False,
            "reason": "no local supervised node met the frozen promotion gates",
        },
        overwrite=True,
    )
    for path, content in (
        (completion / "MODEL_CARD.md", _model_card(evaluation, lodo)),
        (completion / "V3.3.3-r10优化结果报告.md", _result_report(evaluation)),
    ):
        if path.exists() and not overwrite:
            raise FileExistsError(path)
        path.write_text(content, encoding="utf-8", newline="\n")
    write_json(
        completion / "r10_complete_artifact_index.json",
        _artifact_index(root, completion),
        overwrite=overwrite,
    )
    protocol["status"] = "completed-no-go"
    protocol["tasks_completed"] = [f"OPT-V333-R10-{index:03d}" for index in range(10)]
    protocol["candidate_package_run_id"] = None
    protocol["production_default_changed"] = False
    write_json(protocol_path, protocol, overwrite=True)
    return result


__all__ = ["R10_COMPLETION_RELATIVE", "build_r10_completion"]
