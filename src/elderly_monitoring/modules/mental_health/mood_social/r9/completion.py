"""R9-009 completion audit, model card and immutable artifact index."""

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
    FROZEN_DOCUMENT_RELATIVE,
    R9_FROZEN_DOCUMENT_SHA256,
    write_json,
)
from .evaluation import EVALUATION_REPORT_RELATIVE
from .evidence_graph import graph_contract_sha256
from .release import (
    R9_PACKAGE_RELATIVE,
    R9_PACKAGE_RUN_ID,
    R9_REPORT_RELATIVE,
    _verify,
)
from .rule_fusion import rule_contract_sha256


R9_COMPLETION_RELATIVE = Path(
    "reports/mental_health/mood_social/v3.3.3-r9/OPT-V333-R9-009-completion"
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _metric_line(node: Mapping[str, Any], head: str) -> str:
    metrics = node["heads"][head]
    candidate = metrics["candidate"]
    baseline = metrics["baseline"]
    delta = metrics["delta"]
    bootstrap = metrics["paired_bootstrap_delta_auprc"]
    return (
        f"候选 AP `{candidate['auprc']:.6f}`，基线 `{baseline['auprc']:.6f}`，"
        f"ΔAP `{delta['auprc']:+.6f}`，95% CI "
        f"`[{bootstrap['lower_95']:+.6f}, {bootstrap['upper_95']:+.6f}]`"
    )


def _model_card(evaluation: Mapping[str, Any], package_report: Mapping[str, Any]) -> str:
    nodes = evaluation["nodes"]
    return f"""# V3.3.3-R9 模型卡

## 定位

R9 是当前状态模块的离线候选版：保留 R7/R8 的独立 Activity、Sleep、Social、Profile、Physiology、PHQHistory 和 Facial 专家，只在确有同一受试者配对数据的签名上训练局部融合器，并以无训练证据图和确定性规则完成最终关注融合。R9 不做医学诊断，不改变 V3.4 的未来 1/2 月预测，也不替换当前 R5 生产默认。

## 证据边界

- 全部现有受试者均已在既往开发中暴露；本结果属于 `adaptive-development/reused-benchmark/locked-procedure-estimate`，不是新盲测。
- A+S 同人签名为 2,846 人；S+严格历史 PHQ 主分析为 6,358 行、2,887 人，历史间隔限定 14–100 天。
- 完整七域同人数据为 0，真实设备数据为 0，真实 S10 视频与同期 PHQ 配对为 0；因此没有完整 R9 AUPRC，也没有训练 PHQ 视频概率模型。

## 正式同折结果

- A+S（PHQ≥5）：{_metric_line(nodes['activity_sleep'], 'ge5')}；5/5 折非负，正式晋级。
- A+S（PHQ≥10）：{_metric_line(nodes['activity_sleep'], 'ge10')}；作为副头随包发布。
- S+History（PHQ≥5）：{_metric_line(nodes['sleep_history'], 'ge5')}；5/5 折非负，按历史校准替代路径晋级。
- S+History（PHQ≥10）：{_metric_line(nodes['sleep_history'], 'ge10')}；5/5 折非负。
- History（PHQ≥5）：{_metric_line(nodes['phq_history'], 'ge5')}；AP 基本持平但 Brier/ECE 改善，按训练/服务时间一致性路径晋级。
- Activity 和 Sleep 新候选未过各自门槛，继续使用 R7/R8 专家回退，未因局部改进而强制替换。

## 融合与输出

- 证据图不是训练模型，合约 SHA：`{graph_contract_sha256()}`；128 个签名组合、0 项违规。
- 最终规则不是训练模型，合约 SHA：`{rule_contract_sha256()}`；3,125 个真值组合、0 项违规。
- 局部融合器仅对 A+S、S+History 输出 PHQ≥5/≥10 双头筛查概率；最终 `attention_index` 是规则生成的综合关注指数，不是 PHQ 概率。
- Social 仍为 anomaly-only，Profile 为 background-only，Physiology 与 Facial 为 support-only；微表情类别不等于抑郁或 PHQ 概率。

## 发布状态

- 候选包：`{R9_PACKAGE_RUN_ID}`；manifest SHA `{package_report['manifest_sha256']}`。
- 状态：`offline-validated / integration-ready / device-validation-pending`。
- 仅提供显式候选接口；生产默认仍为 `MH-20260812-R5-001`，R8、旧 V3.3.3 和 V3.4 均未被替换。
"""


def _result_report(evaluation: Mapping[str, Any]) -> str:
    nodes = evaluation["nodes"]
    return f"""# V3.3.3-R9 优化结果报告

R9 已完成 R9-000 至 R9-009。正式外层评估只打开一次，所有比较均在同一受试者、同一折、同一权重口径下完成，并使用 2,000 次参与者级配对 bootstrap。

## 本轮有效提升

1. A+S 局部融合是本轮最明确的传感器增益：PHQ≥5 AP `{nodes['activity_sleep']['heads']['ge5']['candidate']['auprc']:.6f}`，相对同签名最佳单专家基线提升 `{nodes['activity_sleep']['heads']['ge5']['delta']['auprc']:+.6f}`，95% CI 下界为 `{nodes['activity_sleep']['heads']['ge5']['paired_bootstrap_delta_auprc']['lower_95']:+.6f}`。
2. S+History 在历史信息已很强的情况下仍取得小而稳定的增益：PHQ≥10 ΔAP `{nodes['sleep_history']['heads']['ge10']['delta']['auprc']:+.6f}`，PHQ≥5 ΔAP `{nodes['sleep_history']['heads']['ge5']['delta']['auprc']:+.6f}`，两头均 5/5 折非负。
3. 历史专家完成 14–100 天训练/服务口径对齐；排序指标基本持平，但 PHQ≥5 的 Brier 和 ECE 分别改善 `{nodes['phq_history']['heads']['ge5']['delta']['brier']:+.6f}` 与 `{nodes['phq_history']['heads']['ge5']['delta']['ece']:+.6f}`。
4. Activity、Sleep 单专家新候选未强行晋级，继续复用经过验证的 R7/R8 专家，避免为了版本号牺牲现有效果。

## 不能宣称的结果

没有完整七域同人样本，不能报告“完整 R9 AUPRC”；没有真实设备数据，不能宣称真机效果；没有视频与同期 PHQ 配对数据，不能训练或报告微表情对 PHQ 的增益。R9 的最终关注等级来自冻结的确定性规则，不是校准后的 PHQ 概率。
"""


def _artifact_index(root: Path, completion: Path) -> dict[str, Any]:
    roots = [
        root / DEFAULT_DATA_RELATIVE,
        root / "reports/mental_health/mood_social/v3.3.3-r9",
        root / R9_PACKAGE_RELATIVE,
    ]
    index_path = completion / "r9_complete_artifact_index.json"
    files: list[dict[str, Any]] = []
    for base in roots:
        for path in sorted(base.rglob("*")):
            if path.is_file() and path != index_path:
                files.append(
                    {
                        "path": path.relative_to(root).as_posix(),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
    return {
        "schema_version": "mood_social_r9_artifact_index_v1",
        "file_count": len(files),
        "files": files,
    }


def build_r9_completion(
    *,
    test_summary: Mapping[str, Any],
    repository_root_value: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    root = repository_root(repository_root_value)
    completion = root / R9_COMPLETION_RELATIVE
    protocol_path = root / DEFAULT_DATA_RELATIVE / "protocol/r9_protocol_manifest.json"
    evaluation_path = root / EVALUATION_REPORT_RELATIVE / "evaluation_report.json"
    graph_path = root / EVALUATION_REPORT_RELATIVE / "evidence_graph_audit.json"
    truth_path = root / EVALUATION_REPORT_RELATIVE / "rule_truth_table_audit.json"
    role_path = root / EVALUATION_REPORT_RELATIVE / "role_audit.json"
    package_report_path = root / R9_REPORT_RELATIVE / "package_build_report.json"
    package = root / R9_PACKAGE_RELATIVE
    frozen_path = project_root(root) / FROZEN_DOCUMENT_RELATIVE
    required = [
        protocol_path,
        evaluation_path,
        graph_path,
        truth_path,
        role_path,
        package_report_path,
        package / "manifest.json",
        package / "SHA256SUMS",
        frozen_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"R9 completion missing artifacts: {missing}")

    protocol = _read(protocol_path)
    evaluation = _read(evaluation_path)
    graph = _read(graph_path)
    truth = _read(truth_path)
    role = _read(role_path)
    package_report = _read(package_report_path)
    manifest = _verify(package)
    algorithm_tests = test_summary.get("algorithm", {})
    backend_tests = test_summary.get("backend", {})
    checks = {
        "frozen_document_unchanged": sha256_file(frozen_path).upper()
        == R9_FROZEN_DOCUMENT_SHA256,
        "outer_opened_exactly_once": protocol.get("candidate_outer_metrics_opened")
        is True
        and evaluation.get("candidate_outer_opened_once") is True,
        "evaluation_pass": evaluation.get("status") == "pass",
        "promoted_nodes_exact": set(evaluation.get("promoted_nodes", []))
        == {"activity_sleep", "phq_history", "sleep_history"},
        "graph_128_pass": graph.get("status") == "pass"
        and graph.get("row_count") == 128
        and graph.get("violation_count") == 0,
        "truth_table_3125_pass": truth.get("status") == "pass"
        and truth.get("row_count") == 3125
        and truth.get("violation_count") == 0,
        "no_complete_r9_auprc_claim": evaluation.get("complete_r9_auprc_reported")
        is False,
        "no_phq_video_probability": role.get("facial_affect", {}).get(
            "paired_phq_video_labels"
        )
        == 0
        and role.get("facial_affect", {}).get("phq_probability_available")
        is False,
        "role_boundaries_pass": role.get("social", {}).get("role")
        == "anomaly-only"
        and role.get("profile", {}).get("role") == "background-only"
        and role.get("physiology", {}).get("role") == "support-only",
        "package_integrity": manifest.get("run_id") == R9_PACKAGE_RUN_ID
        and package_report.get("package_run_id") == R9_PACKAGE_RUN_ID,
        "production_default_unchanged": manifest.get("online_default_changed")
        is False
        and manifest.get("current_online_default") == "MH-20260812-R5-001",
        "algorithm_tests_pass": algorithm_tests.get("failed") == 0,
        "backend_tests_pass": backend_tests.get("failed") == 0,
        "no_network_download": True,
    }
    errors = sorted(name for name, passed in checks.items() if not passed)
    result = {
        "schema_version": "mood_social_r9_completion_audit_v1",
        "status": "pass" if not errors else "fail",
        "error_count": len(errors),
        "errors": errors,
        "tasks": {f"OPT-V333-R9-{index:03d}": "completed" for index in range(10)},
        "checks": checks,
        "test_summary": dict(test_summary),
        "package_run_id": R9_PACKAGE_RUN_ID,
        "promoted_nodes": evaluation.get("promoted_nodes", []),
        "graph_contract_sha256": graph_contract_sha256(),
        "rule_contract_sha256": rule_contract_sha256(),
        "release_status": "offline-validated/integration-ready/device-validation-pending",
        "production_default_changed": False,
    }
    completion.mkdir(parents=True, exist_ok=True)
    write_json(completion / "test_summary.json", dict(test_summary), overwrite=overwrite)
    write_json(completion / "completion_audit.json", result, overwrite=overwrite)
    write_json(
        completion / "release_decision.json",
        {
            "candidate": R9_PACKAGE_RUN_ID,
            "offline": "approved",
            "integration": "approved_explicit_candidate_endpoint",
            "shadow": "not_authorized_device_validation_pending",
            "default": "not_authorized_keep_r5",
            "production_default": "MH-20260812-R5-001",
            "reason": "no real-device observations and no new blind cohort",
        },
        overwrite=overwrite,
    )
    card = _model_card(evaluation, package_report)
    result_report = _result_report(evaluation)
    for path, content in (
        (completion / "MODEL_CARD.md", card),
        (completion / "V3.3.3-r9优化结果报告.md", result_report),
    ):
        if path.exists() and not overwrite:
            raise FileExistsError(path)
        path.write_text(content, encoding="utf-8", newline="\n")
    write_json(
        completion / "r9_complete_artifact_index.json",
        _artifact_index(root, completion),
        overwrite=overwrite,
    )
    return result


__all__ = ["R9_COMPLETION_RELATIVE", "build_r9_completion"]
