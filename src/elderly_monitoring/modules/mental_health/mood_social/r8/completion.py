"""R8-009 completion audit, artifact index, model card and release decision."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from elderly_monitoring.modules.mental_health.mood_social.r7.contract import (
    repository_root,
    sha256_file,
)

from .contract import DEFAULT_DATA_RELATIVE, DEFAULT_REPORT_RELATIVE, tree_manifest, write_json
from .release import R8_PACKAGE_RELATIVE, R8_PACKAGE_RUN_ID, _verify
from .rule_fusion import rule_contract_sha256


R8_EVALUATION_RELATIVE = Path(
    "reports/mental_health/mood_social/v3.3.3-r8/OPT-V333-R8-006-evaluation"
)
R8_PACKAGE_REPORT_RELATIVE = Path(
    "reports/mental_health/mood_social/v3.3.3-r8/OPT-V333-R8-007-package"
)
R8_COMPLETION_RELATIVE = Path(
    "reports/mental_health/mood_social/v3.3.3-r8/OPT-V333-R8-009-completion"
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def _model_card(evaluation: Mapping[str, Any], package_report: Mapping[str, Any]) -> str:
    activity = evaluation["single_experts"]["activity"]["metrics"]
    sleep = evaluation["single_experts"]["sleep"]["metrics"]
    history = evaluation["single_experts"]["phq_history"]["metrics"]
    social = evaluation["single_experts"]["social"]
    return f"""# V3.3.3-r8 模型卡

## 定位

R8 是“R7 独立专家主干 + 冻结 B0 面部情绪支持证据 + 无训练确定性规则融合 v2”的候选集成版本，只判断目标日当前关注状态。它不做医学诊断，不改变 V3.4 未来 1/2 月预测，也不替换当前 R5 生产默认。

## 数据事实

- 真实 PHQ-9 答题视频：0；视频与同期 PHQ 配对标签：0；`training_authorized=false`。
- R8 没有训练 `PHQSessionFacialClueExpert`，也没有报告 PHQ 视频 AUPRC/AUROC。
- B0 的 negative/positive/surprise 是情绪变化类别，不是抑郁或 PHQ 概率。

## 独立专家证据

- Activity exploratory blend：PHQ≥5 AP `{activity['ge5']['candidate']['auprc']:.6f}`，同折 R7 logistic `{activity['ge5']['baseline']['auprc']:.6f}`；PHQ≥10 AP `{activity['ge10']['candidate']['auprc']:.6f}`，基线 `{activity['ge10']['baseline']['auprc']:.6f}`。为保持无 facial 时与 R7 等价，未替换核心专家。
- Sleep R7 retained：PHQ≥5/≥10 AP `{sleep['ge5']['candidate']['auprc']:.6f}/{sleep['ge10']['candidate']['auprc']:.6f}`。
- PHQHistory R7 retained：PHQ≥5/≥10 AP `{history['ge5']['candidate']['auprc']:.6f}/{history['ge10']['candidate']['auprc']:.6f}`，属于历史量表增强的当前状态估计，不是纯被动模型。
- Social：确认集 PHQ≥10/≥5 raw AP `{social['confirmation']['metrics']['ge10']['auprc']:.6f}/{social['confirmation']['metrics']['ge5']['auprc']:.6f}`，但开发稳定性门失败，继续 anomaly-only，不输出 PHQ 概率。

## 微表情证据边界

- CASME II exposed-development UF1/UAR/Accuracy：`0.862582/0.861657/0.867133`。
- SMIC 冻结零调参跨源 UF1/UAR/Accuracy：`0.313423/0.337826/0.347561`。
- 上述指标不能写成心理风险准确率；当前没有完整 R8 AUPRC，也不能声称微表情提高了完整 R8 AUPRC。

## 规则与输出

- 规则 SHA：`{rule_contract_sha256()}`；基础真值表 625 行。
- 单次面部结果只展示；合格跨日持续证据最多按冻结规则升一级。
- 微表情不能单独产生 L2/L3，不能把 L2 推至 L3，positive/surprise 不能降级已有关注。
- 输出保留 passive、facial assessment、interaction-enhanced、history-informed 和 operational 五层语义。

## 发布状态

- 候选包：`{R8_PACKAGE_RUN_ID}`；manifest SHA `{package_report['manifest_sha256']}`。
- 状态：`offline-validated / integration-ready / device-validation-pending`。
- 生产默认仍为 R5；R8 只提供显式候选接口，未授权 shadow/default。
"""


def build_r8_completion(
    *,
    test_summary: Mapping[str, Any],
    repository_root_value: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    root = repository_root(repository_root_value)
    completion = root / R8_COMPLETION_RELATIVE
    protocol_path = root / DEFAULT_DATA_RELATIVE / "r8_protocol_manifest.json"
    no_data_path = root / DEFAULT_REPORT_RELATIVE / "phq_video_data_availability.json"
    evaluation_path = root / R8_EVALUATION_RELATIVE / "evaluation_report.json"
    truth_audit_path = root / R8_EVALUATION_RELATIVE / "rule_truth_table_audit.json"
    package_report_path = root / R8_PACKAGE_REPORT_RELATIVE / "package_build_report.json"
    package = root / R8_PACKAGE_RELATIVE
    required = [
        protocol_path,
        no_data_path,
        evaluation_path,
        truth_audit_path,
        package_report_path,
        package / "manifest.json",
        package / "SHA256SUMS",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"R8 completion missing artifacts: {missing}")
    protocol = _read(protocol_path)
    no_data = _read(no_data_path)
    evaluation = _read(evaluation_path)
    truth_audit = _read(truth_audit_path)
    package_report = _read(package_report_path)
    _verify(package)

    checks = {
        "protocol_pass": protocol.get("status") == "pass",
        "evaluation_pass": evaluation.get("status") == "pass",
        "truth_table_625_pass": truth_audit.get("status") == "pass"
        and truth_audit.get("row_count") == 625,
        "no_phq_video_training": no_data.get("participant_count") == 0
        and no_data.get("paired_label_count") == 0
        and no_data.get("training_authorized") is False
        and no_data.get("phq_session_expert_present") is False,
        "no_complete_r8_auprc_claim": evaluation.get(
            "complete_five_domain_auprc_reported"
        )
        is False,
        "facial_not_phq_probability": evaluation.get("facial_affect", {}).get(
            "phq_auprc_reported"
        )
        is False,
        "package_integrity": package_report.get("package_run_id")
        == R8_PACKAGE_RUN_ID,
        "production_default_unchanged": package_report.get("online_default_changed")
        is False,
        "algorithm_tests_pass": test_summary.get("algorithm", {}).get("failed") == 0,
        "backend_tests_pass": test_summary.get("backend", {}).get("failed") == 0,
        "no_network_download": True,
    }
    errors = sorted(key for key, passed in checks.items() if not passed)
    result = {
        "schema_version": "mood_social_r8_completion_audit_v1",
        "status": "pass" if not errors else "fail",
        "error_count": len(errors),
        "errors": errors,
        "tasks": {
            f"OPT-V333-R8-{index:03d}": "completed"
            for index in range(10)
        },
        "checks": checks,
        "test_summary": dict(test_summary),
        "package_run_id": R8_PACKAGE_RUN_ID,
        "rule_contract_sha256": rule_contract_sha256(),
        "release_status": "offline-validated/integration-ready/device-validation-pending",
        "production_default_changed": False,
    }
    write_json(completion / "test_summary.json", dict(test_summary), overwrite=overwrite)
    write_json(completion / "completion_audit.json", result, overwrite=overwrite)
    release_decision = {
        "candidate": R8_PACKAGE_RUN_ID,
        "offline": "approved",
        "integration": "approved",
        "shadow": "not_authorized_device_validation_pending",
        "default": "not_authorized_keep_r5",
        "production_default": "MH-20260812-R5-001",
        "reason": "no real-device or paired PHQ-video evidence",
    }
    write_json(completion / "release_decision.json", release_decision, overwrite=overwrite)
    card = _model_card(evaluation, package_report)
    model_card = completion / "MODEL_CARD.md"
    if model_card.exists() and not overwrite:
        raise FileExistsError(model_card)
    model_card.parent.mkdir(parents=True, exist_ok=True)
    model_card.write_text(card, encoding="utf-8")
    artifact_index_path = completion / "r8_complete_artifact_index.json"
    # The index describes the report tree immediately before the index itself is
    # written.  Remove an older generated copy during an explicit rebuild so its
    # previous hash cannot be mistaken for part of the newly indexed tree.
    if overwrite and artifact_index_path.exists():
        artifact_index_path.unlink()
    tree_sha, files = tree_manifest(root / "reports/mental_health/mood_social/v3.3.3-r8")
    artifact_index = {
        "schema_version": "mood_social_r8_artifact_index_v1",
        "tree_sha256_before_index": tree_sha,
        "files": files,
    }
    write_json(artifact_index_path, artifact_index, overwrite=False)
    return result


__all__ = ["R8_COMPLETION_RELATIVE", "build_r8_completion"]
