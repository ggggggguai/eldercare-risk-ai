"""R11 no-new-weight engineering package and explicit candidate inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from elderly_monitoring.modules.mental_health.mood_social.r9.release import (
    R9_PACKAGE_RELATIVE,
    R9_PACKAGE_RUN_ID,
    infer_mood_social_r9_candidate,
)

from .contract import (
    DEFAULT_DATA_RELATIVE,
    DEFAULT_REPORT_RELATIVE,
    R11_FROZEN_DOCUMENT_SHA256,
    R11_PROTOCOL_VERSION,
    repository_root,
    sha256_file,
    write_json,
)
from .evidence_orchestrator import (
    graph_contract_sha256,
    orchestrate_evidence,
    validate_routes,
)
from .reliability_gate import reliability_contract_sha256
from .rule_fusion import (
    fuse_r11_current_state,
    rule_contract_sha256,
    validate_truth_table,
)
from .schemas import (
    R11CandidateRequest,
    R11CandidateResponse,
    R11_RESPONSE_SCHEMA,
)


R11_PACKAGE_RUN_ID = "MH-20260814-R11-001"
R11_MODEL_VERSION = "mood-social-current-state-v3.3.3-r11"
R11_PACKAGE_RELATIVE = Path(
    f"models/mental_health/mood_social/v3.3.3-r11/packages/{R11_PACKAGE_RUN_ID}"
)
R11_PACKAGE_REPORT_RELATIVE = DEFAULT_REPORT_RELATIVE / "package"
MOOD_SOCIAL_R11_CANDIDATE_INFER_PATH = "/v1/mental-health/mood-social/r11-candidate/infer"
R11_RELEASE_STATUS = (
    "engineering-validated/reused-model-evidence/integration-ready/device-validation-pending"
)


class R11PackageError(RuntimeError):
    pass


def _write_checksums(package: Path) -> None:
    files = sorted(
        (value for value in package.iterdir() if value.is_file() and value.name != "SHA256SUMS"),
        key=lambda value: value.name,
    )
    (package / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(value)}  {value.name}\n" for value in files),
        encoding="ascii",
        newline="\n",
    )


def _verify_checksums(package: Path) -> None:
    checksum = package / "SHA256SUMS"
    if not checksum.is_file():
        raise R11PackageError("R11 SHA256SUMS missing")
    for line in checksum.read_text(encoding="ascii").splitlines():
        digest, name = line.split("  ", 1)
        path = package / name
        if not path.is_file() or sha256_file(path) != digest:
            raise R11PackageError(f"R11 package checksum mismatch: {name}")


def build_r11_package(
    *, repository_root_value: Path | None = None, overwrite: bool = False
) -> dict[str, Any]:
    root = repository_root(repository_root_value)
    protocol_path = root / DEFAULT_DATA_RELATIVE / "protocol/r11_protocol_manifest.json"
    evaluation_path = root / DEFAULT_REPORT_RELATIVE / "evaluation/evaluation_report.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    if protocol.get("status") not in {"evaluation-complete", "engineering-package-built", "complete"}:
        raise R11PackageError("R11 evaluation has not completed")
    if evaluation.get("promoted_nodes"):
        raise R11PackageError("engineering-only builder refuses unexpected trained R11 node")
    graph_audit, truth_audit = validate_routes(), validate_truth_table()
    if graph_audit["status"] != "pass" or truth_audit["status"] != "pass":
        raise R11PackageError("R11 graph/rule property audit failed")
    package = root / R11_PACKAGE_RELATIVE
    if package.exists() and not overwrite:
        raise FileExistsError(package)
    package.mkdir(parents=True, exist_ok=True)
    r9_manifest = root / R9_PACKAGE_RELATIVE / "manifest.json"
    r9_checksums = root / R9_PACKAGE_RELATIVE / "SHA256SUMS"
    manifest = {
        "version": "mood-social-r11-engineering-package-v1",
        "protocol_version": R11_PROTOCOL_VERSION,
        "run_id": R11_PACKAGE_RUN_ID,
        "model_version": R11_MODEL_VERSION,
        "status": R11_RELEASE_STATUS,
        "new_trained_model": False,
        "model_evidence_source": "R9/R7/R8 frozen packages",
        "graph_rule_source": "R10 audited architecture; R11 v3/v5 reliability hardening",
        "current_state_only": True,
        "diagnosis": False,
        "online_default_changed": False,
        "current_online_default": "MH-20260812-R5-001",
        "r9_core": {
            "run_id": R9_PACKAGE_RUN_ID,
            "path": R9_PACKAGE_RELATIVE.as_posix(),
            "manifest_sha256": sha256_file(r9_manifest),
            "checksums_sha256": sha256_file(r9_checksums),
        },
        "frozen_document_sha256": R11_FROZEN_DOCUMENT_SHA256,
        "evaluation_report_sha256": sha256_file(evaluation_path),
        "graph_contract_sha256": graph_contract_sha256(),
        "rule_contract_sha256": rule_contract_sha256(),
        "reliability_contract_sha256": reliability_contract_sha256(),
        "supervised_release_decision": "no-go",
        "engineering_release_decision": "go",
        "release_limits": [
            "adaptive/reused/post-selection evidence only",
            "no complete seven-domain same-person current-PHQ cohort",
            "no real-device current-PHQ observations",
            "no paired S10 facial-video/current-PHQ observations",
            "operational attention is rule-derived and not a PHQ probability",
        ],
    }
    write_json(package / "manifest.json", manifest, overwrite=True)
    write_json(package / "request_schema.json", R11CandidateRequest.model_json_schema(), overwrite=True)
    write_json(package / "response_schema.json", R11CandidateResponse.model_json_schema(), overwrite=True)
    write_json(
        package / "contracts.json",
        {
            "graph_contract_sha256": graph_contract_sha256(),
            "rule_contract_sha256": rule_contract_sha256(),
            "reliability_contract_sha256": reliability_contract_sha256(),
            "graph_audit": graph_audit,
            "truth_table_audit": truth_audit,
        },
        overwrite=True,
    )
    _write_checksums(package)
    _verify_checksums(package)
    result = {
        "status": R11_RELEASE_STATUS,
        "package_run_id": R11_PACKAGE_RUN_ID,
        "new_trained_model": False,
        "package_sha256": sha256_file(package / "SHA256SUMS"),
        "file_count": len(list(package.iterdir())),
        "graph_routes": graph_audit["row_count"],
        "truth_table_rows": truth_audit["row_count"],
    }
    write_json(root / R11_PACKAGE_REPORT_RELATIVE / "package_report.json", result, overwrite=overwrite)
    protocol["status"] = "engineering-package-built"
    protocol["engineering_package_run_id"] = R11_PACKAGE_RUN_ID
    protocol["tasks_completed"] = list(dict.fromkeys(list(protocol["tasks_completed"]) + [
        "OPT-V333-R11-004",
        "OPT-V333-R11-005",
        "OPT-V333-R11-006-engineering-evaluation",
        "OPT-V333-R11-007-package",
    ]))
    write_json(protocol_path, protocol, overwrite=True)
    return result


def _load_manifest(package: Path) -> dict[str, Any]:
    _verify_checksums(package)
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("run_id") != R11_PACKAGE_RUN_ID or manifest.get("new_trained_model") is not False:
        raise R11PackageError("R11 package identity or model-weight semantics drift")
    if manifest.get("graph_contract_sha256") != graph_contract_sha256():
        raise R11PackageError("R11 graph contract drift")
    if manifest.get("rule_contract_sha256") != rule_contract_sha256():
        raise R11PackageError("R11 rule contract drift")
    if manifest.get("reliability_contract_sha256") != reliability_contract_sha256():
        raise R11PackageError("R11 reliability contract drift")
    return manifest


def _selected_output(
    selected: str | None, experts: Mapping[str, Mapping[str, Any]], joints: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any] | None:
    if selected == "sleep_history_r9":
        value = joints.get("sleep_history")
    elif selected == "activity_sleep_r9":
        value = joints.get("activity_sleep")
    elif selected in {"activity", "sleep", "phq_history"}:
        value = experts.get(selected)
    else:
        return None
    if not value or not value.get("available"):
        return None
    return {
        "node": selected,
        "probability_phq_ge5": value.get("probability_phq_ge5"),
        "probability_phq_ge10": value.get("probability_phq_ge10"),
        "semantics": "selected supervised current-state screening probability",
    }


def infer_mood_social_r11_candidate(
    request: R11CandidateRequest | Mapping[str, Any],
    *,
    package_directory: Path | None = None,
    r9_package_directory: Path | None = None,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[6]
    package = (package_directory or root / R11_PACKAGE_RELATIVE).resolve()
    manifest = _load_manifest(package)
    parsed = request if isinstance(request, R11CandidateRequest) else R11CandidateRequest.model_validate(request)
    core = infer_mood_social_r9_candidate(
        parsed.r9_payload(),
        package_directory=r9_package_directory or root / R9_PACKAGE_RELATIVE,
    )
    experts = {name: dict(value) for name, value in core["expert_assessments"].items()}
    joints = {name: dict(value) for name, value in core["joint_expert_assessments"].items()}
    age = core.get("freshness", {}).get("last_phq_age_days")
    route = orchestrate_evidence(experts, joints, history_age_days=age)
    history_indices = [value.model_dump(mode="python") for value in parsed.history_attention_indices]
    fused = fuse_r11_current_state(
        experts, joints, route, history_attention_indices=history_indices
    )
    selected = route.selected_probability_node
    selected_output = _selected_output(selected, experts, joints)
    used_decisions = [
        route.reliability[name]
        for name in (
            (["sleep_history"] if selected == "sleep_history_r9" else ["activity_sleep"] if selected == "activity_sleep_r9" else list(route.probability_used_sources))
        )
        if name in route.reliability
    ]
    statuses = [value["reliability_status"] for value in used_decisions]
    reliability_status = (
        "abstain" if not statuses or all(value == "abstain" for value in statuses)
        else "degraded" if any(value == "degraded" for value in statuses)
        else "usable"
    )
    operational = fused["operational_attention_state"]
    available = bool(operational["available"])
    response = {
        "schema_version": R11_RESPONSE_SCHEMA,
        "request_id": parsed.request_id,
        "person_id": parsed.person_id,
        "target_date": parsed.target_date.isoformat(),
        "model_version": R11_MODEL_VERSION,
        "package_run_id": R11_PACKAGE_RUN_ID,
        "core_package_run_id": R9_PACKAGE_RUN_ID,
        "new_trained_model": False,
        "model_evidence_source": "R9/R7/R8 frozen packages",
        "available": available,
        "current_state_only": True,
        "diagnosis": False,
        "expert_assessments": experts,
        "joint_expert_assessments": joints,
        "passive_current_state": core["passive_current_state"],
        "history_informed_current_state": operational,
        "operational_attention_state": operational,
        "facial_affect_assessment": core["facial_affect_assessment"],
        "selected_probability_output": selected_output,
        "selected_probability_node": selected,
        "probability_used_sources": list(route.probability_used_sources),
        "supporting_evidence_sources": list(route.supporting_nodes),
        "consumed_nodes": list(route.consumed_nodes),
        "suppressed_nodes": list(route.suppressed_nodes),
        "reliability_status": reliability_status,
        "domain_compatibility": {
            name: bool(value["domain_compatibility"]) for name, value in route.reliability.items()
        },
        "quality_reasons": {
            name: list(value["quality_reasons"]) for name, value in route.reliability.items()
        },
        "evidence_graph": route.to_dict(),
        "rule_fusion": fused["rule_fusion"],
        "fusion_mode": route.fusion_mode,
        "evidence_signature": route.evidence_signature,
        "observed_sources": list(core.get("observed_sources", [])),
        "quality": {
            **dict(core.get("quality", {})),
            "reliability_gate_version": "mood-social-r11-reliability-gate-v1",
            "no_cross_modal_imputation": True,
        },
        "freshness": {
            "history_probability_interval_days": [76, 100],
            "history_rule_only_interval_days": [14, 75],
            "history_background_interval_days": [101, 180],
            "last_phq_age_days": age,
        },
        "lineage": {
            "r11_package_run_id": R11_PACKAGE_RUN_ID,
            "r9_core_package_run_id": R9_PACKAGE_RUN_ID,
            "graph_contract_sha256": graph_contract_sha256(),
            "rule_contract_sha256": rule_contract_sha256(),
            "reliability_contract_sha256": reliability_contract_sha256(),
            "package_sha256": sha256_file(package / "SHA256SUMS"),
            "evaluation_report_sha256": manifest["evaluation_report_sha256"],
        },
        "fallback_chain": {
            "selected": selected,
            "supervised_r11": "no-go/no-new-weight",
            "r9": R9_PACKAGE_RUN_ID,
            "production": "MH-20260812-R5-001 unchanged",
            "legacy": "MH-20260802-013",
            "core_fallbacks": core.get("fallback_chain", {}),
        },
        "abstain_reason": None if available else "no_usable_probability_or_dynamic_evidence_after_reliability_gate",
        "release_status": R11_RELEASE_STATUS,
        "limitations": [
            "adaptive/reused/post-selection evidence; not a blind test",
            "new_trained_model=false; model metrics remain attributable to R9/R7/R8 nodes",
            "no complete seven-domain R11 AUPRC",
            "no real-device current-PHQ validation",
            "facial affect is support-only and not a PHQ probability",
            "rule-derived attention is not diagnosis or calibrated PHQ probability",
        ],
    }
    return R11CandidateResponse.model_validate(response).model_dump(mode="json")


__all__ = [
    "MOOD_SOCIAL_R11_CANDIDATE_INFER_PATH",
    "R11_MODEL_VERSION",
    "R11_PACKAGE_RELATIVE",
    "R11_PACKAGE_RUN_ID",
    "R11PackageError",
    "build_r11_package",
    "infer_mood_social_r11_candidate",
]
