"""R8 integration package, checksum validation and explicit candidate inference."""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict

from elderly_monitoring.modules.mental_health.mood_social.r7.contract import (
    repository_root,
    sha256_file,
)
from elderly_monitoring.modules.mental_health.mood_social.r7.release import (
    R7_PACKAGE_RELATIVE,
    R7_PACKAGE_RUN_ID,
    infer_mood_social_r7_candidate,
)

from .contract import (
    B0_PACKAGE_RELATIVE,
    DEFAULT_DATA_RELATIVE,
    R8_FROZEN_DOCUMENT_SHA256,
    R8_PROTOCOL_VERSION,
    tree_manifest,
    write_json,
)
from .facial_affect import FacialAffectSupportAdapter
from .rule_fusion import (
    apply_facial_support,
    enumerate_truth_table,
    rule_contract_sha256,
    validate_truth_table,
)
from .schemas import (
    B0_MODEL_VERSION,
    B0_PACKAGE_SHA256,
    B0_RUN_LOCK_SHA256,
    R8CandidateRequest,
    R8_RESPONSE_SCHEMA,
)


R8_PACKAGE_RUN_ID = "MH-20260814-R8-001"
R8_MODEL_VERSION = "mood-social-current-state-v3.3.3-r8"
R8_PACKAGE_RELATIVE = Path(
    f"models/mental_health/mood_social/v3.3.3-r8/packages/{R8_PACKAGE_RUN_ID}"
)
R8_REPORT_RELATIVE = Path(
    "reports/mental_health/mood_social/v3.3.3-r8/OPT-V333-R8-007-package"
)
R8_EVALUATION_REPORT_RELATIVE = Path(
    "reports/mental_health/mood_social/v3.3.3-r8/OPT-V333-R8-006-evaluation/evaluation_report.json"
)
MOOD_SOCIAL_R8_CANDIDATE_INFER_PATH = (
    "/v1/mental-health/mood-social/r8-candidate/infer"
)


class R8PackageError(RuntimeError):
    pass


class R8CandidateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str
    request_id: str
    person_id: str
    target_date: str
    model_version: str
    package_run_id: str
    core_package_run_id: str
    available: bool
    current_state_only: bool
    diagnosis: bool
    expert_assessments: dict[str, Any]
    passive_current_state: dict[str, Any]
    facial_affect_assessment: dict[str, Any]
    interaction_enhanced_current_state: dict[str, Any]
    history_informed_integrated_state: dict[str, Any]
    operational_attention_state: dict[str, Any]
    rule_fusion: dict[str, Any]
    probability_used_sources: list[str]
    supporting_evidence_sources: list[str]
    observed_sources: list[str]
    fallback_model_version: str | None
    fallback_reason: str | None
    release_status: str
    limitations: list[str]


def _checksums(package: Path) -> None:
    paths = sorted(
        (
            path
            for path in package.iterdir()
            if path.is_file() and path.name != "SHA256SUMS"
        ),
        key=lambda path: path.name,
    )
    (package / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in paths),
        encoding="ascii",
        newline="\n",
    )


def build_r8_package(
    *, repository_root_value: Path | None = None, overwrite: bool = False
) -> dict[str, Any]:
    root = repository_root(repository_root_value)
    protocol = root / DEFAULT_DATA_RELATIVE / "r8_protocol_manifest.json"
    if not protocol.is_file():
        raise FileNotFoundError("R8-000 protocol must complete before packaging")
    protocol_record = json.loads(protocol.read_text(encoding="utf-8"))
    if (
        protocol_record.get("status") != "pass"
        or protocol_record.get("frozen_document_sha256")
        != R8_FROZEN_DOCUMENT_SHA256
    ):
        raise R8PackageError("R8 protocol manifest is not valid")
    evaluation = root / R8_EVALUATION_REPORT_RELATIVE
    if not evaluation.is_file():
        raise FileNotFoundError("R8-006 evaluation must complete before packaging")
    evaluation_record = json.loads(evaluation.read_text(encoding="utf-8"))
    if evaluation_record.get("status") != "pass":
        raise R8PackageError("R8 evaluation did not pass")

    package = root / R8_PACKAGE_RELATIVE
    report = root / R8_REPORT_RELATIVE
    if package.exists() and not overwrite:
        raise FileExistsError(package)
    package.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    truth = enumerate_truth_table()
    truth_audit = validate_truth_table(truth)
    if truth_audit["status"] != "pass":
        raise R8PackageError("R8 truth-table audit failed")
    r7_tree_sha, _ = tree_manifest(root / R7_PACKAGE_RELATIVE)
    b0_tree_sha, _ = tree_manifest(root / B0_PACKAGE_RELATIVE)
    manifest = {
        "version": "mood-social-r8-integration-package-v1",
        "protocol_version": R8_PROTOCOL_VERSION,
        "run_id": R8_PACKAGE_RUN_ID,
        "model_version": R8_MODEL_VERSION,
        "status": "offline-validated/integration-ready/device-validation-pending",
        "promotion_authorized": False,
        "online_default_changed": False,
        "current_online_default": "MH-20260812-R5-001",
        "current_state_only": True,
        "forecast_changed": False,
        "core": {
            "package_run_id": R7_PACKAGE_RUN_ID,
            "path": R7_PACKAGE_RELATIVE.as_posix(),
            "tree_sha256": r7_tree_sha,
            "no_facial_equivalence_required": True,
        },
        "facial_affect": {
            "model_version": B0_MODEL_VERSION,
            "path": B0_PACKAGE_RELATIVE.as_posix(),
            "tree_sha256": b0_tree_sha,
            "manifest_self_sha256": B0_PACKAGE_SHA256,
            "run_lock_payload_sha256": B0_RUN_LOCK_SHA256,
            "phq_probability_available": False,
            "paired_phq_video_training_count": 0,
        },
        "rule_fusion": {
            "trained": False,
            "version": "mood-social-r8-rule-fusion-v2",
            "rule_contract_sha256": rule_contract_sha256(),
            "truth_table_rows": 625,
            "attention_index_is_phq_probability": False,
        },
        "expert_optimization": evaluation_record.get("expert_decisions", {}),
        "fallback": {
            "core": R7_PACKAGE_RUN_ID,
            "production": "MH-20260812-R5-001",
            "legacy": "MH-20260802-013",
        },
        "release_limits": [
            "no real-device facial-affect validation",
            "no paired facial-video/current-PHQ training data",
            "no complete-five-domain R8 AUPRC",
            "R5 production default remains unchanged",
        ],
        "protocol_manifest_sha256": sha256_file(protocol),
        "evaluation_report_sha256": sha256_file(evaluation),
    }
    write_json(package / "manifest.json", manifest, overwrite=True)
    write_json(
        package / "request_schema.json",
        R8CandidateRequest.model_json_schema(),
        overwrite=True,
    )
    write_json(package / "rule_truth_table.json", truth, overwrite=True)
    write_json(package / "rule_truth_table_audit.json", truth_audit, overwrite=True)
    _checksums(package)
    result = {
        "status": manifest["status"],
        "package_run_id": R8_PACKAGE_RUN_ID,
        "manifest_sha256": sha256_file(package / "manifest.json"),
        "checksums_sha256": sha256_file(package / "SHA256SUMS"),
        "rule_contract_sha256": rule_contract_sha256(),
        "truth_table_rows": len(truth),
        "online_default_changed": False,
    }
    write_json(report / "package_build_report.json", result, overwrite=True)
    return result


def _verify(package: Path) -> dict[str, Any]:
    try:
        lines = (package / "SHA256SUMS").read_text(encoding="ascii").splitlines()
    except OSError as exc:
        raise R8PackageError("R8 package checksums unavailable") from exc
    listed: set[str] = set()
    for line in lines:
        try:
            expected, name = line.split("  ", 1)
            path = (package / name).resolve()
            path.relative_to(package.resolve())
        except (ValueError, OSError) as exc:
            raise R8PackageError("invalid R8 checksum entry") from exc
        if name in listed or not path.is_file() or sha256_file(path) != expected:
            raise R8PackageError(f"R8 checksum failed: {name}")
        listed.add(name)
    expected_files = {
        path.name
        for path in package.iterdir()
        if path.is_file() and path.name != "SHA256SUMS"
    }
    if listed != expected_files:
        raise R8PackageError("R8 checksum coverage incomplete")
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("run_id") != R8_PACKAGE_RUN_ID
        or manifest.get("rule_fusion", {}).get("rule_contract_sha256")
        != rule_contract_sha256()
        or manifest.get("facial_affect", {}).get("manifest_self_sha256")
        != B0_PACKAGE_SHA256
    ):
        raise R8PackageError("R8 package manifest contract drift")
    return manifest


@lru_cache(maxsize=4)
def _load_verified_package(package_text: str) -> dict[str, Any]:
    return _verify(Path(package_text))


def infer_mood_social_r8_candidate(
    request: R8CandidateRequest | Mapping[str, Any],
    *,
    package_directory: Path | None = None,
    r7_package_directory: Path | None = None,
    facial_adapter: FacialAffectSupportAdapter | None = None,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[6]
    package = (package_directory or root / R8_PACKAGE_RELATIVE).resolve()
    manifest = _load_verified_package(str(package))
    parsed = (
        request
        if isinstance(request, R8CandidateRequest)
        else R8CandidateRequest.model_validate(request)
    )
    core = infer_mood_social_r7_candidate(
        parsed.r7_payload(),
        package_directory=r7_package_directory or root / R7_PACKAGE_RELATIVE,
    )
    facial = (facial_adapter or FacialAffectSupportAdapter()).assess(
        parsed.facial_affect_observations,
        person_id=parsed.person_id,
        inference_cutoff=parsed.inference_cutoff,
        authorized=parsed.facial_affect_authorized,
    )
    integrated = apply_facial_support(core, facial)
    expert_assessments = dict(core["expert_assessments"])
    expert_assessments["facial_affect"] = facial
    observed_sources = list(core.get("probability_used_sources", [])) + list(
        core.get("supporting_evidence_sources", [])
    )
    if parsed.facial_affect_observations:
        observed_sources.append("s10_facial_affect")
    observed_sources = list(dict.fromkeys(observed_sources))
    supporting = list(core.get("supporting_evidence_sources", []))
    if facial.get("available"):
        supporting.append("s10_facial_affect")
    supporting = list(dict.fromkeys(supporting))
    operational = integrated["operational_attention_state"]
    available = bool(operational.get("available"))
    facial_vote = bool(integrated["rule_fusion"]["facial_vote_eligible"])
    return {
        "schema_version": R8_RESPONSE_SCHEMA,
        "request_id": parsed.request_id,
        "person_id": parsed.person_id,
        "target_date": parsed.target_date.isoformat(),
        "model_version": R8_MODEL_VERSION,
        "package_run_id": R8_PACKAGE_RUN_ID,
        "core_package_run_id": R7_PACKAGE_RUN_ID,
        "available": available,
        "current_state_only": True,
        "diagnosis": False,
        "expert_assessments": expert_assessments,
        **integrated,
        "probability_used_sources": list(core.get("probability_used_sources", [])),
        "supporting_evidence_sources": supporting,
        "observed_sources": observed_sources,
        "fallback_model_version": (
            core.get("fallback_model_version") if not available else None
        ),
        "fallback_reason": (
            core.get("fallback_reason")
            if not available
            else "facial_unavailable_or_ineligible_r7_equivalent"
            if not facial_vote
            else None
        ),
        "release_status": manifest["status"],
        "limitations": [
            "offline/reused evidence; no real-device validation",
            "facial-affect classes are not PHQ probabilities",
            "no paired PHQ-video data; no PHQ facial expert was trained",
            "rule-derived attention is not medical diagnosis",
        ],
    }


__all__ = [
    "MOOD_SOCIAL_R8_CANDIDATE_INFER_PATH",
    "R8CandidateRequest",
    "R8CandidateResponse",
    "R8PackageError",
    "R8_MODEL_VERSION",
    "R8_PACKAGE_RELATIVE",
    "R8_PACKAGE_RUN_ID",
    "build_r8_package",
    "infer_mood_social_r8_candidate",
]
