"""Independent OPT-V333-R5-008 release and evidence completion audit."""

from __future__ import annotations

from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any
from zoneinfo import ZoneInfo

from elderly_monitoring.modules.mental_health.mood_social.package_selection import (
    load_active_package_selection,
)
from elderly_monitoring.modules.mental_health.mood_social.r4.features import (
    deployable_features_for_signature,
)
from elderly_monitoring.modules.mental_health.mood_social.r5.confirmation import (
    locked_confirmation_recipe,
)
from elderly_monitoring.modules.mental_health.mood_social.r5.contract import (
    R5_CONFIRMATION_SEED,
    R5_FROZEN_DOCUMENT_SHA256,
    R5_PROTOCOL_VERSION,
    assert_r5_deployable_feature_names,
    assert_r5_research_feature_names,
)
from elderly_monitoring.modules.mental_health.mood_social.r5_release import (
    INTEGRATION_ALLOWED_SIGNATURES,
    MOOD_SOCIAL_R5_CANDIDATE_INFER_PATH,
    ONLINE_FALLBACK_RUN_ID,
    R5IntegrationRunner,
    R5_PACKAGE_RELATIVE,
    ROUTE_BLOCKED_SIGNATURES,
    load_r5_fullfit_package,
)


R5_COMPLETION_RELATIVE = Path(
    "reports/mental_health/mood_social/v3.3.3-r5/OPT-V333-R5-008-completion"
)
R5_CONFIRMATION_RELATIVE = Path(
    "reports/mental_health/mood_social/v3.3.3-r5/OPT-V333-R5-006-confirmation"
)
R5_FAILURE_LEDGER_RELATIVE = Path(
    "reports/mental_health/mood_social/v3.3.3-r5/failed_runs.jsonl"
)
R5_FROZEN_DOCUMENT_PARTS = (
    "项目文档",
    "开发协作",
    "情绪与社交关注模块V3.3.3优化开发",
    "00-V3.3.3-r5优化方案冻结说明.md",
)
PRIMARY_TABLES = (
    "multisource_phq_ge10",
    "multisource_phq_ge5_noninferiority",
    "psyche_d_phq_ge10",
    "psyche_d_phq_ge5",
    "joint_101_111_phq_ge10",
)
MAJOR_LODO_SOURCES = (
    "nhanes",
    "nhanes_ssq_2005_2008",
    "psyche_d",
    "shenzhen_elderly",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite r5 completion artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _profile_payload() -> dict[str, Any]:
    target = (
        datetime.now(ZoneInfo("Asia/Shanghai")).date() - timedelta(days=1)
    ).isoformat()
    return {
        "schema_version": "mood_social_infer_request_v3",
        "request_id": "r5_completion_benchmark",
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


def _benchmark(
    package: Path, *, iterations: int
) -> tuple[dict[str, Any], R5IntegrationRunner]:
    if iterations < 10:
        raise ValueError("r5 completion benchmark requires at least 10 iterations")
    start = perf_counter()
    runner = R5IntegrationRunner.from_package(package)
    load_ms = (perf_counter() - start) * 1000.0
    payload = _profile_payload()
    for _ in range(10):
        runner.predict(payload)
    latencies: list[float] = []
    for _ in range(iterations):
        start = perf_counter()
        result = runner.predict(payload)
        latencies.append((perf_counter() - start) * 1000.0)
        if not result["available"] or result["probability_ge10"] > result["probability_ge5"]:
            raise RuntimeError("r5 deterministic benchmark returned an invalid response")
    latencies.sort()
    p95_index = max(0, min(iterations - 1, int(0.95 * iterations) - 1))
    p99_index = max(0, min(iterations - 1, int(0.99 * iterations) - 1))
    return (
        {
            "payload": "synthetic profile-only local replay",
            "iterations": iterations,
            "warmup_iterations": 10,
            "package_bytes": sum(
                path.stat().st_size for path in package.iterdir() if path.is_file()
            ),
            "load_ms": load_ms,
            "p50_ms": median(latencies),
            "p95_ms": latencies[p95_index],
            "p99_ms": latencies[p99_index],
            "max_ms": max(latencies),
            "network_used": False,
            "real_device_claim": False,
        },
        runner,
    )


def run_r5_completion_audit(
    *,
    repository_root: Path,
    project_root: Path | None = None,
    output_directory: Path | None = None,
    overwrite: bool = False,
    benchmark_iterations: int = 500,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    project = Path(project_root).resolve() if project_root else root.parents[1]
    output = (output_directory or root / R5_COMPLETION_RELATIVE).resolve()
    confirmation = root / R5_CONFIRMATION_RELATIVE
    package = root / R5_PACKAGE_RELATIVE
    frozen_document = project.joinpath(*R5_FROZEN_DOCUMENT_PARTS)
    failures: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    require(frozen_document.is_file(), "frozen r5 document is missing")
    frozen_sha = _sha256(frozen_document) if frozen_document.is_file() else None
    require(frozen_sha == R5_FROZEN_DOCUMENT_SHA256.lower(), "frozen r5 document SHA drifted")

    lock = locked_confirmation_recipe(root)
    opened = json.loads((confirmation / "OPENED.json").read_text(encoding="utf-8"))
    evaluation = json.loads(
        (confirmation / "confirmation_evaluation.json").read_text(encoding="utf-8")
    )
    lodo = json.loads(
        (confirmation / "lodo/lodo_report.json").read_text(encoding="utf-8")
    )
    artifact_manifest = json.loads(
        (confirmation / "confirmation_artifact_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    require(opened.get("opened") is True, "confirmation was not opened")
    require(opened.get("reopen_allowed") is False, "confirmation permits reopening")
    require(opened.get("confirmation_seed") == R5_CONFIRMATION_SEED, "confirmation seed drifted")
    require(evaluation.get("status") == "offline_candidate_pass", "offline gate did not pass")
    require(evaluation.get("confirmation_seed") == R5_CONFIRMATION_SEED, "evaluation seed drifted")
    monotonic = evaluation.get("monotonic_violation_count", {})
    require(
        isinstance(monotonic, dict)
        and set(monotonic) == {"multisource_deployable", "psyche_d_single_source_research"}
        and all(value == 0 for value in monotonic.values()),
        "dual-head monotonicity failed",
    )
    require(evaluation.get("offline_gate", {}).get("pass") is True, "offline gate flags failed")
    require(
        artifact_manifest.get("evidence_level")
        == "sealed reused-cohort confirmation; not a new-subject blind test",
        "confirmation evidence level drifted",
    )
    for table_name in PRIMARY_TABLES:
        table = evaluation.get("tables", {}).get(table_name)
        require(isinstance(table, dict), f"missing confirmation table: {table_name}")
        if not isinstance(table, dict):
            continue
        bootstrap = table.get("paired_participant_bootstrap", {})
        require(
            bootstrap.get("requested_repetitions") == 2000
            and bootstrap.get("valid_repetitions") == 2000,
            f"incomplete participant bootstrap: {table_name}",
        )
        require(table.get("common_coverage", 0.0) >= 0.93, f"coverage failed: {table_name}")
    require(
        lodo.get("status") == "complete_with_route_local_warnings",
        "LODO completion status drifted",
    )
    for source in MAJOR_LODO_SOURCES:
        source_result = lodo.get("sources", {}).get(source)
        require(isinstance(source_result, dict), f"missing LODO source: {source}")
        if isinstance(source_result, dict):
            for head in ("ge5", "ge10"):
                result = source_result.get("heads", {}).get(head, {})
                require(
                    result.get("status") == "pass"
                    and result.get("coverage", 0.0) >= 0.93,
                    f"invalid LODO result: {source}/{head}",
                )
    require(
        lodo["sources"]["nhanes"].get("route_warning")
        == "block_related_source_route",
        "NHANES joint-route local block is missing",
    )

    bundle, manifest = load_r5_fullfit_package(package)
    fallback = load_active_package_selection(repository_root=root)
    require(fallback.package_run_id == ONLINE_FALLBACK_RUN_ID, "online fallback identity drifted")
    require(manifest.get("online_default_changed") is False, "r5 changed the online default")
    require(
        manifest.get("status") == "integration-ready/shadow-blocked",
        "package release status drifted",
    )
    require(
        tuple(manifest.get("integration_allowed_signatures", ()))
        == INTEGRATION_ALLOWED_SIGNATURES,
        "integration allowlist drifted",
    )
    require(
        tuple(manifest.get("route_blocked_signatures", ()))
        == ROUTE_BLOCKED_SIGNATURES,
        "route blocklist drifted",
    )
    for signature in ("profile", "sleep", "joint"):
        assert_r5_deployable_feature_names(deployable_features_for_signature(signature))
    assert_r5_research_feature_names(bundle.psyche_features)
    require(
        manifest.get("historical_or_current_phq_used_as_model_input") is False,
        "package claims PHQ leakage",
    )
    require(
        manifest.get("research_track", {}).get("device_api_eligible") is False,
        "PSYCHE research track was incorrectly enabled for devices",
    )

    benchmark, _runner = _benchmark(package, iterations=benchmark_iterations)
    require(benchmark["package_bytes"] < 50 * 1024 * 1024, "package exceeds 50 MiB")
    require(benchmark["p95_ms"] < 50.0, "synthetic profile replay p95 exceeds 50 ms")

    failure_ledger = root / R5_FAILURE_LEDGER_RELATIVE
    ledger_lines = [
        line for line in failure_ledger.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    for line in ledger_lines:
        json.loads(line)
    require(len(ledger_lines) >= 10, "failed-run ledger is incomplete")

    release_decision = {
        "protocol_version": R5_PROTOCOL_VERSION,
        "candidate_package_run_id": manifest["run_id"],
        "offline": "pass",
        "integration": "integration-ready/shadow-blocked",
        "online_default": {
            "changed": False,
            "package_run_id": ONLINE_FALLBACK_RUN_ID,
        },
        "integration_allowed_signatures": list(INTEGRATION_ALLOWED_SIGNATURES),
        "local_route_blocks": {
            "joint": "NHANES LODO ge10 delta AUPRC and AUROC crossed local warning thresholds"
        },
        "research_only": {
            "psyche_d": "no audited real-device feature generator parity"
        },
        "real_device_data_available": False,
        "new_subjects_available": False,
        "shadow_started": False,
        "promotion_authorized": False,
        "next_authorized_step": "local contract replay, then real-device parity when data becomes available",
    }
    completion = {
        "status": "pass" if not failures else "fail",
        "protocol_version": R5_PROTOCOL_VERSION,
        "completed_tasks": [f"OPT-V333-R5-{index:03d}" for index in range(9)],
        "frozen_document": {
            "path": frozen_document.relative_to(project).as_posix(),
            "sha256": frozen_sha,
        },
        "confirmation": {
            "seed": R5_CONFIRMATION_SEED,
            "open_count_policy": "opened once; reopen forbidden",
            "recipe_sha256": opened.get("recipe_sha256"),
            "candidate_recipe": lock["recipe"]["candidates"],
            "evaluation_sha256": _sha256(confirmation / "confirmation_evaluation.json"),
            "lodo_sha256": _sha256(confirmation / "lodo/lodo_report.json"),
        },
        "package": {
            "path": package.relative_to(root).as_posix(),
            "manifest_sha256": _sha256(package / "manifest.json"),
            "model_sha256": _sha256(package / manifest["candidate_model_path"]),
            "checksums_sha256": _sha256(package / "SHA256SUMS"),
        },
        "algorithm_api_path": MOOD_SOCIAL_R5_CANDIDATE_INFER_PATH,
        "release_decision": release_decision,
        "benchmark": benchmark,
        "failed_run_records": len(ledger_lines),
        "network_data_downloads": 0,
        "backend_algorithm_implementation_changes": 0,
        "failures": failures,
    }
    if failures:
        raise RuntimeError("r5 completion audit failed: " + "; ".join(failures))

    _write_json(output / "benchmark.json", benchmark, overwrite=overwrite)
    _write_json(output / "release_decision.json", release_decision, overwrite=overwrite)
    _write_json(output / "completion_audit.json", completion, overwrite=overwrite)
    artifact_rows = []
    for name in ("benchmark.json", "release_decision.json", "completion_audit.json"):
        artifact_rows.append({"path": name, "sha256": _sha256(output / name)})
    final_manifest = {
        "status": "pass",
        "protocol_version": R5_PROTOCOL_VERSION,
        "artifacts": artifact_rows,
    }
    _write_json(output / "artifact_manifest.json", final_manifest, overwrite=overwrite)
    return completion


__all__ = ["R5_COMPLETION_RELATIVE", "run_r5_completion_audit"]
