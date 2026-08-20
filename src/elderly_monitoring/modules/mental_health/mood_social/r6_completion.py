"""Independent R6-008 completion, release-boundary, and evidence audit."""

from __future__ import annotations

from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any
from zoneinfo import ZoneInfo

from elderly_monitoring.modules.mental_health.mood_social.package_selection import load_active_package_selection
from elderly_monitoring.modules.mental_health.mood_social.r6.confirmation import (
    DEFAULT_CONFIRMATION_RELATIVE,
    confirmation_state,
    locked_confirmation_recipe,
)
from elderly_monitoring.modules.mental_health.mood_social.r6.contract import (
    R6_CONFIRMATION_SEED,
    R6_FROZEN_DOCUMENT_SHA256,
    R6_PROTOCOL_VERSION,
)
from elderly_monitoring.modules.mental_health.mood_social.r6_release import (
    MOOD_SOCIAL_R6_CANDIDATE_INFER_PATH,
    ONLINE_FALLBACK_RUN_ID,
    R6IntegrationRunner,
    R6_PACKAGE_RELATIVE,
    load_r6_fullfit_package,
)


R6_COMPLETION_RELATIVE = Path(
    "reports/mental_health/mood_social/v3.3.3-r6/OPT-V333-R6-008-completion"
)
R6_FAILURE_LEDGER_RELATIVE = Path(
    "reports/mental_health/mood_social/v3.3.3-r6/failed_runs.jsonl"
)
PRIMARY_TABLES = (
    "multisource_phq_ge10",
    "multisource_phq_ge5_noninferiority",
    "psyche_d_phq_ge5",
    "psyche_d_phq_ge10",
    "joint_101_111_phq_ge10",
)
MAJOR_SOURCES = ("nhanes", "nhanes_ssq_2005_2008", "psyche_d", "shenzhen_elderly")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite r6 completion artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _payload() -> dict[str, Any]:
    target = (datetime.now(ZoneInfo("Asia/Shanghai")).date() - timedelta(days=1)).isoformat()
    return {
        "schema_version": "mood_social_infer_request_v3",
        "request_id": "r6_completion_benchmark",
        "person_id": "synthetic-local-replay",
        "target_date": target,
        "timezone": "Asia/Shanghai",
        "available_sources": ["camera"],
        "profile": {"age_group": "70_79", "sex": "female"},
        "current_daily_features": {"date": target, "activity": None, "sleep": None, "physiology": None, "social": None},
        "history_daily_features": [],
        "history_attention_indices": [],
        "historical_phq9_assessments": [],
    }


def _benchmark(package: Path, iterations: int) -> dict[str, Any]:
    if iterations < 100:
        raise ValueError("r6 completion benchmark requires at least 100 iterations")
    start = perf_counter()
    runner = R6IntegrationRunner.from_package(package)
    load_ms = (perf_counter() - start) * 1000.0
    payload = _payload()
    for _ in range(10):
        runner.predict(payload)
    latencies: list[float] = []
    for _ in range(iterations):
        start = perf_counter()
        result = runner.predict(payload)
        latencies.append((perf_counter() - start) * 1000.0)
        if not result["available"] or not result["fallback_required"]:
            raise RuntimeError("r6 benchmark violated research fallback policy")
        if result["probability_ge10"] > result["probability_ge5"]:
            raise RuntimeError("r6 benchmark violated dual-head monotonicity")
    latencies.sort()
    return {
        "payload": "synthetic profile-only local replay",
        "iterations": iterations,
        "warmup_iterations": 10,
        "package_bytes": sum(path.stat().st_size for path in package.iterdir() if path.is_file()),
        "load_ms": load_ms,
        "p50_ms": median(latencies),
        "p95_ms": latencies[max(0, min(iterations - 1, int(0.95 * iterations) - 1))],
        "p99_ms": latencies[max(0, min(iterations - 1, int(0.99 * iterations) - 1))],
        "max_ms": max(latencies),
        "network_used": False,
        "real_device_claim": False,
    }


def _failure_ledger(root: Path, *, overwrite: bool) -> tuple[Path, list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    competition = root / "reports/mental_health/mood_social/v3.3.3-r6/OPT-V333-R6-004-competition/failed_runs.json"
    for row in json.loads(competition.read_text(encoding="utf-8")):
        records.append({"task": "OPT-V333-R6-004", **row})
    lodo = root / DEFAULT_CONFIRMATION_RELATIVE / "lodo/failed_runs.json"
    for row in json.loads(lodo.read_text(encoding="utf-8")):
        records.append({"task": "OPT-V333-R6-006-LODO", **row})
    ledger = root / R6_FAILURE_LEDGER_RELATIVE
    if ledger.exists() and not overwrite:
        existing = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip()]
        if existing != records:
            raise ValueError("r6 failure ledger differs from source records")
        return ledger, existing
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in records),
        encoding="utf-8",
    )
    return ledger, records


def _result_markdown(evaluation: dict[str, Any], lodo: dict[str, Any], benchmark: dict[str, Any]) -> str:
    labels = {
        "multisource_phq_ge10": "多源 PHQ≥10",
        "multisource_phq_ge5_noninferiority": "多源 PHQ≥5",
        "psyche_d_phq_ge5": "PSYCHE-D PHQ≥5",
        "psyche_d_phq_ge10": "PSYCHE-D PHQ≥10",
        "joint_101_111_phq_ge10": "Joint 101/111 PHQ≥10",
    }
    rows = []
    for name in PRIMARY_TABLES:
        table = evaluation["tables"][name]
        delta = table["natural"]["delta"]["auprc"]
        ci = table["paired_participant_bootstrap"]["natural_delta"]["auprc"]
        rows.append(
            f"| {labels[name]} | {table['natural']['candidate']['auprc']:.6f} | "
            f"{table['natural']['baseline']['auprc']:.6f} | {delta:+.6f} | "
            f"[{ci['lower_95']:+.6f}, {ci['upper_95']:+.6f}] | {table['natural']['candidate']['auroc']:.6f} |"
        )
    return f"""# V3.3.3-r6 优化结果报告

## 结论

r6 已完成冻结协议中的 R6-000—R6-008，但未通过开发与离线晋级门，状态为 `research-only/integration-ready/shadow-blocked`。五表最佳确认 AUPRC 增益为 `{evaluation['best_confirmation_delta_auprc']:+.6f}`，低于同信息模型继续堆叠停止线 `+0.002`，因此生产默认继续使用 `MH-20260802-013`。

确认结果属于 `sealed reused-cohort confirmation`，不是新受试者盲测；没有真实设备 parity，不能写成线上效果。

## 五张正式表

| 结果表 | r6 AUPRC | 同折 r5 | ΔAUPRC | paired participant bootstrap 95% CI | r6 AUROC |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

所有表 coverage 为 100%，每表完成 2,000 次配对 participant bootstrap，双头 `P10≤P5` 单调违例为 0。Joint 表下降 `{evaluation['tables']['joint_101_111_phq_ge10']['natural']['delta']['auprc']:+.6f}`，所以即使 NHANES LODO 的解除阈值单独通过，也不启用 Joint。

## LODO 与发布边界

- 四主要来源×双头 LODO：`{lodo['head_accounting']['passed']}/8` 成功、`{lodo['head_accounting']['explicit_failures']}` 项显式失败；
- NHANES `≥10`：ΔAUPRC `{lodo['sources']['nhanes']['heads']['ge10']['delta']['auprc']:+.6f}`、ΔAUROC `{lodo['sources']['nhanes']['heads']['ge10']['delta']['auroc']:+.6f}`；
- r6 包为独立显式研究 API，不替换生产 `/infer`；profile/sleep 只做本地影子回放且仍要求 fallback，Joint/no-evidence 直接拒答回退；
- PSYCHE-D `source__` 研究字段不进入设备 API Schema；历史/当前 PHQ 和历史 attention 不进入 V3.3.3 概率。

## 工程结果

- 模型包：`MH-20260812-R6-001`；
- 本地 synthetic profile replay：p95 `{benchmark['p95_ms']:.3f} ms`，包大小 `{benchmark['package_bytes'] / 1024:.1f} KiB`；
- 网络下载、数据下载和真实设备声明均为 0；
- 下一步不是继续堆相同模型，而是等待真实设备特征 parity 或新的信息来源。
"""


def _schema_property_names(value: Any) -> set[str]:
    names: set[str] = set()
    if isinstance(value, dict):
        properties = value.get("properties")
        if isinstance(properties, dict):
            names.update(str(name) for name in properties)
        for child in value.values():
            names.update(_schema_property_names(child))
    elif isinstance(value, list):
        for child in value:
            names.update(_schema_property_names(child))
    return names


def run_r6_completion_audit(
    *,
    repository_root: Path,
    project_root: Path | None = None,
    output_directory: Path | None = None,
    overwrite: bool = False,
    benchmark_iterations: int = 500,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    project = Path(project_root).resolve() if project_root else root.parents[1]
    output = (output_directory or root / R6_COMPLETION_RELATIVE).resolve()
    confirmation = root / DEFAULT_CONFIRMATION_RELATIVE
    package = root / R6_PACKAGE_RELATIVE
    frozen_document = project / "项目文档/开发协作/情绪与社交关注模块V3.3.3优化开发/00-V3.3.3-r6优化方案冻结说明.md"
    failures: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    require(frozen_document.is_file(), "frozen r6 document is missing")
    frozen_sha = _sha256(frozen_document) if frozen_document.is_file() else None
    require(frozen_sha == R6_FROZEN_DOCUMENT_SHA256.lower(), "frozen r6 document SHA drifted")
    state = confirmation_state(root)
    require(state["opened"] and state["sealed"], "r6 confirmation is not sealed")
    opened = json.loads((confirmation / "OPENED.json").read_text(encoding="utf-8"))
    sealed = json.loads((confirmation / "SEALED.json").read_text(encoding="utf-8"))
    evaluation = json.loads((confirmation / "confirmation_evaluation.json").read_text(encoding="utf-8"))
    lodo = json.loads((confirmation / "lodo/lodo_report.json").read_text(encoding="utf-8"))
    lock = locked_confirmation_recipe(root)
    require(opened.get("confirmation_seed") == R6_CONFIRMATION_SEED, "confirmation seed drifted")
    require(opened.get("reopen_allowed") is False and sealed.get("reopen_allowed") is False, "confirmation permits reopening")
    require(evaluation.get("status") == "research_candidate_not_promoted", "r6 promotion decision drifted")
    require(evaluation.get("offline_gate", {}).get("pass") is False, "r6 offline gate unexpectedly passed")
    require(evaluation.get("same_information_model_stacking_stop") is True, "r6 stopping decision drifted")
    require(all(value is False for value in evaluation.get("development_gate_pass", {}).values()), "r6 development gate record drifted")
    require(all(value == 0 for value in evaluation.get("monotonic_violation_count", {}).values()), "r6 monotonicity failed")
    for table_name in PRIMARY_TABLES:
        table = evaluation.get("tables", {}).get(table_name, {})
        bootstrap = table.get("paired_participant_bootstrap", {})
        require(bootstrap.get("requested_repetitions") == 2000 and bootstrap.get("valid_repetitions") == 2000, f"incomplete bootstrap: {table_name}")
        require(table.get("common_coverage") == 1.0, f"coverage drifted: {table_name}")
    require(lodo.get("head_accounting") == {"expected": 8, "passed": 8, "explicit_failures": 0}, "LODO accounting drifted")
    for source in MAJOR_SOURCES:
        for head in ("ge5", "ge10"):
            require(lodo.get("sources", {}).get(source, {}).get("heads", {}).get(head, {}).get("status") == "pass", f"LODO failed: {source}/{head}")

    bundle, manifest = load_r6_fullfit_package(package)
    active = load_active_package_selection(repository_root=root)
    require(active.package_run_id == ONLINE_FALLBACK_RUN_ID, "active production package changed")
    require(manifest.get("status") == "research-only/integration-ready/shadow-blocked", "r6 package status drifted")
    require(manifest.get("promotion_authorized") is False, "r6 package was authorized")
    require(manifest.get("online_default_changed") is False, "r6 changed the online default")
    require(manifest.get("historical_or_current_phq_used_as_model_input") is False, "r6 package claims PHQ input")
    require(manifest.get("history_attention_used_as_model_input") is False, "r6 package claims attention history input")
    require(manifest.get("research_track", {}).get("device_api_eligible") is False, "PSYCHE research fields exposed to API")
    require(manifest.get("joint_route", {}).get("device_api_eligible") is False, "joint route was enabled")
    schema = json.loads((package / manifest["device_schema_path"]).read_text(encoding="utf-8"))
    require(
        not any(name.startswith("source__") for name in _schema_property_names(schema)),
        "PSYCHE source fields leaked into device schema",
    )
    require(len(bundle.psyche_features) == 486, "PSYCHE research feature set drifted")
    benchmark = _benchmark(package, benchmark_iterations)
    require(benchmark["package_bytes"] < 50 * 1024 * 1024, "r6 package exceeds 50 MiB")
    require(benchmark["p95_ms"] < 50.0, "r6 local replay p95 exceeds 50 ms")
    ledger_path, ledger = _failure_ledger(root, overwrite=overwrite)
    require(len(ledger) == 2, "r6 failed-run ledger accounting drifted")
    if failures:
        raise RuntimeError("r6 completion audit failed: " + "; ".join(failures))

    release_decision = {
        "protocol_version": R6_PROTOCOL_VERSION,
        "candidate_package_run_id": manifest["run_id"],
        "offline": "not_promoted",
        "integration": "research-only/integration-ready/shadow-blocked",
        "online_default": {"changed": False, "package_run_id": ONLINE_FALLBACK_RUN_ID},
        "profile_sleep_replay": "explicit research endpoint only; fallback remains required",
        "joint_route": "blocked because confirmation main table regressed despite NHANES LODO threshold pass",
        "psyche_d": "research-only; no device generator parity",
        "new_subjects_available": False,
        "real_device_data_available": False,
        "shadow_started": False,
        "promotion_authorized": False,
        "same_information_model_stacking_stop": True,
        "next_authorized_step": "real-device feature parity or genuinely new information source; do not retune sealed r6",
    }
    completion = {
        "status": "pass",
        "protocol_version": R6_PROTOCOL_VERSION,
        "completed_tasks": [f"OPT-V333-R6-{index:03d}" for index in range(9)],
        "frozen_document": {"path": frozen_document.relative_to(project).as_posix(), "sha256": frozen_sha},
        "confirmation": {
            "seed": R6_CONFIRMATION_SEED,
            "open_count_policy": "opened once and sealed; rerun forbidden",
            "recipe_sha256": opened["recipe_sha256"],
            "artifact_tree_sha256": sealed["artifact_tree_sha256"],
            "evaluation_sha256": _sha256(confirmation / "confirmation_evaluation.json"),
            "lodo_sha256": _sha256(confirmation / "lodo/lodo_report.json"),
            "candidate_recipe": lock["recipe"]["candidates"],
        },
        "metrics": {
            "best_confirmation_delta_auprc": evaluation["best_confirmation_delta_auprc"],
            "five_tables": {
                name: {
                    "candidate_auprc": evaluation["tables"][name]["natural"]["candidate"]["auprc"],
                    "baseline_auprc": evaluation["tables"][name]["natural"]["baseline"]["auprc"],
                    "delta_auprc": evaluation["tables"][name]["natural"]["delta"]["auprc"],
                }
                for name in PRIMARY_TABLES
            },
        },
        "package": {
            "path": package.relative_to(root).as_posix(),
            "manifest_sha256": _sha256(package / "manifest.json"),
            "model_sha256": _sha256(package / manifest["candidate_model_path"]),
            "checksums_sha256": _sha256(package / "SHA256SUMS"),
        },
        "algorithm_api_path": MOOD_SOCIAL_R6_CANDIDATE_INFER_PATH,
        "release_decision": release_decision,
        "benchmark": benchmark,
        "failed_run_ledger": {"path": ledger_path.relative_to(root).as_posix(), "records": len(ledger), "sha256": _sha256(ledger_path)},
        "network_data_downloads": 0,
        "backend_algorithm_implementation_changes": 0,
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "benchmark.json", benchmark, overwrite=overwrite)
    _write_json(output / "release_decision.json", release_decision, overwrite=overwrite)
    _write_json(output / "completion_audit.json", completion, overwrite=overwrite)
    result_report = output / "V3.3.3-r6优化结果报告.md"
    if result_report.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite r6 result report: {result_report}")
    result_report.write_text(_result_markdown(evaluation, lodo, benchmark), encoding="utf-8")
    artifacts = [output / name for name in ("benchmark.json", "release_decision.json", "completion_audit.json", "V3.3.3-r6优化结果报告.md")]
    _write_json(
        output / "artifact_manifest.json",
        {"status": "pass", "protocol_version": R6_PROTOCOL_VERSION, "artifacts": [{"path": path.name, "sha256": _sha256(path)} for path in artifacts]},
        overwrite=overwrite,
    )
    return completion


__all__ = ["R6_COMPLETION_RELATIVE", "run_r6_completion_audit"]
