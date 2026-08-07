from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = (
    PROJECT_ROOT
    / "configs/open_source/microexpression_open_source_lock_v1.json"
)
DEFAULT_NOTICE = PROJECT_ROOT / "THIRD_PARTY_NOTICES_MICROEXPRESSION.md"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "artifacts/microexpression/oss_audit/open_source_audit_v1.json"
)
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit pinned micro-expression open-source reference snapshots."
    )
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--notice", type=Path, default=DEFAULT_NOTICE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def project_path(relative: str) -> Path:
    candidate = (PROJECT_ROOT / relative).resolve()
    try:
        candidate.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"Path escapes project root: {relative}") from exc
    return candidate


def _check_file(
    relative: str,
    expected_sha256: str,
    *,
    project_id: str,
    role: str,
    issues: list[dict[str, str]],
    audited_files: list[dict[str, Any]],
) -> Path | None:
    path = project_path(relative)
    if not path.is_file():
        issues.append(
            {
                "project": project_id,
                "code": "missing_file",
                "path": relative,
                "detail": role,
            }
        )
        return None
    actual = sha256_file(path)
    audited_files.append(
        {
            "project": project_id,
            "role": role,
            "path": relative,
            "sha256": actual,
            "size_bytes": path.stat().st_size,
        }
    )
    if actual != expected_sha256.lower():
        issues.append(
            {
                "project": project_id,
                "code": "sha256_mismatch",
                "path": relative,
                "detail": f"expected={expected_sha256.lower()} actual={actual}",
            }
        )
    return path


def audit_project(
    project: Mapping[str, Any],
    issues: list[dict[str, str]],
    audited_files: list[dict[str, Any]],
) -> dict[str, Any]:
    project_id = str(project["id"])
    commit = str(project["commit"])
    if not COMMIT_PATTERN.fullmatch(commit):
        issues.append(
            {
                "project": project_id,
                "code": "invalid_commit",
                "path": "",
                "detail": commit,
            }
        )

    license_record = project["license"]
    _check_file(
        str(license_record["path"]),
        str(license_record["sha256"]),
        project_id=project_id,
        role="license",
        issues=issues,
        audited_files=audited_files,
    )
    for reference in project["reference_files"]:
        _check_file(
            str(reference["path"]),
            str(reference["sha256"]),
            project_id=project_id,
            role="audit_only_reference",
            issues=issues,
            audited_files=audited_files,
        )

    evidence_results: list[dict[str, Any]] = []
    for evidence in project["evidence_checks"]:
        relative = str(evidence["path"])
        path = project_path(relative)
        missing_patterns: list[str] = []
        if not path.is_file():
            missing_patterns = [str(item) for item in evidence["required_patterns"]]
        else:
            text = path.read_text(encoding="utf-8")
            missing_patterns = [
                str(pattern)
                for pattern in evidence["required_patterns"]
                if str(pattern) not in text
            ]
        if missing_patterns:
            issues.append(
                {
                    "project": project_id,
                    "code": "missing_evidence_pattern",
                    "path": relative,
                    "detail": repr(missing_patterns),
                }
            )
        evidence_results.append(
            {
                "path": relative,
                "finding": str(evidence["finding"]),
                "required_pattern_count": len(evidence["required_patterns"]),
                "missing_patterns": missing_patterns,
                "status": "pass" if not missing_patterns else "fail",
            }
        )

    return {
        "id": project_id,
        "name": str(project["name"]),
        "repository": str(project["repository"]),
        "commit": commit,
        "license": str(license_record["spdx"]),
        "usage_scope": str(project["usage_scope"]),
        "forbidden_usage": list(project["forbidden_usage"]),
        "reference_file_count": len(project["reference_files"]),
        "evidence": evidence_results,
    }


def run_audit(lock_path: Path, notice_path: Path) -> dict[str, Any]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("schema_version") != "microexpression_open_source_lock_v1":
        raise ValueError("Unexpected open-source lock schema")
    if lock.get("task_id") != "OSS-ME-001":
        raise ValueError("Open-source lock has the wrong task id")
    if lock.get("reference_snapshots_executable") is not False:
        raise ValueError("Reference snapshots must be audit-only")

    issues: list[dict[str, str]] = []
    audited_files: list[dict[str, Any]] = []
    projects = [
        audit_project(project, issues, audited_files)
        for project in lock["projects"]
    ]

    notice_text = ""
    if notice_path.is_file():
        notice_text = notice_path.read_text(encoding="utf-8")
    else:
        issues.append(
            {
                "project": "all",
                "code": "missing_notice",
                "path": notice_path.as_posix(),
                "detail": "THIRD_PARTY notice is required",
            }
        )
    missing_notice_tokens = [
        str(token)
        for token in lock["required_notice_tokens"]
        if str(token) not in notice_text
    ]
    if missing_notice_tokens:
        issues.append(
            {
                "project": "all",
                "code": "notice_missing_tokens",
                "path": notice_path.as_posix(),
                "detail": repr(missing_notice_tokens),
            }
        )

    return {
        "schema_version": "microexpression_open_source_audit_v1",
        "task_id": "OSS-ME-001",
        "status": "pass" if not issues else "fail",
        "issue_count": len(issues),
        "issues": issues,
        "reference_snapshots_executable": False,
        "project_count": len(projects),
        "projects": projects,
        "audited_file_count": len(audited_files),
        "audited_files": audited_files,
        "notice": {
            "path": notice_path.resolve().as_posix(),
            "exists": notice_path.is_file(),
            "sha256": sha256_file(notice_path) if notice_path.is_file() else None,
            "missing_tokens": missing_notice_tokens,
        },
        "inputs": {
            "lock_path": lock_path.resolve().as_posix(),
            "lock_sha256": sha256_file(lock_path),
            "audit_script_path": Path(__file__).resolve().as_posix(),
            "audit_script_sha256": sha256_file(Path(__file__)),
        },
    }


def write_json(payload: Mapping[str, Any], output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    return sha256_file(output_path)


def main() -> int:
    args = parse_args()
    result = run_audit(args.lock.resolve(), args.notice.resolve())
    result["output_sha256"] = write_json(result, args.output.resolve())
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
