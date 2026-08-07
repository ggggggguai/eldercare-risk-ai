from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_SCRIPT = PROJECT_ROOT / "scripts/audit_microexpression_open_source.py"
SOURCE_LOCK = (
    PROJECT_ROOT
    / "configs/open_source/microexpression_open_source_lock_v1.json"
)


def _run_audit(lock_path: Path, output_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(AUDIT_SCRIPT),
            "--lock",
            str(lock_path),
            "--output",
            str(output_path),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_open_source_audit_passes_and_detects_hash_tampering(
    tmp_path: Path,
) -> None:
    passing_output = tmp_path / "passing.json"
    passing = _run_audit(SOURCE_LOCK, passing_output)
    assert passing.returncode == 0, passing.stderr or passing.stdout
    passing_payload = json.loads(passing_output.read_text(encoding="utf-8"))
    assert passing_payload["status"] == "pass"
    assert passing_payload["issue_count"] == 0
    assert passing_payload["project_count"] == 2
    assert passing_payload["audited_file_count"] == 11
    assert passing_payload["reference_snapshots_executable"] is False

    tampered_lock = json.loads(SOURCE_LOCK.read_text(encoding="utf-8"))
    tampered_lock["projects"][0]["reference_files"][0]["sha256"] = "0" * 64
    tampered_lock_path = tmp_path / "tampered_lock.json"
    tampered_lock_path.write_text(
        json.dumps(tampered_lock, indent=2) + "\n", encoding="utf-8"
    )
    failing_output = tmp_path / "failing.json"
    failing = _run_audit(tampered_lock_path, failing_output)
    assert failing.returncode == 1
    failing_payload = json.loads(failing_output.read_text(encoding="utf-8"))
    assert failing_payload["status"] == "fail"
    assert any(
        issue["code"] == "sha256_mismatch"
        for issue in failing_payload["issues"]
    )
