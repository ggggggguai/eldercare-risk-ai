from __future__ import annotations

import hashlib
import http.server
import json
import subprocess
import sys
import tempfile
import threading
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from unittest.mock import patch
from urllib.request import urlopen

from elderly_monitoring.modules.fall_risk.kinecal import (
    age_group_mismatches,
    build_download_summary,
    download_kinecal,
    download_file,
    parse_directory_files,
    participant_number,
    read_register_csv,
    select_risk_group_participants,
    sha256_file,
    verify_kinecal_download,
)


REGISTER_CSV = """part_id,group,age,sex,height,weight,BMI,recorded_in_the_lab,clinically-at-risk
SPPB2,HA,54,m,1.82,73.0,22.0,1,0
SPPB40,FHs,67,f,1.65,60.0,22.0,0,1
SPPB62,FHm,64,f,1.63,65.3,24.6,0,0
SPPB307,NF,>89,f,1.60,55.0,21.5,0,1
"""

DIRECTORY_HTML = """<html><body><pre>
<a href="../">../</a>
<a href="100.txt">100.txt</a> 15-Apr-2019 15:07 1673
<a href="101.txt">101.txt</a> 15-Apr-2019 15:07 1681
<a href="ignore.bin">ignore.bin</a> 15-Apr-2019 15:07 434176
</pre></body></html>
"""


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return


class _RecordingHttpClient:
    def __init__(self) -> None:
        self.urls: list[str] = []

    def read(self, url: str, *, timeout: float) -> bytes:
        self.urls.append(url)
        with urlopen(url, timeout=timeout) as response:
            return response.read()


@contextmanager
def local_http_server(root: Path) -> Iterator[str]:
    handler = lambda *args, **kwargs: _QuietHandler(  # noqa: E731
        *args,
        directory=str(root),
        **kwargs,
    )
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


class KinecalDownloadTest(unittest.TestCase):
    def test_cli_exposes_download_and_verify_options(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/annotation/download_kinecal.py",
                "--help",
            ],
            cwd=repo,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--output-dir", completed.stdout)
        self.assertIn("--verify-only", completed.stdout)

    def test_selects_official_risk_groups_and_preserves_source_order(self) -> None:
        rows = read_register_csv(REGISTER_CSV)

        selected = select_risk_group_participants(rows)

        self.assertEqual(
            [row["part_id"] for row in selected],
            ["SPPB40", "SPPB62", "SPPB307"],
        )

    def test_reports_numeric_age_values_that_conflict_with_risk_group(self) -> None:
        rows = select_risk_group_participants(read_register_csv(REGISTER_CSV))

        mismatches = age_group_mismatches(rows)

        self.assertEqual(
            mismatches,
            [{"part_id": "SPPB62", "group": "FHm", "age": "64"}],
        )

    def test_extracts_numeric_participant_directory(self) -> None:
        self.assertEqual(participant_number("SPPB700"), "700")
        with self.assertRaisesRegex(ValueError, "participant ID"):
            participant_number("unknown")

    def test_parses_only_requested_directory_file_suffix(self) -> None:
        files = parse_directory_files(DIRECTORY_HTML, suffix=".txt")

        self.assertEqual(
            [(item.name, item.size) for item in files],
            [("100.txt", 1673), ("101.txt", 1681)],
        )

    def test_downloads_file_atomically_and_skips_matching_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            source = root / "source"
            source.mkdir()
            (source / "pose.txt").write_bytes(b"pose")
            output = root / "download" / "pose.txt"

            with local_http_server(source) as base_url:
                first = download_file(
                    f"{base_url}/pose.txt",
                    output,
                    expected_size=4,
                    retries=0,
                )
                second = download_file(
                    f"{base_url}/pose.txt",
                    output,
                    expected_size=4,
                    retries=0,
                )

            expected_hash = hashlib.sha256(b"pose").hexdigest()
            self.assertEqual(first.status, "downloaded")
            self.assertEqual(second.status, "skipped")
            self.assertEqual(first.sha256, expected_hash)
            self.assertEqual(second.sha256, expected_hash)
            self.assertEqual(sha256_file(output), expected_hash)
            self.assertFalse(output.with_suffix(".txt.part").exists())

    def test_builds_group_recording_and_age_mismatch_summary(self) -> None:
        participants = select_risk_group_participants(read_register_csv(REGISTER_CSV))
        recordings = [
            {
                "part_id": "SPPB40",
                "movement": "3m-walk-Front-View",
                "status": "available",
                "file_count": 2,
                "bytes": 8,
            },
            {
                "part_id": "SPPB62",
                "movement": "STS-5",
                "status": "missing",
                "file_count": 0,
                "bytes": 0,
            },
        ]
        manifest_rows = [
            {"status": "downloaded", "size": 4},
            {"status": "skipped", "size": 4},
        ]

        summary = build_download_summary(participants, recordings, manifest_rows)

        self.assertEqual(summary["participant_count"], 3)
        self.assertEqual(summary["group_counts"], {"NF": 1, "FHs": 1, "FHm": 1})
        self.assertEqual(summary["recording_counts"], {"available": 1, "missing": 1})
        self.assertEqual(summary["file_count"], 2)
        self.assertEqual(summary["total_bytes"], 8)
        self.assertEqual(summary["age_group_mismatch"][0]["part_id"], "SPPB62")
        json.dumps(summary)

    def test_downloads_only_selected_skeleton_recordings_and_verifies_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            (source / "register.csv").write_text(
                "part_id,group,age,sex,height,weight,BMI,recorded_in_the_lab,clinically-at-risk\n"
                "SPPB2,HA,54,m,1.82,73.0,22.0,1,0\n"
                "SPPB40,FHs,67,f,1.65,60.0,22.0,0,1\n",
                encoding="utf-8",
            )
            (source / "LICENSE.txt").write_text("CC0", encoding="utf-8")
            for movement in (
                "3m-walk-Front-View",
                "Get-Up-And-Go-Front-View",
                "STS-5",
            ):
                recording = source / "kinecal" / "40" / f"40_{movement}"
                skeleton_dir = recording / "skel"
                skeleton_dir.mkdir(parents=True)
                (skeleton_dir / "100.txt").write_text(
                    f"SpineBase,Tracked,0.0,0.0,1.0,10,20,{movement}\n",
                    encoding="utf-8",
                )
                skeleton_size = (skeleton_dir / "100.txt").stat().st_size
                (skeleton_dir / "index.html").write_text(
                    '<a href="100.txt">100.txt</a> '
                    f"15-Apr-2019 15:07 {skeleton_size}\n",
                    encoding="utf-8",
                )
                depth_dir = recording / "depth"
                depth_dir.mkdir()
                (depth_dir / "DepthUshort100.bin").write_bytes(b"depth")

            client = _RecordingHttpClient()

            @contextmanager
            def pooled_client(*args: object, **kwargs: object) -> Iterator[object]:
                yield client

            with (
                local_http_server(source) as base_url,
                patch(
                    "elderly_monitoring.modules.fall_risk.kinecal._pooled_http_client",
                    side_effect=pooled_client,
                ) as client_factory,
            ):
                summary = download_kinecal(
                    output,
                    base_url=f"{base_url}/",
                    workers=2,
                    retries=0,
                )

            manifest = [
                json.loads(line)
                for line in (output / "download_manifest.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(summary["participant_count"], 1)
            self.assertEqual(summary["file_count"], 3)
            self.assertEqual({row["movement"] for row in manifest}, {
                "3m-walk-Front-View",
                "Get-Up-And-Go-Front-View",
                "STS-5",
            })
            self.assertFalse(any(output.rglob("*.bin")))
            self.assertEqual(client_factory.call_count, 1)
            self.assertEqual(len(client.urls), 3)

            verification = verify_kinecal_download(output)
            self.assertEqual(verification["errors"], [])
            self.assertEqual(verification["verified_files"], 3)

            damaged = output / manifest[0]["path"]
            damaged.write_bytes(b"x" * damaged.stat().st_size)
            verification = verify_kinecal_download(output)
            self.assertTrue(
                any("SHA-256 mismatch" in error for error in verification["errors"])
            )


if __name__ == "__main__":
    unittest.main()
