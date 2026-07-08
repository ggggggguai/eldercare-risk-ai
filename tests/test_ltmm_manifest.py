import json
import tempfile
import unittest
from pathlib import Path

from elderly_monitoring.modules.fall_risk.ltmm import build_ltmm_manifest, write_ltmm_manifest


class LtmmManifestTest(unittest.TestCase):
    def test_build_manifest_uses_records_and_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = Path(tmpdir)
            (raw / "LabWalks").mkdir()
            (raw / "RECORDS").write_text(
                "\n".join(["CO001", "FL001", "LabWalks/co001_base"]),
                encoding="utf-8",
            )
            (raw / "SHA256SUMS.txt").write_text(
                "\n".join(
                    [
                        "a" * 64 + " ClinicalDemogData_COFL.xlsx",
                        "b" * 64 + " CO001.dat",
                        "c" * 64 + " CO001.hea",
                    ]
                ),
                encoding="utf-8",
            )
            (raw / "ClinicalDemogData_COFL.xlsx").write_bytes(b"placeholder")
            (raw / "CO001.dat").write_bytes(b"data")
            (raw / "CO001.hea").write_text("header", encoding="utf-8")
            (raw / "FL001.hea").write_text("header", encoding="utf-8")
            (raw / "LabWalks" / "co001_base.dat").write_bytes(b"data")
            (raw / "LabWalks" / "co001_base.hea").write_text("header", encoding="utf-8")

            rows = build_ltmm_manifest(raw)

        by_id = {row["record_id"]: row for row in rows}
        self.assertEqual(len(rows), 5)
        self.assertTrue(by_id["ClinicalDemogData_COFL"]["available"])
        self.assertEqual(by_id["ClinicalDemogData_COFL"]["record_type"], "clinical_demographic_table")
        self.assertEqual(by_id["ClinicalDemogData_COFL"]["sha256"], "a" * 64)

        self.assertEqual(by_id["CO001"]["group"], "control")
        self.assertEqual(by_id["CO001"]["subject_id"], "ltmm_co_001")
        self.assertTrue(by_id["CO001"]["available"])
        self.assertEqual(by_id["CO001"]["sha256"], "b" * 64)
        self.assertEqual(by_id["CO001"]["header_sha256"], "c" * 64)

        self.assertEqual(by_id["FL001"]["group"], "faller")
        self.assertFalse(by_id["FL001"]["available"])
        self.assertTrue(by_id["FL001"]["header_available"])
        self.assertFalse(by_id["FL001"]["data_available"])

        self.assertEqual(by_id["LabWalks/co001_base"]["record_type"], "lab_walk_accelerometer")
        self.assertEqual(by_id["LabWalks/co001_base"]["group"], "control")

    def test_write_manifest_outputs_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = Path(tmpdir) / "raw"
            raw.mkdir()
            (raw / "RECORDS").write_text("CO001\n", encoding="utf-8")
            (raw / "CO001.dat").write_bytes(b"data")
            (raw / "CO001.hea").write_text("header", encoding="utf-8")
            output = Path(tmpdir) / "ltmm_manifest.jsonl"

            count = write_ltmm_manifest(raw, output)
            lines = output.read_text(encoding="utf-8").splitlines()

        self.assertEqual(count, 3)
        self.assertEqual(len(lines), 3)
        payloads = [json.loads(line) for line in lines]
        self.assertIn("ltmm", {payload["dataset"] for payload in payloads})
        self.assertIn("CO001", {payload["record_id"] for payload in payloads})


if __name__ == "__main__":
    unittest.main()
