import tempfile
import unittest
from pathlib import Path

from scripts.evaluate.build_fall_runtime_fingerprint import build_fingerprint


class FallRuntimeFingerprintTest(unittest.TestCase):
    def test_repository_acceptance_config_has_hashable_fixed_inputs(self) -> None:
        result = build_fingerprint(
            Path("configs/modules/fall_risk_runtime_acceptance.yaml")
        )

        self.assertEqual(result["acceptance_status"], "development_candidate")
        self.assertEqual(len(result["config_sha256"]), 64)
        self.assertTrue(result["fixed_inputs"])
        self.assertTrue(all(len(item["sha256"]) == 64 for item in result["fixed_inputs"]))
        self.assertEqual(result["external_acceptance"]["status"], "blocked_external")

    def test_missing_fixed_input_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "acceptance.yaml"
            original = Path("configs/modules/fall_risk_runtime_acceptance.yaml").read_text(
                encoding="utf-8"
            )
            config.write_text(
                original.replace(
                    "data/external/le2i_imvia/raw/FallDataset/Home_01/Videos/video (1).avi",
                    "missing-video.avi",
                ),
                encoding="utf-8",
            )
            with self.assertRaises(FileNotFoundError):
                build_fingerprint(config)

    def test_branch_state_contract_must_include_inference_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "acceptance.yaml"
            original = Path("configs/modules/fall_risk_runtime_acceptance.yaml").read_text(
                encoding="utf-8"
            )
            config.write_text(
                original.replace(
                    "branch_states: [valid, unavailable, inference_error]",
                    "branch_states: [valid, unavailable]",
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "inference_error"):
                build_fingerprint(config)


if __name__ == "__main__":
    unittest.main()
