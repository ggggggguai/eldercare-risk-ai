from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys
from typing import Any

import yaml


ALGORITHM_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ALGORITHM_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.evaluation_variants import (  # noqa: E402
    build_evaluation_variants_for_sample,
    write_variant_manifests,
)
from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.manifold_graph import (  # noqa: E402
    sha256_path,
)
from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.paper_preprocess import (  # noqa: E402
    PaperPreprocessConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the frozen-apex EVAL-ME-002 preprocessing/flow/channel variants."
    )
    parser.add_argument(
        "--sequence-manifest",
        type=Path,
        default=ALGORITHM_ROOT
        / "data/manifests/microexpression/sequence_manifest_v1.jsonl",
    )
    parser.add_argument(
        "--frozen-manifest",
        type=Path,
        default=ALGORITHM_ROOT
        / "data/manifests/microexpression/paper_v2"
        / "flow_artifact_manifest_smic_hs_classification_combined_v2.jsonl",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ALGORITHM_ROOT
        / "configs/preprocessing/microexpression_smic_paper_v2.yaml",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ALGORITHM_ROOT
        / "data/processed/microexpression/paper_v2/evaluation_variants",
    )
    parser.add_argument(
        "--report-root",
        type=Path,
        default=ALGORITHM_ROOT
        / "data/manifests/microexpression/paper_v2/evaluation_variants",
    )
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    args = parse_args()
    config_payload = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    valid_fields = {item.name for item in fields(PaperPreprocessConfig)}
    config = PaperPreprocessConfig(
        **{
            key: value
            for key, value in config_payload["preprocess"].items()
            if key in valid_fields
        }
    )
    sequences = {
        str(record["sample_id"]): record
        for record in load_jsonl(args.sequence_manifest)
        if record.get("source_dataset") == "smic_hs"
        and record.get("sample_role") == "classification"
    }
    frozen_records = load_jsonl(args.frozen_manifest)
    if args.limit:
        frozen_records = frozen_records[: args.limit]
    output_records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for index, frozen_record in enumerate(frozen_records, start=1):
        sample_id = str(frozen_record["sample_id"])
        try:
            output_records.extend(
                build_evaluation_variants_for_sample(
                    sequence_record=sequences[sample_id],
                    frozen_record=frozen_record,
                    base_config=config,
                    output_root=args.output_root,
                )
            )
        except Exception as error:  # noqa: BLE001
            failures.append({"sample_id": sample_id, "error": repr(error)})
        if index % 10 == 0 or index == len(frozen_records):
            print(f"processed={index}/{len(frozen_records)} failures={len(failures)}")
    source_hashes = {
        "sequence_manifest": sha256_path(args.sequence_manifest),
        "frozen_manifest": sha256_path(args.frozen_manifest),
        "config": sha256_path(args.config),
        "builder_source": sha256_path(Path(__file__).resolve()),
    }
    summary = write_variant_manifests(
        output_records,
        report_root=args.report_root,
        source_hashes=source_hashes,
        config=config,
    )
    summary["failures"] = failures
    summary["requested_samples"] = len(frozen_records)
    summary["completed_artifacts"] = len(output_records)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not failures and summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
