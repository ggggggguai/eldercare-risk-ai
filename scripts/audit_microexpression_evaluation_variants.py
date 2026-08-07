from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ALGORITHM_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ALGORITHM_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.evaluation_variants import (  # noqa: E402
    audit_evaluation_variants,
)
from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.manifold_graph import (  # noqa: E402
    sha256_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independently audit all EVAL-ME-002 input variants."
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=ALGORITHM_ROOT
        / "data/manifests/microexpression/paper_v2/evaluation_variants"
        / "evaluation_input_variants_summary_v2.json",
    )
    parser.add_argument(
        "--frozen-manifest",
        type=Path,
        default=ALGORITHM_ROOT
        / "data/manifests/microexpression/paper_v2"
        / "flow_artifact_manifest_smic_hs_classification_combined_v2.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ALGORITHM_ROOT
        / "data/manifests/microexpression/paper_v2/evaluation_variants"
        / "evaluation_input_variants_audit_v2.json",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    args = parse_args()
    audit = audit_evaluation_variants(
        summary=json.loads(args.summary.read_text(encoding="utf-8")),
        frozen_records=load_jsonl(args.frozen_manifest),
    )
    audit["inputs"] = {
        "summary": {
            "path": args.summary.resolve().as_posix(),
            "sha256": sha256_path(args.summary),
        },
        "frozen_manifest": {
            "path": args.frozen_manifest.resolve().as_posix(),
            "sha256": sha256_path(args.frozen_manifest),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": audit["status"],
                "variant_count": audit["variant_count"],
                "artifact_count": audit["artifact_count"],
                "p3_reference_exact_count": audit["p3_reference_exact_count"],
                "issue_count": audit["issue_count"],
                "output": args.output.resolve().as_posix(),
                "sha256": sha256_path(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if audit["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
