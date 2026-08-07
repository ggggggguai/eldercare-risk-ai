from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ALGORITHM_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ALGORITHM_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.manifold_graph import (  # noqa: E402
    build_au_cooccurrence_prior,
    sha256_path,
)


def parse_args() -> argparse.Namespace:
    workspace_root = ALGORITHM_ROOT.parents[1]
    source_root = workspace_root / "微表情识别" / "MHSSA-Tranformer-GCN-main"
    parser = argparse.ArgumentParser(
        description="Audit auxiliary AU CSV files and reconstruct the paper-v2 AU prior."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        action="append",
        default=None,
        help="Repeat for each AU-bearing CSV. Defaults to CASME II and SAMM files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ALGORITHM_ROOT
        / "data/manifests/microexpression/paper_v2/au_prior_casme_samm_reconstructed_v2.json",
    )
    parser.set_defaults(
        default_csvs=[
            source_root / "casme_class2_for_optical_flow.csv",
            source_root / "samm_class2_for_optical_flow.csv",
        ]
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    csv_paths = args.csv or args.default_csvs
    _, audit = build_au_cooccurrence_prior(csv_paths)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": output.as_posix(),
                "sha256": sha256_path(output),
                "parsed_rows": audit["parsed_rows"],
                "class_support": audit["class_support"],
                "rows_without_mapped_au": audit["rows_without_mapped_au"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
