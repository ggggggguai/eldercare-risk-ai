from __future__ import annotations

import argparse
from pathlib import Path

from elderly_monitoring.modules.fall_risk.self_collected_scf import (
    build_scf_mvp_v1_candidates,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an isolated, fail-closed SCF_MVP_V1 label candidate."
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=Path(
            "data/documents/fall_risk/self_collected/SCF_MVP_V1/"
            "audit/ingest_20260811/delivery_inventory.jsonl"
        ),
    )
    parser.add_argument(
        "--decision",
        type=Path,
        default=Path("configs/data/self_collected_scf_mvp_v1_decision.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build_scf_mvp_v1_candidates(
        inventory_path=args.inventory,
        decision_path=args.decision,
        output_dir=args.output_dir,
    )
    for name, path in sorted(result.items()):
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
