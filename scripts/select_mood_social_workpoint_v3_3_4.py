"""Run OPT-WORKPOINT-001."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from elderly_monitoring.modules.mental_health.mood_social.workpoint_v3_3_4 import (  # noqa: E402
    select_workpoint,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = select_workpoint(
        ROOT / "configs/evaluation/mood_social_workpoint_v3_3_4.yaml",
        repository_root=ROOT,
        overwrite=args.overwrite,
        command=[sys.executable, __file__, *sys.argv[1:]],
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
