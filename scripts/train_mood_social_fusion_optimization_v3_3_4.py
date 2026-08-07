"""Run OPT-FUSION-003 with frozen strict OOF inputs."""

from __future__ import annotations

import sys
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from elderly_monitoring.modules.mental_health.mood_social.fusion.optimization_v3_3_4 import (  # noqa: E402
    load_v334_config,
    optimize_v334_fusion,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config_path = ROOT / "configs/training/mood_social_fusion_optimization_v3_3_4.yaml"
    config = load_v334_config(config_path, repository_root=ROOT)
    result = optimize_v334_fusion(
        config,
        overwrite=args.overwrite,
        command=[sys.executable, __file__, *sys.argv[1:]],
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
