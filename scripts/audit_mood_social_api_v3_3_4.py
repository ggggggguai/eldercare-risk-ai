"""Run API-003 no-promotion version audit."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from elderly_monitoring.modules.mental_health.mood_social.api_versioning_v3_3_4 import (  # noqa: E402
    audit_api003,
)


def main() -> int:
    result = audit_api003(
        ROOT / "configs/runtime/mood_social_api_v3_3_4.yaml",
        repository_root=ROOT,
        command=[sys.executable, __file__, *sys.argv[1:]],
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
