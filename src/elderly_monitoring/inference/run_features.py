from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from elderly_monitoring.modules.fall_risk import FallRiskPipeline
from elderly_monitoring.modules.mental_health import MentalHealthRiskPipeline


def load_sample(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError("Input feature file must contain a JSON object.")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Run risk inference from a feature JSON file.")
    parser.add_argument("--module", choices=["fall_risk", "mental_health"], required=True)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()

    sample = load_sample(args.input)
    if args.module == "fall_risk":
        event = FallRiskPipeline().predict_from_features(sample)
    else:
        event = MentalHealthRiskPipeline().predict_from_features(sample)

    print(json.dumps(event.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
