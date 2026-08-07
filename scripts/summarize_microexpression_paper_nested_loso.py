from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from typing import Any

import torch


ALGORITHM_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ALGORITHM_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.paper_training import (  # noqa: E402
    sha256_file,
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return sha256_file(path)


def mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.pstdev(values)),
        "values": values,
    }


def summarize(run_root: Path) -> dict[str, Any]:
    all_runs: list[dict[str, Any]] = []
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for result_path in sorted((run_root / "selection").rglob("result.json")):
        result = load_json(result_path)
        row = {
            "fold_id": result["fold_id"],
            "stage": result["stage"],
            "candidate_id": result["candidate_id"],
            "candidate": result["candidate"],
            "status": result["status"],
            "result_path": result_path.resolve().as_posix(),
            "result_sha256": sha256_file(result_path),
        }
        if result["status"] == "completed":
            validation = result["validation"]
            row.update(
                {
                    "selection_score": validation["selection_score"],
                    "validation_loss": validation["loss"],
                    "validation_metrics": validation["metrics"],
                    "best_epoch": result["best_epoch"],
                }
            )
        else:
            row["error"] = result.get("error")
        all_runs.append(row)
        groups.setdefault((str(result["stage"]), str(result["candidate_id"])), []).append(row)

    grouped: list[dict[str, Any]] = []
    for (stage, candidate_id), rows in sorted(groups.items()):
        completed = [row for row in rows if row["status"] == "completed"]
        grouped_row: dict[str, Any] = {
            "stage": stage,
            "candidate_id": candidate_id,
            "candidate": rows[0]["candidate"],
            "run_count": len(rows),
            "completed_count": len(completed),
            "failed_count": len(rows) - len(completed),
            "fold_ids": [row["fold_id"] for row in rows],
        }
        if completed:
            for metric_name, source in (
                ("selection_score", "selection_score"),
                ("validation_loss", "validation_loss"),
            ):
                grouped_row[metric_name] = mean_std(
                    [float(row[source]) for row in completed]
                )
            for metric_name in ("uf1", "uar", "accuracy", "balanced_accuracy"):
                grouped_row[f"validation_{metric_name}"] = mean_std(
                    [
                        float(row["validation_metrics"][metric_name])
                        for row in completed
                    ]
                )
        grouped.append(grouped_row)

    final_rows = [
        load_json(path)
        for path in sorted((run_root / "final").rglob("result.json"))
    ]
    durations = [float(row["duration_seconds"]) for row in final_rows]
    memory = [int(row["peak_gpu_memory_bytes"]) for row in final_rows]
    parameter_count = None
    if final_rows:
        checkpoint_path = Path(str(final_rows[0]["checkpoint"]["path"]))
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        parameter_count = int(
            sum(
                value.numel()
                for name, value in checkpoint["model_state_dict"].items()
                if name != "graph_builder.au_adjacency"
            )
        )

    selected_by_fold = {}
    for path in sorted((run_root / "selection").rglob("selected_config.json")):
        payload = load_json(path)
        selected_by_fold[payload["fold_id"]] = payload["candidate"]

    return {
        "schema_version": "microexpression_nested_loso_ablation_summary_v1",
        "task_id": "EVAL-ME-002",
        "status": "completed",
        "selection_run_count": len(all_runs),
        "selection_completed_count": sum(row["status"] == "completed" for row in all_runs),
        "selection_failed_count": sum(row["status"] != "completed" for row in all_runs),
        "all_runs": all_runs,
        "by_stage_candidate": grouped,
        "selected_by_fold": selected_by_fold,
        "final_resource_summary": {
            "final_run_count": len(final_rows),
            "parameter_count": parameter_count,
            "duration_seconds": mean_std(durations),
            "peak_gpu_memory_bytes": {
                "mean": float(statistics.fmean(memory)),
                "max": max(memory),
                "values": memory,
            },
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize all EVAL-ME-002 ablation runs.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_root = args.run_root.resolve()
    output = (args.output or run_root / "aggregate" / "ablation_summary.json").resolve()
    payload = summarize(run_root)
    digest = write_json(output, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "selection_run_count": payload["selection_run_count"],
                "selection_failed_count": payload["selection_failed_count"],
                "final_run_count": payload["final_resource_summary"]["final_run_count"],
                "parameter_count": payload["final_resource_summary"]["parameter_count"],
                "output": output.as_posix(),
                "sha256": digest,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
