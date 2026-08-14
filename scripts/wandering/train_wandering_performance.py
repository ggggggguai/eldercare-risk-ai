#!/usr/bin/env python3
"""Train, smoke-test, or fresh-reload the joint TopoWander-MPT model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from elderly_monitoring.modules.mental_health.wandering.performance import (
    CheckpointError,
    PerformanceConfigError,
    PerformanceDataError,
    aggregate_m0s_stability,
    evaluate_fresh_checkpoint,
    materialize_development_bundle,
    run_overfit_smoke,
    train_performance_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        required=True,
        choices=("materialize", "overfit-smoke", "train", "evaluate", "aggregate"),
        help="Execution phase; materialize creates train/validation-only data and evaluate is a new process.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/modules/wandering_performance_v1.yaml"),
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--runs-root",
        type=Path,
        help="Required for aggregate; contains seed-20260731, seed-20260801, seed-20260802.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Required for train/evaluate; must be one of 20260731, 20260801, 20260802.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume train mode from latest checkpoint.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.resume and args.mode != "train":
        print("--resume is valid only with --mode train", file=sys.stderr)
        return 2
    if args.mode in {"train", "evaluate"} and args.seed is None:
        print("--seed is required with --mode train/evaluate", file=sys.stderr)
        return 2
    if args.mode not in {"train", "evaluate"} and args.seed is not None:
        print("--seed is valid only with --mode train/evaluate", file=sys.stderr)
        return 2
    if (args.mode == "aggregate") != (args.runs_root is not None):
        print("--runs-root is required only with --mode aggregate", file=sys.stderr)
        return 2
    try:
        if args.mode == "materialize":
            output = materialize_development_bundle(
                args.config,
                project_root=args.project_root,
                output_dir=args.run_dir,
            )
            print(json.dumps({"mode": args.mode, "output_dir": str(output)}, ensure_ascii=False), flush=True)
            return 0
        if args.mode == "overfit-smoke":
            result = run_overfit_smoke(
                args.config,
                project_root=args.project_root,
                run_dir=args.run_dir,
            )
            payload = {
                "mode": args.mode,
                "passed": result.passed,
                "run_dir": str(result.run_dir),
                "checkpoint": str(result.checkpoint),
            }
            print(json.dumps(payload, ensure_ascii=False), flush=True)
            return 0 if result.passed else 1
        if args.mode == "train":
            result = train_performance_model(
                args.config,
                project_root=args.project_root,
                run_dir=args.run_dir,
                seed=args.seed,
                resume=args.resume,
            )
            print(
                json.dumps(
                    {
                        "mode": args.mode,
                        "run_dir": str(result.run_dir),
                        "best_checkpoint": str(result.best_checkpoint),
                        "last_checkpoint": str(result.last_checkpoint),
                        "stopped_reason": result.stopped_reason,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            return 0
        if args.mode == "aggregate":
            result = aggregate_m0s_stability(
                args.config,
                project_root=args.project_root,
                runs_root=args.runs_root,
                output_dir=args.run_dir,
            )
            print(
                json.dumps(
                    {
                        "mode": args.mode,
                        "output_dir": str(result.output_dir),
                        "gate_passed": result.report["gates"]["passed"],
                        "primary_seed_frozen": result.report["primary_seed_frozen"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            return 0 if result.report["gates"]["passed"] else 1
        result = evaluate_fresh_checkpoint(
            args.config,
            project_root=args.project_root,
            run_dir=args.run_dir,
            seed=args.seed,
        )
        print(
            json.dumps(
                {
                    "mode": args.mode,
                    "output_dir": str(result.output_dir),
                    "target_attained": result.metrics["target_attained"],
                    "wp_four_class_macro_f1": result.metrics["joint_metrics"]["wp_four_class"]["macro_f1"],
                    "source_equal_binary_macro_f1": result.metrics["joint_metrics"]["binary"]["source_equal_macro_f1"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 0
    except (PerformanceConfigError, PerformanceDataError, CheckpointError, FileExistsError, OSError) as exc:
        print(f"Wandering performance {args.mode} failed: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
