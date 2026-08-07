from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np


ALGORITHM_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ALGORITHM_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue.microexpression_spotter import (  # noqa: E402
    SPOTTER_TASK_ID,
    SpotterFeatureConfig,
    extract_window_features,
    feature_schema,
    list_frame_paths,
)


DEFAULT_SOURCE = ALGORITHM_ROOT / "data/manifests/microexpression/sequence_manifest_v1.jsonl"
DEFAULT_OUTPUT = ALGORITHM_ROOT / "data/processed/microexpression/spotter_v1"
DEFAULT_MANIFEST = ALGORITHM_ROOT / "data/manifests/microexpression/spotter_v1_manifest.jsonl"
DEFAULT_SUMMARY = ALGORITHM_ROOT / "data/manifests/microexpression/spotter_v1_summary.json"


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
    return sha256_file(path)


def write_npz(path: Path, features: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(temporary, features=features.astype(np.float32))
    temporary.replace(path)
    return sha256_file(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build unified SMIC micro/non_micro short-window features."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    return parser.parse_args()


def load_source(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected: list[dict[str, Any]] = []
    for row in rows:
        if row.get("source_dataset") != "smic_hs":
            continue
        role = str(row.get("sample_role"))
        if role == "classification":
            label = 1
            label_name = "micro"
        elif role == "spotting_negative":
            label = 0
            label_name = "non_micro"
        else:
            continue
        selected.append({**row, "binary_label": label, "binary_label_name": label_name})
    if len(selected) != 328:
        raise ValueError(f"Expected 328 SMIC windows, got {len(selected)}")
    if len({str(row["sample_id"]) for row in selected}) != len(selected):
        raise ValueError("Duplicate SMIC sample_id in source manifest")
    return selected


def main() -> int:
    args = parse_args()
    source_path = args.source.resolve()
    output_root = args.output.resolve()
    config = SpotterFeatureConfig()
    source_hash = sha256_file(source_path)
    rows = load_source(source_path)
    manifest_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for row in rows:
        sample_id = str(row["sample_id"])
        try:
            frame_paths = list_frame_paths(str(row["frame_dir"]))
            features = extract_window_features(frame_paths, config)
            artifact_path = output_root / "smic_hs" / str(row["subject_id"]) / f"{sample_id}.npz"
            artifact_hash = write_npz(artifact_path, features)
            manifest_rows.append(
                {
                    "schema_version": "smic_window_spotter_manifest_v1",
                    "task_id": SPOTTER_TASK_ID,
                    "sample_id": sample_id,
                    "source_dataset": "smic_hs",
                    "subject_id": str(row["subject_id"]),
                    "source_subject_id": str(row["source_subject_id"]),
                    "sequence_id": str(row["sequence_id"]),
                    "sample_role": str(row["sample_role"]),
                    "label": int(row["binary_label"]),
                    "label_name": str(row["binary_label_name"]),
                    "frame_dir": str(row["frame_dir"]),
                    "frame_count": int(row["frame_count"]),
                    "feature_path": str(artifact_path),
                    "feature_sha256": artifact_hash,
                    "feature_dim": int(features.shape[0]),
                    "feature_schema": feature_schema(config),
                    "source_manifest_sha256": source_hash,
                    "quality_status": "pass",
                    "evaluation_scope": "smic_short_window_baseline_only",
                    "long_video_spotting_claim": False,
                }
            )
        except (OSError, ValueError) as error:
            failures.append({"sample_id": sample_id, "error": str(error)})

    if failures or len(manifest_rows) != 328:
        failure_path = output_root / "failures.json"
        write_json(failure_path, {"failures": failures})
        raise RuntimeError(f"Feature preprocessing failed for {len(failures)} windows")

    manifest_rows.sort(key=lambda item: str(item["sample_id"]))
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = args.manifest.with_suffix(args.manifest.suffix + ".tmp")
    temporary_manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest_rows),
        encoding="utf-8",
    )
    temporary_manifest.replace(args.manifest)
    manifest_hash = sha256_file(args.manifest)
    summary = {
        "schema_version": "smic_window_spotter_preprocess_summary_v1",
        "task_id": SPOTTER_TASK_ID,
        "source_manifest": str(source_path),
        "source_manifest_sha256": source_hash,
        "output_root": str(output_root),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": manifest_hash,
        "feature_schema": feature_schema(config),
        "window_count": len(manifest_rows),
        "micro_count": sum(int(row["label"]) == 1 for row in manifest_rows),
        "non_micro_count": sum(int(row["label"]) == 0 for row in manifest_rows),
        "subject_count": len({str(row["subject_id"]) for row in manifest_rows}),
        "failure_count": 0,
        "uses_frame_count_as_feature": False,
        "long_video_spotting_claim": False,
    }
    summary_hash = write_json(args.summary.resolve(), summary)
    print(
        json.dumps(
            {**summary, "summary_sha256": summary_hash}, ensure_ascii=False, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
