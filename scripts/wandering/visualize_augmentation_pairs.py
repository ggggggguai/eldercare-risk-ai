"""Create the fixed step-8 human-review contact set without changing its status."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from elderly_monitoring.modules.mental_health.wandering.augmentation import (
    AugmentationBuildError,
    commit_new_output_directory,
)
from elderly_monitoring.modules.mental_health.wandering.camera_adapter import canonical_json_bytes
from elderly_monitoring.modules.mental_health.wandering.camera_corruption import (
    load_augmentation_config,
    verify_augmentation_trust_roots,
)
from elderly_monitoring.modules.mental_health.wandering.compatibility_report import (
    load_augmentation_bundle,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--augmentation-bundle", required=True)
    parser.add_argument("--expected-augmentation-manifest-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise AugmentationBuildError(f"output directory already exists: {output}")
    config = load_augmentation_config(args.config)
    loaded = load_augmentation_bundle(
        bundle_dir=args.augmentation_bundle,
        expected_manifest_sha256=args.expected_augmentation_manifest_sha256,
        config=config,
        project_root=args.project_root,
    )
    verify_augmentation_trust_roots(config, args.project_root)
    selected = _select_pairs(loaded.train_pair_records, config)
    files: dict[str, bytes] = {}
    selection_rows: list[dict[str, Any]] = []
    for group_name, rows in selected:
        for rank, row in enumerate(rows, start=1):
            relative = f"figures/{group_name}/{row['severity']}_{rank:03d}_{row['child_id'][:12]}.png"
            files[relative] = _pair_png(row)
            selection_rows.append(
                {
                    "source_dataset": row["source_dataset"],
                    "label": row["labels"]["four_class"]
                    if row["source_dataset"] == "wandering_patterns"
                    else row["labels"]["binary"],
                    "severity": row["severity"],
                    "child_id": row["child_id"],
                    "parent_sample_id": row["parent_sample_id"],
                    "figure": relative,
                }
            )
    files["figures/severity_curves/pressure_qc_semantic_rates.png"] = _severity_png(
        loaded.pressure_records
    )
    for fault_type, row in _fault_examples(loaded.fault_records).items():
        files[f"figures/qc_fault_examples/{fault_type}.png"] = _fault_png(row)
    selection_payload = canonical_json_bytes(
        {
            "schema_version": "wandering-augmentation-visual-selection-v3",
            "selection": "sha256_child_id_ascending",
            "augmentation_manifest_sha256": loaded.manifest_sha256,
            "records": selection_rows,
        }
    )
    files["figures/selection_index.json"] = selection_payload
    files["README.md"] = _readme_markdown(loaded.manifest_sha256).encode("utf-8")
    descriptors = {
        name: {"sha256": hashlib.sha256(payload).hexdigest(), "byte_count": len(payload)}
        for name, payload in files.items()
    }
    visual_manifest_payload = canonical_json_bytes(
        {
            "schema_version": "wandering-augmentation-visual-manifest-v3",
            "augmentation_manifest_sha256": loaded.manifest_sha256,
            "artifacts": descriptors,
            "human_review_contract": {
                "path": "HUMAN_REVIEW.md",
                "initial_status": "pending_human_review",
                "required_final_status": "human_review_passed",
                "final_sha256_source": "external_cli",
            },
        }
    )
    visual_manifest_sha256 = hashlib.sha256(visual_manifest_payload).hexdigest()
    files["figures/visual_manifest.json"] = visual_manifest_payload
    files["HUMAN_REVIEW.md"] = _human_review_markdown(
        selection_rows,
        augmentation_manifest_sha256=loaded.manifest_sha256,
        visual_manifest_sha256=visual_manifest_sha256,
        selection_index_sha256=hashlib.sha256(selection_payload).hexdigest(),
    ).encode("utf-8")
    commit_new_output_directory(output, files)
    print(
        json.dumps(
            {
                "output_dir": str(output),
                "human_review_status": "pending_human_review",
                "selected_pair_count": len(selection_rows),
                "visual_manifest_sha256": visual_manifest_sha256,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def _select_pairs(
    pairs: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> list[tuple[str, list[Mapping[str, Any]]]]:
    groups: list[tuple[str, list[Mapping[str, Any]]]] = []
    for label in ("direct", "pacing", "lapping", "random"):
        selected: list[Mapping[str, Any]] = []
        for severity in ("low", "medium"):
            candidates = [
                row
                for row in pairs
                if row["source_dataset"] == "wandering_patterns"
                and row["labels"]["four_class"] == label
                and row["severity"] == severity
            ]
            ordered = sorted(candidates, key=lambda row: _child_hash(str(row["child_id"])))
            count = int(config["visual_review"]["accepted_wp_pairs_per_class_per_severity"])
            if len(ordered) < count:
                raise AugmentationBuildError(f"insufficient WP visual-review pairs: {label}/{severity}")
            selected.extend(ordered[:count])
        groups.append((f"wp_{label}_100_pairs", selected))
    for label in (0, 1):
        selected = []
        for severity in ("low", "medium"):
            candidates = [
                row
                for row in pairs
                if row["source_dataset"] == "smartcare"
                and int(row["labels"]["binary"]) == label
                and row["severity"] == severity
            ]
            ordered = sorted(candidates, key=lambda row: _child_hash(str(row["child_id"])))
            count = int(
                config["visual_review"]["accepted_smartcare_pairs_per_binary_class_per_severity"]
            )
            if len(ordered) < count:
                raise AugmentationBuildError(
                    f"insufficient SmartCare visual-review pairs: {label}/{severity}"
                )
            selected.extend(ordered[:count])
        groups.append((f"smartcare_{label}_20_pairs", selected))
    return groups


def _pair_png(row: Mapping[str, Any]) -> bytes:
    clean = np.asarray(row["clean_shape_normalized_points"], dtype=np.float64)
    corrupted = np.asarray(row["shape_normalized_points"], dtype=np.float64)
    joint = np.vstack((clean, corrupted))
    low = joint.min(axis=0)
    high = joint.max(axis=0)
    padding = np.maximum((high - low) * 0.08, 0.03)
    figure, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=120)
    for axis, points, title, color in (
        (axes[0], clean, "clean", "#1f77b4"),
        (axes[1], corrupted, f"corrupted: {row['severity']}", "#d62728"),
    ):
        axis.plot(points[:, 0], points[:, 1], color=color, linewidth=1.4)
        axis.scatter(points[0, 0], points[0, 1], color="#2ca02c", s=18, label="start")
        axis.scatter(points[-1, 0], points[-1, 1], color="#111111", s=18, label="end")
        axis.set_xlim(low[0] - padding[0], high[0] + padding[0])
        axis.set_ylim(low[1] - padding[1], high[1] + padding[1])
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.2)
        axis.set_title(title)
    label = row["labels"]["four_class"] or f"binary={row['labels']['binary']}"
    figure.suptitle(f"{row['source_dataset']} | {label} | parent={row['parent_sample_id']}", fontsize=9)
    figure.tight_layout()
    return _figure_bytes(figure)


def _severity_png(rows: Sequence[Mapping[str, Any]]) -> bytes:
    severities = ("low", "medium", "high")
    total = Counter(str(row["severity"]) for row in rows)
    ready = Counter(str(row["severity"]) for row in rows if row["qc_status"] == "ready")
    semantic = Counter(
        str(row["severity"]) for row in rows if row["semantic_status"] == "accepted"
    )
    x = np.arange(3)
    figure, axis = plt.subplots(figsize=(7, 4), dpi=120)
    axis.bar(x - 0.18, [ready[s] / total[s] for s in severities], width=0.36, label="QC ready")
    axis.bar(
        x + 0.18,
        [semantic[s] / total[s] for s in severities],
        width=0.36,
        label="semantic accepted",
    )
    axis.set_xticks(x, severities)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("fraction of all frozen validation views")
    axis.set_title("Frozen pressure-view availability")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    return _figure_bytes(figure)


def _fault_examples(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    output: dict[str, Mapping[str, Any]] = {}
    for row in sorted(rows, key=lambda value: _child_hash(str(value["fault_or_control_id"]))):
        output.setdefault(str(row["fault_or_control_type"]), row)
    return output


def _fault_png(row: Mapping[str, Any]) -> bytes:
    points = np.asarray(row["virtual_image_points"], dtype=np.float64)
    observed = np.asarray(row["observed_mask"], dtype=bool)
    figure, axis = plt.subplots(figsize=(5, 5), dpi=120)
    axis.plot(points[:, 0], points[:, 1], color="#999999", linewidth=1.0)
    axis.scatter(points[observed, 0], points[observed, 1], color="#1f77b4", s=8, label="observed")
    if np.any(~observed):
        axis.scatter(points[~observed, 0], points[~observed, 1], color="#d62728", s=20, label="omitted")
    axis.axvline(points[40, 0], color="#ff7f0e", alpha=0.25)
    axis.set_aspect("equal", adjustable="box")
    axis.set_title(
        f"{row['fault_or_control_type']}\nQC={row['observed_qc_status']} reasons={','.join(row['observed_qc_reason_codes'])}",
        fontsize=8,
    )
    axis.grid(alpha=0.2)
    axis.legend(fontsize=7)
    figure.tight_layout()
    return _figure_bytes(figure)


def _figure_bytes(figure: Any) -> bytes:
    buffer = io.BytesIO()
    figure.savefig(
        buffer,
        format="png",
        dpi=120,
        metadata={"Software": "elderly-monitoring-algorithms-step8"},
    )
    plt.close(figure)
    return buffer.getvalue()


def _child_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _human_review_markdown(
    rows: Sequence[Mapping[str, Any]],
    *,
    augmentation_manifest_sha256: str,
    visual_manifest_sha256: str,
    selection_index_sha256: str,
) -> str:
    counts = Counter(
        (str(row["source_dataset"]), str(row["label"]), str(row["severity"])) for row in rows
    )
    if len(rows) != 440:
        raise AugmentationBuildError("v3 human-review template requires exactly 440 pairs")
    machine_header = {
        "schema_version": "wandering-augmentation-human-review-v3",
        "status": "pending_human_review",
        "augmentation_manifest_sha256": augmentation_manifest_sha256,
        "visual_manifest_sha256": visual_manifest_sha256,
        "selection_index_sha256": selection_index_sha256,
        "reviewed_pair_count": 440,
        "reviewer": "pending",
        "review_date": "pending",
        "rejected_child_ids": [],
        "final_decision": "pending_human_review",
    }
    lines = [
        "<!-- wandering-augmentation-human-review-v3",
        json.dumps(machine_header, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        "-->",
        "# Wandering Step 8 Human Review",
        "",
        "This status may be changed to `human_review_passed` only after the project owner actually inspects every selected clean/corrupted pair. Automated tests and AI review do not count as sign-off.",
        "",
        "## Required strata",
        "",
    ]
    for key, count in sorted(counts.items()):
        lines.append(f"- [ ] `{key[0]} / {key[1]} / {key[2]}`: {count} pairs")
    lines.extend(
        [
            "",
            "## Reviewer record",
            "",
            "- Reviewer: pending",
            "- Review date: pending",
            "- Rejected child IDs and reasons: pending",
            "- Final decision: pending_human_review",
            "",
        ]
    )
    return "\n".join(lines)


def _readme_markdown(manifest_sha256: str) -> str:
    return "\n".join(
        [
            "# Wandering Step 8 Diagnostics",
            "",
            f"Augmentation manifest SHA-256: `{manifest_sha256}`",
            "",
            "Evidence scope: `synthetic_camera_corruption`.",
            "",
            "These figures are deterministic diagnostics. They do not validate a target camera, identify people, or establish elderly/clinical performance.",
            "",
            "The human-review gate remains `pending_human_review` until the project owner records an actual inspection in `HUMAN_REVIEW.md`.",
            "",
        ]
    )


if __name__ == "__main__":
    main()
