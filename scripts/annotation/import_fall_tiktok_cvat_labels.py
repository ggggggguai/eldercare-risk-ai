from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

from elderly_monitoring.modules.fall_risk.annotations import (
    convert_cvat_xml,
    write_converted_fall_labels,
)
from elderly_monitoring.modules.fall_risk.fall_tiktok import (
    load_fall_tiktok_collection_decision,
    prepare_fall_tiktok_cvat_export,
    write_prepared_fall_tiktok_export,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the fall_tiktok CVAT export and import source-specific v2 labels."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--source-map",
        type=Path,
        default=Path("configs/data/fall_tiktok_source_map_v1.json"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/fall_risk_video_manifest.jsonl"),
    )
    parser.add_argument(
        "--collection-decision",
        type=Path,
        default=Path("configs/data/fall_tiktok_collection_decision_v1.json"),
    )
    parser.add_argument(
        "--redacted-output",
        type=Path,
        default=Path(
            "data/annotations/fall_risk/cvat_exports/raw/fall_tiktok/"
            "fall_tiktok_cvat_redacted.zip"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/annotations/fall_risk/generated/v2/fall_tiktok_manual"),
    )
    parser.add_argument("--labeler", default="fall_tiktok_labeler_01")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    action_output = args.output_dir / "action_labels.jsonl"
    event_output = args.output_dir / "event_labels.jsonl"
    report_output = args.output_dir / "import_report.json"
    try:
        decision = load_fall_tiktok_collection_decision(args.collection_decision)
        prepared = prepare_fall_tiktok_cvat_export(args.input, args.source_map)
        write_prepared_fall_tiktok_export(
            prepared, args.redacted_output, overwrite=args.overwrite
        )
        converted = convert_cvat_xml(
            args.redacted_output,
            manifest_path=args.manifest,
            fps=None,
            labeler=args.labeler,
        )
        counts = write_converted_fall_labels(
            converted,
            action_output_path=action_output,
            event_output_path=event_output,
            overwrite=args.overwrite,
        )
        report = {
            "schema_version": "fall-tiktok-cvat-import-v1",
            "source_archive": {
                "filename": args.input.name,
                "sha256": prepared.source_sha256,
            },
            "source_map": {
                "path": args.source_map.as_posix(),
                "sha256": _sha256_file(args.source_map),
            },
            "collection_decision": {
                "path": args.collection_decision.as_posix(),
                "sha256": _sha256_file(args.collection_decision),
                "decision_id": decision.decision_id,
                "decided_at": decision.decided_at,
                "decided_by": decision.decided_by,
                "collection_status": decision.collection_status,
                "training_use": decision.training_use,
                "redistribution_use": decision.redistribution_use,
                "consent_status": decision.consent_status,
                "subject_grouping_status": decision.subject_grouping_status,
            },
            "redacted_export": {
                "path": args.redacted_output.as_posix(),
                "sha256": prepared.prepared_sha256,
                "removed_identity_elements": prepared.removed_identity_elements,
            },
            "task_count": prepared.task_count,
            "track_count": prepared.track_count,
            "task_mappings": prepared.task_mappings,
            "source_group_id": decision.source_group_id,
            "provenance_status": decision.provenance_status,
            "training_policy": "eligible_manual_cvat_training",
            "outputs": {
                "action_labels": {
                    "path": action_output.as_posix(),
                    "count": counts["action_labels"],
                    "sha256": _sha256_file(action_output),
                },
                "event_labels": {
                    "path": event_output.as_posix(),
                    "count": counts["event_labels"],
                    "sha256": _sha256_file(event_output),
                },
            },
        }
        _write_json(report_output, report, overwrite=args.overwrite)
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    print(json.dumps(report["outputs"], ensure_ascii=False, sort_keys=True))
    print(f"report_output={report_output}")


def _write_json(path: Path, payload: dict, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
