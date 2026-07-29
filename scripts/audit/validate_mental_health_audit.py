"""Independently validate the aggregate mental-health dataset audit outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    algorithm_root = Path(__file__).resolve().parents[2]
    workspace_root = algorithm_root.parents[1]
    mental_root = workspace_root / "数据集" / "心理"
    report_root = (
        algorithm_root
        / "reports"
        / "mental_health"
        / "dataset_audit_2026-07-25"
    )
    summary = json.loads(
        (report_root / "audit_summary.json").read_text(encoding="utf-8")
    )

    psyche = pd.read_parquet(
        mental_root / "PSYCHE-D" / "anon_processed_df_parquet"
    )
    label_columns = [
        "phq9_score_start",
        "phq9_score_end",
        "phq9_cat_start",
        "phq9_cat_end",
    ]
    labeled = psyche.loc[psyche[label_columns].notna().all(axis=1)]
    participant = pd.Series(
        [str(sample_id).rsplit("_", 1)[0] for sample_id in labeled.index],
        index=labeled.index,
    )
    repeat_distribution = (
        participant.value_counts().value_counts().sort_index().to_dict()
    )
    assert psyche.shape == (35_694, 154)
    assert len(labeled) == 10_866
    assert participant.nunique() == 4_036
    assert int((labeled["phq9_score_end"] >= 10).sum()) == 3_444
    assert int(
        (labeled["phq9_cat_end"] > labeled["phq9_cat_start"]).sum()
    ) == 2_252
    assert repeat_distribution == {1: 929, 2: 774, 3: 943, 4: 1_390}

    resilient = pd.read_csv(
        mental_root / "RESILIENT" / "Demographics.csv",
        dtype={"user_id": "string"},
    )
    sensor_stats = pd.read_csv(
        mental_root / "RESILIENT" / "summary_stats_per_participant.csv",
        dtype={"participant": "string"},
    )
    phq_dates = pd.Series(
        pd.to_datetime(
            resilient["phq_date"], format="%d/%m/%Y", errors="raise"
        ).to_numpy(),
        index=resilient["user_id"],
    )
    starts = (
        sensor_stats.assign(
            parsed_start=pd.to_datetime(
                sensor_stats["earliest_date"], errors="coerce"
            )
        )
        .pivot(index="participant", columns="file", values="parsed_start")
        .max(axis=1, skipna=False)
    )
    offsets = (starts - phq_dates.reindex(starts.index)).dt.days
    assert len(resilient) == 73
    assert int((resilient["phq_total"] >= 10).sum()) == 10
    assert offsets.value_counts().sort_index().to_dict() == {
        0.0: 28,
        1.0: 37,
        2.0: 3,
        5.0: 1,
        9.0: 1,
        10.0: 1,
        20.0: 1,
    }
    assert int(offsets.isna().sum()) == 1
    assert summary["resilient"]["strict_past_only_windows"] == {
        "7": 0,
        "14": 0,
        "28": 0,
    }
    assert (
        summary["resilient"][
            "ace_paired_with_all_four_modalities_ge28_days"
        ]
        == 45
    )

    nhanes_root = mental_root / "NHANES"
    dpq = pd.concat(
        [
            pd.read_sas(nhanes_root / "DPQ_G.xpt", format="xport"),
            pd.read_sas(nhanes_root / "DPQ_H.xpt", format="xport"),
        ],
        ignore_index=True,
    )
    day = pd.read_csv(nhanes_root / "NHANES Preliminary Day Level Output.csv")
    timestamp_columns = [
        column
        for column in dpq.columns
        if "date" in column.lower() or "time" in column.lower()
    ]
    assert not timestamp_columns
    assert (
        sorted(pd.to_datetime(day["calendar_date"]).dt.year.unique().tolist())
        == [2000]
    )
    assert len(day) == 89_104
    assert day["SEQN"].nunique() == 14_264

    manifest = pd.read_csv(report_root / "dataset_file_manifest.csv")
    for row in manifest.itertuples(index=False):
        source_path = workspace_root / Path(row.relative_path)
        assert source_path.stat().st_size == row.bytes
        assert sha256(source_path) == row.sha256

    assert summary["validation"]["passed"] is True
    print(
        json.dumps(
            {
                "psyche_d": "pass",
                "resilient": "pass",
                "nhanes_local": "pass",
                "file_hashes": f"pass ({len(manifest)} files)",
                "reported_validation": "pass",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
