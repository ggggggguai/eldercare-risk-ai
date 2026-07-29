from __future__ import annotations

import pandas as pd

from scripts.audit.audit_mental_health_datasets import (
    _canonical_collection_hash,
    _parse_psyche_index,
    build_psyche_feature_mapping,
)


def test_parse_psyche_index_preserves_participant_and_month() -> None:
    parsed = _parse_psyche_index(pd.Index(["abc_3", "participant_with_gap_12"]))

    assert parsed["participant_id"].tolist() == ["abc", "participant_with_gap"]
    assert parsed["month"].tolist() == [3, 12]


def test_feature_mapping_freezes_documented_sensor_statistics_only() -> None:
    dictionary = pd.DataFrame(
        [
            {
                "Feature name": "sleep_efficiency",
                "Category": "Sleep",
                "Subcategory": "Statistics",
                "Description": "documented",
            },
            {
                "Feature name": "sleep_regression_score",
                "Category": "Sleep",
                "Subcategory": "Linear regression",
                "Description": None,
            },
            {
                "Feature name": "phq9_score_end",
                "Category": "PHQ-9",
                "Subcategory": "PHQ-9 score",
                "Description": "target",
            },
        ]
    )

    mapping = build_psyche_feature_mapping(dictionary).set_index("feature")

    assert bool(mapping.loc["sleep_efficiency", "include_state_sensor_v1"])
    assert not bool(
        mapping.loc["sleep_regression_score", "include_state_sensor_v1"]
    )
    assert mapping.loc["sleep_regression_score", "role"] == (
        "quarantine_undefined_derived"
    )
    assert mapping.loc["phq9_score_end", "role"] == "target_only"


def test_collection_hash_is_order_independent() -> None:
    rows = [
        {"relative_path": "b.csv", "bytes": 2, "sha256": "bbb"},
        {"relative_path": "a.csv", "bytes": 1, "sha256": "aaa"},
    ]

    assert _canonical_collection_hash(rows) == _canonical_collection_hash(
        list(reversed(rows))
    )
