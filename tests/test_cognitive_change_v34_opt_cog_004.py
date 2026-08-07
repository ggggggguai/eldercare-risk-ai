from __future__ import annotations

from training.cognitive_change_clue.opt_cog_004 import _a4_rows
from training.cognitive_change_clue.train_v34 import _validate_prediction_rows


def _prediction_row(candidate_id: str, *, raw_logit: float, face_logit: float | None) -> dict:
    return {
        "schema_version": "cognitive_v34_raw_prediction_v1",
        "candidate_id": candidate_id,
        "outer_fold": 2,
        "split_role": "inner_validation",
        "sample_id": "HC_subj_001__pic_1",
        "subject_id": "HC_subj_001",
        "diagnosis": "HC",
        "label_hc_vs_non_hc": 0,
        "raw_logit": raw_logit,
        "audio_logit": 0.2,
        "text_logit": 0.3,
        "face_logit": face_logit,
        "q_audio": 0.8,
        "q_text": 0.7,
        "q_face": 0.9,
        "audio_missing_mask": 0,
        "text_missing_mask": 0,
        "face_missing_mask": 0,
    }


def test_a4_rows_rebind_candidate_identity_for_prediction_bundle() -> None:
    audio_text = _prediction_row("V34-A2", raw_logit=0.4, face_logit=None)
    face = _prediction_row("V34-Face", raw_logit=0.6, face_logit=0.6)

    rows = _a4_rows([audio_text], [face], alpha=0.25)

    assert rows[0]["candidate_id"] == "V34-A4"
    assert audio_text["candidate_id"] == "V34-A2"
    assert face["candidate_id"] == "V34-Face"
    _validate_prediction_rows(
        rows,
        candidate_id="V34-A4",
        outer_fold=2,
        split_role="inner_validation",
    )
