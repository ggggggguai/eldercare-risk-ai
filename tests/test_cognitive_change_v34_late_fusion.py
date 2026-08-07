import numpy as np

from training.cognitive_change_clue import select_v34_backbone, v34_late_fusion


def _subject_records():
    rows = []
    for diagnosis in ("HC", "MCI", "AD"):
        for index in range(5):
            subject_id = f"{diagnosis}-{index}"
            rows.append({"subject_id": subject_id, "diagnosis_label": diagnosis})
    return rows


def _prediction(sample_id, subject_id, label, raw_logit, *, modality):
    return {
        "schema_version": "cognitive_v34_raw_prediction_v1",
        "candidate_id": f"V34-A3-{modality.title()}",
        "outer_fold": 0,
        "split_role": "inner_validation",
        "sample_id": sample_id,
        "subject_id": subject_id,
        "diagnosis": "HC" if label == 0 else "MCI",
        "label_hc_vs_non_hc": label,
        "raw_logit": raw_logit,
        "audio_logit": raw_logit if modality == "audio" else None,
        "text_logit": raw_logit if modality == "text" else None,
        "face_logit": None,
        "q_audio": 0.8,
        "q_text": 0.7,
        "q_face": 0.6,
        "audio_missing_mask": 0,
        "text_missing_mask": 0,
        "face_missing_mask": 0,
    }


def test_inner_oof_split_is_subject_safe_deterministic_and_complete():
    records = _subject_records()
    subjects = [row["subject_id"] for row in records]
    first = v34_late_fusion.build_inner_subject_folds(records, subjects, outer_fold=2)
    second = v34_late_fusion.build_inner_subject_folds(records, subjects, outer_fold=2)
    assert first == second
    evaluation = []
    for fold in first:
        assert set(fold["train_subjects"]).isdisjoint(fold["evaluation_subjects"])
        evaluation.extend(fold["evaluation_subjects"])
    assert sorted(evaluation) == sorted(subjects)


def test_late_fusion_uses_two_logits_and_quality_and_preserves_degradation():
    audio_rows = [
        _prediction(f"s-{index}", f"p-{index}", index % 2, float(index - 2), modality="audio")
        for index in range(6)
    ]
    text_rows = [
        _prediction(f"s-{index}", f"p-{index}", index % 2, float(2 - index), modality="text")
        for index in range(6)
    ]
    classifier, fit_rows = v34_late_fusion._fit_late_fusion(audio_rows, text_rows)
    assert len(fit_rows) == 6
    assert classifier.coef_.shape == (1, 4)
    text_rows[-1]["raw_logit"] = None
    text_rows[-1]["text_missing_mask"] = 1
    combined = v34_late_fusion.combine_component_rows(
        audio_rows,
        text_rows,
        classifier,
        outer_fold=0,
        split_role="outer_evaluation",
        schema_version="cognitive_v34_raw_prediction_v1",
    )
    assert np.isfinite(combined[0]["raw_logit"])
    assert combined[-1]["raw_logit"] == audio_rows[-1]["raw_logit"]
    assert combined[-1]["text_logit"] is None


def test_backbone_ranking_uses_frozen_tie_break_order():
    def summary(auc, std, specificity, task_auc):
        return {
            "subject_raw_auc": {"mean": auc, "population_std": std},
            "pooled_temporary_workpoint": {"specificity": specificity},
            "task_raw_auc": {"mean": task_auc},
        }

    selected, rows = select_v34_backbone._ranking(
        {
            "V34-A1": summary(0.82, 0.05, 0.75, 0.80),
            "V34-A2": summary(0.817, 0.04, 0.70, 0.79),
            "V34-A3": summary(0.70, 0.01, 0.90, 0.90),
        }
    )
    assert selected == "V34-A2"
    assert rows[0]["candidate_id"] == "V34-A2"
