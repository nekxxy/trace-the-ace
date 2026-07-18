from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from trace_ace.features import (  # noqa: E402
    DENSE_FEATURE_NAMES,
    FeatureConfig,
    TRANSCRIPT_COLUMNS,
    extract_objective_features,
)


def _transcript() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("session-a", 0, "background", "Lesson begins", "00:00:00"),
            ("session-a", 1, "tutor", "What is the numerator of this fraction?", "00:00:05"),
            ("session-a", 2, "student", "I think the numerator is three because it is on top.", "00:00:09"),
            ("session-a", 3, "tutor", "Yes, exactly. The numerator is three.", "00:00:12"),
            ("session-a", 4, "student", "Then the denominator is four.", "00:00:15"),
            ("session-a", 5, "tutor", "Correct, well done.", "00:00:18"),
            ("session-a", 6, "student", "I am not sure about an improper fraction [unclear].", "00:00:21"),
            ("session-a", 7, "tutor", "Check that again: the numerator can be larger.", "00:00:25"),
            ("session-a", 8, "student", "So five over four is improper.", "00:00:29"),
            ("session-a", 9, "tutor", "Great, that is right.", "00:00:32"),
        ],
        columns=TRANSCRIPT_COLUMNS,
    )


def test_extraction_is_deterministic_with_fixed_tie_breaking() -> None:
    transcript = _transcript()
    objective = "Identify numerator and denominator in a fraction"

    first = extract_objective_features(transcript, objective)
    second = extract_objective_features(
        transcript.sample(frac=1.0, random_state=19).reset_index(drop=True), objective
    )

    assert first == second
    assert first.objective_context
    assert first.dense_feature_names == DENSE_FEATURE_NAMES


def test_every_text_view_respects_its_character_budget() -> None:
    config = FeatureConfig(
        context_char_budget=73,
        role_char_budget=61,
        window_char_budget=67,
        opening_closing_char_budget=43,
        top_k_lines=8,
        neighbor_radius=2,
        opening_closing_lines=5,
        max_window_gap=4,
    )
    features = extract_objective_features(
        _transcript(), "numerator denominator fraction", config=config
    )

    assert len(features.objective_context) <= config.context_char_budget
    assert len(features.tutor_relevant_text) <= config.role_char_budget
    assert len(features.student_relevant_text) <= config.role_char_budget
    assert len(features.tutor_question_student_answer_windows) <= config.window_char_budget
    assert len(features.student_answer_tutor_feedback_windows) <= config.window_char_budget
    assert len(features.opening_evidence) <= config.opening_closing_char_budget
    assert len(features.closing_evidence) <= config.opening_closing_char_budget

    no_edge_evidence = extract_objective_features(
        _transcript(),
        "fraction",
        config=FeatureConfig(opening_closing_lines=0),
    )
    assert no_edge_evidence.opening_evidence == ""
    assert no_edge_evidence.closing_evidence == ""


def test_empty_objective_and_empty_content_are_safe() -> None:
    transcript = pd.DataFrame(
        [
            ("empty-session", 0, "tutor", None, "00:00:00"),
            ("empty-session", 1, "student", np.nan, "not-a-time"),
            ("empty-session", 2, "background", "   ", "00:00:02"),
        ],
        columns=TRANSCRIPT_COLUMNS,
    )

    features = extract_objective_features(transcript, "")

    assert features.objective_context == ""
    assert features.tutor_relevant_text == ""
    assert features.student_relevant_text == ""
    assert features.opening_evidence == ""
    assert features.closing_evidence == ""
    assert len(features.dense) == len(DENSE_FEATURE_NAMES)
    assert np.isfinite(np.asarray(features.dense)).all()


def test_role_ordered_windows_are_extracted() -> None:
    features = extract_objective_features(_transcript(), "fractions and numerator")

    qa = features.tutor_question_student_answer_windows
    feedback = features.student_answer_tutor_feedback_windows
    assert "tutor: What is the numerator" in qa
    assert "student: I think the numerator" in qa
    assert qa.index("tutor:") < qa.index("student:")
    assert "student: I think the numerator" in feedback
    assert "tutor: Yes, exactly" in feedback
    assert feedback.index("student:") < feedback.index("tutor:")


def test_dense_vector_is_fixed_length_named_and_finite() -> None:
    features = extract_objective_features(
        _transcript(), "Identify numerator and denominator in a fraction"
    )
    dense = np.asarray(features.dense)

    assert dense.shape == (len(DENSE_FEATURE_NAMES),)
    assert len(DENSE_FEATURE_NAMES) == len(set(DENSE_FEATURE_NAMES))
    assert np.isfinite(dense).all()
    values = features.dense_dict()
    assert values["objective_coverage"] > 0
    assert values["duration_seconds"] == 32.0
    assert values["tutor_question_student_answer_window_count"] >= 1
    assert values["student_answer_tutor_feedback_window_count"] >= 1
    assert "student_reasoning_ratio_late_minus_early" in values


def test_calls_are_batch_independent_and_do_not_leak_state() -> None:
    target = _transcript()
    objective = "numerator in a fraction"
    before = extract_objective_features(target, objective)

    unrelated = pd.DataFrame(
        [
            ("other-session", 0, "tutor", "Discuss an unrelated geometry topic?", "00:00:00"),
            ("other-session", 1, "student", "A triangle has three sides.", "00:00:02"),
        ],
        columns=TRANSCRIPT_COLUMNS,
    )
    extract_objective_features(unrelated, "triangle geometry")
    after = extract_objective_features(target, objective)

    assert before == after
    assert "triangle" not in after.objective_context.casefold()


def test_required_schema_and_single_session_are_enforced() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        extract_objective_features(pd.DataFrame({"content": ["hello"]}), "hello")

    transcript = _transcript()
    transcript.loc[1, "session_id"] = "different-session"
    with pytest.raises(ValueError, match="exactly one"):
        extract_objective_features(transcript, "fraction")
