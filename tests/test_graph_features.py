from __future__ import annotations

import inspect
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from trace_ace.features import TRANSCRIPT_COLUMNS  # noqa: E402
from trace_ace.graph_features import (  # noqa: E402
    GRAPH_DENSE_FEATURE_NAMES,
    extract_graph_features,
)


def _transcript() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("session-a", 0, "background", "Lesson begins", "00:00:00"),
            (
                "session-a",
                1,
                "tutor",
                "What is the numerator of this fraction? Think about the top number.",
                "00:00:05",
            ),
            ("session-a", 2, "student", "I am not sure, maybe three?", "00:00:09"),
            ("session-a", 3, "tutor", "Not quite, check that again.", "00:00:12"),
            (
                "session-a",
                4,
                "student",
                "I think it is three because it is on top.",
                "00:00:15",
            ),
            ("session-a", 5, "tutor", "Correct, well done.", "00:00:18"),
            (
                "session-a",
                6,
                "student",
                "I am not sure about improper fractions.",
                "00:00:21",
            ),
            (
                "session-a",
                7,
                "tutor",
                "Remember that the numerator can be larger than the denominator.",
                "00:00:25",
            ),
            (
                "session-a",
                8,
                "student",
                "So five over four is improper, I am confident now.",
                "00:00:29",
            ),
            ("session-a", 9, "tutor", "Great, that is right.", "00:00:32"),
        ],
        columns=TRANSCRIPT_COLUMNS,
    )


def test_signature_has_no_label_shaped_input() -> None:
    """Structural leakage guard: the extractor cannot see any row's outcome.

    This function only ever receives a transcript and an objective string -
    there is no parameter it could use to read ``is_correct`` (this row's or
    a sibling's), so a leakage audit reduces to checking this signature
    never grows one.
    """

    parameters = list(inspect.signature(extract_graph_features).parameters)
    assert parameters == ["transcript", "objective"]


def test_extraction_is_deterministic_under_row_reordering() -> None:
    transcript = _transcript()
    objective = "Identify numerator and denominator in a fraction"

    first = extract_graph_features(transcript, objective)
    second = extract_graph_features(
        transcript.sample(frac=1.0, random_state=7).reset_index(drop=True), objective
    )

    assert first == second
    assert first.dense_feature_names == GRAPH_DENSE_FEATURE_NAMES
    assert all(np.isfinite(value) for value in first.dense)


def test_empty_transcript_content_yields_all_zero_features() -> None:
    transcript = pd.DataFrame(
        [("session-b", 0, "background", "", "00:00:00")],
        columns=TRANSCRIPT_COLUMNS,
    )
    result = extract_graph_features(transcript, "Any objective")
    assert result.dense == tuple(0.0 for _ in GRAPH_DENSE_FEATURE_NAMES)


def test_fixture_reflects_expected_tutor_and_student_signals() -> None:
    transcript = _transcript()
    result = extract_graph_features(
        transcript, "Identify numerator and denominator in a fraction"
    )
    values = result.dense_dict()

    # 2 tutor hint-style lines ("think about", "remember that").
    assert values["graph_hint_count"] == 2.0
    # 1 tutor correction line ("not quite, check that again").
    assert values["graph_correction_chain_length"] == 1.0
    # A later uncertainty line after the correction counts as persistence.
    assert values["graph_misconception_persistence"] >= 1.0
    # More than one distinct tutor action type appears (hint/question/correction/feedback).
    assert values["graph_strategy_diversity"] >= 2.0
    assert values["graph_node_count"] > 0.0
    assert 0.0 <= values["graph_density"] <= 1.0


def test_missing_objective_does_not_crash_and_stays_finite() -> None:
    transcript = _transcript()
    result = extract_graph_features(transcript, None)
    assert all(np.isfinite(value) for value in result.dense)


def test_graph_density_and_degree_are_bounded() -> None:
    """Self-transitions (e.g. two consecutive tutor hints) must not count as
    edges, or density/degree can exceed their intended [0, 1] / [0, node_count-1]
    bounds - regression test for a bug where repeated same-type turns inflated
    both past their conventional simple-directed-graph ranges."""

    transcript = pd.DataFrame(
        [
            ("session-d", 0, "tutor", "Think about the numerator.", "00:00:00"),
            ("session-d", 1, "tutor", "Consider what the top number means.", "00:00:05"),
            ("session-d", 2, "tutor", "Remember that fractions have two parts.", "00:00:10"),
        ],
        columns=TRANSCRIPT_COLUMNS,
    )
    result = extract_graph_features(transcript, "fractions")
    values = result.dense_dict()
    assert 0.0 <= values["graph_density"] <= 1.0
    assert values["graph_max_node_degree"] <= max(values["graph_node_count"] - 1.0, 0.0)


def test_single_role_transcript_has_zero_cross_role_edges() -> None:
    transcript = pd.DataFrame(
        [
            ("session-c", 0, "student", "I think this equals four because it is doubled.", "00:00:00"),
            ("session-c", 1, "student", "Actually I am not sure now.", "00:00:05"),
        ],
        columns=TRANSCRIPT_COLUMNS,
    )
    result = extract_graph_features(transcript, "Doubling numbers")
    values = result.dense_dict()
    assert values["graph_hint_count"] == 0.0
    assert values["graph_tutor_intervention_count"] == 0.0
