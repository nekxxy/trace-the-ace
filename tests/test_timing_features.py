from __future__ import annotations

import inspect
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from trace_ace.features import TRANSCRIPT_COLUMNS  # noqa: E402
from trace_ace.timing_features import (  # noqa: E402
    TIMING_DENSE_FEATURE_NAMES,
    extract_timing_features,
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
                "00:00:45",
            ),
            ("session-a", 5, "tutor", "Correct, well done.", "00:00:48"),
            (
                "session-a",
                6,
                "student",
                "I am not sure about improper fractions.",
                "00:00:50",
            ),
            (
                "session-a",
                7,
                "tutor",
                "Remember that the numerator can be larger than the denominator.",
                "00:00:55",
            ),
            (
                "session-a",
                8,
                "student",
                "So five over four is improper, I am confident now.",
                "00:00:58",
            ),
            ("session-a", 9, "tutor", "Great, that is right.", "00:01:01"),
        ],
        columns=TRANSCRIPT_COLUMNS,
    )


def test_signature_has_no_label_shaped_input() -> None:
    """Structural leakage guard: the extractor cannot see any row's outcome."""

    parameters = list(inspect.signature(extract_timing_features).parameters)
    assert parameters == ["transcript", "objective"]


def test_extraction_is_deterministic_under_row_reordering() -> None:
    transcript = _transcript()
    objective = "Identify numerator and denominator in a fraction"

    first = extract_timing_features(transcript, objective)
    second = extract_timing_features(
        transcript.sample(frac=1.0, random_state=7).reset_index(drop=True), objective
    )

    assert first == second
    assert first.dense_feature_names == TIMING_DENSE_FEATURE_NAMES
    assert all(np.isfinite(value) for value in first.dense)


def test_empty_transcript_content_yields_all_zero_features() -> None:
    transcript = pd.DataFrame(
        [("session-b", 0, "background", "", "00:00:00")],
        columns=TRANSCRIPT_COLUMNS,
    )
    result = extract_timing_features(transcript, "Any objective")
    assert result.dense == tuple(0.0 for _ in TIMING_DENSE_FEATURE_NAMES)


def test_single_line_transcript_yields_all_zero_features() -> None:
    transcript = pd.DataFrame(
        [("session-e", 0, "student", "Hello", "00:00:00")],
        columns=TRANSCRIPT_COLUMNS,
    )
    result = extract_timing_features(transcript, "Any objective")
    assert result.dense == tuple(0.0 for _ in TIMING_DENSE_FEATURE_NAMES)


def test_fixture_reflects_expected_latency_signals() -> None:
    transcript = _transcript()
    result = extract_timing_features(
        transcript, "Identify numerator and denominator in a fraction"
    )
    values = result.dense_dict()

    # tutor@5 -> student@9 = 4s, tutor@12 -> student@45 = 33s,
    # tutor@48 -> student@50 = 2s, tutor@55 -> student@58 = 3s
    assert values["timing_student_mean_latency"] == (4 + 33 + 2 + 3) / 4
    assert values["timing_max_gap_seconds"] == 33.0
    # two hint-tagged tutor lines ("think about...", 4s gap; "remember that...", 3s gap)
    assert values["timing_gap_after_hint"] == (4.0 + 3.0) / 2
    # the correction-tagged tutor line ("not quite...") precedes the 33s gap
    assert values["timing_gap_after_correction"] == 33.0
    assert 0.0 <= values["timing_fast_response_ratio"] <= 1.0
    assert 0.0 <= values["timing_slow_response_ratio"] <= 1.0


def test_missing_objective_does_not_crash_and_stays_finite() -> None:
    transcript = _transcript()
    result = extract_timing_features(transcript, None)
    assert all(np.isfinite(value) for value in result.dense)


def test_out_of_order_timestamps_are_ignored_not_negative() -> None:
    """A backwards timestamp (bad data) must not produce a negative gap that
    corrupts mean/std/max - it should just be excluded, like a missing one."""

    transcript = pd.DataFrame(
        [
            ("session-f", 0, "tutor", "Question one?", "00:00:10"),
            ("session-f", 1, "student", "Answer one.", "00:00:05"),
            ("session-f", 2, "tutor", "Question two?", "00:00:20"),
            ("session-f", 3, "student", "Answer two.", "00:00:25"),
        ],
        columns=TRANSCRIPT_COLUMNS,
    )
    result = extract_timing_features(transcript, "topic")
    values = result.dense_dict()
    assert values["timing_student_mean_latency"] == 5.0
    assert all(np.isfinite(value) and value >= 0.0 for name, value in values.items() if "trend" not in name)
