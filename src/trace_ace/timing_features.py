"""Deterministic, per-response transcript timing/latency features.

Like ``features.py`` and ``graph_features.py``, every function here is a
pure, stateless function of *one* transcript and *one* learning objective's
response row. Nothing is fit or aggregated across responses or sessions, and
the response's own correctness label is never an input.

Existing dense features (``duration_seconds``, ``utterances_per_minute`` in
``features.py``) only capture whole-session totals. Everything here is about
*local, per-turn* pacing: how long a student takes to respond to a specific
tutor turn, whether that changes over the session, and whether it changes
around hints/corrections specifically - none of which a whole-session
average or rate can express.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
import pandas as pd

from trace_ace.features import _Line, _prepare_lines, tokenize


FAST_RESPONSE_SECONDS = 3.0
SLOW_RESPONSE_SECONDS = 30.0

HINT_OR_QUESTION_PHRASES = (
    "here's a hint",
    "heres a hint",
    "think about",
    "remember that",
    "what if you",
    "consider",
    "recall that",
    "a clue",
)
CORRECTION_PHRASES = (
    "not quite",
    "not correct",
    "incorrect",
    "actually,",
    "that's not",
    "thats not",
    "close, but",
    "try again",
)

TIMING_DENSE_FEATURE_NAMES = (
    "timing_student_mean_latency",
    "timing_student_latency_std",
    "timing_student_latency_trend",
    "timing_max_gap_seconds",
    "timing_gap_after_hint",
    "timing_gap_after_correction",
    "timing_pace_cv",
    "timing_fast_response_ratio",
    "timing_slow_response_ratio",
)


@dataclass(frozen=True)
class TimingConditionedFeatures:
    """Fixed-order immutable numeric vector of timing/latency features."""

    dense: tuple[float, ...]

    @property
    def dense_feature_names(self) -> tuple[str, ...]:
        return TIMING_DENSE_FEATURE_NAMES

    def dense_dict(self) -> dict[str, float]:
        return dict(zip(TIMING_DENSE_FEATURE_NAMES, self.dense, strict=True))


def _contains_any(content: str, phrases: Sequence[str]) -> bool:
    haystack = f" {' '.join(tokenize(content, remove_stopwords=False))} "
    return any(
        f" {' '.join(tokenize(phrase, remove_stopwords=False))} " in haystack
        for phrase in phrases
    )


def _is_hint_or_question_local(content: str) -> bool:
    return _contains_any(content, HINT_OR_QUESTION_PHRASES) or content.strip().endswith("?")


def _is_correction_local(content: str) -> bool:
    return _contains_any(content, CORRECTION_PHRASES)


def _gap(left: _Line, right: _Line) -> float:
    if not math.isfinite(left.timestamp_seconds) or not math.isfinite(right.timestamp_seconds):
        return math.nan
    gap = right.timestamp_seconds - left.timestamp_seconds
    return gap if gap >= 0.0 else math.nan


def _late_minus_early(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    thirds = np.array_split(np.arange(len(values)), 3)
    early, _mid, late = thirds
    early_mean = float(np.mean([values[i] for i in early])) if len(early) else 0.0
    late_mean = float(np.mean([values[i] for i in late])) if len(late) else 0.0
    return late_mean - early_mean


def extract_timing_features(
    transcript: pd.DataFrame,
    objective: object,
) -> TimingConditionedFeatures:
    """Extract deterministic timing-latency features from one transcript.

    Parameters mirror ``features.extract_objective_features`` and
    ``graph_features.extract_graph_features``: a single session transcript
    and the response row's own learning-objective text (unused here, kept
    only so callers can pass the same two arguments as the sibling
    extractors). No other row's data, and no correctness label of any kind,
    is an input.
    """

    del objective  # timing is a property of the transcript alone
    lines = _prepare_lines(transcript)
    nonempty = [line for line in lines if line.content]
    if len(nonempty) < 2:
        return TimingConditionedFeatures(dense=tuple(0.0 for _ in TIMING_DENSE_FEATURE_NAMES))

    consecutive_gaps = [
        _gap(left, right) for left, right in zip(nonempty, nonempty[1:])
    ]
    valid_gaps = [gap for gap in consecutive_gaps if math.isfinite(gap)]

    student_latencies: list[float] = []
    hint_latencies: list[float] = []
    correction_latencies: list[float] = []
    for left, right, gap in zip(nonempty, nonempty[1:], consecutive_gaps):
        if right.role != "student" or left.role != "tutor" or not math.isfinite(gap):
            continue
        student_latencies.append(gap)
        if _is_hint_or_question_local(left.content):
            hint_latencies.append(gap)
        if _is_correction_local(left.content):
            correction_latencies.append(gap)

    mean_latency = float(np.mean(student_latencies)) if student_latencies else 0.0
    std_latency = float(np.std(student_latencies)) if len(student_latencies) > 1 else 0.0
    latency_trend = _late_minus_early(student_latencies)
    max_gap = float(max(valid_gaps)) if valid_gaps else 0.0
    gap_after_hint = float(np.mean(hint_latencies)) if hint_latencies else mean_latency
    gap_after_correction = (
        float(np.mean(correction_latencies)) if correction_latencies else mean_latency
    )
    gap_mean = float(np.mean(valid_gaps)) if valid_gaps else 0.0
    gap_std = float(np.std(valid_gaps)) if len(valid_gaps) > 1 else 0.0
    pace_cv = gap_std / gap_mean if gap_mean > 0.0 else 0.0
    fast_ratio = (
        float(np.mean([latency <= FAST_RESPONSE_SECONDS for latency in student_latencies]))
        if student_latencies
        else 0.0
    )
    slow_ratio = (
        float(np.mean([latency >= SLOW_RESPONSE_SECONDS for latency in student_latencies]))
        if student_latencies
        else 0.0
    )

    values = {
        "timing_student_mean_latency": mean_latency,
        "timing_student_latency_std": std_latency,
        "timing_student_latency_trend": latency_trend,
        "timing_max_gap_seconds": max_gap,
        "timing_gap_after_hint": gap_after_hint,
        "timing_gap_after_correction": gap_after_correction,
        "timing_pace_cv": pace_cv,
        "timing_fast_response_ratio": fast_ratio,
        "timing_slow_response_ratio": slow_ratio,
    }
    vector = np.asarray([values[name] for name in TIMING_DENSE_FEATURE_NAMES], dtype=np.float64)
    vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
    return TimingConditionedFeatures(dense=tuple(float(value) for value in vector))


__all__ = [
    "TIMING_DENSE_FEATURE_NAMES",
    "TimingConditionedFeatures",
    "extract_timing_features",
]
