"""Deterministic, per-response transcript feature extraction.

The functions in this module are deliberately stateless.  They inspect exactly
one transcript and one learning objective at a time, and never fit or update
corpus-level state.  This keeps the resulting features legal for independent
competition-test inference.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


TRANSCRIPT_COLUMNS = (
    "session_id",
    "utterance_id",
    "role",
    "content",
    "timestamp",
)

TOKEN_RE = re.compile(r"[a-z0-9]+(?:['’][a-z0-9]+)?", re.IGNORECASE)
SPACE_RE = re.compile(r"\s+")
TIME_RE = re.compile(
    r"^\s*(?P<hours>\d+):(?P<minutes>[0-5]\d):(?P<seconds>[0-5]\d(?:\.\d+)?)\s*$"
)
UNCLEAR_RE = re.compile(r"\[\s*unclear\s*\]", re.IGNORECASE)

# Frozen locally rather than imported from an NLP package so token selection is
# stable across machines and package versions.
STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "being",
        "by",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "hers",
        "him",
        "his",
        "i",
        "in",
        "into",
        "is",
        "it",
        "its",
        "me",
        "my",
        "of",
        "on",
        "or",
        "our",
        "ours",
        "she",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "they",
        "this",
        "those",
        "to",
        "us",
        "was",
        "we",
        "were",
        "with",
        "you",
        "your",
        "yours",
    }
)

QUESTION_STARTERS = frozenset(
    {
        "am",
        "are",
        "can",
        "could",
        "did",
        "do",
        "does",
        "explain",
        "has",
        "have",
        "how",
        "is",
        "shall",
        "should",
        "tell",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "would",
    }
)

FEEDBACK_PHRASES = (
    "correct",
    "exactly",
    "excellent",
    "good job",
    "great",
    "nice",
    "perfect",
    "right",
    "that's it",
    "thats it",
    "well done",
    "yes",
)
CORRECTION_PHRASES = (
    "be careful",
    "check that",
    "incorrect",
    "instead",
    "mistake",
    "no,",
    "not quite",
    "try again",
    "wrong",
)
REASONING_PHRASES = (
    "because",
    "equals",
    "first",
    "i think",
    "if ",
    "means",
    "since",
    "so ",
    "then",
    "therefore",
)
UNCERTAINTY_PHRASES = (
    "confused",
    "don't know",
    "dont know",
    "guess",
    "i'm not sure",
    "im not sure",
    "maybe",
    "not sure",
    "unsure",
)


@dataclass(frozen=True)
class FeatureConfig:
    """Fixed extraction limits; all units are characters or utterance rows."""

    context_char_budget: int = 4096
    role_char_budget: int = 3072
    window_char_budget: int = 4096
    opening_closing_char_budget: int = 2048
    top_k_lines: int = 12
    neighbor_radius: int = 1
    opening_closing_lines: int = 10
    max_window_gap: int = 4

    def __post_init__(self) -> None:
        integer_fields = (
            "context_char_budget",
            "role_char_budget",
            "window_char_budget",
            "opening_closing_char_budget",
            "top_k_lines",
            "neighbor_radius",
            "opening_closing_lines",
            "max_window_gap",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")


SEGMENT_METRICS = (
    "student_share",
    "student_mean_chars",
    "student_reasoning_ratio",
    "student_uncertainty_ratio",
    "tutor_question_ratio",
    "tutor_feedback_ratio",
    "tutor_correction_ratio",
)

_BASE_DENSE_FEATURE_NAMES = (
    "objective_token_count",
    "objective_unique_token_count",
    "matched_objective_unique_count",
    "objective_coverage",
    "relevant_line_count",
    "relevant_line_ratio",
    "max_line_overlap",
    "mean_positive_line_overlap",
    "first_match_position",
    "last_match_position",
    "mean_match_position",
    "selected_context_char_ratio",
    "utterance_count",
    "nonempty_utterance_count",
    "tutor_count",
    "student_count",
    "background_count",
    "other_role_count",
    "tutor_ratio",
    "student_ratio",
    "background_ratio",
    "other_role_ratio",
    "role_transition_count",
    "role_transition_ratio",
    "unclear_occurrence_count",
    "unclear_line_count",
    "unclear_line_ratio",
    "total_char_count",
    "total_token_count",
    "mean_line_chars",
    "median_line_chars",
    "max_line_chars",
    "duration_seconds",
    "utterances_per_minute",
    "tutor_char_ratio",
    "student_char_ratio",
    "tutor_question_count",
    "tutor_question_ratio",
    "student_question_count",
    "student_question_ratio",
    "tutor_feedback_count",
    "tutor_feedback_ratio",
    "tutor_correction_count",
    "tutor_correction_ratio",
    "tutor_question_student_answer_window_count",
    "student_answer_tutor_feedback_window_count",
)

DENSE_FEATURE_NAMES = (
    _BASE_DENSE_FEATURE_NAMES
    + tuple(
        f"{segment}_{metric}"
        for segment in ("early", "mid", "late")
        for metric in SEGMENT_METRICS
    )
    + tuple(
        f"{metric}_{delta}"
        for metric in SEGMENT_METRICS
        for delta in ("mid_minus_early", "late_minus_early")
    )
)


@dataclass(frozen=True)
class ObjectiveConditionedFeatures:
    """Independent text views and a fixed-order immutable numeric vector."""

    objective_context: str
    tutor_relevant_text: str
    student_relevant_text: str
    tutor_question_student_answer_windows: str
    student_answer_tutor_feedback_windows: str
    opening_evidence: str
    closing_evidence: str
    dense: tuple[float, ...]

    @property
    def dense_feature_names(self) -> tuple[str, ...]:
        return DENSE_FEATURE_NAMES

    def dense_dict(self) -> dict[str, float]:
        return dict(zip(DENSE_FEATURE_NAMES, self.dense, strict=True))


@dataclass(frozen=True)
class _Line:
    order: int
    utterance_id: str
    role: str
    content: str
    timestamp_seconds: float
    tokens: tuple[str, ...]

    @property
    def rendered(self) -> str:
        return f"{self.role}: {self.content}" if self.content else ""


@dataclass(frozen=True)
class _Window:
    start: int
    score: float
    text: str


def tokenize(text: object, *, remove_stopwords: bool = True) -> tuple[str, ...]:
    """Return frozen, version-independent lexical tokens."""

    normalized = "" if text is None else str(text).casefold().replace("’", "'")
    tokens = tuple(match.group(0) for match in TOKEN_RE.finditer(normalized))
    if remove_stopwords:
        tokens = tuple(token for token in tokens if token not in STOPWORDS)
    return tokens


def extract_objective_features(
    transcript: pd.DataFrame,
    objective: object,
    *,
    config: FeatureConfig | None = None,
) -> ObjectiveConditionedFeatures:
    """Extract deterministic features from one transcript/objective pair.

    Parameters
    ----------
    transcript:
        A single-session frame with the exact competition transcript columns.
        Extra columns are ignored. Rows are deterministically ordered by
        ``utterance_id`` with content-based tie breakers.
    objective:
        The response row's learning-objective text.
    config:
        Optional fixed limits. No value is learned from another response.
    """

    cfg = config or FeatureConfig()
    lines = _prepare_lines(transcript)
    objective_tokens = tokenize(objective)
    if not objective_tokens and _normalize_text(objective):
        objective_tokens = tokenize(objective, remove_stopwords=False)

    objective_counts = Counter(objective_tokens)
    line_scores = tuple(_line_overlap(line.tokens, objective_counts) for line in lines)
    positive_indices = tuple(index for index, score in enumerate(line_scores) if score > 0.0)
    anchor_indices = tuple(
        sorted(positive_indices, key=lambda index: (-line_scores[index], index))[
            : cfg.top_k_lines
        ]
    )

    context_priority: list[int] = []
    for anchor in anchor_indices:
        context_priority.append(anchor)
        for distance in range(1, cfg.neighbor_radius + 1):
            previous = anchor - distance
            following = anchor + distance
            if previous >= 0:
                context_priority.append(previous)
            if following < len(lines):
                context_priority.append(following)

    objective_context = _select_lines(lines, context_priority, cfg.context_char_budget)
    tutor_anchors = sorted(
        (index for index in positive_indices if lines[index].role == "tutor"),
        key=lambda index: (-line_scores[index], index),
    )[: cfg.top_k_lines]
    student_anchors = sorted(
        (index for index in positive_indices if lines[index].role == "student"),
        key=lambda index: (-line_scores[index], index),
    )[: cfg.top_k_lines]
    tutor_relevant_text = _select_lines(lines, tutor_anchors, cfg.role_char_budget)
    student_relevant_text = _select_lines(lines, student_anchors, cfg.role_char_budget)

    qa_windows = _question_answer_windows(lines, line_scores, cfg.max_window_gap)
    feedback_windows = _answer_feedback_windows(lines, line_scores, cfg.max_window_gap)
    qa_text = _select_windows(qa_windows, cfg.window_char_budget)
    feedback_text = _select_windows(feedback_windows, cfg.window_char_budget)

    nonempty_indices = tuple(index for index, line in enumerate(lines) if line.content)
    opening_indices = nonempty_indices[: cfg.opening_closing_lines]
    closing_indices = (
        nonempty_indices[-cfg.opening_closing_lines :]
        if cfg.opening_closing_lines
        else ()
    )
    opening_evidence = _join_lines(
        (lines[index] for index in opening_indices), cfg.opening_closing_char_budget
    )
    closing_evidence = _join_lines(
        (lines[index] for index in closing_indices), cfg.opening_closing_char_budget
    )

    dense = _dense_features(
        lines=lines,
        objective_tokens=objective_tokens,
        line_scores=line_scores,
        positive_indices=positive_indices,
        selected_context=objective_context,
        context_budget=cfg.context_char_budget,
        qa_window_count=len(qa_windows),
        feedback_window_count=len(feedback_windows),
    )
    return ObjectiveConditionedFeatures(
        objective_context=objective_context,
        tutor_relevant_text=tutor_relevant_text,
        student_relevant_text=student_relevant_text,
        tutor_question_student_answer_windows=qa_text,
        student_answer_tutor_feedback_windows=feedback_text,
        opening_evidence=opening_evidence,
        closing_evidence=closing_evidence,
        dense=dense,
    )


def _prepare_lines(transcript: pd.DataFrame) -> tuple[_Line, ...]:
    if not isinstance(transcript, pd.DataFrame):
        raise TypeError("transcript must be a pandas DataFrame")
    missing = [column for column in TRANSCRIPT_COLUMNS if column not in transcript.columns]
    if missing:
        raise ValueError(f"transcript is missing required columns: {missing}")

    records: list[dict[str, object]] = []
    session_ids: set[str] = set()
    for row in transcript.loc[:, TRANSCRIPT_COLUMNS].itertuples(index=False, name=None):
        session_id = _normalize_text(row[0])
        if session_id:
            session_ids.add(session_id)
        utterance_id = _normalize_text(row[1])
        try:
            numeric_id = float(utterance_id)
            invalid_numeric_id = not math.isfinite(numeric_id)
        except ValueError:
            numeric_id = 0.0
            invalid_numeric_id = True
        role = _normalize_text(row[2]).casefold() or "other"
        content = _normalize_text(row[3])
        timestamp_seconds = _parse_timestamp(row[4])
        records.append(
            {
                "session_id": session_id,
                "utterance_id": utterance_id,
                "numeric_id": numeric_id,
                "invalid_numeric_id": invalid_numeric_id,
                "role": role,
                "content": content,
                "timestamp_seconds": timestamp_seconds,
            }
        )

    if len(session_ids) > 1:
        raise ValueError("transcript must contain exactly one non-empty session_id")

    records.sort(
        key=lambda record: (
            bool(record["invalid_numeric_id"]),
            float(record["numeric_id"]),
            str(record["utterance_id"]),
            _finite_or(record["timestamp_seconds"], math.inf),
            str(record["role"]),
            str(record["content"]),
            str(record["session_id"]),
        )
    )
    return tuple(
        _Line(
            order=order,
            utterance_id=str(record["utterance_id"]),
            role=str(record["role"]),
            content=str(record["content"]),
            timestamp_seconds=float(record["timestamp_seconds"]),
            tokens=tokenize(record["content"], remove_stopwords=False),
        )
        for order, record in enumerate(records)
    )


def _normalize_text(value: object) -> str:
    if value is None or (not isinstance(value, (list, tuple, dict)) and pd.isna(value)):
        return ""
    return SPACE_RE.sub(" ", str(value)).strip()


def _parse_timestamp(value: object) -> float:
    text = _normalize_text(value)
    match = TIME_RE.match(text)
    if not match:
        return math.nan
    return (
        float(match.group("hours")) * 3600.0
        + float(match.group("minutes")) * 60.0
        + float(match.group("seconds"))
    )


def _line_overlap(tokens: Sequence[str], objective_counts: Counter[str]) -> float:
    if not tokens or not objective_counts:
        return 0.0
    content_counts = Counter(tokens)
    overlap = set(content_counts).intersection(objective_counts)
    if not overlap:
        return 0.0
    unique_coverage = len(overlap) / len(objective_counts)
    frequency_coverage = sum(
        min(content_counts[token], objective_counts[token]) for token in overlap
    ) / sum(objective_counts.values())
    return float(0.8 * unique_coverage + 0.2 * frequency_coverage)


def _select_lines(lines: Sequence[_Line], priority: Iterable[int], budget: int) -> str:
    if budget <= 0:
        return ""
    selected: list[int] = []
    seen: set[int] = set()
    used = 0
    for index in priority:
        if index in seen or index < 0 or index >= len(lines):
            continue
        seen.add(index)
        rendered = lines[index].rendered
        if not rendered:
            continue
        extra = len(rendered) + (1 if selected else 0)
        if used + extra <= budget:
            selected.append(index)
            used += extra
        elif not selected:
            return rendered[:budget]
    return _join_lines((lines[index] for index in sorted(selected)), budget)


def _join_lines(lines: Iterable[_Line], budget: int) -> str:
    return _join_chunks((line.rendered for line in lines if line.rendered), budget)


def _join_chunks(chunks: Iterable[str], budget: int) -> str:
    if budget <= 0:
        return ""
    output: list[str] = []
    used = 0
    for chunk in chunks:
        if not chunk:
            continue
        separator = 1 if output else 0
        remaining = budget - used - separator
        if remaining <= 0:
            break
        if len(chunk) <= remaining:
            output.append(chunk)
            used += separator + len(chunk)
        else:
            output.append(chunk[:remaining])
            used += separator + remaining
            break
    return "\n".join(output)


def _question_answer_windows(
    lines: Sequence[_Line], line_scores: Sequence[float], max_gap: int
) -> tuple[_Window, ...]:
    windows: list[_Window] = []
    for index, line in enumerate(lines):
        if line.role != "tutor" or not _is_question(line.content):
            continue
        stop = min(len(lines), index + max_gap + 1)
        for answer_index in range(index + 1, stop):
            answer = lines[answer_index]
            if answer.role == "student" and answer.content:
                windows.append(
                    _Window(
                        start=index,
                        score=line_scores[index] + line_scores[answer_index],
                        text=f"{line.rendered}\n{answer.rendered}",
                    )
                )
                break
    return tuple(windows)


def _answer_feedback_windows(
    lines: Sequence[_Line], line_scores: Sequence[float], max_gap: int
) -> tuple[_Window, ...]:
    windows: list[_Window] = []
    for index, line in enumerate(lines):
        if line.role != "student" or not line.content:
            continue
        stop = min(len(lines), index + max_gap + 1)
        for feedback_index in range(index + 1, stop):
            feedback = lines[feedback_index]
            if feedback.role != "tutor" or not feedback.content:
                continue
            if _is_feedback(feedback.content) or _is_correction(feedback.content):
                windows.append(
                    _Window(
                        start=index,
                        score=line_scores[index] + line_scores[feedback_index],
                        text=f"{line.rendered}\n{feedback.rendered}",
                    )
                )
                break
    return tuple(windows)


def _select_windows(windows: Sequence[_Window], budget: int) -> str:
    if budget <= 0 or not windows:
        return ""
    ranked = sorted(windows, key=lambda window: (-window.score, window.start, window.text))
    chosen: list[_Window] = []
    used = 0
    for window in ranked:
        separator = 2 if chosen else 0
        if used + separator + len(window.text) <= budget:
            chosen.append(window)
            used += separator + len(window.text)
        elif not chosen:
            return window.text[:budget]
    chosen.sort(key=lambda window: (window.start, window.text))
    return _join_chunks((window.text for window in chosen), budget).replace("\n", "\n", 0)


def _is_question(content: str) -> bool:
    if "?" in content:
        return True
    tokens = tokenize(content, remove_stopwords=False)
    return bool(tokens and tokens[0] in QUESTION_STARTERS)


def _contains_phrase(content: str, phrases: Sequence[str]) -> bool:
    haystack = f" {' '.join(tokenize(content, remove_stopwords=False))} "
    return any(
        f" {' '.join(tokenize(phrase, remove_stopwords=False))} " in haystack
        for phrase in phrases
    )


def _is_feedback(content: str) -> bool:
    return _contains_phrase(content, FEEDBACK_PHRASES)


def _is_correction(content: str) -> bool:
    return _contains_phrase(content, CORRECTION_PHRASES)


def _has_reasoning(content: str) -> bool:
    return _contains_phrase(content, REASONING_PHRASES)


def _has_uncertainty(content: str) -> bool:
    return _contains_phrase(content, UNCERTAINTY_PHRASES)


def _dense_features(
    *,
    lines: Sequence[_Line],
    objective_tokens: Sequence[str],
    line_scores: Sequence[float],
    positive_indices: Sequence[int],
    selected_context: str,
    context_budget: int,
    qa_window_count: int,
    feedback_window_count: int,
) -> tuple[float, ...]:
    count = len(lines)
    nonempty = [line for line in lines if line.content]
    nonempty_count = len(nonempty)
    role_counts = Counter(line.role for line in lines)
    tutor_lines = [line for line in lines if line.role == "tutor"]
    student_lines = [line for line in lines if line.role == "student"]
    background_count = role_counts.get("background", 0)
    other_count = count - role_counts.get("tutor", 0) - role_counts.get("student", 0) - background_count

    transcript_token_set = {token for line in lines for token in line.tokens}
    objective_unique = set(objective_tokens)
    matched_unique = objective_unique.intersection(transcript_token_set)
    positions = [_safe_ratio(index, count - 1) for index in positive_indices]
    positive_scores = [line_scores[index] for index in positive_indices]

    roles = [line.role for line in lines]
    role_transitions = sum(left != right for left, right in zip(roles, roles[1:]))
    unclear_occurrences = sum(len(UNCLEAR_RE.findall(line.content)) for line in lines)
    unclear_lines = sum(bool(UNCLEAR_RE.search(line.content)) for line in lines)
    line_lengths = [len(line.content) for line in lines]
    total_chars = sum(line_lengths)
    total_tokens = sum(len(line.tokens) for line in lines)
    valid_times = [line.timestamp_seconds for line in lines if math.isfinite(line.timestamp_seconds)]
    duration = max(valid_times) - min(valid_times) if valid_times else 0.0
    duration = max(0.0, duration)
    tutor_chars = sum(len(line.content) for line in tutor_lines)
    student_chars = sum(len(line.content) for line in student_lines)

    tutor_questions = sum(_is_question(line.content) for line in tutor_lines)
    student_questions = sum(_is_question(line.content) for line in student_lines)
    tutor_feedback = sum(_is_feedback(line.content) for line in tutor_lines)
    tutor_corrections = sum(_is_correction(line.content) for line in tutor_lines)

    values: dict[str, float] = {
        "objective_token_count": float(len(objective_tokens)),
        "objective_unique_token_count": float(len(objective_unique)),
        "matched_objective_unique_count": float(len(matched_unique)),
        "objective_coverage": _safe_ratio(len(matched_unique), len(objective_unique)),
        "relevant_line_count": float(len(positive_indices)),
        "relevant_line_ratio": _safe_ratio(len(positive_indices), nonempty_count),
        "max_line_overlap": max(positive_scores, default=0.0),
        "mean_positive_line_overlap": _mean(positive_scores),
        "first_match_position": positions[0] if positions else 0.0,
        "last_match_position": positions[-1] if positions else 0.0,
        "mean_match_position": _mean(positions),
        "selected_context_char_ratio": _safe_ratio(len(selected_context), context_budget),
        "utterance_count": float(count),
        "nonempty_utterance_count": float(nonempty_count),
        "tutor_count": float(role_counts.get("tutor", 0)),
        "student_count": float(role_counts.get("student", 0)),
        "background_count": float(background_count),
        "other_role_count": float(other_count),
        "tutor_ratio": _safe_ratio(role_counts.get("tutor", 0), count),
        "student_ratio": _safe_ratio(role_counts.get("student", 0), count),
        "background_ratio": _safe_ratio(background_count, count),
        "other_role_ratio": _safe_ratio(other_count, count),
        "role_transition_count": float(role_transitions),
        "role_transition_ratio": _safe_ratio(role_transitions, count - 1),
        "unclear_occurrence_count": float(unclear_occurrences),
        "unclear_line_count": float(unclear_lines),
        "unclear_line_ratio": _safe_ratio(unclear_lines, nonempty_count),
        "total_char_count": float(total_chars),
        "total_token_count": float(total_tokens),
        "mean_line_chars": _mean(line_lengths),
        "median_line_chars": float(np.median(line_lengths)) if line_lengths else 0.0,
        "max_line_chars": float(max(line_lengths, default=0)),
        "duration_seconds": float(duration),
        "utterances_per_minute": _safe_ratio(nonempty_count * 60.0, duration),
        "tutor_char_ratio": _safe_ratio(tutor_chars, total_chars),
        "student_char_ratio": _safe_ratio(student_chars, total_chars),
        "tutor_question_count": float(tutor_questions),
        "tutor_question_ratio": _safe_ratio(tutor_questions, len(tutor_lines)),
        "student_question_count": float(student_questions),
        "student_question_ratio": _safe_ratio(student_questions, len(student_lines)),
        "tutor_feedback_count": float(tutor_feedback),
        "tutor_feedback_ratio": _safe_ratio(tutor_feedback, len(tutor_lines)),
        "tutor_correction_count": float(tutor_corrections),
        "tutor_correction_ratio": _safe_ratio(tutor_corrections, len(tutor_lines)),
        "tutor_question_student_answer_window_count": float(qa_window_count),
        "student_answer_tutor_feedback_window_count": float(feedback_window_count),
    }

    segment_indices = np.array_split(np.arange(count, dtype=int), 3)
    segment_values: dict[str, dict[str, float]] = {}
    for name, indices in zip(("early", "mid", "late"), segment_indices, strict=True):
        metrics = _segment_metrics([lines[int(index)] for index in indices])
        segment_values[name] = metrics
        for metric, value in metrics.items():
            values[f"{name}_{metric}"] = value
    for metric in SEGMENT_METRICS:
        early = segment_values["early"][metric]
        values[f"{metric}_mid_minus_early"] = segment_values["mid"][metric] - early
        values[f"{metric}_late_minus_early"] = segment_values["late"][metric] - early

    vector = np.asarray([values[name] for name in DENSE_FEATURE_NAMES], dtype=np.float64)
    vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
    return tuple(float(value) for value in vector)


def _segment_metrics(lines: Sequence[_Line]) -> dict[str, float]:
    student = [line for line in lines if line.role == "student"]
    tutor = [line for line in lines if line.role == "tutor"]
    return {
        "student_share": _safe_ratio(len(student), len(lines)),
        "student_mean_chars": _mean([len(line.content) for line in student]),
        "student_reasoning_ratio": _safe_ratio(
            sum(_has_reasoning(line.content) for line in student), len(student)
        ),
        "student_uncertainty_ratio": _safe_ratio(
            sum(_has_uncertainty(line.content) for line in student), len(student)
        ),
        "tutor_question_ratio": _safe_ratio(
            sum(_is_question(line.content) for line in tutor), len(tutor)
        ),
        "tutor_feedback_ratio": _safe_ratio(
            sum(_is_feedback(line.content) for line in tutor), len(tutor)
        ),
        "tutor_correction_ratio": _safe_ratio(
            sum(_is_correction(line.content) for line in tutor), len(tutor)
        ),
    }


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not denominator:
        return 0.0
    value = float(numerator) / float(denominator)
    return value if math.isfinite(value) else 0.0


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _finite_or(value: object, fallback: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if math.isfinite(number) else fallback


__all__ = [
    "DENSE_FEATURE_NAMES",
    "FeatureConfig",
    "ObjectiveConditionedFeatures",
    "TRANSCRIPT_COLUMNS",
    "extract_objective_features",
    "tokenize",
]
