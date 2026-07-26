"""Deterministic, per-response knowledge-graph-inspired transcript features.

Like ``features.py``, every function here is a pure, stateless function of
*one* transcript and *one* learning objective. Nothing is fit or aggregated
across responses or sessions, and the response's own correctness label is
never an input - these functions have no way to see it, by construction.
That keeps every feature legal for independent, session-disjoint inference.

The "graph" is a lightweight, implicit structure over the ordered utterances
of one session: each non-empty line is tagged with a small set of heuristic
action/state labels (tutor hint, tutor question, tutor correction, tutor
positive feedback, student reasoning, student uncertainty, student
confidence), and consecutive-line transitions between those labels form the
graph's edges. Nothing here reads or infers exam-style "misconception
category" labels (e.g. "Fractions") - that would require a taxonomy or an
LLM pass this project does not have; instead the *concept* axis reuses the
already-safe, already-produced objective-relevance scoring from
``features.py`` (each row already carries its own ``learning_objective``).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from typing import Sequence

import numpy as np
import pandas as pd

from trace_ace.features import (
    TRANSCRIPT_COLUMNS,
    _Line,
    _is_correction,
    _is_feedback,
    _is_question,
    _line_overlap,
    _prepare_lines,
    tokenize,
)


HINT_PHRASES = (
    "here's a hint",
    "heres a hint",
    "think about",
    "remember that",
    "what if you",
    "consider",
    "recall that",
    "a clue",
)
DIRECT_EXPLANATION_PHRASES = (
    "the answer is",
    "you need to",
    "here is how",
    "here's how",
    "the rule is",
    "this means",
)
CONFIDENCE_PHRASES = (
    "i know",
    "i'm sure",
    "im sure",
    "definitely",
    "easy",
    "of course",
    "i'm confident",
    "im confident",
)

# Fixed, canonical action-type vocabulary. Order is part of the contract:
# node-degree and density statistics are computed over this exact label set.
TUTOR_ACTIONS = ("hint", "question", "direct_explanation", "correction", "feedback")
STUDENT_ACTIONS = ("reasoning", "uncertainty", "confidence")
ACTION_TYPES = TUTOR_ACTIONS + STUDENT_ACTIONS + ("other",)

GRAPH_DENSE_FEATURE_NAMES = (
    "graph_hint_count",
    "graph_hint_depth",
    "graph_strategy_diversity",
    "graph_correction_chain_length",
    "graph_tutor_intervention_count",
    "graph_concept_transition_count",
    "graph_reasoning_improvement_score",
    "graph_confidence_change",
    "graph_misconception_persistence",
    "graph_unresolved_misconception_count",
    "graph_density",
    "graph_mean_node_degree",
    "graph_max_node_degree",
    "graph_node_count",
)


@dataclass(frozen=True)
class GraphConditionedFeatures:
    """Fixed-order immutable numeric vector of graph-inspired features."""

    dense: tuple[float, ...]

    @property
    def dense_feature_names(self) -> tuple[str, ...]:
        return GRAPH_DENSE_FEATURE_NAMES

    def dense_dict(self) -> dict[str, float]:
        return dict(zip(GRAPH_DENSE_FEATURE_NAMES, self.dense, strict=True))


def _contains_any(content: str, phrases: Sequence[str]) -> bool:
    haystack = f" {' '.join(tokenize(content, remove_stopwords=False))} "
    return any(
        f" {' '.join(tokenize(phrase, remove_stopwords=False))} " in haystack
        for phrase in phrases
    )


def _is_hint(content: str) -> bool:
    return _contains_any(content, HINT_PHRASES)


def _is_direct_explanation(content: str) -> bool:
    return _contains_any(content, DIRECT_EXPLANATION_PHRASES)


def _is_confident(content: str) -> bool:
    return _contains_any(content, CONFIDENCE_PHRASES)


def _has_reasoning_local(content: str) -> bool:
    # Mirrors features.py's REASONING_PHRASES without importing a private
    # constant name; kept identical in spirit (cheap discourse markers).
    markers = ("because", "equals", "first", "i think", "since", "so ", "then", "therefore")
    return _contains_any(content, markers)


def _has_uncertainty_local(content: str) -> bool:
    markers = ("confused", "don't know", "dont know", "guess", "not sure", "unsure", "maybe")
    return _contains_any(content, markers)


def _action_types(line: _Line) -> tuple[str, ...]:
    """Return every action/state label that applies to one utterance.

    A line may carry more than one label (e.g. a tutor line can be both a
    question and a hint); it may also carry none, in which case it is
    tagged only as "other" for graph-node purposes.
    """

    content = line.content
    if not content:
        return ()
    labels: list[str] = []
    if line.role == "tutor":
        if _is_hint(content):
            labels.append("hint")
        if _is_question(content):
            labels.append("question")
        if _is_direct_explanation(content):
            labels.append("direct_explanation")
        if _is_correction(content):
            labels.append("correction")
        if _is_feedback(content):
            labels.append("feedback")
    elif line.role == "student":
        if _has_reasoning_local(content):
            labels.append("reasoning")
        if _has_uncertainty_local(content):
            labels.append("uncertainty")
        if _is_confident(content):
            labels.append("confidence")
    return tuple(labels) or ("other",)


def extract_graph_features(
    transcript: pd.DataFrame,
    objective: object,
) -> GraphConditionedFeatures:
    """Extract deterministic graph-inspired features from one transcript.

    Parameters mirror ``features.extract_objective_features``: a single
    session transcript and the response row's own learning-objective text.
    No other row's data - and no correctness label of any kind - is an
    input, so a leakage audit only needs to confirm this signature never
    grows a label-shaped parameter.
    """

    lines = _prepare_lines(transcript)
    nonempty = [line for line in lines if line.content]
    if not nonempty:
        return GraphConditionedFeatures(dense=tuple(0.0 for _ in GRAPH_DENSE_FEATURE_NAMES))

    objective_tokens = tokenize(objective)
    objective_counts = Counter(objective_tokens)
    relevance = [
        _line_overlap(line.tokens, objective_counts) > 0.0 if objective_counts else False
        for line in nonempty
    ]

    line_actions = [_action_types(line) for line in nonempty]

    tutor_action_set = {
        action for actions in line_actions for action in actions if action in TUTOR_ACTIONS
    }
    strategy_diversity = float(len(tutor_action_set))

    hint_count = sum("hint" in actions for actions in line_actions)
    tutor_intervention_count = sum(
        any(action in ("hint", "question", "correction") for action in actions)
        for actions in line_actions
    )

    hint_depth = _max_run(
        ["hint" in actions or "question" in actions for actions in line_actions]
    )
    correction_chain_length = _max_run(["correction" in actions for actions in line_actions])

    concept_transition_count = sum(
        left != right for left, right in zip(relevance, relevance[1:])
    )

    reasoning_flags = [
        "reasoning" in actions
        for line, actions in zip(nonempty, line_actions)
        if line.role == "student"
    ]
    confidence_flags = [
        "confidence" in actions
        for line, actions in zip(nonempty, line_actions)
        if line.role == "student"
    ]
    reasoning_improvement_score = _late_minus_early(reasoning_flags)
    confidence_change = _late_minus_early(confidence_flags)

    misconception_persistence = 0
    unresolved_misconception_count = 0
    seen_correction = False
    for index, actions in enumerate(line_actions):
        if "correction" in actions:
            seen_correction = True
        if "uncertainty" in actions:
            if seen_correction:
                misconception_persistence += 1
            addressed = any(
                "feedback" in later or "correction" in later
                for later in line_actions[index + 1 : index + 5]
            )
            if not addressed:
                unresolved_misconception_count += 1

    node_counter: Counter[str] = Counter()
    edge_set: set[tuple[str, str]] = set()
    for actions in line_actions:
        for action in actions:
            node_counter[action] += 1
    for previous_actions, current_actions in zip(line_actions, line_actions[1:]):
        for left in previous_actions:
            for right in current_actions:
                if left != right:
                    edge_set.add((left, right))

    node_count = len(node_counter)
    possible_edges = node_count * (node_count - 1)
    graph_density = float(len(edge_set)) / possible_edges if possible_edges > 0 else 0.0

    out_degree: Counter[str] = Counter()
    for left, _right in edge_set:
        out_degree[left] += 1
    degrees = [out_degree.get(node, 0) for node in node_counter]
    mean_node_degree = float(np.mean(degrees)) if degrees else 0.0
    max_node_degree = float(max(degrees)) if degrees else 0.0

    values = {
        "graph_hint_count": float(hint_count),
        "graph_hint_depth": float(hint_depth),
        "graph_strategy_diversity": strategy_diversity,
        "graph_correction_chain_length": float(correction_chain_length),
        "graph_tutor_intervention_count": float(tutor_intervention_count),
        "graph_concept_transition_count": float(concept_transition_count),
        "graph_reasoning_improvement_score": reasoning_improvement_score,
        "graph_confidence_change": confidence_change,
        "graph_misconception_persistence": float(misconception_persistence),
        "graph_unresolved_misconception_count": float(unresolved_misconception_count),
        "graph_density": graph_density,
        "graph_mean_node_degree": mean_node_degree,
        "graph_max_node_degree": max_node_degree,
        "graph_node_count": float(node_count),
    }
    vector = np.asarray([values[name] for name in GRAPH_DENSE_FEATURE_NAMES], dtype=np.float64)
    vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
    return GraphConditionedFeatures(dense=tuple(float(value) for value in vector))


def _max_run(flags: Sequence[bool]) -> int:
    best = 0
    current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        best = max(best, current)
    return best


def _late_minus_early(flags: Sequence[bool]) -> float:
    if len(flags) < 2:
        return 0.0
    thirds = np.array_split(np.arange(len(flags)), 3)
    early, _mid, late = thirds
    early_ratio = float(np.mean([flags[i] for i in early])) if len(early) else 0.0
    late_ratio = float(np.mean([flags[i] for i in late])) if len(late) else 0.0
    return late_ratio - early_ratio


__all__ = [
    "ACTION_TYPES",
    "GRAPH_DENSE_FEATURE_NAMES",
    "GraphConditionedFeatures",
    "STUDENT_ACTIONS",
    "TUTOR_ACTIONS",
    "TRANSCRIPT_COLUMNS",
    "extract_graph_features",
]
