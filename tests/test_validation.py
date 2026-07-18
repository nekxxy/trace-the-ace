"""Focused synthetic tests for leakage-safe validation."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from trace_ace.validation import (  # noqa: E402
    DEFAULT_SEED,
    assert_fold_isolation,
    assign_semantic_families,
    evaluate_probabilities,
    expected_calibration_error,
    objective_disjoint_folds,
    semantic_family_disjoint_folds,
    session_grouped_folds,
)


def synthetic_responses() -> pd.DataFrame:
    """Return balanced rows with both pure and cross-objective sessions."""

    rows: list[dict[str, object]] = []
    objectives = [f"objective_{number}" for number in range(6)]

    # Pure sessions guarantee training rows remain after cross-session purges.
    for objective_number, objective in enumerate(objectives):
        for repetition in range(3):
            session = f"pure_{objective_number}_{repetition}"
            rows.extend(
                [
                    {
                        "response_id": f"{session}_0",
                        "session_id": session,
                        "learning_objective_id": objective,
                        "is_correct": 0,
                    },
                    {
                        "response_id": f"{session}_1",
                        "session_id": session,
                        "learning_objective_id": objective,
                        "is_correct": 1,
                    },
                ]
            )

    # These sessions span objective boundaries and must be purged when only one
    # of their objectives lands in validation.
    cross_pairs = [(0, 2), (1, 4), (2, 5), (3, 0), (4, 2), (5, 1)]
    for number, (left, right) in enumerate(cross_pairs):
        session = f"cross_{number}"
        rows.extend(
            [
                {
                    "response_id": f"{session}_0",
                    "session_id": session,
                    "learning_objective_id": objectives[left],
                    "is_correct": 0,
                },
                {
                    "response_id": f"{session}_1",
                    "session_id": session,
                    "learning_objective_id": objectives[right],
                    "is_correct": 1,
                },
            ]
        )
    return pd.DataFrame(rows)


class ValidationSplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = synthetic_responses()

    def test_session_folds_are_deterministic_and_disjoint(self) -> None:
        first = session_grouped_folds(self.frame, n_splits=3)
        second = session_grouped_folds(self.frame, n_splits=3)

        self.assertEqual(DEFAULT_SEED, 20260716)
        self.assertEqual(len(first), 3)
        coverage = np.zeros(len(self.frame), dtype=int)
        for left, right in zip(first, second, strict=True):
            np.testing.assert_array_equal(left.train_mask, right.train_mask)
            np.testing.assert_array_equal(left.validation_mask, right.validation_mask)
            self.assertEqual(left.purged_count, 0)
            assert_fold_isolation(
                self.frame,
                left.train_mask,
                left.validation_mask,
            )
            coverage += left.validation_mask
        np.testing.assert_array_equal(coverage, np.ones(len(self.frame), dtype=int))

    def test_objective_folds_purge_validation_sessions(self) -> None:
        folds = objective_disjoint_folds(self.frame, n_splits=3)
        total_purged = 0

        for fold in folds:
            assert_fold_isolation(
                self.frame,
                fold.train_mask,
                fold.validation_mask,
                objective_col="learning_objective_id",
            )
            validation_sessions = set(
                self.frame.loc[fold.validation_mask, "session_id"]
            )
            self.assertFalse(
                self.frame.loc[fold.train_mask, "session_id"].isin(validation_sessions).any()
            )
            total_purged += fold.purged_count

        self.assertGreater(total_purged, 0)

    def test_semantic_family_assignment_and_folds_are_deterministic(self) -> None:
        objective_ids = [f"objective_{number}" for number in range(6)]
        embeddings = pd.DataFrame(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [10.0, 10.0],
                [10.1, 10.0],
                [20.0, 0.0],
                [20.1, 0.0],
            ],
            index=objective_ids,
        )
        first = assign_semantic_families(embeddings, n_families=3)
        shuffled = assign_semantic_families(
            embeddings.sample(frac=1.0, random_state=91),
            n_families=3,
        )
        pd.testing.assert_series_equal(first.sort_index(), shuffled.sort_index())
        self.assertEqual(first.nunique(), 3)

        folds = semantic_family_disjoint_folds(
            self.frame,
            first,
            n_splits=3,
            protocol_name="semantic_k3",
        )
        row_families = self.frame["learning_objective_id"].map(first)
        for fold in folds:
            self.assertEqual(fold.protocol, "semantic_k3")
            assert_fold_isolation(
                self.frame,
                fold.train_mask,
                fold.validation_mask,
                objective_col="learning_objective_id",
                semantic_families=row_families,
            )

    def test_missing_family_assignment_is_rejected(self) -> None:
        incomplete = pd.Series(
            {f"objective_{number}": number for number in range(5)},
            name="semantic_family",
        )
        with self.assertRaisesRegex(ValueError, "missing"):
            semantic_family_disjoint_folds(self.frame, incomplete, n_splits=3)


class ProbabilityMetricTests(unittest.TestCase):
    def test_probability_metrics(self) -> None:
        targets = np.asarray([0, 1, 1, 0])
        probabilities = np.asarray([0.1, 0.8, 0.6, 0.3])
        metrics = evaluate_probabilities(targets, probabilities)

        self.assertEqual(list(metrics), ["log_loss", "roc_auc", "brier", "ece10"])
        self.assertAlmostEqual(
            metrics["log_loss"],
            log_loss(targets, probabilities, labels=[0, 1]),
        )
        self.assertAlmostEqual(metrics["roc_auc"], roc_auc_score(targets, probabilities))
        self.assertAlmostEqual(metrics["brier"], brier_score_loss(targets, probabilities))
        self.assertAlmostEqual(metrics["ece10"], 0.25)
        self.assertAlmostEqual(
            expected_calibration_error(targets, probabilities),
            metrics["ece10"],
        )

    def test_auc_is_nan_for_a_single_class(self) -> None:
        metrics = evaluate_probabilities([1, 1], [0.7, 0.8])
        self.assertTrue(math.isnan(metrics["roc_auc"]))
        self.assertTrue(math.isfinite(metrics["log_loss"]))

    def test_invalid_probabilities_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            evaluate_probabilities([0, 1], [0.2, 1.1])


if __name__ == "__main__":
    unittest.main()
