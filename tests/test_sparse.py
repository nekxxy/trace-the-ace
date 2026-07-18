from __future__ import annotations

import numpy as np
import pandas as pd

from trace_ace.config import ModelConfig
from trace_ace.sparse import (
    append_scaled_dense,
    positive_probability,
    transform_full_text,
    transform_role_text,
)
from trace_ace.training import fit_dense_scaler, new_sparse_classifier


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "full_text": [
                "tutor: explain the fraction student: one half",
                "tutor: multiply student: I am unsure",
                "tutor: good reasoning student: because they are equal",
                "tutor: try again student: three quarters",
            ],
            "tutor_text": ["explain the fraction", "multiply", "good reasoning", "try again"],
            "student_text": ["one half", "I am unsure", "because they are equal", "three quarters"],
            "learning_objective": ["fractions", "multiplication", "equivalence", "fractions"],
        }
    )


def test_hash_features_are_deterministic_and_separated() -> None:
    cfg = ModelConfig(full_hash_features=2**10, role_hash_features=2**9, objective_hash_features=2**8)
    frame = _frame()
    first = transform_full_text(frame, cfg)
    second = transform_full_text(frame, cfg)
    assert first.shape == (4, 2**10 + 2**8)
    assert (first != second).nnz == 0
    assert transform_role_text(frame, cfg).shape == (4, 2 * 2**9 + 2**8)


def test_incremental_classifier_and_dense_append_are_valid() -> None:
    cfg = ModelConfig(full_hash_features=2**10, role_hash_features=2**9, objective_hash_features=2**8)
    frame = _frame()
    dense = np.asarray([[1.0, 0.0], [2.0, 1.0], [3.0, 0.0], [4.0, 1.0]], dtype=np.float32)
    scaler = fit_dense_scaler(dense)
    matrix = append_scaled_dense(transform_role_text(frame, cfg), dense, scaler)
    target = np.asarray([1, 0, 1, 0], dtype=np.int8)
    model = new_sparse_classifier(alpha=cfg.role_alpha, config=cfg)
    model.partial_fit(matrix, target, classes=np.asarray([0, 1], dtype=np.int8))
    probability = positive_probability(model, matrix)
    assert probability.shape == (4,)
    assert np.isfinite(probability).all()
    assert ((probability >= 0.0) & (probability <= 1.0)).all()


def test_batch_prediction_is_invariant() -> None:
    cfg = ModelConfig(full_hash_features=2**10, role_hash_features=2**9, objective_hash_features=2**8)
    frame = _frame()
    matrix = transform_full_text(frame, cfg)
    target = np.asarray([1, 0, 1, 0], dtype=np.int8)
    model = new_sparse_classifier(alpha=cfg.full_alpha, config=cfg)
    model.partial_fit(matrix, target, classes=np.asarray([0, 1], dtype=np.int8))
    together = positive_probability(model, matrix)
    separately = np.concatenate([positive_probability(model, matrix[index : index + 1]) for index in range(4)])
    np.testing.assert_allclose(together, separately, rtol=0.0, atol=1e-12)
