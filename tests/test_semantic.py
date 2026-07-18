"""Focused tests for offline semantic-model utilities."""

from __future__ import annotations

import contextlib
import io
import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from trace_ace.semantic import (  # noqa: E402
    DEFAULT_SEED,
    LOGISTIC_C,
    SemanticLogisticModel,
    build_semantic_interaction_matrix,
    encode_texts,
    load_sentence_transformer,
)
from trace_ace.semantic_training import fit_semantic_model  # noqa: E402


class DummyTransformer:
    """Small local stand-in for SentenceTransformer."""

    calls: list[tuple[str, dict[str, object]]] = []

    def __init__(self, path: str, **kwargs: object) -> None:
        self.calls.append((path, kwargs))
        self.max_seq_length: int | None = None
        self.eval_calls = 0
        self.encode_kwargs: dict[str, object] | None = None

    def eval(self) -> DummyTransformer:
        self.eval_calls += 1
        return self

    def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
        self.encode_kwargs = kwargs
        return np.asarray(
            [[float(len(text)), float(index + 1), 2.0] for index, text in enumerate(texts)],
            dtype=np.float64,
        )


class OfflineEncodingTests(unittest.TestCase):
    def setUp(self) -> None:
        DummyTransformer.calls.clear()

    def test_loader_is_local_only_and_configures_sequence_length(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "trace_ace.semantic._sentence_transformer_type",
                return_value=DummyTransformer,
            ):
                encoder = load_sentence_transformer(
                    directory,
                    max_seq_length=384,
                    device="cpu",
                )

        self.assertEqual(encoder.max_seq_length, 384)
        self.assertEqual(len(DummyTransformer.calls), 1)
        path, kwargs = DummyTransformer.calls[0]
        self.assertTrue(Path(path).is_absolute())
        self.assertEqual(kwargs["device"], "cpu")
        self.assertIs(kwargs["local_files_only"], True)

    def test_loader_rejects_missing_local_assets(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_sentence_transformer(
                "/path/that/does/not/exist",
                max_seq_length=128,
            )

    def test_encoding_is_quiet_deterministic_normalized_float32(self) -> None:
        encoder = DummyTransformer("/local", local_files_only=True)
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            first = encode_texts(encoder, ["alpha", "beta", "gamma"], batch_size=2)
            second = encode_texts(encoder, ["alpha", "beta", "gamma"], batch_size=2)

        self.assertEqual(captured.getvalue(), "")
        self.assertEqual(first.dtype, np.float32)
        np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1.0, atol=1e-6)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(encoder.eval_calls, 2)
        self.assertEqual(encoder.encode_kwargs["batch_size"], 2)
        self.assertIs(encoder.encode_kwargs["show_progress_bar"], False)
        self.assertIs(encoder.encode_kwargs["convert_to_numpy"], True)
        self.assertIs(encoder.encode_kwargs["normalize_embeddings"], False)

    def test_encoding_rejects_invalid_output(self) -> None:
        class InvalidEncoder:
            def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
                del texts, kwargs
                return np.asarray([[0.0, 0.0]], dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "zero-norm"):
            encode_texts(InvalidEncoder(), ["text"])


class SemanticInteractionTests(unittest.TestCase):
    def test_interaction_layout_and_values(self) -> None:
        contexts = np.asarray([[3.0, 4.0], [0.0, 2.0]], dtype=np.float64)
        objectives = np.asarray([[0.0, 5.0], [2.0, 0.0]], dtype=np.float64)
        dense = np.asarray([[7.0, 8.0], [9.0, 10.0]], dtype=np.float64)
        interactions = build_semantic_interaction_matrix(
            contexts,
            objectives,
            dense_features=dense,
        )

        self.assertEqual(interactions.shape, (2, 11))
        self.assertEqual(interactions.dtype, np.float32)
        context_norm = np.asarray([[0.6, 0.8], [0.0, 1.0]], dtype=np.float32)
        objective_norm = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        np.testing.assert_allclose(interactions[:, 0:2], context_norm)
        np.testing.assert_allclose(interactions[:, 2:4], objective_norm)
        np.testing.assert_allclose(
            interactions[:, 4:6],
            np.abs(context_norm - objective_norm),
        )
        np.testing.assert_allclose(
            interactions[:, 6:8],
            context_norm * objective_norm,
        )
        np.testing.assert_allclose(interactions[:, 8], [0.8, 0.0])
        np.testing.assert_allclose(interactions[:, 9:11], dense)

    def test_interactions_reject_bad_shapes_and_non_finite_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "identical shapes"):
            build_semantic_interaction_matrix(
                np.ones((2, 3)),
                np.ones((2, 4)),
            )
        with self.assertRaisesRegex(ValueError, "non-finite"):
            build_semantic_interaction_matrix(
                np.asarray([[1.0, np.nan]]),
                np.ones((1, 2)),
            )
        with self.assertRaisesRegex(ValueError, "must have 2 rows"):
            build_semantic_interaction_matrix(
                np.ones((2, 2)),
                np.ones((2, 2)),
                dense_features=np.ones((1, 2)),
            )


class SemanticLogisticModelTests(unittest.TestCase):
    def setUp(self) -> None:
        generator = np.random.default_rng(42)
        self.features = generator.normal(size=(40, 7)).astype(np.float32)
        self.targets = (self.features[:, 0] + 0.5 * self.features[:, 1] > 0).astype(int)

    def test_fold_fit_predict_and_metadata_are_deterministic(self) -> None:
        first = fit_semantic_model(
            self.features, self.targets, c=LOGISTIC_C, seed=DEFAULT_SEED
        )
        second = fit_semantic_model(
            self.features, self.targets, c=LOGISTIC_C, seed=DEFAULT_SEED
        )
        first_probabilities = first.predict_proba(self.features)
        second_probabilities = second.predict_proba(self.features)

        self.assertEqual(first_probabilities.shape, (40, 2))
        np.testing.assert_allclose(first_probabilities.sum(axis=1), 1.0)
        np.testing.assert_array_equal(first_probabilities, second_probabilities)
        np.testing.assert_allclose(first.scaler_.mean_, self.features.mean(axis=0), atol=1e-6)

        metadata = first.metadata_dict()
        self.assertEqual(metadata["feature_count"], 7)
        self.assertEqual(metadata["classes"], [0, 1])
        self.assertEqual(metadata["logistic_c"], LOGISTIC_C)
        self.assertEqual(metadata["random_seed"], DEFAULT_SEED)
        json.dumps(metadata)

        restored = pickle.loads(pickle.dumps(first))
        np.testing.assert_array_equal(
            first_probabilities,
            restored.predict_proba(self.features),
        )

    def test_predict_before_fit_and_invalid_fit_are_rejected(self) -> None:
        with self.assertRaisesRegex(Exception, "not been fitted"):
            SemanticLogisticModel().predict_proba(self.features)
        with self.assertRaisesRegex(ValueError, "both binary classes"):
            fit_semantic_model(
                self.features,
                np.zeros(40, dtype=int),
                c=LOGISTIC_C,
                seed=DEFAULT_SEED,
            )
        invalid = self.features.copy()
        invalid[0, 0] = np.inf
        with self.assertRaisesRegex(ValueError, "non-finite"):
            fit_semantic_model(
                invalid,
                self.targets,
                c=LOGISTIC_C,
                seed=DEFAULT_SEED,
            )


if __name__ == "__main__":
    unittest.main()
