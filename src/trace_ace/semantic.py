"""Offline semantic encoding, interactions, and a deterministic classifier.

This module never logs text, embeddings, or data summaries.  The optional
SentenceTransformer dependency is imported only when a local model is loaded,
which keeps the remaining utilities usable in lightweight environments.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.exceptions import NotFittedError


DEFAULT_SEED = 20260716
LOGISTIC_C = 0.1
METADATA_VERSION = 1


def load_sentence_transformer(
    asset_path: str | Path,
    *,
    max_seq_length: int,
    device: str = "cpu",
) -> Any:
    """Load a SentenceTransformer strictly from an existing local directory.

    ``local_files_only=True`` is mandatory and there is deliberately no remote
    model-name fallback.  Missing assets or a missing optional dependency fail
    before any text is encoded.
    """

    path = Path(asset_path).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"local SentenceTransformer asset directory not found: {path}")
    if not isinstance(max_seq_length, int) or max_seq_length < 1:
        raise ValueError("max_seq_length must be a positive integer")
    if not isinstance(device, str) or not device.strip():
        raise ValueError("device must be a non-empty string")

    transformer_type = _sentence_transformer_type()
    model = transformer_type(
        str(path),
        device=device,
        local_files_only=True,
    )
    model.max_seq_length = max_seq_length
    return model


def encode_texts(
    encoder: Any,
    texts: Sequence[str],
    *,
    batch_size: int = 32,
) -> np.ndarray:
    """Encode text deterministically and return L2-normalized float32 rows."""

    if isinstance(texts, (str, bytes)):
        raise TypeError("texts must be a sequence of strings, not one string")
    values = list(texts)
    if not values:
        raise ValueError("texts must contain at least one item")
    if any(not isinstance(value, str) for value in values):
        raise TypeError("every text value must be a string")
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if not callable(getattr(encoder, "encode", None)):
        raise TypeError("encoder must provide a callable encode method")

    eval_method = getattr(encoder, "eval", None)
    if callable(eval_method):
        eval_method()
    encoded = encoder.encode(
        values,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    matrix = _finite_matrix(encoded, "encoded embeddings", expected_rows=len(values))
    return _l2_normalize(matrix, "encoded embeddings")


def build_semantic_interaction_matrix(
    context_embeddings: np.ndarray | Sequence[Sequence[float]],
    objective_embeddings: np.ndarray | Sequence[Sequence[float]],
    *,
    dense_features: np.ndarray | Sequence[Sequence[float]] | None = None,
) -> np.ndarray:
    """Build normalized context/objective semantic interactions.

    The output columns are, in order: normalized context, normalized objective,
    absolute difference, elementwise product, cosine similarity, and optional
    caller-supplied dense features.
    """

    contexts = _finite_matrix(context_embeddings, "context_embeddings")
    objectives = _finite_matrix(
        objective_embeddings,
        "objective_embeddings",
        expected_rows=contexts.shape[0],
    )
    if contexts.shape != objectives.shape:
        raise ValueError("context and objective embeddings must have identical shapes")

    contexts = _l2_normalize(contexts, "context_embeddings")
    objectives = _l2_normalize(objectives, "objective_embeddings")
    difference = np.abs(contexts - objectives)
    product = contexts * objectives
    cosine = np.sum(product, axis=1, keepdims=True)
    pieces = [contexts, objectives, difference, product, cosine]

    if dense_features is not None:
        dense = _finite_matrix(
            dense_features,
            "dense_features",
            expected_rows=contexts.shape[0],
        )
        pieces.append(dense)

    interactions = np.concatenate(pieces, axis=1)
    if not np.isfinite(interactions).all():
        raise ValueError("semantic interaction matrix contains non-finite values")
    return np.asarray(interactions, dtype=np.float32)


@dataclass(frozen=True)
class SemanticModelMetadata:
    """JSON/joblib-friendly description of a fitted semantic classifier."""

    format_version: int
    estimator: str
    feature_count: int
    classes: tuple[int, ...]
    logistic_c: float
    random_seed: int
    max_iter: int
    sklearn_version: str

    def to_dict(self) -> dict[str, object]:
        """Return metadata using only standard serializable value types."""

        values = asdict(self)
        values["classes"] = list(self.classes)
        return values


class SemanticLogisticModel:
    """Runtime container for a pre-fitted scaler and logistic classifier."""

    def __init__(
        self,
        *,
        c: float = LOGISTIC_C,
        seed: int = DEFAULT_SEED,
        max_iter: int = 1_000,
    ) -> None:
        if not isinstance(max_iter, int) or max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        if not np.isfinite(c) or c <= 0:
            raise ValueError("c must be a positive finite value")
        if not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        self.c = float(c)
        self.seed = seed
        self.max_iter = max_iter

    def predict_proba(
        self,
        features: np.ndarray | Sequence[Sequence[float]],
    ) -> np.ndarray:
        """Return two-column class probabilities without refitting anything."""

        self._require_fitted()
        matrix = _finite_matrix(features, "features")
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError(
                f"features has {matrix.shape[1]} columns; expected {self.n_features_in_}"
            )
        probabilities = self.classifier_.predict_proba(self.scaler_.transform(matrix))
        if probabilities.shape != (matrix.shape[0], 2) or not np.isfinite(probabilities).all():
            raise ValueError("classifier returned invalid probabilities")
        return np.asarray(probabilities, dtype=np.float64)

    def metadata_dict(self) -> dict[str, object]:
        """Return fitted model metadata as a plain serializable dictionary."""

        self._require_fitted()
        return self.metadata_.to_dict()

    def _require_fitted(self) -> None:
        if not all(
            hasattr(self, attribute)
            for attribute in ("scaler_", "classifier_", "n_features_in_", "metadata_")
        ):
            raise NotFittedError("SemanticLogisticModel has not been fitted")


def _sentence_transformer_type() -> type[Any]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise ImportError(
            "sentence-transformers is required only for offline semantic encoding"
        ) from error
    return SentenceTransformer


def _finite_matrix(
    values: np.ndarray | Sequence[Sequence[float]],
    name: str,
    *,
    expected_rows: int | None = None,
) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix")
    if expected_rows is not None and matrix.shape[0] != expected_rows:
        raise ValueError(f"{name} must have {expected_rows} rows")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains non-finite values")
    return matrix


def _l2_normalize(matrix: np.ndarray, name: str) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0):
        raise ValueError(f"{name} contains a zero-norm or invalid row")
    normalized = matrix / norms
    return np.asarray(normalized, dtype=np.float32)


__all__ = [
    "DEFAULT_SEED",
    "LOGISTIC_C",
    "SemanticLogisticModel",
    "SemanticModelMetadata",
    "build_semantic_interaction_matrix",
    "encode_texts",
    "load_sentence_transformer",
]
