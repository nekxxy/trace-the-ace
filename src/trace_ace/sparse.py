"""Stateless sparse text features and serializable linear components."""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
import scipy
from scipy import sparse
import sklearn
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

from trace_ace.config import ModelConfig


REQUIRED_TEXT_COLUMNS = (
    "full_text",
    "tutor_text",
    "student_text",
    "learning_objective",
)


def _texts(frame: pd.DataFrame, column: str) -> list[str]:
    if column not in frame:
        raise ValueError(f"missing text column: {column}")
    if frame[column].isna().any():
        raise ValueError(f"text column contains missing values: {column}")
    return frame[column].astype(str).tolist()


def _vectorizer(n_features: int, config: ModelConfig) -> HashingVectorizer:
    return HashingVectorizer(
        n_features=n_features,
        analyzer="word",
        ngram_range=(config.ngram_min, config.ngram_max),
        lowercase=True,
        strip_accents="unicode",
        token_pattern=r"(?u)\b\w\w+\b",
        alternate_sign=False,
        norm="l2",
        binary=False,
        dtype=np.float32,
    )


def transform_full_text(
    frame: pd.DataFrame, config: ModelConfig | None = None
) -> sparse.csr_matrix:
    """Hash role-preserving full transcript and objective into separate spaces."""

    cfg = config or ModelConfig()
    full = _vectorizer(cfg.full_hash_features, cfg).transform(_texts(frame, "full_text"))
    objective = _vectorizer(cfg.objective_hash_features, cfg).transform(
        _texts(frame, "learning_objective")
    )
    return sparse.hstack((full, objective), format="csr", dtype=np.float32)


def transform_role_text(
    frame: pd.DataFrame, config: ModelConfig | None = None
) -> sparse.csr_matrix:
    """Hash tutor, student, and objective text into independent namespaces."""

    cfg = config or ModelConfig()
    tutor = _vectorizer(cfg.role_hash_features, cfg).transform(_texts(frame, "tutor_text"))
    student = _vectorizer(cfg.role_hash_features, cfg).transform(
        _texts(frame, "student_text")
    )
    objective = _vectorizer(cfg.objective_hash_features, cfg).transform(
        _texts(frame, "learning_objective")
    )
    return sparse.hstack((tutor, student, objective), format="csr", dtype=np.float32)


def append_scaled_dense(
    text_matrix: sparse.spmatrix,
    dense: np.ndarray,
    scaler: StandardScaler,
) -> sparse.csr_matrix:
    """Append fold-fitted dense features to a sparse text matrix."""

    values = _validate_dense(dense)
    if text_matrix.shape[0] != values.shape[0]:
        raise ValueError("text and dense row counts differ")
    scaled = scaler.transform(values).astype(np.float32, copy=False)
    if not np.isfinite(scaled).all():
        raise ValueError("scaled dense features contain non-finite values")
    return sparse.hstack(
        (text_matrix, sparse.csr_matrix(scaled)), format="csr", dtype=np.float32
    )


def positive_probability(model: SGDClassifier, matrix: sparse.spmatrix) -> np.ndarray:
    """Return finite positive-class probabilities."""

    probabilities = np.asarray(model.predict_proba(matrix)[:, 1], dtype=np.float64)
    if probabilities.ndim != 1 or not np.isfinite(probabilities).all():
        raise ValueError("model produced invalid probabilities")
    return probabilities


@dataclass
class SparseArtifact:
    """Runtime-serializable full and role sparse components."""

    config: dict[str, object]
    full_model: SGDClassifier
    role_model: SGDClassifier
    role_dense_scaler: StandardScaler
    dense_feature_names: tuple[str, ...]
    build_versions: dict[str, str]

    @classmethod
    def create(
        cls,
        *,
        config: ModelConfig,
        full_model: SGDClassifier,
        role_model: SGDClassifier,
        role_dense_scaler: StandardScaler,
        dense_feature_names: tuple[str, ...],
    ) -> "SparseArtifact":
        return cls(
            config=config.to_dict(),
            full_model=full_model,
            role_model=role_model,
            role_dense_scaler=role_dense_scaler,
            dense_feature_names=dense_feature_names,
            build_versions={
                "joblib": joblib.__version__,
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "scikit_learn": sklearn.__version__,
            },
        )

    def validate_runtime(self) -> None:
        """Fail clearly when loading under a different serialization stack."""

        expected = self.build_versions
        actual = {
            "joblib": joblib.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        }
        mismatches = [
            f"{name}: built={expected.get(name)} runtime={version}"
            for name, version in actual.items()
            if expected.get(name) != version
        ]
        if mismatches:
            raise RuntimeError("model/runtime version mismatch: " + "; ".join(mismatches))


def _validate_dense(dense: np.ndarray) -> np.ndarray:
    values = np.asarray(dense, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("dense features must be a 2D matrix")
    if not np.isfinite(values).all():
        raise ValueError("dense features contain non-finite values")
    return values
