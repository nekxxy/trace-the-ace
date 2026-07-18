"""Final serializable ensemble and leakage-free runtime inference."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterator

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.utils.validation import check_is_fitted

from trace_ace.config import BGE_DIMENSION, ModelConfig
from trace_ace.features import DENSE_FEATURE_NAMES, FeatureConfig, extract_objective_features
from trace_ace.io import read_transcript
from trace_ace.provenance import asset_tree_sha256, selected_source_sha256
from trace_ace.semantic import (
    SemanticLogisticModel,
    build_semantic_interaction_matrix,
    encode_texts,
    load_sentence_transformer,
)
from trace_ace.sparse import (
    SparseArtifact,
    append_scaled_dense,
    positive_probability,
    transform_full_text,
    transform_role_text,
)


ARTIFACT_FORMAT_VERSION = 1
ROLE_DENSE_SCALER_GEOMETRY = "standardized-unit-expected-l2-v1"
SEMANTIC_SCALER_GEOMETRY = "natural-semantic-unit-dense-l2-v1"
RUNTIME_SOURCE_FILES = (
    "__init__.py",
    "config.py",
    "ensemble.py",
    "features.py",
    "io.py",
    "provenance.py",
    "semantic.py",
    "sparse.py",
)


def blend_probabilities(
    full: np.ndarray,
    role: np.ndarray,
    semantic: np.ndarray,
    *,
    weights: tuple[float, float, float],
    probability_floor: float,
) -> np.ndarray:
    components = [np.asarray(values, dtype=np.float64) for values in (full, role, semantic)]
    if any(values.ndim != 1 for values in components):
        raise ValueError("component probabilities must be one-dimensional")
    if len({len(values) for values in components}) != 1:
        raise ValueError("component probability lengths differ")
    if len(weights) != 3 or min(weights) < 0 or not np.isclose(sum(weights), 1.0):
        raise ValueError("ensemble weights must be nonnegative and sum to one")
    if not np.isfinite(probability_floor) or not 0 <= probability_floor < 0.5:
        raise ValueError("probability_floor must be in [0, 0.5)")
    if any(not np.isfinite(values).all() for values in components):
        raise ValueError("component probabilities contain non-finite values")
    blended = sum(weight * values for weight, values in zip(weights, components, strict=True))
    return np.clip(blended, probability_floor, 1.0 - probability_floor)


@dataclass
class FinalEnsembleArtifact:
    format_version: int
    config: dict[str, object]
    feature_config: dict[str, object]
    sparse: SparseArtifact
    semantic: SemanticLogisticModel
    weights: tuple[float, float, float]
    training_metadata: dict[str, object]

    def validate_runtime(self) -> ModelConfig:
        if self.format_version != ARTIFACT_FORMAT_VERSION:
            raise RuntimeError("unsupported model artifact format")
        self.sparse.validate_runtime()
        if self.sparse.config != self.config:
            raise RuntimeError("top-level and sparse model configurations differ")
        config = ModelConfig(**self.config)
        try:
            feature_config = FeatureConfig(**self.feature_config)
        except (TypeError, ValueError) as error:
            raise RuntimeError("invalid serialized feature configuration") from error
        expected_dense_names = tuple(f"dense__{name}" for name in DENSE_FEATURE_NAMES)
        if self.sparse.dense_feature_names != expected_dense_names:
            raise RuntimeError("dense feature names/order differ from the runtime")
        expected_full = config.full_hash_features + config.objective_hash_features
        expected_role = (
            2 * config.role_hash_features
            + config.objective_hash_features
            + len(DENSE_FEATURE_NAMES)
        )
        for estimator, expected in (
            (self.sparse.full_model, expected_full),
            (self.sparse.role_model, expected_role),
        ):
            try:
                check_is_fitted(estimator)
            except Exception as error:
                raise RuntimeError("sparse estimator is not fitted") from error
            if getattr(estimator, "n_features_in_", None) != expected:
                raise RuntimeError("sparse estimator feature dimension is invalid")
            if not np.array_equal(getattr(estimator, "classes_", None), [0, 1]):
                raise RuntimeError("sparse estimator classes are invalid")
        try:
            check_is_fitted(self.sparse.role_dense_scaler)
        except Exception as error:
            raise RuntimeError("dense scaler is not fitted") from error
        if getattr(self.sparse.role_dense_scaler, "n_features_in_", None) != len(
            DENSE_FEATURE_NAMES
        ):
            raise RuntimeError("dense scaler feature dimension is invalid")
        if getattr(self.semantic, "metadata_", None) is None:
            raise RuntimeError("semantic model is not fitted")
        if self.semantic.metadata_.sklearn_version != sklearn.__version__:
            raise RuntimeError("semantic model/runtime scikit-learn version mismatch")
        if len(self.weights) != 3 or not np.isclose(sum(self.weights), 1.0):
            raise RuntimeError("invalid ensemble weights")
        if min(self.weights) < 0:
            raise RuntimeError("ensemble weights must be nonnegative")
        expected_weights = (
            config.full_weight,
            config.role_weight,
            config.semantic_weight,
        )
        if not np.allclose(self.weights, expected_weights, rtol=0.0, atol=0.0):
            raise RuntimeError("ensemble weights differ from model configuration")
        expected_semantic = 4 * BGE_DIMENSION + 1 + len(DENSE_FEATURE_NAMES)
        if getattr(self.semantic, "n_features_in_", None) != expected_semantic:
            raise RuntimeError("semantic model feature dimension is invalid")
        _validate_scaler_geometry(
            role_scaler=self.sparse.role_dense_scaler,
            semantic_scaler=getattr(self.semantic, "scaler_", None),
            semantic_feature_count=4 * BGE_DIMENSION + 1,
            dense_block_width=len(DENSE_FEATURE_NAMES),
        )
        if self.semantic.metadata_.logistic_c != config.semantic_c:
            raise RuntimeError("semantic regularization differs from model configuration")
        if self.semantic.metadata_.random_seed != config.seed:
            raise RuntimeError("semantic seed differs from model configuration")
        expected_source_hash = self.training_metadata.get("runtime_source_sha256")
        actual_source_hash = selected_source_sha256(
            Path(__file__).resolve().parent,
            RUNTIME_SOURCE_FILES,
        )
        if expected_source_hash != actual_source_hash:
            raise RuntimeError("runtime source differs from the trained artifact")
        if feature_config.context_char_budget != config.context_char_budget:
            raise RuntimeError("feature and model context budgets differ")
        return config


def _validate_scaler_geometry(
    *,
    role_scaler: object,
    semantic_scaler: object,
    semantic_feature_count: int,
    dense_block_width: int,
) -> None:
    """Reject fitted artifacts that predate or violate the locked geometry."""

    if (
        getattr(role_scaler, "trace_ace_geometry_", None)
        != ROLE_DENSE_SCALER_GEOMETRY
        or getattr(role_scaler, "trace_ace_block_width_", None)
        != dense_block_width
        or getattr(role_scaler, "n_features_in_", None) != dense_block_width
    ):
        raise RuntimeError("role dense scaler geometry is invalid")
    role_mean = np.asarray(getattr(role_scaler, "mean_", []))
    role_scale = np.asarray(getattr(role_scaler, "scale_", []))
    if (
        role_mean.shape != (dense_block_width,)
        or role_scale.shape != (dense_block_width,)
        or not np.isfinite(role_mean).all()
        or not np.isfinite(role_scale).all()
        or np.any(role_scale <= 0)
    ):
        raise RuntimeError("role dense scaler parameters are invalid")
    expected_semantic_total = semantic_feature_count + dense_block_width
    if (
        getattr(semantic_scaler, "trace_ace_geometry_", None)
        != SEMANTIC_SCALER_GEOMETRY
        or getattr(semantic_scaler, "trace_ace_semantic_feature_count_", None)
        != semantic_feature_count
        or getattr(semantic_scaler, "trace_ace_dense_block_width_", None)
        != dense_block_width
        or getattr(semantic_scaler, "n_features_in_", None)
        != expected_semantic_total
    ):
        raise RuntimeError("semantic scaler geometry is invalid")
    semantic_mean = np.asarray(getattr(semantic_scaler, "mean_", []))
    semantic_scale = np.asarray(getattr(semantic_scaler, "scale_", []))
    if (
        semantic_mean.shape != (expected_semantic_total,)
        or semantic_scale.shape != (expected_semantic_total,)
        or not np.isfinite(semantic_mean).all()
        or not np.isfinite(semantic_scale).all()
        or np.any(semantic_scale <= 0)
        or not np.array_equal(
            semantic_mean[:semantic_feature_count],
            np.zeros(semantic_feature_count),
        )
        or not np.array_equal(
            semantic_scale[:semantic_feature_count],
            np.ones(semantic_feature_count),
        )
    ):
        raise RuntimeError("semantic scaler parameters violate locked geometry")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_rows(
    features: pd.DataFrame,
    transcripts_dir: Path,
    *,
    feature_config: FeatureConfig,
) -> Iterator[dict[str, object]]:
    required = [
        "response_id",
        "session_id",
        "learning_objective_id",
        "learning_objective",
    ]
    if list(features.columns) != required:
        raise ValueError("invalid test feature schema")
    if features[required].isna().any().any() or features["response_id"].duplicated().any():
        raise ValueError("invalid test feature rows")
    for _, session_rows in features.groupby("session_id", sort=False):
        session_id = str(session_rows.iloc[0]["session_id"])
        transcript_path = transcripts_dir / f"{session_id}.csv"
        if not transcript_path.is_file():
            raise FileNotFoundError("a required transcript file is missing")
        transcript = read_transcript(
            transcript_path, expected_session_id=session_id
        )
        roles = transcript["role"].astype(str).str.casefold().tolist()
        contents = transcript["content"].astype(str).tolist()
        if any(role not in {"tutor", "student", "background"} for role in roles):
            raise ValueError("transcript contains an unsupported role")
        full_text = "\n".join(
            f"{role.upper()}: {content}" for role, content in zip(roles, contents, strict=True)
        )
        tutor_text = "\n".join(
            content for role, content in zip(roles, contents, strict=True) if role == "tutor"
        )
        student_text = "\n".join(
            content for role, content in zip(roles, contents, strict=True) if role == "student"
        )
        for response in session_rows.itertuples(index=False):
            extracted = extract_objective_features(
                transcript,
                response.learning_objective,
                config=feature_config,
            )
            yield {
                "response_id": str(response.response_id),
                "full_text": full_text,
                "tutor_text": tutor_text,
                "student_text": student_text,
                "learning_objective": str(response.learning_objective),
                "objective_context": extracted.objective_context,
                "dense": extracted.dense,
            }


def _batches(iterator: Iterator[dict[str, object]], size: int) -> Iterator[list[dict[str, object]]]:
    batch: list[dict[str, object]] = []
    for row in iterator:
        batch.append(row)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def predict_test_directory(
    *,
    data_dir: str | Path,
    artifact_path: str | Path,
    bge_asset_path: str | Path,
    runtime_batch_size: int = 128,
) -> pd.DataFrame:
    """Predict independently transformed test responses without fitting anything."""

    if runtime_batch_size < 1:
        raise ValueError("runtime_batch_size must be positive")
    data_root = Path(data_dir)
    artifact = joblib.load(artifact_path)
    if not isinstance(artifact, FinalEnsembleArtifact):
        raise RuntimeError("invalid model artifact")
    config = artifact.validate_runtime()
    if artifact.training_metadata.get("bge_asset_tree_sha256") != asset_tree_sha256(
        bge_asset_path
    ):
        raise RuntimeError("BGE asset tree differs from the trained artifact")
    features = pd.read_csv(data_root / "test_features.csv")
    submission_format = pd.read_csv(data_root / "submission_format.csv")
    if list(submission_format.columns) != ["response_id", "probability"]:
        raise ValueError("invalid submission format schema")
    if submission_format["response_id"].duplicated().any():
        raise ValueError("submission format contains duplicate IDs")
    if set(features["response_id"].astype(str)) != set(submission_format["response_id"].astype(str)):
        raise ValueError("test features and submission format IDs differ")

    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    encoder = load_sentence_transformer(
        bge_asset_path,
        max_seq_length=config.bge_max_seq_length,
        device=device,
    )
    encoding_batch_size = (
        config.bge_batch_size_gpu if device == "cuda" else config.bge_batch_size_cpu
    )
    feature_config = FeatureConfig(**artifact.feature_config)
    probabilities: dict[str, float] = {}
    iterator = _runtime_rows(
        features,
        data_root / "test_transcripts",
        feature_config=feature_config,
    )
    for rows in _batches(iterator, runtime_batch_size):
        frame = pd.DataFrame(
            {
                key: [row[key] for row in rows]
                for key in ("full_text", "tutor_text", "student_text", "learning_objective")
            }
        )
        dense = np.asarray([row["dense"] for row in rows], dtype=np.float32)
        full_probability = positive_probability(
            artifact.sparse.full_model,
            transform_full_text(frame, config),
        )
        role_matrix = append_scaled_dense(
            transform_role_text(frame, config),
            dense,
            artifact.sparse.role_dense_scaler,
        )
        role_probability = positive_probability(artifact.sparse.role_model, role_matrix)
        context_embeddings = encode_texts(
            encoder,
            [str(row["objective_context"]) for row in rows],
            batch_size=encoding_batch_size,
        )
        objective_embeddings = encode_texts(
            encoder,
            [str(row["learning_objective"]) for row in rows],
            batch_size=encoding_batch_size,
        )
        semantic_features = build_semantic_interaction_matrix(
            context_embeddings,
            objective_embeddings,
            dense_features=dense,
        )
        semantic_probability = artifact.semantic.predict_proba(semantic_features)[:, 1]
        blended = blend_probabilities(
            full_probability,
            role_probability,
            semantic_probability,
            weights=artifact.weights,
            probability_floor=config.probability_floor,
        )
        for row, probability in zip(rows, blended, strict=True):
            probabilities[str(row["response_id"])] = float(probability)

    output = submission_format[["response_id"]].copy()
    output["probability"] = output["response_id"].astype(str).map(probabilities)
    if output["probability"].isna().any() or not np.isfinite(output["probability"]).all():
        raise RuntimeError("inference did not produce every required probability")
    return output
