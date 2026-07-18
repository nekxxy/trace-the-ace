"""Cross-validation and final fitting for the BGE interaction component."""

from __future__ import annotations

from dataclasses import dataclass
import gc
import hashlib
from pathlib import Path
from typing import Callable, Sequence

import joblib
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from trace_ace.config import ModelConfig
from trace_ace.provenance import (
    SEMANTIC_TRAINING_SOURCE_FILES,
    trace_ace_source_sha256,
    validate_store_manifest_source,
)
from trace_ace.semantic import (
    METADATA_VERSION,
    SemanticLogisticModel,
    SemanticModelMetadata,
    build_semantic_interaction_matrix,
)
from trace_ace.semantic_store import (
    SemanticArrays,
    load_semantic_arrays,
    load_semantic_manifest,
)
from trace_ace.validation import FoldMasks, evaluate_probabilities


@dataclass(frozen=True)
class SemanticCVResult:
    oof: np.ndarray
    fold_metrics: tuple[dict[str, object], ...]


@dataclass
class SemanticFoldCheckpoint:
    model: SemanticLogisticModel
    fold: int
    protocol: str
    store_manifest_sha256: str
    train_mask_sha256: str
    validation_mask_sha256: str
    config: dict[str, object]
    trainer_source_sha256: str


def fit_semantic_model(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    c: float,
    seed: int,
    max_iter: int = 1_000,
) -> SemanticLogisticModel:
    """Fit the semantic component outside the packaged inference modules."""

    matrix = np.asarray(features, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("features must be a non-empty two-dimensional matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("features contains non-finite values")
    labels = np.asarray(targets)
    if labels.ndim != 1 or labels.size != matrix.shape[0]:
        raise ValueError("targets must match the feature rows")
    if not np.isfinite(labels.astype(np.float64)).all() or not set(
        np.unique(labels)
    ).issubset({0, 1, False, True}):
        raise ValueError("targets must contain only binary 0/1 values")
    labels = labels.astype(np.int8, copy=False)
    if np.unique(labels).size != 2:
        raise ValueError("training targets must contain both binary classes")

    model = SemanticLogisticModel(c=c, seed=seed, max_iter=max_iter)
    model.scaler_ = StandardScaler()
    scaled = model.scaler_.fit_transform(matrix)
    model.classifier_ = LogisticRegression(
        C=model.c,
        random_state=model.seed,
        solver="liblinear",
        max_iter=model.max_iter,
    )
    model.classifier_.fit(scaled, labels)
    model.n_features_in_ = int(matrix.shape[1])
    model.metadata_ = SemanticModelMetadata(
        format_version=METADATA_VERSION,
        estimator="StandardScaler+LogisticRegression",
        feature_count=model.n_features_in_,
        classes=tuple(int(value) for value in model.classifier_.classes_),
        logistic_c=model.c,
        random_seed=model.seed,
        max_iter=model.max_iter,
        sklearn_version=sklearn.__version__,
    )
    return model


def interaction_matrix(arrays: SemanticArrays) -> np.ndarray:
    """Materialize the bounded training interaction matrix in stable row order."""

    objectives_by_row = np.asarray(
        arrays.objective_embeddings[arrays.objective_index_by_row], dtype=np.float32
    )
    return build_semantic_interaction_matrix(
        arrays.context_embeddings,
        objectives_by_row,
        dense_features=arrays.dense,
    )


def train_semantic_cv(
    *,
    store_dir: str | Path,
    folds: Sequence[FoldMasks],
    config: ModelConfig | None = None,
    checkpoint_dir: str | Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> SemanticCVResult:
    cfg = config or ModelConfig()
    store_manifest = load_semantic_manifest(store_dir)
    validate_store_manifest_source(store_manifest, store="semantic")
    if store_manifest["source"]["config"] != cfg.to_dict():
        raise ValueError("semantic store configuration differs from training")
    arrays = load_semantic_arrays(store_dir)
    features = interaction_matrix(arrays)
    oof = np.full(len(arrays.targets), np.nan, dtype=np.float64)
    records: list[dict[str, object]] = []
    checkpoint_root = Path(checkpoint_dir).resolve() if checkpoint_dir else None
    if checkpoint_root:
        checkpoint_root.mkdir(parents=True, exist_ok=True)
    store_manifest_sha256 = _sha256_file(Path(store_dir) / "manifest.json")
    trainer_source_sha256 = trace_ace_source_sha256(SEMANTIC_TRAINING_SOURCE_FILES)
    for fold in folds:
        train_mask_sha256 = _sha256_array(fold.train_mask)
        validation_mask_sha256 = _sha256_array(fold.validation_mask)
        checkpoint = checkpoint_root / f"semantic_fold_{fold.fold}.joblib" if checkpoint_root else None
        if checkpoint and checkpoint.is_file():
            candidate = joblib.load(checkpoint)
            if (
                not isinstance(candidate, SemanticFoldCheckpoint)
                or candidate.store_manifest_sha256 != store_manifest_sha256
                or candidate.train_mask_sha256 != train_mask_sha256
                or candidate.validation_mask_sha256 != validation_mask_sha256
                or candidate.config != cfg.to_dict()
                or candidate.fold != fold.fold
                or candidate.protocol != fold.protocol
                or getattr(candidate, "trainer_source_sha256", None)
                != trainer_source_sha256
            ):
                raise ValueError(f"invalid semantic checkpoint: {checkpoint}")
            model = candidate.model
            if progress:
                progress(f"semantic CV: resumed fold {fold.fold}")
        else:
            model = fit_semantic_model(
                features[fold.train_mask],
                arrays.targets[fold.train_mask],
                c=cfg.semantic_c,
                seed=cfg.seed,
            )
            if checkpoint:
                temporary = checkpoint.with_suffix(".joblib.tmp")
                joblib.dump(
                    SemanticFoldCheckpoint(
                        model=model,
                        fold=fold.fold,
                        protocol=fold.protocol,
                        store_manifest_sha256=store_manifest_sha256,
                        train_mask_sha256=train_mask_sha256,
                        validation_mask_sha256=validation_mask_sha256,
                        config=cfg.to_dict(),
                        trainer_source_sha256=trainer_source_sha256,
                    ),
                    temporary,
                    compress=3,
                )
                temporary.replace(checkpoint)
        validation_rows = fold.validation_indices
        oof[validation_rows] = model.predict_proba(features[validation_rows])[:, 1]
        records.append(
            {
                "fold": fold.fold,
                "train_rows": int(fold.train_mask.sum()),
                "validation_rows": int(fold.validation_mask.sum()),
                "purged_rows": fold.purged_count,
                "semantic": evaluate_probabilities(
                    arrays.targets[validation_rows], oof[validation_rows]
                ),
            }
        )
        if progress:
            progress(f"semantic CV: fold {fold.fold} complete")
    del features
    gc.collect()
    if not np.isfinite(oof).all():
        raise ValueError("semantic validation did not produce complete OOF predictions")
    return SemanticCVResult(oof, tuple(records))


def train_final_semantic(
    store_dir: str | Path,
    *,
    config: ModelConfig | None = None,
    progress: Callable[[str], None] | None = None,
) -> SemanticLogisticModel:
    cfg = config or ModelConfig()
    store_manifest = load_semantic_manifest(store_dir)
    validate_store_manifest_source(store_manifest, store="semantic")
    if store_manifest["source"]["config"] != cfg.to_dict():
        raise ValueError("semantic store configuration differs from training")
    arrays = load_semantic_arrays(store_dir)
    features = interaction_matrix(arrays)
    model = fit_semantic_model(
        features,
        arrays.targets,
        c=cfg.semantic_c,
        seed=cfg.seed,
    )
    del features
    gc.collect()
    if progress:
        progress("final semantic: fit complete")
    return model


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).view(np.uint8)).hexdigest()
