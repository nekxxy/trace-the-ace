"""Memory-bounded sparse cross-validation and final fitting."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Callable, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

from trace_ace.config import ModelConfig
from trace_ace.feature_store import dense_columns
from trace_ace.provenance import (
    SPARSE_TRAINING_SOURCE_FILES,
    trace_ace_source_sha256,
    validate_store_manifest_source,
)
from trace_ace.sparse import (
    SparseArtifact,
    append_scaled_dense,
    positive_probability,
)
from trace_ace.sparse_store import (
    iter_sparse_parts,
    load_dense_by_row,
    validate_sparse_store,
)
from trace_ace.validation import FoldMasks, evaluate_probabilities


@dataclass
class SparseFoldModel:
    fold: int
    protocol: str
    full_model: object
    role_model: object
    role_dense_scaler: object
    config: dict[str, object]
    store_manifest_sha256: str
    train_mask_sha256: str
    validation_mask_sha256: str
    trainer_source_sha256: str


@dataclass(frozen=True)
class SparseCVResult:
    full_oof: np.ndarray
    role_oof: np.ndarray
    fold_metrics: tuple[dict[str, object], ...]


def new_sparse_classifier(*, alpha: float, config: ModelConfig | None = None) -> SGDClassifier:
    """Create the deterministic incremental logistic model used for training."""

    cfg = config or ModelConfig()
    return SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=alpha,
        fit_intercept=True,
        learning_rate="optimal",
        average=cfg.sparse_average,
        shuffle=False,
        random_state=cfg.seed,
        tol=None,
    )


def fit_dense_scaler(dense: np.ndarray) -> StandardScaler:
    """Fit a finite float32 dense-feature scaler using training rows only."""

    values = np.asarray(dense, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("dense features must be a finite 2D matrix")
    return StandardScaler(copy=True).fit(values)


def train_sparse_cv(
    *,
    store_dir: str | Path,
    folds: Sequence[FoldMasks],
    config: ModelConfig | None = None,
    checkpoint_dir: str | Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> SparseCVResult:
    """Train or resume full/role sparse models for leakage-safe folds."""

    cfg = config or ModelConfig()
    store_manifest = validate_sparse_store(store_dir)
    validate_store_manifest_source(store_manifest, store="sparse")
    if store_manifest["source"]["config"] != cfg.to_dict():
        raise ValueError("sparse store configuration differs from training")
    targets, dense = load_dense_by_row(store_dir)
    row_count = len(targets)
    full_oof = np.full(row_count, np.nan, dtype=np.float64)
    role_oof = np.full(row_count, np.nan, dtype=np.float64)
    metrics: list[dict[str, object]] = []
    checkpoint_root = Path(checkpoint_dir).resolve() if checkpoint_dir else None
    if checkpoint_root:
        checkpoint_root.mkdir(parents=True, exist_ok=True)

    store_manifest_sha256 = _sha256_file(Path(store_dir) / "manifest.json")
    trainer_source_sha256 = trace_ace_source_sha256(SPARSE_TRAINING_SOURCE_FILES)
    for fold in folds:
        train_mask_sha256 = _sha256_array(fold.train_mask)
        validation_mask_sha256 = _sha256_array(fold.validation_mask)
        checkpoint = checkpoint_root / f"sparse_fold_{fold.fold}.joblib" if checkpoint_root else None
        fold_model: SparseFoldModel
        if checkpoint and checkpoint.is_file():
            candidate = joblib.load(checkpoint)
            if (
                not isinstance(candidate, SparseFoldModel)
                or candidate.config != cfg.to_dict()
                or candidate.fold != fold.fold
                or candidate.protocol != fold.protocol
                or candidate.store_manifest_sha256 != store_manifest_sha256
                or candidate.train_mask_sha256 != train_mask_sha256
                or candidate.validation_mask_sha256 != validation_mask_sha256
                or getattr(candidate, "trainer_source_sha256", None)
                != trainer_source_sha256
            ):
                raise ValueError(f"incompatible sparse checkpoint: {checkpoint}")
            fold_model = candidate
            if progress:
                progress(f"sparse CV: resumed fold {fold.fold}")
        else:
            scaler = fit_dense_scaler(dense[fold.train_mask])
            full_model = new_sparse_classifier(alpha=cfg.full_alpha, config=cfg)
            role_model = new_sparse_classifier(alpha=cfg.role_alpha, config=cfg)
            first_full = True
            first_role = True
            for epoch in range(cfg.sparse_epochs):
                updates = 0
                for part in iter_sparse_parts(store_dir):
                    local_train = fold.train_mask[part.row_ids]
                    if not local_train.any():
                        continue
                    y = part.targets[local_train]
                    full_kwargs = {"classes": np.asarray([0, 1], dtype=np.int8)} if first_full else {}
                    full_model.partial_fit(part.full[local_train], y, **full_kwargs)
                    first_full = False
                    role_matrix = append_scaled_dense(
                        part.role[local_train], part.dense[local_train], scaler
                    )
                    role_kwargs = {"classes": np.asarray([0, 1], dtype=np.int8)} if first_role else {}
                    role_model.partial_fit(role_matrix, y, **role_kwargs)
                    first_role = False
                    updates += len(y)
                if updates == 0:
                    raise ValueError(f"fold {fold.fold} has no training rows")
                if progress:
                    progress(
                        f"sparse CV: fold {fold.fold} epoch {epoch + 1}/"
                        f"{cfg.sparse_epochs} complete"
                    )
            fold_model = SparseFoldModel(
                fold=fold.fold,
                protocol=fold.protocol,
                full_model=full_model,
                role_model=role_model,
                role_dense_scaler=scaler,
                config=cfg.to_dict(),
                store_manifest_sha256=store_manifest_sha256,
                train_mask_sha256=train_mask_sha256,
                validation_mask_sha256=validation_mask_sha256,
                trainer_source_sha256=trainer_source_sha256,
            )
            if checkpoint:
                temporary = checkpoint.with_suffix(".joblib.tmp")
                joblib.dump(fold_model, temporary, compress=3)
                temporary.replace(checkpoint)

        for part in iter_sparse_parts(store_dir):
            local_validation = fold.validation_mask[part.row_ids]
            if not local_validation.any():
                continue
            row_ids = part.row_ids[local_validation]
            full_oof[row_ids] = positive_probability(
                fold_model.full_model, part.full[local_validation]
            )
            role_matrix = append_scaled_dense(
                part.role[local_validation],
                part.dense[local_validation],
                fold_model.role_dense_scaler,
            )
            role_oof[row_ids] = positive_probability(fold_model.role_model, role_matrix)

        validation_rows = fold.validation_indices
        fold_record: dict[str, object] = {
            "fold": fold.fold,
            "train_rows": int(fold.train_mask.sum()),
            "validation_rows": int(fold.validation_mask.sum()),
            "purged_rows": fold.purged_count,
            "full": evaluate_probabilities(targets[validation_rows], full_oof[validation_rows]),
            "role": evaluate_probabilities(targets[validation_rows], role_oof[validation_rows]),
        }
        metrics.append(fold_record)
        if progress:
            progress(f"sparse CV: fold {fold.fold} predictions complete")

    if not np.isfinite(full_oof).all() or not np.isfinite(role_oof).all():
        raise ValueError("validation folds did not produce complete OOF predictions")
    return SparseCVResult(full_oof, role_oof, tuple(metrics))


def train_final_sparse(
    *,
    store_dir: str | Path,
    config: ModelConfig | None = None,
    progress: Callable[[str], None] | None = None,
) -> SparseArtifact:
    """Fit both sparse components on every training row."""

    cfg = config or ModelConfig()
    store_manifest = validate_sparse_store(store_dir)
    validate_store_manifest_source(store_manifest, store="sparse")
    if store_manifest["source"]["config"] != cfg.to_dict():
        raise ValueError("sparse store configuration differs from training")
    _, dense = load_dense_by_row(store_dir)
    scaler = fit_dense_scaler(dense)
    full_model = new_sparse_classifier(alpha=cfg.full_alpha, config=cfg)
    role_model = new_sparse_classifier(alpha=cfg.role_alpha, config=cfg)
    first_full = True
    first_role = True
    for epoch in range(cfg.sparse_epochs):
        updates = 0
        for part in iter_sparse_parts(store_dir):
            full_kwargs = {"classes": np.asarray([0, 1], dtype=np.int8)} if first_full else {}
            full_model.partial_fit(part.full, part.targets, **full_kwargs)
            first_full = False
            role_matrix = append_scaled_dense(part.role, part.dense, scaler)
            role_kwargs = {"classes": np.asarray([0, 1], dtype=np.int8)} if first_role else {}
            role_model.partial_fit(role_matrix, part.targets, **role_kwargs)
            first_role = False
            updates += len(part.targets)
        if updates == 0:
            raise ValueError("sparse store is empty")
        if progress:
            progress(f"final sparse: epoch {epoch + 1}/{cfg.sparse_epochs} complete")
    return SparseArtifact.create(
        config=cfg,
        full_model=full_model,
        role_model=role_model,
        role_dense_scaler=scaler,
        dense_feature_names=dense_columns(),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).view(np.uint8)).hexdigest()
