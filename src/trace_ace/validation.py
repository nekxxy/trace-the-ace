"""Leakage-safe validation splits and probability metrics.

The competition has multiple response rows per tutoring session.  Every split in
this module therefore removes *all* training rows whose session occurs in the
validation fold.  Objective- and semantic-family protocols add stricter group
boundaries on top of that session purge.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Hashable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


DEFAULT_SEED = 20260716


@dataclass(frozen=True)
class FoldMasks:
    """Boolean row masks for one validation fold.

    Rows purged because their session appears in validation are false in both
    masks.  This is intentional: train and validation masks need not be
    exhaustive for objective- or family-disjoint protocols.
    """

    protocol: str
    fold: int
    train_mask: np.ndarray
    validation_mask: np.ndarray
    purged_count: int

    @property
    def train_indices(self) -> np.ndarray:
        """Return positional training indices."""

        return np.flatnonzero(self.train_mask)

    @property
    def validation_indices(self) -> np.ndarray:
        """Return positional validation indices."""

        return np.flatnonzero(self.validation_mask)


def session_grouped_folds(
    frame: pd.DataFrame,
    *,
    target_col: str = "is_correct",
    session_col: str = "session_id",
    n_splits: int = 5,
    seed: int = DEFAULT_SEED,
) -> list[FoldMasks]:
    """Create deterministic stratified folds grouped by tutoring session."""

    return _grouped_folds(
        frame,
        group_values=frame[session_col] if session_col in frame else None,
        protocol="session",
        target_col=target_col,
        session_col=session_col,
        n_splits=n_splits,
        seed=seed,
    )


def objective_disjoint_folds(
    frame: pd.DataFrame,
    *,
    target_col: str = "is_correct",
    session_col: str = "session_id",
    objective_col: str = "learning_objective_id",
    n_splits: int = 5,
    seed: int = DEFAULT_SEED,
) -> list[FoldMasks]:
    """Create objective-disjoint folds and purge validation sessions from train."""

    return _grouped_folds(
        frame,
        group_values=frame[objective_col] if objective_col in frame else None,
        protocol="objective",
        target_col=target_col,
        session_col=session_col,
        objective_col=objective_col,
        n_splits=n_splits,
        seed=seed,
    )


def assign_semantic_families(
    objective_embeddings: pd.DataFrame | np.ndarray,
    *,
    objective_ids: Sequence[Hashable] | None = None,
    n_families: int,
    seed: int = DEFAULT_SEED,
) -> pd.Series:
    """Cluster precomputed objective embeddings into deterministic families.

    A DataFrame may carry objective IDs in its index.  For a NumPy array,
    ``objective_ids`` is required.  Inputs are sorted by a stable representation
    before KMeans is fitted so assignment does not depend on caller row order;
    cluster labels are then canonicalized by centroid order.
    """

    if isinstance(objective_embeddings, pd.DataFrame):
        matrix = objective_embeddings.to_numpy(dtype=np.float64, copy=True)
        ids = list(objective_embeddings.index if objective_ids is None else objective_ids)
    else:
        matrix = np.asarray(objective_embeddings, dtype=np.float64)
        if objective_ids is None:
            raise ValueError("objective_ids is required for array embeddings")
        ids = list(objective_ids)

    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("objective_embeddings must be a non-empty 2D matrix")
    if len(ids) != matrix.shape[0]:
        raise ValueError("objective_ids length must match embedding rows")
    index = pd.Index(ids, name="learning_objective_id")
    if index.has_duplicates:
        raise ValueError("objective_ids must be unique")
    if not np.isfinite(matrix).all():
        raise ValueError("objective_embeddings contains non-finite values")
    if not isinstance(n_families, int) or not 1 <= n_families <= matrix.shape[0]:
        raise ValueError("n_families must be between 1 and the number of objectives")
    if np.unique(matrix, axis=0).shape[0] < n_families:
        raise ValueError("n_families exceeds the number of unique embeddings")

    stable_order = np.asarray(
        sorted(range(len(ids)), key=lambda position: _stable_key(ids[position])),
        dtype=np.int64,
    )
    sorted_matrix = matrix[stable_order]
    model = KMeans(
        n_clusters=n_families,
        random_state=seed,
        n_init=10,
        algorithm="lloyd",
    )
    sorted_labels = model.fit_predict(sorted_matrix)

    centroid_order = sorted(
        range(n_families),
        key=lambda label: tuple(model.cluster_centers_[label].tolist()),
    )
    canonical = {old_label: new_label for new_label, old_label in enumerate(centroid_order)}
    sorted_labels = np.asarray([canonical[int(label)] for label in sorted_labels], dtype=np.int32)

    labels = np.empty(len(ids), dtype=np.int32)
    labels[stable_order] = sorted_labels
    return pd.Series(labels, index=index, name="semantic_family")


def semantic_family_disjoint_folds(
    frame: pd.DataFrame,
    objective_families: pd.Series | Mapping[Hashable, Hashable],
    *,
    target_col: str = "is_correct",
    session_col: str = "session_id",
    objective_col: str = "learning_objective_id",
    n_splits: int = 5,
    seed: int = DEFAULT_SEED,
    protocol_name: str = "semantic_family",
) -> list[FoldMasks]:
    """Create family-disjoint folds and purge validation sessions from train."""

    row_families = _map_objective_families(frame, objective_col, objective_families)
    return _grouped_folds(
        frame,
        group_values=row_families,
        protocol=protocol_name,
        target_col=target_col,
        session_col=session_col,
        objective_col=objective_col,
        semantic_families=row_families,
        n_splits=n_splits,
        seed=seed,
    )


def assert_fold_isolation(
    frame: pd.DataFrame,
    train_mask: Sequence[bool] | np.ndarray,
    validation_mask: Sequence[bool] | np.ndarray,
    *,
    session_col: str = "session_id",
    objective_col: str | None = None,
    semantic_families: Sequence[Hashable] | pd.Series | None = None,
) -> None:
    """Assert zero row/session and optional objective/family overlap."""

    required = [session_col]
    if objective_col is not None:
        required.append(objective_col)
    _require_columns(frame, required)
    train = _coerce_mask(train_mask, len(frame), "train_mask")
    validation = _coerce_mask(validation_mask, len(frame), "validation_mask")

    if np.any(train & validation):
        raise AssertionError("training and validation row masks overlap")
    _assert_value_disjoint(
        frame[session_col].to_numpy()[train],
        frame[session_col].to_numpy()[validation],
        "session",
    )

    if objective_col is not None:
        objectives = frame[objective_col].to_numpy()
        _assert_value_disjoint(objectives[train], objectives[validation], "objective")

    if semantic_families is not None:
        families = _coerce_row_values(semantic_families, frame)
        _assert_value_disjoint(families[train], families[validation], "semantic family")


def expected_calibration_error(
    y_true: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    """Return equal-width expected calibration error."""

    targets, predictions = _validate_predictions(y_true, probabilities)
    if not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")

    bin_ids = np.minimum(np.floor(predictions * n_bins).astype(np.int64), n_bins - 1)
    error = 0.0
    for bin_id in range(n_bins):
        members = bin_ids == bin_id
        if members.any():
            error += float(members.mean()) * abs(
                float(targets[members].mean()) - float(predictions[members].mean())
            )
    return float(error)


def evaluate_probabilities(
    y_true: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
) -> dict[str, float]:
    """Evaluate binary probabilities, returning log loss as the primary metric."""

    targets, predictions = _validate_predictions(y_true, probabilities)
    auc = (
        float(roc_auc_score(targets, predictions))
        if np.unique(targets).size == 2
        else float("nan")
    )
    return {
        "log_loss": float(log_loss(targets, predictions, labels=[0, 1])),
        "roc_auc": auc,
        "brier": float(brier_score_loss(targets, predictions)),
        "ece10": expected_calibration_error(targets, predictions, n_bins=10),
    }


def _grouped_folds(
    frame: pd.DataFrame,
    *,
    group_values: pd.Series | np.ndarray | None,
    protocol: str,
    target_col: str,
    session_col: str,
    n_splits: int,
    seed: int,
    objective_col: str | None = None,
    semantic_families: pd.Series | np.ndarray | None = None,
) -> list[FoldMasks]:
    required = [target_col, session_col]
    if objective_col is not None:
        required.append(objective_col)
    _require_columns(frame, required)
    if frame.empty:
        raise ValueError("frame must contain at least one row")
    if not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError("n_splits must be an integer of at least 2")
    if group_values is None:
        raise KeyError("grouping column is missing")

    targets = _validate_binary_target(frame[target_col])
    groups = _coerce_row_values(group_values, frame)
    if pd.isna(groups).any():
        raise ValueError("group values may not be missing")
    if pd.unique(groups).size < n_splits:
        raise ValueError("n_splits exceeds the number of unique groups")

    session_values = frame[session_col].to_numpy()
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    validation_coverage = np.zeros(len(frame), dtype=np.int16)
    folds: list[FoldMasks] = []

    for fold_number, (base_train_indices, validation_indices) in enumerate(
        splitter.split(np.zeros(len(frame), dtype=np.uint8), targets, groups)
    ):
        base_train_mask = np.zeros(len(frame), dtype=bool)
        validation_mask = np.zeros(len(frame), dtype=bool)
        base_train_mask[base_train_indices] = True
        validation_mask[validation_indices] = True

        validation_sessions = set(session_values[validation_mask].tolist())
        shares_validation_session = pd.Series(session_values, copy=False).isin(
            validation_sessions
        ).to_numpy()
        train_mask = base_train_mask & ~shares_validation_session
        purged_count = int(base_train_mask.sum() - train_mask.sum())

        if not train_mask.any():
            raise ValueError(
                f"session purge left fold {fold_number} without training rows; "
                "use fewer splits or a less entangled synthetic/data grouping"
            )
        if not validation_mask.any():
            raise AssertionError(f"fold {fold_number} has no validation rows")

        assert_fold_isolation(
            frame,
            train_mask,
            validation_mask,
            session_col=session_col,
            objective_col=objective_col,
            semantic_families=semantic_families,
        )
        train_mask.setflags(write=False)
        validation_mask.setflags(write=False)
        folds.append(
            FoldMasks(
                protocol=protocol,
                fold=fold_number,
                train_mask=train_mask,
                validation_mask=validation_mask,
                purged_count=purged_count,
            )
        )
        validation_coverage += validation_mask

    if not np.all(validation_coverage == 1):
        raise AssertionError("validation folds must cover every row exactly once")
    return folds


def _map_objective_families(
    frame: pd.DataFrame,
    objective_col: str,
    objective_families: pd.Series | Mapping[Hashable, Hashable],
) -> pd.Series:
    _require_columns(frame, [objective_col])
    if isinstance(objective_families, pd.Series):
        if objective_families.index.has_duplicates:
            raise ValueError("objective_families index must be unique")
        mapping = objective_families
    else:
        mapping = pd.Series(dict(objective_families), dtype=object)
    row_families = frame[objective_col].map(mapping)
    if row_families.isna().any():
        missing = pd.unique(frame.loc[row_families.isna(), objective_col])
        preview = ", ".join(repr(value) for value in missing[:5])
        raise ValueError(f"semantic family is missing for objective(s): {preview}")
    return row_families.rename("semantic_family")


def _validate_binary_target(target: pd.Series) -> np.ndarray:
    if target.isna().any():
        raise ValueError("target may not contain missing values")
    values = target.to_numpy()
    if not set(pd.unique(values)).issubset({0, 1, False, True}):
        raise ValueError("target must contain only binary 0/1 values")
    return values.astype(np.int8, copy=False)


def _validate_predictions(
    y_true: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    targets = np.asarray(y_true)
    predictions = np.asarray(probabilities, dtype=np.float64)
    if targets.ndim != 1 or predictions.ndim != 1:
        raise ValueError("targets and probabilities must be one-dimensional")
    if targets.size == 0 or targets.size != predictions.size:
        raise ValueError("targets and probabilities must have the same non-zero length")
    if pd.isna(targets).any() or not set(np.unique(targets)).issubset({0, 1, False, True}):
        raise ValueError("targets must contain only binary 0/1 values")
    if not np.isfinite(predictions).all() or np.any((predictions < 0) | (predictions > 1)):
        raise ValueError("probabilities must be finite values in [0, 1]")
    return targets.astype(np.int8, copy=False), predictions


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"missing required column(s): {', '.join(missing)}")
    if frame[list(columns)].isna().any(axis=None):
        raise ValueError("split columns may not contain missing values")


def _coerce_mask(mask: Sequence[bool] | np.ndarray, length: int, name: str) -> np.ndarray:
    values = np.asarray(mask)
    if values.ndim != 1 or values.size != length:
        raise ValueError(f"{name} must be a one-dimensional mask with {length} entries")
    if values.dtype != np.bool_:
        raise ValueError(f"{name} must have boolean dtype")
    return values


def _coerce_row_values(values: Sequence[Hashable] | pd.Series, frame: pd.DataFrame) -> np.ndarray:
    if isinstance(values, pd.Series):
        if len(values) != len(frame):
            raise ValueError("row-aligned values must match frame length")
        if values.index.equals(frame.index):
            array = values.to_numpy()
        else:
            try:
                array = values.reindex(frame.index).to_numpy()
            except ValueError as error:
                raise ValueError("row-aligned values must have a unique compatible index") from error
    else:
        array = np.asarray(values)
    if array.ndim != 1 or array.size != len(frame):
        raise ValueError("row-aligned values must be one-dimensional and match frame length")
    return array


def _assert_value_disjoint(train_values: np.ndarray, validation_values: np.ndarray, name: str) -> None:
    overlap = set(train_values.tolist()).intersection(validation_values.tolist())
    if overlap:
        preview = ", ".join(repr(value) for value in list(overlap)[:5])
        raise AssertionError(f"{name} overlap detected: {preview}")


def _stable_key(value: Hashable) -> tuple[str, str]:
    return type(value).__name__, repr(value)


__all__ = [
    "DEFAULT_SEED",
    "FoldMasks",
    "assert_fold_isolation",
    "assign_semantic_families",
    "evaluate_probabilities",
    "expected_calibration_error",
    "objective_disjoint_folds",
    "semantic_family_disjoint_folds",
    "session_grouped_folds",
]
