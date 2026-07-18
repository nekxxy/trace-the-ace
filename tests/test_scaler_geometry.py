"""Fail-closed runtime checks for the locked training block geometry."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from trace_ace.ensemble import _validate_scaler_geometry


def _valid_scalers() -> tuple[StandardScaler, StandardScaler]:
    role = StandardScaler().fit(
        np.asarray([[0.0, 1.0], [1.0, 0.0], [2.0, 3.0]], dtype=np.float32)
    )
    role.trace_ace_geometry_ = "standardized-unit-expected-l2-v1"
    role.trace_ace_block_width_ = 2

    semantic = StandardScaler().fit(
        np.asarray(
            [[0.1, 0.2, 0.3, 1.0, 2.0], [0.3, 0.4, 0.5, 2.0, 4.0]],
            dtype=np.float32,
        )
    )
    semantic.mean_[:3] = 0.0
    semantic.scale_[:3] = 1.0
    semantic.trace_ace_geometry_ = "natural-semantic-unit-dense-l2-v1"
    semantic.trace_ace_semantic_feature_count_ = 3
    semantic.trace_ace_dense_block_width_ = 2
    return role, semantic


def test_locked_scaler_geometry_is_accepted() -> None:
    role, semantic = _valid_scalers()

    _validate_scaler_geometry(
        role_scaler=role,
        semantic_scaler=semantic,
        semantic_feature_count=3,
        dense_block_width=2,
    )


def test_legacy_role_scaler_is_rejected() -> None:
    role, semantic = _valid_scalers()
    del role.trace_ace_geometry_

    with pytest.raises(RuntimeError, match="role dense scaler geometry"):
        _validate_scaler_geometry(
            role_scaler=role,
            semantic_scaler=semantic,
            semantic_feature_count=3,
            dense_block_width=2,
        )


def test_invalid_role_scaler_parameters_are_rejected() -> None:
    role, semantic = _valid_scalers()
    role.scale_[0] = np.nan

    with pytest.raises(RuntimeError, match="role dense scaler parameters"):
        _validate_scaler_geometry(
            role_scaler=role,
            semantic_scaler=semantic,
            semantic_feature_count=3,
            dense_block_width=2,
        )


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("trace_ace_geometry_", "legacy-coordinate-scaling"),
        ("trace_ace_semantic_feature_count_", 4),
        ("trace_ace_dense_block_width_", 1),
    ],
)
def test_malformed_semantic_geometry_is_rejected(
    attribute: str, value: object
) -> None:
    role, semantic = _valid_scalers()
    setattr(semantic, attribute, value)

    with pytest.raises(RuntimeError, match="semantic scaler geometry"):
        _validate_scaler_geometry(
            role_scaler=role,
            semantic_scaler=semantic,
            semantic_feature_count=3,
            dense_block_width=2,
        )


def test_semantic_natural_prefix_transform_is_enforced() -> None:
    role, semantic = _valid_scalers()
    semantic.mean_[1] = 0.25

    with pytest.raises(RuntimeError, match="parameters violate locked geometry"):
        _validate_scaler_geometry(
            role_scaler=role,
            semantic_scaler=semantic,
            semantic_feature_count=3,
            dense_block_width=2,
        )
