"""Tests for the full-component monotone Platt calibration."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import log_loss

from trace_ace.calibration import (
    apply_full_calibration,
    logit,
    validate_full_calibration,
)
from trace_ace.calibration_training import fit_full_calibration


def _overconfident(seed: int = 0, n: int = 4000) -> tuple[np.ndarray, np.ndarray]:
    """Return over-scaled probabilities with real signal and their labels."""

    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    base = rng.normal(0.0, 1.0, n) + (y - 0.5) * 1.5
    over = 1.0 / (1.0 + np.exp(-(base * 2.0 + 0.5)))  # over-confident + biased high
    return over, y


def test_identity_calibration_is_a_no_op() -> None:
    p = np.linspace(0.01, 0.99, 25)
    assert np.allclose(apply_full_calibration(p, (1.0, 0.0)), p, atol=1e-9)


def test_apply_is_monotone_and_row_independent() -> None:
    params = (0.7, 0.2)
    p = np.linspace(0.01, 0.99, 50)
    calibrated = apply_full_calibration(p, params)
    assert np.all(np.diff(calibrated) > 0.0)
    for index in (0, 17, 49):
        single = apply_full_calibration([p[index]], params)[0]
        assert np.isclose(single, calibrated[index])


def test_fit_reduces_log_loss_on_overconfident_input() -> None:
    p, y = _overconfident()
    params = fit_full_calibration(p, y)
    assert params[0] > 0.0
    calibrated = apply_full_calibration(p, params)
    floor = lambda values: np.clip(values, 1e-6, 1 - 1e-6)
    assert log_loss(y, floor(calibrated)) < log_loss(y, floor(p))


@pytest.mark.parametrize(
    "bad",
    [(0.0, 0.1), (-1.0, 0.0), (float("nan"), 0.0), (1.0, float("inf")), (1.0,), (1.0, 2.0, 3.0), "x", None],
)
def test_validate_rejects_invalid_calibration(bad: object) -> None:
    with pytest.raises(ValueError):
        validate_full_calibration(bad)


def test_apply_rejects_invalid_params_and_nonfinite_input() -> None:
    with pytest.raises(ValueError):
        apply_full_calibration([0.5], (0.0, 0.0))
    with pytest.raises(ValueError):
        apply_full_calibration([np.nan], (1.0, 0.0))


def test_fit_requires_both_classes_and_alignment() -> None:
    with pytest.raises(ValueError):
        fit_full_calibration(np.full(10, 0.6), np.zeros(10, dtype=int))
    with pytest.raises(ValueError):
        fit_full_calibration(np.full(10, 0.6), np.ones(5, dtype=int))


def test_logit_clips_to_runtime_floor() -> None:
    assert np.isfinite(logit([0.0, 1.0])).all()
