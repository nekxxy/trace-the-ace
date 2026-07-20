"""Runtime application of the full sparse component's monotone calibration.

The full-transcript sparse head is trained with ``learning_rate="optimal"`` and a
small ``alpha``; that schedule over-scales its logits, leaving the component
over-confident and biased high (out-of-fold log loss worse than the fold prior
despite healthy ranking). A single two-parameter monotone Platt calibrator,
fit only on out-of-fold ``full`` predictions during training, corrects the scale
before the fixed blend.

This module is packaged in the inference runtime and therefore performs **no**
fitting: it only validates and applies the two frozen scalars. Fitting lives in
the training-only :mod:`trace_ace.calibration_training` module. The transform is
row-independent (``sigmoid(slope * logit(p) + intercept)``), so no information
crosses between test samples and it is legal for independent competition
inference.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


FULL_CALIBRATION_VERSION = 1
_FLOOR = 1e-6


def logit(probabilities: object) -> np.ndarray:
    """Return the finite logit of probabilities clipped to the runtime floor."""

    values = np.asarray(probabilities, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("probabilities must be finite")
    clipped = np.clip(values, _FLOOR, 1.0 - _FLOOR)
    return np.log(clipped) - np.log1p(-clipped)


def apply_full_calibration(
    full_prob: Sequence[float] | np.ndarray,
    calibration: Sequence[float],
) -> np.ndarray:
    """Return ``sigmoid(slope * logit(full) + intercept)`` row-independently."""

    slope, intercept = validate_full_calibration(calibration)
    z = logit(full_prob)
    calibrated = 1.0 / (1.0 + np.exp(-(slope * z + intercept)))
    if not np.isfinite(calibrated).all():
        raise ValueError("calibrated probabilities are not finite")
    return np.asarray(calibrated, dtype=np.float64)


def validate_full_calibration(calibration: object) -> tuple[float, float]:
    """Validate and return a ``(slope, intercept)`` calibration pair."""

    if not isinstance(calibration, (tuple, list)) or len(calibration) != 2:
        raise ValueError("full calibration must be a (slope, intercept) pair")
    slope = float(calibration[0])
    intercept = float(calibration[1])
    if not np.isfinite(slope) or not np.isfinite(intercept):
        raise ValueError("full calibration parameters must be finite")
    if slope <= 0.0:
        raise ValueError("full calibration slope must be strictly positive")
    return slope, intercept


__all__ = [
    "FULL_CALIBRATION_VERSION",
    "apply_full_calibration",
    "logit",
    "validate_full_calibration",
]
