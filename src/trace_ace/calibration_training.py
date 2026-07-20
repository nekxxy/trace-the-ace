"""Training-only fitting of the full component's monotone Platt calibrator.

Separated from :mod:`trace_ace.calibration` (which is packaged for inference) so
that no fitting logic or scikit-learn dependency is present in the runtime
modules. Only cross-validation and final-training code import from here.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from trace_ace.calibration import logit, validate_full_calibration


def fit_full_calibration(
    full_oof: Sequence[float] | np.ndarray,
    targets: Sequence[int] | np.ndarray,
) -> tuple[float, float]:
    """Fit a monotone Platt calibrator ``(slope, intercept)`` on full OOF preds.

    A logistic regression on the single feature ``logit(full)`` yields the
    slope/intercept. The slope must be strictly positive so the calibrated
    probability is a monotone increasing function of the raw probability.
    """

    from sklearn.linear_model import LogisticRegression

    features = logit(full_oof).reshape(-1, 1)
    labels = np.asarray(targets)
    if features.shape[0] == 0 or features.shape[0] != labels.shape[0]:
        raise ValueError("full_oof and targets must be non-empty and aligned")
    if not set(np.unique(labels).tolist()).issubset({0, 1}):
        raise ValueError("targets must contain only binary 0/1 values")
    if np.unique(labels).size != 2:
        raise ValueError("targets must contain both classes to fit calibration")
    model = LogisticRegression(C=1e6, max_iter=5000)
    model.fit(features, labels)
    slope = float(model.coef_[0, 0])
    intercept = float(model.intercept_[0])
    return validate_full_calibration((slope, intercept))


__all__ = ["fit_full_calibration"]
