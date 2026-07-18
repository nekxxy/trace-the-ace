"""Focused tests for the final-training validation evidence gates."""

from __future__ import annotations

import math

import pytest

from scripts.train_final import _gate_passed, _validate_metric_evidence


COMPONENTS = ("baseline", "full", "role", "semantic", "ensemble")


def _valid_evidence() -> dict[str, object]:
    metrics = {
        "log_loss": 0.42,
        "roc_auc": 0.73,
        "brier": 0.14,
        "ece10": 0.03,
    }
    return {
        "promotion_gate": {"passed": True},
        **{component: dict(metrics) for component in COMPONENTS},
        "folds": [{"fold": fold} for fold in range(5)],
    }


def test_valid_five_fold_finite_evidence_passes() -> None:
    evidence = _valid_evidence()

    assert _gate_passed(evidence) is True
    _validate_metric_evidence(evidence, "primary")


def test_metric_evidence_rejects_missing_component() -> None:
    evidence = _valid_evidence()
    del evidence["semantic"]

    with pytest.raises(ValueError, match="primary validation lacks semantic metrics"):
        _validate_metric_evidence(evidence, "primary")


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_metric_evidence_rejects_nonfinite_metric(value: float) -> None:
    evidence = _valid_evidence()
    evidence["ensemble"]["log_loss"] = value  # type: ignore[index]

    with pytest.raises(ValueError, match="primary validation contains non-finite metrics"):
        _validate_metric_evidence(evidence, "primary")


@pytest.mark.parametrize("folds", [[], [{"fold": fold} for fold in range(4)], None])
def test_metric_evidence_rejects_wrong_fold_count(folds: object) -> None:
    evidence = _valid_evidence()
    evidence["folds"] = folds

    with pytest.raises(ValueError, match="primary validation must contain five fold records"):
        _validate_metric_evidence(evidence, "primary")


@pytest.mark.parametrize(
    "evidence",
    [
        {"promotion_gate": {"passed": False}},
        {"promotion_gate": {}},
        {"promotion_gate": None},
        {"promotion_gate": {"passed": 1}},
        {},
        None,
    ],
)
def test_gate_passed_rejects_false_or_malformed_gate(evidence: object) -> None:
    assert _gate_passed(evidence) is False
