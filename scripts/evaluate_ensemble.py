#!/usr/bin/env python3
"""Evaluate the fixed documented 25/25/50 OOF ensemble."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from trace_ace.config import ModelConfig
from trace_ace.ensemble import blend_probabilities
from trace_ace.provenance import response_identity_sha256
from trace_ace.validation import evaluate_probabilities, objective_disjoint_folds


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, default=PROJECT_ROOT / "experiments/runs/cleanroom_v02"
    )
    parser.add_argument(
        "--responses",
        type=Path,
        default=PROJECT_ROOT / "data/interim/cache/responses.parquet",
    )
    args = parser.parse_args()
    sparse = pd.read_parquet(args.run_dir / "sparse_oof.parquet").sort_values("row_id")
    semantic = pd.read_parquet(args.run_dir / "semantic_oof.parquet").sort_values("row_id")
    if not np.array_equal(sparse["row_id"], semantic["row_id"]):
        raise ValueError("OOF row IDs differ")
    if not np.array_equal(sparse["is_correct"], semantic["is_correct"]):
        raise ValueError("OOF targets differ")
    config = ModelConfig()
    probability = blend_probabilities(
        sparse["full_probability"].to_numpy(),
        sparse["role_probability"].to_numpy(),
        semantic["semantic_probability"].to_numpy(),
        weights=(config.full_weight, config.role_weight, config.semantic_weight),
        probability_floor=config.probability_floor,
    )
    targets = sparse["is_correct"].to_numpy()
    responses = pq.read_table(args.responses).to_pandas().sort_values("row_id")
    if not np.array_equal(responses["row_id"].to_numpy(), sparse["row_id"].to_numpy()):
        raise ValueError("OOF and response row IDs differ")
    if not np.array_equal(responses["is_correct"].to_numpy(), targets):
        raise ValueError("OOF and response targets differ")
    folds = objective_disjoint_folds(responses)
    baseline_probability = np.full(len(targets), np.nan, dtype=np.float64)
    for fold in folds:
        baseline_probability[fold.validation_mask] = float(
            targets[fold.train_mask].mean()
        )
    if not np.isfinite(baseline_probability).all():
        raise ValueError("fold baseline did not cover every OOF row")
    component_predictions = {
        "baseline": baseline_probability,
        "full": sparse["full_probability"].to_numpy(),
        "role": sparse["role_probability"].to_numpy(),
        "semantic": semantic["semantic_probability"].to_numpy(),
        "ensemble": probability,
    }
    fold_metrics = []
    for fold in folds:
        rows = fold.validation_indices
        fold_metrics.append(
            {
                "fold": fold.fold,
                "train_rows": int(fold.train_mask.sum()),
                "validation_rows": int(fold.validation_mask.sum()),
                "purged_rows": fold.purged_count,
                **{
                    name: evaluate_probabilities(targets[rows], values[rows])
                    for name, values in component_predictions.items()
                },
            }
        )
    metric_names = ("log_loss", "roc_auc", "brier", "ece10")
    fold_dispersion = {
        component: {
            f"{metric}_std": float(
                np.std([record[component][metric] for record in fold_metrics])
            )
            for metric in metric_names
        }
        for component in component_predictions
    }
    for component in component_predictions:
        fold_dispersion[component]["worst_fold_log_loss"] = float(
            max(record[component]["log_loss"] for record in fold_metrics)
        )
    metrics = {
        "config": config.to_dict(),
        "weights": [config.full_weight, config.role_weight, config.semantic_weight],
        "protocol": "objective-disjoint SGKF with validation-session purge",
        "response_identity_sha256": response_identity_sha256(responses),
        "baseline": evaluate_probabilities(targets, component_predictions["baseline"]),
        "full": evaluate_probabilities(targets, component_predictions["full"]),
        "role": evaluate_probabilities(targets, component_predictions["role"]),
        "semantic": evaluate_probabilities(targets, component_predictions["semantic"]),
        "ensemble": evaluate_probabilities(targets, probability),
        "folds": fold_metrics,
        "fold_dispersion": fold_dispersion,
    }
    metrics["promotion_gate"] = {
        "criterion": (
            "ensemble OOF log loss is strictly lower than the leakage-safe "
            "fold-prior baseline and every individual component"
        ),
        "comparators": {
            name: metrics[name]["log_loss"]
            for name in ("baseline", "full", "role", "semantic")
        },
        "passed": bool(
            all(
                metrics["ensemble"]["log_loss"] < metrics[name]["log_loss"]
                for name in ("baseline", "full", "role", "semantic")
            )
        ),
    }
    pd.DataFrame(
        {
            "row_id": sparse["row_id"],
            "is_correct": targets,
            "baseline_probability": baseline_probability,
            "full_probability": sparse["full_probability"],
            "role_probability": sparse["role_probability"],
            "semantic_probability": semantic["semantic_probability"],
            "ensemble_probability": probability,
        }
    ).to_parquet(args.run_dir / "ensemble_oof.parquet", index=False)
    (args.run_dir / "ensemble_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
