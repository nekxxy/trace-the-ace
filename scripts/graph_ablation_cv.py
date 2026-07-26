#!/usr/bin/env python3
"""v08 go/no-go ablation: does appending graph-inspired dense features
(scripts/build_graph_features.py) to an existing semantic store's dense
block improve objective-disjoint and session-disjoint CV log loss?

Uses the SAME fit_semantic_model / folds / metrics as production so the
comparison is apples-to-apples. Does not touch feature_store.py, does not
rebuild any cached store, does not write anything back into the production
pipeline - this is a read-only-of-production-artifacts, throwaway comparison.

Only run this when no other heavy job (a store rebuild, another CV run) is
using significant RAM concurrently - fitting the augmented interaction matrix
needs the same peak memory class that required the swap file for Stage 3 of
the main pipeline.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from trace_ace.config import ModelConfig
from trace_ace.semantic import build_semantic_interaction_matrix
from trace_ace.semantic_training import fit_semantic_model
from trace_ace.semantic_store import load_semantic_arrays
from trace_ace.validation import (
    evaluate_probabilities,
    objective_disjoint_folds,
    session_grouped_folds,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--semantic-store",
        type=Path,
        default=PROJECT_ROOT / "data/processed/semantic_store_bge_base_v06",
    )
    parser.add_argument(
        "--graph-features",
        type=Path,
        default=PROJECT_ROOT / "data/interim/graph_features/graph_dense.parquet",
    )
    parser.add_argument(
        "--responses",
        type=Path,
        default=PROJECT_ROOT / "data/interim/cache/responses.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "experiments/runs/v08_graph_ablation.json",
    )
    args = parser.parse_args()

    cfg = ModelConfig()
    responses = (
        pq.read_table(args.responses).to_pandas().sort_values("row_id").reset_index(drop=True)
    )
    if not np.array_equal(responses["row_id"].to_numpy(), np.arange(len(responses))):
        raise ValueError("responses row_id must be contiguous")

    arrays = load_semantic_arrays(args.semantic_store)
    if not np.array_equal(arrays.targets, responses["is_correct"].to_numpy(dtype=np.int8)):
        raise ValueError("responses and semantic store targets differ")

    graph = pd.read_parquet(args.graph_features).sort_values("row_id").reset_index(drop=True)
    if not np.array_equal(graph["row_id"].to_numpy(), np.arange(len(responses))):
        raise ValueError("graph feature row_id must be contiguous and match responses")
    graph_matrix = graph.drop(columns=["row_id"]).to_numpy(dtype=np.float32)
    print(f"graph feature matrix shape: {graph_matrix.shape}", flush=True)

    objectives_by_row = np.asarray(
        arrays.objective_embeddings[arrays.objective_index_by_row], dtype=np.float32
    )

    baseline_features = build_semantic_interaction_matrix(
        arrays.context_embeddings, objectives_by_row, dense_features=arrays.dense
    )
    augmented_features = build_semantic_interaction_matrix(
        arrays.context_embeddings,
        objectives_by_row,
        dense_features=np.concatenate([arrays.dense, graph_matrix], axis=1),
    )
    print(
        f"baseline features: {baseline_features.shape}, "
        f"augmented features: {augmented_features.shape}",
        flush=True,
    )

    def run_cv(features: np.ndarray, folds, label: str) -> dict:
        oof = np.full(len(arrays.targets), np.nan, dtype=np.float64)
        start = time.time()
        for fold in folds:
            model = fit_semantic_model(
                features[fold.train_mask],
                arrays.targets[fold.train_mask],
                c=cfg.semantic_c,
                seed=cfg.seed,
            )
            oof[fold.validation_indices] = model.predict_proba(
                features[fold.validation_indices]
            )[:, 1]
            print(
                f"  {label}: fold {fold.fold} done ({time.time() - start:.1f}s elapsed)",
                flush=True,
            )
        if not np.isfinite(oof).all():
            raise ValueError(f"{label}: OOF predictions did not cover every row")
        return evaluate_probabilities(arrays.targets, oof)

    results: dict[str, dict] = {}

    print("=== PRIMARY (objective-disjoint) protocol ===", flush=True)
    primary_folds = objective_disjoint_folds(responses, seed=cfg.seed)
    results["primary_baseline"] = run_cv(baseline_features, primary_folds, "primary_baseline")
    print(json.dumps(results["primary_baseline"]), flush=True)
    results["primary_with_graph"] = run_cv(
        augmented_features, primary_folds, "primary_with_graph"
    )
    print(json.dumps(results["primary_with_graph"]), flush=True)

    print("=== SESSION-DISJOINT protocol ===", flush=True)
    session_folds = session_grouped_folds(responses, seed=cfg.seed)
    results["session_baseline"] = run_cv(baseline_features, session_folds, "session_baseline")
    print(json.dumps(results["session_baseline"]), flush=True)
    results["session_with_graph"] = run_cv(
        augmented_features, session_folds, "session_with_graph"
    )
    print(json.dumps(results["session_with_graph"]), flush=True)

    print("=== SUMMARY ===", flush=True)
    for name, metrics in results.items():
        print(f"{name}: log_loss={metrics['log_loss']:.5f} roc_auc={metrics['roc_auc']:.4f}", flush=True)

    primary_delta = (
        results["primary_with_graph"]["log_loss"] - results["primary_baseline"]["log_loss"]
    )
    session_delta = (
        results["session_with_graph"]["log_loss"] - results["session_baseline"]["log_loss"]
    )
    print(f"primary delta (with_graph - baseline): {primary_delta:+.5f}", flush=True)
    print(f"session delta (with_graph - baseline): {session_delta:+.5f}", flush=True)
    print("ABLATION_DONE", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
